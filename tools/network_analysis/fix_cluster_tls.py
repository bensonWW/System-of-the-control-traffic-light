"""
Fix TLS attributes and connections for specific cluster junctions.
This script repairs known issues where junctions lose their TLS assignments or have uncontrolled connections.
"""
import xml.etree.ElementTree as ET
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.config import get_network_config

def fix_cluster_tls(network_name: str):
    config = get_network_config(network_name)
    net_file = config["net_path"]
    
    print(f"Fixing TLS for network: {net_file}")
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    updates = 0
    
    # Define repairs
    # Format: { junction_id: { 'tls_id': str, 'logic_extend': bool } }
    repairs = {
        "cluster655375228_655375240": {
            "tls_id": "joinedS_655375228_655375239_655375240",
            "logic_extend": True
        },
        "cluster619136899_619136900_cluster_619136929_619136942": {
            "tls_id": "cluster619136899_619136900_cluster_619136929_619136942",
            "logic_extend": True
        }
    }
    
    for junction_id, spec in repairs.items():
        tls_id = spec['tls_id']
        print(f"\nProcessing {junction_id} -> TLS {tls_id}")
        
        # 1. Fix Junction Attribute
        junction_found = False
        for j in root.findall('junction'):
            if j.get('id') == junction_id:
                current_tl = j.get('tl')
                if current_tl != tls_id:
                    j.set('tl', tls_id)
                    j.set('programID', '0')
                    print(f"  [Fix] Updated junction 'tl' from '{current_tl}' to '{tls_id}'")
                    updates += 1
                junction_found = True
        
        if not junction_found:
            print(f"  [Warn] Junction {junction_id} not found in network.")
            continue

        # 2. Find Max Link Index
        max_idx = -1
        for c in root.findall('connection'):
            if c.get('tl') == tls_id:
                try:
                    idx = int(c.get('linkIndex') or -1)
                    if idx > max_idx: max_idx = idx
                except ValueError: pass
        
        print(f"  Current max linkIndex: {max_idx}")
        next_idx = max_idx + 1
        

        # 3. Fix Uncontrolled Connections
        # Connections originating from the cluster or passing through via
        fixed_conn_count = 0
        
        # [NEW] Specific fix for 49073272#4 (South Arm) -> 668037874 (West Arm)
        # Issue: Only Lane 2 connects. Need Lane 1 to connect as well.
        if junction_id == "cluster619136899_619136900_cluster_619136929_619136942":
            source_edge = "49073272#4"
            dest_edge = "668037874"
            
            # Check if connection already exists
            existing_conns = []
            for c in root.findall('connection'):
                if c.get('from') == source_edge and c.get('to') == dest_edge:
                    existing_conns.append(c)
            
            has_lane_1 = any(c.get('fromLane') == '1' for c in existing_conns)
            
            if not has_lane_1:
                print("  [Fix] Injecting missing left turn connection: Lane 1 -> Lane 1")
                new_conn = ET.SubElement(root, 'connection')
                new_conn.set('from', source_edge)
                new_conn.set('to', dest_edge)
                new_conn.set('fromLane', '1')
                new_conn.set('toLane', '1') # Assuming Lane 1 is valid target
                new_conn.set('dir', 'l')
                # Will be assigned TL/Index in loop below
                updates += 1

        for c in root.findall('connection'):
            # Check 'via' attribute to identify internal paths through this junction
            c_via = c.get('via', '')
            # Also check if 'from' edge belongs to the cluster (harder without edge logic, but via is good proxy)
            
            if junction_id in c_via or (c.get('tl') == tls_id and tls_id.startswith('joinedS')):
                if not c.get('tl') or c.get('tl') != tls_id:
                    # Only fix if it's currently uncontrolled or wrong TLS
                    # (Prevent overwriting if it's already correct, but our logic assumes 
                    # we are the authority).
                    # Netedit sometimes leaves tl=""
                    
                    if not c.get('tl'): # Priorities fixing completely uncontrolled ones
                        c.set('tl', tls_id)
                        c.set('linkIndex', str(next_idx))
                        # print(f"    [Fix] Assigned {c.get('from')}->{c.get('to')} to index {next_idx}")
                        next_idx += 1
                        fixed_conn_count += 1
                        updates += 1
                    elif c.get('tl') != tls_id and c.get('tl') != junction_id: 
                         # If it has a WRONG tls, fix it. 
                         # But be careful not to steal from valid neighbors.
                         # Generally trusted-logic: if via junction matches target, it should be controlled by target.
                         c.set('tl', tls_id)
                         c.set('linkIndex', str(next_idx))
                         next_idx += 1
                         fixed_conn_count += 1
                         updates += 1

        if fixed_conn_count > 0:
            print(f"  [Fix] Assigned TLS/Index to {fixed_conn_count} connections.")
            
        # 4. Extend Logic
        if spec.get('logic_extend') and next_idx > 0:
            required_len = next_idx
            for tl in root.findall('tlLogic'):
                if tl.get('id') == tls_id:
                    phases = tl.findall('phase')
                    if phases:
                        for p in phases:
                            state = p.get('state')
                            if len(state) < required_len:
                                diff = required_len - len(state)
                                p.set('state', state + 'r' * diff)
                                print(f"    [Fix] Extended phase state length by {diff}")
                                updates += 1

    # [NEW] Force Protected Green (G) for -1334905223#2
    # User complains cars don't move (Stuck yielding).
    target_edge = '-1334905223#2'
    target_tls = 'joinedS_655375228_655375239_655375240'
    target_indices = []
    
    for c in root.findall('connection'):
        if c.get('from') == target_edge and c.get('tl') == target_tls:
             idx = c.get('linkIndex')
             if idx: target_indices.append(int(idx))

    if target_indices:
        print(f"  [Fix] Upgrading signals to 'G' for edge {target_edge} (indices: {target_indices})")
        for tl in root.findall('tlLogic'):
            if tl.get('id') == target_tls:
                for p in tl.findall('phase'):
                    state = list(p.get('state'))
                    changed = False
                    for idx in target_indices:
                        if idx < len(state) and state[idx] == 'g':
                            state[idx] = 'G'
                            changed = True
                    if changed:
                         p.set('state', "".join(state))
                         updates += 1



    # [NEW] Create Exclusive Right Turn Phase (Phase 9)
    # Problem: Straight (Idx 1) and Right (Idx 14,15,16) are both G in Phase 3 & 9.
    # Result: Right yields to Straight forever.
    # Fix: Set Straight (Idx 1) to Red in Phase 9, giving Right Turn exclusive flows.
    target_tls = 'joinedS_655375228_655375239_655375240'
    straight_idx = 1
    target_phase_idx = 9
    
    for tl in root.findall('tlLogic'):
        if tl.get('id') == target_tls:
            phases = tl.findall('phase')
            if len(phases) > target_phase_idx:
                p = phases[target_phase_idx]
                state = list(p.get('state'))
                if len(state) > straight_idx and state[straight_idx].lower() == 'g':
                    print(f"  [Fix] Setting Straight Traffic (Idx {straight_idx}) to Red in Phase {target_phase_idx} for Exclusive Turn.")
                    state[straight_idx] = 'r'
                    p.set('state', "".join(state))
                    updates += 1

    # [NEW] Explicit Cleanup
    # Remove unwanted connection from -1334905223#2 Lane 0 (User says it's a sidewalk)
    # netconvert -x might not have cleared it if it came from -s
    toremove = []
    for c in root.findall('connection'):
        # Remove Lane 0 (Sidewalk)
        if c.get('from') == '-1334905223#2' and c.get('fromLane') == '0':
            print("  [Cleanup] Removing unwanted connection from Lane 0 (Sidewalk)")
            toremove.append(c)
            updates += 1
        # Remove Lane 1 Straight (User wants exclusive usage)
        # Lane 1 should only go Right (to 263217603#0).
        # Remove if to == '-1334905225'
        if c.get('from') == '-1334905223#2' and c.get('fromLane') == '1' and c.get('to') == '-1334905225':
            print("  [Cleanup] Removing mixed Straight connection from Lane 1 (Enforcing Exclusive Right)")
            toremove.append(c)
            updates += 1
            
    for c in toremove:
        root.remove(c)


    if updates > 0:
        tree.write(net_file, encoding='UTF-8', xml_declaration=True)
        print(f"\nSaved network file with {updates} fixes.")
    else:
        print("\nNo fixes required.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', '-n', default='legacy')
    args = parser.parse_args()
    fix_cluster_tls(args.network)
