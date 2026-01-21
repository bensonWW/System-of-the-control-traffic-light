
import xml.etree.ElementTree as ET

import math
import os
import sys
import json
import re
import argparse

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.config import get_network_config, PROJECT_ROOT

def get_edge_angle(shape_str):
    coords = [list(map(float, p.split(","))) for p in shape_str.split(" ")]
    if len(coords) < 2: return 0
    p2 = coords[-1]; p1 = coords[-2]
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360

def get_cardinal_direction(angle):
    if 45 <= angle < 135: return "North"
    elif 135 <= angle < 225: return "West"
    elif 225 <= angle < 315: return "South"
    else: return "East"

def get_tls_info(root):
    info = {}
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        phases = tl.findall("phase")
        if phases:
            state = phases[0].get("state")
            if state:
                info[tid] = len(state)
    return info

def extract_topology(root, tls_id):
    edges = {}
    for edge in root.findall("edge"):
        eid = edge.get("id")
        lane = edge.find("lane")
        if lane is not None: edges[eid] = lane.get("shape")
            
    topology = []
    active_indices = set()
    
    for conn in root.findall("connection"):
        if conn.get("tl") == tls_id:
            idx = int(conn.get("linkIndex"))
            from_edge = conn.get("from")
            to_lane = conn.get("to")
            turn_dir = conn.get("dir", "s").lower()
            
            shape = edges.get(from_edge, "")
            angle = get_edge_angle(shape)
            direction = get_cardinal_direction(angle)
            
            axis = "NS" if direction in ["North", "South"] else "EW"
            is_left = (turn_dir == "l")
            group = f"{axis}_{'L' if is_left else 'S'}"
            
            topology.append({
                "index": idx,
                "from": from_edge,
                "to": to_lane,
                "turn": turn_dir,
                "cardinal": direction,
                "group": group
            })
            active_indices.add(idx)

    return active_indices, topology

def generate_program_xml(tls_id, active_indices, topology, expected_len, subplans):
    sorted_subplans = sorted(subplans, key=lambda x: x.get('green', 0), reverse=True)
    num_phases = len(sorted_subplans)
    
    groups = {"NS_S": set(), "NS_L": set(), "EW_S": set(), "EW_L": set()}
    for t in topology:
        g = t["group"]
        groups[g].add(t["index"])

    # Phase Assignments
    active_group_names = [g for g in ["NS_S", "EW_S", "NS_L", "EW_L"] if groups[g]]
    phase_assignments = [set() for _ in range(num_phases)]
    
    if num_phases == 2 and len(active_group_names) > 2:
        ns_size = len(groups["NS_S"] | groups["NS_L"])
        ew_size = len(groups["EW_S"] | groups["EW_L"])
        main, cross = ("NS", "EW") if ns_size >= ew_size else ("EW", "NS")
        phase_assignments[0].update(groups[f"{main}_S"] | groups[f"{main}_L"])
        phase_assignments[1].update(groups[f"{cross}_S"] | groups[f"{cross}_L"])
    else:
        sorted_groups = sorted(active_group_names, key=lambda g: len(groups[g]), reverse=True)
        for i, g_name in enumerate(sorted_groups):
            phase_assignments[i % num_phases].update(groups[g_name])

    # --- AGGRESSIVE COVERAGE CHECK ---
    # Ensure every index 0..expected_len is assigned to AT LEAST one phase
    covered = set()
    for s in phase_assignments: covered.update(s)
    
    missing_indices = []
    for idx in range(expected_len):
        if idx not in covered:
            missing_indices.append(idx)
    
    if missing_indices:
        # Force add to Phase 0 (Main Phase)
        phase_assignments[0].update(missing_indices)
        # Also add to topology logically for consistency (optional but good for debug)
        # print(f"TLS {tls_id}: Forced indices {missing_indices} to Phase 0")

    phases_xml = []
    is_cluster = "joined" in tls_id or "cluster" in tls_id
    
    for i, sp in enumerate(sorted_subplans):
        duration_g = sp.get('green', 10)
        duration_y = sp.get('yellow', 3)
        duration_r = sp.get('allred', 2)
        if duration_g > 0 and duration_y < 3: duration_y = 3
        
        green_indices = phase_assignments[i]
        
        # State Construction
        state_g = ['r'] * expected_len
        
        for idx in range(expected_len):
            if idx not in green_indices: continue  # <--- CRITICAL FIX: Only set green for active indices
            
            # Standard Logic: Left = Yield (g), Straight/Right = Priority (G)
            is_left = any(t["turn"] == 'l' for t in topology if t["index"] == idx)
            
            # SAFETY OVERRIDE: For Joined/Cluster TLS, use 'g' for ALL movements to let SUMO internal logic handle right-of-way.
            # This prevents "Unsafe green phase... Lane is targeted by 2 'G'-links" warnings.
            if is_cluster:
                 state_g[idx] = 'g'
            else:
                 state_g[idx] = 'g' if is_left else 'G'

        if duration_g > 0:
            phases_xml.append(f'        <phase duration="{duration_g}" state="{"".join(state_g)}"/>')
        
        state_y = ['r'] * expected_len
        for idx in range(expected_len):
            if idx in green_indices: state_y[idx] = 'y'
        if duration_y > 0:
            phases_xml.append(f'        <phase duration="{duration_y}" state="{"".join(state_y)}"/>')

        state_r = ['r'] * expected_len
        if duration_r > 0:
            phases_xml.append(f'        <phase duration="{duration_r}" state="{"".join(state_r)}"/>')

    xml_str = f'    <tlLogic id="{tls_id}" type="static" programID="integrated_0" offset="0">\n'
    xml_str += '        <param key="tls.ignore-internal-junction-jam" value="true"/>\n'
    xml_str += '\n'.join(phases_xml)
    xml_str += '\n    </tlLogic>'
    return xml_str

def find_controlling_tls(root, junction_id):
    junction = None
    for j in root.findall("junction"):
        if j.get("id") == junction_id:
            junction = j; break
    if not junction: return None
    inc_edges = set()
    for lane in junction.get("incLanes", "").split(" "):
        if "_" in lane: inc_edges.add(lane.rsplit("_", 1)[0])
    candidates = {}
    for conn in root.findall("connection"):
        if conn.get("from") in inc_edges:
            tl = conn.get("tl")
            if tl: candidates[tl] = candidates.get(tl, 0) + 1
    return max(candidates, key=candidates.get) if candidates else None

def main(network: str = "legacy"):
    config = get_network_config(network)
    net_xml = config["net_path"]
    mapping_csv = config["mapping_path"]
    timing_json = os.path.join(PROJECT_ROOT, "data", "source", "integrated_timing_plan.json")
    output_add_xml = os.path.join(PROJECT_ROOT, "data", network, "traffic_light_integrated.add.xml")
    
    print(f"Loading data for {config['name']}...")
    print(f"  Network: {net_xml}")
    print(f"  Mapping: {mapping_csv}")
    
    tree = ET.parse(net_xml)
    root = tree.getroot()
    matches = []
    import csv
    with open(mapping_csv, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches.append(row)
    
    with open(timing_json, 'r', encoding='utf-8') as f:
        timing_map = {r['icid']: r['subplan'] for r in json.load(f) if 'subplan' in r and r.get('icid')}
    
    tls_info = get_tls_info(root)
    available_tls_ids = set(tls_info.keys())
    
    output = ['<additional>']
    processed = set(); count_gen = 0
    
    for row in matches:
        sumo_id = str(row.get('tls_id', row.get('sumo_id', '')))
        icid = row.get('icid', '')
        if not sumo_id or not icid:
            continue
        target = sumo_id if sumo_id in available_tls_ids else find_controlling_tls(root, sumo_id)
        if not target: target = sumo_id
        
        if target in processed: continue
        if icid not in timing_map: continue
        
        active_indices, topo = extract_topology(root, target)
        
        expected_len = tls_info.get(target, max(active_indices)+1 if active_indices else 0)
        
        if expected_len == 0:
            print(f"Skipping {target}: No links found.")
            continue
            
        xml_block = generate_program_xml(target, active_indices, topo, expected_len, timing_map[icid])
        output.append(xml_block)
        processed.add(target)
        count_gen += 1
        
    output.append('</additional>')
    with open(output_add_xml, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    print(f"Generated {count_gen} programs to {output_add_xml}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply integrated timing plans to SUMO network")
    parser.add_argument('--network', '-n', choices=['legacy', 'real_world'], default='legacy',
                        help='Network to process (default: legacy)')
    args = parser.parse_args()
    main(args.network)
