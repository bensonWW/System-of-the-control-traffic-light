"""
Updated TLS update script with smart phase generation logic.
Parses network connections to determine Safe Green, Permitted Green, and proper transitions.
"""
import xml.etree.ElementTree as ET
import sys
import os
import csv
import json
import math
import argparse
from datetime import datetime
import re

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tools.tls_timing.timing_schedule as timing_schedule


def get_tls_connections(net_file, tls_id):
    """
    Parse the network file to get connections for the given TLS ID.
    Calculates incoming angle to classify directions.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Pre-load Edges and Lanes to check permissions and shapes
    edges = {}
    for edge in root.findall('edge'):
        eid = edge.get('id')
        lanes = {}
        for lane in edge.findall('lane'):
            lid = lane.get('id')
            index = lane.get('index')
            lanes[index] = {
                'id': lid,
                'allow': lane.get('allow'),
                'disallow': lane.get('disallow'),
                'shape': lane.get('shape')
            }
        
        # Get Edge Shape (fallback to lane 0)
        shape_str = edge.get('shape')
        if not shape_str and '0' in lanes:
            shape_str = lanes['0']['shape']
            
        coords = []
        if shape_str:
            for pair in shape_str.split():
                try:
                    x, y = map(float, pair.split(','))
                    coords.append((x, y))
                except: pass
        
        edges[eid] = {
            'shape': coords,
            'lanes': lanes
        }

    # 1. Get raw connections
    raw_conns = []
    max_index = 0
    for conn in root.findall('connection'):
        if conn.get('tl') == tls_id:
            idx = int(conn.get('linkIndex'))
            if idx > max_index:
                max_index = idx
            
            from_edge = conn.get('from')
            to_edge = conn.get('to')
            from_lane_idx = conn.get('fromLane')
            
            # Check if Pedestrian
            is_pedestrian = False
            if from_edge in edges:
                lane_info = edges[from_edge]['lanes'].get(from_lane_idx)
                if lane_info:
                    allow = lane_info.get('allow') or ''
                    if 'pedestrian' in allow:
                        is_pedestrian = True
            
            # Pedestrians usually on internal edges or crossings
            # If to_edge is a crossing, it is definitely pedestrian
            # But checking 'allow' is safer.
            
            raw_conns.append({
                'index': idx,
                'from_edge': from_edge,
                'to_edge': to_edge,
                'dir': conn.get('dir'), 
                'lane': from_lane_idx,
                'is_pedestrian': is_pedestrian
            })

    # 3. Calculate Angles and Classify
    connections = []
    for c in raw_conns:
        # For Pedestrians, use TO shape (Crossing direction)
        # For Vehicles, use FROM shape (Incoming direction)
        target_eid = c['to_edge'] if c['is_pedestrian'] else c['from_edge']
        
        angle = -1 # Default invalid
        
        if target_eid in edges and len(edges[target_eid]['shape']) >= 2:
            coords = edges[target_eid]['shape']
            
            if c['is_pedestrian']:
                # Walking direction: Start to End of crossing
                p1 = coords[0]
                p2 = coords[-1]
            else:
                # Vehicle Incoming: 2nd last to Last
                p1 = coords[-2]
                p2 = coords[-1]
                
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            deg = math.degrees(math.atan2(dy, dx))
            if deg < 0: deg += 360
            angle = deg
        
        # Classify Arm
        # If angle is invalid (-1), default to U (Unknown)
        if angle == -1:
            arm = 'U'
        elif (angle >= 315 or angle < 45): arm = 'E' 
        elif (angle >= 45 and angle < 135): arm = 'N' 
        elif (angle >= 135 and angle < 225): arm = 'W' 
        elif (angle >= 225 and angle < 315): arm = 'S'
        else: arm = 'U' # Should not happen
        
        c['angle'] = angle if angle != -1 else 0 # Keep 0 for safety in other logic, but arm is U
        c['arm'] = arm
        connections.append(c)
        
    return connections, max_index


def generate_tls_phases_smart(tls_id, timing_plan, connections, max_index):
    """
    Generate phases using smart classification logic.
    Ensures Red-Yellow-Green-Red transitions.
    """
    # 1. Group indices by Arm and Type
    # arms = {'N': {'s': [], 'l': [], 'r': []}, ...}
    arms = {d: {'s': [], 'l': [], 'r': [], 'u': [], 't': [], 'other': []} for d in ['N', 'S', 'E', 'W', 'U']}
    
    for c in connections:
        d = c['dir']
        # normalize dir
        if d not in ['s', 'l', 'r', 'u', 't']: d = 'other'
        if d == 't': d = 'l' # Treat turn as left often? Or check to/from. Assume turning left usually.
        
        arms[c['arm']][d].append(c['index'])

    # 2. Assign Indices to Main Phases
    # Phase A (N/S): Straight/Right = G. Left = g (if conflict) or G (if safe/protected).
    # Phase B (E/W): Straight/Right = G. ...
    
    # Analyze if we need permitted lefts
    # If N has Straight and S has Straight, then N-Left and S-Left should be 'g' (yield).
    # If only N has Straight (T-intersection), S-Left (if exists) might be 'G' (protected)?
    # For safety/simplicity in generated code:
    # Use 'g' for Left turns if there is *any* opposing traffic.
    
    def get_phase_state(active_arms):
        # 'active_arms' = ['N', 'S'] or ['E', 'W']
        # Returns a list of chars ['r'] * (max_index+1) with G/g set
        state = ['r'] * (max_index + 1)
        
        for arm in active_arms:
            # Straight = G
            for idx in arms[arm]['s']:
                state[idx] = 'G'
            
            # Right Turn = g (Yield) - Fixes "Unsafe green... targeted by 2 G links"
            for idx in arms[arm]['r']:
                state[idx] = 'g'
            
            # Left Turns
            opp_arm = {'N':'S', 'S':'N', 'E':'W', 'W':'E'}.get(arm)
            # If opposite has Straight traffic, usually yield (g).
            # BUT, for major intersections in Taiwan, Left is often Protected (G)
            # or yielding causes gridlock with Peds. 
            # Strategy: Default to G (Protected) to solve 'stuck' issue.
            # has_opposing_straight = (len(arms[opp_arm]['s']) > 0)
            # left_signal = 'g' if has_opposing_straight else 'G'
            left_signal = 'G' # Force Protected Left
            
            for idx in arms[arm]['l'] + arms[arm]['t'] + arms[arm]['u']:
                state[idx] = left_signal
                
            # Internal/Others associated with this arm
            for idx in arms[arm]['other']:
                state[idx] = 'g' 
        
        # Handle Unclassified (U) Indices
        # Strategy: Assign 'g' during the "Main" phase (Phase 1 / N-S).
        # We need to know if we are in the "Main" phase.
        # Heuristic: If 'N' or 'S' is active, AND N/S actually exist in this intersection.
        # If intersection is only E-W (e.g. straight road with Ped crossing), then N/S might not exist.
        
        has_ns = (len(arms['N']['s'])+len(arms['N']['l'])+len(arms['N']['r']) + 
                  len(arms['S']['s'])+len(arms['S']['l'])+len(arms['S']['r'])) > 0
                  
        is_ns_phase = ('N' in active_arms or 'S' in active_arms)
        
        # If we have N/S arms, give U green during N/S phase.
        # If we don't have N/S arms (e.g. only E/W), give U green during E/W phase?
        # Or just give U green in Phase 1 always? (arm_groups[0] is N/S)
        
        assign_u = False
        if has_ns and is_ns_phase: assign_u = True
        elif not has_ns and ('E' in active_arms): assign_u = True # Fallback for E-W only nodes
        
        if assign_u:
            for idx in arms['U']['s'] + arms['U']['l'] + arms['U']['r'] + arms['U']['other']:
                 state[idx] = 'g'
                 
        return state

    phases_xml = []
    
    # Decide which Plan Phase maps to which Arms
    # Standard Taipei: Phase 1 usually Main St (N/S?), Phase 2 Cross St (E/W?)
    # We need a heuristic or just rotate.
    # Plan 1: N/S. Plan 2: E/W. 
    # If Plan has 3 phases? maybe N/S(L) protected?
    # Given we only have timing *duration*, not semantic meaning...
    # We assume standard 2-phase: 1=Main, 2=Cross.
    # What is Main? Usually the one with more lanes or flux?
    # Or just N/S = Phase 1.
    
    # We will simply toggle:
    # Subplan 1 -> Arms Group 1 (N/S)
    # Subplan 2 -> Arms Group 2 (E/W)
    # Subplan 3 -> Arms Group 1 (Repeat? or Lefts?) -> Safety: Repeat N/S for now or Lefts?
    # This is the tricky part without semantic data.
    
    arm_groups = [['N', 'S'], ['E', 'W']]
    
    subplans = timing_plan.get('subplan', [])
    if not subplans:
        return ""

    for i, sub in enumerate(subplans):
        # Determine active arms
        # Cycle through [N,S], [E,W], [N,S]...
        active_arms = arm_groups[i % 2] 
        
        green_time = float(sub.get('green', 30))
        yellow_time = float(sub.get('yellow', 3))
        allred_time = float(sub.get('allred', 2))
        
        # Enforce minimums for safety
        if yellow_time < 3: yellow_time = 3
        if allred_time < 2: allred_time = 2
        
        # 1. Green Phase
        state_g = get_phase_state(active_arms)
        if green_time > 0:
            phases_xml.append(f'        <phase duration="{int(green_time)}" state="{"".join(state_g)}"/>')
        
        # 2. Yellow Phase
        # Convert G/g -> y
        state_y = ['y' if s in ['G', 'g'] else s for s in state_g]
        phases_xml.append(f'        <phase duration="{int(yellow_time)}" state="{"".join(state_y)}"/>')
        
        # 3. All-Red Phase
        state_r = ['r'] * (max_index + 1)
        phases_xml.append(f'        <phase duration="{int(allred_time)}" state="{"".join(state_r)}"/>')

    return "\n".join(phases_xml)


def update_tls_for_network(network_path, output_path, mapping_path, time_override=None):
    print(f"Updating TLS for network: {network_path}")
    
    # 1. Load Mappings
    # Same as before...
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8-sig') as f:
             reader = csv.DictReader(f)
             for row in reader:
                 # Support both 'sumo_id' and 'tls_id' headers
                 sid = row.get('sumo_id', row.get('tls_id', '')).strip()
                 icid = row.get('icid', '').strip()
                 if sid and icid: mapping[sid] = icid
    
    # 2. Identify Network TLS
    tree = ET.parse(network_path)
    root = tree.getroot()
    tls_ids = [tl.get('id') for tl in root.findall('.//tlLogic')]
    
    # 3. Process
    xml_content = ["<additional>"]
    generated_tls_ids = []  # Track for WAUT generation
    
    count = 0
    current_time = time_override if time_override else datetime.now()
    
    for tid in tls_ids:
        # Match ID
        icid = mapping.get(tid)
        if not icid:
             # Try partial
             for m_sid, m_icid in mapping.items():
                 if m_sid in tid:
                     icid = m_icid
                     break
        
        if not icid:
            continue
            
        # Get Original Type from Network
        # Locate the tlLogic element
        tl_node = root.find(f".//tlLogic[@id='{tid}']")
        tls_type = tl_node.get('type', 'static') if tl_node is not None else 'static'

        # Get Connections
        conns, max_idx = get_tls_connections(network_path, tid)
        
        # Get Plan
        plan_id = timing_schedule.get_plan_for_time(icid, current_time)
        details = timing_schedule.get_timing_details(icid, plan_id)
        
        if not details: continue
        
        # Generate XML
        # Use programID="1" to avoid conflict with network's programID="0".
        # We will use WAUT to auto-switch from 0 to 1 at simulation start.
        xml_content.append(f'    <tlLogic id="{tid}" type="{tls_type}" programID="1" offset="0">')
        xml_content.append(f'        <param key="tls.ignore-internal-junction-jam" value="true"/>')
        
        phases = generate_tls_phases_smart(tid, details, conns, max_idx)
        xml_content.append(phases)
        
        xml_content.append('    </tlLogic>')
        generated_tls_ids.append(tid)
        count += 1
        print(f"  Generated smart phases for {tid} -> {icid} (Plan {plan_id})")

    # 4. Add WAUT (Waveform Areawide Uniform Traffic) for auto-switching
    # This automatically switches from program 0 to program 1 at simulation time 0
    if generated_tls_ids:
        xml_content.append('')
        xml_content.append('    <!-- WAUT: Auto-switch from default (0) to custom plan (1) at simulation start -->')
        xml_content.append('    <WAUT refTime="0" id="taipei_timing" startProg="0">')
        xml_content.append('        <wautSwitch to="1" time="0"/>')
        xml_content.append('    </WAUT>')
        
        for tid in generated_tls_ids:
            xml_content.append(f'    <wautJunction wautID="taipei_timing" junctionID="{tid}" procedure="Stretch"/>')

    xml_content.append("</additional>")
    
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("\n".join(xml_content))
    
    print(f"Written {count} TLS programs to {output_path} (with WAUT auto-switch)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=['legacy', 'realworld'], default='legacy')
    parser.add_argument('--time', help='Override time "YYYY-MM-DD HH:MM"')
    args = parser.parse_args()
    
    # Configuration
    NETWORKS = {
        "legacy": {
            "net_xml": "data/legacy/ntut_network_split.net.xml",
            "output_add_xml": "data/legacy/traffic_light.add.xml",
            "mapping_csv": "data/legacy/ntut_mapping.csv"
        },
        "realworld": {
            "net_xml": "data/real_world/realworld_network.net.xml",
            "output_add_xml": "data/real_world/traffic_light.add.xml",
            "mapping_csv": "data/real_world/realworld_mapping.csv"
        }
    }
    
    nw = NETWORKS[args.network]
    t_obj = None
    if args.time:
        if " " in args.time: t_obj = datetime.strptime(args.time, "%Y-%m-%d %H:%M")
        else: t_obj = datetime.combine(datetime.today(), datetime.strptime(args.time, "%H:%M").time())
        print(f"Using override time: {t_obj}")
        
    update_tls_for_network(nw['net_xml'], nw['output_add_xml'], nw['mapping_csv'], t_obj)
