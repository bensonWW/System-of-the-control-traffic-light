
import xml.etree.ElementTree as ET
import pandas as pd
import math
import os
import json

NET_XML = "ntut-the way.net.xml"
MAPPING_CSV = "sumo_json_mapping.csv"
TIMING_JSON = "integrated_timing_plan.json"
OUTPUT_ADD_XML = "traffic_light.add.xml"

def get_edge_angle(shape_str):
    # shape="x1,y1 x2,y2 ..."
    # We care about the *end* of the incoming edge or the *start* of the outgoing edge?
    # For incoming edge (fromEdge), we want the angle of the LAST segment pointing towards junction.
    coords = [list(map(float, p.split(","))) for p in shape_str.split(" ")]
    if len(coords) < 2:
        return 0
    p2 = coords[-1]
    p1 = coords[-2]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    # Standardize to 0-360, 0=East, 90=North? 
    # math.atan2(dy,dx): 0=East, 90=North, 180=West, -90=South
    # Let's map to cardinal: 0=E, 90=N, 180=W, 270=S
    if angle < 0:
        angle += 360
    return angle

def get_cardinal_direction(angle):
    # 0=East, 90=North, 180=West, 270=South
    # Allow +/- 45 deg deviation
    if 45 <= angle < 135:
        return "North"
    elif 135 <= angle < 225:
        return "West"
    elif 225 <= angle < 315:
        return "South"
    else:
        return "East"

def extract_topology(root, tls_id):
    # Find all connections controlled by this TLS
    connections = [] # (linkIndex, fromEdge, dir)
    
    # We need edge shapes to determine direction
    edges = {} # id -> shape
    for edge in root.findall("edge"):
        # internal edges don't matter much for 'from', usually standard edges
        eid = edge.get("id")
        # shape is in <lane> child usually, or edge if simplified
        # SUMO .net.xml: <edge ...> <lane shape="..."/></edge>
        # We take the first lane's shape for approximation
        lane = edge.find("lane")
        if lane is not None:
            edges[eid] = lane.get("shape")
            
    # Iterate connections
    link_indices = set()
    
    for conn in root.findall("connection"):
        if conn.get("tl") == tls_id:
            idx = int(conn.get("linkIndex"))
            from_edge = conn.get("from")
            shape = edges.get(from_edge, "")
            angle = get_edge_angle(shape)
            direction = get_cardinal_direction(angle)
            
            connections.append({
                "index": idx,
                "from": from_edge,
                "dir": direction,
                "angle": angle
            })
            link_indices.add(idx)

    # If no explicitly marked connections, we can't generate a program easily.
    # Return grouped indices
    
    # Group by index? No, index IS the group (all connections with same index switch together).
    # We need to group indices by Direction.
    # Usually: 
    # Phase 1: N+S go (indices for N and S edges)
    # Phase 2: E+W go (indices for E and W edges)
    
    # Let's group indices by their dominant direction
    # index_map: index -> set of directions it serves
    
    index_dirs = {}
    for c in connections:
        idx = c["index"]
        if idx not in index_dirs:
            index_dirs[idx] = set()
        index_dirs[idx].add(c["dir"])
        
    return index_dirs, max(link_indices) + 1 if link_indices else 0

    return index_dirs, max(link_indices) + 1 if link_indices else 0

def generate_program_xml(tls_id, index_dirs, total_links, subplans):
    # 1. Strategy: Sort Directions by weight (number of links)
    #    Sort Subplans by Green Time
    #    Map Top Direction -> Top Subplan
    
    # Group indices by 'Main Directions'
    # N+S usually go together, E+W usually go together
    # But let's keep them separate for now, or group N/S and E/W
    
    dir_groups = {
        "NS": set(),
        "EW": set()
    }
    
    for idx, dirs in index_dirs.items():
        if "North" in dirs or "South" in dirs:
            dir_groups["NS"].add(idx)
        if "East" in dirs or "West" in dirs:
            dir_groups["EW"].add(idx)
            
    # Which group is bigger?
    ns_count = len(dir_groups["NS"])
    ew_count = len(dir_groups["EW"])
    
    groups = []
    if ns_count > 0: groups.append( ("NS", dir_groups["NS"]) )
    if ew_count > 0: groups.append( ("EW", dir_groups["EW"]) )
    
    # Sort groups by size desc
    groups.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Sort subplans by green time desc (assuming main street gets more green)
    # subplan is list of dicts
    sorted_subplans = sorted(subplans, key=lambda x: x.get('green', 0), reverse=True)
    
    # Map
    # Problem: what if 4 subplans but only 2 groups?
    # Or 2 subplans but 1 group?
    # We cycle assignments.
    
    phases_xml = []
    
    # Calculate cycle time (sum of green+yellow+allred)
    # Actually each subphase has its own times.
    
    for i, sp in enumerate(sorted_subplans):
        duration_g = sp.get('green', 10)
        duration_y = sp.get('yellow', 3)
        duration_r = sp.get('allred', 2) # All Red
        
        # Determine which links are GREEN in this phase
        # If we have more subplans than groups, we reuse groups or leave some Red?
        # Let's modulo the groups.
        if groups:
            target_group_idx = i % len(groups)
            green_indices = groups[target_group_idx][1]
        else:
            green_indices = set()
            
        # Build State String
        # Length = total_links
        # State: 'G' for green, 'y' for yellow, 'r' for red
        
        # Green Phase
        state_g = ['r'] * total_links
        for idx in green_indices:
            if idx < total_links:
                state_g[idx] = 'G'
        phases_xml.append(f'        <phase duration="{duration_g}" state="{"".join(state_g)}"/>')
        
        # Yellow Phase (Transition)
        state_y = ['r'] * total_links
        for idx in green_indices:
            if idx < total_links:
                state_y[idx] = 'y'
        phases_xml.append(f'        <phase duration="{duration_y}" state="{"".join(state_y)}"/>')
        
        # All Red Phase
        state_r = ['r'] * total_links
        phases_xml.append(f'        <phase duration="{duration_r}" state="{"".join(state_r)}"/>')

    # Construct the full logic element
    xml_str = f'    <tlLogic id="{tls_id}" type="static" programID="integrated_0" offset="0">\n'
    xml_str += '\n'.join(phases_xml)
    xml_str += '\n    </tlLogic>'
    return xml_str

def find_controlling_tls(root, junction_id):
    # Find incoming set of edges for this junction
    junction = None
    for j in root.findall("junction"):
        if j.get("id") == junction_id:
            junction = j
            break
            
    if not junction:
        return None
        
    inc_lanes = junction.get("incLanes", "").split(" ")
    inc_edges = set()
    for lane in inc_lanes:
        if "_" in lane:
            edge = lane.rsplit("_", 1)[0]
            inc_edges.add(edge)
            
    # Check connections originating from these edges
    # We want a connection that has a 'tl' attribute.
    # Count occurrences to find the dominant one if multiple?
    candidates = {}
    
    for conn in root.findall("connection"):
        if conn.get("from") in inc_edges:
            tl = conn.get("tl")
            if tl:
                candidates[tl] = candidates.get(tl, 0) + 1
                
    if not candidates:
        return None
        
    # Return the most frequent TL ID
    best_tl = max(candidates, key=candidates.get)
    return best_tl

def main():
    print("Loading data...")
    tree = ET.parse(NET_XML)
    root = tree.getroot()
    
    matches = pd.read_csv(MAPPING_CSV)
    
    # Load JSON
    with open(TIMING_JSON, 'r', encoding='utf-8') as f:
        timing_data = json.load(f)
    
    # Index JSON by icid for fast lookup
    timing_map = {} # icid -> subplan list
    for record in timing_data:
        # Just take the first valid one or merge? 
        # Typically one ID has one plan for a given time. We use the one in the file.
        icid = record.get('icid')
        if icid and 'subplan' in record:
            timing_map[icid] = record['subplan']
            
    print(f"Loaded {len(timing_map)} timing plans.")
    
    # 2. Collect all available tlLogic IDs from XML for resolution
    available_tls_ids = set()
    for tl in root.findall("tlLogic"):
        available_tls_ids.add(tl.get("id"))
        
    output_content = ['<additional>']
    processed_tls_ids = set()
    count_generated = 0
    count_covered_matches = 0
    
    for _, row in matches.iterrows():
        sumo_id = str(row['sumo_id']) # ensure string
        icid = row['icid']
        
        # Resolve Actual Control ID
        target_id = sumo_id
        
        # Strategy 1: Direct Match
        if sumo_id in available_tls_ids:
            target_id = sumo_id
        else:
            # Strategy 2: Topology Search (Most Robust)
            found_tl = find_controlling_tls(root, sumo_id)
            if found_tl:
                target_id = found_tl
            else:
                 # Strategy 3: Substring fallback (for clusters not yet found?)
                 # usually Strategy 2 covers it if connections exist.
                 pass

        if target_id in processed_tls_ids:
            # We count this as "covered" because the logic for its cluster is generated
            count_covered_matches += 1
            print(f"Match {sumo_id} -> Covered by existing {target_id}")
            continue
            
        if icid not in timing_map:
            print(f"Skipping {sumo_id} -> {icid}: ICID not found in timing map")
            continue
            
        subplans = timing_map[icid]
        if not subplans:
            continue
            
        index_dirs, total_links = extract_topology(root, target_id)
        if total_links == 0:
            print(f"Skipping {sumo_id} (Target: {target_id}): No connections found.")
            continue
            
        # Generate XML
        xml_block = generate_program_xml(target_id, index_dirs, total_links, subplans)
        output_content.append(xml_block)
        processed_tls_ids.add(target_id)
        count_generated += 1
        count_covered_matches += 1
        
    output_content.append('</additional>')
    
    with open(OUTPUT_ADD_XML, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))
        
    print(f"\nSuccessfully generated {OUTPUT_ADD_XML}")
    print(f"Unique Programs Generated: {count_generated}")
    print(f"Total Matches Covered: {count_covered_matches} / {len(matches)}")

if __name__ == "__main__":
    main()
