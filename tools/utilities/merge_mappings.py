"""
Merge junction-TLS mapping with ICID and road names from sumo_json_mapping.
Fixed: Use junction_id as key instead of tls_id to avoid duplicates.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.config import get_network_config, PROJECT_ROOT

def merge_mappings(network: str):
    config = get_network_config(network)
    
    # First, regenerate junction mapping from network file
    import xml.etree.ElementTree as ET
    net_file = config["net_path"]
    
    print(f"Network: {config['name']}")
    print(f"Input: {net_file}")
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Extract all traffic_light junctions
    junctions = []
    for junction in root.findall('.//junction'):
        jid = junction.get('id', '')
        jtype = junction.get('type', '')
        
        if jtype != 'traffic_light':
            continue
        
        x = junction.get('x', '0')
        y = junction.get('y', '0')
        
        # Find TLS ID
        tls_ids = set()
        incoming_edges = set()
        for edge in root.findall('.//edge'):
            eid = edge.get('id', '')
            if eid.startswith(':'): continue
            if edge.get('to') == jid:
                incoming_edges.add(eid)
        
        for conn in root.findall('.//connection'):
            if conn.get('from') in incoming_edges:
                tl = conn.get('tl')
                if tl:
                    tls_ids.add(tl)
        
        tl_logic = root.find(f".//tlLogic[@id='{jid}']")
        if tl_logic is not None:
            tls_ids.add(jid)
        
        if len(tls_ids) == 0:
            tls_id = "NO_TLS"
        elif len(tls_ids) == 1:
            tls_id = list(tls_ids)[0]
        else:
            tls_id = ";".join(sorted(tls_ids))
        
        conn_count = len([c for c in root.findall('.//connection') if c.get('tl') == tls_id])
        
        junctions.append({
            'junction_id': jid,
            'tls_id': tls_id,
            'x': x,
            'y': y,
            'connections': conn_count
        })
    
    print(f"Found {len(junctions)} traffic_light junctions")
    
    # Load ICID mapping - also build a junction-based lookup
    icid_mapping = {}
    junction_to_icid = {}  # Map individual junction IDs to ICID
    icid_file = os.path.join(PROJECT_ROOT, "data", "sumo_json_mapping_fixed.csv")
    if os.path.exists(icid_file):
        with open(icid_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sumo_id = row.get('sumo_id', '')
                icid = row.get('icid', '')
                name = row.get('name', '')
                junction_id_str = row.get('junction_id', '')
                if sumo_id and icid:
                    icid_mapping[sumo_id] = {'icid': icid, 'name': name}
                    # Also map individual junction IDs to ICID
                    if junction_id_str:
                        for jid in junction_id_str.split(';'):
                            if jid.strip():
                                junction_to_icid[jid.strip()] = {'icid': icid, 'name': name}
    
    print(f"Loaded {len(icid_mapping)} ICID mappings, {len(junction_to_icid)} junction mappings")
    
    # Merge ICID into junctions
    for j in junctions:
        tls_id = j['tls_id']
        jid = j['junction_id']
        
        # First try exact TLS ID match
        icid_data = icid_mapping.get(tls_id, {})
        
        # If not found, try junction ID match
        if not icid_data:
            icid_data = junction_to_icid.get(jid, {})
        
        # If still not found, try extracting numbers from TLS ID and match
        if not icid_data and 'joinedS_' in tls_id:
            # Extract junction numbers from TLS ID like "joinedS_622618015_622618108_..."
            import re
            numbers = re.findall(r'\d{6,}', tls_id)
            for num in numbers:
                if num in junction_to_icid:
                    icid_data = junction_to_icid[num]
                    break
        
        j['icid'] = icid_data.get('icid', '')
        j['name'] = icid_data.get('name', '')
    
    # Sort by junction_id
    junctions.sort(key=lambda m: m['junction_id'])
    
    # Write
    mapping_file = config["mapping_path"]
    with open(mapping_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['junction_id', 'tls_id', 'icid', 'name', 'x', 'y', 'connections'])
        writer.writeheader()
        writer.writerows(junctions)
    
    # Stats
    with_icid = len([m for m in junctions if m['icid']])
    print(f"Total: {len(junctions)} junctions, {with_icid} with ICID")
    print(f"Saved to: {mapping_file}")

if __name__ == "__main__":
    print("=" * 50)
    merge_mappings("legacy")
    print()
    print("=" * 50)
    merge_mappings("real_world")
    print("=" * 50)

