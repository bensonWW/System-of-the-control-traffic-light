"""
Extract Junction ID to TLS ID mappings from network file.
Supports --network argument for selecting legacy or real_world.
"""
import sys
import os
import argparse
import xml.etree.ElementTree as ET
import csv

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.config import get_network_config, PROJECT_ROOT

def extract_mapping(net_file: str, output_file: str):
    """Extract junction to TLS ID mapping from network file."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    mappings = []
    
    for junction in root.findall('.//junction'):
        jid = junction.get('id', '')
        jtype = junction.get('type', '')
        
        if jtype != 'traffic_light':
            continue
        
        x = junction.get('x', '0')
        y = junction.get('y', '0')
        
        tls_ids = set()
        
        # Get incoming edges
        incoming_edges = set()
        for edge in root.findall('.//edge'):
            eid = edge.get('id', '')
            if eid.startswith(':'): continue
            if edge.get('to') == jid:
                incoming_edges.add(eid)
        
        # Find TLS IDs from connections
        for conn in root.findall('.//connection'):
            if conn.get('from') in incoming_edges:
                tl = conn.get('tl')
                if tl:
                    tls_ids.add(tl)
        
        # Check tlLogic
        tl_logic = root.find(f".//tlLogic[@id='{jid}']")
        if tl_logic is not None:
            tls_ids.add(jid)
        
        # Determine TLS ID
        if len(tls_ids) == 0:
            tls_id = "NO_TLS"
        elif len(tls_ids) == 1:
            tls_id = list(tls_ids)[0]
        else:
            tls_id = ";".join(sorted(tls_ids))
        
        conn_count = len([c for c in root.findall('.//connection') if c.get('tl') == tls_id])
        
        mappings.append({
            'junction_id': jid,
            'tls_id': tls_id,
            'x': x,
            'y': y,
            'connections': conn_count
        })
    
    # Sort and save
    mappings.sort(key=lambda m: m['junction_id'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['junction_id', 'tls_id', 'x', 'y', 'connections'])
        writer.writeheader()
        writer.writerows(mappings)
    
    return mappings

def main():
    parser = argparse.ArgumentParser(description="Extract Junction-TLS mapping")
    parser.add_argument("--network", "-n", choices=["legacy", "real_world"], 
                        default="legacy", help="Which network to use")
    args = parser.parse_args()
    
    config = get_network_config(args.network)
    net_file = config["net_path"]
    output_file = config["mapping_path"]
    
    print(f"Network: {config['name']}")
    print(f"Input: {net_file}")
    print(f"Output: {output_file}")
    
    if not os.path.exists(net_file):
        print(f"Error: Network file not found: {net_file}")
        return
    
    mappings = extract_mapping(net_file, output_file)
    
    # Summary
    no_tls = [m for m in mappings if m['tls_id'] == 'NO_TLS']
    has_tls = [m for m in mappings if m['tls_id'] != 'NO_TLS']
    different_id = [m for m in has_tls if m['junction_id'] != m['tls_id']]
    
    print(f"\n=== Summary ===")
    print(f"Total junctions: {len(mappings)}")
    print(f"With TLS: {len(has_tls)}")
    print(f"Without TLS: {len(no_tls)}")
    print(f"Different ID: {len(different_id)}")
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()
