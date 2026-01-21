"""
Add traffic lights to specified junctions.
Supports --network argument and --junctions for junction IDs.
"""
import sys
import os
import argparse
import xml.etree.ElementTree as ET

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.config import get_network_config, PROJECT_ROOT

def add_tls_to_junctions(net_file: str, junction_ids: list, output_file: str = None):
    """
    Add traffic lights to specified junctions.
    
    Args:
        net_file: Path to network file
        junction_ids: List of junction IDs to add TLS to
        output_file: Output file path (defaults to overwriting input)
    """
    if output_file is None:
        output_file = net_file
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    for junction_id in junction_ids:
        print(f"\n=== Processing: {junction_id} ===")
        
        # Find junction
        junction = root.find(f".//junction[@id='{junction_id}']")
        if junction is None:
            print(f"  Junction not found!")
            continue
        
        # Get current type
        current_type = junction.get('type')
        print(f"  Current type: {current_type}")
        
        # Set to traffic_light if not already
        if current_type != 'traffic_light':
            junction.set('type', 'traffic_light')
            print(f"  Changed type to: traffic_light")
        
        # Find incoming edges
        incoming_edges = set()
        for edge in root.findall('.//edge'):
            eid = edge.get('id', '')
            if eid.startswith(':'): continue
            if edge.get('to') == junction_id:
                incoming_edges.add(eid)
        
        print(f"  Incoming edges: {len(incoming_edges)}")
        
        # Assign tl and linkIndex to connections from incoming edges
        link_index = 0
        connections_fixed = 0
        for conn in root.findall('.//connection'):
            from_edge = conn.get('from')
            if from_edge in incoming_edges:
                conn.set('tl', junction_id)
                conn.set('linkIndex', str(link_index))
                link_index += 1
                connections_fixed += 1
        
        print(f"  Assigned TLS to {connections_fixed} connections")
        
        # Check if tlLogic exists, if not create one
        tl = root.find(f".//tlLogic[@id='{junction_id}']")
        if tl is None and link_index > 0:
            print(f"  Creating tlLogic with {link_index} links")
            
            # Create a simple 2-phase static TLS
            state_len = link_index
            half = state_len // 2
            state1 = 'G' * half + 'r' * (state_len - half)
            state2 = 'r' * half + 'G' * (state_len - half)
            yellow1 = 'y' * half + 'r' * (state_len - half)
            yellow2 = 'r' * half + 'y' * (state_len - half)
            
            tl_logic = ET.Element('tlLogic')
            tl_logic.set('id', junction_id)
            tl_logic.set('type', 'static')
            tl_logic.set('programID', '0')
            tl_logic.set('offset', '0')
            
            for phase_state, duration in [(state1, '30'), (yellow1, '3'), 
                                          (state2, '30'), (yellow2, '3')]:
                phase = ET.SubElement(tl_logic, 'phase')
                phase.set('duration', duration)
                phase.set('state', phase_state)
            
            root.insert(1, tl_logic)
        elif tl is not None:
            print(f"  tlLogic already exists")
        else:
            print(f"  No connections to control - cannot create TLS")
    
    # Save
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"\nSaved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Add traffic lights to junctions")
    parser.add_argument("--network", "-n", choices=["legacy", "real_world"], 
                        default="legacy", help="Which network to use")
    parser.add_argument("--junctions", "-j", nargs="+", required=True,
                        help="Junction IDs to add TLS to")
    args = parser.parse_args()
    
    config = get_network_config(args.network)
    net_file = config["net_path"]
    
    print(f"Network: {config['name']}")
    print(f"File: {net_file}")
    
    if not os.path.exists(net_file):
        print(f"Error: Network file not found: {net_file}")
        return
    
    add_tls_to_junctions(net_file, args.junctions)

if __name__ == "__main__":
    main()
