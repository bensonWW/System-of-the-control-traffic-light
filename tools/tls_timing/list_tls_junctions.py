"""
List all traffic_light junctions in a network.
Supports --network argument for selecting legacy or real_world.
"""
import sys
import os
import xml.etree.ElementTree as ET

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.config import get_network_config, parse_network_arg, PROJECT_ROOT

def main():
    network = parse_network_arg()
    config = get_network_config(network)
    
    net_file = config["net_path"]
    
    print(f"Listing TLS junctions for: {config['name']}")
    print(f"Network file: {net_file}")
    
    if not os.path.exists(net_file):
        print(f"Error: Network file not found: {net_file}")
        return
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Find all traffic_light junctions
    tls_junctions = []
    for j in root.findall('.//junction'):
        if j.get('type') == 'traffic_light':
            jid = j.get('id')
            tls_junctions.append(jid)
    
    lines = [f"=== Traffic Light Junctions ({len(tls_junctions)} total) ===\n"]
    for i, jid in enumerate(tls_junctions, 1):
        # Check if tlLogic exists
        tl = root.find(f".//tlLogic[@id='{jid}']")
        # Count controlled connections
        conns = len([c for c in root.findall('.//connection') if c.get('tl') == jid])
        status = "OK" if tl is not None and conns > 0 else "NO"
        lines.append(f"{i:2}. [{status}] {jid}  (conns: {conns})")
    
    # Save to file in network directory
    output_file = os.path.join(config["dir"], "tls_list.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nSaved to: {output_file}")
    print("\n".join(lines[:10]))
    if len(lines) > 10:
        print(f"... and {len(lines) - 10} more")

if __name__ == "__main__":
    main()
