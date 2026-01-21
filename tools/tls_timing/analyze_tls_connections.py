"""
Analyze TLS connections to determine direction and type (straight/left/right).
"""
import xml.etree.ElementTree as ET
import sys

def analyze_tls(net_file, tls_id):
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    connections = []
    # Find all connections controlled by this TLS
    for conn in root.findall('connection'):
        if conn.get('tl') == tls_id:
            connections.append(conn.attrib)
            
    # Sort by linkIndex
    connections.sort(key=lambda x: int(x.get('linkIndex')))
    
    print(f"Analysis for TLS: {tls_id}")
    print(f"{'Index':<6} {'From':<20} {'To':<20} {'Dir':<5} {'Link Index'}")
    print("-" * 60)
    
    max_index = 0
    if connections:
        max_index = int(connections[-1].get('linkIndex'))
        
    for c in connections:
        idx = c.get('linkIndex')
        dir_ = c.get('dir')
        print(f"{idx:<6} {c.get('from')[:18]:<20} {c.get('to')[:18]:<20} {dir_:<5} {c.get('linkIndex')}")

    print(f"\nMax Index found: {max_index}")
    return connections

if __name__ == "__main__":
    net = 'data/legacy/ntut-the way.net.xml'
    # Default ID or argument
    tid = 'GS_cluster_9383263990_9383263991_9383263992_9383263993' 
    if len(sys.argv) > 1:
        tid = sys.argv[1]
    
    analyze_tls(net, tid)
