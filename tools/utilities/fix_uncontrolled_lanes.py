import xml.etree.ElementTree as ET
import os

def fix_uncontrolled_lanes(net_file):
    print(f"Scanning {net_file} for uncontrolled connections in known clusters...")
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    updates = 0
    
    # Configuration
    targets = {
        "cluster619136899_619136900_cluster_619136929_619136942": {
            "start_index": 62, 
            "nodes": ["619136899", "619136900", "619136929", "619136942"]
        },
        "cluster655375228_655375240": {
            "start_index": 9,
            "nodes": ["655375228", "655375240", "655375239"]
        }
    }
    
    for tls_id, config in targets.items():
        start_index = config["start_index"]
        nodes = config["nodes"]
        
        # 1. Identify Target Edges
        edge_to_node = {}
        for e in root.findall('edge'):
            to = e.get('to')
            if to: edge_to_node[e.get('id')] = to
            
        target_edges = set()
        for eid, node in edge_to_node.items():
            if node in nodes: target_edges.add(eid)
        for c in root.findall('connection'):
            if c.get('tl') == tls_id:
                target_edges.add(c.get('from'))
        
        print(f"\nTargeting Cluster {tls_id}")
        
        # 2. Find Max Index (Dynamic)
        current_max = -1
        for c in root.findall('connection'):
             if c.get('tl') == tls_id:
                 idx = int(c.get('linkIndex') or -1)
                 if idx > current_max: current_max = idx
        
        if current_max >= start_index:
             start_index = current_max + 1
            
        assigned_count = 0
        for c in root.findall('connection'):
            c_from = c.get('from')
            # Fix everything (Do NOT filter internal here)
            if c_from in target_edges:
                tl = c.get('tl')
                if not tl or (tl != tls_id and "cluster" not in (tl or "")):
                    c.set('tl', tls_id)
                    c.set('linkIndex', str(start_index))
                    print(f"  Fixed: {c_from}->{c.get('to')} (Dir:{c.get('dir')}) => Index {start_index}")
                    start_index += 1
                    assigned_count += 1
                    updates += 1
        
        print(f"  Applied {assigned_count} fixes for this cluster.")

    if updates > 0:
        tree.write(net_file, encoding='UTF-8', xml_declaration=True)
        print(f"\nSaved {net_file} with {updates} total fixes.")
    else:
        print("\nNo new fixes applied to Net File.")

    # Always export SAFE connections to .con.xml
    con_file = os.path.join(os.path.dirname(net_file), 'fixed_connections.con.xml')
    with open(con_file, 'w', encoding='utf-8') as f:
        f.write('<connections>\n')
        count = 0
        for c in root.findall('connection'):
            if c.get('tl') in targets:
                # Filter Internal (Source OR Dest)
                if c.get('to').startswith(':') or c.get('from').startswith(':'):
                    continue
                f.write(f'    <connection from="{c.get("from")}" to="{c.get("to")}" fromLane="{c.get("fromLane")}" toLane="{c.get("toLane")}" tl="{c.get("tl")}" linkIndex="{c.get("linkIndex")}" dir="{c.get("dir")}"/>\n')
                count += 1
        f.write('</connections>\n')
    print(f"Exported {count} safe connections to {con_file}")

if __name__ == "__main__":
    net_path = os.path.join('data', 'legacy', 'ntut_network_split.net.xml')
    fix_uncontrolled_lanes(net_path)
