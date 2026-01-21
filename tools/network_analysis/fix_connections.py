"""
Remove connections for edges that were connected through junction 655375336.
"""
import xml.etree.ElementTree as ET

JUNCTION_ID = "655375336"

# First, get the edge IDs that were connected to this junction
edg_file = "data/legacy/plain.edg.xml"
tree = ET.parse(edg_file)
root = tree.getroot()

# Find edges that NOW point to new dead_end nodes (meaning they were connected to the junction)
affected_edges = set()
for edge in root.findall('.//edge'):
    eid = edge.get('id', '')
    from_node = edge.get('from', '')
    to_node = edge.get('to', '')
    
    # Check if from/to points to a split node
    if JUNCTION_ID in from_node or JUNCTION_ID in to_node:
        affected_edges.add(eid)
        print(f"Affected edge: {eid}")

print(f"\nTotal affected edges: {len(affected_edges)}")

# Now remove connections that have these edges
con_file = "data/legacy/plain.con.xml"
tree = ET.parse(con_file)
root = tree.getroot()

conns_to_remove = []
for conn in root.findall('.//connection'):
    from_edge = conn.get('from', '')
    to_edge = conn.get('to', '')
    
    # If connection involves an affected edge, remove it
    if from_edge in affected_edges or to_edge in affected_edges:
        conns_to_remove.append(conn)

for conn in conns_to_remove:
    root.remove(conn)

tree.write(con_file, encoding='UTF-8', xml_declaration=True)
print(f"Removed {len(conns_to_remove)} connections from {con_file}")

print("\nNow retry: netconvert --node-files data/legacy/plain.nod.xml --edge-files data/legacy/plain.edg.xml --connection-files data/legacy/plain.con.xml --tllogic-files data/legacy/plain.tll.xml -o data/legacy/ntut-the_way_split.net.xml")
