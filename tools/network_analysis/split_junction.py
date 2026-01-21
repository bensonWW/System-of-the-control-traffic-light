"""
Properly split junction 655375336 using plain XML files.
This modifies plain.nod.xml and plain.edg.xml to remove the junction
and create new dead-end nodes for each connected edge.
"""
import xml.etree.ElementTree as ET

JUNCTION_ID = "655375336"

# === 1. Modify nodes file ===
nod_file = "data/legacy/plain.nod.xml"
tree = ET.parse(nod_file)
root = tree.getroot()

# Find and remove the junction
junction = root.find(f".//node[@id='{JUNCTION_ID}']")
if junction is not None:
    jx = float(junction.get('x'))
    jy = float(junction.get('y'))
    print(f"Found node {JUNCTION_ID} at ({jx}, {jy})")
    root.remove(junction)
    print(f"Removed node {JUNCTION_ID}")
else:
    print(f"Node {JUNCTION_ID} not found!")
    exit(1)

# === 2. Modify edges file ===
edg_file = "data/legacy/plain.edg.xml"
edg_tree = ET.parse(edg_file)
edg_root = edg_tree.getroot()

# Find edges that reference this junction
node_counter = 0
offset = 10  # meters apart
new_nodes = []

for edge in edg_root.findall('.//edge'):
    eid = edge.get('id', '')
    from_node = edge.get('from')
    to_node = edge.get('to')
    
    if to_node == JUNCTION_ID:
        new_node_id = f"{JUNCTION_ID}_end_{node_counter}"
        edge.set('to', new_node_id)
        new_nodes.append((new_node_id, jx + node_counter * offset, jy))
        print(f"Edge {eid}: to -> {new_node_id}")
        node_counter += 1
        
    if from_node == JUNCTION_ID:
        new_node_id = f"{JUNCTION_ID}_end_{node_counter}"
        edge.set('from', new_node_id)
        new_nodes.append((new_node_id, jx - node_counter * offset, jy))
        print(f"Edge {eid}: from -> {new_node_id}")
        node_counter += 1

edg_tree.write(edg_file, encoding='UTF-8', xml_declaration=True)
print(f"\nUpdated {edg_file}")

# === 3. Add new nodes ===
for node_id, x, y in new_nodes:
    new_node = ET.SubElement(root, 'node')
    new_node.set('id', node_id)
    new_node.set('x', str(x))
    new_node.set('y', str(y))
    new_node.set('type', 'dead_end')

tree.write(nod_file, encoding='UTF-8', xml_declaration=True)
print(f"Added {len(new_nodes)} new nodes to {nod_file}")

# === 4. Remove connections through this junction from plain.con.xml ===
con_file = "data/legacy/plain.con.xml"
con_tree = ET.parse(con_file)
con_root = con_tree.getroot()

# Get IDs of affected edges
affected_to = {edge.get('id') for edge in edg_root.findall('.//edge') 
               if JUNCTION_ID in edge.get('to', '')}
affected_from = {edge.get('id') for edge in edg_root.findall('.//edge') 
                 if JUNCTION_ID in edge.get('from', '')}

conns_to_remove = []
for conn in con_root.findall('.//connection'):
    from_edge = conn.get('from', '')
    to_edge = conn.get('to', '')
    
    # Remove if both ends involve affected edges (was through this junction)
    if from_edge in affected_to or to_edge in affected_from:
        conns_to_remove.append(conn)

for conn in conns_to_remove:
    con_root.remove(conn)

con_tree.write(con_file, encoding='UTF-8', xml_declaration=True)
print(f"Removed {len(conns_to_remove)} connections from {con_file}")

# === 5. Remove tlLogic ===
tll_file = "data/legacy/plain.tll.xml"
tll_tree = ET.parse(tll_file)
tll_root = tll_tree.getroot()

for tl in tll_root.findall(f".//tlLogic[@id='{JUNCTION_ID}']"):
    tll_root.remove(tl)
    print(f"Removed tlLogic for {JUNCTION_ID}")

tll_tree.write(tll_file, encoding='UTF-8', xml_declaration=True)

print("\n=== Done! ===")
print("Now run: netconvert --node-files data/legacy/plain.nod.xml --edge-files data/legacy/plain.edg.xml --connection-files data/legacy/plain.con.xml --tllogic-files data/legacy/plain.tll.xml -o data/legacy/ntut_network_split.net.xml --ignore-errors")
