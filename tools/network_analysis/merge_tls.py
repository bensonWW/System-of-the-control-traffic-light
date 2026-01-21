"""
Clean up and merge all 622618xxx junctions into one TLS.
"""
import xml.etree.ElementTree as ET

NET_FILE = "data/legacy/ntut_network_split.net.xml"
OUTPUT_FILE = "data/legacy/ntut_network_split.net.xml"

# All junction IDs that should be in one TLS
JUNCTIONS = ["622618015", "622618108", "622618112", "622618144", "622618197", "622618230"]
NEW_TLS_ID = "joinedS_622618015_622618108_622618112_622618144_622618197_622618230"

tree = ET.parse(NET_FILE)
root = tree.getroot()

# Step 1: Remove ALL tlLogic containing 622618
tls_to_remove = []
for tl in root.findall('.//tlLogic'):
    if '622618' in tl.get('id', ''):
        tls_to_remove.append(tl)

for tl in tls_to_remove:
    root.remove(tl)
    print(f"Removed tlLogic: {tl.get('id')}")

# Step 2: Update all connections from these junctions
conns_updated = 0
for conn in root.findall('.//connection'):
    tl = conn.get('tl')
    if tl and '622618' in tl:
        conn.set('tl', NEW_TLS_ID)
        conns_updated += 1

print(f"Updated {conns_updated} connections")

# Step 3: Also find connections from edges going TO these junctions
for jid in JUNCTIONS:
    incoming_edges = set()
    for edge in root.findall('.//edge'):
        eid = edge.get('id', '')
        if eid.startswith(':'): continue
        if edge.get('to') == jid:
            incoming_edges.add(eid)
    
    for conn in root.findall('.//connection'):
        if conn.get('from') in incoming_edges:
            if conn.get('tl') != NEW_TLS_ID:
                conn.set('tl', NEW_TLS_ID)
                conns_updated += 1

print(f"Total connections: {conns_updated}")

# Step 4: Create new tlLogic
num_conns = len([c for c in root.findall('.//connection') if c.get('tl') == NEW_TLS_ID])
print(f"Creating tlLogic with {num_conns} connections")

# Create simple 2-phase static TLS
half = num_conns // 2
state1 = 'G' * half + 'r' * (num_conns - half)
state2 = 'r' * half + 'G' * (num_conns - half)
yellow1 = 'y' * half + 'r' * (num_conns - half)
yellow2 = 'r' * half + 'y' * (num_conns - half)

tl_logic = ET.Element('tlLogic')
tl_logic.set('id', NEW_TLS_ID)
tl_logic.set('type', 'static')
tl_logic.set('programID', '0')
tl_logic.set('offset', '0')

for phase_state, duration in [(state1, '30'), (yellow1, '3'), 
                              (state2, '30'), (yellow2, '3')]:
    phase = ET.SubElement(tl_logic, 'phase')
    phase.set('duration', duration)
    phase.set('state', phase_state)

root.insert(1, tl_logic)
print(f"Created tlLogic: {NEW_TLS_ID}")

# Step 5: Reassign linkIndex
link_index = 0
for conn in root.findall('.//connection'):
    if conn.get('tl') == NEW_TLS_ID:
        conn.set('linkIndex', str(link_index))
        link_index += 1

print(f"Reassigned {link_index} linkIndex values")

# Save
tree.write(OUTPUT_FILE, encoding='UTF-8', xml_declaration=True)
print(f"\nSaved to: {OUTPUT_FILE}")
