"""Check and add TLS to specified junctions."""
import xml.etree.ElementTree as ET

NET_FILE = "data/legacy/ntut_network_split.net.xml"
JUNCTIONS = [
    "cluster619136899_619136900_cluster_619136929_619136942",
    "cluster655375228_655375240"
]

tree = ET.parse(NET_FILE)
root = tree.getroot()

for jid in JUNCTIONS:
    j = None
    for junction in root.findall('.//junction'):
        if junction.get('id') == jid:
            j = junction
            break
    
    if j is None:
        print(f"NOT FOUND: {jid}")
    else:
        print(f"FOUND: {jid}")
        print(f"  Type: {j.get('type')}")
        
        # Find incoming edges
        incoming = []
        for edge in root.findall('.//edge'):
            if edge.get('id', '').startswith(':'): continue
            if edge.get('to') == jid:
                incoming.append(edge.get('id'))
        print(f"  Incoming edges: {len(incoming)} - {incoming[:3]}...")
        
        # Count connections with TLS
        conns = [c for c in root.findall('.//connection') if c.get('tl') == jid]
        print(f"  TLS connections: {len(conns)}")
        
        # Check tlLogic
        tl = root.find(f".//tlLogic[@id='{jid}']")
        print(f"  tlLogic exists: {tl is not None}")
