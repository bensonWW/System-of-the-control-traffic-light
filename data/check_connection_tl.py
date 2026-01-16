
import xml.etree.ElementTree as ET

NET_XML = "ntut-the way.net.xml"
TARGET_JID = "655375336"

def check():
    print(f"Checking connections for Junction {TARGET_JID}...")
    tree = ET.parse(NET_XML)
    root = tree.getroot()
    
    # 1. Find Junction and incoming lanes
    junction = None
    for j in root.findall("junction"):
        if j.get("id") == TARGET_JID:
            junction = j
            break
            
    if not junction:
        print("Junction not found.")
        return

    inc_lanes = junction.get("incLanes", "").split(" ")
    inc_edges = set()
    for lane in inc_lanes:
        if "_" in lane:
            edge = lane.rsplit("_", 1)[0]
            inc_edges.add(edge)
            
    print(f"Incoming Edges: {inc_edges}")
    
    # 2. Find connections from these edges
    print("\nConnections from incoming edges:")
    found_tl_attr = False
    for conn in root.findall("connection"):
        if conn.get("from") in inc_edges:
            print(f"  From: {conn.get('from')}, To: {conn.get('to')}, TL: {conn.get('tl')}, LinkIndex: {conn.get('linkIndex')}")
            if conn.get("tl"):
                found_tl_attr = True
                
    if not found_tl_attr:
        print("\nWARNING: No 'tl' attribute found on connections. Implicit control?")

if __name__ == "__main__":
    check()
