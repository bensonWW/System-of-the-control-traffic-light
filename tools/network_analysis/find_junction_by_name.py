
import xml.etree.ElementTree as ET

NET_FILE = "data/real_world.net.xml"

# Street names to find
STREETS = ["忠孝東路三段", "建國南路一段"]

def main():
    print(f"Loading {NET_FILE}...")
    # Use iterparse for memory efficiency
    found_edges = set()
    
    # 1. Find Edges with matching names
    for event, elem in ET.iterparse(NET_FILE, events=("start", "end")):
        if event == "start" and elem.tag == "edge":
            name = elem.get("name", "")
            if any(s in name for s in STREETS):
                eid = elem.get("id")
                # ignore internal edges
                if not eid.startswith(":"):
                    found_edges.add(eid)
        elem.clear()

    print(f"Found {len(found_edges)} edges matching names.")
    
    # 2. Find Junctions that these edges feed into
    target_junctions = {}
    
    for event, elem in ET.iterparse(NET_FILE, events=("start", "end")):
        if event == "start" and elem.tag == "connection":
            from_edge = elem.get("from")
            if from_edge in found_edges:
                tl = elem.get("tl")
                if tl:
                    target_junctions[tl] = target_junctions.get(tl, 0) + 1
        elem.clear()
        
    print("\nTop Traffic Lights by Connection Count:")
    for tl, count in sorted(target_junctions.items(), key=lambda x: x[1], reverse=True):
        print(f"TL: {tl} (Count: {count})")

if __name__ == "__main__":
    main()
