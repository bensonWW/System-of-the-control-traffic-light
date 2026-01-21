
import xml.etree.ElementTree as ET

OSM_FILE = "data/source/map_no_expressway.osm"

def main():
    print(f"Scanning {OSM_FILE} for lane-related tags...")
    
    lane_tags = {}
    
    for event, elem in ET.iterparse(OSM_FILE, events=("start", "end")):
        if event == "start" and elem.tag == "tag":
            k = elem.get("k", "")
            v = elem.get("v", "")
            if "lane" in k.lower() or "bus" in k.lower() or "psv" in k.lower():
                key = f"{k}={v}"
                lane_tags[key] = lane_tags.get(key, 0) + 1
        elem.clear()
    
    print("\nLane-related tags found:")
    for tag, count in sorted(lane_tags.items(), key=lambda x: x[1], reverse=True)[:30]:
        print(f"  {tag}: {count}")

if __name__ == "__main__":
    main()
