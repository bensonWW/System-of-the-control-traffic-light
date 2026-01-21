
import xml.etree.ElementTree as ET

NET_FILE = "data/real_world.net.xml"

def main():
    print(f"Scanning {NET_FILE} for largest traffic light junction...")
    
    tls_counts = {}
    
    context = ET.iterparse(NET_FILE, events=("start", "end"))
    for event, elem in context:
        if event == "start" and elem.tag == "connection":
            tl = elem.get("tl")
            if tl:
                tls_counts[tl] = tls_counts.get(tl, 0) + 1
        elem.clear()

    # Sort
    sorted_tls = sorted(tls_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Largest Traffic Lights:")
    for i, (tl, count) in enumerate(sorted_tls[:10]):
        print(f"{i+1}. {tl} (Connections: {count})")

if __name__ == "__main__":
    main()
