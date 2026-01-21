
import xml.etree.ElementTree as ET

NET_FILE = "data/legacy/ntut-the way.net.xml"

def main():
    print(f"Extracting TLS IDs from {NET_FILE}...")
    
    tls_ids = set()
    for event, elem in ET.iterparse(NET_FILE, events=("start",)):
        if elem.tag == "tlLogic":
            tls_id = elem.get("id")
            if tls_id:
                tls_ids.add(tls_id)
    
    print(f"\nFound {len(tls_ids)} TLS controllers:")
    for tls_id in sorted(tls_ids):
        print(f"  {tls_id}")

if __name__ == "__main__":
    main()
