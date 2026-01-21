
import xml.etree.ElementTree as ET

NET_FILE = "data/real_world.net.xml"
TARGET = "1446842483"

def main():
    print(f"Checking node {TARGET} in {NET_FILE}...")
    found = False
    
    for event, elem in ET.iterparse(NET_FILE, events=("start",)):
        if elem.tag == "junction" and elem.get("id") == TARGET:
            print(f"Junction Found:")
            print(f"  Type: {elem.get('type')}")
            print(f"  TL: {elem.get('tl')}") # If controlled by another TLS
            print(f"  IncLanes: {elem.get('incLanes')}")
            found = True
            break
            
    if not found:
        print("Junction NOT found in XML stream.")
    
    # Check connections
    print("Checking connections...")
    count = 0
    for event, elem in ET.iterparse(NET_FILE, events=("start",)):
        if elem.tag == "connection":
            if elem.get("tl") == TARGET:
                count += 1
    print(f"Connections with tl='{TARGET}': {count}")

if __name__ == "__main__":
    main()
