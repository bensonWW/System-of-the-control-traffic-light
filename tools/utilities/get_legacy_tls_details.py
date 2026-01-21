
import xml.etree.ElementTree as ET

NET_FILE = "data/legacy/ntut-the way.net.xml"

def main():
    print(f"Extracting full TLS details from {NET_FILE}...")
    
    tls_details = []
    for event, elem in ET.iterparse(NET_FILE, events=("start",)):
        if elem.tag == "tlLogic":
            tls_id = elem.get("id")
            if tls_id:
                # Get phase count and state length
                phases = elem.findall("phase")
                state_len = 0
                if phases:
                    state = phases[0].get("state", "")
                    state_len = len(state)
                tls_details.append({
                    "id": tls_id,
                    "phases": len(phases),
                    "state_len": state_len
                })
    
    print(f"\nFound {len(tls_details)} TLS controllers:")
    print(f"{'TLS ID':<70} {'Phases':<8} {'Links'}")
    print("-" * 90)
    for tl in tls_details:
        print(f"{tl['id']:<70} {tl['phases']:<8} {tl['state_len']}")

if __name__ == "__main__":
    main()
