"""
Extract TLS rules for a specific cluster from the legacy network.
"""
import xml.etree.ElementTree as ET

NET_FILE = "data/legacy/ntut-the way.net.xml"
TARGET_TLS = "cluster_5372642442_631668976_631668981_631669076"

def main():
    print(f"Searching for TLS: {TARGET_TLS}")
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    
    found = False
    for tl in root.findall(".//tlLogic"):
        tls_id = tl.get("id", "")
        if TARGET_TLS in tls_id or tls_id == TARGET_TLS:
            found = True
            print(f"\n{'='*60}")
            print(f"TLS ID: {tls_id}")
            print(f"Type: {tl.get('type')}")
            print(f"Program ID: {tl.get('programID')}")
            print(f"Offset: {tl.get('offset', '0')}")
            print(f"{'='*60}")
            print("\nPhases:")
            phases = tl.findall("phase")
            for i, phase in enumerate(phases):
                duration = phase.get("duration")
                state = phase.get("state")
                print(f"  Phase {i:2d}: duration={duration:>3}s  state=\"{state}\"")
            print(f"\nTotal phases: {len(phases)}")
            print(f"State length: {len(phases[0].get('state', '')) if phases else 0} (number of controlled links)")
    
    if not found:
        print(f"\nNo TLS found matching '{TARGET_TLS}'")
        print("\nAvailable TLS IDs containing 'cluster':")
        for tl in root.findall(".//tlLogic"):
            tls_id = tl.get("id", "")
            if "cluster" in tls_id.lower():
                print(f"  - {tls_id}")

if __name__ == "__main__":
    main()
