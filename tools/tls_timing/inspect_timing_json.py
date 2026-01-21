"""
Inspect the integrated_timing_plan.json to understand its structure.
"""
import json
import os

# Path to JSON file
JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "source", "integrated_timing_plan.json")

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Get unique ICIDs
    icids = set()
    for entry in data:
        icids.add(entry.get("icid", ""))
    print(f"Unique ICIDs: {len(icids)}")
    
    # Find IJLJW entries
    ijljw = [x for x in data if x.get("icid") == "IJLJW"]
    print(f"\nIJLJW entries: {len(ijljw)}")
    
    if ijljw:
        # Show structure of first entry
        entry = ijljw[0]
        print(f"\nFirst IJLJW entry keys: {list(entry.keys())}")
        print(f"  icid: {entry.get('icid')}")
        print(f"  icname_plan: {entry.get('icname_plan')}")
        print(f"  deviceid_plan: {entry.get('deviceid_plan')}")
        print(f"  planid: {entry.get('planid（SeqNo）')}")
        print(f"  cycletime: {entry.get('cycletime')}")
        print(f"  segmenttype: {entry.get('segmenttype')}")
        
        # Show subplan structure
        subplan = entry.get("subplan", [])
        print(f"\n  subplan (phases): {len(subplan)} phases")
        for sp in subplan:
            print(f"    Phase {sp.get('subphaseid')}: green={sp.get('green')}, yellow={sp.get('yellow')}, allred={sp.get('allred')}")
        
        # Show subsegment (time schedule)
        subseg = entry.get("subsegment", [])
        print(f"\n  subsegment (time schedule): {len(subseg)} entries")
        for ss in subseg[:5]:  # Show first 5
            print(f"    time={ss.get('time')} -> planid={ss.get('planid（SeqNo）')}")
        if len(subseg) > 5:
            print(f"    ... and {len(subseg) - 5} more")
    
    # Check different planids for IJLJW
    print("\n--- Different planids for IJLJW ---")
    planids = set()
    segmenttypes = set()
    for entry in ijljw:
        planids.add(entry.get("planid（SeqNo）"))
        segmenttypes.add(entry.get("segmenttype"))
    print(f"Unique planids: {sorted([x for x in planids if x is not None])}")
    print(f"Unique segmenttypes: {sorted([x for x in segmenttypes if x is not None])}")

if __name__ == "__main__":
    main()
