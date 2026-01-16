
import xml.etree.ElementTree as ET
import pandas as pd
import json

NET_XML = "ntut-the way.net.xml"
TIMING_JSON = "integrated_timing_plan.json"
TARGET_SUMO_ID = "655375336"
TARGET_ICID = "IKJJP" # Matched ID from CSV

def analyze():
    print(f"Analyzing structure for SUMO TLS {TARGET_SUMO_ID} and JSON ICID {TARGET_ICID}")
    
    # 1. Parse SUMO XML
    tree = ET.parse(NET_XML)
    root = tree.getroot()
    
    sumo_phases = []
    found_tls = False
    for tl in root.findall("tlLogic"):
        if tl.get("id") == TARGET_SUMO_ID:
            found_tls = True
            print(f"SUMO TLS Found: Type={tl.get('type')}, ProgramID={tl.get('programID')}")
            for phase in tl.findall("phase"):
                sumo_phases.append({
                    "duration": phase.get("duration"),
                    "state": phase.get("state")
                })
            break
            
    print(f"\nSearching for Junction {TARGET_SUMO_ID}...")
    target_junction = None
    for j in root.findall("junction"):
        if j.get("id") == TARGET_SUMO_ID:
            target_junction = j
            break
            
    if target_junction:
         print(f"Junction Found. Attributes: {target_junction.attrib}")
         # Attempt to find connections passing through this junction to find 'tl' attribute
         # Connections are linking edges.
    else:
        print("Junction not found.")
        
    # Check if there is a tlLogic that seems related (e.g. joinedS_...)
    # We will iterate connections to find the tlLogic ID for this junction's edges.
    print("\nScanning connections for tlLogic ID...")
    controlling_tls = set()
    # We need to know which edges enter/leave this junction. 
    # But simpler: scan all connections, if 'tl' attribute is present, gather it.
    # Actually, we want to know the textual ID mapping.
    # Let's search for any 'tl' attribute in connections where 'to' or 'from' might relate to our junction (harder).
    # But usually <junction> element has 'tl' attribute if it differs from ID? No, not always.
    
    # Let's just list ALL tlLogic IDs in the file to see if we missed it or if it's joined.
    all_tls_ids = [tl.get("id") for tl in root.findall("tlLogic")]
    with open("tls_ids_log.txt", "w") as f:
        f.write(str(all_tls_ids))
    
    # Check if 3086736518 is in any set


    # 2. Parse JSON
    with open(TIMING_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find records for this ICID
    records = [r for r in data if r.get('icid') == TARGET_ICID]
    print(f"\nJSON Records for {TARGET_ICID}: {len(records)}")
    
    if records:
        # Take the first plan as example
        plan = records[0]
        subplan = plan.get('subplan', [])
        print(f"Plan ID: {plan.get('planid（SeqNo）')}")
        print(f"Subplans (Real-world Phases): {len(subplan)}")
        for sp in subplan:
            print(f"  SubPhase {sp.get('subphaseid')}: Green={sp.get('green')}, Yellow={sp.get('yellow')}, AllRed={sp.get('allred')}, Total={sp.get('green',0)+sp.get('yellow',0)+sp.get('allred',0)}")

if __name__ == "__main__":
    analyze()
