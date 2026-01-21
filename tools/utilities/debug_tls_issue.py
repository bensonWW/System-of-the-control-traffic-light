"""
Debug script to check IJLJW Plan 5 and mapping validity.
"""
import xml.etree.ElementTree as ET
import csv
import json
import re

def check_mappings():
    print('Checking mapping file against network...')
    # 1. Get real network TLS IDs
    tree = ET.parse('data/legacy/ntut-the way.net.xml')
    real_tls = set(tl.get('id') for tl in tree.findall('.//tlLogic'))
    print(f"Loaded {len(real_tls)} TLS IDs from network.")

    # 2. Check mapping file
    with open('data/sumo_json_mapping.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sumo_id = row.get('sumo_id', '').strip()
            # Check exact and partial matches
            found = False
            if sumo_id in real_tls:
                found = True
            else:
                for real in real_tls:
                    if sumo_id in real:  # Partial match check (simple substring)
                        found = True
                        break
                    # Check if all numeric parts match
                    s_nums = set(re.findall(r'\d+', sumo_id))
                    r_nums = set(re.findall(r'\d+', real))
                    if s_nums and s_nums.issubset(r_nums):
                         found = True
                         break
            
            if not found:
                print(f'❌ MAPPING ERROR: {sumo_id} not found in network (nor as substring/subset)')
            else:
                # print(f'✓ Found {sumo_id}')
                pass

def check_ijljw_plan():
    print('\nChecking IJLJW Plan 5 details...')
    with open('data/source/integrated_timing_plan.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    found_any = False
    for entry in data:
        if entry.get('icid') == 'IJLJW' and entry.get('planid（SeqNo）') == 5.0:
            found_any = True
            print(f'Found IJLJW Plan 5 (Segment Type {entry.get("segmenttype")}):')
            print(f'  Cycle Time: {entry.get("cycletime")}')
            print('  Subplans:')
            for sub in entry.get('subplan', []):
                print(f'    Phase {sub.get("subphaseid")}: G={sub.get("green")} Y={sub.get("yellow")} R={sub.get("allred")}')
    
    if not found_any:
        print("❌ IJLJW Plan 5 not found in JSON data.")

if __name__ == '__main__':
    check_mappings()
    check_ijljw_plan()
