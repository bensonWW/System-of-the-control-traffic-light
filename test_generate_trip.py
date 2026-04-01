#!/usr/bin/env python3
import json, os, sys

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, 'tools'))
import convertToRou as CTR
import searchnetdata as SD

with open('data/trafficData/traffic_data_20260126_112230.json', 'r', encoding='utf-8') as f:
    record = json.load(f)
road_data = record.get('data', {})
selected = road_data

# Remap junctions
remapped = 0
for rname in selected:
    old_from = selected[rname]["from"]
    old_to = selected[rname]["to"]
    selected[rname]["from"] = SD.remap_junction(old_from)
    selected[rname]["to"] = SD.remap_junction(old_to)
    if selected[rname]["from"] != old_from or selected[rname]["to"] != old_to:
        remapped += 1

print(f'Preprocessed: {len(selected)} roads, {remapped} junctions remapped')

# Test generate_trip
CTR.generate_trip(selected)

# Check if trips.xml was created
if os.path.exists('./data/trips.xml'):
    import xml.etree.ElementTree as ET
    tree = ET.parse('./data/trips.xml')
    root = tree.getroot()
    print(f'\nGenerated trips.xml:')
    print(f'  Root tag: {root.tag}')
    print(f'  Number of children: {len(root)}')
    
    # Count flows
    flows = [e for e in root if e.tag == 'flow']
    print(f'  Flows: {len(flows)}')
    
    if flows:
        print(f'\n  First flow:')
        first = flows[0]
        for k, v in first.attrib.items():
            print(f'    {k}: {v}')
else:
    print('ERROR: trips.xml not created')
