#!/usr/bin/env python3
import json, os, sys

PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, 'tools'))
import searchnetdata as SD

with open('data/trafficData/traffic_data_20260126_112230.json', 'r', encoding='utf-8') as f:
    record = json.load(f)
road_data = record.get('data', {})

# Check if preprocessed
first_item = next(iter(road_data.values()), {})
is_preprocessed = 'from' in first_item and 'to' in first_item

print(f'Is preprocessed: {is_preprocessed}')

if is_preprocessed:
    selected = road_data
    print(f'Selected roads: {len(selected)}')
    
    # Check first road
    name = list(selected.keys())[0]
    print(f'\nFirst road: {name}')
    print(f'  from (before): {selected[name]["from"]}')
    print(f'  to (before): {selected[name]["to"]}')
    
    # Try remap
    new_from = SD.remap_junction(selected[name]['from'])
    new_to = SD.remap_junction(selected[name]['to'])
    print(f'  from (after): {new_from}')
    print(f'  to (after): {new_to}')
    
    # Show all keys
    print(f'\nAll keys in first road:')
    for k in selected[name].keys():
        print(f'  - {k}')
