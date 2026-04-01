#!/usr/bin/env python3
import os
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "tools"))

import convertToRou as CTR
import selectRoad as ST

# Get first JSON file
files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "trafficData", "*.json")))
if not files:
    print("No JSON files found")
    sys.exit(1)

json_path = files[0]
print(f"Testing with: {json_path}")

# Load and process
with open(json_path, "r", encoding="utf-8") as f:
    record = json.load(f)
road_data = record.get("data", {})
print(f"Road data items: {len(road_data)}")

# Monkey patch
original = ST.gb.getData
ST.gb.getData = lambda: road_data

try:
    selected = ST.select()
    print(f"Selected roads: {len(selected)}")
    
    # Show first item
    if selected:
        name = list(selected.keys())[0]
        info = selected[name]
        print(f"\nFirst road: {name}")
        print(f"  from: {info.get('from')}")
        print(f"  to: {info.get('to')}")
        print(f"  TotalVol: {info.get('TotalVol')}")
        print(f"  Keys: {list(info.keys())}")
finally:
    ST.gb.getData = original
