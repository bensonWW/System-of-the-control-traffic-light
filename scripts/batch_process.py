#!/usr/bin/env python3
"""
Batch script: process all JSON files in data/trafficData through the SUMO pipeline
and save generated .rou.xml files to data/VehicleData.
"""
import os
import sys
import json
import shutil
import datetime
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "tools"))

import convertToRou as CTR
import selectRoad as ST
import fixRoadData as FRD

def process_one(json_path, output_dir):
    """Process a single trafficData JSON file into a .rou.xml file."""
    basename = os.path.basename(json_path)
    # e.g. traffic_data_20260211_021820.json -> traffic_data_20260211_021820.rou.xml
    out_name = basename.replace(".json", ".rou.xml")
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        return "skip"

    # Load raw data
    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    road_data = record.get("data", {})
    if not road_data:
        return "empty"

    # Monkey-patch grabapi.getData to return our data
    original = ST.gb.getData
    ST.gb.getData = lambda: road_data

    try:
        selected = ST.select()
        CTR.generate_trip(selected)

        net = "./data/ntut_network_split.net copy.xml"
        CTR.run_duarouter(net, "./data/trips.xml", "./data/output.rou.alt.xml", "./data/output.rou.xml")

        edges_volume = FRD.fixtheRoadData()
        CTR.generate_trip(edges_volume)

        final_alt = "./data/final_output.rou.alt.xml"
        final_rou = "./data/final_output.rou.xml"
        CTR.run_duarouter(net, "./data/trips.xml", final_alt, final_rou)

        if os.path.exists(final_rou):
            shutil.copy2(final_rou, out_path)
            return "ok"
        else:
            return "no_output"
    except Exception as e:
        print(f"  Error: {e}")
        return "error"
    finally:
        ST.gb.getData = original


def main():
    input_dir = os.path.join(PROJECT_ROOT, "data", "trafficData")
    output_dir = os.path.join(PROJECT_ROOT, "data", "VehicleData")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    total = len(files)
    print(f"Found {total} JSON files in {input_dir}")
    print(f"Output to {output_dir}")
    print("=" * 50)

    ok = skip = err = 0
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        print(f"[{i}/{total}] {name} ... ", end="", flush=True)
        result = process_one(f, output_dir)
        print(result)
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
        else:
            err += 1

    print("=" * 50)
    print(f"Done! OK={ok}, Skipped={skip}, Errors={err}")


if __name__ == "__main__":
    main()
