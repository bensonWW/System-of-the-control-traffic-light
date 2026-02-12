#!/usr/bin/env python3
"""
Standalone script for collecting traffic data from Taipei City Open Data API,
filtering roads within the SUMO network boundary, and saving results to data/trafficData.
"""
import requests
import xml.etree.ElementTree as ET
import gzip
import shutil
import os
import sys
import json
import datetime
import math

# Ensure we run from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


# Data URL for Taipei City traffic data
DATA_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"

# Network file path
NETWORK_FILE = "./data/ntut_network_split.net copy.xml"


# ========== Network helpers (inlined from searchnetdata.py) ==========

def get_network_root():
    """Parse and return the root of the SUMO network XML."""
    tree = ET.parse(NETWORK_FILE)
    return tree.getroot()


def get_map_scope():
    """Get the WGS84 boundary (lon_min, lat_min, lon_max, lat_max) from the network file."""
    root = get_network_root()
    boundary_str = root[0].attrib["origBoundary"]
    parts = boundary_str.split(",")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def get_conv_boundary():
    """Get the converted boundary (SUMO local coordinate system) from the network file."""
    root = get_network_root()
    parts = root[0].attrib["convBoundary"].split(",")
    return float(parts[2]), float(parts[3])  # cDeltax, cDeltay


def search_nearest_junction(position_x, position_y):
    """Find the nearest traffic_light junction node to the given position."""
    root = get_network_root()
    road_info = {}
    position_x, position_y = float(position_x), float(position_y)

    for child in root:
        if (child.tag == "junction"
                and not child.attrib["id"].startswith(":")
                and child.attrib["type"] == "traffic_light"):
            x = float(child.attrib["x"])
            y = float(child.attrib["y"])
            dist = math.dist((x, y), (position_x, position_y))
            road_info[child.attrib["id"]] = [x, y, dist]

    if not road_info:
        return None
    return min(road_info, key=lambda k: road_info[k][2])


# ========== Road filtering (inlined from selectRoad.py) ==========

def in_range(lon_min, lat_min, lon_max, lat_max, sx, sy, ex, ey):
    """Check if start or end point falls within the map boundary."""
    start_in = (lon_min <= sx <= lon_max and lat_min <= sy <= lat_max)
    end_in = (lon_min <= ex <= lon_max and lat_min <= ey <= lat_max)
    return start_in or end_in


def filter_and_convert(road_info):
    """
    Filter roads within the SUMO network boundary,
    convert WGS84 coords to SUMO local coords,
    and find the nearest traffic_light junction for each endpoint.
    """
    lon_min, lat_min, lon_max, lat_max = get_map_scope()
    c_delta_x, c_delta_y = get_conv_boundary()

    # Step 1: Filter roads within boundary
    filtered = {}
    for name, data in road_info.items():
        sx = float(data["StartWgsX"])
        sy = float(data["StartWgsY"])
        ex = float(data["EndWgsX"])
        ey = float(data["EndWgsY"])
        if in_range(lon_min, lat_min, lon_max, lat_max, sx, sy, ex, ey):
            filtered[name] = data

    # Step 2: Convert coordinates to SUMO local
    for name in filtered:
        lon = float(filtered[name]["StartWgsX"])
        lat = float(filtered[name]["StartWgsY"])
        filtered[name]["StartWgsX"] = ((lon - lon_min) / (lon_max - lon_min)) * c_delta_x
        filtered[name]["StartWgsY"] = ((lat - lat_min) / (lat_max - lat_min)) * c_delta_y
        lon = float(filtered[name]["EndWgsX"])
        lat = float(filtered[name]["EndWgsY"])
        filtered[name]["EndWgsX"] = ((lon - lon_min) / (lon_max - lon_min)) * c_delta_x
        filtered[name]["EndWgsY"] = ((lat - lat_min) / (lat_max - lat_min)) * c_delta_y

    # Step 3: Find nearest junction for each endpoint
    for name in filtered:
        filtered[name]["from"] = search_nearest_junction(
            filtered[name]["StartWgsX"], filtered[name]["StartWgsY"])
        filtered[name]["to"] = search_nearest_junction(
            filtered[name]["EndWgsX"], filtered[name]["EndWgsY"])

    return filtered


# ========== Download & Parse ==========

def download_and_parse_data():
    """Download traffic data from Taipei City API and parse it."""
    print(f"[{datetime.datetime.now()}] Downloading traffic data...")

    try:
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error downloading data: {e}")
        return None

    # Save and extract gz file
    gz_file = "GetVD.xml.gz"
    xml_file = "GetVD.xml"

    with open(gz_file, "wb") as f:
        f.write(response.content)

    with gzip.open(gz_file, "rb") as f_in:
        with open(xml_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Clean up gz file
    if os.path.exists(gz_file):
        os.remove(gz_file)

    # Parse XML
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        if os.path.exists(xml_file):
            os.remove(xml_file)
        return None

    road_info = {}
    try:
        if len(root) > 2:
            target_element = root[2]
        else:
            target_element = root

        for child1 in target_element:
            temp_dict = {}
            name = ""
            for child2 in child1:
                tag = child2.tag.split("}")[1] if "}" in child2.tag else child2.tag
                if tag == "SectionName":
                    if child2.text and ("高" in child2.text or "快" in child2.text):
                        name = " "
                    else:
                        name = child2.text if child2.text else ""
                else:
                    temp_dict[tag] = child2.text if child2.text else ""
            if name and name != " ":
                road_info[name] = temp_dict
    except Exception as e:
        print(f"Error processing XML structure: {e}")
        return None

    # Clean up xml file
    if os.path.exists(xml_file):
        os.remove(xml_file)

    print(f"[{datetime.datetime.now()}] Downloaded {len(road_info)} road sections")
    return road_info


# ========== Save ==========

def save_traffic_data(data, output_dir="data/trafficData"):
    """Saves the filtered traffic data to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_time = datetime.datetime.now()
    filename = f"traffic_data_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    record = {
        "timestamp": current_time.isoformat(),
        "data": data
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=4)
        print(f"[{current_time.strftime('%H:%M:%S')}] Saved traffic data to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving data: {e}")
        return None


# ========== Main ==========

def main():
    """Main function to collect, filter, and save traffic data."""
    print("=" * 50)
    print("Traffic Data Collection Script")
    print(f"Started at: {datetime.datetime.now()}")
    print("=" * 50)

    # Check network file exists
    if not os.path.exists(NETWORK_FILE):
        print(f"Error: Network file not found: {NETWORK_FILE}")
        return 1

    # 1. Download and parse data from API
    road_info = download_and_parse_data()
    if road_info is None:
        print("Failed to download data")
        return 1

    # 2. Filter roads in map boundary + convert coords + find junction nodes
    print(f"[{datetime.datetime.now()}] Filtering roads within map boundary...")
    filtered_data = filter_and_convert(road_info)
    print(f"[{datetime.datetime.now()}] {len(filtered_data)} roads matched (out of {len(road_info)})")

    # 3. Save to data/trafficData
    filepath = save_traffic_data(filtered_data)

    if filepath:
        print("=" * 50)
        print(f"Done! Saved to: {filepath}")
        print("=" * 50)
        return 0
    else:
        print("Failed to save traffic data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())