#!/usr/bin/env python3
"""
Standalone script for collecting traffic data from Taipei City Open Data API.
Designed to run in GitHub Actions environment.
"""
import requests
import xml.etree.ElementTree as ET
import gzip
import shutil
import os
import json
import datetime
import math

# Data URL for Taipei City traffic data
DATA_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"

# Map boundaries from ntut-the way.net.xml
# These are the origBoundary values
MAP_BOUNDARY = {
    "lon_min": 121.52312,
    "lat_min": 25.03645,
    "lon_max": 121.54575,
    "lat_max": 25.05055
}

# Conversion boundary from convBoundary
CONV_BOUNDARY = {
    "x_min": 0.0,
    "y_min": 0.0,
    "x_max": 1942.75,
    "y_max": 1643.91
}


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
    for child1 in root[2]:
        temp_dict = {}
        name = ""
        for child2 in child1:
            tag = child2.tag.split("}")[1] if "}" in child2.tag else child2.tag
            if tag == "SectionName":
                # Skip highway data
                if "高" in child2.text or "快" in child2.text:
                    name = " "
                else:
                    name = child2.text
            else:
                temp_dict[tag] = child2.text
        if name != " ":
            road_info[name] = temp_dict
    
    # Clean up xml file
    if os.path.exists(xml_file):
        os.remove(xml_file)
    
    print(f"[{datetime.datetime.now()}] Downloaded {len(road_info)} road sections")
    return road_info


def is_in_range(start_x, start_y, end_x, end_y):
    """Check if road segment is within the map boundary."""
    lon_min = MAP_BOUNDARY["lon_min"]
    lat_min = MAP_BOUNDARY["lat_min"]
    lon_max = MAP_BOUNDARY["lon_max"]
    lat_max = MAP_BOUNDARY["lat_max"]
    
    start_in = (lon_min <= start_x <= lon_max and lat_min <= start_y <= lat_max)
    end_in = (lon_min <= end_x <= lon_max and lat_min <= end_y <= lat_max)
    
    return start_in or end_in


def convert_coordinates(road_info):
    """Convert WGS84 coordinates to local simulation coordinates."""
    lon_min = MAP_BOUNDARY["lon_min"]
    lat_min = MAP_BOUNDARY["lat_min"]
    lon_max = MAP_BOUNDARY["lon_max"]
    lat_max = MAP_BOUNDARY["lat_max"]
    
    c_delta_x = CONV_BOUNDARY["x_max"]
    c_delta_y = CONV_BOUNDARY["y_max"]
    
    for name in road_info:
        lon = float(road_info[name]["StartWgsX"])
        lat = float(road_info[name]["StartWgsY"])
        road_info[name]["StartWgsX"] = ((lon - lon_min) / (lon_max - lon_min)) * c_delta_x
        road_info[name]["StartWgsY"] = ((lat - lat_min) / (lat_max - lat_min)) * c_delta_y
        
        lon = float(road_info[name]["EndWgsX"])
        lat = float(road_info[name]["EndWgsY"])
        road_info[name]["EndWgsX"] = ((lon - lon_min) / (lon_max - lon_min)) * c_delta_x
        road_info[name]["EndWgsY"] = ((lat - lat_min) / (lat_max - lat_min)) * c_delta_y
    
    return road_info


def filter_and_process_roads(road_info):
    """Filter roads within map boundary and convert coordinates."""
    filtered = {}
    
    for road_name, data in road_info.items():
        try:
            start_x = float(data.get("StartWgsX", 0))
            start_y = float(data.get("StartWgsY", 0))
            end_x = float(data.get("EndWgsX", 0))
            end_y = float(data.get("EndWgsY", 0))
            
            if is_in_range(start_x, start_y, end_x, end_y):
                filtered[road_name] = data
        except (ValueError, TypeError):
            continue
    
    # Convert coordinates
    filtered = convert_coordinates(filtered)
    
    print(f"[{datetime.datetime.now()}] Filtered to {len(filtered)} roads in target area")
    return filtered


def save_traffic_data(data, output_dir="data/trafficData"):
    """Save traffic data to JSON file with timestamp."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.datetime.now()
    filename = f"traffic_data_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    
    record = {
        "timestamp": current_time.isoformat(),
        "data": data
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=4)
    
    print(f"[{current_time.strftime('%H:%M:%S')}] Saved data to {filename}")
    return filepath


def main():
    """Main function to collect and save traffic data."""
    print("=" * 50)
    print("Traffic Data Collection Script")
    print(f"Started at: {datetime.datetime.now()}")
    print("=" * 50)
    
    # Download and parse data
    road_info = download_and_parse_data()
    if road_info is None:
        print("Failed to download data")
        return 1
    
    # Filter and process roads
    processed_data = filter_and_process_roads(road_info)
    
    if not processed_data:
        print("No roads found in target area")
        return 1
    
    # Save data
    filepath = save_traffic_data(processed_data)
    
    print("=" * 50)
    print(f"Data collection completed successfully!")
    print(f"Output file: {filepath}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
