#!/usr/bin/env python3
"""
Standalone script for collecting traffic data from Taipei City Open Data API,
processing it into SUMO routes, and saving results to data/VehicleData.
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
import time

# Ensure we run from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Add tools directory to sys.path
sys.path.append(os.path.join(PROJECT_ROOT, "tools"))

# Now import tool modules
try:
    import convertToRou as CTR
    import selectRoad as ST
    import fixRoadData as FRD
    import grabapi as GB
except ImportError as e:
    print(f"Error importing tools: {e}")
    # If tools dir structure is System-of-the-control-traffic-light-main/tools
    # and this script is System-.../scripts/collect.py
    # sys.path should be correct.
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Tools Path: {os.path.join(PROJECT_ROOT, 'tools')}")
    sys.exit(1)



# Data URL for Taipei City traffic data
DATA_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"


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
        # Based on observed structure (root[2] contains VDs)
        # But we verify if root has children
        if len(root) > 2:
            target_element = root[2]
        else:
            target_element = root

        for child1 in target_element:
            temp_dict = {}
            name = ""
            for child2 in child1:
                # Handle namespaces if present
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



def ensure_network_file():
    """
    Ensures that 'data/ntut_network_split.net copy.xml' exists for tools.
    If missing, tries to copy from 'data/ntut_network_split.net.xml'.
    """
    legacy_target = "./data/ntut_network_split.net copy.xml"
    source_file = "./data/ntut_network_split.net.xml"
    
    # Create data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    
    # Check if improved network file exists
    if not os.path.exists(source_file):
        print(f"[{datetime.datetime.now()}] Warning: '{source_file}' not found.")
        # Fallback to check legacy file if source is missing (maybe user only has legacy)
        if os.path.exists(legacy_target):
            print(f"[{datetime.datetime.now()}] Found '{legacy_target}'. Using it as source.")
            source_file = legacy_target
        else:
            print(f"[{datetime.datetime.now()}] Error: No network file found (checked '{source_file}' and '{legacy_target}').")
            return source_file # Return whatever we have, even if missing
            
    # Copy to legacy target if needed
    if not os.path.exists(legacy_target) and os.path.exists(source_file):
        print(f"[{datetime.datetime.now()}] Found '{source_file}'. Copying to '{legacy_target}' for legacy tools...")
        try:
            shutil.copy2(source_file, legacy_target)
        except Exception as e:
             print(f"[{datetime.datetime.now()}] Failed to copy network file: {e}")
    
    return source_file

def save_raw_traffic_data(data, output_dir="data/trafficData"):
    """
    Saves the raw downloaded traffic data to a JSON file, similar to tools/CollectData.py.
    """
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
        print(f"[{current_time.strftime('%H:%M:%S')}] Saved RAW traffic data to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving raw data: {e}")
        return None

def run_main_process(road_data):
    """
    Executes the logic from tools/main.py but uses the provided road_data.
    """
    print("Starting SUMO Route Generation...")
    
    # 1. Save RAW data first (as requested)
    save_raw_traffic_data(road_data)
    
    # 2. Ensure network file exists for legacy tools
    net_file = ensure_network_file()

    # Monkey-patch grabapi.getData in selectRoad to return our data
    # selectRoad imports grabapi as gb, so we patch ST.gb.getData
    if hasattr(ST, 'gb') and hasattr(ST.gb, 'getData'):
        original_getData = ST.gb.getData
        ST.gb.getData = lambda: road_data
    else:
        print("Warning: selectRoad module structure unexpected. Patching might not work.")

    try:
        print("Generating initial trips...")
        # ST.select() inside calls gb.getData() (which we patched),
        # filters by boundary (XY coordinates), converts coords, maps to Node IDs.
        selected_roads = ST.select()
        
        CTR.generate_trip(selected_roads)
        
        print("Running DUARouter (1st pass)...")
        # Ensure ./data directory exists
            
        # Note: We use the LEGACY network file name here because 'fixRoadData' might depend on it
        # or other tools might expect it in the './data' folder with that specific name.
        # But we can try using the 'net_file' variable if 'convertToRou' supports it.
        # Check convertToRou.run_duarouter signature: def run_duarouter(net_file, ...)
        
        legacy_net_path = "./data/ntut_network_split.net copy.xml"
        
        CTR.run_duarouter(
            legacy_net_path, 
            "./data/trips.xml", 
            "./data/output.rou.alt.xml",
            "./data/output.rou.xml"
        )
        
        print("Refining road data (imputation)...")
        # fixRoadData calls methods that use ST.select(), which is patched above
        # It also reads ./data/output.rou.xml which we just generated
        # fixRoadData reads "./data/ntut_network_split.net copy.xml" inside getMapData()
        edges_volume = FRD.fixtheRoadData()
        
        print("Generating final trips based on imputed volume...")
        CTR.generate_trip(edges_volume)
        
        print("Running DUARouter (Final pass)...")
        output_alt = "./data/final_output.rou.alt.xml"
        output_rou = "./data/final_output.rou.xml"
        CTR.run_duarouter(
            legacy_net_path, 
            "./data/trips.xml", 
            output_alt,
            output_rou
        )
        
        # Restore patched method
        if hasattr(ST, 'gb') and hasattr(ST.gb, 'getData'):
             ST.gb.getData = original_getData
        
        return output_rou

    except Exception as e:
        print(f"Error during route generation: {e}")
        # Restore patched method
        if hasattr(ST, 'gb') and hasattr(ST.gb, 'getData'):
             ST.gb.getData = original_getData
        import traceback
        traceback.print_exc()
        return None


def save_final_output(source_file, target_dir="data/VehicleData"):
    """Save the generated route file to the target directory."""
    if not source_file or not os.path.exists(source_file):
        print(f"Source file not found: {source_file}")
        return None
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    current_time = datetime.datetime.now()
    # Format: traffic_data_YYYYMMDD_HHMMSS.rou.xml
    filename = f"traffic_data_{current_time.strftime('%Y%m%d_%H%M%S')}.rou.xml"
    target_path = os.path.join(target_dir, filename)
    
    try:
        shutil.copy2(source_file, target_path)
        print(f"[{current_time.strftime('%H:%M:%S')}] Saved final route file to {target_path}")
        return target_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def main():
    """Main function to collect, process, and save traffic data."""
    print("=" * 50)
    print("Traffic Data Collection & Processing Script")
    print(f"Started at: {datetime.datetime.now()}")
    print("=" * 50)
    
    # 1. Download and parse data
    road_info = download_and_parse_data()
    if road_info is None:
        print("Failed to download data")
        return 1
    
    # 2. Process data (run main.py logic)
    output_rou_file = run_main_process(road_info)
    
    if output_rou_file:
        # 3. Save to VehicleData
        filepath = save_final_output(output_rou_file)
        
        if filepath:
            print("=" * 50)
            print(f"Process completed successfully!")
            print(f"Output saved to: {filepath}")
            print("=" * 50)
            return 0
        else:
            print("Failed to save output file.")
            return 1
    else:
        print("Process failed to generate traffic routes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
