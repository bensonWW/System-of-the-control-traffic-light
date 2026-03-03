import os
import sys
import glob
import time
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime
import multiprocessing
from tqdm import tqdm

# Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

# Configuration
VEHICLE_DATA_DIR = "./data/VehicleData"
OUTPUT_DIR = "./data/simulation_check_data"
BASE_SUMOCFG = "./data/ntut-the way.sumocfg"

def get_timestamp(filename):
    """Extracts timestamp (YYYYMMDD_HHMMSS) from filename."""
    parts = os.path.basename(filename).split('_')
    return f"{parts[-2]}_{parts[-1].split('.')[0]}" if len(parts) >= 4 else None

def create_temp_cfg(route_file, base_cfg, temp_cfg_path):
    """Creates a temporary sumocfg pointing to the route file."""
    try:
        tree = ET.parse(base_cfg)
        tree.getroot().find('input/route-files').set('value', os.path.abspath(route_file))
        tree.write(temp_cfg_path)
        return True
    except AttributeError:
        print("Error: Invalid sumocfg structure.")
        return False

def filter_routes(input_rou, output_rou, valid_edges):
    """Filters out vehicles with invalid edges. Supports both <route> and <routeDistribution>."""
    try:
        tree = ET.parse(input_rou)
        root = tree.getroot()
        
        valid_vehicles = []
        for v in root.findall('vehicle'):
            # Case 1: Direct route
            route = v.find('route')
            if route is not None:
                edges = route.get('edges', '').split()
                if edges and all(e in valid_edges for e in edges):
                    valid_vehicles.append(v)
                continue
            
            # Case 2: routeDistribution
            rd = v.find('routeDistribution')
            if rd is not None:
                routes = rd.findall('route')
                # If ANY route in the distribution is valid, we keep the vehicle 
                # (or we could filter the distribution itself, but keeping it simpler for now)
                all_routes_valid = True
                for r in routes:
                    edges = r.get('edges', '').split()
                    if not edges or not all(e in valid_edges for e in edges):
                        all_routes_valid = False
                        break
                
                if routes and all_routes_valid:
                    valid_vehicles.append(v)
        
        # Identify other elements to keep (vTypes, etc.)
        children_to_keep = []
        for child in root:
            if child.tag == 'vehicle':
                if child in valid_vehicles:
                    children_to_keep.append(child)
            else:
                children_to_keep.append(child)
        
        # Rebuild root
        root.clear()
        root.tag = 'routes'
        for child in children_to_keep:
            root.append(child)
            
        tree.write(output_rou)
        return len(valid_vehicles)
    except Exception as e:
        print(f"Filter error {input_rou}: {e}")
        return 0

def run_simulation(config_file, output_csv):
    """Runs SUMO simulation and saves traffic data."""
    # Each process has its own traci instance state, so we just need to ensure start/close
    
    sumoBinary = sumolib.checkBinary('sumo')
    # Using a unique label can help if we needed to access specific instances, 
    # but separate processes usually isolate default instance fine.
    # To be safe against port conflicts/race conditions, let traci pick ports.
    cmd = [sumoBinary, "-c", config_file, "--start", "--quit-on-end", "--no-warnings"]
    
    try:
        traci.start(cmd)
        end_time = traci.simulation.getEndTime()
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write("time,edge_id,vehicle_count\n")
            
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Skip first 50s and last 100s
                if 50 <= current_time <= (end_time - 100):
                    if current_time % 20 == 0:
                        lines = []
                        for edge in traci.edge.getIDList():
                            count = traci.edge.getLastStepVehicleNumber(edge)
                            if count > 0:
                                lines.append(f"{current_time},{edge},{count}\n")
                        f.writelines(lines)
        return True
    except Exception as e:
        print(f"Sim error {config_file}: {e}")
        return False
    finally:
        try:
            traci.close()
        except:
            pass
        sys.stdout.flush()

def process_file_wrapper(args):
    """Wrapper to unpack arguments for process_file."""
    return process_file(*args)

def process_file(rou_file, valid_edges):
    """Worker function to process a single file."""
    ts = get_timestamp(rou_file)
    if not ts: return f"Skipped: {os.path.basename(rou_file)}"

    # Unique temp filenames using timestamp AND process ID to be extra safe (though TS should be unique)
    pid = os.getpid()
    temp_rou = os.path.join("./data", f"temp_routes_{ts}_{pid}.rou.xml")
    temp_cfg = os.path.join("./data", f"temp_{ts}_{pid}.sumocfg")
    out_csv = os.path.join(OUTPUT_DIR, f"traffic_data_{ts}.csv")

    try:
        if filter_routes(rou_file, temp_rou, valid_edges) > 0:
            if create_temp_cfg(temp_rou, BASE_SUMOCFG, temp_cfg):
                if run_simulation(os.path.abspath(temp_cfg), out_csv):
                    return f"Finished: {os.path.basename(rou_file)}"
                else:
                    return f"Failed Sim: {os.path.basename(rou_file)}"
            else:
                return f"Failed Config: {os.path.basename(rou_file)}"
        else:
             return f"Skipped (0 valid): {os.path.basename(rou_file)}"
    except Exception as e:
        return f"Error {os.path.basename(rou_file)}: {e}"
    finally:
        # Cleanup
        for f in [temp_cfg, temp_rou]:
            if os.path.exists(f): 
                try:
                    os.remove(f)
                except:
                    pass

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load network edges
    print("Loading network validation...")
    try:
        tree = ET.parse(BASE_SUMOCFG)
        net_rel = urllib.parse.unquote(tree.getroot().find("input/net-file").get("value"))
        net_path = os.path.normpath(os.path.join(os.path.dirname(BASE_SUMOCFG), net_rel))
        
        valid_edges = {e.getID() for e in sumolib.net.readNet(net_path).getEdges()}
        print(f"Loaded {len(valid_edges)} valid edges.")
    except Exception as e:
        print(f"Network load failed: {e}")
        return

    # Process files
    files = sorted(glob.glob(os.path.join(VEHICLE_DATA_DIR, "*.rou.xml")))
    print(f"Found {len(files)} files. Starting pool of 16 processes...")

    # Prepare arguments for each task
    tasks = [(f, valid_edges) for f in files]
    
    # Run pool
    start_time = time.time()
    with multiprocessing.Pool(processes=16) as pool:
        # Wrap the imap with tqdm for a progress bar
        for i, result in enumerate(tqdm(pool.imap_unordered(process_file_wrapper, tasks), total=len(files), desc="Processing"), 1):
            pass  # The progress bar handles the display

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows
    main()
