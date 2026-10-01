import os
import sys
import glob
import time
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime
import multiprocessing

# Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

# Configuration
# VEHICLE_DATA_DIR = "./data/VehicleData"
# OUTPUT_DIR = "./data/simulation_data"
VEHICLE_DATA_DIR = "./data/VehicleData_check"
OUTPUT_DIR = "./data/simulation_data_check"
BASE_SUMOCFG = "./data/ntut_config.sumocfg"

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
    """Filters out vehicles with invalid edges."""
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
    # Explicit free port + numRetries: with 16 parallel workers, traci's
    # default random port picker has a non-trivial collision rate. Asking the
    # OS for a free port narrows the race window; numRetries=10 absorbs the
    # remaining TOCTOU window between port acquisition and SUMO bind.
    port = sumolib.miscutils.getFreeSocketPort()
    label = f"vd_{os.getpid()}_{port}"
    cmd = [sumoBinary, "-c", config_file, "--start", "--quit-on-end"]

    try:
        traci.start(cmd, port=port, label=label, numRetries=10)
        traci.switch(label)
        data_buffer = []  # Buffer to store data for filtering later

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            # Record data every 20 seconds
            if current_time % 20 == 0:
                for edge in traci.edge.getIDList():
                    count = traci.edge.getLastStepVehicleNumber(edge)
                    if count > 0:
                        # 取該 edge 上所有車輛的平均速度 (TraCI 回 m/s) → 轉成 km/h
                        # 只在 count > 0 時取,避免 TraCI 對空 edge 回傳特殊值 (-1001 等)
                        mean_speed_ms = traci.edge.getLastStepMeanSpeed(edge)
                        avg_speed_kmh = mean_speed_ms * 3.6
                        data_buffer.append((current_time, edge, count, avg_speed_kmh))

        if not data_buffer:
            return True # No data, but simulation finished

        actual_end_time = data_buffer[-1][0]

        # Write filtered data to CSV
        # 新增 avg_speed_kmh 欄位; 下游 (train_model / predict_to_csv / traffic_light_optimizer)
        # 都用具名欄位讀 vehicle_count,多一欄不影響既有功能。
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write("time,edge_id,vehicle_count,avg_speed_kmh\n")
            lines = []
            # Skip first 60s and last 100s
            for t, edge, count, speed in data_buffer:
                if 60 <= t <= (actual_end_time - 100):
                    lines.append(f"{t},{edge},{count},{speed:.2f}\n")
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
        input_net = tree.getroot().find("input/net-file")
        if input_net is None:
            print("Error: Could not find 'input/net-file' in sumocfg.")
            return

        net_rel = urllib.parse.unquote(input_net.get("value"))
        net_path = os.path.normpath(os.path.join(os.path.dirname(BASE_SUMOCFG), net_rel))
        
        if not os.path.exists(net_path):
            print(f"Error: Network file not found at {net_path}")
            return
        
        valid_edges = {e.getID() for e in sumolib.net.readNet(net_path).getEdges()}
        print(f"Loaded {len(valid_edges)} valid edges.")
    except Exception as e:
        print(f"Network load failed: {e}")
        return

    # Process files
    files = sorted(glob.glob(os.path.join(VEHICLE_DATA_DIR, "*.rou.xml")))

    # Adaptive worker count: each SUMO process pulls ~300-500 MB; on a 16 GB
    # dev box, 16 workers is right at the edge of OOM. Use CPU count as a
    # ceiling (caller can override via env if they know what they're doing).
    # Cap at 16 to preserve the existing tuned upper bound for large servers.
    DEFAULT_WORKERS = min(20, multiprocessing.cpu_count() or 1)
    workers = int(os.environ.get("TRAFFICVISION_VD_WORKERS", DEFAULT_WORKERS))
    workers = max(1, min(workers, len(files) or 1))
    print(f"Found {len(files)} files. Starting pool of {workers} processes "
          f"(cpu_count={multiprocessing.cpu_count()}, override via TRAFFICVISION_VD_WORKERS)")

    # Prepare arguments for each task
    tasks = [(f, valid_edges) for f in files]

    # Run pool
    start_time = time.time()
    with multiprocessing.Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file_wrapper, tasks), 1):
            elapsed = time.time() - start_time
            print(f"[{i}/{len(files)}] {result} (Time: {elapsed:.2f}s)")

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows
    main()