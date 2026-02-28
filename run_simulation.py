import os
import sys
import traci

# Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary

def run():
    # Create output directories
    base_dir = "traffic_data"
    en_dir = os.path.join(base_dir, "english")
    ch_dir = os.path.join(base_dir, "chinese")
    
    for d in [en_dir, ch_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Initialize output files with headers
    en_filepath = os.path.join(en_dir, "traffic_data_english.csv")
    ch_filepath = os.path.join(ch_dir, "traffic_data_chinese.csv")
    
    with open(en_filepath, "w", encoding="utf-8") as f:
        f.write("time,edge_id,vehicle_count\n")
        
    with open(ch_filepath, "w", encoding="utf-8-sig") as f:
        f.write("時間,路段ID,車輛數\n")

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        current_time = traci.simulation.getTime()
        
        # Capture data every 20 seconds
        if current_time % 20 == 0:
            lines_en = []
            lines_ch = []
            
            # Iterate over all edges
            for edge_id in traci.edge.getIDList():
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                if vehicle_count > 0:
                    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                    
                    # Append data line
                    lines_en.append(f"{current_time},{edge_id},{vehicle_count}\n")
                    lines_ch.append(f"{current_time},{edge_id},{vehicle_count}\n")
            
            # Append to English file
            with open(en_filepath, "a", encoding="utf-8") as f:
                f.writelines(lines_en)
                
            # Append to Chinese file
            with open(ch_filepath, "a", encoding="utf-8-sig") as f:
                f.writelines(lines_ch)
            
            print(f"Exported data for time {current_time} to consolidated files")
        
        step += 1

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    # Choose sumo binary
    # Use sumo-gui if you want to see it, or sumo for command line only
    sumoBinary = checkBinary('sumo')
    
    # Or use sumo-gui
    #sumoBinary = checkBinary('sumo-gui')

    # Start TraCI
    # Added --start to auto-start simulation
    # Added --delay 100 to slow down simulation (100ms per step) for visibility
    #traci.start([sumoBinary, "-c", "data/ntut-the way.sumocfg", "--tripinfo-output", "tripinfo.xml", "--start", "--delay", "100"])
    traci.start([sumoBinary, "-c", "data/ntut_config.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    run()
