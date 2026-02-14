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
    base_dir = "./data/simulation_data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    en_filepath = os.path.join(base_dir, "traffic_data_english.csv")
    
    with open(en_filepath, "w", encoding="utf-8") as f:
        f.write("time,edge_id,vehicle_count\n")

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        current_time = traci.simulation.getTime()
        
        if current_time % 20 == 0:
            lines_en = []
            
            for edge_id in traci.edge.getIDList():
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                if vehicle_count > 0:
                    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                    
                    lines_en.append(f"{current_time},{edge_id},{vehicle_count}\n")
            
            with open(en_filepath, "a", encoding="utf-8") as f:
                f.writelines(lines_en)
                

            
            print(f"Exported data for time {current_time} to consolidated files")
        
        step += 1

    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    sumoBinary = checkBinary('sumo')
    
    traci.start([sumoBinary, "-c", "data/ntut-the way.sumocfg"])
    run()
