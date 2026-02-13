import xml.etree.ElementTree as ET
import os 
import sys
import subprocess
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
import selectRoad as ST
def run_duarouter(net_file, initial_trip_file, final_output_alt, final_output_rou):
    current_input = initial_trip_file
    temp_output_alt = final_output_alt + ".tmp" 
    for time in range(5):
        print(f"--- Running DUARouter Iteration {time} ---")
        print(f"Input: {current_input}")
        print(f"Output: {temp_output_alt}")
        cmd = [
            "duarouter",
            "-n", net_file,
            "-r", current_input,
            "-o", final_output_rou,
            "--alternatives-output", temp_output_alt,
            "--max-alternatives", "3",
            "--weights.random-factor", "12",
            "--junction-taz",
            "--ignore-errors"
        ]
        p = subprocess.run(cmd, text=True, capture_output=True)
        if p.stderr:
            print("STDERR:\n", p.stderr[:2000])
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
        else:
            print("Iteration success.")
        if os.path.exists(temp_output_alt):
            if os.path.exists(final_output_alt):
                os.remove(final_output_alt)
            os.rename(temp_output_alt, final_output_alt)
            current_input = final_output_alt
        else:
            print("Error: Output file was not generated.")
            break
    if os.path.exists("./data/trips.xml"):
        os.remove("./data/trips.xml")
    
    print("DUA Loop Completed.")
def generate_trip(road_info):  
    trip = ET.Element("routes")
    percentage = 0.8
    simulation_duration = 300
    for index,roadName in  enumerate(road_info):
        ET.SubElement(trip,
                      "vType",id = f"Car{index}",
                      vClass = "passenger",color = "0,1,0",
                      length = "5")
        observed_volume_per_hour = float(road_info[roadName].get("TotalVol", 0))
        if observed_volume_per_hour > 0:
            ET.SubElement(trip, 
                          "flow",
                          id=f"Flow{index}",
                          type=f"Car{index}",
                          fromJunction=road_info[roadName]["from"],
                          toJunction=road_info[roadName]["to"],
                          begin="0",                
                          end=str(simulation_duration), 
                          number=str(int(observed_volume_per_hour)), # 使用實際的車流量 (輛/小時)
                          )
    ET.ElementTree(trip).write("./data/trips.xml",encoding="utf-8",xml_declaration=True)
    print("已生成")
if __name__ == "__main__":
    generate_trip(ST.select())
    run_duarouter("./data/ntut_network_split.net copy.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")