import xml.etree.ElementTree as ET
import os 
import sys
import subprocess
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
import select as ST
def run_duarouter(net_file, trip_file,outputalt_file,output_file):
    for time in range(5):
        cmd = [
            "duarouter",
            "-n", net_file,
            "-r", trip_file,          # 用 -r：route input (可含 trips/flows/vehicles)
            "-o", output_file,
            "--alternatives-output",outputalt_file,
            "--max-alternatives", "3",
            "--weights.random-factor", "1.75",
            "--junction-taz",      # ← 加這一行
            "--ignore-errors"
        ]
        p = subprocess.run(cmd, text=True, capture_output=True)
        print("STDOUT:\n", p.stdout)
        print("STDERR:\n", p.stderr)   # ← 這行會告訴你真正的錯
        trip_file = outputalt_file
    p.check_returncode()
def generate_trip(road_info):  
    trip = ET.Element("routes")
    percentage = 0.8
    for index,roadName in  enumerate(road_info):
        ET.SubElement(trip,
                      "vType",id = f"Car{index}",
                      vClass = "passenger",color = "0,1,0",
                      length = "5",
                      maxSpeed = str((float(road_info[roadName]["AvgSpd"]) / percentage) /3.6))
        ET.SubElement(trip,"trip",
                      id = f"Trip{index}",
                      type = f"Car{index}",
                      fromJunction = road_info[roadName]["from"],
                      toJunction = road_info[roadName]["to"],
                      depart = "0")
    ET.ElementTree(trip).write("./data/trips.xml",encoding="utf-8",xml_declaration=True)
    print("已生成")


generate_trip(ST.select())
run_duarouter("./data/ntut-the way.net.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")