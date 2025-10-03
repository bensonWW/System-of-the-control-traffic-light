import xml.etree.ElementTree as ET
import grabapi as gb
import searchnetdata as sd
import select
from xml.dom import minidom
from pathlib import Path

def create_route_from_road(road_data):
    route = " ".join([road_data["from"], road_data["to"]]) 
    return route

def generate_rou_file(road_info):  
    routes = ET.Element("routes")

    vType = ET.SubElement(routes, "vType", id="car", vClass="passenger", color="1,0,0", length="5", minGap="2.5", maxSpeed="13.9")

    for road_name, road_data in road_info.items():
        route_id = f"route_{road_name}"
        route_edges = create_route_from_road(road_data)  

        route = ET.SubElement(routes, "route", id=route_id, edges=route_edges)

        vehicle_id = f"vehicle_{road_name}"
        ET.SubElement(routes, "vehicle", id=vehicle_id, type="car", route=route_id, depart="0", departLane="best", departPos="0")

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)  

    out_path = out_dir / "output.rou.xml"

    tree = ET.ElementTree(routes)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    tree = ET.parse(out_path)
    root = tree.getroot()
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(reparsed.toprettyxml(indent="  "))

    print(f"[INFO] `.rou.xml` 文件已儲存至：{out_path.resolve()}")

road_info = gb.getData()

lon_min, lat_min, lon_max, lat_max = sd.getmapscope()

filtered_road_info = {}

for road_name, road_data in road_info.items():
    StartWgsX = float(road_data["StartWgsX"])
    StartWgsY = float(road_data["StartWgsY"])
    EndWgsX = float(road_data["EndWgsX"])
    EndWgsY = float(road_data["EndWgsY"])
    if select.inrange(lon_min, lat_min, lon_max, lat_max, StartWgsX, StartWgsY, EndWgsX, EndWgsY):
        filtered_road_info[road_name] = road_data

converted_road_info = select.convert(filtered_road_info, lon_min, lat_min, lon_max, lat_max)

for road_name, road_data in converted_road_info.items():
    road_data["from"] = sd.search(road_data["StartWgsX"], road_data["StartWgsY"])  
    road_data["to"] = sd.search(road_data["EndWgsX"], road_data["EndWgsY"])  
    
generate_rou_file(converted_road_info)