
import xml.etree.ElementTree as ET
import math

NET_FILE = "data/real_world.net.xml"
# Target: Zhongxiao/Jianguo (IJHKR)
TARGET_LAT = 25.0415
TARGET_LON = 121.5365

def main():
    print("Reading location info...")
    location_elem = None
    for event, elem in ET.iterparse(NET_FILE, events=("start",)):
        if elem.tag == "location":
            location_elem = elem
            break
            
    if location_elem is None:
        print("No location element found.")
        return

    orig = [float(x) for x in location_elem.get("origBoundary").split(",")]
    conv = [float(x) for x in location_elem.get("convBoundary").split(",")]
    
    # orig: lon_min, lat_min, lon_max, lat_max
    # conv: x_min, y_min, x_max, y_max
    
    lon_min, lat_min, lon_max, lat_max = orig
    x_min, y_min, x_max, y_max = conv
    
    # Linear projection
    def to_xy(lon, lat):
        x = x_min + (lon - lon_min) * (x_max - x_min) / (lon_max - lon_min)
        y = y_min + (lat - lat_min) * (y_max - y_min) / (lat_max - lat_min)
        return x, y

    tx, ty = to_xy(TARGET_LON, TARGET_LAT)
    print(f"Target projected: {tx}, {ty}")
    
    # Find closest junction
    print("Scanning junctions...")
    min_dist = float('inf')
    best_id = None
    
    # Re-parse to find junctions
    for event, elem in ET.iterparse(NET_FILE, events=("start", "end")):
        if event == "start" and elem.tag == "junction":
            try:
                jx = float(elem.get("x"))
                jy = float(elem.get("y"))
                dist = math.sqrt((jx - tx)**2 + (jy - ty)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_id = elem.get("id")
            except:
                pass
        elem.clear()
        
    print(f"Closest Junction: {best_id}")
    print(f"Distance: {min_dist}")

if __name__ == "__main__":
    main()
