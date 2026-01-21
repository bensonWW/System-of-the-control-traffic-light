
import sumolib

NET_FILE = "data/real_world.net.xml"

# Targets (lat, lon)
TARGETS = {
    "Zhongxiao_Jianguo (IJHKR)": (25.0415, 121.5365),
    "Renai_Xinsheng": (25.0383, 121.5330)
}

def main():
    print(f"Loading {NET_FILE}...")
    net = sumolib.net.readNet(NET_FILE)
    
    for name, (lat, lon) in TARGETS.items():
        # Convert lat/lon to query x/y match
        # sumolib handles conversion if projection is properly set in net
        try:
            x, y = net.convertLonLat2XY(lon, lat)
            nodes = net.getNeighboringNodes(x, y, 100) # 100m radius
            if nodes:
                closest_node, dist = nodes[0]
                print(f"\nTarget: {name}")
                print(f"  Closest Junction ID: {closest_node.getID()}")
                print(f"  Distance: {dist:.2f}m")
                print(f"  Type: {closest_node.getType()}")
            else:
                print(f"\nTarget: {name} - No nodes found within 100m")
        except Exception as e:
            print(f"Error checking {name}: {e}")

if __name__ == "__main__":
    main()
