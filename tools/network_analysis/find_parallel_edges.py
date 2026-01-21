
import xml.etree.ElementTree as ET
import math

NET_FILE = "data/real_world.net.xml"

def get_edge_endpoints(shape_str):
    """Extract start and end points from a shape string."""
    try:
        points = shape_str.split()
        if len(points) < 2:
            return None, None
        start = [float(x) for x in points[0].split(',')]
        end = [float(x) for x in points[-1].split(',')]
        return tuple(start), tuple(end)
    except:
        return None, None

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    print(f"Loading {NET_FILE}...")
    
    edges = {}  # id -> (start, end, shape)
    
    for event, elem in ET.iterparse(NET_FILE, events=("start",)):
        if elem.tag == "edge" and not elem.get("id", "").startswith(":"):
            shape = elem.get("shape")
            if shape:
                start, end = get_edge_endpoints(shape)
                if start and end:
                    edges[elem.get("id")] = (start, end, shape)
    
    print(f"Found {len(edges)} edges.")
    
    # Find parallel edges (close start/end points, similar direction)
    print("\nLooking for parallel edge pairs within 30m...")
    parallel_pairs = []
    
    edge_list = list(edges.items())
    for i, (id1, (s1, e1, _)) in enumerate(edge_list):
        for id2, (s2, e2, _) in edge_list[i+1:]:
            # Check if starts and ends are close
            if distance(s1, s2) < 30 and distance(e1, e2) < 30:
                parallel_pairs.append((id1, id2, distance(s1, s2), distance(e1, e2)))
    
    print(f"Found {len(parallel_pairs)} potential parallel edge pairs.")
    for id1, id2, d1, d2 in parallel_pairs[:20]:
        print(f"  {id1} <-> {id2} (start dist: {d1:.1f}m, end dist: {d2:.1f}m)")

if __name__ == "__main__":
    main()
