
import xml.etree.ElementTree as ET
import pandas as pd
import math
import os

NET_XML = "ntut-the way.net.xml"
TIMING_JSON = "integrated_timing_plan.json"

def parse_net_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # 1. Get Boundaries for Coordinate Conversion
    location = root.find("location")
    if location is None:
        print("Error: No <location> tag found in .net.xml")
        return None, []
    
    conv_boundary = [float(v) for v in location.get("convBoundary").split(",")]
    orig_boundary = [float(v) for v in location.get("origBoundary").split(",")]
    
    # map (x,y) from convBoundary to (lon,lat) in origBoundary
    # conv: minX, minY, maxX, maxY
    # orig: minLon, minLat, maxLon, maxLat
    
    min_x, min_y, max_x, max_y = conv_boundary
    min_lon, min_lat, max_lon, max_lat = orig_boundary
    
    x_range = max_x - min_x
    y_range = max_y - min_y
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    def xy_to_lonlat(x, y):
        # Linear interpolation
        if x_range == 0 or y_range == 0:
            return min_lon, min_lat
        lon = min_lon + ((x - min_x) / x_range) * lon_range
        lat = min_lat + ((y - min_y) / y_range) * lat_range
        return lon, lat

    # 2. Get Traffic Light Junctions
    tls_junctions = []
    
    # Find junctions with type="traffic_light"
    for junction in root.findall("junction"):
        if junction.get("type") == "traffic_light":
            jid = junction.get("id")
            x = float(junction.get("x"))
            y = float(junction.get("y"))
            lon, lat = xy_to_lonlat(x, y)
            tls_junctions.append({
                "sumo_id": jid,
                "x": x,
                "y": y,
                "sumo_lon": lon,
                "sumo_lat": lat
            })
            
    # Also finding tlLogic IDs in case they differ (e.g. joined TLS)
    # But usually we map physical location (junction) to real world data.
    
    return tls_junctions

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000 # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def compare_data():
    print("--- Comparing SUMO Network with Integrated Traffic Data ---")
    
    # 1. Parse SUMO Network
    print(f"Parsing {NET_XML}...")
    sumo_tls = parse_net_xml(NET_XML)
    print(f"Found {len(sumo_tls)} traffic light junctions in SUMO network.")
    
    # 2. Load Integrated Data
    print(f"Loading {TIMING_JSON}...")
    if not os.path.exists(TIMING_JSON):
        print(f"Error: {TIMING_JSON} not found.")
        return
        
    df_plan = pd.read_json(TIMING_JSON)
    # Ensure coordinates are float
    df_plan['Longitude'] = pd.to_numeric(df_plan['Longitude'], errors='coerce')
    df_plan['Latitude'] = pd.to_numeric(df_plan['Latitude'], errors='coerce')
    
    # Filter out records without coordinates
    df_plan = df_plan.dropna(subset=['Longitude', 'Latitude'])
    print(f"Loaded {len(df_plan)} records with valid coordinates from JSON.")
    
    # 3. Match
    print("\nMatching Results (Threshold: 50m):")
    print(f"{'SUMO ID':<20} | {'Matched ICID':<10} | {'Dist (m)':<10} | {'Name':<30}")
    print("-" * 80)
    
    matches = []
    
    for tls in sumo_tls:
        best_match = None
        min_dist = float('inf')
        
        # Iterate through all plan records (optimization: could use spatial index, but N is small enough for now)
        # Using a subset if efficient, or just loop
        # Vectorized distance calculation is faster
        
        # Calculate distances to all points
        # For simplicity in this script without heavy scipy deps, we'll just loop or use pandas apply
        # Since 100 * 25k = 2.5m, it might take a few seconds. Let's filter by bounding box first?
        
        # Filter candidate box (approx +/- 0.01 deg) to speed up
        lat_bound = 0.01
        lon_bound = 0.01
        candidates = df_plan[
            (df_plan['Latitude'] > tls['sumo_lat'] - lat_bound) & 
            (df_plan['Latitude'] < tls['sumo_lat'] + lat_bound) &
            (df_plan['Longitude'] > tls['sumo_lon'] - lon_bound) &
            (df_plan['Longitude'] < tls['sumo_lon'] + lon_bound)
        ]
        
        for _, row in candidates.iterrows():
            dist = haversine_distance(tls['sumo_lat'], tls['sumo_lon'], row['Latitude'], row['Longitude'])
            if dist < min_dist:
                min_dist = dist
                best_match = row
        
        if min_dist < 50: # Threshold 50 meters
            matches.append({
                "sumo_id": tls['sumo_id'],
                "icid": best_match['icid'],
                "dist": min_dist,
                "name": best_match.get('IntersectionName', 'N/A') or best_match.get('路口名稱', 'N/A')
            })
            print(f"{tls['sumo_id']:<20} | {best_match['icid']:<10} | {min_dist:.1f}       | {best_match.get('IntersectionName', '')[:28]}")
        else:
            print(f"{tls['sumo_id']:<20} | {'No Match':<10} | {min_dist:.1f} (nearest)")

    print("-" * 80)
    print(f"Total Matches: {len(matches)} / {len(sumo_tls)}")
    
    matched_ids = set(m['sumo_id'] for m in matches)
    unmatched = [tls for tls in sumo_tls if tls['sumo_id'] not in matched_ids]

    print("\n--- Unmatched SUMO Traffic Lights ---")
    with open("unmatched_list.txt", "w", encoding="utf-8") as f:
        for u in unmatched:
             msg = f"ID: {u['sumo_id']}, Loc: ({u['x']:.2f}, {u['y']:.2f}) -> Approx Lon/Lat: ({u['sumo_lon']:.5f}, {u['sumo_lat']:.5f})"
             print(msg)
             f.write(msg + "\n")

    if matches:
        df_matches = pd.DataFrame(matches)
        output_csv = "sumo_json_mapping.csv"
        df_matches.to_csv(output_csv, index=False, encoding='utf_8_sig')
        print(f"\nMapping saved to {output_csv}")

if __name__ == "__main__":
    compare_data()
