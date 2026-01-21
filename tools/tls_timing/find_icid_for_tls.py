"""
Find matching ICIDs for unmapped TLS by coordinate comparison.
Uses the network's location offset to convert SUMO coords to WGS84.
"""
import xml.etree.ElementTree as ET
import json
import math

# Load network
tree = ET.parse('data/legacy/ntut-the way.net.xml')
root = tree.getroot()

# Get network location offset (for coordinate conversion)
location = root.find('.//location')
net_offset = location.get('netOffset', '0,0').split(',')
offset_x, offset_y = float(net_offset[0]), float(net_offset[1])
orig_boundary = location.get('origBoundary', '0,0,0,0').split(',')
conv_boundary = location.get('convBoundary', '0,0,0,0').split(',')

print(f"Network offset: ({offset_x}, {offset_y})")
print(f"Original boundary (WGS84): {orig_boundary}")
print(f"Converted boundary: {conv_boundary}")

# Parse boundaries
orig_min_lon, orig_min_lat, orig_max_lon, orig_max_lat = map(float, orig_boundary)
conv_min_x, conv_min_y, conv_max_x, conv_max_y = map(float, conv_boundary)

# Scale factors
scale_lon = (orig_max_lon - orig_min_lon) / (conv_max_x - conv_min_x) if (conv_max_x - conv_min_x) != 0 else 1
scale_lat = (orig_max_lat - orig_min_lat) / (conv_max_y - conv_min_y) if (conv_max_y - conv_min_y) != 0 else 1

def sumo_to_wgs84(x, y):
    """Convert SUMO coordinates to WGS84 (lon, lat)."""
    lon = orig_min_lon + (x - conv_min_x) * scale_lon
    lat = orig_min_lat + (y - conv_min_y) * scale_lat
    return lon, lat

# Load timing plan JSON
with open('data/source/integrated_timing_plan.json', 'r', encoding='utf-8') as f:
    timing_data = json.load(f)

# Build ICID -> (lon, lat, name) mapping (unique ICIDs)
icid_locations = {}
for entry in timing_data:
    icid = entry.get('icid')
    if icid and icid not in icid_locations:
        lon = entry.get('Longitude')
        lat = entry.get('Latitude')
        name = entry.get('IntersectionName', entry.get('icname_plan', ''))
        if lon and lat:
            icid_locations[icid] = (lon, lat, name)

print(f"\nLoaded {len(icid_locations)} unique ICIDs with coordinates")

# Unmapped TLS
unmapped = [
    ('3208478121', 742.06, 292.27),
    ('619136999', 795.46, 295.24),
    ('cluster_4374470814_4374470816_656132411_656132475', 402.66, 678.20),
]

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two WGS84 points."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

print("\n" + "=" * 80)
print("未對應 TLS 與最近 ICID 配對結果：")
print("=" * 80)

for tls_id, x, y in unmapped:
    lon, lat = sumo_to_wgs84(x, y)
    print(f"\nTLS: {tls_id}")
    print(f"  SUMO座標: ({x}, {y})")
    print(f"  WGS84座標: ({lon:.6f}, {lat:.6f})")
    
    # Find nearest ICIDs
    distances = []
    for icid, (ic_lon, ic_lat, ic_name) in icid_locations.items():
        dist = haversine_distance(lon, lat, ic_lon, ic_lat)
        distances.append((dist, icid, ic_name, ic_lon, ic_lat))
    
    distances.sort()
    print("  最近的 5 個 ICID:")
    for dist, icid, name, ic_lon, ic_lat in distances[:5]:
        print(f"    {icid}: {name} (距離: {dist:.1f}m)")
