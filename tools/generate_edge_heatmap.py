#!/usr/bin/env python3
"""
Parse final_output.rou.alt.xml + ntut_network_split.net copy.xml
and produce edge_heatmap.json for the TrafficVision heatmap layer.
"""
import xml.etree.ElementTree as ET
import json, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

NET_FILE = os.path.join(ROOT, 'data', 'ntut_network_split.net copy.xml')
ROU_FILE = os.path.join(ROOT, 'data', 'final_output.rou.alt.xml')
OUT_FILE = os.path.join(ROOT, 'TrafficVision Design System', 'data', 'edge_heatmap.json')

# ── 1. Count vehicles per edge (use the selected route at index `last`) ────────
print('Parsing routes...')
rou_root = ET.parse(ROU_FILE).getroot()
edge_counts = defaultdict(int)

for vehicle in rou_root.findall('vehicle'):
    rd = vehicle.find('routeDistribution')
    if rd is None:
        continue
    last = int(rd.get('last', 0))
    routes = rd.findall('route')
    if last < len(routes):
        for eid in routes[last].get('edges', '').split():
            edge_counts[eid] += 1

total_v = len(rou_root.findall('vehicle'))
print(f'  {total_v} vehicles, {len(edge_counts)} unique edges with traffic')

# ── 2. Parse network edges → WGS84 shapes ──────────────────────────────────────
print('Parsing network...')
net_root = ET.parse(NET_FILE).getroot()

loc   = net_root.find('location')
conv  = list(map(float, loc.get('convBoundary').split(',')))
orig  = list(map(float, loc.get('origBoundary').split(',')))
cx, cy = conv[2], conv[3]          # 895.34, 692.75
lon0, lat0, lon1, lat1 = orig      # WGS84 bounds

def to_latlon(x, y):
    return [round((y / cy) * (lat1 - lat0) + lat0, 7),
            round((x / cx) * (lon1 - lon0) + lon0, 7)]

def parse_shape(s):
    pts = []
    for tok in s.strip().split():
        xy = tok.split(',')
        if len(xy) == 2:
            pts.append(to_latlon(float(xy[0]), float(xy[1])))
    return pts

edges_out = {}
skipped   = 0

for edge in net_root.findall('edge'):
    eid = edge.get('id', '')
    if eid.startswith(':'):
        continue
    shape_str = edge.get('shape', '')
    if not shape_str:
        lane = edge.find('lane')
        if lane is not None:
            shape_str = lane.get('shape', '')
    if not shape_str:
        skipped += 1
        continue
    shape = parse_shape(shape_str)
    if len(shape) < 2:
        skipped += 1
        continue
    edges_out[eid] = {
        'name':  edge.get('name', ''),
        'shape': shape,
        'count': edge_counts.get(eid, 0),
    }

print(f'  {len(edges_out)} edges with geometry ({skipped} skipped, no shape)')

max_count = max((v['count'] for v in edges_out.values()), default=1)
used      = sum(1 for v in edges_out.values() if v['count'] > 0)
print(f'  count range: 0–{max_count}  edges with traffic: {used}')

# ── 3. Write JSON ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
out = {
    'meta': {
        'total_vehicles': total_v,
        'max_count':      max_count,
        'bounds': {'lat_min': lat0, 'lat_max': lat1, 'lon_min': lon0, 'lon_max': lon1},
    },
    'edges': edges_out,
}
# Atomic write — the frontend polls this file live; a partial write would
# crash JSON.parse on the dashboard.
tmp_out = OUT_FILE + '.tmp'
with open(tmp_out, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_out, OUT_FILE)

kb = os.path.getsize(OUT_FILE) // 1024
print(f'Written → {OUT_FILE}  ({kb} KB)')
