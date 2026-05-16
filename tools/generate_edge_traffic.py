#!/usr/bin/env python3
"""
Combine:
  final_output.rou.alt.xml        → vehicle counts per edge
  edgedata_output.xml             → speed / occupancy / density
  ntut_network_split.net copy.xml → edge geometry (WGS84)

Output: edge_heatmap.json  (served via GET /api/edge-heatmap)

可直接執行（使用預設路徑）：
    python tools/generate_edge_traffic.py

或從 runtime_pipeline.py 呼叫：
    from generate_edge_traffic import generate
    generate(net_file=..., rou_file=..., edge_file=..., out_file=...)
"""
import xml.etree.ElementTree as ET
import json, os
from collections import defaultdict

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)

_DEFAULT_NET_FILE  = os.path.join(_ROOT, 'data', 'ntut_network_split.net copy.xml')
_DEFAULT_ROU_FILE  = os.path.join(_ROOT, 'data', 'final_output.rou.alt.xml')
_DEFAULT_EDGE_FILE = os.path.join(_ROOT, 'data', 'edgedata_output.xml')
_DEFAULT_OUT_FILE  = os.path.join(_ROOT, 'TrafficVision Design System', 'data', 'edge_heatmap.json')


def generate(net_file=None, rou_file=None, edge_file=None, out_file=None):
    """
    生成 edge_heatmap.json。

    Args:
        net_file:  SUMO net.xml 路徑（含邊道幾何）
        rou_file:  rou.alt.xml 路徑（含選定路由）
        edge_file: edgedata_output.xml 路徑（SUMO 輸出的速度/佔有率）
        out_file:  輸出 JSON 路徑

    Returns:
        str: out_file 路徑
    """
    net_file  = net_file  or _DEFAULT_NET_FILE
    rou_file  = rou_file  or _DEFAULT_ROU_FILE
    edge_file = edge_file or _DEFAULT_EDGE_FILE
    out_file  = out_file  or _DEFAULT_OUT_FILE

    # ── 1. Vehicle counts per edge ──────────────────────────────────────────
    print('Parsing routes ...')
    rou_root = ET.parse(rou_file).getroot()
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
    print(f'  {total_v} vehicles, {len(edge_counts)} edges with route traffic')

    # ── 2. Traffic metrics from edgedata_output.xml ─────────────────────────
    print('Parsing edgedata ...')
    edge_metrics = {}
    try:
        with open(edge_file, 'r', encoding='utf-8') as fh:
            raw = fh.read()
        end_tag = '</meandata>'
        idx = raw.find(end_tag)
        if idx != -1:
            raw = raw[:idx + len(end_tag)]
        edata_root = ET.fromstring(raw)
        for interval in edata_root.iter('interval'):
            for edge_el in interval.findall('edge'):
                eid = edge_el.get('id', '')

                def _f(attr, default=0.0, _el=edge_el):
                    v = _el.get(attr)
                    try:
                        return float(v) if v else default
                    except ValueError:
                        return default

                edge_metrics[eid] = {
                    'spd':     round(_f('speed') * 3.6, 1),
                    'occ':     round(_f('occupancy'), 2),
                    'density': round(_f('density'), 2),
                    'vol':     int(_f('entered')),
                    'wait':    round(_f('waitingTime'), 1),
                    'tloss':   round(_f('timeLoss'), 1),
                }
        print(f'  {len(edge_metrics)} edges with edgedata metrics')
    except Exception as e:
        print(f'  WARNING: could not parse edgedata ({e}); speed data will be 0')

    # ── 3. Edge geometry from net.xml ────────────────────────────────────────
    print('Parsing network geometry ...')
    net_root = ET.parse(net_file).getroot()

    loc  = net_root.find('location')
    conv = list(map(float, loc.get('convBoundary').split(',')))
    orig = list(map(float, loc.get('origBoundary').split(',')))
    cx, cy = conv[2], conv[3]
    lon0, lat0, lon1, lat1 = orig

    def to_latlon(x, y):
        return [round((y / cy) * (lat1 - lat0) + lat0, 7),
                round((x / cx) * (lon1 - lon0) + lon0, 7)]

    def parse_shape(s):
        pts = []
        for tok in s.strip().split():
            xy = tok.split(',')
            if len(xy) == 2:
                try:
                    pts.append(to_latlon(float(xy[0]), float(xy[1])))
                except ValueError:
                    pass
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
        m = edge_metrics.get(eid, {})
        edges_out[eid] = {
            'name':    edge.get('name', ''),
            'shape':   shape,
            'count':   edge_counts.get(eid, 0),
            'spd':     m.get('spd',     0),
            'occ':     m.get('occ',     0),
            'density': m.get('density', 0),
            'vol':     m.get('vol',     0),
            'wait':    m.get('wait',    0),
            'tloss':   m.get('tloss',   0),
        }
    print(f'  {len(edges_out)} edges with geometry ({skipped} skipped)')

    # ── 4. Summary stats ─────────────────────────────────────────────────────
    max_count = max((v['count'] for v in edges_out.values()), default=1)
    max_vol   = max((v['vol']   for v in edges_out.values()), default=1)
    speeds    = [v['spd'] for v in edges_out.values() if v['spd'] > 0]
    max_speed = max(speeds, default=50.0)
    print(f'  speed range: 0–{max_speed:.1f} km/h  ({len(speeds)} edges with data)')

    # ── 5. Write JSON ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    payload = {
        'meta': {
            'total_vehicles': total_v,
            'max_count':      max_count,
            'max_vol':        max_vol,
            'max_speed':      round(max_speed, 1),
            'bounds': {
                'lat_min': lat0, 'lat_max': lat1,
                'lon_min': lon0, 'lon_max': lon1,
            },
        },
        'edges': edges_out,
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    kb = os.path.getsize(out_file) // 1024
    print(f'Written → {out_file}  ({kb} KB)')
    return out_file


if __name__ == '__main__':
    generate()
    print('Done.')
