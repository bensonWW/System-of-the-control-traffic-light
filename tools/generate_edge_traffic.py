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
_DEFAULT_VD_DIR    = os.path.join(_ROOT, 'TrafficVision Design System', 'data', 'trafficData')

# 顯示用車流量縮放倍數（模擬累積車流量偏大時除以此值；VD 真實車流不受影響）
VOL_SCALE = 3

# VD 涵蓋的幹道前綴（與 serve_api._ROAD_PREFIX_MAP 一致；新生北路併入新生南路）
_ROAD_PREFIXES = ['忠孝東路', '八德路', '市民大道', '建國北路', '建國南路', '新生南路', '新生北路', '松江路']
_PREFIX_CANON  = {'新生北路': '新生南路'}


def _road_prefix(name):
    """edge/VD 路段名 → 大分類前綴（無對應回 None）。"""
    for p in _ROAD_PREFIXES:
        if name.startswith(p):
            return _PREFIX_CANON.get(p, p)
    return None


def _load_vd_road_speeds(vd_dir=None):
    """讀最新的 VD 快取 JSON，回傳 {大分類路名: 平均車速 km/h}。"""
    import glob
    vd_dir = vd_dir or _DEFAULT_VD_DIR
    files = sorted(glob.glob(os.path.join(vd_dir, '*.json')))
    if not files:
        return {}
    data = json.load(open(files[-1], encoding='utf-8')).get('data', {})
    acc = {}
    for section, v in data.items():
        pref = _road_prefix(section)
        spd = v.get('AvgSpd', 0) or 0
        if pref and spd > 0:
            acc.setdefault(pref, []).append(spd)
    return {p: sum(xs) / len(xs) for p, xs in acc.items()}


def generate(net_file=None, rou_file=None, edge_file=None, out_file=None,
             calibrate=True, vd_dir=None):
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
        intervals = list(edata_root.iter('interval'))

        # 只取「最繁忙的時間窗」（進入車數最多），代表尖峰 5 分鐘車況，
        # 而非整場模擬累積（會把車流量灌大、速度被全程平均拉低）。
        def _interval_load(iv):
            return sum(
                float(e.get('entered', 0) or 0)
                for e in iv.findall('edge')
            )

        chosen = max(intervals, key=_interval_load) if intervals else None
        if chosen is not None:
            for edge_el in chosen.findall('edge'):
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
            print(f'  使用時間窗 begin={chosen.get("begin")} end={chosen.get("end")}；'
                  f'{len(edge_metrics)} edges（共 {len(intervals)} 個窗）')
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
        # 車流量顯示縮放：模擬車流量除以 VOL_SCALE（VD 當前車流不受影響，那是真實感測值）
        edges_out[eid] = {
            'name':    edge.get('name', ''),
            'shape':   shape,
            'count':   round(edge_counts.get(eid, 0) / VOL_SCALE),
            'spd':     m.get('spd',     0),
            'occ':     m.get('occ',     0),
            'density': m.get('density', 0),
            'vol':     round(m.get('vol', 0) / VOL_SCALE),
            'wait':    m.get('wait',    0),
            'tloss':   m.get('tloss',   0),
        }
    print(f'  {len(edges_out)} edges with geometry ({skipped} skipped)')

    # ── 3.5 VD 校準：以實測車速錨定 SUMO 速度 ────────────────────────────────
    # SUMO 的 edge 均速含路口停等，普遍比 VD 點速度低（約 1.5x）。用 VD 當地面真值，
    # 各幹道算 VD/SUMO 比作為校準係數，使顯示速度貼近實測（其餘 edge 套全域中位數）。
    if calibrate:
        try:
            vd_spd = _load_vd_road_speeds(vd_dir)
        except Exception as exc:
            vd_spd = {}
            print(f'  VD 校準略過（讀取失敗）: {exc}')
        if vd_spd:
            sumo_by_pref = {}
            for ed in edges_out.values():
                pref = _road_prefix(ed.get('name', ''))
                if pref and ed['spd'] > 0:
                    sumo_by_pref.setdefault(pref, []).append(ed['spd'])
            factors = {}
            for pref, sp in sumo_by_pref.items():
                avg = sum(sp) / len(sp)
                if pref in vd_spd and avg > 0:
                    factors[pref] = max(0.5, min(3.0, vd_spd[pref] / avg))
            if factors:
                vals = sorted(factors.values())
                global_f = vals[len(vals) // 2]   # 中位數，給無 VD 的 edge 用
                vd_vals = sorted(vd_spd.values())
                vd_median = vd_vals[len(vd_vals) // 2]
                # 每條 edge 收斂到該路 VD 的 [LO, HI] 倍：壓低過大的 per-edge 變異，
                # 讓地圖各 edge 速度貼近列表顯示的路段均速（解決「列表與地圖對不上」）。
                LO, HI = 0.6, 1.25
                for ed in edges_out.values():
                    if ed['spd'] <= 0:
                        continue
                    pref = _road_prefix(ed.get('name', ''))
                    f = factors.get(pref, global_f)
                    vd = vd_spd.get(pref, vd_median)
                    scaled = ed['spd'] * f
                    ed['spd'] = round(min(max(scaled, vd * LO), vd * HI), 1)
                print(f'  VD 校準: {len(factors)} 幹道，中位數×{global_f:.2f}，'
                      f'每邊收斂到 VD×[{LO},{HI}]')
            else:
                print('  VD 校準：無重疊幹道可錨定，略過')
        else:
            print('  VD 校準：無 VD 資料，略過')

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
