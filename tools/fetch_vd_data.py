#!/usr/bin/env python3
"""一次性抓取台北市 VD 車流資料，存至 trafficData/。"""
import requests, gzip, io, json
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent.parent
TRAFFIC_DIR = ROOT / "TrafficVision Design System" / "data" / "trafficData"
TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)

NTUT_PREFIXES = ["忠孝東路", "八德路", "市民大道", "建國北路", "建國南路", "新生南路", "新生北路", "松江路"]

print("正在從台北市 VD API 抓取資料...")
resp = requests.get("https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz", timeout=30)
resp.raise_for_status()

with gzip.open(io.BytesIO(resp.content)) as gz:
    root = ET.fromstring(gz.read())

data = {}
for item in root[2]:
    section_name = ""
    fields = {}
    for child in item:
        tag = child.tag.split("}")[-1]
        if tag == "SectionName":
            txt = (child.text or "").strip()
            if txt and "高" not in txt and "快" not in txt:
                section_name = txt
        else:
            fields[tag] = child.text
    if not section_name:
        continue
    if not any(section_name.startswith(p) for p in NTUT_PREFIXES):
        continue
    try:
        data[section_name] = {
            "AvgSpd":    round(float(fields.get("AvgSpd") or 0), 2),
            "TotalVol":  int(float(fields.get("TotalVol") or 0)),
            "AvgOcc":    round(float(fields.get("AvgOcc") or 0), 2),
            "MOELevel":  int(fields.get("MOELevel") or 0),
            "SectionId": fields.get("SectionId", ""),
        }
    except (ValueError, TypeError):
        continue

fname = TRAFFIC_DIR / f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(fname, "w", encoding="utf-8") as fp:
    json.dump({"timestamp": datetime.now().isoformat(), "data": data}, fp, ensure_ascii=False, indent=2)

print(f"已抓取 {len(data)} 個 NTUT 周邊路段")
print(f"存檔：{fname}")
for k, v in list(data.items())[:5]:
    print(f"  {k}: MOE={v['MOELevel']}, spd={v['AvgSpd']}, vol={v['TotalVol']}")
