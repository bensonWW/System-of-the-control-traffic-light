#!/usr/bin/env python3
"""一次性抓取台北市 VD 車流資料，存至 trafficData/。"""
import os
import requests, gzip, io, json
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).parent.parent
TRAFFIC_DIR = ROOT / "TrafficVision Design System" / "data" / "trafficData"
TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)

NTUT_PREFIXES = ["忠孝東路", "八德路", "市民大道", "建國北路", "建國南路", "新生南路", "新生北路", "松江路"]


def _retrying_session():
    """Session with exponential backoff against transient Taipei API failures.

    backoff_factor=1 → sleeps 1s, 2s, 4s between attempts. status_forcelist
    covers the 5xx range plus 429 (rate limit). raise_on_status=False lets the
    caller's resp.raise_for_status() decide; allowed_methods includes GET so
    idempotent reads retry without needing POST whitelisting.
    """
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://",  adapter)
    s.mount("https://", adapter)
    return s


# Capture fetch timestamp once — used for both the file name and the payload
# so that the filename, the manifest timestamp, and the source-of-truth
# acquisition time all match.
fetch_time = datetime.now()

print("正在從台北市 VD API 抓取資料...")
resp = _retrying_session().get(
    "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz",
    timeout=30,
)
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

fname = TRAFFIC_DIR / f"traffic_{fetch_time.strftime('%Y%m%d_%H%M%S')}.json"
# Atomic write: a partial JSON would crash serve_api.py's /api/traffic reader.
# Write to a sibling .tmp first, fsync, then os.replace into place so a crash
# mid-write leaves the previous good snapshot intact.
tmp_path = fname.with_suffix(fname.suffix + ".tmp")
with open(tmp_path, "w", encoding="utf-8") as fp:
    json.dump({"timestamp": fetch_time.isoformat(), "data": data}, fp, ensure_ascii=False, indent=2)
    fp.flush()
    os.fsync(fp.fileno())
os.replace(tmp_path, fname)

print(f"已抓取 {len(data)} 個 NTUT 周邊路段")
print(f"存檔：{fname}")
for k, v in list(data.items())[:5]:
    print(f"  {k}: MOE={v['MOELevel']}, spd={v['AvgSpd']}, vol={v['TotalVol']}")
