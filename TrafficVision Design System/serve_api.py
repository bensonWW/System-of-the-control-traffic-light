"""
serve_api.py — TrafficVision 數據橋接 API
============================================
將 runtime_pipeline.py 的輸出轉為網頁可讀取的 JSON API。

安裝依賴：
    pip install fastapi uvicorn

啟動方式：
    python serve_api.py
    # 預設在 http://localhost:8000 啟動
    # 儀表板會自動從此 API 拉取最新數據

API 端點：
    GET  /api/traffic              最新即時車流數據（來自 trafficData/*.json）
    GET  /api/prediction           最新 GRU 預測結果（來自 handoff/prediction*.csv）
    GET  /api/signals              最新號誌計畫（來自 handoff/*_signal_plan_summary.csv）
    GET  /api/comparison           最佳策略對比（來自 handoff/*_comparison_summary.csv）
    GET  /api/handoff              完整最新 handoff 資料夾摘要
    GET  /api/status               系統狀態（上次更新時間、檔案是否存在）
    POST /api/simulation/start     啟動 SUMO 即時串流（TraCI 背景執行緒）
    POST /api/simulation/stop      停止模擬
    GET  /api/simulation/status    模擬執行狀態
    WS   /ws/simulation            每 10 模擬秒推送一次路段 + 號誌資料
"""

import asyncio
import json
import os
import glob
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import RedirectResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    import pandas as pd
    import requests
    import gzip
    import io
except ImportError:
    print("請安裝依賴：pip install fastapi uvicorn[standard] pandas requests aiofiles")
    raise

# ─── 路徑設定（相對於此檔案所在目錄）───────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
TRAFFIC_DIR   = DATA_DIR / "trafficData"
RUNTIME_DIR   = BASE_DIR.parent / "data" / "runtime_data"

# ─── SUMO 資料路徑 ────────────────────────────────
SUMO_DATA_DIR     = BASE_DIR.parent / "data"
SUMO_NET_XML      = SUMO_DATA_DIR / "ntut_network_split.net.xml"
DEFAULT_SUMOCFG   = SUMO_DATA_DIR / "ntut_config.sumocfg"
EDGE_HEATMAP_FILE = BASE_DIR / "data" / "edge_heatmap.json"
EDGE_HEATMAP_BASELINE_FILE = BASE_DIR / "data" / "edge_heatmap_baseline.json"
EDGE_HEATMAP_CURRENT_FILE = BASE_DIR / "data" / "edge_heatmap_current.json"

# 路段名稱前綴 → 前端用的大分類名稱
_ROAD_PREFIX_MAP = {
    "忠孝東路": "忠孝東路",
    "八德路":   "八德路",
    "市民大道": "市民大道",
    "建國北路": "建國北路",
    "建國南路": "建國南路",
    "新生南路": "新生南路",
    "新生北路": "新生南路",  # 合併至同一前端路段
    "松江路":   "松江路",
}


_TAIPEI_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
_NTUT_ROAD_PREFIXES = list(_ROAD_PREFIX_MAP.keys())


# Module-level retrying session for the Taipei VD API.
# total=3, backoff_factor=1 → waits 1s / 2s / 4s on transient 5xx or 429.
# Built once at import; reused for every /api/traffic and /api/traffic/refresh
# call to avoid per-request socket/adapter churn.
from requests.adapters import HTTPAdapter as _HTTPAdapter
from urllib3.util.retry import Retry as _Retry

_taipei_session = requests.Session()
_taipei_session.mount(
    "https://",
    _HTTPAdapter(
        max_retries=_Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
    ),
)


# ─── JSON file cache (mtime-keyed) ─────────────────────────────────────────────
# Without this, every /api/traffic, /api/edge-heatmap*, and /api/roads/forecast
# request re-parses the same on-disk JSON. Under 100+ clients/min that's 100×
# disk reads + 100× json.loads per minute on files up to ~100KB.
# Cache invalidates as soon as the pipeline writes a new version (mtime changes),
# so freshness is preserved.
_json_cache: dict = {}
_json_cache_lock = threading.Lock()


def _load_json_cached(path):
    """Return parsed JSON for `path`, cached until the file's mtime changes.

    Returns None if the file doesn't exist. Any parse error invalidates the
    cache entry and re-raises so the caller can decide how to handle it.
    """
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except FileNotFoundError:
        return None

    key = str(p)
    with _json_cache_lock:
        entry = _json_cache.get(key)
        if entry is not None and entry[0] == mtime:
            return entry[1]

    # Read outside the lock — large JSONs would otherwise serialize all readers.
    with open(p, encoding="utf-8") as fp:
        data = json.load(fp)

    with _json_cache_lock:
        _json_cache[key] = (mtime, data)
    return data


def _to_float_or_none(v):
    """安全轉 float；None / 空字串 / 非數值一律回 None。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    return f if f == f else None  # 過濾 NaN


def _fetch_taipei_traffic() -> dict:
    """從台北市 Open Data API 抓取 VD 資料，回傳 NTUT 周邊路段字典。
    每段除了流量指標外，另附 StartLon/Lat、EndLon/Lat（WGS84）方便前端逐段繪製。"""
    resp = _taipei_session.get(_TAIPEI_VD_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as gz:
        root = __import__("xml.etree.ElementTree", fromlist=["ElementTree"]).fromstring(gz.read())
    data: dict = {}
    # Schema-drift visibility (mirrors tools/fetch_vd_data.py).
    parse_errors: list = []
    for item in root[2]:
        section_name = ""
        fields: dict = {}
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
        if not any(section_name.startswith(p) for p in _NTUT_ROAD_PREFIXES):
            continue
        try:
            data[section_name] = {
                "AvgSpd":    round(float(fields.get("AvgSpd") or 0), 2),
                "TotalVol":  int(float(fields.get("TotalVol") or 0)),
                "AvgOcc":    round(float(fields.get("AvgOcc") or 0), 2),
                "MOELevel":  int(fields.get("MOELevel") or 0),
                "SectionId": fields.get("SectionId", ""),
                "StartLon":  _to_float_or_none(fields.get("StartWgsX")),
                "StartLat":  _to_float_or_none(fields.get("StartWgsY")),
                "EndLon":    _to_float_or_none(fields.get("EndWgsX")),
                "EndLat":    _to_float_or_none(fields.get("EndWgsY")),
            }
        except (ValueError, TypeError) as exc:
            parse_errors.append((section_name, str(exc)))
            continue
    if parse_errors:
        print(f"⚠  _fetch_taipei_traffic: {len(parse_errors)} 筆 NTUT 路段解析失敗（schema 漂移?）")
        for section, reason in parse_errors[:5]:
            print(f"     - {section}: {reason}")
    return data


def _save_traffic_json(data: dict, timestamp: str) -> None:
    TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(TRAFFIC_DIR / fname, "w", encoding="utf-8") as fp:
        json.dump({"timestamp": timestamp, "data": data}, fp, ensure_ascii=False, indent=2)


def _build_edge_road_map(net_xml: Path) -> dict:
    """從 SUMO net.xml 建立 {edge_id: 大分類路名} 對應表。"""
    import xml.etree.ElementTree as ET
    result = {}
    try:
        for edge in ET.parse(str(net_xml)).getroot().findall("edge"):
            eid = edge.get("id", "")
            if eid.startswith(":"):
                continue
            name = edge.get("name", "")
            for prefix, broad in _ROAD_PREFIX_MAP.items():
                if name.startswith(prefix):
                    result[eid] = broad
                    break
    except Exception:
        pass
    return result


# ─── 模擬串流狀態（全域，執行緒共用）────────────────
# Lazy-init: _edge_road_map is populated by _lifespan() at FastAPI startup but
# also accessed by sync endpoints called pre-startup (test harness, REPL).
# _ensure_edge_road_map() guarantees it's built on first use either way.
_edge_road_map: dict = {}


def _ensure_edge_road_map():
    """Build _edge_road_map on first access if startup hasn't populated it yet."""
    global _edge_road_map
    if not _edge_road_map and SUMO_NET_XML.exists():
        _edge_road_map = _build_edge_road_map(SUMO_NET_XML)
    return _edge_road_map

_sim_clients: set = set()
_sim_snapshot: dict = {}
_sim_running  = threading.Event()
_sim_thread: Optional[threading.Thread] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None

async def _periodic_traffic_refresh(interval: int = 300):
    """每 interval 秒從台北 API 抓一次資料並存檔。"""
    while True:
        await asyncio.sleep(interval)
        try:
            data = await asyncio.get_running_loop().run_in_executor(None, _fetch_taipei_traffic)
            _save_traffic_json(data, datetime.now().isoformat())
            print(f"[traffic] 已更新 {len(data)} 路段")
        except Exception as exc:
            print(f"[traffic] 自動更新失敗: {exc}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _edge_road_map, _event_loop
    _event_loop = asyncio.get_running_loop()
    if SUMO_NET_XML.exists():
        _edge_road_map = _build_edge_road_map(SUMO_NET_XML)

    # 啟動時若無本地快取檔，立即抓一次
    if not latest_file(str(TRAFFIC_DIR / "*.json")):
        try:
            data = await asyncio.get_running_loop().run_in_executor(None, _fetch_taipei_traffic)
            _save_traffic_json(data, datetime.now().isoformat())
            print(f"[traffic] 初始化完成，已載入 {len(data)} 路段")
        except Exception as exc:
            print(f"[traffic] 初始化失敗: {exc}")

    refresh_task = asyncio.create_task(_periodic_traffic_refresh(300))
    yield
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="TrafficVision API", version="1.0", lifespan=_lifespan)

# ─── 靜態文件 ─────────────────────────────────────────────────
# 只掛載前端資源目錄；資料一律經 /api/* 端點提供，避免整個 data/ 目錄(含原始 VD 快取)外洩。
app.mount("/ui_kits", StaticFiles(directory=str(BASE_DIR / "ui_kits")), name="ui_kits")


@app.get("/", include_in_schema=False)
def root():
    """重定向到儀表板首頁。"""
    return RedirectResponse("/ui_kits/traffic-dashboard/dashboard.html")


async def _broadcast(data: dict):
    """廣播給所有已連線的 WebSocket 客戶端。"""
    dead: set = set()
    for ws in list(_sim_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _sim_clients.difference_update(dead)


def _traci_worker(cfg_path: str, loop: asyncio.AbstractEventLoop) -> None:
    """在背景執行緒中執行 SUMO TraCI，每 10 模擬秒推送一筆資料。"""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "error", "message": "環境變數 SUMO_HOME 未設定"}), loop
        )
        _sim_running.clear()
        return

    sys.path.append(os.path.join(sumo_home, "tools"))
    try:
        import traci       # type: ignore
        import sumolib     # type: ignore
    except ImportError as exc:
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "error", "message": f"找不到 traci/sumolib: {exc}"}), loop
        )
        _sim_running.clear()
        return

    cmd = [
        sumolib.checkBinary("sumo"),
        "-c", cfg_path,
        "--start", "--quit-on-end", "--no-warnings", "--no-step-log",
    ]
    try:
        traci.start(cmd)
        last_broadcast_t = -999.0
        MAX_SIM_SECONDS = 7200
        import time as _time
        # 每條 edge 在廣播視窗內的累積車輛通過次數（每模擬步驟加總，比單步快照更能反映車流）
        _edge_window_acc: dict = {}

        while _sim_running.is_set():
            traci.simulationStep()
            t = traci.simulation.getTime()

            if traci.simulation.getMinExpectedNumber() == 0 and t > 30.0:
                break
            if t > MAX_SIM_SECONDS:
                break

            # 每步累積 edge 計數（不論是否到廣播時間）
            for eid in traci.edge.getIDList():
                if not eid.startswith(":"):
                    cnt = traci.edge.getLastStepVehicleNumber(eid)
                    if cnt:
                        _edge_window_acc[eid] = _edge_window_acc.get(eid, 0) + cnt

            if t - last_broadcast_t < 10.0:
                continue

            _time.sleep(0.5)  # 每 10 模擬秒暫停 0.5 秒，讓前端有時間顯示
            last_broadcast_t = t

            # ── 路段資料彙總（按大分類路名平均）
            road_acc: dict = {}
            for eid in traci.edge.getIDList():
                if eid.startswith(":"):
                    continue
                road_name = _edge_road_map.get(eid)
                if not road_name:
                    continue
                spd = traci.edge.getLastStepMeanSpeed(eid) * 3.6  # m/s → km/h
                cnt = traci.edge.getLastStepVehicleNumber(eid)
                occ = traci.edge.getLastStepOccupancy(eid)
                if road_name not in road_acc:
                    road_acc[road_name] = {"speeds": [], "count": 0, "occ": []}
                road_acc[road_name]["speeds"].append(spd)
                road_acc[road_name]["count"] += cnt
                road_acc[road_name]["occ"].append(occ)

            roads_out: dict = {}
            for name, d in road_acc.items():
                avg_spd = sum(d["speeds"]) / len(d["speeds"]) if d["speeds"] else 0.0
                avg_occ = sum(d["occ"])    / len(d["occ"])    if d["occ"]    else 0.0
                roads_out[name] = {
                    "speed_kmh": round(avg_spd, 1),
                    "count":     d["count"],
                    "occupancy": round(avg_occ, 2),
                    "moe": 0 if avg_spd >= 40 else (1 if avg_spd >= 20 else 2),
                }

            # ── 每條 edge：10 秒視窗累積車次 + 當前速度（供前端熱力圖使用）
            edges_out: dict = {}
            for eid in traci.edge.getIDList():
                if eid.startswith(":"):
                    continue
                spd = traci.edge.getLastStepMeanSpeed(eid) * 3.6
                edges_out[eid] = {
                    "count":     _edge_window_acc.get(eid, 0),
                    "speed_kmh": round(spd, 1),
                }
            _edge_window_acc.clear()  # 重置，準備下一個視窗

            # ── 號誌資料
            signals_out: dict = {}
            for tl_id in traci.trafficlight.getIDList():
                state      = traci.trafficlight.getRedYellowGreenState(tl_id)
                phase      = traci.trafficlight.getPhase(tl_id)
                next_sw    = traci.trafficlight.getNextSwitch(tl_id) - t
                signals_out[tl_id] = {
                    "state":       state,
                    "phase":       phase,
                    "next_switch": round(next_sw, 1),
                }

            payload = {"type": "step", "sim_time": t, "roads": roads_out, "signals": signals_out, "edges": edges_out}
            _sim_snapshot.update(payload)
            asyncio.run_coroutine_threadsafe(_broadcast(payload), loop)

    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "error", "message": str(exc)}), loop
        )
    finally:
        try:
            traci.close()
        except Exception:
            pass
        _sim_running.clear()
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "end", "sim_time": _sim_snapshot.get("sim_time", 0)}),
            loop,
        )

# CORS 已移除：前端由本服務同源送出（/ui_kits），不需要跨來源中介層。
# 若日後有跨來源需求，請在此加回 CORSMiddleware 並限制 allow_origins。

# ─── 工具函數 ─────────────────────────────────────
def latest_file(pattern: str):
    """找到符合 glob pattern 中最新的檔案"""
    files = sorted(glob.glob(str(pattern)))
    return files[-1] if files else None

def latest_handoff_dir():
    """找到最新的 handoff 目錄"""
    pattern = str(RUNTIME_DIR / "*" / "handoff")
    dirs = sorted(glob.glob(pattern))
    return Path(dirs[-1]) if dirs else None

def csv_to_records(path: str):
    """CSV → list of dicts（JSON-serializable）"""
    try:
        df = pd.read_csv(path)
        return df.where(df.notna(), None).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


# ─── 端點 ─────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """系統狀態：最新數據時間、各輸出是否存在"""
    latest_traffic = latest_file(str(TRAFFIC_DIR / "*.json"))
    handoff = latest_handoff_dir()
    return {
        "status": "ok",
        "server_time": datetime.now().isoformat(),
        "latest_traffic_file": latest_traffic,
        "latest_handoff_dir": str(handoff) if handoff else None,
        "traffic_data_exists": latest_traffic is not None,
        "handoff_exists": handoff is not None,
    }


@app.get("/api/health")
def get_health():
    """深度健康檢查 — 給 dashboard / ops 用。

    回傳每個資料源的「上次更新何時、距現在幾秒」：
      vd            ← TrafficVision Design System/data/trafficData/*.json
      predict       ← runtime_pipeline 的 *_predict.csv
      handoff       ← 整個 handoff/ 目錄
      pipeline_lock ← runtime_pipeline scheduler 是否在跑
      ollama        ← 本機 Ollama 是否可連線
    """
    now = datetime.now()
    out = {"server_time": now.isoformat(timespec="seconds")}

    # ── VD freshness (Taipei API 每 5 分鐘自動抓)
    vd_file = latest_file(str(TRAFFIC_DIR / "*.json"))
    if vd_file:
        age = (now - datetime.fromtimestamp(os.path.getmtime(vd_file))).total_seconds()
        out["vd"] = {
            "file": os.path.basename(vd_file),
            "age_sec": int(age),
            "stale": age > 600,  # > 10 分鐘算 stale
        }
    else:
        out["vd"] = {"file": None, "age_sec": None, "stale": True}

    # ── Predict freshness (runtime_pipeline 跑完才更新)
    predict_csv = _latest_predict_csv()
    if predict_csv and predict_csv.exists():
        age = (now - datetime.fromtimestamp(predict_csv.stat().st_mtime)).total_seconds()
        out["predict"] = {
            "file": predict_csv.name,
            "age_sec": int(age),
            "stale": age > 600,
        }
    else:
        out["predict"] = {"file": None, "age_sec": None, "stale": True}

    # ── Scheduler liveness (lockfile + last metric)
    lock_path = BASE_DIR.parent / "data" / "runtime_data" / ".pipeline.lock"
    last_metric = _latest_run_metric() or {}
    pipeline_age = None
    if last_metric.get("ts"):
        try:
            pipeline_age = (now - datetime.fromisoformat(last_metric["ts"])).total_seconds()
        except Exception:
            pass
    out["pipeline"] = {
        "lock_exists": lock_path.exists(),
        "last_run_ts": last_metric.get("ts"),
        "last_run_age_sec": int(pipeline_age) if pipeline_age else None,
        "last_run_success": last_metric.get("success"),
        "last_run_step3_skipped": last_metric.get("step3_skipped"),
        "last_strategy": last_metric.get("strategy"),
        "last_composite_score": last_metric.get("composite_score"),
        # Scheduler likely alive if a run happened within last 10 min (default interval 5 min)
        "scheduler_alive_guess": pipeline_age is not None and pipeline_age < 600,
    }

    # ── Ollama reachability (best-effort, < 1s timeout)
    try:
        r = _taipei_session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.0)
        out["ollama"] = {
            "reachable": r.ok,
            "url": OLLAMA_BASE_URL,
            "model_required": OLLAMA_MODEL,
        }
    except Exception as exc:
        out["ollama"] = {
            "reachable": False,
            "url": OLLAMA_BASE_URL,
            "error": str(exc)[:120],
        }

    # ── Overall status: ok / degraded / critical
    if out["vd"]["stale"] or out["pipeline"]["last_run_age_sec"] is None:
        out["status"] = "critical"
    elif out["predict"]["stale"] or not out["pipeline"]["scheduler_alive_guess"]:
        out["status"] = "degraded"
    else:
        out["status"] = "ok"
    return out


@app.get("/api/traffic")
def get_latest_traffic(response: Response):
    """
    最新即時車流數據。
    格式：{ timestamp, data: { "路段名稱": { AvgSpd, TotalVol, AvgOcc, MOELevel, ... } } }
    優先讀本地快取（5 分鐘內，對齊 VD 更新週期），過期或不存在則直接打台北 API。
    """
    # 阻擋瀏覽器 / 中介層 cache — VD 資料 5 分鐘變動，cache 會讓 dashboard 顯示舊值。
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    raw_data: dict = {}
    timestamp: Optional[str] = None
    source = "unknown"

    f = latest_file(str(TRAFFIC_DIR / "*.json"))
    if f:
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(f))).total_seconds()
        # 台北 VD API 約 5 分鐘更新一次，本地快取守 < 5 分鐘才當作 fresh。
        # 過去設 < 10 分鐘會讓 dashboard 拿到 5-9 分鐘前的舊資料，看起來像「不更新」。
        if age < 300:
            raw = _load_json_cached(f) or {}
            raw_data = raw.get("data", {})
            timestamp = raw.get("timestamp")
            source = os.path.basename(f)

    if not raw_data:
        try:
            raw_data = _fetch_taipei_traffic()
            timestamp = datetime.now().isoformat()
            source = "live_api"
            _save_traffic_json(raw_data, timestamp)
        except Exception as exc:
            return {"error": f"資料擷取失敗: {exc}", "data": {}}

    data = {}
    for road, v in raw_data.items():
        data[road] = {
            "AvgSpd":    round(float(v.get("AvgSpd") or 0), 2),
            "TotalVol":  int(float(v.get("TotalVol") or 0)),
            "AvgOcc":    round(float(v.get("AvgOcc") or 0), 2),
            "MOELevel":  int(v.get("MOELevel") or 0),
            "SectionId": v.get("SectionId", ""),
            # 舊版快取沒有座標欄位 → 回 None，前端會自動跳過該段不繪線。
            "StartLon":  v.get("StartLon"),
            "StartLat":  v.get("StartLat"),
            "EndLon":    v.get("EndLon"),
            "EndLat":    v.get("EndLat"),
        }
    return {"timestamp": timestamp, "source_file": source, "road_count": len(data), "data": data}


@app.post("/api/traffic/refresh")
async def refresh_traffic():
    """手動觸發從台北 API 立即抓取最新車流數據並存檔。"""
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, _fetch_taipei_traffic)
        ts = datetime.now().isoformat()
        _save_traffic_json(data, ts)
        return {"status": "ok", "road_count": len(data), "timestamp": ts}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ─── Run history helpers (read tools/runtime_pipeline.py's _metrics.jsonl) ──
_METRICS_PATH = BASE_DIR.parent / "data" / "runtime_data" / "_metrics.jsonl"


def _latest_run_metric():
    """Return the most recent JSONL entry from the pipeline metrics log,
    or None if the log doesn't exist or is empty."""
    if not _METRICS_PATH.exists():
        return None
    try:
        with open(_METRICS_PATH, "rb") as fp:
            # tail-read last 16KB so we don't load the whole history
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - 16384))
            tail = fp.read().decode("utf-8", errors="replace")
        last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
        if not last_line:
            return None
        return json.loads(last_line)
    except Exception:
        return None


def _diagnose_missing_handoff():
    """Compose a friendlier error payload when handoff is absent.

    Inspects the metrics log to tell the user *why* nothing's there:
    pipeline hasn't run yet, last run failed, or last run skipped Step 3
    due to data shortage."""
    last = _latest_run_metric()
    if last is None:
        return {
            "error": "pipeline_never_ran",
            "message": "尚無任何 pipeline 執行紀錄。請執行 python tools/runtime_pipeline.py --once",
            "hint": "通常每 5 分鐘自動跑一次；如未啟用 scheduler，請先手動跑一輪。",
        }
    if last.get("error"):
        return {
            "error": "last_run_failed",
            "message": f"上一次 pipeline 在 {last.get('ts')} 失敗：{last.get('error')}",
            "hint": "看 serve_api / scheduler log 取得完整 traceback。",
            "duration_sec": last.get("duration_sec"),
        }
    if last.get("step3_skipped"):
        return {
            "error": "step3_skipped",
            "message": f"上一次 pipeline 於 {last.get('ts')} 跳過 Step 3（預測 + 號誌優化）。",
            "reason": last.get("step3_skipped"),
            "hint": "常見原因：低車流時段 SUMO 模擬太短，無法產生 15 個時間步給 pair model。",
            "duration_sec": last.get("duration_sec"),
        }
    return {
        "error": "handoff_not_found",
        "message": f"上一次 pipeline 於 {last.get('ts')} 成功，但找不到對應的 handoff 目錄。",
        "hint": f"預期位置：{last.get('run_dir')}/handoff/。可能被外部刪掉了。",
        "last_strategy": last.get("strategy"),
        "duration_sec": last.get("duration_sec"),
    }


@app.get("/api/prediction")
def get_prediction():
    """GRU 預測結果 CSV → JSON"""
    handoff = latest_handoff_dir()
    if not handoff:
        return _diagnose_missing_handoff()
    f = latest_file(str(handoff / "*_predict.csv"))
    if not f:
        return {"error": "找不到 prediction CSV", "handoff_dir": str(handoff),
                "hint": "Step 3 可能被跳過或失敗；查 /api/status 或 _metrics.jsonl。"}
    return {"source_file": os.path.basename(f), "records": csv_to_records(f)}


@app.get("/api/signals")
def get_signal_plan():
    """號誌計畫摘要"""
    handoff = latest_handoff_dir()
    if not handoff:
        return _diagnose_missing_handoff()
    f = latest_file(str(handoff / "*signal_plan_summary*.csv"))
    if not f:
        return {"error": "找不到 signal plan CSV", "handoff_dir": str(handoff),
                "hint": "如果 best strategy 是 no_control，signal_plan_summary 會是空的（無號誌調整）。"}
    return {"source_file": os.path.basename(f), "records": csv_to_records(f)}


@app.get("/api/comparison")
def get_comparison():
    """前後對比：baseline vs 最佳策略的 8 個指標"""
    handoff = latest_handoff_dir()
    if not handoff:
        return _diagnose_missing_handoff()
    f = latest_file(str(handoff / "*comparison_summary*.csv"))
    if not f:
        return {"error": "找不到 comparison CSV", "handoff_dir": str(handoff)}
    records = csv_to_records(f)
    # 整理成 { metric: { before, after, delta } } 格式
    formatted = {}
    for r in (records if isinstance(records, list) else []):
        formatted[r.get("metric", "?")] = {
            "before": r.get("before"),
            "after":  r.get("after"),
            "delta":  r.get("delta"),
        }
    return {"source_file": os.path.basename(f), "metrics": formatted}


@app.get("/api/handoff")
def get_full_handoff():
    """完整 handoff 目錄摘要（所有可用 CSV）"""
    handoff = latest_handoff_dir()
    if not handoff:
        return _diagnose_missing_handoff()
    result = {}
    for csv_file in sorted(handoff.glob("*.csv")):
        key = csv_file.stem
        result[key] = csv_to_records(str(csv_file))
    manifest = handoff / "handoff_manifest.csv"
    return {
        "handoff_dir": str(handoff),
        "run_time": handoff.parent.name,
        "files": result,
    }


@app.get("/api/handoff/download")
def download_handoff_file(name: str):
    """下載最新 handoff 目錄內的單一檔案（含目錄穿越防護）。"""
    handoff = latest_handoff_dir()
    if not handoff:
        return {"error": "找不到 handoff 目錄"}
    safe = Path(name).name                       # 去除任何路徑成分，防目錄穿越
    target = (handoff / safe).resolve()
    if target.parent != handoff.resolve() or not target.exists():
        return {"error": "檔案不存在"}
    return FileResponse(str(target), filename=safe, media_type="application/octet-stream")


@app.get("/api/edge-heatmap")
def get_edge_heatmap():
    """
    邊道熱力圖 JSON（由 generate_edge_traffic.py / runtime_pipeline.py 產生）。
    格式：{ meta: {...}, edges: { edge_id: { name, shape, count, spd, occ, density, vol, wait, tloss } } }
    """
    data = _load_json_cached(EDGE_HEATMAP_FILE)
    if data is None:
        return {
            "error": "edge_heatmap.json 尚未生成",
            "hint": "請先執行 runtime_pipeline.py 或 python tools/generate_edge_traffic.py",
        }
    return data


@app.get("/api/edge-heatmap/baseline")
def get_edge_heatmap_baseline():
    """基準（no_control）邊道熱力圖 — 未優化的 5 分鐘預測全路網車況，供「5分鐘預測」使用。"""
    data = _load_json_cached(EDGE_HEATMAP_BASELINE_FILE)
    if data is None:
        return {
            "error": "edge_heatmap_baseline.json 尚未生成",
            "hint": "請先執行 runtime_pipeline.py（會同時產生基準熱力圖）",
        }
    return data


@app.get("/api/edge-heatmap/current")
def get_edge_heatmap_current():
    """當前需求（Step 2）邊道熱力圖 — 當前車流的全路網模擬，供「當前車流」使用。"""
    data = _load_json_cached(EDGE_HEATMAP_CURRENT_FILE)
    if data is None:
        return {
            "error": "edge_heatmap_current.json 尚未生成",
            "hint": "請先執行 runtime_pipeline.py（會同時產生當前熱力圖）",
        }
    return data


# ─── GRU prediction CSV aggregation ─────────────────────────────────────────
# 之前 /api/roads/forecast.pred 拿的是 SUMO no_control 模擬輸出（edge_heatmap_baseline.json），
# 完全沒接到 GRU 預測 CSV — 命名「預測車流」但實際是「模擬車流」，name 不副實。
# 這層 helper 從最近 handoff 的 *_predict.csv 讀出 pair model 的 15 步 × 20 秒
# 預測，aggregate 到 road 級 / edge 級，供 forecast endpoints 真正用到 GRU。

def _latest_predict_csv() -> Optional[Path]:
    """Find the canonical 15-step GRU prediction CSV in the latest handoff dir.
    Excludes *_predict_full.csv (the unfiltered full-horizon variant)."""
    handoff = latest_handoff_dir()
    if not handoff:
        return None
    files = [f for f in handoff.glob("*_predict.csv") if "_predict_full" not in f.name]
    return sorted(files)[-1] if files else None


def _gru_predict_by_edge() -> dict:
    """Sum GRU predicted vehicle_count across all 15 timesteps per edge.

    The pair model emits one row per (time, edge_id) over a 5-min horizon
    (15 × 20 s). Summing across time gives "predicted total vehicle-snapshots
    seen on this edge during the next 5 min" — a useful magnitude that scales
    intuitively with how busy the edge is expected to be.

    Returns {edge_id: float}. Cached against the CSV's mtime via
    _load_json_cached's mechanism reused — but pandas needs a separate path,
    so we just re-read each request (the file is small ~100 KB).
    """
    csv_path = _latest_predict_csv()
    if csv_path is None:
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "vehicle_count" not in df.columns or "edge_id" not in df.columns:
            return {}
        return df.groupby("edge_id")["vehicle_count"].sum().to_dict()
    except Exception as exc:
        print(f"  ⚠ _gru_predict_by_edge: 解析 predict CSV 失敗 ({exc})")
        return {}


def _spd_by_edge(heatmap_path: Path) -> dict:
    """Return {edge_id: spd} from a SUMO edge heatmap JSON. Used to build a
    fallback map for edges where the primary source (baseline) has spd=0
    because that strategy's SUMO sim didn't route vehicles through them."""
    data = _load_json_cached(heatmap_path)
    if data is None:
        return {}
    out = {}
    for eid, ed in (data.get("edges") or {}).items():
        try:
            s = float(ed.get("spd") or 0)
        except (TypeError, ValueError):
            s = 0.0
        if s > 0:
            out[eid] = round(s, 1)
    return out


def _gru_predict_by_road() -> dict:
    """Aggregate edge-level GRU predictions to broad road names via _edge_road_map."""
    by_edge = _gru_predict_by_edge()
    road_map = _ensure_edge_road_map()
    by_road: dict = {}
    for eid, total in by_edge.items():
        road = road_map.get(eid)
        if not road:
            continue
        by_road[road] = by_road.get(road, 0.0) + float(total)
    return {r: round(v, 1) for r, v in by_road.items()}


def _aggregate_edges_to_roads(heatmap_file: Path) -> dict:
    """把 edge 級熱力圖依 _edge_road_map 聚合成 { 大分類路名: {spd, vol, occ} }。

    spd/occ 取有資料 edge 的平均，vol 取總和（無 vol 時退回 count）。
    """
    try:
        data = _load_json_cached(heatmap_file)
    except Exception:
        return {}
    if data is None:
        return {}
    edges = data.get("edges", {})

    acc: dict = {}
    for eid, ed in edges.items():
        road = _edge_road_map.get(eid)
        if not road:
            continue
        bucket = acc.setdefault(road, {"spds": [], "occs": [], "vol": 0.0})
        spd = ed.get("spd", 0) or 0
        occ = ed.get("occ", 0) or 0
        vol = ed.get("vol", 0) or ed.get("count", 0) or 0
        if spd > 0:
            bucket["spds"].append(spd)
        if occ > 0:
            bucket["occs"].append(occ)
        bucket["vol"] += vol

    out: dict = {}
    for road, b in acc.items():
        out[road] = {
            "spd": round(sum(b["spds"]) / len(b["spds"]), 1) if b["spds"] else 0.0,
            "occ": round(sum(b["occs"]) / len(b["occs"]), 2) if b["occs"] else 0.0,
            "vol": int(round(b["vol"])),
        }
    return out


@app.get("/api/roads/forecast")
def get_road_forecast():
    """路段級「5 分鐘預測」/ 優化後車況。

    pred 欄位來源（修正前所有都是 SUMO，前端名稱「預測車流」實際與 GRU 無關）：
        vol      ← GRU pair model 預測（最近 handoff 的 *_predict.csv，
                   15 步 × 20 秒 snapshot 合計）— 真正的「預測」
        spd, occ ← SUMO no_control 基準模擬（GRU 不輸出這兩個物理量）
        vol_sumo ← 舊版的 SUMO baseline 流量，保留方便對比

    opt 欄位（best strategy 已套用後的模擬輸出，仍是 SUMO，因為 GRU 預測
    輸入給 traffic_light_optimizer 後產生的是優化動作，最終效果仍要由
    SUMO 重新模擬）：
        vol, spd, occ ← SUMO best-strategy 模擬

    meta 欄位明示資料來源，避免再被命名誤導。
    """
    sumo_baseline = _aggregate_edges_to_roads(EDGE_HEATMAP_BASELINE_FILE)
    sumo_best     = _aggregate_edges_to_roads(EDGE_HEATMAP_FILE)
    gru_by_road   = _gru_predict_by_road()

    # pred.vol is the GRU model's raw output unit: sum of vehicle_count across
    # 15 × 20-sec snapshots per edge, then aggregated by road. This is NOT the
    # same physical quantity as VD's TotalVol (5-min pass-through flow) — by
    # Little's Law the two differ by a factor of ~dwell_time/window ≈ 1/6.
    # We deliberately do NOT scale: a multiplier would invent absolute magnitude
    # the model can't actually predict. The meta block tells consumers the unit.
    pred_merged: dict = {}
    for road in set(sumo_baseline) | set(gru_by_road):
        sumo_row = sumo_baseline.get(road, {})
        gru_raw = gru_by_road.get(road, 0)
        pred_merged[road] = {
            "vol": int(round(gru_raw)),
            "spd": sumo_row.get("spd", 0.0),
            "occ": sumo_row.get("occ", 0.0),
        }

    # Opt scenario: physically, signal optimization redistributes timing — it
    # doesn't make vehicles appear or disappear. So opt.vol = pred.vol; the
    # optimization shows up in spd/occ (faster speed, lower occupancy), not vol.
    opt_merged: dict = {}
    for road in set(pred_merged) | set(sumo_best):
        pred_row = pred_merged.get(road, {})
        sumo_row = sumo_best.get(road, {})
        opt_merged[road] = {
            "vol": pred_row.get("vol", 0),  # vehicles preserved
            "spd": sumo_row.get("spd", pred_row.get("spd", 0.0)),
            "occ": sumo_row.get("occ", pred_row.get("occ", 0.0)),
        }

    return {
        "pred": pred_merged,
        "opt":  opt_merged,
        "meta": {
            "pred_vol_unit":         "vehicle-snapshots (GRU pair model: Σ vehicle_count over 15 × 20-sec timesteps per edge)",
            "pred_spd_occ_source":   "SUMO no_control 基準模擬",
            "opt_vol_source":        "= pred.vol（優化改變流速，不改變車輛數）",
            "opt_spd_occ_source":    "SUMO best-strategy 模擬（顯示優化後的速度/佔有率）",
            "vs_vd_note":            "NOT directly comparable to VD TotalVol (which is 5-min pass-through flow). Ratio ~1/6 by Little's Law (dwell_time/window).",
            "predict_csv":           str(_latest_predict_csv() or ""),
        },
    }


def _moe_from_spd(spd) -> Optional[int]:
    """Mirror dashboard.html `_moeFromSpd`: <10→2, <30→1, else 0 (None if spd unknown)."""
    if spd is None or spd <= 0:
        return None
    if spd < 10:
        return 2
    if spd < 30:
        return 1
    return 0


def _flatten_heatmap_for_monitor(path: Path) -> list:
    """Edge-level forecast list shaped for the dashboard monitor table.

    Returns one row per named edge so pred/opt mode can show edge-grained
    detail instead of the 7-prefix aggregation in /api/roads/forecast.
    Unnamed edges (internal SUMO link IDs) are skipped — they have no
    user-meaningful label to display in the table.
    """
    data = _load_json_cached(path)
    if data is None:
        return []
    out = []
    for eid, ed in (data.get("edges") or {}).items():
        name = (ed.get("name") or "").strip()
        if not name:
            continue
        spd_raw = ed.get("spd", 0) or 0
        try:
            spd = float(spd_raw)
        except (TypeError, ValueError):
            spd = 0.0
        try:
            occ = float(ed.get("occ") or 0)
        except (TypeError, ValueError):
            occ = 0.0
        try:
            vol = float(ed.get("vol") or ed.get("count") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        out.append({
            "id":   eid,
            "name": name,
            "spd":  round(spd, 1),
            "vol":  int(round(vol)),
            "occ":  round(occ, 2),
            "moe":  _moe_from_spd(spd),
        })
    return out


@app.get("/api/edges/forecast")
def get_edge_forecast():
    """Edge 級「5 分鐘預測」/ 優化 — 細到 SUMO 每條 named edge。

    pred 欄位（同 /api/roads/forecast 的修正）：
        vol      ← GRU pair model 預測（per-edge 15 步合計）
        spd, occ ← SUMO no_control 基準模擬
    opt 欄位：SUMO best-strategy 模擬。

    前端 pred / opt 模式的監控列表用這個 endpoint 直接顯示 ~130 條 named edges。
    """
    # SUMO baseline gives physics (spd/occ) + the named-edge structure we need
    sumo_baseline_edges = _flatten_heatmap_for_monitor(EDGE_HEATMAP_BASELINE_FILE)
    gru_by_edge = _gru_predict_by_edge()

    # spd fallback chain for edges where the baseline strategy's SUMO sim
    # didn't route vehicles (spd=0 = "no measurement", not "stopped"):
    #   primary: edgedata_baseline.xml (no_control strategy sim)
    #   fallback 1: edgedata_current.xml (Step 2 current-demand sim)
    #   final: null  (so dashboard renders "---" instead of fake "0 km/h")
    spd_fallback_current = _spd_by_edge(EDGE_HEATMAP_CURRENT_FILE)

    def _resolved_spd(eid, primary_spd):
        if primary_spd and primary_spd > 0:
            return primary_spd
        fb = spd_fallback_current.get(eid)
        return fb if fb and fb > 0 else None

    # pred.vol is the GRU's raw per-edge 15-step snapshot sum — the model's
    # native output. Deliberately NOT scaled to match VD TotalVol: by Little's
    # Law the two quantities legitimately differ by ~dwell_time/window (~1/6).
    pred_edges = []
    for row in sumo_baseline_edges:
        eid = row.get("id")
        gru_vol = gru_by_edge.get(eid)
        resolved_spd = _resolved_spd(eid, row.get("spd"))
        # MOE only meaningful when we have a real speed
        moe = _moe_from_spd(resolved_spd) if resolved_spd is not None else None
        base = {**row, "spd": resolved_spd, "moe": moe, "vol_sumo": row.get("vol", 0)}
        if gru_vol is not None:
            base["vol"] = int(round(float(gru_vol)))
        pred_edges.append(base)

    # Opt edges: vol = pred.vol (vehicles preserved), spd/occ from SUMO best
    # with same fallback chain so the table doesn't show fake 0 km/h.
    pred_by_id = {r["id"]: r for r in pred_edges}
    opt_edges = []
    for row in _flatten_heatmap_for_monitor(EDGE_HEATMAP_FILE):
        eid = row.get("id")
        pred_row = pred_by_id.get(eid, {})
        resolved_spd = _resolved_spd(eid, row.get("spd"))
        moe = _moe_from_spd(resolved_spd) if resolved_spd is not None else None
        opt_edges.append({
            **row,
            "spd":      resolved_spd,
            "moe":      moe,
            "vol":      pred_row.get("vol", row.get("vol", 0)),  # carry pred.vol forward
            "vol_sumo": row.get("vol", 0),
        })

    return {
        "pred": pred_edges,
        "opt":  opt_edges,
        "meta": {
            "pred_vol_unit":       "vehicle-snapshots (GRU pair model: Σ vehicle_count over 15 × 20-sec timesteps per edge)",
            "pred_spd_occ_source": "SUMO no_control 基準模擬（spd=0 時 fallback 取 current sim，仍 0 則回傳 null）",
            "opt_vol_source":      "= pred.vol（優化改變流速，不改變車輛數）",
            "opt_spd_occ_source":  "SUMO best-strategy 模擬（同 spd fallback chain）",
            "vs_vd_note":          "NOT directly comparable to VD TotalVol (5-min pass-through flow). Ratio ~1/6 by Little's Law.",
            "predict_csv":         str(_latest_predict_csv() or ""),
            "gru_edges_total":     len(gru_by_edge),
        },
    }


# ─── LLM Chat（本地 Ollama Gemma 4）──────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",    "trafficvision-gemma4")

class ChatRequest(BaseModel):
    message: str
    history: list = []   # [{"role": "user"|"assistant", "content": "..."}]

def _compact(obj, limit: int = 2000) -> str:
    """提取 data/records/metrics 欄位並序列化，截短至 limit 字元。"""
    if not isinstance(obj, dict):
        return str(obj)[:500]
    inner = obj.get("data") or obj.get("records") or obj.get("metrics") or obj
    s = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
    return s[:limit] + "…" if len(s) > limit else s

def _top_congested_edges(n: int = 10) -> str:
    """從 edge_heatmap.json 取佔有率最高的前 n 條 edge，序列化為簡潔 JSON。"""
    try:
        heatmap = _load_json_cached(EDGE_HEATMAP_FILE)
    except Exception:
        heatmap = None
    if heatmap is None:
        return "（尚無 SUMO edge 數據）"
    try:
        edges = heatmap.get("edges", {})
        top = sorted(
            ((eid, ed) for eid, ed in edges.items() if ed.get("occ", 0) > 0 or ed.get("spd", 0) > 0),
            key=lambda x: x[1].get("occ", 0),
            reverse=True,
        )[:n]
        summary = {
            eid: {
                "name":    ed.get("name", ""),
                "spd":     ed.get("spd", 0),     # km/h
                "occ":     ed.get("occ", 0),     # %
                "vol":     ed.get("vol", 0),     # 輛
                "wait":    ed.get("wait", 0),    # s
            }
            for eid, ed in top
        }
        s = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        return s[:1500] + "…" if len(s) > 1500 else s
    except Exception:
        return "（edge 數據讀取失敗）"

_MOE_LABEL = {0: "暢通", 1: "緩行", 2: "壅塞"}

def _precompute_summary(road_data: dict) -> str:
    """Pre-compute all key answers server-side so the model only needs to rephrase."""
    if not road_data:
        return "目前無即時車流資料，VD API 尚未回應。"

    sorted_roads = sorted(road_data.items(), key=lambda x: -int(x[1].get("MOELevel", 0)))
    congested  = [(r, v) for r, v in sorted_roads if int(v.get("MOELevel", 0)) == 2]
    slow       = [(r, v) for r, v in sorted_roads if int(v.get("MOELevel", 0)) == 1]
    clear      = [(r, v) for r, v in sorted_roads if int(v.get("MOELevel", 0)) == 0]

    lines = []

    # Overall status
    lines.append(
        f"整體概況：共監測 {len(road_data)} 個路段，"
        f"壅塞 {len(congested)} 個、緩行 {len(slow)} 個、暢通 {len(clear)} 個。"
    )

    # Most congested
    if congested:
        worst_road, worst_v = congested[0]
        lines.append(
            f"[壅塞] 最嚴重路段：{worst_road}，"
            f"車速 {worst_v.get('AvgSpd', 0):.1f} km/h，"
            f"佔有率 {worst_v.get('AvgOcc', 0):.1f}%，"
            f"車流量 {worst_v.get('TotalVol', 0)} 輛（MOE 2）。"
        )
        for r, v in congested[1:]:
            lines.append(
                f"[壅塞] {r}：車速 {v.get('AvgSpd', 0):.1f} km/h，"
                f"佔有率 {v.get('AvgOcc', 0):.1f}%（MOE 2）。"
            )
    else:
        lines.append("監測範圍內目前無壅塞路段（無 MOE 2）。")

    if slow:
        lines.append(f"[緩行] 共 {len(slow)} 條路段（MOE 1）：")
        for r, v in slow:
            lines.append(
                f"  {r}：車速 {v.get('AvgSpd', 0):.1f} km/h，"
                f"佔有率 {v.get('AvgOcc', 0):.1f}%，車流量 {v.get('TotalVol', 0)} 輛。"
            )
    else:
        lines.append("監測範圍內目前無緩行路段（無 MOE 1）。")

    # Per-road detail
    lines.append("\n各路段即時數據：")
    for road, v in sorted_roads:
        moe = int(v.get("MOELevel", 0))
        lines.append(
            f"  {road}：{_MOE_LABEL.get(moe, '未知')}（MOE {moe}）"
            f"｜車速 {v.get('AvgSpd', 0):.1f} km/h"
            f"｜車流量 {v.get('TotalVol', 0)} 輛"
            f"｜佔有率 {v.get('AvgOcc', 0):.1f}%"
        )

    return "\n".join(lines)


def _build_system_prompt(traffic: dict) -> str:
    """Build a grounded system prompt with pre-computed answers to prevent hallucination."""
    road_data: dict = traffic.get("data", {}) if isinstance(traffic.get("data"), dict) else {}
    road_names = list(road_data.keys())
    road_list_str = "、".join(road_names) if road_names else "（無資料）"
    summary = _precompute_summary(road_data)

    return f"""你是 TrafficVision AI 助理，專門回答台北市北科大（NTUT）周邊路網的交通問題。

═══ 鐵則（違反即為錯誤回答）═══
1. 只能提及以下 {len(road_names)} 個路段，絕對禁止提及清單以外的路段名稱：
   {road_list_str}
2. 所有數值（車速、佔有率、流量）必須直接抄自「即時交通摘要」，不得自行計算或捏造。
3. 若問及未監測的路段，回答：「該路段不在本系統監測範圍。目前監測：{road_list_str}」
4. 若摘要顯示資料缺失，回答「尚無該資料」，不得推測。

═══ 即時交通摘要（伺服器已計算完畢，直接引用）═══
{summary}

═══ 其他系統資料 ═══
SUMO 壅塞 Edge（佔有率排序前 10）：{_top_congested_edges(10)}
GRU 預測：{_compact(get_prediction())}
號誌優化：{_compact(get_signal_plan())}
優化效益：{_compact(get_comparison())}

═══ 回答格式 ═══
使用繁體中文。引用數值時標明單位（km/h、%、輛）。簡潔為主，不重複摘要已說明的內容。"""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """呼叫本地 Ollama Gemma 4 模型，附帶即時交通 context 回覆問題。"""
    loop = asyncio.get_running_loop()
    traffic = await loop.run_in_executor(None, get_latest_traffic)
    system = _build_system_prompt(traffic)

    messages = [{"role": "system", "content": system}]
    for turn in req.history[-10:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": str(turn.get("content", ""))})
    messages.append({"role": "user", "content": req.message})

    try:
        import json as _json
        _payload = _json.dumps(
            {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,                # 關閉 thinking：此模型為 Gemma 4 思考版，關閉可省去每次回答前的英文推理，速度約快 2.5 倍
                "options": {"num_ctx": 16384},  # grounded system prompt 實測約 9300 tokens（規則+全路段+SUMO+GRU+號誌+效益），預設 2048／8192 都會截斷掉車流資料導致模型亂答；16384 可完整容納並保留多輪對話餘裕
            },
            ensure_ascii=False,
        ).encode("utf-8")
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                data=_payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=120,
            ),
        )
        resp.raise_for_status()
        reply = resp.json()["message"]["content"]
        return {"reply": reply}
    except requests.exceptions.ConnectTimeout:
        return {
            "error": "ollama_connect_timeout",
            "reply": "AI 服務連線逾時，請確認 Ollama 已啟動（ollama serve）並可從本伺服器存取。",
        }
    except requests.exceptions.ReadTimeout:
        return {
            "error": "ollama_read_timeout",
            "reply": f"AI 模型 {OLLAMA_MODEL} 回應超過 120 秒。可能模型過大或主機 CPU/GPU 不足，請改用較小模型或檢查資源。",
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "ollama_connection_refused",
            "reply": f"無法連線到 Ollama ({OLLAMA_BASE_URL})。請執行 `ollama serve` 並確認位址正確。",
        }
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", "?")
        body = ""
        try:
            body = (exc.response.text or "")[:200]
        except Exception:
            pass
        if status == 404 or "not found" in body.lower():
            hint = f"請先執行：ollama create {OLLAMA_MODEL} -f Modelfile"
        else:
            hint = "請檢查 Ollama 服務狀態或模型設定。"
        return {
            "error": f"ollama_http_{status}",
            "reply": f"AI 服務回傳錯誤 ({status})：{body}。{hint}",
        }
    except requests.RequestException as exc:
        return {
            "error": str(exc),
            "reply": "AI 服務暫時無法使用，請確認 Ollama 已啟動（ollama serve）。",
        }


# ─── WebSocket / 模擬控制端點 ──────────────────────

# Optional bearer-token auth for the WebSocket. Set TRAFFICVISION_WS_TOKEN to
# enable; leave unset to allow anonymous connections (current default — fine
# for localhost deployment). Two ways to supply: ?token=... query param OR
# Sec-WebSocket-Protocol: bearer.<token>  (some browsers won't let you set
# the Authorization header on a WS handshake, hence the query-param fallback).
WS_TOKEN = os.environ.get("TRAFFICVISION_WS_TOKEN", "").strip() or None


def _extract_ws_token(websocket: WebSocket) -> Optional[str]:
    # 1. Query string
    qp = websocket.query_params.get("token")
    if qp:
        return qp
    # 2. Sec-WebSocket-Protocol: bearer.<token>
    proto = websocket.headers.get("sec-websocket-protocol", "")
    if proto.startswith("bearer."):
        return proto[len("bearer."):]
    # 3. Authorization: Bearer <token> (works for native ws clients, not browsers)
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


@app.websocket("/ws/simulation")
async def simulation_ws(websocket: WebSocket):
    """即時模擬串流：每 10 模擬秒推送 {type, sim_time, roads, signals}。

    若設定 TRAFFICVISION_WS_TOKEN 環境變數，要求 client 帶 bearer token
    （?token=… 或 Sec-WebSocket-Protocol: bearer.…）。否則開放連線。
    """
    if WS_TOKEN is not None:
        supplied = _extract_ws_token(websocket)
        if supplied != WS_TOKEN:
            # 1008 = policy violation, per RFC 6455
            await websocket.close(code=1008, reason="unauthorized")
            return
    await websocket.accept()
    _sim_clients.add(websocket)
    if _sim_snapshot:
        await websocket.send_json(_sim_snapshot)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # 心跳：送 ping；送不出去代表連線已死，立即跳出迴圈清理
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _sim_clients.discard(websocket)


@app.post("/api/simulation/start")
def start_simulation(cfg: str = None):
    """啟動 SUMO 模擬。cfg 可指定 .sumocfg 路徑；不指定則用預設。"""
    global _sim_thread
    if _sim_running.is_set():
        return {"status": "already_running", "sim_time": _sim_snapshot.get("sim_time")}
    cfg_path = cfg or str(DEFAULT_SUMOCFG)
    if not Path(cfg_path).exists():
        return {"status": "error", "message": f"設定檔不存在: {cfg_path}"}
    _sim_snapshot.clear()
    _sim_running.set()
    _sim_thread = threading.Thread(
        target=_traci_worker, args=(cfg_path, _event_loop), daemon=True
    )
    _sim_thread.start()
    return {"status": "started", "cfg": cfg_path}


@app.post("/api/simulation/stop")
def stop_simulation():
    """送出停止信號；TraCI 會在目前 step 結束後中斷。"""
    _sim_running.clear()
    return {"status": "stopping"}


@app.get("/api/simulation/status")
def simulation_status():
    """模擬執行狀態：是否執行中、已連線 WebSocket 客戶端數量、目前模擬時間。"""
    return {
        "running":    _sim_running.is_set(),
        "clients":    len(_sim_clients),
        "sim_time":   _sim_snapshot.get("sim_time"),
        "road_count": len(_sim_snapshot.get("roads", {})),
    }


# ─── 啟動 ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("TrafficVision API Server")
    print(f"數據目錄: {DATA_DIR}")
    print("文件: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
