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
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    import pandas as pd
    import requests
    import gzip
    import io
except ImportError:
    print("請安裝依賴：pip install fastapi uvicorn pandas requests")
    raise

# ─── 路徑設定（相對於此檔案所在目錄）───────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
TRAFFIC_DIR   = DATA_DIR / "trafficData"
RUNTIME_DIR   = DATA_DIR / "runtime_data"

# ─── SUMO 主專案路徑 ──────────────────────────────
SUMO_DATA_DIR   = BASE_DIR.parent / "System-of-the-control-traffic-light-kaven" / "data"
SUMO_NET_XML    = SUMO_DATA_DIR / "ntut_network_split.net.xml"
DEFAULT_SUMOCFG = SUMO_DATA_DIR / "ntut_config.sumocfg"

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


def _fetch_taipei_traffic() -> dict:
    """從台北市 Open Data API 抓取 VD 資料，回傳 NTUT 周邊路段字典。"""
    resp = requests.get(_TAIPEI_VD_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as gz:
        root = __import__("xml.etree.ElementTree", fromlist=["ElementTree"]).fromstring(gz.read())
    data: dict = {}
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
            }
        except (ValueError, TypeError):
            continue
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
_edge_road_map: dict = {}
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
            data = await asyncio.get_event_loop().run_in_executor(None, _fetch_taipei_traffic)
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
            data = await asyncio.get_event_loop().run_in_executor(None, _fetch_taipei_traffic)
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

        while _sim_running.is_set():
            traci.simulationStep()
            t = traci.simulation.getTime()

            if traci.simulation.getMinExpectedNumber() == 0 and t > 30.0:
                break
            if t > MAX_SIM_SECONDS:
                break

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

            payload = {"type": "step", "sim_time": t, "roads": roads_out, "signals": signals_out}
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 開發用；生產環境請限制來源
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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


@app.get("/api/traffic")
def get_latest_traffic():
    """
    最新即時車流數據。
    格式：{ timestamp, data: { "路段名稱": { AvgSpd, TotalVol, AvgOcc, MOELevel, ... } } }
    優先讀本地快取（10 分鐘內），過期或不存在則直接打台北 API。
    """
    raw_data: dict = {}
    timestamp: Optional[str] = None
    source = "unknown"

    f = latest_file(str(TRAFFIC_DIR / "*.json"))
    if f:
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(f))).total_seconds()
        if age < 600:
            with open(f, encoding="utf-8") as fp:
                raw = json.load(fp)
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
        }
    return {"timestamp": timestamp, "source_file": source, "road_count": len(data), "data": data}


@app.post("/api/traffic/refresh")
async def refresh_traffic():
    """手動觸發從台北 API 立即抓取最新車流數據並存檔。"""
    try:
        data = await asyncio.get_event_loop().run_in_executor(None, _fetch_taipei_traffic)
        ts = datetime.now().isoformat()
        _save_traffic_json(data, ts)
        return {"status": "ok", "road_count": len(data), "timestamp": ts}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/api/prediction")
def get_prediction():
    """GRU 預測結果 CSV → JSON"""
    handoff = latest_handoff_dir()
    if not handoff:
        return {"error": "找不到 handoff 目錄，請先執行 runtime_pipeline.py"}
    f = latest_file(str(handoff / "prediction*.csv"))
    if not f:
        return {"error": "找不到 prediction CSV"}
    return {"source_file": os.path.basename(f), "records": csv_to_records(f)}


@app.get("/api/signals")
def get_signal_plan():
    """號誌計畫摘要"""
    handoff = latest_handoff_dir()
    if not handoff:
        return {"error": "找不到 handoff 目錄"}
    f = latest_file(str(handoff / "*signal_plan_summary*.csv"))
    if not f:
        return {"error": "找不到 signal plan CSV"}
    return {"source_file": os.path.basename(f), "records": csv_to_records(f)}


@app.get("/api/comparison")
def get_comparison():
    """前後對比：baseline vs 最佳策略的 8 個指標"""
    handoff = latest_handoff_dir()
    if not handoff:
        return {"error": "找不到 handoff 目錄"}
    f = latest_file(str(handoff / "*comparison_summary*.csv"))
    if not f:
        return {"error": "找不到 comparison CSV"}
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
        return {"error": "找不到 handoff 目錄，請先執行 runtime_pipeline.py"}
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


# ─── WebSocket / 模擬控制端點 ──────────────────────

@app.websocket("/ws/simulation")
async def simulation_ws(websocket: WebSocket):
    """即時模擬串流：每 10 模擬秒推送 {type, sim_time, roads, signals}。"""
    await websocket.accept()
    _sim_clients.add(websocket)
    if _sim_snapshot:
        await websocket.send_json(_sim_snapshot)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
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
