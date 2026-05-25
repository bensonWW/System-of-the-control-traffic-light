import argparse
import csv
import errno
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime

from predict_main import run_full_pipeline
from traffic_optimizer_io import create_temp_sumo_cfg, run_sumo_simulation_with_end_time
from generate_edge_traffic import generate as _generate_edge_heatmap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RUNTIME_DATA_DIR = os.path.join(DATA_DIR, "runtime_data")
# Lockfile path — prevents two pipeline runs from overlapping. A previous run
# that overruns the scheduler interval (>300s default) would otherwise launch a
# second SUMO batch concurrently, racing for the same edgedata XML outputs.
LOCKFILE_PATH = os.path.join(RUNTIME_DATA_DIR, ".pipeline.lock")
BASE_SUMOCFG = os.path.join(DATA_DIR, "ntut_config.sumocfg")
GENERATED_ROUTE_XML = os.path.join(DATA_DIR, "final_output.rou.xml")
# Shared edgeData add file + the per-flow output it gets redirected to.
EDGEDATA_ADD_FILE = os.path.join(DATA_DIR, "edgedata.add.xml")
# Current-demand (Step 2) edgeData — drives the "當前車流" full-network heatmap.
EDGEDATA_CURRENT_FILE = os.path.join(DATA_DIR, "edgedata_current.xml")
# Append-only run history. One JSON line per run_once invocation. Used for
# computing success rate, average duration, and which strategies win over time.
METRICS_PATH = os.path.join(RUNTIME_DATA_DIR, "_metrics.jsonl")


def _read_best_strategy(handoff_dir):
    """Extract best_strategy + composite_score from a handoff's summary CSV.
    Returns (strategy_name, composite_score) or (None, None) on failure."""
    if not handoff_dir or not os.path.isdir(handoff_dir):
        return None, None
    candidates = [f for f in os.listdir(handoff_dir) if f.endswith("_best_strategy.csv")]
    if not candidates:
        candidates = ["best_result_summary.csv"] if os.path.exists(os.path.join(handoff_dir, "best_result_summary.csv")) else []
    if not candidates:
        return None, None
    try:
        path = os.path.join(handoff_dir, candidates[0])
        with open(path, encoding="utf-8-sig") as fp:
            row = next(csv.DictReader(fp), None)
        if not row:
            return None, None
        strat = row.get("strategy")
        try:
            score = float(row.get("composite_score", "nan"))
        except (TypeError, ValueError):
            score = None
        return strat, score
    except Exception:
        return None, None


def _append_metric(record):
    """Append one JSON line to the metrics log. Best-effort; never raises.

    Schema (each line is independent — no header, easy to tail):
        {
          "ts": "2026-05-26T01:23:45.678",
          "duration_sec": 71.3,
          "success": true,
          "step3_skipped": null | "資料長度不足...",
          "strategy": "baseline_more_edges_more_tls" | null,
          "composite_score": 0.99 | null,
          "run_dir": "data/runtime_data/traffic_data_..." | null,
          "error": null | "<exception class>: <msg>"
        }
    """
    try:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Logging must never break the pipeline itself
        _log(f"  指標寫入失敗（非致命）: {exc}")


def _write_edgedata_add(base_add_file, out_add_file, edgedata_output_path):
    """Copy the base edgeData add file, redirecting its output to an absolute path."""
    tree = ET.parse(base_add_file)
    node = tree.getroot().find("edgeData")
    if node is None:
        raise RuntimeError(f"edgeData add 檔缺少 <edgeData> 節點: {base_add_file}")
    node.set("file", os.path.abspath(edgedata_output_path))
    tree.write(out_add_file)


def _log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


class PipelineBusy(RuntimeError):
    """Raised by _pipeline_lock when a previous run is still in progress."""


def _pid_alive(pid):
    """Cross-platform PID liveness check. Returns True if pid maps to a live
    process *whose argv looks like our pipeline* (best-effort to avoid PID
    reuse false positives). Falls back to bare PID check if psutil missing."""
    try:
        import psutil  # type: ignore
    except ImportError:
        # No psutil → use os.kill(pid, 0) which works on POSIX. On Windows
        # without psutil the safest assumption is "alive" (don't steal a
        # potentially-real lock); operator can still rm the file manually.
        if os.name == "nt":
            return True
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return True
    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline()).lower()
        # Best-effort guard: only treat as ours if the cmdline mentions our
        # script. Otherwise the PID was reused by some unrelated process and
        # we should still respect the lock (caller can rm manually).
        if "runtime_pipeline" in cmdline or "pipeline_lock" in cmdline:
            return True
        return False
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        # AccessDenied or similar → can't tell, be conservative
        return True


def _parse_lockfile_pid(lock_path):
    """Extract the PID from a lockfile. Returns None if absent or malformed."""
    try:
        content = open(lock_path, "r", encoding="utf-8").read()
    except OSError:
        return None
    for token in content.split():
        if token.startswith("pid="):
            try:
                return int(token.split("=", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


@contextmanager
def _pipeline_lock(lock_path=LOCKFILE_PATH):
    """Exclusive file lock — only one run_once may execute at a time.

    Uses O_CREAT|O_EXCL for cross-platform atomicity (works on Windows where
    fcntl.flock is unavailable). Writes PID + start time to the lockfile so a
    stuck run can be diagnosed. On encountering an existing lockfile, checks
    whether the owning PID is actually alive — if not, reclaims the lock
    instead of failing (the previous process crashed without cleanup).
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break  # acquired
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            # Lock exists — check if its owning PID is alive
            stale_pid = _parse_lockfile_pid(lock_path)
            if stale_pid is None:
                # Malformed lockfile; take it over
                _log(f"  發現損壞的 lockfile（無法解析 PID），接管: {lock_path}")
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            if not _pid_alive(stale_pid):
                _log(f"  發現 stale lockfile（PID {stale_pid} 已死），接管: {lock_path}")
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            # Still alive — genuine conflict
            try:
                existing = open(lock_path, "r", encoding="utf-8").read().strip()
            except OSError:
                existing = "<unreadable>"
            raise PipelineBusy(
                f"另一輪 pipeline 仍在執行（lock: {lock_path}, 內容: {existing}）"
            ) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
        yield
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _build_stem(now=None):
    now = now or datetime.now()
    return f"traffic_data_{now.strftime('%Y%m%d_%H%M%S')}"


def _run_route_generation():
    """Run existing route generation pipeline (grab + convert + duarouter)."""
    command = [sys.executable, os.path.join(BASE_DIR, "main.py")]
    result = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "route 產生流程失敗\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not os.path.exists(GENERATED_ROUTE_XML):
        raise FileNotFoundError(f"找不到產生後的路線檔: {GENERATED_ROUTE_XML}")


def _simulate_to_csv(route_xml, output_csv, temp_cfg, stats_xml):
    # Redirect this current-demand run's edgeData to a dedicated file so it isn't
    # clobbered by the strategy runs; this is the "當前車流" full-network state.
    current_edgedata_add = os.path.join(os.path.dirname(temp_cfg), "edgedata_current.add.xml")
    _write_edgedata_add(EDGEDATA_ADD_FILE, current_edgedata_add, EDGEDATA_CURRENT_FILE)

    ok_cfg = create_temp_sumo_cfg(
        route_file=route_xml,
        base_cfg=BASE_SUMOCFG,
        temp_cfg_path=temp_cfg,
        additional_files=[current_edgedata_add],
        exclude_additional_basenames=[os.path.basename(EDGEDATA_ADD_FILE)],
        output_overrides={
            "output-prefix": "",
            "statistic-output": stats_xml,
        },
    )
    if not ok_cfg:
        raise RuntimeError("建立臨時 SUMO 設定檔失敗")

    ok_sim, end_time = run_sumo_simulation_with_end_time(temp_cfg, output_csv)
    if not ok_sim:
        raise RuntimeError("SUMO 模擬失敗")

    if not os.path.exists(output_csv):
        raise FileNotFoundError(f"找不到模擬輸出 CSV: {output_csv}")

    return end_time


def run_once(model_path=None):
    os.makedirs(RUNTIME_DATA_DIR, exist_ok=True)
    started_at = time.time()
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": None,
        "success": False,
        "step3_skipped": None,
        "strategy": None,
        "composite_score": None,
        "run_dir": None,
        "error": None,
    }
    try:
        with _pipeline_lock():
            summary = _run_once_locked(model_path=model_path)
        record["success"] = True
        record["step3_skipped"] = summary.get("step3_skipped")
        record["run_dir"] = summary.get("run_dir")
        strat, score = _read_best_strategy(summary.get("handoff_dir"))
        record["strategy"] = strat
        record["composite_score"] = score
        return summary
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["duration_sec"] = round(time.time() - started_at, 2)
        _append_metric(record)


def _run_once_locked(model_path=None):
    stem = _build_stem()
    run_dir = os.path.join(RUNTIME_DATA_DIR, stem)
    os.makedirs(run_dir, exist_ok=True)
    vehicle_data_dir = os.path.join(run_dir, "VehicleData")
    os.makedirs(vehicle_data_dir, exist_ok=True)

    input_csv = os.path.join(run_dir, f"{stem}.csv")
    temp_cfg = os.path.join(run_dir, f"{stem}.sumocfg")
    stats_xml = os.path.join(run_dir, f"{stem}_stats.xml")
    route_copy = os.path.join(vehicle_data_dir, f"{stem}.rou.xml")

    _log("Step 1/3: 抓取資料並產生 route")
    _run_route_generation()

    # process_prediction_csv 需要在 VehicleData_check 有與 CSV stem 對應的 .rou.xml。
    shutil.copy2(GENERATED_ROUTE_XML, route_copy)

    _log("Step 2/3: 進行 SUMO 模擬並輸出 CSV")
    end_time = _simulate_to_csv(
        route_xml=route_copy,
        output_csv=input_csv,
        temp_cfg=temp_cfg,
        stats_xml=stats_xml,
    )

    _log("Step 3/3: 執行預測 + 訊號優化 + 交付輸出")
    prediction_work_dir = os.path.join(run_dir, f"{stem}_predict_dynamic")
    handoff_dir = os.path.join(run_dir, "handoff")
    pipeline_result = None
    pipeline_skipped_reason = None
    try:
        pipeline_result = run_full_pipeline(
            input_csv=input_csv,
            model_path=model_path,
            work_dir=prediction_work_dir,
            handoff_dir=handoff_dir,
            route_xml_dir=vehicle_data_dir,
        )
    except Exception as exc:
        # Distinguish error categories so we don't silently swallow real bugs.
        # Imported lazily to avoid cycles (predict_to_csv imports torch).
        from predict_to_csv import PredictDataShortageError, PredictModelMismatchError
        if isinstance(exc, PredictDataShortageError):
            # Adaptive trim handled most low-traffic cases; this only fires when
            # sim ends < 360s (essentially no cars). Graceful skip + still run
            # Step 4 heatmaps from whatever XMLs exist (stale predict/baseline,
            # fresh current).
            pipeline_skipped_reason = f"data_shortage: {exc}"
            _log(f"Step 3 跳過（資料不足，無法預測）: {exc}")
            _log("  → 號誌維持現行，僅更新「當前車流」熱力圖；其他兩張保留上輪資料")
        elif isinstance(exc, PredictModelMismatchError):
            # Model is fundamentally incompatible (edge count / feature width
            # mismatch). Skipping won't help — the operator must fix the model.
            # But we still want the pipeline to complete the current heatmap
            # so the dashboard isn't blank.
            pipeline_skipped_reason = f"model_mismatch: {exc}"
            _log(f"Step 3 跳過（模型不相容，需操作員介入）: {exc}")
        else:
            # Unknown failure mode — re-raise so the scheduler logs it and
            # operators see the actual stack trace. Don't pretend everything's
            # fine when it isn't.
            raise

    _log("Step 4/4: 生成邊道熱力圖 JSON")
    tvds_data_dir = os.path.join(ROOT_DIR, "TrafficVision Design System", "data")
    edge_heatmap_out = os.path.join(tvds_data_dir, "edge_heatmap.json")
    net_file = os.path.join(DATA_DIR, "ntut_network_split.net.xml")
    rou_file = os.path.join(DATA_DIR, "final_output.rou.alt.xml")
    try:
        _generate_edge_heatmap(
            net_file  = net_file,
            rou_file  = rou_file,
            edge_file = os.path.join(DATA_DIR, "edgedata_output.xml"),
            out_file  = edge_heatmap_out,
        )
        _log(f"邊道熱力圖已更新: {edge_heatmap_out}")
    except Exception as exc:
        _log(f"邊道熱力圖生成失敗（非致命）: {exc}")

    # 基準（no_control）熱力圖 — 供前端「5 分鐘預測」視圖聚合使用
    edge_heatmap_baseline_out = os.path.join(tvds_data_dir, "edge_heatmap_baseline.json")
    try:
        _generate_edge_heatmap(
            net_file  = net_file,
            rou_file  = rou_file,
            edge_file = os.path.join(DATA_DIR, "edgedata_baseline.xml"),
            out_file  = edge_heatmap_baseline_out,
        )
        _log(f"基準邊道熱力圖已更新: {edge_heatmap_baseline_out}")
    except Exception as exc:
        _log(f"基準邊道熱力圖生成失敗（非致命）: {exc}")

    # 當前需求（Step 2）熱力圖 — 供前端「當前車流」全路網視圖使用
    edge_heatmap_current_out = os.path.join(tvds_data_dir, "edge_heatmap_current.json")
    try:
        _generate_edge_heatmap(
            net_file  = net_file,
            rou_file  = rou_file,
            edge_file = EDGEDATA_CURRENT_FILE,
            out_file  = edge_heatmap_current_out,
        )
        _log(f"當前邊道熱力圖已更新: {edge_heatmap_current_out}")
    except Exception as exc:
        _log(f"當前邊道熱力圖生成失敗（非致命）: {exc}")

    summary = {
        "run_dir": run_dir,
        "input_csv": input_csv,
        "route_xml": route_copy,
        "simulation_end_time": end_time,
        "handoff_dir": pipeline_result["handoff"]["handoff_dir"] if pipeline_result else None,
        "edge_heatmap": edge_heatmap_out,
        "step3_skipped": pipeline_skipped_reason,
    }
    if pipeline_skipped_reason:
        _log(f"本輪完成（Step 3 已跳過）: {summary['run_dir']}")
    else:
        _log(f"本輪完成: {summary['run_dir']}")
    return summary


def run_scheduler(interval_seconds=300, model_path=None):
    _log(f"啟動排程，每 {interval_seconds} 秒執行一次")
    while True:
        started = time.time()
        try:
            run_once(model_path=model_path)
        except PipelineBusy as exc:
            # Previous run still in progress — skip this tick rather than queue
            # a second concurrent SUMO batch that would race for the same files.
            _log(f"本輪跳過: {exc}")
        except Exception as exc:
            _log(f"本輪執行失敗: {exc}")

        elapsed = time.time() - started
        sleep_seconds = max(0.0, float(interval_seconds) - elapsed)
        next_time = datetime.now().timestamp() + sleep_seconds
        _log(
            "等待下一輪，預計於 "
            + datetime.fromtimestamp(next_time).strftime("%Y-%m-%d %H:%M:%S")
            + " 開始"
        )
        time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="Runtime traffic pipeline scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="只執行一次，不進入每 5 分鐘排程",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="排程間隔秒數（預設 300 秒）",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(ROOT_DIR, "gru_traffic_model_pair.pth"),
        help="模型路徑，未指定時使用 pair-based 模型 (gru_traffic_model_pair.pth)。"
             "若要 rollback 到舊 sliding 模型,顯式指定 --model-path gru_traffic_model.pth",
    )
    args = parser.parse_args()

    if args.once:
        run_once(model_path=args.model_path)
        return

    run_scheduler(interval_seconds=args.interval, model_path=args.model_path)


if __name__ == "__main__":
    main()
