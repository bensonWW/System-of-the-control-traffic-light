import argparse
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

from predict_main import run_full_pipeline
from traffic_optimizer_io import create_temp_sumo_cfg, run_sumo_simulation_with_end_time
from generate_edge_traffic import generate as _generate_edge_heatmap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RUNTIME_DATA_DIR = os.path.join(DATA_DIR, "runtime_data")
BASE_SUMOCFG = os.path.join(DATA_DIR, "ntut_config.sumocfg")
GENERATED_ROUTE_XML = os.path.join(DATA_DIR, "final_output.rou.xml")
# Shared edgeData add file + the per-flow output it gets redirected to.
EDGEDATA_ADD_FILE = os.path.join(DATA_DIR, "edgedata.add.xml")
# Current-demand (Step 2) edgeData — drives the "當前車流" full-network heatmap.
EDGEDATA_CURRENT_FILE = os.path.join(DATA_DIR, "edgedata_current.xml")


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
    pipeline_result = run_full_pipeline(
        input_csv=input_csv,
        model_path=model_path,
        work_dir=prediction_work_dir,
        handoff_dir=handoff_dir,
        route_xml_dir=vehicle_data_dir,
    )

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
        "handoff_dir": pipeline_result["handoff"]["handoff_dir"],
        "edge_heatmap": edge_heatmap_out,
    }
    _log(f"本輪完成: {summary['run_dir']}")
    return summary


def run_scheduler(interval_seconds=300, model_path=None):
    _log(f"啟動排程，每 {interval_seconds} 秒執行一次")
    while True:
        started = time.time()
        try:
            run_once(model_path=model_path)
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
        default=None,
        help="模型路徑，未指定時使用預設 gru_traffic_model.pth",
    )
    args = parser.parse_args()

    if args.once:
        run_once(model_path=args.model_path)
        return

    run_scheduler(interval_seconds=args.interval, model_path=args.model_path)


if __name__ == "__main__":
    main()
