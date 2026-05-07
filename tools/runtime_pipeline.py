import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from predict_main import run_full_pipeline
from traffic_optimizer_io import create_temp_sumo_cfg, run_sumo_simulation_with_end_time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RUNTIME_DATA_DIR = os.path.join(DATA_DIR, "runtime_data")
BASE_SUMOCFG = os.path.join(DATA_DIR, "ntut_config.sumocfg")
GENERATED_ROUTE_XML = os.path.join(DATA_DIR, "final_output.rou.xml")


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
    ok_cfg = create_temp_sumo_cfg(
        route_file=route_xml,
        base_cfg=BASE_SUMOCFG,
        temp_cfg_path=temp_cfg,
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

    summary = {
        "run_dir": run_dir,
        "input_csv": input_csv,
        "route_xml": route_copy,
        "simulation_end_time": end_time,
        "handoff_dir": pipeline_result["handoff"]["handoff_dir"],
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
