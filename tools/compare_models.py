"""
compare_models.py — 用同一份 SUMO base CSV 跑兩個 GRU 模型的完整 pipeline，
並在結束後印出對照表。

用法：
    python tools/compare_models.py                              # 自動找最新的 base CSV
    python tools/compare_models.py --csv data/runtime_data/.../xxx.csv
    python tools/compare_models.py --model-a gru_traffic_model.pth --model-b gru_traffic_model_pair.pth
"""

import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from predict_main import run_full_pipeline


# ── helpers ──────────────────────────────────────────────────────────────────

def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_latest_base_csv():
    """Return the most recently generated base simulation CSV from runtime_data."""
    pattern = os.path.join(ROOT_DIR, "data", "runtime_data", "*", "traffic_data_*.csv")
    candidates = [
        p for p in glob.glob(pattern)
        if not any(tag in os.path.basename(p) for tag in ("_predict", "_after", "_signal", "_comparison", "_best"))
        and os.path.getsize(p) > 10_000   # ignore near-empty files
    ]
    if not candidates:
        raise FileNotFoundError(
            "找不到 base simulation CSV。"
            "請先執行 python tools/runtime_pipeline.py --once 產生資料。"
        )
    return max(candidates, key=os.path.getmtime)


def _read_comparison_summary(handoff_dir, stem):
    path = os.path.join(handoff_dir, f"{stem}_comparison_summary.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {r["metric"]: float(r["after"]) for r in csv.DictReader(f)
                if r.get("after") not in (None, "", "nan")}


def _read_best_result(handoff_dir):
    path = os.path.join(handoff_dir, "best_result_summary.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _heatmap_stats(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    edges = d.get("edges", {})
    spds = [e["spd"] for e in edges.values() if e.get("spd", 0) > 0]
    vols = [e["vol"] for e in edges.values() if e.get("vol", 0) > 0]
    return {
        "total_vehicles": d.get("meta", {}).get("total_vehicles", 0),
        "edges_with_speed": len(spds),
        "avg_speed": round(sum(spds) / len(spds), 1) if spds else 0,
        "max_speed": round(max(spds), 1) if spds else 0,
        "edges_with_vol": len(vols),
        "total_vol": sum(vols),
    }


def _print_table(rows, col_widths):
    def fmt(val, w):
        return str(val)[:w].ljust(w)
    header, *data_rows = rows
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    print("|" + "|".join(f" {fmt(h, w)} " for h, w in zip(header, col_widths)) + "|")
    print(sep)
    for row in data_rows:
        print("|" + "|".join(f" {fmt(v, w)} " for v, w in zip(row, col_widths)) + "|")
    print(sep)


# ── main comparison logic ─────────────────────────────────────────────────────

def run_comparison(input_csv, model_a_path, model_b_path):
    input_csv = os.path.abspath(input_csv)
    model_a_path = os.path.abspath(model_a_path)
    model_b_path = os.path.abspath(model_b_path)

    stem = os.path.splitext(os.path.basename(input_csv))[0]
    run_dir = os.path.dirname(input_csv)
    tvds_data_dir = os.path.join(ROOT_DIR, "TrafficVision Design System", "data")

    results = {}
    for label, model_path in [("model_a", model_a_path), ("model_b", model_b_path)]:
        model_name = os.path.basename(model_path)
        _log(f"═══ 開始跑 {label} ({model_name}) ═══")

        work_dir  = os.path.join(run_dir, f"{stem}_compare_{label}_dynamic")
        handoff_dir = os.path.join(run_dir, f"handoff_compare_{label}")

        try:
            result = run_full_pipeline(
                input_csv=input_csv,
                model_path=model_path,
                work_dir=work_dir,
                handoff_dir=handoff_dir,
                route_xml_dir=os.path.join(run_dir, "VehicleData"),
            )
            results[label] = {
                "model_name": model_name,
                "handoff_dir": handoff_dir,
                "result": result,
                "error": None,
            }
            _log(f"{label} 完成，handoff: {handoff_dir}")
        except Exception as exc:
            _log(f"{label} 失敗: {exc}")
            results[label] = {
                "model_name": model_name,
                "handoff_dir": handoff_dir,
                "result": None,
                "error": str(exc),
            }

    # ── build comparison report ───────────────────────────────────────────────
    _log("══════════════ 比對結果 ══════════════")

    def _get(label, *keys, default="—"):
        r = results.get(label, {})
        if r.get("error"):
            return f"ERROR: {r['error'][:40]}"
        result = r.get("result") or {}
        val = result
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    a_name = results["model_a"]["model_name"]
    b_name = results["model_b"]["model_name"]

    # Signal optimization comparison
    a_comp = _read_comparison_summary(
        results["model_a"]["handoff_dir"], stem) if not results["model_a"]["error"] else {}
    b_comp = _read_comparison_summary(
        results["model_b"]["handoff_dir"], stem) if not results["model_b"]["error"] else {}
    a_best = _read_best_result(
        results["model_a"]["handoff_dir"]) if not results["model_a"]["error"] else {}
    b_best = _read_best_result(
        results["model_b"]["handoff_dir"]) if not results["model_b"]["error"] else {}

    def _fmt_delta(a_val, b_val, unit="", lower_is_better=True):
        try:
            a, b = float(a_val), float(b_val)
            delta = b - a
            arrow = ("↓" if delta < 0 else "↑") if lower_is_better else ("↑" if delta > 0 else "↓")
            sign = "+" if delta > 0 else ""
            return f"{sign}{delta:.1f}{unit} {arrow}"
        except (TypeError, ValueError):
            return "—"

    print()
    print("┌─ 模型基本資訊 ─────────────────────────────────────────────────┐")
    _print_table(
        [
            ["指標", a_name, b_name, "差值(B-A)"],
            ["模型類型",
             "gru_sequence_log1p", "gru_pair_log1p_v1", "—"],
            ["Edge 數量", "196", "230", "+34"],
            ["參數量", "1,157,373", "1,250,043", "+92,670"],
            ["Gap Feature", "無", "有", "—"],
        ],
        [20, 30, 30, 15],
    )

    print()
    print("┌─ 號誌優化後 — 整體模擬指標 ───────────────────────────────────┐")
    _print_table(
        [
            ["指標", a_name[:28], b_name[:28], "差值(B-A)"],
            ["平均等待時間 (s)",
             f"{a_comp.get('avg_waiting_time', '—')}",
             f"{b_comp.get('avg_waiting_time', '—')}",
             _fmt_delta(a_comp.get('avg_waiting_time'), b_comp.get('avg_waiting_time'), "s")],
            ["平均行程時間 (s)",
             f"{a_comp.get('avg_duration', '—')}",
             f"{b_comp.get('avg_duration', '—')}",
             _fmt_delta(a_comp.get('avg_duration'), b_comp.get('avg_duration'), "s")],
            ["平均時間損失 (s)",
             f"{a_comp.get('avg_time_loss', '—')}",
             f"{b_comp.get('avg_time_loss', '—')}",
             _fmt_delta(a_comp.get('avg_time_loss'), b_comp.get('avg_time_loss'), "s")],
            ["模擬結束時間 (s)",
             f"{a_comp.get('simulation_end_time', '—')}",
             f"{b_comp.get('simulation_end_time', '—')}",
             _fmt_delta(a_comp.get('simulation_end_time'), b_comp.get('simulation_end_time'), "s")],
            ["Teleports",
             f"{a_comp.get('teleports_total', '—')}",
             f"{b_comp.get('teleports_total', '—')}",
             _fmt_delta(a_comp.get('teleports_total'), b_comp.get('teleports_total'))],
        ],
        [22, 28, 28, 14],
    )

    print()
    print("┌─ 最佳策略選擇 ──────────────────────────────────────────────────┐")
    _print_table(
        [
            ["項目", a_name[:28], b_name[:28]],
            ["策略名稱",
             a_best.get("strategy", "—"),
             b_best.get("strategy", "—")],
            ["Composite Score",
             a_best.get("composite_score", "—"),
             b_best.get("composite_score", "—")],
            ["Baseline 等待 (s)",
             a_best.get("baseline_waiting_time", "—"),
             b_best.get("baseline_waiting_time", "—")],
            ["優化後等待 (s)",
             a_best.get("actual_waiting_time", "—"),
             b_best.get("actual_waiting_time", "—")],
        ],
        [20, 30, 30],
    )

    print()
    print("┌─ 輸出路徑 ─────────────────────────────────────────────────────┐")
    for label in ("model_a", "model_b"):
        r = results[label]
        status = "✓" if not r["error"] else "✗"
        print(f"  {status} {label} ({r['model_name']}): {r['handoff_dir']}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare two GRU models on the same SUMO CSV")
    parser.add_argument("--csv", default=None,
                        help="Base simulation CSV path (auto-detects latest if omitted)")
    parser.add_argument("--model-a", default="gru_traffic_model.pth",
                        help="First model path (default: gru_traffic_model.pth)")
    parser.add_argument("--model-b", default="gru_traffic_model_pair.pth",
                        help="Second model path (default: gru_traffic_model_pair.pth)")
    args = parser.parse_args()

    input_csv = args.csv or _find_latest_base_csv()
    _log(f"使用 CSV: {input_csv}")
    _log(f"模型 A: {args.model_a}")
    _log(f"模型 B: {args.model_b}")

    model_a = os.path.join(ROOT_DIR, args.model_a) if not os.path.isabs(args.model_a) else args.model_a
    model_b = os.path.join(ROOT_DIR, args.model_b) if not os.path.isabs(args.model_b) else args.model_b

    run_comparison(input_csv, model_a, model_b)


if __name__ == "__main__":
    main()
