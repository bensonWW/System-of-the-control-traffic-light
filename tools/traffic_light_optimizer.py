import os
import shutil
from concurrent.futures import ProcessPoolExecutor
import math
import xml.etree.ElementTree as ET

import pandas as pd

from traffic_optimizer_io import (
    create_temp_sumo_cfg,
    filter_route_file,
    find_route_xml_for_prediction,
    get_source_stem_from_prediction_csv,
    get_valid_edge_ids,
    run_sumo_simulation_with_end_time,
    summarize_stats_xml,
)
from traffic_optimizer_signal import (
    build_signal_change_detail_df,
    get_signal_plan_from_prediction,
    write_empty_override_xml,
    write_signal_override_xml,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_SUMOCFG = os.path.join(ROOT_DIR, "data", "ntut_config.sumocfg")
TEMP_ROUTE_DIR = os.path.join(ROOT_DIR, "data", "VehicleData_check")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT_DIR, "data", "prediction_runs")


def _resolve_net_file_from_sumocfg(base_sumocfg):
    try:
        root = ET.parse(base_sumocfg).getroot()
        input_node = root.find("input")
        net_node = input_node.find("net-file") if input_node is not None else None
        net_value = net_node.get("value") if net_node is not None else None
        if not net_value:
            raise ValueError("Missing input/net-file in sumocfg")
        if os.path.isabs(net_value):
            return net_value
        return os.path.normpath(os.path.join(os.path.dirname(base_sumocfg), net_value))
    except Exception:
        # Safe fallback for legacy setups.
        return os.path.join(ROOT_DIR, "data", "ntut_network_split.net.xml")


NET_FILE = _resolve_net_file_from_sumocfg(BASE_SUMOCFG)

DEFAULT_TOP_N_TLS = 6
DEFAULT_TOP_EDGE_COUNT = 20
TOP_PHASES_TO_ADJUST = 2
PRIMARY_PHASE_EXTRA_SECONDS = 10
SECONDARY_PHASE_EXTRA_SECONDS = 6
MIN_EDGE_SCORE = 0.01
FROM_EDGE_WEIGHT = 1.0
TO_EDGE_WEIGHT = 0.8

DEFAULT_STRATEGY = {
    "name": "default",
    "top_n_tls": DEFAULT_TOP_N_TLS,
    "top_phases_to_adjust": TOP_PHASES_TO_ADJUST,
    "primary_phase_extra": PRIMARY_PHASE_EXTRA_SECONDS,
    "secondary_phase_extra": SECONDARY_PHASE_EXTRA_SECONDS,
}

UNSAFE_TLS_IDS = {
    "joinedS_3086736519_3086736520_631668954_631668971_#3more",
    "GS_cluster_9383263990_9383263991_9383263992_9383263993",
    "cluster_2038168688_655375573",
    "cluster_2528018119_655375236",
    "cluster_3431431575_4777363357_655375231_6982059025",
    "cluster_623980090_623980094",
}

METRICS = [
    "simulation_end_time",
    "vehicle_count",
    "avg_duration",
    "avg_waiting_time",
    "avg_time_loss",
    "avg_depart_delay",
    "teleports_total",
    "teleports_wrong_lane",
]

# Composite strategy score: lower is better.
# score = w_wait * (waiting_time / baseline_waiting_time)
#       + w_loss * (time_loss / baseline_time_loss)
SCORE_WEIGHT_WAITING_TIME = 0.6
SCORE_WEIGHT_TIME_LOSS = 0.4


def resolve_strategy(strategy=None):
    return {**DEFAULT_STRATEGY, **(strategy or {})}


def select_strategy_from_prediction(_prediction_csv):
    ranked_rows = [
        {"strategy": "no_control", "no_control": True, "predicted_score": 0.0},
        {
            "strategy": "baseline_original",
            "is_proportional": True,
            "proportional_pool": 6.0,
            "top_n_tls": 3,
            "use_downstream_penalty": True,
            "update_interval": 180.0,
            "top_edge_count_override": 12,
            "predicted_score": 0.0,
        },
        {
            "strategy": "baseline_more_edges",
            "is_proportional": True,
            "proportional_pool": 6.0,
            "top_n_tls": 3,
            "use_downstream_penalty": True,
            "update_interval": 180.0,
            "top_edge_count_override": 20,
            "predicted_score": 0.0,
        },
        {
            "strategy": "baseline_more_edges_more_tls",
            "is_proportional": True,
            "proportional_pool": 8.0,
            "top_n_tls": 4,
            "use_downstream_penalty": True,
            "update_interval": 180.0,
            "top_edge_count_override": 20,
            "predicted_score": 0.0,
        },
    ]
    return ranked_rows, {}


def process_prediction_csv(
    prediction_csv,
    work_dir=None,
    run_simulations=True,
    strategy=None,
    baseline_summary=None,
    route_xml_dir=None,
):
    prediction_csv = os.path.abspath(prediction_csv)
    source_stem = get_source_stem_from_prediction_csv(prediction_csv)
    strategy = resolve_strategy(strategy)
    work_dir = os.path.abspath(work_dir or os.path.join(DEFAULT_OUTPUT_ROOT, f"{source_stem}_predict"))
    os.makedirs(work_dir, exist_ok=True)

    local_prediction_csv = os.path.join(work_dir, os.path.basename(prediction_csv))
    if prediction_csv != local_prediction_csv:
        shutil.copy2(prediction_csv, local_prediction_csv)

    prediction_df = pd.read_csv(local_prediction_csv)
    no_control = bool(strategy.get("no_control", False))

    if no_control:
        time_signal_plans = []
        signal_plan_summary_df = pd.DataFrame(columns=["time", "apply_time", "tl_id", "score"])
        signal_change_detail_df = build_signal_change_detail_df([])
        override_program_map = {}
    else:
        to_edge_weight = TO_EDGE_WEIGHT if strategy.get("use_downstream_penalty", True) else 0.0
        time_signal_plans, signal_plan_summary_df, tls_road_hint_map = get_signal_plan_from_prediction(
            prediction_df,
            strategy.get("top_n_tls", DEFAULT_TOP_N_TLS),
            strategy,
            NET_FILE,
            DEFAULT_TOP_EDGE_COUNT,
            MIN_EDGE_SCORE,
            FROM_EDGE_WEIGHT,
            to_edge_weight,
            UNSAFE_TLS_IDS,
            TOP_PHASES_TO_ADJUST,
            PRIMARY_PHASE_EXTRA_SECONDS,
            SECONDARY_PHASE_EXTRA_SECONDS,
        )
        signal_change_detail_df = build_signal_change_detail_df(time_signal_plans, tls_road_hint_map)
        override_program_map = {}

    signal_plan_summary_csv = os.path.join(work_dir, f"{source_stem}_signal_plan_summary.csv")
    signal_plan_summary_df.to_csv(signal_plan_summary_csv, index=False, encoding="utf-8-sig")

    signal_change_detail_csv = os.path.join(work_dir, f"{source_stem}_signal_change_detail.csv")
    signal_change_detail_df.to_csv(signal_change_detail_csv, index=False, encoding="utf-8-sig")

    override_xml = os.path.join(work_dir, f"{source_stem}_signal_override.xml")
    override_program_map = write_empty_override_xml(override_xml) if no_control else write_signal_override_xml(time_signal_plans, override_xml)

    route_xml_src = find_route_xml_for_prediction(
        prediction_csv,
        os.path.abspath(route_xml_dir or TEMP_ROUTE_DIR),
    )
    route_xml_dst = os.path.join(work_dir, os.path.basename(route_xml_src))
    shutil.copy2(route_xml_src, route_xml_dst)

    filtered_route_xml = os.path.join(work_dir, f"{source_stem}_filtered.rou.xml")
    filter_route_file(route_xml_dst, filtered_route_xml, get_valid_edge_ids(NET_FILE))

    after_cfg = os.path.join(work_dir, f"{source_stem}_after.sumocfg")
    after_stats_xml = os.path.join(work_dir, f"{source_stem}_after_stats.xml")
    comparison_summary_csv = os.path.join(work_dir, f"{source_stem}_comparison_summary.csv")

    cfg_kwargs = {
        "output_overrides": {
            # Disable inherited output-prefix so SUMO writes exactly the path we set.
            "output-prefix": "",
            "statistic-output": after_stats_xml,
        }
    }
    if not no_control:
        cfg_kwargs["additional_files"] = [override_xml]

    create_temp_sumo_cfg(filtered_route_xml, BASE_SUMOCFG, after_cfg, **cfg_kwargs)

    result = {
        "work_dir": work_dir,
        "prediction_csv": local_prediction_csv,
        "signal_plan_summary_csv": signal_plan_summary_csv,
        "signal_change_detail_csv": signal_change_detail_csv,
        "route_xml": route_xml_dst,
        "filtered_route_xml": filtered_route_xml,
        "after_cfg": after_cfg,
        "override_xml": override_xml,
        "comparison_summary_csv": comparison_summary_csv,
        "after_end_time": None,
    }

    if not run_simulations:
        return result

    after_output_csv = os.path.join(work_dir, f"{source_stem}_after.csv")
    sim_kwargs = {"override_program_map": override_program_map} if not no_control else {}
    after_ok, after_end_time = run_sumo_simulation_with_end_time(after_cfg, after_output_csv, **sim_kwargs)
    result["after_end_time"] = after_end_time if after_ok else None

    after_summary = summarize_stats_xml(after_stats_xml) if after_ok else {}
    effective_baseline = baseline_summary or (after_summary if no_control else {})

    comparison_df = pd.DataFrame([
        {
            "metric": metric,
            "before": effective_baseline.get(metric),
            "after": after_summary.get(metric),
            "delta": (
                after_summary.get(metric) - effective_baseline.get(metric)
                if effective_baseline.get(metric) is not None and after_summary.get(metric) is not None
                else None
            ),
        }
        for metric in METRICS
    ])
    comparison_df.to_csv(comparison_summary_csv, index=False, encoding="utf-8-sig")
    result["comparison_df"] = comparison_df
    return result


def evaluate_strategy_worker(args):
    prediction_csv, base_work_dir, run_simulations, strategy, baseline_summary, route_xml_dir = args
    strat_work_dir = os.path.join(base_work_dir, strategy["strategy"])

    result = process_prediction_csv(
        prediction_csv,
        work_dir=strat_work_dir,
        run_simulations=run_simulations,
        strategy=strategy,
        baseline_summary=baseline_summary,
        route_xml_dir=route_xml_dir,
    )

    waiting_time_after = end_time_after = waiting_time_before = end_time_before = float("inf")
    time_loss_after = time_loss_before = float("inf")
    df_comp = result.get("comparison_df")
    if df_comp is None and os.path.exists(result["comparison_summary_csv"]):
        df_comp = pd.read_csv(result["comparison_summary_csv"])

    if df_comp is not None and not df_comp.empty:
        wait_row = df_comp[df_comp["metric"] == "avg_waiting_time"]
        loss_row = df_comp[df_comp["metric"] == "avg_time_loss"]
        end_row = df_comp[df_comp["metric"] == "simulation_end_time"]
        if not wait_row.empty:
            wait_after = pd.to_numeric(wait_row.iloc[0]["after"], errors="coerce")
            wait_before = pd.to_numeric(wait_row.iloc[0]["before"], errors="coerce")
            if not pd.isna(wait_after):
                waiting_time_after = float(wait_after)
            if not pd.isna(wait_before):
                waiting_time_before = float(wait_before)
        if not end_row.empty:
            end_after = pd.to_numeric(end_row.iloc[0]["after"], errors="coerce")
            end_before = pd.to_numeric(end_row.iloc[0]["before"], errors="coerce")
            if not pd.isna(end_after):
                end_time_after = float(end_after)
            if not pd.isna(end_before):
                end_time_before = float(end_before)
        if not loss_row.empty:
            loss_after = pd.to_numeric(loss_row.iloc[0]["after"], errors="coerce")
            loss_before = pd.to_numeric(loss_row.iloc[0]["before"], errors="coerce")
            if not pd.isna(loss_after):
                time_loss_after = float(loss_after)
            if not pd.isna(loss_before):
                time_loss_before = float(loss_before)

    strategy["actual_waiting_time"] = waiting_time_after
    strategy["actual_end_time"] = end_time_after
    strategy["baseline_waiting_time"] = waiting_time_before
    strategy["baseline_end_time"] = end_time_before
    strategy["actual_time_loss"] = time_loss_after
    strategy["baseline_time_loss"] = time_loss_before
    return strategy, result


def compute_composite_score(strategy_row):
    actual_wait = strategy_row.get("actual_waiting_time")
    base_wait = strategy_row.get("baseline_waiting_time")
    actual_loss = strategy_row.get("actual_time_loss")
    base_loss = strategy_row.get("baseline_time_loss")

    if (
        actual_wait is None or base_wait in (None, 0) or
        actual_loss is None or base_loss in (None, 0) or
        math.isinf(actual_wait) or math.isinf(base_wait) or
        math.isinf(actual_loss) or math.isinf(base_loss)
    ):
        return float("inf")

    wait_ratio = float(actual_wait) / float(base_wait)
    loss_ratio = float(actual_loss) / float(base_loss)
    return SCORE_WEIGHT_WAITING_TIME * wait_ratio + SCORE_WEIGHT_TIME_LOSS * loss_ratio


def run_prediction_driven_strategy(
    prediction_csv,
    work_dir=None,
    run_simulations=True,
    route_xml_dir=None,
):
    ranked_rows, profile = select_strategy_from_prediction(prediction_csv)
    source_stem = get_source_stem_from_prediction_csv(prediction_csv)
    base_work_dir = os.path.abspath(work_dir or os.path.join(DEFAULT_OUTPUT_ROOT, f"{source_stem}_predict_dynamic"))
    os.makedirs(base_work_dir, exist_ok=True)

    print(f"\n>>> 啟動策略評估 (共 {len(ranked_rows)} 個策略) <<<")

    no_control_strategy = next((row for row in ranked_rows if row.get("no_control")), None)
    remaining_strategies = [row for row in ranked_rows if not row.get("no_control")]
    if no_control_strategy is None:
        raise RuntimeError("策略清單缺少 no_control，無法建立 baseline。")

    strategy_rows, best_result, best_strategy = [], None, None

    baseline_strategy, baseline_result = evaluate_strategy_worker(
        (prediction_csv, base_work_dir, run_simulations, no_control_strategy, None, route_xml_dir)
    )
    strategy_rows.append(baseline_strategy)

    # 以 no_control 作為基準，使用綜合指標分數選最佳（分數越小越好）。
    best_score = compute_composite_score(baseline_strategy)
    baseline_strategy["composite_score"] = best_score

    baseline_df = baseline_result.get("comparison_df")
    baseline_summary = {str(row.metric): row.after for row in baseline_df.itertuples(index=False)} if baseline_df is not None and not baseline_df.empty else {}

    args_list = [
        (prediction_csv, base_work_dir, run_simulations, strategy, baseline_summary, route_xml_dir)
        for strategy in remaining_strategies
    ]
    if args_list:
        with ProcessPoolExecutor(max_workers=len(args_list)) as executor:
            for strategy, result in executor.map(evaluate_strategy_worker, args_list):
                strategy_rows.append(strategy)
                strategy_score = compute_composite_score(strategy)
                strategy["composite_score"] = strategy_score
                if strategy_score < best_score:
                    best_score, best_result, best_strategy = strategy_score, result, strategy

    if best_strategy is None:
        best_strategy, best_result = baseline_strategy, baseline_result
    if best_strategy is None or best_result is None:
        raise RuntimeError("策略評估失敗，找不到最佳解。")

    baseline_waiting_time = baseline_strategy.get("actual_waiting_time")
    baseline_end_time = baseline_strategy.get("actual_end_time")
    baseline_time_loss = baseline_strategy.get("actual_time_loss")
    for row in strategy_rows:
        row["baseline_waiting_time"] = baseline_waiting_time
        row["baseline_end_time"] = baseline_end_time
        row["baseline_time_loss"] = baseline_time_loss
        if "composite_score" not in row:
            row["composite_score"] = compute_composite_score(row)

    best_strategy_csv = os.path.join(base_work_dir, f"{source_stem}_best_strategy.csv")
    pd.DataFrame([best_strategy]).to_csv(best_strategy_csv, index=False, encoding="utf-8-sig")

    actual_wait = best_strategy.get("actual_waiting_time")
    base_wait = best_strategy.get("baseline_waiting_time")
    actual_end = best_strategy.get("actual_end_time")
    base_end = best_strategy.get("baseline_end_time")

    wait_diff = (
        actual_wait - base_wait
        if actual_wait is not None and base_wait is not None and not math.isinf(actual_wait) and not math.isinf(base_wait)
        else float("nan")
    )
    end_diff = (
        actual_end - base_end
        if actual_end is not None and base_end is not None and not math.isinf(actual_end) and not math.isinf(base_end)
        else float("nan")
    )

    wait_diff_text = f"{wait_diff:+.2f}" if not math.isnan(wait_diff) else "N/A"
    end_diff_text = f"{end_diff:+.2f}" if not math.isnan(end_diff) else "N/A"
    print(
        f"最佳策略: {best_strategy['strategy']} | "
        f"綜合分數 = {best_strategy.get('composite_score', float('nan')):.4f} | "
        f"結束時間 = {best_strategy.get('actual_end_time')} 秒 ({end_diff_text}) | "
        f"等待時間 = {best_strategy.get('actual_waiting_time')} 秒 ({wait_diff_text}) | "
        f"TimeLoss = {best_strategy.get('actual_time_loss')}"
    )

    return {
        "base_work_dir": base_work_dir,
        "strategy_summary_csv": best_strategy_csv,
        "selection_mode": "prediction_dynamic",
        "prediction_profile": profile,
        "best_strategy": best_strategy,
        "best_result": best_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to prediction CSV")
    args = parser.parse_args()

    if args.csv:
        run_prediction_driven_strategy(args.csv)
    else:
        print("請提供預測 CSV 路徑，例如: python traffic_light_optimizer.py --csv output.csv")
