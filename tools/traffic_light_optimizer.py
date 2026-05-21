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
# Shared edgeData additional file inherited from BASE_SUMOCFG. Each strategy gets
# its own copy so parallel SUMO runs never write the same edgedata output file.
EDGEDATA_ADD_FILE = os.path.join(ROOT_DIR, "data", "edgedata.add.xml")
EDGEDATA_OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "edgedata_output.xml")
# Baseline (no_control) edgeData — the forecast traffic state without signal optimization.
EDGEDATA_BASELINE_FILE = os.path.join(ROOT_DIR, "data", "edgedata_baseline.xml")


def _write_strategy_edgedata_add(base_add_file, out_add_file, edgedata_output_path):
    """Copy the base edgeData add file, redirecting its output to a per-strategy path."""
    tree = ET.parse(base_add_file)
    edge_node = tree.getroot().find("edgeData")
    if edge_node is None:
        raise RuntimeError(f"edgeData add 檔缺少 <edgeData> 節點: {base_add_file}")
    edge_node.set("file", os.path.abspath(edgedata_output_path))
    tree.write(out_add_file)


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
        return os.path.join(ROOT_DIR, "data", "ntut_network_split.net copy.xml")


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


def _compute_congestion_profile(prediction_csv: str) -> dict:
    """
    從 GRU 預測 CSV 提取壅塞特徵，供策略選擇使用。

    回傳 dict 包含：
        congestion_ratio   : 0..1，壅塞 edge 佔總 edge 的比例
        concentration      : 0..1，前 20% edge 承載的流量比例（高 = 集中）
        temporal_ratio     : 0..1，時間軸車流標準差 / 均值（高 = 快速變動）
        mean_vol           : 平均每步每 edge 預測車輛數
        max_vol            : 最大單 edge 預測均值
        n_congested_edges  : 高於 75 百分位的 edge 數
        n_total_edges      : 總 edge 數
    """
    try:
        df = pd.read_csv(prediction_csv)
        if "時間" in df.columns:
            df = df.rename(columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"})

        edge_means = df.groupby("edge_id")["vehicle_count"].mean()
        time_means = df.groupby("time")["vehicle_count"].mean()

        if edge_means.empty:
            raise ValueError("prediction CSV 無有效 edge 資料")

        threshold      = float(edge_means.quantile(0.75))
        n_congested    = int((edge_means > threshold).sum())
        n_total        = len(edge_means)
        congestion_ratio = n_congested / max(n_total, 1)

        # Concentration：前 20% 高流量 edge 承載的流量占比（Pareto 指標）
        sorted_desc    = edge_means.sort_values(ascending=False)
        top_k          = max(1, int(n_total * 0.2))
        concentration  = float(sorted_desc.iloc[:top_k].sum() / max(sorted_desc.sum(), 1e-9))

        # Temporal ratio：時間維度的變異係數
        t_std  = float(time_means.std()) if len(time_means) > 1 else 0.0
        t_mean = float(time_means.mean())
        temporal_ratio = t_std / max(t_mean, 1e-9)

        return {
            "congestion_ratio":  round(min(congestion_ratio, 1.0), 4),
            "concentration":     round(min(concentration, 1.0), 4),
            "temporal_ratio":    round(min(temporal_ratio, 1.0), 4),
            "mean_vol":          round(float(edge_means.mean()), 4),
            "max_vol":           round(float(edge_means.max()), 4),
            "n_congested_edges": n_congested,
            "n_total_edges":     n_total,
        }

    except Exception as exc:
        print(f"  WARNING: 無法解析預測 CSV ({exc})，使用預設 profile")
        return {
            "congestion_ratio":  0.5,
            "concentration":     0.5,
            "temporal_ratio":    0.2,
            "mean_vol":          0.0,
            "max_vol":           0.0,
            "n_congested_edges": 0,
            "n_total_edges":     0,
        }


def _strategy_intensity(strategy: dict) -> float:
    """
    將策略參數歸一化為 0..1 的「激進程度」。
    0 = 完全不控制, 1 = 最大力度介入。
    """
    if strategy.get("no_control"):
        return 0.0
    top_n  = strategy.get("top_n_tls",             3) / 6.0    # 3..6 → 0.5..1.0
    pool   = strategy.get("proportional_pool",    6.0) / 10.0  # 6..10 → 0.6..1.0
    edges  = strategy.get("top_edge_count_override", 12) / 24.0 # 12..24 → 0.5..1.0
    return round((top_n + pool + edges) / 3.0, 4)


def _predicted_score(strategy: dict, cr: float, tvr: float) -> float:
    """
    預估分數：策略激進程度與當前壅塞強度的匹配距離，越小代表越適合當前情境。

    no_control：壅塞越嚴重分數越高（表示越不適合）。
    控制策略：激進程度越接近 cr 越好；更新間隔越短對快速變動越有利。
    """
    if strategy.get("no_control"):
        return round(float(cr), 4)

    intensity     = _strategy_intensity(strategy)
    interval_norm = strategy.get("update_interval", 180.0) / 300.0  # 60..300 → 0.2..1.0
    # 時間變異大 → 偏好短間隔（interval_norm 低）；tvr 越大，interval_fit 越高
    interval_fit  = abs(interval_norm - (1.0 - min(tvr, 1.0)))

    return round(0.70 * abs(intensity - cr) + 0.30 * interval_fit, 4)


def select_strategy_from_prediction(prediction_csv: str):
    """
    根據 GRU 預測 CSV 計算各策略的預估分數，並動態生成 adaptive 策略。

    策略排序邏輯（predicted_score 越小 = 越適合當前情境）：
      - no_control       : 適合低壅塞，壅塞越重分數越高
      - baseline_original: 保守，適合 cr ≈ 0.25、集中分布
      - baseline_more_edges      : 中度，適合 cr ≈ 0.45、分散分布
      - baseline_more_edges_more_tls: 積極，適合 cr ≈ 0.65
      - adaptive         : 參數依本輪預測即時生成，永遠貼近當前情境
    """
    profile = _compute_congestion_profile(prediction_csv)
    cr  = profile["congestion_ratio"]
    tvr = profile["temporal_ratio"]

    # ── 動態計算 adaptive 策略參數 ─────────────────────────────────────────
    # 壅塞越重 → 更多 TLS、更大 pool、監控更多 edge
    # 時間變異大 → 縮短更新間隔（讓號誌更頻繁響應）
    adaptive_top_n    = int(3 + round(cr * 3))          # 3..6
    adaptive_pool     = round(5.0 + cr * 5.0, 1)        # 5.0..10.0
    adaptive_edges    = int(12 + round(cr * 12))         # 12..24
    adaptive_interval = float(max(60, int(180 - tvr * 120)))  # 60..180 s

    candidates = [
        {
            "strategy":   "no_control",
            "no_control": True,
        },
        {
            "strategy":               "baseline_original",
            "is_proportional":        True,
            "proportional_pool":      6.0,
            "top_n_tls":              3,
            "use_downstream_penalty": True,
            "update_interval":        180.0,
            "top_edge_count_override": 12,
        },
        {
            "strategy":               "baseline_more_edges",
            "is_proportional":        True,
            "proportional_pool":      6.0,
            "top_n_tls":              3,
            "use_downstream_penalty": True,
            "update_interval":        180.0,
            "top_edge_count_override": 20,
        },
        {
            "strategy":               "baseline_more_edges_more_tls",
            "is_proportional":        True,
            "proportional_pool":      8.0,
            "top_n_tls":              4,
            "use_downstream_penalty": True,
            "update_interval":        180.0,
            "top_edge_count_override": 20,
        },
        {
            "strategy":               "adaptive",
            "is_proportional":        True,
            "proportional_pool":      adaptive_pool,
            "top_n_tls":              adaptive_top_n,
            "use_downstream_penalty": True,
            "update_interval":        adaptive_interval,
            "top_edge_count_override": adaptive_edges,
        },
    ]

    # 計算每個策略的預估分數
    for s in candidates:
        s["predicted_score"] = _predicted_score(s, cr, tvr)

    # no_control 永遠排第一（作為 baseline），其餘按 predicted_score 升序
    no_ctrl  = [s for s in candidates if s.get("no_control")]
    ctrl     = sorted([s for s in candidates if not s.get("no_control")],
                      key=lambda s: s["predicted_score"])
    ranked_rows = no_ctrl + ctrl

    print(
        f"  壅塞 profile: ratio={cr:.3f}, concentration={profile['concentration']:.3f}, "
        f"temporal_var={tvr:.3f}, congested_edges={profile['n_congested_edges']}/{profile['n_total_edges']}"
    )
    print(
        f"  Adaptive 策略: top_n_tls={adaptive_top_n}, pool={adaptive_pool}, "
        f"edges={adaptive_edges}, interval={adaptive_interval}s"
    )
    score_str = " | ".join(f"{s['strategy']}={s['predicted_score']:.4f}" for s in ranked_rows)
    print(f"  predicted_score 排序: {score_str}")

    return ranked_rows, profile


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

    # Per-strategy edgeData output so parallel SUMO runs never collide on the shared file.
    strategy_edgedata_xml = os.path.join(work_dir, f"{source_stem}_edgedata.xml")
    strategy_edgedata_add = os.path.join(work_dir, f"{source_stem}_edgedata.add.xml")
    _write_strategy_edgedata_add(EDGEDATA_ADD_FILE, strategy_edgedata_add, strategy_edgedata_xml)

    additional = [strategy_edgedata_add]
    if not no_control:
        additional.append(override_xml)

    cfg_kwargs = {
        "output_overrides": {
            # Disable inherited output-prefix so SUMO writes exactly the path we set.
            "output-prefix": "",
            "statistic-output": after_stats_xml,
        },
        "additional_files": additional,
        # Drop the shared edgedata.add.xml inherited from BASE_SUMOCFG; use the per-strategy copy.
        "exclude_additional_basenames": [os.path.basename(EDGEDATA_ADD_FILE)],
    }

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
        "edgedata_output": strategy_edgedata_xml,
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

    # Publish the winning strategy's edgeData to the canonical path (single writer, no race).
    best_edgedata = best_result.get("edgedata_output")
    if best_edgedata and os.path.exists(best_edgedata):
        try:
            shutil.copy2(best_edgedata, EDGEDATA_OUTPUT_FILE)
        except Exception as exc:
            print(f"複製最佳策略 edgedata 失敗（非致命）: {exc}")

    # Publish the no_control baseline edgeData — drives the frontend's "5-min forecast" view.
    baseline_edgedata = baseline_result.get("edgedata_output")
    if baseline_edgedata and os.path.exists(baseline_edgedata):
        try:
            shutil.copy2(baseline_edgedata, EDGEDATA_BASELINE_FILE)
        except Exception as exc:
            print(f"複製基準 edgedata 失敗（非致命）: {exc}")

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
