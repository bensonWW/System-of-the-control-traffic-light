"""加權權重敏感度分析 (Weight Sensitivity Analysis)

驗證號誌策略選擇對「綜合評分權重」的穩健性。

背景
----
`traffic_light_optimizer.py` 以下列綜合分數挑選最佳策略（越低越好）：

    composite = w_wait * (actual_waiting_time / baseline_waiting_time)
              + w_loss * (actual_time_loss    / baseline_time_loss)

預設 w_wait=0.6、w_loss=0.4。本腳本檢驗：若改變權重，原本勝出的策略
是否仍優於無控制基準 (no_control，其 composite 恆為 1.0)。

資料來源
--------
- data/runtime_data/_metrics.jsonl          : 每次 run 一行，記錄勝出策略
- data/runtime_data/<stem>/handoff/
      <stem>_best_result_summary.csv         : 勝出策略的 wait/loss 原始值

限制
----
系統僅保存「勝出策略」的指標，落敗策略未落地。因此本分析能驗證
「勝出策略 vs no_control」的穩定性，但無法驗證四種非基準策略之間的
完整排序穩定性 —— 後者需重跑 SUMO。

用法
----
    python tools/sensitivity_analysis.py
    python tools/sensitivity_analysis.py --weights 0.5 0.6 0.7 --out-dir data/analysis
    python tools/sensitivity_analysis.py --no-figure
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = REPO_ROOT / "data" / "runtime_data" / "_metrics.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "analysis"
BASELINE_STRATEGY = "no_control"

# 表 5.2 指標：comparison_summary 的 metric 名稱 → 論文標籤
RESULTS_METRICS = [
    ("avg_waiting_time", "平均等待時間 (s)"),
    ("avg_time_loss", "行程延誤 (s)"),
    ("avg_duration", "平均行程時間 (s)"),
    ("avg_depart_delay", "出發延誤 (s)"),
    ("simulation_end_time", "模擬結束時間 (s)"),
    ("teleports_total", "車輛 teleport 次數"),
]


def load_winner_rows(metrics_path: Path) -> pd.DataFrame:
    """讀取每次有效 run 的勝出策略 wait/loss 原始值。"""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"找不到 metrics 檔: {metrics_path}")

    runtime_dir = metrics_path.parent
    rows = [json.loads(line) for line in metrics_path.open(encoding="utf-8")]
    valid = [r for r in rows if r.get("strategy") and r.get("composite_score") is not None]

    recs, missing = [], 0
    for r in valid:
        stem = os.path.basename(r["run_dir"])
        f = runtime_dir / stem / "handoff" / "best_result_summary.csv"
        if not f.is_file():
            missing += 1
            continue
        try:
            d = pd.read_csv(f).iloc[0]
            bw, aw = float(d["baseline_waiting_time"]), float(d["actual_waiting_time"])
            bl, al = float(d["baseline_time_loss"]), float(d["actual_time_loss"])
        except Exception:
            missing += 1
            continue
        if bw <= 0 or bl <= 0 or not np.isfinite([bw, aw, bl, al]).all():
            continue
        recs.append(
            dict(
                stem=stem,
                strategy=str(d["strategy"]),
                wait_ratio=aw / bw,
                loss_ratio=al / bl,
                base_wait=bw, act_wait=aw,
                base_loss=bl, act_loss=al,
            )
        )

    df = pd.DataFrame(recs)
    df.attrs["n_records"] = len(rows)
    df.attrs["n_valid"] = len(valid)
    df.attrs["n_missing"] = missing
    return df


def run_sensitivity(df: pd.DataFrame, weights: list[float]) -> tuple[pd.DataFrame, dict]:
    """對每個權重重算 composite，回傳 (含 composite 欄的 df, 摘要 dict)。"""
    for w in weights:
        df[f"composite_w{w:.2f}"] = w * df["wait_ratio"] + (1.0 - w) * df["loss_ratio"]

    opt = df[df["strategy"] != BASELINE_STRATEGY].copy()  # 真正做了優化的 run
    r = float(opt["wait_ratio"].corr(opt["loss_ratio"])) if len(opt) > 1 else float("nan")

    per_weight = {}
    ref_w = 0.60 if any(abs(w - 0.60) < 1e-9 for w in weights) else weights[0]
    ref_col = f"composite_w{ref_w:.2f}"
    ref_wins = opt[ref_col] < 1.0
    for w in weights:
        col = f"composite_w{w:.2f}"
        beats = opt[col] < 1.0
        flips = int((ref_wins & ~beats).sum())  # 在參考權重下贏、此權重下輸
        per_weight[w] = dict(
            pct_beats_baseline=100.0 * beats.mean(),
            mean_composite=float(opt[col].mean()),
            flips_vs_ref=flips,
            flip_pct_vs_ref=100.0 * flips / max(1, int(ref_wins.sum())),
        )

    summary = dict(
        n_records=df.attrs.get("n_records"),
        n_valid=df.attrs.get("n_valid"),
        n_missing=df.attrs.get("n_missing"),
        n_analyzable=len(df),
        n_optimized=len(opt),
        winner_distribution=df["strategy"].value_counts().to_dict(),
        pearson_r_wait_loss=r,
        ref_weight=ref_w,
        mean_wait_improve_pct=100.0 * float((1 - opt["wait_ratio"]).mean()),
        mean_loss_improve_pct=100.0 * float((1 - opt["loss_ratio"]).mean()),
        per_weight=per_weight,
    )
    return df, summary


def print_report(summary: dict, weights: list[float]) -> None:
    print("=" * 64)
    print("加權權重敏感度分析報告")
    print("=" * 64)
    print(f"總紀錄 {summary['n_records']} | 有效 run {summary['n_valid']} "
          f"| 缺檔 {summary['n_missing']} | 可分析 {summary['n_analyzable']}")
    print(f"非基準(優化)贏家: {summary['n_optimized']}")
    print("贏家分布:")
    for k, v in summary["winner_distribution"].items():
        print(f"  {k:32s} {v}")
    print()
    print(f"等待改善 vs 延誤改善 Pearson r = {summary['pearson_r_wait_loss']:.4f}")
    print(f"平均改善: 等待 {summary['mean_wait_improve_pct']:.2f}% | "
          f"延誤 {summary['mean_loss_improve_pct']:.2f}%")
    print()
    print(f"各權重下仍優於 {BASELINE_STRATEGY} 之比例 (參考權重={summary['ref_weight']:.2f}):")
    for w in weights:
        pw = summary["per_weight"][w]
        print(f"  w_wait={w:.2f}: {pw['pct_beats_baseline']:6.1f}%  "
              f"composite={pw['mean_composite']:.4f}  "
              f"翻盤={pw['flips_vs_ref']} ({pw['flip_pct_vs_ref']:.1f}%)")
    print("=" * 64)


def build_results_table(metrics_path: Path) -> tuple[pd.DataFrame, dict]:
    """從各 run 的 comparison_summary 聚合六項指標，產出表 5.2。"""
    from collections import Counter

    runtime_dir = metrics_path.parent
    rows = [json.loads(line) for line in metrics_path.open(encoding="utf-8")]
    valid = [r for r in rows if r.get("strategy") and r.get("composite_score") is not None]

    acc = {m: {"before": [], "after": []} for m, _ in RESULTS_METRICS}
    strat = []
    for r in valid:
        stem = os.path.basename(r["run_dir"])
        f = runtime_dir / stem / "handoff" / f"{stem}_comparison_summary.csv"
        if not f.is_file():
            continue
        try:
            d = pd.read_csv(f).set_index("metric")
        except Exception:
            continue
        for m, _ in RESULTS_METRICS:
            if m in d.index:
                acc[m]["before"].append(float(d.loc[m, "before"]))
                acc[m]["after"].append(float(d.loc[m, "after"]))
        strat.append(r["strategy"])

    table = []
    for m, label in RESULTS_METRICS:
        b = float(np.mean(acc[m]["before"])) if acc[m]["before"] else float("nan")
        a = float(np.mean(acc[m]["after"])) if acc[m]["after"] else float("nan")
        imp = 100.0 * (b - a) / b if b else float("nan")
        table.append(dict(metric=label, baseline_avg=round(b, 2),
                          best_avg=round(a, 2), improvement_pct=round(imp, 2)))

    n = len(strat)
    dist = dict(Counter(strat))
    meta = dict(
        n_runs=n,
        adaptive_wins=dist.get("adaptive", 0),
        adaptive_pct=round(100.0 * dist.get("adaptive", 0) / max(1, n), 1),
        winner_distribution=dist,
    )
    return pd.DataFrame(table), meta


def build_congestion_strata(metrics_path: Path,
                            quantiles=(0.0, 0.75, 0.90)) -> pd.DataFrame:
    """以 baseline 等待時間分層，比較不同壅塞程度下的改善幅度。

    壅塞代理 = 無控制基準的平均等待時間（越高越塞）。
    """
    runtime_dir = metrics_path.parent
    rows = [json.loads(line) for line in metrics_path.open(encoding="utf-8")]
    valid = [r for r in rows if r.get("strategy") and r.get("composite_score") is not None]

    recs = []
    for r in valid:
        stem = os.path.basename(r["run_dir"])
        f = runtime_dir / stem / "handoff" / f"{stem}_comparison_summary.csv"
        if not f.is_file():
            continue
        try:
            d = pd.read_csv(f).set_index("metric")
        except Exception:
            continue
        if "avg_waiting_time" not in d.index or "avg_time_loss" not in d.index:
            continue
        recs.append(dict(
            strategy=r["strategy"],
            base_wait=float(d.loc["avg_waiting_time", "before"]),
            best_wait=float(d.loc["avg_waiting_time", "after"]),
            base_loss=float(d.loc["avg_time_loss", "before"]),
            best_loss=float(d.loc["avg_time_loss", "after"]),
        ))
    df = pd.DataFrame(recs)
    out = []
    for q in quantiles:
        thr = float(df["base_wait"].quantile(q))
        sub = df[df["base_wait"] >= thr]
        label = "all" if q == 0.0 else f"top_{int(round((1 - q) * 100))}pct"
        out.append(dict(
            stratum=label,
            congestion_quantile=q,
            wait_threshold_s=round(thr, 1),
            n=len(sub),
            wait_improve_pct=round(100 * (1 - sub["best_wait"].mean() / sub["base_wait"].mean()), 2),
            delay_improve_pct=round(100 * (1 - sub["best_loss"].mean() / sub["base_loss"].mean()), 2),
            adaptive_pct=round(100 * (sub["strategy"] == "adaptive").mean(), 1),
            no_control_pct=round(100 * (sub["strategy"] == "no_control").mean(), 1),
        ))
    return pd.DataFrame(out)


ALL_STRATEGIES = [
    "no_control", "baseline_original", "baseline_more_edges",
    "baseline_more_edges_more_tls", "adaptive",
]


def build_oracle_comparison(metrics_path: Path) -> tuple[pd.DataFrame, dict]:
    """用各策略子目錄 (predict_dynamic/<strategy>/) 的逐策略指標，證明
    GRU 動態選擇優於任何單一固定策略，並做完整 winner-flip 敏感度。
    """
    from scipy import stats

    runtime_dir = metrics_path.parent
    rows = [json.loads(line) for line in metrics_path.open(encoding="utf-8")]
    valid = [r for r in rows if r.get("strategy") and r.get("composite_score") is not None]

    runs = []
    for r in valid:
        stem = os.path.basename(r["run_dir"])
        pdir = runtime_dir / stem / f"{stem}_predict_dynamic"
        if not pdir.is_dir():
            continue
        wait, loss = {}, {}
        base_w = base_l = None
        ok = True
        for s in ALL_STRATEGIES:
            f = pdir / s / f"{stem}_comparison_summary.csv"
            if not f.is_file():
                ok = False
                break
            try:
                d = pd.read_csv(f).set_index("metric")
                wait[s] = float(d.loc["avg_waiting_time", "after"])
                loss[s] = float(d.loc["avg_time_loss", "after"])
                if s == "no_control":
                    base_w = float(d.loc["avg_waiting_time", "before"])
                    base_l = float(d.loc["avg_time_loss", "before"])
            except Exception:
                ok = False
                break
        if ok and base_w and base_l:
            runs.append(dict(winner=r["strategy"], wait=wait, loss=loss,
                             base_w=base_w, base_l=base_l))

    fixed_means = {s: float(np.nanmean([x["loss"][s] for x in runs])) for s in ALL_STRATEGIES}
    best_fixed = min(fixed_means, key=fixed_means.get)
    dynamic = np.array([x["loss"][x["winner"]] for x in runs])
    bf = np.array([x["loss"][best_fixed] for x in runs])

    mask = dynamic != bf
    _, p = stats.wilcoxon(dynamic[mask], bf[mask]) if mask.sum() else (None, float("nan"))

    def winner_at(x, w):
        sc = {s: w * (x["wait"][s] / x["base_w"]) + (1 - w) * (x["loss"][s] / x["base_l"])
              for s in ALL_STRATEGIES}
        return min(sc, key=sc.get)

    w5 = [winner_at(x, 0.5) for x in runs]
    w6 = [winner_at(x, 0.6) for x in runs]
    w7 = [winner_at(x, 0.7) for x in runs]
    same = sum(1 for a, b, c in zip(w5, w6, w7) if a == b == c)

    fixed_df = pd.DataFrame(
        [dict(strategy=s, mean_time_loss=round(v, 2)) for s, v in
         sorted(fixed_means.items(), key=lambda kv: kv[1])]
    )
    summary = dict(
        n_runs=len(runs),
        best_fixed=best_fixed,
        best_fixed_time_loss=round(fixed_means[best_fixed], 2),
        dynamic_time_loss=round(float(dynamic.mean()), 2),
        improvement_pct=round(100 * (fixed_means[best_fixed] - dynamic.mean()) / fixed_means[best_fixed], 2),
        dynamic_wins=int((dynamic < bf).sum()),
        dynamic_win_pct=round(100 * (dynamic < bf).mean(), 1),
        ties=int((dynamic == bf).sum()),
        wilcoxon_p=p,
        winner_identical_3weights_pct=round(100 * same / len(runs), 1),
    )
    return fixed_df, summary


def make_figure(df: pd.DataFrame, summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opt = df[df["strategy"] != BASELINE_STRATEGY]
    r = summary["pearson_r_wait_loss"]

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=200)
    ax.scatter(opt["wait_ratio"], opt["loss_ratio"], s=18, alpha=0.55,
               color="#0072B2", edgecolor="none", label=f"Optimized runs (n={len(opt)})")
    lo = float(min(opt["wait_ratio"].min(), opt["loss_ratio"].min())) - 0.01
    hi = float(max(opt["wait_ratio"].max(), opt["loss_ratio"].max())) + 0.01
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", lw=1.0, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Waiting-time ratio  (after / baseline)")
    ax.set_ylabel("Time-loss ratio  (after / baseline)")
    ax.set_title("Robustness of strategy selection to scoring weights")
    ax.text(0.05, 0.92, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"圖已輸出: {out_path}  (+ .pdf)")


def main() -> None:
    p = argparse.ArgumentParser(description="號誌策略加權權重敏感度分析")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.6, 0.7],
                   help="要測試的 w_wait 權重 (w_loss = 1 - w_wait)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--no-oracle", action="store_true",
                   help="跳過 oracle 比較（讀逐策略子目錄較慢）")
    args = p.parse_args()

    df = load_winner_rows(args.metrics)
    if df.empty:
        print("無可分析資料。")
        return
    df, summary = run_sensitivity(df, args.weights)
    print_report(summary, args.weights)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "sensitivity_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path = args.out_dir / "sensitivity_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明細已輸出: {csv_path}")
    print(f"摘要已輸出: {json_path}")

    # 表 5.2（各指標 無控制基準 vs 最佳策略平均）
    tdf, tmeta = build_results_table(args.metrics)
    table_path = args.out_dir / "results_table_5_2.csv"
    tdf.to_csv(table_path, index=False, encoding="utf-8-sig")
    print()
    print("表 5.2（自動產生，無控制基準 vs 最佳策略平均）:")
    print(tdf.to_string(index=False))
    print(f"n_runs={tmeta['n_runs']} | adaptive={tmeta['adaptive_wins']} "
          f"({tmeta['adaptive_pct']}%)")
    print(f"表已輸出: {table_path}")

    # 壅塞分層（驗證「高壅塞下改善更大」）
    cdf = build_congestion_strata(args.metrics)
    cong_path = args.out_dir / "congestion_strata.csv"
    cdf.to_csv(cong_path, index=False, encoding="utf-8-sig")
    print()
    print("壅塞分層（baseline 等待時間分位）:")
    print(cdf.to_string(index=False))
    print(f"分層已輸出: {cong_path}")

    # Oracle 比較 + 完整 winner-flip 敏感度（用逐策略子目錄資料）
    if not args.no_oracle:
        fixed_df, osum = build_oracle_comparison(args.metrics)
        fixed_df.to_csv(args.out_dir / "fixed_strategy_means.csv", index=False, encoding="utf-8-sig")
        (args.out_dir / "oracle_summary.json").write_text(
            json.dumps(osum, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print("Oracle 比較（GRU 動態選擇 vs 永遠固定單一策略，指標=time_loss）:")
        print(fixed_df.to_string(index=False))
        print(f"  最佳固定={osum['best_fixed']} ({osum['best_fixed_time_loss']}s) "
              f"→ 動態={osum['dynamic_time_loss']}s ({osum['improvement_pct']}% 改善)")
        print(f"  動態勝過最佳固定: {osum['dynamic_wins']}/{osum['n_runs']} "
              f"({osum['dynamic_win_pct']}%), Wilcoxon p={osum['wilcoxon_p']:.2e}")
        print(f"  三權重 winner 一致: {osum['winner_identical_3weights_pct']}%")

    if not args.no_figure:
        make_figure(df, summary, args.out_dir / "wait_loss_scatter.png")


if __name__ == "__main__":
    main()
