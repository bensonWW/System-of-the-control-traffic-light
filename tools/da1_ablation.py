"""DA-1 消融實驗：GRU 預測驅動選擇 vs 當前觀測（persistence）驅動選擇

魔鬼代言人 DA-1 的質疑：論文證明 GRU 在預測層優於 persistence，但未證明
「以 GRU 預測選策略」在端到端優化上勝過「以當前觀測選策略」。後者更簡單、
不需訓練。本腳本實際比較兩者。

方法
----
對每個 run：
  1. GRU 路線：使用實際 GRU 預測（沿用既有 best_result_summary 的最佳 time_loss）。
  2. persistence 路線：以最後一個觀測 timestep 的逐邊車流量，沿用 GRU 預測檔的
     (time, edge_id) 網格建構 persistence 預測，重新呼叫優化器（同一份 routes、
     同一個 SUMO），取其最佳策略之 time_loss。
  3. 比較兩者的端到端 time_loss（越低越好）。

兩條路線的 SUMO 需求（routes）完全相同，差異僅在「號誌計畫由 GRU 預測還是
persistence 預測驅動」，因此為乾淨的 ablation。

用法
----
    python tools/da1_ablation.py --sample 15
    python tools/da1_ablation.py --sample 15 --seed 0   # seed 影響抽樣
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from traffic_light_optimizer import run_prediction_driven_strategy  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(REPO, "data", "runtime_data")


def _metrics_universe_stems():
    """正文 (表5.2/5.4、oracle) 所用的有效 run 宇集 = _metrics.jsonl 中有勝出策略者。

    僅靠檔案系統掃描 (obs/VehicleData/predict/best_result) 會納入排程器判定為
    無效、未寫入 _metrics.jsonl 的 gridlock run（time_loss 動輒數百秒）。這些 run
    不在正文 44.43s 的宇集內，混入會把消融均值灌高 3 倍（曾誤得 GRU 140.9s）。
    故此處與正文同宇集，確保消融數字與正文一致、可重現。
    """
    mj = os.path.join(RUNTIME, "_metrics.jsonl")
    if not os.path.exists(mj):
        return None
    stems = set()
    for line in open(mj, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("strategy") and r.get("composite_score") is not None:
            stems.add(os.path.basename(r["run_dir"]))
    return stems


def eligible_runs():
    universe = _metrics_universe_stems()  # 與正文同宇集；None 時退回純檔案系統掃描
    out = []
    for d in sorted(os.listdir(RUNTIME)):
        rd = os.path.join(RUNTIME, d)
        if not d.startswith("traffic_data_") or not os.path.isdir(rd):
            continue
        if universe is not None and d not in universe:
            continue
        obs = os.path.join(rd, f"{d}.csv")
        vdir = os.path.join(rd, "VehicleData")
        pc = os.path.join(rd, "handoff", f"{d}_predict.csv")
        bs = os.path.join(rd, "handoff", "best_result_summary.csv")
        if all(os.path.exists(p) for p in (obs, vdir, pc, bs)):
            out.append(d)
    return out


def build_persistence_csv(stem, out_csv):
    """以最後觀測 timestep 的逐邊車流，沿用 GRU 預測網格建 persistence 預測。"""
    rd = os.path.join(RUNTIME, stem)
    obs = pd.read_csv(os.path.join(rd, f"{stem}.csv"))
    recent = obs.sort_values("time").groupby("edge_id")["vehicle_count"].last().to_dict()
    last_t = obs["time"].max()
    last = obs[obs["time"] == last_t].set_index("edge_id")["vehicle_count"].to_dict()

    gru = pd.read_csv(os.path.join(rd, "handoff", f"{stem}_predict.csv"))
    gru["vehicle_count"] = gru["edge_id"].map(
        lambda e: float(last.get(e, recent.get(e, 0.0)))
    )
    gru.to_csv(out_csv, index=False)
    return out_csv


def gru_best_time_loss(stem):
    bs = pd.read_csv(os.path.join(RUNTIME, stem, "handoff", "best_result_summary.csv")).iloc[0]
    return float(bs["actual_time_loss"]), str(bs["strategy"])


def run_persistence(stem):
    rd = os.path.join(RUNTIME, stem)
    work = tempfile.mkdtemp(prefix=f"da1_{stem}_")
    # 命名為 <stem>_predict.csv，使優化器能推導出正確 stem 並找到對應 route
    pers_csv = build_persistence_csv(stem, os.path.join(work, f"{stem}_predict.csv"))
    res = run_prediction_driven_strategy(
        pers_csv, work_dir=work, run_simulations=True,
        route_xml_dir=os.path.join(rd, "VehicleData"),
    )
    bs = res.get("best_strategy", {}) if isinstance(res, dict) else {}
    return float(bs.get("actual_time_loss")), str(bs.get("strategy"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(REPO, "data", "analysis", "da1_ablation.csv"))
    args = p.parse_args()

    runs = eligible_runs()
    print(f"合格 run 數: {len(runs)}")
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(runs, size=min(args.sample, len(runs)), replace=False).tolist())

    rows = []
    for i, stem in enumerate(pick, 1):
        try:
            gru_tl, gru_s = gru_best_time_loss(stem)
            pers_tl, pers_s = run_persistence(stem)
            rows.append(dict(stem=stem, gru_time_loss=round(gru_tl, 3), gru_strategy=gru_s,
                             pers_time_loss=round(pers_tl, 3), pers_strategy=pers_s,
                             gru_better=gru_tl < pers_tl))
            print(f"[{i}/{len(pick)}] {stem}: GRU={gru_tl:.2f}({gru_s}) "
                  f"vs persist={pers_tl:.2f}({pers_s}) "
                  f"{'GRU勝' if gru_tl < pers_tl else ('平' if gru_tl==pers_tl else 'persist勝')}")
        except Exception as exc:
            print(f"[{i}/{len(pick)}] {stem}: 失敗 {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("無有效結果。")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    n = len(df)
    gru_wins = int(df["gru_better"].sum())
    ties = int((df["gru_time_loss"] == df["pers_time_loss"]).sum())
    print("\n" + "=" * 56)
    print(f"DA-1 消融結果 (n={n})")
    print(f"GRU 平均 time_loss   = {df['gru_time_loss'].mean():.3f}")
    print(f"persist 平均 time_loss = {df['pers_time_loss'].mean():.3f}")
    delta = df["pers_time_loss"].mean() - df["gru_time_loss"].mean()
    print(f"GRU 相對 persistence 改善 = {100*delta/df['pers_time_loss'].mean():.2f}%")
    print(f"GRU 較佳 {gru_wins}/{n} | 平手 {ties} | persist 較佳 {n-gru_wins-ties}")
    if n - ties >= 1:
        from scipy import stats
        m = df["gru_time_loss"] != df["pers_time_loss"]
        w, pv = stats.wilcoxon(df.loc[m, "gru_time_loss"], df.loc[m, "pers_time_loss"])
        print(f"Wilcoxon (排除平手 n={m.sum()}): p={pv:.4f}")
    print(f"明細: {args.out}")
    print("=" * 56)


if __name__ == "__main__":
    main()
