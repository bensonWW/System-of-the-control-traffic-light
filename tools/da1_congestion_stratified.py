"""DA-1 壅塞分層再分析：GRU 預測驅動選擇是否「在高壅塞 run 才」勝過 persistence？

魔鬼代言人 DA-1 的後續質疑：即便端到端整體上 GRU 與 persistence 打平，
也許 GRU 的價值「集中在高壅塞情境」——塞車時提前看見車流堆積，才值得用
神經網路。本腳本以既有的 60 組成對消融 (da1_ablation_n60.csv) 為基礎，
依「無控制基準的平均等待時間 (base_wait)」這個壅塞代理分層，檢驗此假設。

重要：本腳本不重跑任何 SUMO，純粹是對既有兩份 CSV 的再分析。

資料來源
--------
1. data/analysis/da1_ablation_n60.csv
     每 run 一列 (n=60)。gru_time_loss / pers_time_loss = 號誌選擇分別由
     GRU 預測 / persistence 驅動時的最佳策略端到端 time_loss（越低越好）。
2. data/analysis/sensitivity_results.csv
     每 stem 一列（勝出策略）。base_wait = no_control 基準平均等待時間
     = 壅塞代理。沿用此值對 60 組 run 分層。

方法
----
- 每 run 的有號誌優勢 adv = pers_time_loss - gru_time_loss（正值=GRU 較佳）。
- 以 stem inner join 取得 base_wait（回報缺漏）。
- 沿用論文既定固定門檻（不在 n=60 上重新求分位數）：
      top_25pct = base_wait >= 51.4
      top_10pct = base_wait >= 76.2
      low       = base_wait <  51.4
- 各層回報：n、n_nontie、gru_mean、pers_mean、gru_minus_pers_pct、
  gru_wins、pers_wins、ties，以及僅對非平手成對做的 Wilcoxon p
  （n_nontie < 6 視為檢定力不足，p 設為 None）。
- 連續檢定：adv vs base_wait 的 Spearman 相關（GRU 優勢是否隨壅塞增加）。

用法
----
    python tools/da1_congestion_stratified.py
    python tools/da1_congestion_stratified.py --top25 51.4 --top10 76.2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION = REPO_ROOT / "data" / "analysis" / "da1_ablation_n60.csv"
DEFAULT_SENSITIVITY = REPO_ROOT / "data" / "analysis" / "sensitivity_results.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "analysis" / "da1_congestion_stratified.csv"

MIN_NONTIE_FOR_WILCOXON = 6  # 非平手成對少於此數，檢定力不足，不報 p


def load_joined(ablation_path: Path, sensitivity_path: Path) -> tuple[pd.DataFrame, dict]:
    """讀兩份 CSV（皆 utf-8-sig 含 BOM），以 stem inner join 取 base_wait。"""
    a = pd.read_csv(ablation_path, encoding="utf-8-sig")
    s = pd.read_csv(sensitivity_path, encoding="utf-8-sig")

    # 驗證 base_wait 在同一 stem 內跨策略應為常數（取一個值）。
    spread = s.groupby("stem")["base_wait"].agg(lambda x: float(x.max() - x.min()))
    max_spread = float(spread.max()) if len(spread) else 0.0
    bad = spread[spread > 1e-6]
    if len(bad):
        raise ValueError(
            f"base_wait 在 {len(bad)} 個 stem 內跨策略不一致（最大 spread={max_spread:.4f}），"
            "壅塞代理不可靠。"
        )
    base_wait = s.groupby("stem")["base_wait"].first()

    abl_stems = set(a["stem"])
    sen_stems = set(base_wait.index)
    missing = sorted(abl_stems - sen_stems)

    m = a.merge(base_wait.rename("base_wait"), left_on="stem", right_index=True, how="inner")
    m["adv"] = m["pers_time_loss"] - m["gru_time_loss"]  # 正值 = GRU 較佳

    meta = dict(
        n_ablation=len(a),
        n_sensitivity_stems=len(sen_stems),
        n_joined=len(m),
        n_missing=len(missing),
        missing_stems=missing,
        base_wait_max_intra_stem_spread=max_spread,
    )
    return m, meta


def stratum_stats(name: str, desc: str, sub: pd.DataFrame) -> dict:
    """單一壅塞層的成對統計。"""
    n = len(sub)
    gru = sub["gru_time_loss"].to_numpy(dtype=float)
    pers = sub["pers_time_loss"].to_numpy(dtype=float)
    nontie_mask = gru != pers
    n_nontie = int(nontie_mask.sum())

    gru_mean = float(gru.mean()) if n else float("nan")
    pers_mean = float(pers.mean()) if n else float("nan")
    gmp_pct = (100.0 * (pers_mean - gru_mean) / pers_mean) if (n and pers_mean) else float("nan")

    gru_wins = int((gru < pers).sum())
    pers_wins = int((pers < gru).sum())
    ties = int((gru == pers).sum())

    wilcoxon_p: float | None
    if n_nontie >= MIN_NONTIE_FOR_WILCOXON:
        _, wilcoxon_p = stats.wilcoxon(gru[nontie_mask], pers[nontie_mask])
        wilcoxon_p = float(wilcoxon_p)
    else:
        wilcoxon_p = None  # 檢定力不足

    return dict(
        name=name,
        threshold_desc=desc,
        n=n,
        n_nontie=n_nontie,
        gru_mean=round(gru_mean, 3),
        pers_mean=round(pers_mean, 3),
        gru_minus_pers_pct=round(gmp_pct, 3) if np.isfinite(gmp_pct) else float("nan"),
        gru_wins=gru_wins,
        pers_wins=pers_wins,
        ties=ties,
        wilcoxon_p=wilcoxon_p,
    )


def build_strata(m: pd.DataFrame, top25: float, top10: float) -> pd.DataFrame:
    rows = [
        stratum_stats("all", "all runs", m),
        stratum_stats("low", f"base_wait < {top25}", m[m["base_wait"] < top25]),
        stratum_stats("top_25pct", f"base_wait >= {top25}", m[m["base_wait"] >= top25]),
        stratum_stats("top_10pct", f"base_wait >= {top10}", m[m["base_wait"] >= top10]),
    ]
    return pd.DataFrame(rows)


def spearman_adv_vs_congestion(m: pd.DataFrame) -> tuple[float, float]:
    """adv vs base_wait 的 Spearman 相關（正值=GRU 優勢隨壅塞增加）。"""
    rho, p = stats.spearmanr(m["adv"], m["base_wait"])
    return float(rho), float(p)


def tie_breakdown(ablation_path: Path) -> dict:
    """檢查全部 60 組平手的成因（多為兩路線都選 no_control）。"""
    a = pd.read_csv(ablation_path, encoding="utf-8-sig")
    tie = a["gru_time_loss"] == a["pers_time_loss"]
    same = a["gru_strategy"] == a["pers_strategy"]
    both_nc = tie & (a["gru_strategy"] == "no_control") & (a["pers_strategy"] == "no_control")
    return dict(
        n_total=len(a),
        n_ties=int(tie.sum()),
        ties_both_no_control=int(both_nc.sum()),
        ties_same_strategy=int((tie & same).sum()),
        ties_different_strategy=int((tie & ~same).sum()),
    )


def print_report(strata: pd.DataFrame, rho: float, sp_p: float,
                 meta: dict, tie: dict, top25: float, top10: float) -> None:
    print("=" * 78)
    print("DA-1 壅塞分層再分析：GRU 預測驅動選擇 vs persistence（按壅塞分層）")
    print("=" * 78)
    print(f"消融 run {meta['n_ablation']} | sensitivity stems {meta['n_sensitivity_stems']} "
          f"| inner join {meta['n_joined']} | 缺漏 {meta['n_missing']}")
    print(f"base_wait 同 stem 內跨策略最大 spread = {meta['base_wait_max_intra_stem_spread']:.4g}（應為 0）")
    if meta["n_missing"]:
        print(f"缺漏 stems（在 da1 但不在 sensitivity，共 {meta['n_missing']}）:")
        for s in meta["missing_stems"]:
            print(f"  - {s}")
    print(f"固定門檻：top_25pct = base_wait>={top25} | top_10pct = base_wait>={top10} "
          f"| low = base_wait<{top25}")
    print()

    print("各壅塞層成對結果（gru_minus_pers_pct 正值 = GRU 較佳；越低 time_loss 越好）:")
    cols = ["name", "threshold_desc", "n", "n_nontie", "gru_mean", "pers_mean",
            "gru_minus_pers_pct", "gru_wins", "pers_wins", "ties", "wilcoxon_p"]
    disp = strata[cols].copy()
    disp["wilcoxon_p"] = disp["wilcoxon_p"].map(
        lambda v: "None(檢定力不足)" if v is None or (isinstance(v, float) and np.isnan(v))
        else f"{v:.4f}")
    print(disp.to_string(index=False))
    print()

    print(f"連續檢定 Spearman(adv, base_wait) = {rho:.4f}  (p={sp_p:.4f})")
    print("  正 rho = GRU 優勢隨壅塞增加；負 rho = 高壅塞反而對 GRU 不利。")
    print()

    print("平手成因（全部 60 組，未經 join）:")
    print(f"  平手 {tie['n_ties']}/{tie['n_total']}；其中兩路線都選 no_control "
          f"{tie['ties_both_no_control']}；同策略 {tie['ties_same_strategy']}；"
          f"不同策略卻同分 {tie['ties_different_strategy']}")
    print("=" * 78)


def main() -> None:
    p = argparse.ArgumentParser(description="DA-1 壅塞分層再分析（不重跑 SUMO）")
    p.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    p.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top25", type=float, default=51.4, help="top_25pct 壅塞門檻（固定，沿用論文）")
    p.add_argument("--top10", type=float, default=76.2, help="top_10pct 壅塞門檻（固定，沿用論文）")
    args = p.parse_args()

    m, meta = load_joined(args.ablation, args.sensitivity)
    strata = build_strata(m, args.top25, args.top10)
    rho, sp_p = spearman_adv_vs_congestion(m)
    tie = tie_breakdown(args.ablation)

    print_report(strata, rho, sp_p, meta, tie, args.top25, args.top10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = strata.copy()
    # CSV 以空字串表示 None（檢定力不足），其餘照寫。
    out_df["wilcoxon_p"] = out_df["wilcoxon_p"].map(
        lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"分層表已輸出: {args.out}")


if __name__ == "__main__":
    main()
