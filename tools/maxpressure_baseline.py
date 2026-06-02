"""Max-Pressure 基準（Varaiya 2013，相位式）vs 預測驅動之動態選擇。

審查委員要求補一個「標準自適應基準」，現況只比了 persistence 與手刻 fixed。
本腳本以 TraCI 實作去中心化、相位式（phase-based）Max-Pressure 控制器，並在
**與動態選擇完全相同的 run／路網／路徑／time_loss 指標**上評估，因此為公平對照。

Max-Pressure 原理
-----------------
對每個受控路口，定義每個「綠燈相位」的壓力 (pressure)：

    pressure(phase) = Σ_{該相位放行的轉向 m} [ q_in(m) − q_out(m) ]

其中 q_in/q_out 為上游／下游車道的車輛數。每個控制週期選擇壓力最大的相位
（受最小綠燈、黃燈清道約束）。此為已證明可穩定路網的去中心化貪婪策略。

公平性設計
----------
- 與動態選擇共用同一份 filtered routes、同一張 net、同一個 BASE_SUMOCFG、
  同一個 summarize_stats_xml（timeLoss = SUMO vehicleTripStatistics）。
- 套用與系統相同的安全下限：MIN_GREEN / MIN_YELLOW（見 traffic_optimizer_signal）。
- 沿用既有相位的綠燈 state 字串 + 計算式黃燈轉場，故不會產生非法相位或破壞
  行人/清道相位（行人相位結構被原樣繼承）。
- 宇集 = data/runtime_data/_metrics.jsonl 之有效 run（與正文 44.43s 同一宇集）。

用法
----
    python tools/maxpressure_baseline.py --sample 60
    python tools/maxpressure_baseline.py --sample 5 --seed 0    # 先小樣本驗證
    python tools/maxpressure_baseline.py --stems-from-ablation  # 用消融的同一批 run 配對比較
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib  # noqa: E402
import traci  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

from traffic_optimizer_io import (  # noqa: E402
    create_temp_sumo_cfg,
    filter_route_file,
    get_valid_edge_ids,
    summarize_stats_xml,
)
from traffic_light_optimizer import UNSAFE_TLS_IDS  # noqa: E402
from traffic_optimizer_signal import (  # noqa: E402
    MIN_GREEN_DURATION,
    MIN_YELLOW_DURATION,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(REPO, "data", "runtime_data")
BASE_SUMOCFG = os.path.join(REPO, "data", "ntut_config.sumocfg")
EDGEDATA_ADD = os.path.join(REPO, "data", "edgedata.add.xml")


def _net_file() -> str:
    root = ET.parse(BASE_SUMOCFG).getroot()
    nf = root.find("input").find("net-file").get("value")
    return nf if os.path.isabs(nf) else os.path.normpath(os.path.join(os.path.dirname(BASE_SUMOCFG), nf))


NET = _net_file()


# ─────────────────────────── 宇集 / 對照數值 ───────────────────────────
def metrics_universe() -> list[str]:
    mj = os.path.join(RT, "_metrics.jsonl")
    rows = [json.loads(l) for l in open(mj, encoding="utf-8")]
    valid = [r for r in rows if r.get("strategy") and r.get("composite_score") is not None]
    out = []
    for r in valid:
        stem = os.path.basename(r["run_dir"])
        rou = os.path.join(RT, stem, "VehicleData", f"{stem}.rou.xml")
        brs = os.path.join(RT, stem, "handoff", "best_result_summary.csv")
        if os.path.exists(rou) and os.path.exists(brs):
            out.append(stem)
    return sorted(out)


def reference_values(stem: str) -> dict:
    """動態選擇 (winner) 與 no_control 基準的 time_loss，以及（若有）最佳固定策略。"""
    out = {"dyn_tl": None, "dyn_strategy": None, "no_control_tl": None, "best_fixed_tl": None}
    brs = os.path.join(RT, stem, "handoff", "best_result_summary.csv")
    try:
        d = pd.read_csv(brs).iloc[0]
        out["dyn_tl"] = float(d["actual_time_loss"])
        out["dyn_strategy"] = str(d["strategy"])
        out["no_control_tl"] = float(d["baseline_time_loss"])
    except Exception:
        pass
    # 最佳固定策略 (baseline_more_edges_more_tls) 的逐策略 time_loss
    pdir = os.path.join(RT, stem, f"{stem}_predict_dynamic")
    f = os.path.join(pdir, "baseline_more_edges_more_tls", f"{stem}_comparison_summary.csv")
    if os.path.isfile(f):
        try:
            dd = pd.read_csv(f).set_index("metric")
            out["best_fixed_tl"] = float(dd.loc["avg_time_loss", "after"])
        except Exception:
            pass
    return out


# ─────────────────────────── Max-Pressure 控制器 ───────────────────────────
def _is_green(ch: str) -> bool:
    return ch in ("G", "g")


def _distinct_green_states(phases) -> list[str]:
    """取出相異的綠燈 state（含至少一個 G/g、且無黃燈 y），保序去重。"""
    seen, out = set(), []
    for p in phases:
        s = p.state
        if ("G" in s or "g" in s) and "y" not in s.lower():
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _make_yellow(cur: str, target: str) -> str:
    """由 cur → target 的黃燈轉場：失去綠燈的位置轉黃，續綠者維持，其餘紅。"""
    y = []
    for c, t in zip(cur, target):
        if _is_green(c) and not _is_green(t):
            y.append("y")
        elif _is_green(c) and _is_green(t):
            y.append(c)
        else:
            y.append("r")
    return "".join(y)


def _setup_controllers() -> dict:
    """為每個可控 TLS 建立 Max-Pressure 控制狀態。需 >=2 個相異綠燈相位才接管。"""
    ctrl = {}
    for t in traci.trafficlight.getIDList():
        if t in UNSAFE_TLS_IDS:
            continue
        logic = traci.trafficlight.getAllProgramLogics(t)[0]
        greens = _distinct_green_states(logic.phases)
        if len(greens) < 2:
            continue  # 退化路口（單一綠相或全綠）→ 交回原程式，較安全
        links = traci.trafficlight.getControlledLinks(t)
        ctrl[t] = dict(greens=greens, links=links, cur=greens[0],
                       green_time=0.0, yellow=None, yellow_time=0.0, target=None)
        traci.trafficlight.setRedYellowGreenState(t, greens[0])  # 接管
    return ctrl


def _pressure(state: str, links) -> float:
    p = 0.0
    for i, ch in enumerate(state):
        if not _is_green(ch) or i >= len(links) or not links[i]:
            continue
        in_lane, out_lane = links[i][0][0], links[i][0][1]
        try:
            qin = traci.lane.getLastStepVehicleNumber(in_lane)
            qout = traci.lane.getLastStepVehicleNumber(out_lane) if out_lane else 0
        except traci.TraCIException:
            continue
        p += qin - qout
    return p


def run_maxpressure(cfg: str, stats_xml: str, sim_cap: float = 3600.0,
                    control_interval: float = 10.0) -> dict:
    """執行一場 Max-Pressure 控制的 SUMO 模擬。

    control_interval：兩次壓力再評估的最小間隔（秒）。設為 >= 最小綠燈可避免
    每秒切相位造成的抖動 (thrashing)；綠燈仍受 MIN_GREEN_DURATION 硬性下限保護。
    """
    cmd = [sumolib.checkBinary("sumo"), "-c", cfg, "--start", "--quit-on-end", "--no-warnings"]
    port = sumolib.miscutils.getFreeSocketPort()
    label = f"mp_{os.getpid()}_{port}"
    reeval = max(control_interval, MIN_GREEN_DURATION)
    try:
        traci.start(cmd, port=port, label=label, numRetries=10)
        traci.switch(label)
        ctrl = _setup_controllers()
        end_t = 0.0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            end_t = traci.simulation.getTime()
            if end_t > sim_cap:
                break
            for t, st in ctrl.items():
                if st["yellow"] is not None:
                    st["yellow_time"] += 1.0
                    if st["yellow_time"] >= MIN_YELLOW_DURATION:
                        traci.trafficlight.setRedYellowGreenState(t, st["target"])
                        st["cur"], st["yellow"], st["green_time"] = st["target"], None, 0.0
                    continue
                st["green_time"] += 1.0
                if st["green_time"] < reeval:
                    continue
                # 已達再評估間隔 → 依壓力決定是否切換（綠燈受 MIN_GREEN 保護）
                best = max(st["greens"], key=lambda s: _pressure(s, st["links"]))
                if best != st["cur"]:
                    yellow = _make_yellow(st["cur"], best)
                    traci.trafficlight.setRedYellowGreenState(t, yellow)
                    st["yellow"], st["yellow_time"], st["target"] = yellow, 0.0, best
                else:
                    st["green_time"] = MIN_GREEN_DURATION  # 維持綠燈，下一秒再評估
    except Exception as exc:
        print(f"    MP 模擬失敗: {exc}")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        sys.stdout.flush()
    # 統計檔由 SUMO 在程序結束 (traci.close) 後才寫出，故於此處解析。
    return summarize_stats_xml(stats_xml)


def eval_one(stem: str, work_root: str) -> dict | None:
    rd = os.path.join(RT, stem)
    rou = os.path.join(rd, "VehicleData", f"{stem}.rou.xml")
    work = os.path.join(work_root, stem)
    os.makedirs(work, exist_ok=True)
    filt = os.path.join(work, f"{stem}_filtered.rou.xml")
    filter_route_file(rou, filt, get_valid_edge_ids(NET))
    cfg = os.path.join(work, f"{stem}_mp.sumocfg")
    stats = os.path.join(work, f"{stem}_mp_stats.xml")
    # 每 strategy 各自的 edgeData 輸出（避免覆蓋共用檔）
    edge_add = os.path.join(work, f"{stem}_edgedata.add.xml")
    tree = ET.parse(EDGEDATA_ADD)
    tree.getroot().find("edgeData").set("file", os.path.join(work, f"{stem}_edgedata.xml"))
    tree.write(edge_add)
    ok = create_temp_sumo_cfg(filt, BASE_SUMOCFG, cfg,
                              additional_files=[edge_add],
                              output_overrides={"output-prefix": "", "statistic-output": stats},
                              exclude_additional_basenames=[os.path.basename(EDGEDATA_ADD)])
    if not ok:
        return None
    summ = run_maxpressure(cfg, stats)
    if not summ:
        return None
    ref = reference_values(stem)
    return dict(stem=stem,
                mp_tl=round(summ.get("avg_time_loss", float("nan")), 3),
                mp_wait=round(summ.get("avg_waiting_time", float("nan")), 3),
                mp_teleports=summ.get("teleports_total"),
                dyn_tl=ref["dyn_tl"], dyn_strategy=ref["dyn_strategy"],
                no_control_tl=ref["no_control_tl"], best_fixed_tl=ref["best_fixed_tl"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stems-from-ablation", action="store_true",
                    help="改用 da1_ablation_n60.csv 中、且在 metrics 宇集內的 run（配對比較）")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "analysis", "maxpressure_vs_dynamic.csv"))
    args = ap.parse_args()

    universe = metrics_universe()
    print(f"metrics 宇集可用 run: {len(universe)}")

    if args.stems_from_ablation:
        abl = pd.read_csv(os.path.join(REPO, "data", "analysis", "da1_ablation_n60.csv"))
        uni = set(universe)
        pick = sorted([s for s in abl["stem"].tolist() if s in uni])
        print(f"消融 ∩ metrics 宇集: {len(pick)} run（配對比較）")
    else:
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(universe, size=min(args.sample, len(universe)), replace=False).tolist())

    work_root = os.path.join(REPO, "data", "analysis", "_mp_runs")
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for i, stem in enumerate(pick, 1):
        r = eval_one(stem, work_root)
        if r is None:
            print(f"[{i}/{len(pick)}] {stem}: 略過")
            continue
        rows.append(r)
        dyn = r["dyn_tl"]
        flag = "MP勝" if (dyn is not None and r["mp_tl"] < dyn) else ("平" if dyn == r["mp_tl"] else "動態勝")
        print(f"[{i}/{len(pick)}] {stem}: MP={r['mp_tl']:.2f} vs 動態={dyn:.2f}({r['dyn_strategy']}) "
              f"no_ctrl={r['no_control_tl']:.2f} → {flag}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("無有效結果。")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # ── 統計摘要（僅用 mp 與 dyn 皆有效的配對）──
    m = df.dropna(subset=["mp_tl", "dyn_tl"])
    n = len(m)
    mp_mean, dyn_mean = m["mp_tl"].mean(), m["dyn_tl"].mean()
    mp_med, dyn_med = m["mp_tl"].median(), m["dyn_tl"].median()
    dyn_wins = int((m["dyn_tl"] < m["mp_tl"]).sum())
    mp_wins = int((m["mp_tl"] < m["dyn_tl"]).sum())
    ties = int((m["mp_tl"] == m["dyn_tl"]).sum())
    summary = dict(n=n, mp_mean_tl=round(mp_mean, 3), dyn_mean_tl=round(dyn_mean, 3),
                   mp_median_tl=round(mp_med, 3), dyn_median_tl=round(dyn_med, 3),
                   dyn_wins=dyn_wins, mp_wins=mp_wins, ties=ties,
                   no_control_mean_tl=round(m["no_control_tl"].mean(), 3))
    if "best_fixed_tl" in m and m["best_fixed_tl"].notna().any():
        summary["best_fixed_mean_tl"] = round(m["best_fixed_tl"].mean(), 3)
    if n - ties >= 1:
        from scipy import stats
        mask = m["mp_tl"] != m["dyn_tl"]
        w, pv = stats.wilcoxon(m.loc[mask, "mp_tl"], m.loc[mask, "dyn_tl"])
        summary["wilcoxon_p"] = float(pv)

    print("\n" + "=" * 60)
    print(f"Max-Pressure vs 動態選擇 (n={n}，指標 time_loss，越低越好)")
    print(f"MP     平均={mp_mean:.2f}  中位數={mp_med:.2f}")
    print(f"動態   平均={dyn_mean:.2f}  中位數={dyn_med:.2f}")
    print(f"no_control 平均={summary['no_control_mean_tl']:.2f}")
    if "best_fixed_mean_tl" in summary:
        print(f"最佳固定 平均={summary['best_fixed_mean_tl']:.2f}")
    print(f"動態較佳 {dyn_wins}/{n} | MP 較佳 {mp_wins}/{n} | 平手 {ties}")
    if "wilcoxon_p" in summary:
        print(f"配對 Wilcoxon p={summary['wilcoxon_p']:.4f}")
    print(f"明細: {args.out}")
    (os.path.splitext(args.out)[0] + "_summary.json")
    open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8").write(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
