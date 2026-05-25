#!/usr/bin/env python3
"""
從 handoff CSV 與即時車流 JSON 自動生成 Gemma 4 指令微調資料集。
輸出格式：data/finetune_dataset.jsonl（Unsloth chat template）

執行方式：
    python tools/generate_finetune_dataset.py

設計原則（vs 舊版）：
    - 答案**全部來自實際 handoff CSV 內容**，不再使用「if MOE=2 → 建議延長綠燈」
      這種硬編模板（會教 LLM 講系統實際上沒做的話）。
    - 欄位名稱對齊 traffic_light_optimizer.py 與 traffic_optimizer_signal.py
      的實際 schema（舊版猜 "green" / "strat" / "speed" 全部對不到 → "?"）。
    - 號誌策略答案區分 no_control（不介入）vs 主動優化兩種情境；no_control 時
      明白說「優化器試了但沒能超越 baseline」而不是「我們調了綠燈」。
    - prediction 答案把 edge_id 透過 SUMO net.xml 聚合回中文路名。
"""
import argparse
import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
ROOT = BASE.parent
TRAFFIC_DIR = ROOT / "TrafficVision Design System" / "data" / "trafficData"
RUNTIME_DIR = ROOT / "data" / "runtime_data"
NET_XML     = ROOT / "data" / "ntut_network_split.net.xml"
OUT_FILE    = ROOT / "data" / "finetune_dataset.jsonl"

# 取樣上限：歷史資料越多，模型越能學到「不同時段該講什麼」。
# 跑得慢就把這兩個調小；訓練效果不夠就調大（500 / 50 是合理上限）。
MAX_TRAFFIC_FILES = int(os.environ.get("FT_MAX_TRAFFIC_FILES", "200"))
MAX_HANDOFF_DIRS  = int(os.environ.get("FT_MAX_HANDOFF_DIRS",  "30"))

SYSTEM_PROMPT = (
    "你是 TrafficVision AI 助理，專門分析台北市北科大周邊路網的即時車流、"
    "GRU 預測與號誌優化結果。請使用繁體中文回答，數據需引用具體數值，"
    "建議要有依據，避免含糊描述。"
)
MOE_LABELS = {0: "暢通", 1: "緩行", 2: "壅塞"}


# ─── 共用工具 ──────────────────────────────────────────────────────────────────

def qa(question, answer):
    return {"conversations": [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human",  "value": question},
        {"from": "gpt",    "value": answer},
    ]}


def fmt(x, digits=2, default="—"):
    """安全把任意值格式化為 X.XX；None / 非數值回 default。"""
    try:
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f"{f:.{digits}f}" if digits else str(int(f))
    except (TypeError, ValueError):
        return default


def load_csv(path):
    """讀 CSV 為 list of dict；任何錯誤回空 list。"""
    try:
        import pandas as pd
        return pd.read_csv(path).where(lambda df: df.notna(), None).to_dict(orient="records")
    except Exception:
        return []


def load_edge_name_map():
    """從 SUMO net.xml 建 edge_id → 中文路名（無路名就回 edge_id 本身）。"""
    if not NET_XML.exists():
        print(f"  警告: 找不到 {NET_XML}，predict 答案無法聚合到路名")
        return {}
    try:
        root = ET.parse(str(NET_XML)).getroot()
    except Exception as exc:
        print(f"  警告: 解析 net.xml 失敗 ({exc})，predict 答案無法聚合到路名")
        return {}
    result = {}
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):  # internal edges
            continue
        name = (edge.get("name") or "").strip()
        result[eid] = name or eid
    return result


EDGE_NAME_MAP = load_edge_name_map()


# ─── 1. 即時車流 snapshot Q&A ──────────────────────────────────────────────────

def load_all_traffic_files():
    files = sorted(glob.glob(str(TRAFFIC_DIR / "*.json")))
    if len(files) > MAX_TRAFFIC_FILES:
        step = len(files) / MAX_TRAFFIC_FILES
        files = [files[int(i * step)] for i in range(MAX_TRAFFIC_FILES)]
    result = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                raw = json.load(fp)
            data = raw.get("data", {})
            if data:
                result.append((Path(f).name, raw.get("timestamp", ""), data))
        except Exception:
            continue
    return result


def make_traffic_records(fname, timestamp, traffic):
    """從一份 traffic snapshot 生成 Q&A。

    與舊版差別：移除「if MOE=2 → 建議延長綠燈」這種硬編建議，因為號誌實際是否
    調整由 traffic_light_optimizer 決定，不是 MOE 直接對應。模型應該學「描述
    狀態」而不是「替系統做未經驗證的承諾」。
    """
    records = []
    if not traffic:
        return records

    ts_hint = timestamp.split("T")[0] if timestamp else fname.replace("traffic_", "").replace(".json", "")
    sorted_by_moe = sorted(traffic.items(), key=lambda x: (-x[1].get("MOELevel", 0), x[1].get("AvgSpd", 0)))

    # 為避免 Q&A 爆量，每份 snapshot 只取 6 個路段做個別 Q&A（壅塞最重 3 + 隨機 3）
    severe = sorted_by_moe[:3]
    pool   = sorted_by_moe[3:]
    rest   = random.sample(pool, min(3, len(pool))) if pool else []
    sample_roads = severe + rest

    for road, v in sample_roads:
        spd = v.get("AvgSpd", 0) or 0
        vol = v.get("TotalVol", 0) or 0
        occ = v.get("AvgOcc", 0) or 0
        moe = v.get("MOELevel", 0) or 0
        lbl = MOE_LABELS.get(moe, "未知")

        # 純描述，不做政策建議
        a = (
            f"根據 {ts_hint} 的 VD 監測，**{road}** 目前為 **{lbl}（MOE {moe}）**：\n\n"
            f"- 平均車速：**{fmt(spd, 1)} km/h**\n"
            f"- 車流量：**{int(vol)} 輛**\n"
            f"- 佔有率：**{fmt(occ, 1)}%**"
        )
        # 4 種問法（之前 5 種，移除「請描述」太冗）
        for q_tmpl in [
            f"請問 {road} 目前的車流狀況？",
            f"{road} 現在塞車嗎？",
            f"目前 {road} 的平均車速是多少？",
            f"{road} 的 MOE 等級是什麼？",
        ]:
            records.append(qa(q_tmpl, a))

    # 整體概況
    total     = len(traffic)
    congested = sum(1 for v in traffic.values() if v.get("MOELevel") == 2)
    slow      = sum(1 for v in traffic.values() if v.get("MOELevel") == 1)
    smooth    = total - congested - slow
    worst_road, worst_v = sorted_by_moe[0]

    overall_a = (
        f"監測 **{total}** 個路段：\n\n"
        f"- 壅塞（MOE 2）：**{congested}** 個\n"
        f"- 緩行（MOE 1）：**{slow}** 個\n"
        f"- 暢通（MOE 0）：**{smooth}** 個\n\n"
        f"當前壓力最大的是 **{worst_road}**（車速 {fmt(worst_v.get('AvgSpd'), 1)} km/h、"
        f"佔有率 {fmt(worst_v.get('AvgOcc'), 1)}%）。"
    )
    for q_text in ["目前整體交通狀況如何？", "整個區域現在塞嗎？", "全區路況摘要？"]:
        records.append(qa(q_text, overall_a))

    # 最嚴重
    worst_a = (
        f"目前壅塞最嚴重的是 **{worst_road}**，MOE 等級 {worst_v.get('MOELevel')}，"
        f"平均車速 {fmt(worst_v.get('AvgSpd'), 1)} km/h，"
        f"佔有率 {fmt(worst_v.get('AvgOcc'), 1)}%。"
    )
    for q_text in ["目前哪條路段最嚴重？", "哪裡塞車？", "現在壅塞最嚴重的路段？"]:
        records.append(qa(q_text, worst_a))

    return records


# ─── 2. handoff CSV → 真實的預測/號誌/對比 Q&A ─────────────────────────────────

def all_handoff_dirs():
    dirs = sorted(glob.glob(str(RUNTIME_DIR / "*" / "handoff")))
    return [Path(d) for d in dirs[-MAX_HANDOFF_DIRS:]]


def latest_file(handoff: Path, pattern):
    files = sorted(glob.glob(str(handoff / pattern)))
    return files[-1] if files else None


def aggregate_predict_by_road(predict_rows):
    """把 edge_id × 時間的預測資料聚合為 {road_name: mean_vehicle_count}。"""
    by_road = defaultdict(list)
    for r in predict_rows:
        eid = r.get("edge_id", "")
        vol = r.get("vehicle_count")
        if vol is None:
            continue
        road = EDGE_NAME_MAP.get(eid, "")
        if not road or road == eid:  # 沒對應到中文路名就略過
            continue
        try:
            by_road[road].append(float(vol))
        except (TypeError, ValueError):
            continue
    return {road: sum(vs) / len(vs) for road, vs in by_road.items() if vs}


def make_prediction_records(handoff: Path):
    """從 predict.csv 生成預測 Q&A（用真實的 edge_id → 路名聚合）。"""
    records = []
    pred_file = latest_file(handoff, "*_predict.csv")
    if not pred_file:
        return records
    rows = load_csv(pred_file)
    if not rows:
        return records

    # 預測涵蓋的時間範圍（pair model 約 360-640s, ~5 分鐘）
    times = sorted({float(r["time"]) for r in rows if r.get("time") is not None})
    if not times:
        return records
    horizon_min = max(1, round((times[-1] - times[0]) / 60))

    road_avg = aggregate_predict_by_road(rows)
    if not road_avg:
        # 沒聚合到任何路名（可能 net.xml 缺 name），回退到 edge_id 列表
        records.append(qa(
            f"未來 {horizon_min} 分鐘車流預測？",
            f"GRU 模型預測未來 {horizon_min} 分鐘共 {len(rows)} 筆 edge × time 紀錄，"
            f"涵蓋 {len({r.get('edge_id') for r in rows})} 個路段。詳細資料請查 handoff/*_predict.csv。"
        ))
        return records

    top = sorted(road_avg.items(), key=lambda x: -x[1])[:5]
    summary_lines = [f"- **{road}**：預測平均 **{fmt(avg, 1)}** 輛 / 20 秒視窗" for road, avg in top]
    summary = "\n".join(summary_lines)

    for q_text in [
        f"{horizon_min} 分鐘後各路段車流預測如何？",
        f"未來 {horizon_min} 分鐘的車流預測結果？",
        f"GRU 模型預測下一時段車流情況？",
    ]:
        records.append(qa(
            q_text,
            f"根據 GRU 模型預測（共 {len(times)} 個時間步、{horizon_min} 分鐘 horizon），"
            f"預測車流量最高的 5 條路段：\n\n{summary}\n\n"
            f"以上為模型輸出之 baseline 預測，實際數值會受號誌方案影響。"
        ))

    return records


def make_strategy_records(handoff: Path):
    """從 best_strategy.csv + signal_change_detail.csv + comparison_summary.csv
    生成「實際採用了什麼策略、變動了哪些號誌」的 Q&A。"""
    records = []

    best_file = latest_file(handoff, "*_best_strategy.csv")
    change_file = latest_file(handoff, "*_signal_change_detail.csv")
    comp_file = latest_file(handoff, "*_comparison_summary.csv")

    best_rows = load_csv(best_file) if best_file else []
    change_rows = load_csv(change_file) if change_file else []
    comp_rows = load_csv(comp_file) if comp_file else []

    if not best_rows:
        return records
    best = best_rows[0]
    strategy_name = best.get("strategy", "unknown")
    is_no_control = str(best.get("no_control", "False")).lower() in ("true", "1")
    composite = fmt(best.get("composite_score"), 4)
    wait_after  = fmt(best.get("actual_waiting_time"), 2)
    wait_before = fmt(best.get("baseline_waiting_time"), 2)
    loss_after  = fmt(best.get("actual_time_loss"), 2)
    loss_before = fmt(best.get("baseline_time_loss"), 2)

    # ── A. 「採用了什麼策略？」
    if is_no_control:
        a = (
            f"本輪採用 **no_control（不介入）**。\n\n"
            f"原因：5 個候選策略並行模擬後，沒有任何優化方案能在「平均等待時間」與「平均時間損失」"
            f"上明顯超越基準（composite_score = {composite}，越小越好；=1.0 代表與基準持平）。\n\n"
            f"- 平均等待時間：{wait_after} 秒（與基準相同）\n"
            f"- 平均時間損失：{loss_after} 秒（與基準相同）\n\n"
            f"低車流時段或 GRU 預測未顯示明顯壅塞時，no_control 是合理的選擇 — 主動調整反而可能擾動其他相位。"
        )
    else:
        a = (
            f"本輪採用 **{strategy_name}** 策略（composite_score = {composite}）。\n\n"
            f"優化前後對比：\n\n"
            f"- 平均等待時間：{wait_before} 秒 → **{wait_after} 秒**\n"
            f"- 平均時間損失：{loss_before} 秒 → **{loss_after} 秒**"
        )
    for q_text in ["本輪採用了什麼號誌策略？", "最佳策略是哪個？", "號誌優化方案為何？"]:
        records.append(qa(q_text, a))

    # ── B. 「優化了哪些號誌？」（真實 phase change 細節）
    if change_rows and not is_no_control:
        # 取前 5 個 phase change，避免一次塞太長
        sample = change_rows[:5]
        change_lines = []
        for c in sample:
            tl = c.get("tl_id", "?")
            hint = c.get("road_hint", "")
            phase_idx = c.get("phase_index", "?")
            state = c.get("state", "?")
            old_d = fmt(c.get("old_duration"), 1)
            new_d = fmt(c.get("new_duration"), 1)
            # delta with explicit sign; if missing, just show old → new
            try:
                d_val = float(c.get("delta_duration"))
                delta_str = f"（{d_val:+.1f}s）"
            except (TypeError, ValueError):
                delta_str = ""
            label = f"{tl}（{hint}）" if hint else tl
            change_lines.append(
                f"- **{label}** phase #{phase_idx} `{state}`：{old_d}s → {new_d}s{delta_str}"
            )
        change_summary = "\n".join(change_lines)
        total_changes = len(change_rows)
        suffix = f"\n\n（共 {total_changes} 個 phase 被調整，僅顯示前 {len(sample)} 個）" if total_changes > len(sample) else ""

        for q_text in ["優化了哪幾個路口的號誌？", "具體調整了哪些紅綠燈？", "號誌變更的細節？"]:
            records.append(qa(
                q_text,
                f"本輪 **{strategy_name}** 策略對以下路口的 phase 做了時長調整：\n\n{change_summary}{suffix}",
            ))
    elif is_no_control:
        for q_text in ["優化了哪幾個路口的號誌？", "具體調整了哪些紅綠燈？"]:
            records.append(qa(
                q_text,
                "本輪未調整任何號誌（採用 no_control）。最佳策略選擇 no_control 代表優化器在 5 個候選方案中"
                "找不到顯著優於基準的方案，因此保留現行號誌時制。"
            ))

    # ── C. 「改善了多少？」（用真實 comparison_summary 的 delta）
    if comp_rows:
        metric_map = {r.get("metric"): r for r in comp_rows if isinstance(r, dict)}
        # 抓三個主要指標
        wait_row = metric_map.get("avg_waiting_time", {})
        loss_row = metric_map.get("avg_time_loss", {})
        depart_row = metric_map.get("avg_depart_delay", {})

        def line(name, row, unit="秒"):
            before = fmt(row.get("before"), 2)
            after  = fmt(row.get("after"), 2)
            delta  = fmt(row.get("delta"), 2)
            arrow = ""
            try:
                d = float(row.get("delta"))
                arrow = "（改善）" if d < 0 else ("（惡化）" if d > 0 else "（持平）")
            except (TypeError, ValueError):
                pass
            return f"- **{name}**：{before} {unit} → {after} {unit}（Δ {delta} {unit}）{arrow}"

        compare_summary = "\n".join([
            line("平均等待時間", wait_row),
            line("平均時間損失", loss_row),
            line("平均出發延誤", depart_row),
        ])
        for q_text in ["號誌優化前後差異多少？", "優化效果如何？", "採用最佳策略後改善了多少？"]:
            records.append(qa(
                q_text,
                f"本輪策略：**{strategy_name}**\n\n優化前後對比：\n\n{compare_summary}\n\n"
                f"composite_score = **{composite}**（< 1.0 代表優於基準；本輪 = {composite}）。",
            ))

    return records


# ─── 3. 通用知識 Q&A ─────────────────────────────────────────────────────────

GENERAL_QAS = [
    (
        "什麼是 MOE 等級？",
        "MOE（Measure of Effectiveness）是台北市 VD 系統的服務水準指標：\n\n"
        "- **MOE 0**：暢通（車速高、佔有率低）\n"
        "- **MOE 1**：緩行（車速中等、佔有率上升）\n"
        "- **MOE 2**：壅塞（車速低、佔有率高）\n\n"
        "MOE 由台北市原始 API 直接給出，本系統不重新計算。"
    ),
    (
        "TrafficVision 系統架構是什麼？",
        "TrafficVision 包含三層：\n\n"
        "1. **資料層**：從台北市 Open Data VD API 抓即時車流，SUMO 模擬產生訓練資料\n"
        "2. **分析層**：pair-based GRU（15 步 → 15 步，5 分鐘 horizon）預測車流；"
        "traffic_light_optimizer 並行模擬 5 個號誌策略，依 composite_score 選最佳\n"
        "3. **視覺層**：FastAPI + Leaflet 熱力圖 + 即時車流面板 + 本 AI 助理"
    ),
    (
        "GRU 模型如何預測車流？",
        "目前部署的是 **pair-based GRU**：訓練樣本是「同日相隔 3–15 分鐘的兩份 SUMO CSV 對」"
        "(`(CSV_A 前 15 步, CSV_B 前 15 步)`)，學習「現在的車流分佈 → 5 分鐘後的車流分佈」。"
        "輸入維度為 `num_edges + 3`（多了 gap_minutes 欄位），輸出為 `(15 步, num_edges)`，"
        "對應未來 5 分鐘各路段每 20 秒一筆預測。"
    ),
    (
        "如何查看即時車流？",
        "請查看儀表板左側的「車流監測」面板，或切換到地圖的「當前車流」模式 —"
        "顏色越紅代表佔有率越高、車速越低。"
    ),
    (
        "系統多久更新一次？",
        "runtime_pipeline.py 預設每 5 分鐘執行一輪：抓 VD → 產生 routes → SUMO 模擬 → "
        "GRU 預測 → 號誌策略並行評估 → 輸出 handoff。實際數值看排程設定。"
    ),
    (
        "佔有率代表什麼意思？",
        "佔有率（Occupancy）= 路段上車輛佔用感應器的時間比例（%）。值越高表示車流越密集，"
        "與壅塞呈正相關。經驗值：> 30% 需注意、> 50% 為嚴重壅塞，但實際門檻依路段而異。"
    ),
    (
        "號誌優化是怎麼做的？",
        "流程如下：\n\n"
        "1. 以 GRU 預測 5 分鐘後車流作為 input\n"
        "2. 並行模擬 5 個策略（no_control 作為 baseline，及 baseline_original、"
        "baseline_more_edges、baseline_more_edges_more_tls、adaptive 4 個主動策略）\n"
        "3. 計算 **composite_score = 0.6 × (waiting_time_ratio) + 0.4 × (time_loss_ratio)**，"
        "其中 ratio = strategy / baseline；分數 < 1 代表優於基準\n"
        "4. 選 composite_score 最小者；若所有方案皆 ≥ 1，採用 no_control\n\n"
        "高風險路口在 `UNSAFE_TLS_IDS` 中，永不被優化器調整。"
    ),
    (
        "如何解讀熱力圖顏色？",
        "顏色對應 SUMO 模擬輸出的車速：\n\n"
        "- **綠色**：車速高（接近自由流）\n"
        "- **黃色**：車速中等\n"
        "- **紅色**：車速低（壅塞）\n\n"
        "線條粗細表示車流量。實際門檻看 generate_edge_traffic.py 的計算。"
    ),
    (
        "什麼情況下系統會建議調整號誌？",
        "當 GRU 預測顯示 congestion_ratio（壅塞 edge 比例）偏高、temporal_ratio（時間變異）大時，"
        "adaptive 策略會自動上調 top_n_tls / proportional_pool / top_edge_count，"
        "並縮短 update_interval（最短 60 秒）。最終是否採用仍由 composite_score 決定 —"
        "若優化方案沒能贏過 no_control 基準，系統就不會建議調整。"
    ),
    (
        "SUMO 模擬在系統中扮演什麼角色？",
        "兩個角色：\n\n"
        "1. **訓練資料生成**：`VehicleData.py` 並行（16 worker）模擬大量歷史車流 → CSV 訓練 pair-based GRU\n"
        "2. **策略驗證**：`traffic_light_optimizer.py` 用 ProcessPoolExecutor 並行跑 5 個號誌策略，"
        "由 SUMO 真實算出每個方案的等待時間與時間損失，再依 composite_score 選最佳"
    ),
    (
        "為什麼有時候採用 no_control？",
        "no_control 代表系統嘗試了所有候選策略，但都無法在 composite_score 上明顯超越基準。"
        "常見情境：低車流時段、車流分佈均勻、預測 horizon 內車流變化平緩。"
        "此時主動調整可能反而擾亂相鄰路口的綠燈協調，所以「不動」是最佳解。"
    ),
    (
        "composite_score 怎麼算？分數高好還是低好？",
        "**分數越低越好**。公式：\n\n"
        "`composite_score = 0.6 × (策略 avg_waiting_time / baseline) + "
        "0.4 × (策略 avg_time_loss / baseline)`\n\n"
        "- composite_score < 1.0：策略優於基準\n"
        "- composite_score = 1.0：與基準相同（通常是 no_control）\n"
        "- composite_score > 1.0：策略反而更糟（會被淘汰）"
    ),
]


# ─── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Gemma 4 fine-tune dataset from real handoff data")
    parser.add_argument("--out", type=str, default=str(OUT_FILE), help="輸出 JSONL 路徑")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible sampling")
    parser.add_argument("--target-balance", action="store_true", default=True,
                        help="downsample snapshot bucket so it doesn't dominate (default on)")
    parser.add_argument("--no-balance", dest="target_balance", action="store_false",
                        help="disable balance sampling, keep all generated records")
    args = parser.parse_args()

    random.seed(args.seed)

    # 各類別分開生成，最後再做 balanced sampling，避免 snapshot 壓垮 strategy
    snapshot_records, prediction_records, strategy_records, general_records = [], [], [], []

    # 1. 即時車流 snapshot
    traffic_files = load_all_traffic_files()
    print(f"載入 {len(traffic_files)} 個 traffic JSON")
    for fname, timestamp, traffic in traffic_files:
        snapshot_records.extend(make_traffic_records(fname, timestamp, traffic))

    # 2. handoff 資料（真實 optimizer 輸出）
    handoff_dirs = all_handoff_dirs()
    print(f"載入 {len(handoff_dirs)} 個 handoff 目錄")
    for handoff in handoff_dirs:
        prediction_records.extend(make_prediction_records(handoff))
        strategy_records.extend(make_strategy_records(handoff))

    # 3. 通用知識
    for q_text, a_text in GENERAL_QAS:
        general_records.append(qa(q_text, a_text))

    # ── Balanced sampling: snapshot 太多會把 strategy/prediction 蓋掉
    # 目標分布: snapshot 40%, strategy 35%, prediction 15%, general 10%
    # 以 strategy 數量為錨點來推算 snapshot 的目標數
    if args.target_balance and strategy_records:
        target_strategy = len(strategy_records)
        # snapshot 上限 = strategy × (40/35)
        snapshot_cap = int(target_strategy * 40 / 35)
        if len(snapshot_records) > snapshot_cap:
            print(f"  Balance: downsample snapshot {len(snapshot_records)} → {snapshot_cap}")
            snapshot_records = random.sample(snapshot_records, snapshot_cap)

    records = snapshot_records + prediction_records + strategy_records + general_records
    random.shuffle(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 統計摘要（協助看 class balance）
    print(f"\n生成 {len(records)} 筆 Q&A → {out_path}")
    # 簡易類別計數：用問題首詞分桶
    from collections import Counter
    bucket = Counter()
    for r in records:
        q = r["conversations"][1]["value"]
        if "MOE" in q or "塞" in q or "車速" in q or "車流狀況" in q:
            bucket["snapshot"] += 1
        elif "預測" in q:
            bucket["prediction"] += 1
        elif "策略" in q or "號誌" in q or "優化" in q or "紅綠燈" in q:
            bucket["strategy"] += 1
        else:
            bucket["general"] += 1
    print("\nClass balance:")
    for k, v in bucket.most_common():
        pct = v / len(records) * 100
        print(f"  {k:12s}: {v:5d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
