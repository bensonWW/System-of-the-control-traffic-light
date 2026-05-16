#!/usr/bin/env python3
"""
從 handoff CSV 與即時車流 JSON 自動生成 Gemma 4 指令微調資料集。
輸出格式：data/finetune_dataset.jsonl（Unsloth chat template）

執行方式：
    python tools/generate_finetune_dataset.py
"""
import json, glob, os, random
from pathlib import Path

BASE = Path(__file__).parent
ROOT = BASE.parent
TRAFFIC_DIR = ROOT / "TrafficVision Design System" / "data" / "trafficData"
RUNTIME_DIR = ROOT / "data" / "runtime_data"
OUT_FILE    = ROOT / "data" / "finetune_dataset.jsonl"

SYSTEM_PROMPT = (
    "你是 TrafficVision AI 助理，專門分析台北市北科大周邊路網的即時車流、"
    "GRU 預測與號誌優化結果。請使用繁體中文回答，數據需引用具體數值，"
    "建議要有依據，避免含糊描述。"
)
MOE_LABELS = {0: "暢通", 1: "緩行", 2: "壅塞"}


def latest_file(pattern):
    files = sorted(glob.glob(str(pattern)))
    return files[-1] if files else None


def latest_handoff():
    dirs = sorted(glob.glob(str(RUNTIME_DIR / "*" / "handoff")))
    return Path(dirs[-1]) if dirs else None


def load_traffic():
    f = latest_file(TRAFFIC_DIR / "*.json")
    if not f:
        return {}
    with open(f, encoding="utf-8") as fp:
        raw = json.load(fp)
    return raw.get("data", {})


def load_csv(path):
    try:
        import pandas as pd
        return pd.read_csv(path).where(lambda df: df.notna(), None).to_dict(orient="records")
    except Exception:
        return []


def qa(question, answer):
    return {"conversations": [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human",  "value": question},
        {"from": "gpt",    "value": answer},
    ]}


records = []

# ── 1. 即時車流 Q&A ───────────────────────────────────────────────────────────
traffic = load_traffic()
if traffic:
    sorted_by_moe = sorted(traffic.items(), key=lambda x: -x[1].get("MOELevel", 0))

    for road, v in traffic.items():
        spd = v.get("AvgSpd", 0)
        vol = v.get("TotalVol", 0)
        occ = v.get("AvgOcc", 0)
        moe = v.get("MOELevel", 0)
        lbl = MOE_LABELS.get(moe, "未知")

        a = (
            f"根據最新監測數據，**{road}** 目前處於 **{lbl}（MOE {moe}）** 狀態。\n\n"
            f"- 平均車速：**{spd:.1f} km/h**\n"
            f"- 車流量：**{vol} 輛**\n"
            f"- 佔有率：**{occ:.1f}%**\n\n"
        )
        if moe == 2:
            a += "建議調整該路段號誌時制，延長主幹道綠燈時間以疏解壅塞。"
        elif moe == 1:
            a += "車流偏緩，可持續觀察後續變化。"
        else:
            a += "車流通暢，無需特別處置。"

        for q_tmpl in [
            f"請問{road}目前的車流狀況？",
            f"{road}現在塞車嗎？",
            f"目前{road}的平均車速是多少？",
            f"{road}的MOE等級是什麼？",
            f"請描述{road}的即時交通狀態。",
        ]:
            records.append(qa(q_tmpl, a))

    # 最嚴重路段
    worst_road, worst_v = sorted_by_moe[0]
    records.append(qa(
        "目前哪條路段最嚴重？",
        f"目前壅塞最嚴重的是 **{worst_road}**，MOE 等級 {worst_v.get('MOELevel')}，"
        f"平均車速僅 {worst_v.get('AvgSpd', 0):.1f} km/h，"
        f"佔有率 {worst_v.get('AvgOcc', 0):.1f}%。"
    ))
    records.append(qa(
        "哪裡塞車？",
        f"目前最嚴重的壅塞發生在 **{worst_road}**，"
        f"車速 {worst_v.get('AvgSpd', 0):.1f} km/h（MOE {worst_v.get('MOELevel')}）。"
    ))

    # 整體概況
    total = len(traffic)
    congested = sum(1 for v in traffic.values() if v.get("MOELevel") == 2)
    slow      = sum(1 for v in traffic.values() if v.get("MOELevel") == 1)
    records.append(qa(
        "目前整體交通狀況如何？",
        f"監測 **{total}** 個路段，其中：\n\n"
        f"- 壅塞（MOE 2）：**{congested}** 個\n"
        f"- 緩行（MOE 1）：**{slow}** 個\n"
        f"- 暢通（MOE 0）：**{total - congested - slow}** 個\n\n"
        f"最嚴重路段：**{worst_road}**（車速 {worst_v.get('AvgSpd', 0):.1f} km/h）。"
    ))

# ── 2. 預測 Q&A ───────────────────────────────────────────────────────────────
handoff = latest_handoff()
if handoff:
    pred_file = latest_file(handoff / "*predict.csv")
    if pred_file:
        pred = load_csv(pred_file)
        if pred:
            spd_key = next((k for k in pred[0] if "speed" in k.lower()), None)
            vol_key = next((k for k in pred[0] if "volume" in k.lower() or "vol" in k.lower()), None)
            road_key = next((k for k in pred[0] if "road" in k.lower() or "edge" in k.lower()), None)

            top = sorted(pred[:10], key=lambda r: -(float(r.get(vol_key) or 0) if vol_key else 0))
            summary_lines = []
            for r in top[:5]:
                road_name = r.get(road_key, "?") if road_key else "?"
                spd_val   = float(r.get(spd_key) or 0) if spd_key else 0
                vol_val   = r.get(vol_key, 0) if vol_key else 0
                summary_lines.append(
                    f"- **{road_name}**：預測車速 {spd_val:.1f} km/h，車流量 {vol_val} 輛"
                )
            summary = "\n".join(summary_lines) if summary_lines else "（資料待更新）"

            for q_text in [
                "5分鐘後各路段車流預測如何？",
                "未來5分鐘的車流預測結果？",
                "GRU模型預測下一時段車流情況？",
            ]:
                records.append(qa(
                    q_text,
                    f"根據 GRU Seq2Seq 模型（15步×20秒=5分鐘）預測，未來車流如下：\n\n"
                    f"{summary}\n\n以上為集成預測，請結合即時數據綜合判斷。"
                ))

    # 號誌計畫 Q&A
    sig_file = latest_file(handoff / "*signal_plan_summary*.csv")
    if sig_file:
        signals = load_csv(sig_file)
        if signals:
            tl_key    = next((k for k in signals[0] if "tl" in k.lower() or "intersection" in k.lower()), None)
            strat_key = next((k for k in signals[0] if "strat" in k.lower()), None)
            green_key = next((k for k in signals[0] if "green" in k.lower()), None)

            sig_lines = []
            for r in signals[:4]:
                tl    = r.get(tl_key, "?") if tl_key else "?"
                strat = r.get(strat_key, "?") if strat_key else "?"
                green = r.get(green_key, "?") if green_key else "?"
                sig_lines.append(f"- **{tl}**：策略 {strat}，綠燈時間 {green} 秒")

            sig_summary = "\n".join(sig_lines) if sig_lines else "（資料待更新）"
            for q_text in [
                "號誌優化建議是什麼？",
                "最佳號誌策略為何？",
                "目前的號誌優化方案？",
            ]:
                records.append(qa(
                    q_text,
                    f"根據多策略優化（composite score = 0.6×等待時間 + 0.4×時間損失），"
                    f"建議號誌調整如下：\n\n{sig_summary}"
                ))

    # 對比 Q&A
    comp_file = latest_file(handoff / "*comparison_summary*.csv")
    if comp_file:
        comp = load_csv(comp_file)
        if comp:
            metrics = {r.get("metric", "?"): r for r in comp if isinstance(r, dict)}

            def _m(key, field, default="?"):
                return metrics.get(key, {}).get(field, default)

            wait_b = _m("avg_waiting_time", "before")
            wait_a = _m("avg_waiting_time", "after")
            wait_d = _m("avg_waiting_time", "delta")
            loss_b = _m("avg_time_loss",    "before")
            loss_a = _m("avg_time_loss",    "after")

            for q_text in [
                "號誌優化前後差異多少？",
                "優化效果如何？等待時間降低多少？",
                "採用最佳策略後改善了多少？",
            ]:
                records.append(qa(
                    q_text,
                    f"優化前後對比摘要：\n\n"
                    f"- **平均等待時間**：{wait_b}s → {wait_a}s（改善 {wait_d}）\n"
                    f"- **平均時間損失**：{loss_b}s → {loss_a}s\n\n"
                    f"採用最佳號誌策略後，整體交通效率顯著提升，建議持續套用最佳方案。"
                ))

# ── 3. 通用知識 Q&A ───────────────────────────────────────────────────────────
general_qas = [
    (
        "什麼是MOE等級？",
        "MOE（Measure of Effectiveness）是交通服務水準指標：\n\n"
        "- **MOE 0**：暢通（車速高、佔有率低）\n"
        "- **MOE 1**：緩行（車速中等、開始擁擠）\n"
        "- **MOE 2**：壅塞（車速低、佔有率高，需介入）"
    ),
    (
        "TrafficVision系統架構是什麼？",
        "TrafficVision 系統包含三層：\n\n"
        "1. **資料層**：從台北市 Open Data API 即時抓取 VD 感應器數據\n"
        "2. **分析層**：GRU Seq2Seq 預測未來5分鐘車流，SUMO 模擬優化號誌時制\n"
        "3. **視覺層**：Leaflet 地圖熱力圖 + 即時車流面板 + AI 問答助理"
    ),
    (
        "GRU模型如何預測車流？",
        "GRU Seq2Seq 模型以 20 秒為步長，輸入最近 10 個時間視窗（滑動集成），"
        "預測未來 15 步（共 5 分鐘）的各路段車速、車流量與佔有率。"
        "使用 Log1p 縮放與 expm1 逆變換以處理長尾分布。"
    ),
    (
        "如何查看即時車流？",
        "請查看儀表板左側的「車流監測」面板，或切換至「當前車流」模式的地圖，"
        "顏色越紅表示壅塞程度越高。"
    ),
    (
        "系統多久更新一次？",
        "即時車流數據每 5 分鐘從台北市 Open Data VD API 更新一次，"
        "GRU 預測與號誌優化由 runtime_pipeline.py 每 5 分鐘自動執行一輪。"
    ),
    (
        "佔有率代表什麼意思？",
        "佔有率（Occupancy）是指路段上車輛佔用感應器的時間比例（%）。"
        "佔有率越高代表車流越密集，通常與壅塞呈正相關。"
        "一般而言，佔有率超過 30% 即需注意，超過 50% 則為嚴重壅塞。"
    ),
    (
        "號誌優化是怎麼做的？",
        "號誌優化使用 SUMO 模擬搭配多策略搜索：\n\n"
        "1. 以 GRU 預測結果作為輸入車流\n"
        "2. 在各路口嘗試不同綠燈時間方案\n"
        "3. 以 composite score（0.6×等待時間 + 0.4×時間損失）評分\n"
        "4. ProcessPoolExecutor 並行計算，取最佳方案\n\n"
        "部分高風險路口（UNSAFE_TLS_IDS）被排除在外以確保安全。"
    ),
    (
        "如何解讀熱力圖顏色？",
        "地圖熱力圖顏色代表：\n\n"
        "- **綠色**：暢通（車速 ≥ 40 km/h）\n"
        "- **黃色**：緩行（20–40 km/h）\n"
        "- **紅色**：壅塞（< 20 km/h）\n\n"
        "線條粗細代表車流量，顏色深淺代表速度快慢。"
    ),
]
for q_text, a_text in general_qas:
    records.append(qa(q_text, a_text))

# ── 輸出 ──────────────────────────────────────────────────────────────────────
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
random.shuffle(records)
with open(OUT_FILE, "w", encoding="utf-8") as fp:
    for r in records:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"生成 {len(records)} 筆 Q&A → {OUT_FILE}")
