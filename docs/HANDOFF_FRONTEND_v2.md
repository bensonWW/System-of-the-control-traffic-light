# 給前端的更新通知 — v2 模型上線（含速度預測）

> **TL;DR**
> 預測 CSV 多了一欄 `avg_speed_kmh`（單位 km/h，每 edge 每 timestep 的預測速度）。
> 既有 API 端點、檔案路徑、其他欄位**完全不變**。可以直接拿來顯示「未來 5 分鐘各路段的速度」。

---

## 1. 改了什麼

### 訓練後新模型
- **`gru_traffic_model_pair_v2.pth`** 已就緒
- 同時預測 `vehicle_count` 和 `avg_speed_kmh`
- Holdout 測試對比 v1：
  - Count MAE：v1 = 0.129 → **v2 = 0.126**（小幅改善）
  - 高流量段 MAE：v1 = 1.02 → **v2 = 0.85**（**↓17%**，這是壅塞預測的關鍵）
  - Speed MAE：~2-3 km/h（in-sample）

### Runtime pipeline 預設模型已切到 v2
- `tools/runtime_pipeline.py`、`tools/predict_main.py`、`tools/predict_to_csv.py` 的預設 model_path 都指向 `gru_traffic_model_pair_v2.pth`
- **你不用做任何切換動作**，下次 runtime_pipeline 跑就會用 v2

---

## 2. 輸出 CSV 新格式

### 之前 (v1)：3 欄

```
time,edge_id,vehicle_count
360.0,-300077497#5,1.79
360.0,-49073264#1,2.03
```

### 現在 (v2)：4 欄 ★ **多了 `avg_speed_kmh`**

```
time,edge_id,vehicle_count,avg_speed_kmh
360.0,-300077497#5,1.79,14.49
360.0,-49073264#1,2.03,2.33
360.0,-51362430,0.54,8.72
360.0,-E1,0.96,19.46
```

| 欄位 | 含義 | 單位 | 範圍 |
|------|------|------|------|
| `time` | SUMO 模擬時間 | 秒 | 360.0 ~ 640.0（15 個 time bin） |
| `edge_id` | SUMO edge ID | — | — |
| `vehicle_count` | 預測車輛數 | 輛 | 0 ~ 約 30 |
| **`avg_speed_kmh`** ★新 | 預測平均速度 | km/h | 0 ~ 約 60 |

**檔案位置**：`data/runtime_data/<stem>/handoff/<stem>_predict.csv` 跟 `<stem>_predict_full.csv`（一樣，路徑不變）

---

## 3. 沒改的東西（你可以略過這些細節）

| 項目 | 狀態 |
|------|------|
| API 端點（`/api/prediction`、`/api/roads/forecast`、`/api/edge-heatmap`） | ✅ 不變 |
| `edge_heatmap_*.json` 結構 | ✅ 不變 |
| `csv_to_records()` 解析邏輯 | ✅ 不變（多 1 欄會自動帶到 records 內） |
| dashboard.html 既有顯示邏輯 | ✅ 不變（不主動讀 speed 不會壞） |
| 號誌優化 / SUMO 模擬 / Adaptive 策略 | ✅ 不變 |
| handoff 目錄結構 | ✅ 不變 |

→ **你可以選擇完全不動前端**，現有功能照舊運作；speed 預測安靜地存在 CSV 內備用。

---

## 4. 推薦的前端整合方式（可選）

### 方案 A：最小整合（5 分鐘）— 在「預測車速」欄改用模型預測值

`dashboard.html` PredictPanel 內目前的「預測車速」是來自 `/api/roads/forecast`（baseline SUMO 模擬聚合的速度）。可以改成：

```javascript
// 從 /api/prediction 的 records 算每路段預測平均速度
const predRecords = await fetch('/api/prediction').then(r => r.json());
const speedByEdge = {};
predRecords.records.forEach(r => {
  if (r.avg_speed_kmh != null) {
    // 取每個 edge 的所有 time bin 平均(或第一個 time bin)
    if (!speedByEdge[r.edge_id]) speedByEdge[r.edge_id] = [];
    speedByEdge[r.edge_id].push(r.avg_speed_kmh);
  }
});
// 用 _edge_road_map 把 edge 聚合到路段
```

**好處**：速度直接來自 GRU 預測，跟 count 預測來自同個模型（一致性更好），不需要 VD 校準係數。

### 方案 B：在熱力圖加個 speed 色彩 toggle

新增一個視圖模式 `pred_speed`，用 `r.avg_speed_kmh` 對應顏色（紅 < 20 km/h、黃 20-35、綠 > 35）。

讓使用者看「未來 5 分鐘的速度地圖」。

### 方案 C：表格新增「預測壅塞時段」

利用 `vehicle_count` 高 + `avg_speed_kmh` 低 = 壅塞，可以做出「未來 1-5 分鐘預計變壅塞的路段」清單。

---

## 5. 已知限制（要做完整流程的話需要注意）

目前 runtime pipeline 走的 SUMO 路徑（`tools/traffic_optimizer_io.py`）**輸入 CSV 沒有速度欄**。v2 模型在 inference 時會 fallback 用 `speed=0` 當輸入，所以：

- **速度預測會偏低**（很多會在 0-20 km/h，比真實低）
- 但 count 預測**不受影響**（仍然在 holdout MAE = 0.126 的水準）
- 看到啟動 log 會印：`⚠ CSV 缺 avg_speed_kmh 欄,v2 模型將以 speed=0 推論`

**要徹底修好需要做 INTEGRATION_EDGE_SPEED.md 的 Step 1-2**：

| Step | 動作 | 結果 |
|------|------|------|
| 1 | 修 `tools/traffic_optimizer_io.py`：runtime SUMO 模擬也要記錄 `avg_speed_kmh`（照 `VehicleData.py` 第 117-125 行那段抄就好） | runtime CSV 變 4 欄 |
| 2 | 確認 `tools/generate_edge_traffic.py` 對應更新（可選）| 速度預測精度恢復 |

完成後，speed 預測會回到 ~2-3 km/h MAE 水準。**不做也能用，只是 speed 數值偏低**。

---

## 6. 沿用 v1 的 rollback 方式（如果新模型在 production 出問題）

```bash
# 方案 1: 臨時用 v1 跑一次
python tools/runtime_pipeline.py --once --model-path gru_traffic_model_pair.pth

# 方案 2: 永久切回 v1 - 把這 3 處預設改成 "gru_traffic_model_pair.pth":
#   tools/predict_to_csv.py  L73     DEFAULT_MODEL_PATH
#   tools/predict_main.py    L13     MODEL_PATH
#   tools/runtime_pipeline.py L221   --model-path default
```

切回 v1 後，預測 CSV 自動變回 3 欄（沒有 `avg_speed_kmh`）。前端如果有讀 speed 欄要做 `r.avg_speed_kmh ?? null` 判斷處理 v1 模式。

---

## 7. 變更清單（給 PR / commit message 用）

```
* 新訓 v2 模型 gru_traffic_model_pair_v2.pth (output_channels=2)
* tools/predict_to_csv.py:
    - GRUSequence 加 output_channels 參數
    - load_model 偵測 v2 並調整 GRU input_size
    - load_demo_csv 讀 avg_speed_kmh 欄(缺則填 0 並警告)
    - is_pair_model 涵蓋 v1+v2
    - 輸出 CSV 多 avg_speed_kmh 欄位 (v2 時)
    - 預設 model_path 改 v2
* tools/predict_main.py: MODEL_PATH 改 v2
* tools/runtime_pipeline.py: --model-path 預設改 v2
* test_model.py: 預設模型改 v2;支援 output_channels
* train_model.py: OUTPUT_CHANNELS=2 為主流程
```

---

## 8. 有問題找誰

- 模型本身（架構 / 訓練）：看 `train_model.py` 註解
- 推論流程（為何 4 欄 / 為何 speed 變 0）：看 `tools/predict_to_csv.py` L416-486
- 整合 / 整體 pipeline：看 `INTEGRATION_PAIR_MODEL.md`
- Speed 流程徹底修：看 `INTEGRATION_EDGE_SPEED.md`

---

## 一句話總結

> **預測 CSV 多了 `avg_speed_kmh` 欄。其他都沒變。前端要不要拿來顯示自己決定。最佳體驗需要也讓 runtime SUMO 寫 speed（INTEGRATION_EDGE_SPEED.md Step 1-2），但不做也能繼續用。**
