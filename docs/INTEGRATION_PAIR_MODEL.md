# Pair-Based GRU 模型整合手冊

本文件指引：當新的 `gru_traffic_model_pair.pth`（pair-based 5 分鐘預測模型）經 `train_model.py --mode eval-batch` 驗證表現符合預期後，如何把它接入既有的 runtime pipeline。

> **設計原則**：訓練階段（`train_model.py`）完全獨立，不影響 runtime；整合需要修改 `tools/predict_to_csv.py`（4 處）、視情況調整 `tools/traffic_light_optimizer.py`、選擇模型檔切換策略，本文件按步驟說明。

---

## Step 1 — 驗收標準

在執行整合前，先確認新模型在 `eval-batch` 上達到以下表現：

執行：
```bash
python train_model.py --mode eval-batch --dir data/simulation_data/
```

驗收條件：
- [ ] **整體 MAE**：mean ≤ 3.0、median ≤ 2.5（5 分鐘預測每邊每步車輛數誤差）
- [ ] **逐步 MAE 單調遞增**：印出的 `Monotonic per-step degradation: YES`
- [ ] **無系統性發散**：第 1 步 MAE 與第 15 步 MAE 比值 < 2.5
- [ ] **整體 RMSE**：mean ≤ 6.0
- [ ] **失敗 pair 比例**：`skipped` 數 < 5% of total pairs

對比 baseline（可選但推薦）：
```bash
# 手算「全 0 預測」的 MAE 作為 sanity baseline
# 若新模型 MAE 沒明顯低於此值,代表模型沒學到東西
```

若有任一項不達標，**不要進行 Step 2 之後的整合**，回頭調 hyperparameters（先試 `LR=1e-4` 或 `EPOCHS=120`）。

---

## Step 2 — 修改 `tools/predict_to_csv.py`

需要動四處：

### 2.1 `load_model()` — 依 `model_type` 決定 input 維度

**位置**：`tools/predict_to_csv.py` 約 L83-113 的 `load_model()` 函式內

**修改前**（節錄關鍵段）：
```python
model = GRUSequence(num_edges, hidden_dim, num_layers, pred_horizon).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
```

**修改後**：
```python
model_type = config.get("model_type", "gru_sequence_log1p")
gap_feature = config.get("gap_feature", False)
input_extra = 3 if gap_feature else 2

# GRUSequence 建構子需新增 input_extra_features 參數,值同上
model = GRUSequence(num_edges, hidden_dim, num_layers, pred_horizon,
                    input_extra_features=input_extra).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"  model_type={model_type}, gap_feature={gap_feature}")
```

同時 `GRUSequence` 類別本身也要相容，把 `input_size=num_edges + 2` 改成 `input_size=num_edges + input_extra_features`（與 `train_model.py` 內定義保持一致）。

另外 `load_model()` 的回傳值要加上 `config`（給 Step 2.4 判斷模型類別用）：
```python
return model, scaler, edge_ids, input_len, pred_horizon, config
```
並同步更新所有 `load_model()` 的呼叫點（`export_prediction_csv`、`run_prediction_pipeline` 等）。

### 2.2 `build_time_features()` — 新增選用的 `gap_minutes` 參數

**位置**：`tools/predict_to_csv.py` 約 L151-163

**修改前**：
```python
def build_time_features(file_name, num_steps):
    ...
    return np.stack([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)
```

**修改後**：
```python
def build_time_features(file_name, num_steps, gap_minutes=None):
    ...
    sin_t = np.sin(theta).astype(np.float32)
    cos_t = np.cos(theta).astype(np.float32)
    if gap_minutes is None:
        return np.stack([sin_t, cos_t], axis=1)
    gap_col = np.full(num_steps, float(gap_minutes), dtype=np.float32)
    return np.stack([sin_t, cos_t, gap_col], axis=1)
```

### 2.3 推算 `gap_for_inference` 並傳入

在 `export_prediction_csv()` 內，load 完 model 後：
```python
gap_for_inference = 5.0 if config.get("gap_feature", False) else None
```
`gap_for_inference = 5.0` 對齊 `runtime_pipeline.py --interval 300`（5 分鐘排程週期）。

### 2.4 `export_prediction_csv()` — 把滑動視窗改成單次預測 ⚠️ 關鍵

**為何必要**：新模型訓練時只見過「CSV 前 15 步（SUMO 暖機段，time 60-340s）」這種分布。若仍用滑動視窗把 CSV 中段（如 time 3000-3340s）當輸入，模型會輸出垃圾（訓練分布外）。**必須改成只用前 15 步、跑一次預測**。

**位置**：`tools/predict_to_csv.py` 的 `export_prediction_csv()` 內，原本呼叫 `_aggregate_predictions(...)` 的部分

**修改方式**：依 `config.get("model_type")` 分支：

```python
is_pair_model = config.get("model_type") == "gru_pair_log1p_v1"

if is_pair_model:
    # ============ 新模型: 單次預測 (固定用 CSV 前 input_len 步) ============
    if len(pivot) < input_len:
        raise ValueError(f"CSV 不足 {input_len} 步,無法預測")

    input_traf = scaled_traffic[:input_len]
    input_time = build_time_features(
        os.path.basename(input_csv), input_len, gap_minutes=gap_for_inference
    )
    pred_seq_real = _predict_sequence(model, device, input_traf, input_time, scaler)
    # pred_seq_real shape = (pred_horizon, num_edges)

    # 構造輸出 rows: time 接續輸入 CSV 之後 (與舊模型第一個 window 的輸出對齊)
    last_input_time = float(time_index[input_len - 1])  # 通常 ≈ 340.0
    full_rows, rows = [], []
    for step_index in range(pred_horizon):
        current_time = float(last_input_time + 20.0 * (step_index + 1))
        step_values = pred_seq_real[step_index]
        for edge_index, edge_id in enumerate(edge_ids):
            vol = round(float(step_values[edge_index]), 4)
            full_rows.append({
                "time": current_time, "edge_id": edge_id, "vehicle_count": vol,
            })
            if vol > PREDICTION_MIN_THRESHOLD:
                rows.append({
                    "time": current_time, "edge_id": edge_id, "vehicle_count": vol,
                })
    # 後續寫檔邏輯 (prediction_df / prediction_full_df) 與舊版相同

else:
    # ============ 舊模型: 維持原本的滑動視窗 ============
    control_sum, control_count = _aggregate_predictions(
        model=model, device=device, scaler=scaler,
        scaled_traffic=scaled_traffic, time_data=time_data,
        time_index=time_index, input_len=input_len, pred_horizon=pred_horizon,
        max_windows=PREDICTION_FUSION_LAST_WINDOWS,
    )
    full_sum, full_count = _aggregate_predictions(
        ...,
        max_windows=None,
    )
    # 原本的 rows / full_rows 構造邏輯
```

**為什麼用 `last_input_time + 20*(step+1)` 當輸出時間軸？**
舊模型第一個 window（`window_end = input_len`）的輸出時間就是 `time_index[input_len-1] + 20, +40, ..., +300` = 360-660s。新模型用相同 base 確保 prediction CSV 的 `time` 欄位語意與舊模型「第一段 prediction」對齊，避免下游 `traffic_light_optimizer` 的 `base_prediction_time` 行為突變。

---

## Step 3 — 號誌優化器（traffic_light_optimizer.py）調整

新模型的 prediction CSV 時間軸只涵蓋 5 分鐘（約 360-660s），比舊模型滑動視窗版本（橫跨整個 SUMO 模擬，可達 6000+s）短得多。對 `traffic_light_optimizer.py` 的影響如下：

### 3.1 影響評估

| 項目 | 舊模型行為 | 新模型行為 |
|------|------------|------------|
| Prediction time 範圍 | 360 ~ 6000+s（整輪 SUMO 都有預測） | 360 ~ 660s（僅 5 分鐘） |
| Bin 數量（預設 `update_interval=180s`） | ~30+ 個 | 1-2 個 |
| 號誌切換點 | 散落整輪模擬 | 集中在前 5 分鐘 |
| 5 分鐘後的 SUMO 模擬 | 持續切換 program | 維持最後一組 program 跑完剩下時間 |

### 3.2 是否需要修改？

**選項 A — 不改（推薦先試）**：
新模型的「5 分鐘預測對應 5 分鐘決策」符合 runtime pipeline 每 5 分鐘重跑一輪的設計本意。下游 `apply_time = current_bin - base_prediction_time` 邏輯本就用相對時間，1-2 個 bin 不會造成 crash，只是優化粒度變粗。先用 Step 5 回歸測試驗證這條路能否跑通。

**選項 B — 降低 `update_interval` 換取更細決策粒度**：
若 Step 5 發現「號誌切換太稀疏導致 adaptive/baseline 策略永遠輸給 no_control」，再改 `tools/traffic_light_optimizer.py` 約 L240-270 的 5 個策略定義，把 `update_interval` 從 `180.0` 改成 `60.0`，讓 5 分鐘內有 5 個切換點：

```python
{
    "strategy":               "baseline_original",
    "is_proportional":        True,
    "proportional_pool":      6.0,
    "top_n_tls":              3,
    "use_downstream_penalty": True,
    "update_interval":        60.0,        # ← 從 180.0 改為 60.0
    "top_edge_count_override": 12,
},
# baseline_more_edges、baseline_more_edges_more_tls 同步調整
# adaptive 策略內的計算式也建議下修:
adaptive_interval = float(max(30, int(60 - tvr * 30)))  # 原本 max(60, 180 - tvr*120)
```

### 3.3 「5 分鐘後 SUMO 行為」說明

`traffic_light_optimizer.py` 對每個策略仍執行完整 SUMO 模擬（100+ 分鐘），但號誌覆蓋只在 prediction 涵蓋的時段（前 5 分鐘）內切換。5 分鐘後，SUMO 維持最後一組覆蓋 program 跑完剩下時間。這是合理的：
- Runtime 每 5 分鐘會重跑一次，現實情境本來就 5 分鐘後重新決策
- 對「策略評估」而言，前 5 分鐘的優化效應已能反映在整輪 SUMO 模擬的 `waiting_time` / `timeLoss` 指標上

### 3.4 Rollback hooks

若 Step 3.2 選了選項 B，rollback 時記得把 `update_interval` 改回原值（180.0、180.0、180.0），否則就算改回舊模型，號誌切換頻率也會跟舊行為不一致。

---

## Step 4 — 模型檔切換策略（二選一）

### 選項 A — 直接覆寫（簡單，但失去 rollback 能力）

```bash
# 在專案根目錄
mv gru_traffic_model.pth gru_traffic_model_sliding_legacy.pth
mv gru_traffic_model_pair.pth gru_traffic_model.pth
```

之後所有 `predict_to_csv.py` 與 `runtime_pipeline.py` 預設都會用新模型。

### 選項 B — 共存（推薦，可隨時 rollback）

修改三處 `DEFAULT_MODEL_PATH` 與預設值：

**`tools/predict_to_csv.py`** 約 L60：
```python
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "gru_traffic_model_pair.pth")
```

**`tools/predict_main.py`** 約 L11：
```python
MODEL_PATH = os.path.join(ROOT_DIR, "gru_traffic_model_pair.pth")
```

**`tools/runtime_pipeline.py`** 約 L218-222（`--model-path` 預設值）：
```python
parser.add_argument(
    "--model-path",
    default=os.path.join(ROOT_DIR, "gru_traffic_model_pair.pth"),
    help="模型路徑，未指定時使用 pair-based 模型",
)
```

兩個 checkpoint 都保留，若新模型出問題只要把上述路徑改回舊檔即可立刻 rollback。

---

## Step 5 — 回歸測試清單

整合後依序執行以下驗證：

### 5.1 單次 pipeline 跑通
```bash
python tools/runtime_pipeline.py --once
```
觀察：
- [ ] Step 1/4 route 產生成功
- [ ] Step 2/4 SUMO 模擬完成
- [ ] Step 3/4 預測 + 訊號優化 + 交付輸出成功（**新模型載入無 shape mismatch 錯誤、prediction CSV 只有 15 個 time bin**）
- [ ] Step 4/4 三個 heatmap JSON 都更新
- [ ] `data/runtime_data/<stem>/handoff/` 內所有 CSV 存在

### 5.2 Prediction CSV 結構檢查
打開 `data/runtime_data/<stem>/handoff/*_predict.csv`：
- [ ] `time` 欄位 unique 值只有 15 個（不是舊模型那種上百個）
- [ ] `time` 範圍約 360-660s（依輸入 CSV 而定）
- [ ] 各 time 的 row 數合理（每個 time bin 對應 top-K 個 edge）

### 5.3 五策略並行模擬
- [ ] `traffic_light_optimizer.py` 五個策略全部跑完無 crash
- [ ] `best_strategy.csv` 內有合理的策略選擇（**不是永遠 no_control**；若是，回 Step 3.2 考慮選項 B）
- [ ] `signal_change_detail.csv` 內 `apply_time` 範圍在 0-280s 之間（前 5 分鐘）

### 5.4 前端 API
```bash
python "TrafficVision Design System/serve_api.py"
```
- [ ] `GET /api/prediction` 回傳有 records（非空）
- [ ] `GET /api/edge-heatmap` 回傳有 edges
- [ ] AI 助理 `POST /api/chat` 能正常回答

### 5.5 新舊模型策略選擇差異
跑兩次 `runtime_pipeline.py --once`，分別用新舊模型，比較：
- [ ] 兩個 handoff 目錄的 `best_strategy.csv` 是否合理（可不一樣，但不該天差地遠）
- [ ] 預測車流量級是否在同個 order of magnitude
- [ ] 新模型的 `waiting_time` / `timeLoss` 是否未大幅惡化（若惡化超過 20%，考慮 Step 3.2 選項 B）

---

## Step 6 — Rollback 程序

任一階段失敗時：

### 若採 Step 4 選項 A（直接覆寫）
```bash
mv gru_traffic_model.pth gru_traffic_model_pair_failed.pth
mv gru_traffic_model_sliding_legacy.pth gru_traffic_model.pth
```
restore 完畢，runtime 立即回到舊版行為。

### 若採 Step 4 選項 B（共存）
把 Step 4 改過的三處 `DEFAULT_MODEL_PATH` 改回 `gru_traffic_model.pth`（舊檔名）即可。

### 若 Step 3.2 選了選項 B 改了 `update_interval`
把 `tools/traffic_light_optimizer.py` 內五個策略的 `update_interval` 改回 `180.0`，並把 `adaptive_interval` 計算式還原。

### 若是 `predict_to_csv.py` 改壞了
```bash
git checkout tools/predict_to_csv.py
```
（前提：整合前已 commit 過此檔的修改）

---

## 附錄 A — 檔案異動清單

整合完成後預期會修改的檔案：
- `tools/predict_to_csv.py`（Step 2 四處修改 — load_model、build_time_features、gap 傳遞、單次預測分支）
- `tools/predict_main.py`（Step 4 選項 B 的 MODEL_PATH）
- `tools/runtime_pipeline.py`（Step 4 選項 B 的 --model-path 預設值）
- `tools/traffic_light_optimizer.py`（Step 3.2 選項 B 的 update_interval，**可選**）

整合完成後預期會新增的檔案：
- `gru_traffic_model_sliding_legacy.pth`（Step 4 選項 A）— 舊模型備份
- 或無新增檔（Step 4 選項 B，兩個 checkpoint 共存於原路徑）

整合**不會**修改：
- `tools/traffic_optimizer_io.py`
- `tools/traffic_optimizer_signal.py`
- `tools/export_handoff.py`
- `TrafficVision Design System/serve_api.py`
- `TrafficVision Design System/ui_kits/traffic-dashboard/dashboard.html`
- `train_model.py`（已是新版）

---

## 附錄 B — 新舊模型語意差異與下游影響

| 項目 | 舊模型（sliding） | 新模型（pair） |
|------|-------------------|----------------|
| 訓練資料 | 單 CSV 內滑動視窗 | 跨 CSV pair（同日 3-15 分鐘間隔） |
| 輸入 | 15 步車流 + sin/cos | 15 步車流 + sin/cos + **gap_minutes** |
| Input dim | `num_edges + 2` | `num_edges + 3` |
| 學習目標 | 「SUMO 模擬如何延伸 5 分鐘」 | 「5 分鐘後的另一輪模擬看起來怎樣」 |
| Inference 模式 | 滑動視窗 + 多 window 平均 | **單次預測（用 CSV 前 15 步）** |
| Prediction CSV time 範圍 | 360 ~ 6000+s | 360 ~ 660s（僅 5 分鐘） |
| Prediction CSV time bin 數 | 數十至數百 | 15 |
| Checkpoint config | `model_type: "gru_sequence_log1p"` | `model_type: "gru_pair_log1p_v1"` |
| Inference 預設 gap | N/A | `5.0` 分鐘（對齊 scheduler 週期） |
| 號誌切換點分布 | 整輪 SUMO 模擬 | 集中前 5 分鐘 |
| 號誌優化器是否需改 | 不需 | **建議先不改**，視 Step 5.3/5.5 結果決定 |

兩個模型的單次模型輸出 shape 完全相同 `(B, 15, num_edges)`；但是 inference 流程的呼叫方式（單次 vs 滑動）與輸出 prediction CSV 的時間軸範圍不同，**這是 Step 2.4 必要修改的核心理由**。

---

## 附錄 C — Hyperparameters 調整建議

若 Step 1 驗收未達標，依以下優先序調整 `train_model.py` 內常數：

1. **LR 太大**：`LR = 1e-4`（從 3e-4 降低）
2. **訓練不夠久**：`EPOCHS = 120, PATIENCE = 15`
3. **過擬合**：`DROPOUT = 0.3`
4. **batch 太大導致 gradient 不穩**：`BATCH_SIZE = 64`
5. **模型容量不足**：`HIDDEN_DIM = 384, NUM_LAYERS = 3`

每次調整重訓後，重跑 Step 1 驗收即可，**不需要動 `predict_to_csv.py` 或 `traffic_light_optimizer.py`**。
