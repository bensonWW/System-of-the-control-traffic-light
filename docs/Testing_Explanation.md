# 交通預測模型：測試與驗證過程解析 (`test_model.py`)

這份文件詳細解釋了 `test_model.py` 的運作原理。該腳本旨在載入經過優化的序列預測模型，以精確測量其實際效能，並支援動態驗證「指定未來特定分鐘數」的準確率。

---

## 1. 核心觀念：嚴格對齊 (Alignment)
推論 (Inference) 階段最重要的原則是**必須與訓練階段的處理邏輯一模一樣**。
為了確保這一點：
1.  **相同類別 (`Log1pScaler`, `GRUSequence`)**：程式重新宣告了相同的 Class，因為 PyTorch 反序列化需要。
2.  **提取附屬檔案**：`load_model` 執行時，它不只載入模型的神經網路權重。它拆開了我們之前打包的 `.pth` 字典，取出 **`edge_ids` (路口對齊標籤)** 、 `config` (例如 `input_len`, `horizon` 長度) 和最關鍵的 **`scaler` (對數轉換器)**。

---

## 2. 記憶體保護機制 (`load_sample_data_generator`)
在處理大量的驗證資料 (如 `./data/simulation_check_data`) 時，如果一次全讀，記憶體仍然會吃緊。
*   **Python Generator (產生器)**: 程式使用了 `yield` 來設計資料讀取函式。它會讀取一個 CSV，算出 2D Pivot Table 與時間特徵 (sin/cos)，**交付出去給下游運算，算完就丟掉釋放記憶體**，再去讀下一個 CSV。
*   這保證了不管資料有幾百 GB，RAM 原則上只會佔用一個 CSV 檔案和模型需要的空間。

---

## 3. 滑動視窗推估 (Sliding Window Check)
測試迴圈是針對每個檔案做線性掃描：
```python
for i in range(0, T - input_len - pred_horizon, 5):
```
*   `stride = 5` 代表每 5 個時間步 (100 秒) 抽取一小段歷史來考模型。
*   **輸入特徵 `x`**：結合 `scaled_traffic` (經過 Log1p 的歷史車流) 和 `time_data` (週期性特徵)。送進模型神經網路。
*   **輸出 `pred_seq_scaled`**：模型吐出 `(Horizon, Edges)` 的二維預測預測目標長度，但這仍是對數空間的數字。

### Log1p 逆轉換
這是還原車輛數的關鍵：
```python
pred_seq_real = scaler.inverse_transform(pred_seq_scaled)
```
透過 `np.expm1(y)` 將對數值變回我們看得懂的**真實車輛數 (Real count)**。

---

## 4. 各項誤差指標 (Metrics Calculation)
測試程式採用 MAE (Mean Absolute Error, 平均絕對誤差) 作為直觀的準確率標準 (1.0 就代表平均猜錯 1 輛車)。

### 總體序列誤差 (Overall Sequence Metrics)
1.  **Average MAE**: 把預測出的這 15 個時間步陣列與真正的 15 步陣列相減，取全部的平均誤差。
2.  **Peak MAE (>10)**: 第一個**過濾評估 (Masked Evaluation)**。只找出「真實檔案在這個時間點，這條路上車流量 > 10 輛」的格子，針對這些重點高峰時段計算誤差。這能反映出模型「會不會壓平高峰導致塞車」。

### 重點功能：特定時間精準度攔截 (`TEST_MINUTES`)
這是應需求所添加的動態變數。可以單獨抽出預測序列中的**特定一個時間步**進行評核。
*   **設定**: 例如 `TEST_MINUTES = 3.0`。
*   **計算對應索引**: 模型是每 20 秒一步。3 分鐘等於 180 秒，除以 20 = 第 9 個步驟 (在系統中 Index = 8)。
    `target_step_idx = int(TEST_MINUTES * 60 / 20) - 1`
*   **抽出比對**:
    取出第 `target_step_idx` 項的結果，分別計算：
    1.  在這個指定未來的時間點上，所有路口的平均 MAE (`Step Avg MAE`)。
    2.  在這個指定未來的時間點上，只針對當下處於高車流 (>10輛) 區間的高峰 MAE (`Step Peak MAE`)。

透過這個功能，使用者可以直接問：**「我如果要在 3 分鐘後給綠燈，模型預測 3 分鐘後的誤差到底大不大？」**，為紅綠燈即時變換提供直接的信心評估標準。
