# Per-Edge Speed 整合手冊

## 背景

`VehicleData.py` 已新增 `avg_speed_kmh` 欄位，每 20 秒從 TraCI 取每個 edge 上所有車輛的平均速度（單位 km/h）。新的 CSV schema：

```
time, edge_id, vehicle_count, avg_speed_kmh
60, 30620492#0, 15, 32.45
60, -E0, 7, 28.10
80, 30620492#0, 12, 35.20
...
```

**目前狀態（2026-05-24）：**
- 此變更**只影響 `data/simulation_data/`（訓練資料）**
- **runtime_pipeline 那條路徑、前端 dashboard 都還沒用上這個新欄位**
- 訓練模型本身仍只用 `vehicle_count`，需要時可以把 `avg_speed_kmh` 當作 input feature 之一

---

## 整體資料流（目前 vs 目標）

### 目前（前端看到的速度來源）

```
SUMO 模擬
   ↓ (--additional-files edgedata.add.xml, period="300")
data/edgedata_baseline.xml  ← 5 分鐘區間,每 edge 一個 mean speed (m/s)
   ↓ tools/generate_edge_traffic.py
      - 取「最忙時間窗」(busiest interval)
      - speed × 3.6 → km/h
      - 套用「VD 校準」(用 VD 真實速度錨定)
   ↓
TVDS/data/edge_heatmap_baseline.json  ← spd 欄位 = SUMO 5min 平均速 × 校準
   ↓ serve_api.py /api/roads/forecast
   ↓ dashboard.html PredictPanel
   → 路段 → 預測車速顯示
```

**問題**：
- SUMO 的 mean speed 是「5 分鐘區間平均」，會把號誌停等都平均進去
- 所以 SUMO 速度比 VD 點測速度低很多（~1.5x），需要 hard-coded 校準係數
- 校準是粗糙的「全路段同一比例」，無法反應 edge 間動態差異

### 目標（用新的 per-edge 即時速度）

```
SUMO 模擬
   ↓ TraCI 每 20 秒採樣
   ↓ traci.edge.getLastStepMeanSpeed(edge) × 3.6
runtime CSV (含 avg_speed_kmh)  ← 每 edge 每 20 秒一筆瞬時平均速度
   ↓ tools/generate_edge_traffic.py (要改:從 CSV 算 speed)
   ↓
edge_heatmap.json  ← spd 欄位 = TraCI 採樣平均(更精細)
   ↓ 前端
   → 路段速度 (不需校準係數,因為直接取 TraCI 真實值)
```

---

## 需要改的 4 處

### Step 1: `tools/traffic_optimizer_io.py` — runtime SUMO 模擬也要採速度

`runtime_pipeline.py` 走的是 `tools/traffic_optimizer_io.py` 的 `run_sumo_simulation_with_end_time()`，目前**只寫 `vehicle_count`**。

需要照 `VehicleData.py` 的做法，加上：

```python
# 在 SUMO loop 內
if current_time % 20 == 0:
    for edge in traci.edge.getIDList():
        count = traci.edge.getLastStepVehicleNumber(edge)
        if count > 0:
            mean_speed_ms = traci.edge.getLastStepMeanSpeed(edge)
            avg_speed_kmh = mean_speed_ms * 3.6
            data_buffer.append((current_time, edge, count, avg_speed_kmh))
```

CSV header 改成：
```python
f.write("time,edge_id,vehicle_count,avg_speed_kmh\n")
```

**影響檔案位置**：搜 `tools/traffic_optimizer_io.py` 內所有 `traci.edge.getLastStepVehicleNumber` 並對應的 CSV 寫檔處。

**下游相容性**：
- `train_model.py` / `predict_to_csv.py` 用具名欄位讀,多 1 欄不影響 ✓
- `traffic_light_optimizer.py` 同上 ✓
- 已存在的舊 CSV (沒有 speed 欄位) 仍能正常讀,只是聚合時 speed 為 0

---

### Step 2: `tools/generate_edge_traffic.py` — 從新 CSV 算 speed,取代 edgedata

**位置**：L98-142 的「Traffic metrics from edgedata_output.xml」段落

**現有邏輯**（取最忙 interval 的 SUMO edge speed）：
```python
edge_metrics[eid] = {
    'spd': round(_f('speed') * 3.6, 1),    # ← SUMO edgedata 5min 平均
    ...
}
```

**改成從 runtime CSV 算 speed**：
```python
import pandas as pd

# 同時讀 SUMO edgedata (給 vol/occ/wait/tloss) 和 runtime CSV (給 speed)
# 找對應的 runtime CSV
runtime_csv = ...  # 例如從 stem 推斷,或從參數傳入

if os.path.exists(runtime_csv):
    df = pd.read_csv(runtime_csv)
    if 'avg_speed_kmh' in df.columns:
        # 對每個 edge 取所有 timestep 平均 (只取有車的時刻,因為 count==0 不在 CSV 內)
        speed_per_edge = df.groupby('edge_id')['avg_speed_kmh'].mean()
        for eid in edge_metrics:
            if eid in speed_per_edge.index:
                edge_metrics[eid]['spd'] = round(float(speed_per_edge[eid]), 1)
                # 標記這個 spd 是「直接量測」而非「校準後」
                edge_metrics[eid]['spd_source'] = 'traci_direct'
```

**為什麼這樣比較準**：
- TraCI 的 `getLastStepMeanSpeed` 是「該 simulation step 上 edge 上所有車輛的算數平均速度」
- 過濾掉 count==0 的 timestep 後,代表「有車時候的平均速度」
- 不會被「沒車的 5 分鐘」拉平,跟 VD 點測語意更接近

---

### Step 3: 移除或調整「VD 校準」邏輯（可選）

**位置**：`tools/generate_edge_traffic.py` L202-???（「VD 校準」段落）

現有：因為 SUMO speed 太低,套用一個 VD/SUMO 比例係數放大。

**新方案**：
- 如果 Step 2 直接用 TraCI 速度,**這個校準係數可能就不需要了**（甚至會反過來太高）
- 建議先**保留校準邏輯但加開關**，跑一輪看新 speed 跟 VD 對比：

```python
USE_TRACI_SPEED = True  # 新功能旗標
ENABLE_VD_CALIBRATION = not USE_TRACI_SPEED  # TraCI 已直測,不需校準
```

跑 2-3 次 runtime_pipeline 後,看 dashboard 顯示速度是否合理（NTUT 區白天約 25-45 km/h, 凌晨 35-55 km/h）。
- 若合理 → 移除校準
- 若偏高 → 改回校準係數但用較小的乘數（例如 0.8x 而非 1.5x）

---

### Step 4: 前端視覺微調（選擇性）

`edge_heatmap.json` schema 不變（仍然有 `spd` 欄位），所以**前端讀檔不用改**。
但可以加一個 metadata 標記讓使用者知道資料來源：

**`tools/generate_edge_traffic.py` 最末段**：
```python
return {
    'meta': {
        ...
        'speed_source': 'traci_direct' if USE_TRACI_SPEED else 'sumo_edgedata_calibrated',
        'speed_unit': 'km/h',
    },
    'edges': edges_out,
}
```

**`dashboard.html`** 在 PredictPanel 表頭可加小字標籤：
```jsx
<span>GRU 5 分鐘流量預測</span>
<span style={{fontSize:10, color:'#4e6080'}}>
  {heatmap.meta.speed_source === 'traci_direct' ? '速度: 直測' : '速度: 校準'}
</span>
```

---

## 整合測試清單

按順序執行,任一步驟失敗回到該步驟修正：

### 5.1 確認 runtime CSV 有 speed 欄位
```bash
python tools/runtime_pipeline.py --once
ls data/runtime_data/<latest>/
head -3 data/runtime_data/<latest>/traffic_data_*.csv
# 應該看到 4 欄: time,edge_id,vehicle_count,avg_speed_kmh
```

### 5.2 確認 edge_heatmap.json 的 spd 改用 TraCI 直測
```bash
python -c "
import json
with open('TrafficVision Design System/data/edge_heatmap_baseline.json') as f:
    d = json.load(f)
print('speed_source:', d.get('meta', {}).get('speed_source'))
sample = next(iter(d['edges'].values()))
print('範例 edge spd:', sample.get('spd'), 'km/h')
"
```

### 5.3 對比新舊 spd 是否合理
跑兩次 runtime,一次用舊邏輯 (USE_TRACI_SPEED=False),一次用新邏輯,比 `edge_heatmap.json` 內幾個重點 edge 的 spd：

| edge | VD 真實速 (km/h) | 舊 SUMO+校準 | 新 TraCI 直測 |
|------|-----------------|-------------|---------------|
| (主要幹道) | 35-45 | ? | ? |
| (次要道路) | 20-30 | ? | ? |

新 TraCI 應該**比舊邏輯更接近 VD**（差距 < 10%）才合格。

### 5.4 前端顯示驗證
- 打開 dashboard → 「5 分鐘預測」分頁
- 看「預測車速」欄位數字是否合理（25-50 km/h 區間）
- 若顯示 70+ km/h 代表 TraCI 速度沒考慮停等（理論上 TraCI 平均會降）→ 檢查 Step 2 是否正確過濾 count==0
- 若顯示 0-5 km/h 代表速度全被低速車輛拉低 → 檢查單位是否誤用 m/s 而非 km/h

---

## Rollback 程序

任一步驟失敗：

```bash
# 回到 SUMO edgedata 速度
# Step 2 的 generate_edge_traffic.py 加旗標:
USE_TRACI_SPEED = False

# Step 1 的 traffic_optimizer_io.py 不影響速度顯示,
# 多寫一欄 CSV 不會壞任何東西,可保留。
# 若要完全 rollback,把 CSV header 改回 "time,edge_id,vehicle_count" 即可,
# 但建議保留新欄位,只是不被消費。
```

---

## 為什麼沒有立即動下游

`VehicleData.py` 跑一輪是「**準備新一批訓練資料**」級別的工作，影響 21K 個 CSV。  
前端整合是「**runtime 與顯示層**」工作，影響使用者體驗。

兩者**目的不同、可獨立進度**：
- 你可以先讓 VehicleData.py 帶 speed 跑訓練資料生成 → 訓練新模型 (含 speed feature)
- 之後再決定要不要把 runtime / 前端也接上

短期內 (1-2 週)：
- ✅ VehicleData.py 已改完（這次提交）
- ⏳ Step 1-2 (runtime + generate_edge_traffic) 等下次有空再做
- ⏳ Step 3-4 (校準調整 + 前端) 跑通後再評估

---

## 附錄：CSV 欄位含義速查

| 欄位 | 含義 | 單位 | 範例 |
|------|------|------|------|
| `time` | SUMO 模擬時間（從 0 起算） | 秒 | 60.0 |
| `edge_id` | SUMO 路網 edge ID | 字串 | `30620492#0` |
| `vehicle_count` | 該 timestep 該 edge 上的車輛數（瞬時） | 輛 | 15 |
| `avg_speed_kmh` | 該 timestep 該 edge 上所有車的平均速度 | km/h | 32.45 |

**注意**：
- 只有 `vehicle_count > 0` 的 row 才會被寫入（CSV 是稀疏格式）
- 同一個 `(time, edge_id)` 只會出現一次
- 速度是「同 timestep 上各車速度的算數平均」，**不是該 edge 整段時間的平均**
- 若要算「edge 整段時間的平均速度」，需用 `groupby('edge_id')['avg_speed_kmh'].mean()`，但要注意只算了「有車」的時間點，會比真實平均高（因為塞車期間車多但速度低）
