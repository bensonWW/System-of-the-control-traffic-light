# 交通號誌控制系統模擬 (Traffic Control System Simulation)

本專案利用台北市開放資料 API (Taipei City Open Data API) 的即時數據，結合 **SUMO (Simulation of Urban MObility)**，模擬台北科技大學 (NTUT) 周邊區域的交通流量與號誌控制。

---

## 🚀 快速開始 (Quick Start)

### 1. 先決條件 (Prerequisites)
- 已安裝 **Python 3.x**。
- 已安裝 **SUMO** 並將其加入系統環境變數 `PATH` 中。[下載 SUMO](https://sumo.dlr.de/docs/Downloads.php)

### 2. 安裝依賴 (Installation)
安裝所需的 Python 套件：
```bash
pip install -r requirements.txt
```

### 3. 執行模擬 (Running the Simulation)

#### **步驟 1：收集與處理交通數據**
執行主數據收集腳本，抓取即時數據、進行處理，並生成 SUMO 的路由檔案。
```bash
python scripts/collect_traffic_data.py
```
- **原始數據輸出**：`data/trafficData/` (JSON 格式)
- **SUMO 路由輸出**：`data/VehicleData/` (SUMO `.rou.xml` 格式)

#### **步驟 2 (可選)：批次處理歷史數據**
若需將 `data/trafficData/` 中已有的所有 JSON 檔案批次轉換為路由檔案，可執行：
```bash
python scripts/batch_process.py
```

#### **步驟 3：啟動 SUMO 模擬**
在 SUMO-GUI 中開啟主要的模擬設定檔。
```bash
sumo-gui -c data/ntut_config.sumocfg
```

或執行整合式模擬腳本：
```bash
python run_simulation.py
```

---

## 📂 專案結構與工具說明 (Project Structure & Tools)

### 核心腳本 (`/scripts`)

| 腳本檔案 | 功能描述 |
| :--- | :--- |
| **`collect_traffic_data.py`** | **主要入口**：協調整個數據收集流程，從 API 下載數據、處理並生成 SUMO 路由檔案。 |
| **`batch_process.py`** | **批次處理器**：將 `data/trafficData/` 中所有 JSON 歷史數據批次處理，轉換為 `.rou.xml` 路由檔案並輸出至 `data/VehicleData/`。 |
| **`add_boundary_detectors.py`** | **偵測器設置**：解析路網檔，在路網邊界的入口/出口 edge 上自動放置車輛偵測器 (e1detectors)，用於流量測量。 |
| **`fix_emitters.py`** | **排放器修復**：修復 SUMO emitters 設定檔中的格式或數據問題。 |
| **`CollectData.py`** | **數據收集輔助**：提供數據收集相關的輔助功能。 |

### 工具列表 (`/tools`)

| 工具檔案 | 功能描述 |
| :--- | :--- |
| **`grabapi.py`** | **數據抓取器**：從台北市開放資料 API 下載 `GetVD.xml.gz`，解壓縮並解析 XML，提取交通流量與速度數據。 |
| **`selectRoad.py`** | **數據篩選器**：根據地圖邊界篩選原始交通數據，使用 `searchnetdata.py` 取得座標範圍，僅保留落在範圍內的道路數據。 |
| **`searchnetdata.py`** | **路網查詢**：解析 SUMO 路網檔 (`.net.xml`) 以決定地圖邊界 (經緯度)，並提供根據座標搜尋 Edge ID 的功能。 |
| **`convertToRou.py`** | **路由生成器**：將處理後的交通數據轉換為 SUMO 路由檔 (`.rou.xml`)，利用 `duarouter` 計算有效路徑。 |
| **`fixRoadData.py`** | **數據補全**：處理缺失或不完整的交通數據，根據歷史或鄰近道路數據補值，確保模擬順利運行。 |
| **`apply_sumo_timings.py`** | **號誌控制**：根據定義的時制計畫，生成紅綠燈時序設定檔 (`traffic_light.add.xml`)。 |
| **`connections_out.py`** | **路網分析**：從路網檔案中提取並分析連接數據 (Connections)，了解車道間的連接關係。 |
| **`aggregate_traffic_lights.py`** | **號誌數據彙整**：搜尋並合併多個 CSV 模擬輸出檔案，統計分析號誌運作數據。 |
| **`analyze_with_ollama.py`** | **AI 分析**：使用 Ollama 本地 LLM 對交通模擬數據進行分析與洞察。 |
| **`main.py`** | **工具主程式**：tools 模組的主要執行入口。 |

### 資料目錄 (`/data`)

- **`ntut_config.sumocfg`**: 主要的 SUMO 模擬設定檔。
- **`ntut_network_split.net.xml`**: 用於模擬的主要路網檔案。
- **`ntut_network_split.net copy.xml`**: 路網副本，用於讀取地圖邊界座標。
- **`trafficData/`**: 儲存從 API 下載的原始交通數據 (JSON)。
- **`VehicleData/`**: 儲存處理後的 SUMO 路由檔案 (`.rou.xml`)。
- **`config/`**: 存放模擬相關設定檔。
- **`source/`**: 存放原始來源數據 (如原始 OSM 地圖)。
- **`legacy/`**: 舊版路網檔案的備份。
- **`timing_plan.json`** / **`timing_plan_table.json`**: 號誌時制計畫設定檔。
- **`臺北市政府交通局路口時制號誌資料.csv`**: 台北市官方路口號誌時制資料。

---

## 🛠️ 設定 (Configuration)

- **地圖邊界 (Map Boundary)**：在 `data/ntut_network_split.net copy.xml` 中動態定義，並由 `searchnetdata.py` 讀取。
- **API URL**：配置於 `scripts/collect_traffic_data.py` 中。
- **號誌時制**：配置於 `data/timing_plan.json` 中。
