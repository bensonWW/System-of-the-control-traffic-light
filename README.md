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
- **SUMO 路由輸出**：`data/VehicleData/` (SUMO .rou.xml 格式)

#### **步驟 2：啟動 SUMO 模擬**
在 SUMO-GUI 中開啟主要的模擬設定檔。
```bash
sumo-gui -c data/ntut_config.sumocfg
```

---

## 📂 專案結構與工具說明 (Project Structure & Tools)

### 核心腳本 (Core Scripts)

#### `scripts/collect_traffic_data.py`
這是數據收集的主要入口點。它負責協調整個流程：
1.  從開放資料 API 下載即時交通數據 (XML)。
2.  將原始數據儲存至 `data/trafficData/` (例如：`traffic_data_YYYYMMDD_HHMMSS.json`)。
3.  呼叫多個工具 (詳見下表) 進行篩選、處理，並生成 SUMO 路由檔案。
4.  將最終的路由檔案儲存至 `data/VehicleData/` (例如：`traffic_data_YYYYMMDD_HHMMSS.rou.xml`)。

### 工具列表 (`/tools`)

| 工具檔案 | 功能描述 |
| :--- | :--- |
| **`grabapi.py`** | **數據抓取器**：從台北市開放資料 API 下載 `GetVD.xml.gz`，解壓縮並解析 XML，提取交通流量與速度數據。 |
| **`selectRoad.py`** | **數據篩選器**：根據地圖邊界篩選原始交通數據。它使用 `searchnetdata.py` 取得地圖的座標範圍，僅保留落在此範圍內的道路數據 (使用 XY 座標判斷)。 |
| **`searchnetdata.py`** | **路網查詢**：解析 SUMO 路網檔 (`.net.xml`) 以決定地圖邊界 (經緯度)，並提供根據座標搜尋 Edge ID 的功能。 |
| **`convertToRou.py`** | **路由生成器**：將處理後的交通數據轉換為 SUMO 路由檔 (`.rou.xml`)。它利用 SUMO 的 `duarouter` 根據交通流量計算路網上的有效路徑。 |
| **`fixRoadData.py`** | **數據補全**：處理缺失或不完整的交通數據，根據歷史數據或鄰近道路數據進行補值，確保模擬順利運行。 |
| **`apply_sumo_timings.py`**| **號誌控制**：根據定義的時制計畫，生成紅綠燈時序設定檔 (`traffic_light.add.xml`)。 |
| **`connections_out.py`** | **路網分析**：從路網檔案中提取並分析連接數據 (Connections)，了解車道間的連接關係。 |

### 資料目錄 (`/data`)

- **`ntut_config.sumocfg`**: 主要的 SUMO 模擬設定檔。
- **`ntut_network_split.net.xml`**: 用於模擬的主要路網檔案。
- **`trafficData/`**: 儲存從 API 下載的原始交通數據 (JSON)。
- **`VehicleData/`**: 儲存處理後的 SUMO 路由檔案 (`.rou.xml`)。
- **`source/`**: 存放原始來源數據 (如原始 OSM 地圖)。
- **`legacy/`**: 舊版路網檔案的備份。

---

## 🛠️ 設定 (Configuration)

- **地圖邊界 (Map Boundary)**：在 `ntut_network_split.net copy.xml` 中動態定義，並由 `searchnetdata.py` 讀取。
- **API URL**：配置於 `scripts/collect_traffic_data.py` 中。