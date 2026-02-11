# Traffic Control System Simulation (Chester)

> **High-Fidelity Traffic Simulation based on Real-World Data (Taipei)**

## 1. Quick Start (快速開始)

本專案已遷移至「真實路網 (Real World Network)」環境。請使用以下指令或在 SUMO-GUI 中開啟設定檔：

```bash
sumo-gui -c data/real_world.sumocfg
```

## 2. Simulation Environment (模擬環境)

### A. High-Fidelity Network (`real_world.net.xml`)
*   **Source**: Raw OpenStreetMap Data (`map.osm`, ~19MB).
*   **Generation**: Produced via `netconvert` with high-fidelity settings to preserve realistic geometry:
    *   `--geometry.remove false`: 保留真實道路彎曲度。
    *   `--ramps.guess true`: 自動識別匝道。
    *   `--tls.guess true` & `--junctions.join`: 自動識別並合併複雜路口號誌。
*   **Manual Upgrades**:
    *   **Junction 619136880**: Detected as a high-volume intersection (fed by 3-lane artery). Manually upgraded from `priority` to `traffic_light` to prevent traffic jams.

### B. Dynamic Route Generation (`output.rou.xml`)
*   Routes are generated dynamically using `tools/convertToRou.py`, which fetches real-time traffic data (Volume/Speed) and maps it to the new network using `duarouter`.
*   **Compliance**: Validated against SUMO routing standards.

### C. Traffic Light Logic (`traffic_light.add.xml`)
*   **Coverage Check**: Ensures every link index (0..N) has at least one Green phase to prevent "Missing Green" warnings.
*   **Flow Optimization**: Straight movements in complex clusters are given Priority Green ('G') to reduce waiting time (Teleporting), while strictly handling deadlock prevention via `<tls.ignore-internal-junction-jam>`.

---

## 3. Compliance with SUMO Standards (合規性說明)

本專案之建置與優化流程均嚴格遵循 [SUMO Application Manuals](https://sumo.dlr.de/docs/index.html#application_manuals)：

1.  **Network**: 使用 `netconvert` 的標準旗標處理 OSM 資料。
2.  **Routing**: 使用 `duarouter` 處理路徑需求，並正確處理 `vType` 與 `maxSpeed`。
3.  **Configuration**: `.sumocfg` 結構符合 XSD 規範，並移除了不適用於 Static TLS 的 `actuated` 參數。

---

## 4. Tools & Data Pipeline (工具說明)

以下為本專案使用的核心 Python 工具：

### 1. `grabapi.py` - 車流數據抓取
- **功能**：從遠端抓取 `.xml.gz` 車流數據並解析。
- **輸出**：Python 字典格式的即時車流資訊。

### 2. `tools/convertToRou.py` - 路由生成器
- **功能**：整合車流數據與路網，呼叫 `duarouter` 生成 SUMO 路由檔 (`output.rou.xml`)。
- **特點**：自動處理座標轉換與路網匹配。

### 3. `tools/apply_sumo_timings.py` - 號誌生成器
- **功能**：根據 `sumo_json_mapping.csv` 與時制計畫 JSON，生成 `traffic_light.add.xml`。
- **特點**：包含「安全覆蓋檢查 (Coverage Check)」與「叢集優化 (Cluster Optimization)」邏輯。

### 4. `searchnetdata.py` & `selectRoad.py`
- **功能**：提供座標搜尋與路網邊界定義，協助將 GPS 座標映射至 SUMO Edge ID。

## 5. Directory Structure (檔案結構)

為確保專案整潔，`data/` 目錄已進行以下分類：

*   **`data/source/`**：存放原始圖資 (`map.osm`)、大型 JSON 時制計畫表等來源檔案。
*   **`data/logs/`**：存放所有 `.txt` 記錄檔與除錯日誌。
*   **`data/analysis_scripts/`**：存放各種一次性的 Python 驗證與分析腳本 (`check_*.py`, `analyze_*.py`)。
*   **`data/legacy/`**：封存舊版的 `ntut-the way` 檔案。
*   **`data/` (Root)**：保留核心模擬檔案 (`real_world.sumocfg`, `real_world.net.xml`, `traffic_light.add.xml`)，確保 `Quick Start` 指令可直接執行。

---

## 5. Troubleshooting (疑難排解)

*   **Teleporting (車輛瞬移)**: 若仍有少量瞬移，通常是因為真實路網的物理瓶頸（如車道縮減）。系統已透過優化號誌盡量緩解此現象。
*   **Unused States Warning**: 這是 SUMO 的正常提示，表示某些號誌狀態未被目前的連接路段使用，不影響模擬正確性。