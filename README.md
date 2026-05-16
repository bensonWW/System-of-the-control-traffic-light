# TrafficVision — NTUT 交通號誌智慧控制系統

以台北科技大學周邊路網為範圍，整合台北市 VD 即時車流 API、SUMO 交通模擬、GRU 預測模型、號誌週期優化、FastAPI 後端，以及 Unsloth 微調的 Gemma 4 LLM 助理。

---

## 完成狀態

| 模組 | 狀態 |
|------|------|
| 儀表板 `dashboard.html` | ✅ 完成 |
| `serve_api.py` FastAPI 後端 | ✅ 完成 |
| `runtime_pipeline.py` 完整處理流水線 | ✅ 完成 |
| 號誌週期優化器 `traffic_optimizer_signal.py` | ✅ 完成 |
| GRU 車流預測訓練程式 `train_model.py` | ✅ 完成 |
| GRU 模型權重 `gru_traffic_model.pth` | ⚠️ 已訓練，未入 git（需手動傳輸） |
| Unsloth Gemma 4 微調程式 `tools/finetune_gemma.py` | ✅ 完成 |
| GGUF 匯出 + Ollama 匯入腳本 `tools/export_to_gguf.sh` | ✅ 完成 |
| Docker 部署 `Dockerfile` / `docker-compose.yml` | ✅ 完成 |
| Gemma 4 微調資料集 `data/finetune_dataset.jsonl` | ✅ 已生成（135 筆 Q&A） |
| Gemma 4 微調執行 `models/trafficvision-gemma4/` | ✅ 已完成（Gemma 4 E4B，loss 0.14，51 steps） |
| GGUF 匯出 `models/gguf/*.gguf` | ✅ 已完成（Q8_0，7.45 GB，需 Ollama 匯入） |

---

## 系統架構

系統有兩條完全獨立的資料流，分工如下：

### Flow A — 即時 VD 車流（serve_api.py 自動處理）

```
台北市 VD Open Data API（外部）
        │
        │  serve_api.py 啟動時：若無快取，立即抓取一次
        │  serve_api.py 背景排程：每 5 分鐘自動重抓
        │  GET /api/traffic：讀快取（10 分鐘 TTL），過期則立即重抓
        ▼
TrafficVision Design System/data/trafficData/*.json
        │
        ▼
儀表板 路段數據表（速度 / 流量 / 占有率）
```

### Flow B — SUMO 模擬 + GRU 預測 + 號誌優化（runtime_pipeline.py）

```
台北市 VD Open Data API（外部）
        │
        │  tools/grabapi.py   抓取 VD 資料
        │  tools/selectRoad.py   篩選北科大周邊路段
        │  tools/convertToRou.py + duarouter   產生 SUMO 車輛路線
        ▼
data/final_output.rou.xml
        │
        │  SUMO 模擬
        ▼
data/runtime_data/<timestamp>/traffic_data_*.csv
        │
        │  tools/predict_main.py   GRU 模型推論（需 gru_traffic_model.pth）
        │  tools/traffic_optimizer_signal.py   號誌週期優化
        │  tools/export_handoff.py   整理輸出
        ▼
data/runtime_data/<timestamp>/handoff/
├── prediction*.csv          → serve_api.py GET /api/prediction
├── *signal_plan_summary*.csv → serve_api.py GET /api/signals
└── *comparison_summary*.csv → serve_api.py GET /api/comparison

        │
        │  tools/generate_edge_heatmap.py
        ▼
TrafficVision Design System/data/edge_heatmap.json
        │
        ▼  serve_api.py GET /api/edge-heatmap
儀表板 Leaflet 邊道熱圖
```

### 儀表板 ↔ API 對應

```
瀏覽器 dashboard.html
    │
    │ HTTP port 8000
    ▼
serve_api.py（本機執行）
    ├── GET /api/traffic        ← Flow A：trafficData/*.json
    ├── GET /api/edge-heatmap   ← Flow B：edge_heatmap.json
    ├── GET /api/signals        ← Flow B：handoff/*signal*.csv
    ├── GET /api/prediction     ← Flow B：handoff/prediction*.csv
    ├── GET /api/comparison     ← Flow B：handoff/*comparison*.csv
    ├── POST /api/chat          → Ollama :11434（Gemma 4）
    └── WS /ws/simulation       → SUMO TraCI 即時串流
```

### LLM 流水線（Unsloth 微調 → Ollama 部署）

```
tools/generate_finetune_dataset.py
    ▼  data/finetune_dataset.jsonl
tools/finetune_gemma.py（Unsloth + LoRA，需 NVIDIA GPU）
    ▼  models/trafficvision-gemma4/（LoRA adapters）
tools/export_to_gguf.py（Unsloth merge + llama.cpp Q8_0 量化，Windows 原生）
    ▼  models/gguf/trafficvision-gemma4-q8_0.gguf
init_ollama.sh 或手動 ollama create（ollama create）
    ▼  Ollama serve :11434
serve_api.py POST /api/chat → 注入即時交通 context → 回覆
```

---

## 快速啟動

### 方法一：雙擊 `start.bat`（Windows，推薦）

自動開啟兩個終端視窗：
- **視窗 1**：`serve_api.py`（HTTP API，port 8000）
- **視窗 2**：`runtime_pipeline.py`（每 5 分鐘執行完整流水線）

前置條件：
- Python 已安裝：`pip install -r requirements.txt`
- SUMO 已安裝且 `SUMO_HOME` 環境變數已設定（Flow B 需要）
- `gru_traffic_model.pth` 在專案根目錄（Flow B 需要）

### 方法二：手動分兩個終端執行

```bash
# 終端 1：啟動 API server
pip install -r requirements.txt
python "TrafficVision Design System/serve_api.py"

# 終端 2：啟動 pipeline 排程（每 5 分鐘）
cd tools
python runtime_pipeline.py

# 只跑一次（不進入排程）
python runtime_pipeline.py --once

# 自訂間隔（秒）
python runtime_pipeline.py --interval 180
```

### 方法三：Docker（僅啟動 API server，不含 pipeline）

```bash
cp .env.example .env
docker-compose up -d
# health check: http://localhost:8000/api/status
```

> Docker 目前只啟動 `serve_api.py` 和 `Ollama`。
> `runtime_pipeline.py` 需要 SUMO，不在 Docker 容器內，需在本機另外執行。

### 開啟儀表板

直接雙擊以下 HTML 檔案（不需 build）：
```
TrafficVision Design System/ui_kits/traffic-dashboard/dashboard.html
```

> API 未啟動時，路段數據顯示 `--`，SUMO 邊道以灰色骨架線渲染，LLM Chat 顯示服務未啟動。

---

## LLM 設定（Unsloth 微調 → Ollama）

### 步驟 1：生成微調資料集

```bash
python tools/generate_finetune_dataset.py
# 輸出：data/finetune_dataset.jsonl
```

### 步驟 2：微調（需 NVIDIA GPU）

```bash
pip install -r requirements-training.txt
python tools/finetune_gemma.py
# 輸出：models/trafficvision-gemma4/（LoRA adapters）
```

| 參數 | 值 |
|------|----|
| 基底模型 | `unsloth/gemma-4-12b-it-unsloth-bnb-4bit` |
| LoRA rank | 16 |
| Effective batch size | 8（2 × 4 grad accum） |
| Epochs | 3 |

### 步驟 3：匯出 GGUF 並匯入 Ollama

```bash
# Windows（推薦，不需要 cmake）
python tools/export_to_gguf.py
# 輸出：models/gguf/trafficvision-gemma4-q8_0.gguf（約 7.5 GB，Q8_0 格式）
# 注意：首次執行會下載 fp16 基底模型（~8 GB），需等候約 20 分鐘
```

### 步驟 4：啟動 Ollama

```bash
docker-compose up -d ollama
bash init_ollama.sh          # 將 GGUF 載入 Ollama
docker-compose up -d api
```

或直接安裝 Ollama 後跑：

```bash
ollama serve &
bash init_ollama.sh
```

### 替代：先用通用模型測試（無需微調）

```bash
ollama pull gemma3:12b
# 修改 .env：OLLAMA_MODEL=gemma3:12b
```

---

## 換機後待辦事項

### 必做

- [ ] **傳輸 `gru_traffic_model.pth`**（4.6 MB，未入 git）  
  用 USB 或 `scp` 複製到新機，放在專案根目錄。

- [ ] **安裝 Python 依賴**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **建立 `.env`**
  ```bash
  cp .env.example .env
  ```

- [ ] **啟動系統**（雙擊 `start.bat` 或手動執行兩個終端）

### Flow B 需額外安裝 SUMO

- [ ] 安裝 SUMO：<https://sumo.dlr.de/docs/Downloads.php>
- [ ] 設定 `SUMO_HOME` 環境變數指向 SUMO 安裝目錄
- [ ] 執行一次 `python tools/runtime_pipeline.py --once` 驗證

### LLM Chat 需額外設定

- [ ] 安裝 Ollama：<https://ollama.com/download>
- [ ] 完成微調流程（步驟 1–4），或先用 `gemma3:12b` 測試
- [ ] 確認 `/api/chat` 可正常回應

---

## API 端點

| 端點 | 說明 |
|------|------|
| `GET /api/status` | 各資料檔是否存在、最後更新時間 |
| `GET /api/traffic` | VD 即時車流（10 分鐘快取，過期自動重抓） |
| `POST /api/traffic/refresh` | 強制立即重新抓取台北 VD API |
| `GET /api/edge-heatmap` | SUMO 邊道流量熱圖 |
| `GET /api/prediction` | GRU 5 分鐘預測結果 |
| `GET /api/signals` | 號誌週期優化方案 |
| `GET /api/comparison` | 優化前後效益對比 |
| `GET /api/handoff` | 完整 handoff 目錄所有 CSV |
| `POST /api/chat` | LLM 對話（注入即時交通 context，呼叫 Ollama） |
| `WS /ws/simulation` | SUMO TraCI 即時模擬串流 |
| `POST /api/simulation/start` | 啟動 SUMO 模擬 |
| `POST /api/simulation/stop` | 停止 SUMO 模擬 |
| `GET /api/simulation/status` | 模擬執行狀態 |

---

## 目錄結構

```
.
├── start.bat                          ← 本機一鍵啟動（API + pipeline 排程）
├── TrafficVision Design System/
│   ├── ui_kits/traffic-dashboard/
│   │   └── dashboard.html             ← 主儀表板（單一 HTML，無 build 步驟）
│   ├── data/
│   │   ├── trafficData/               ← VD 快取（serve_api.py 自動存檔）
│   │   └── edge_heatmap.json          ← SUMO 邊道流量（runtime_pipeline.py 更新）
│   └── serve_api.py                   ← FastAPI 後端
├── data/
│   ├── ntut_network_split.net.xml     ← SUMO 北科大路網
│   ├── ntut_config.sumocfg            ← SUMO 模擬設定
│   └── runtime_data/                  ← pipeline 每輪輸出（gitignore）
│       └── <timestamp>/
│           ├── traffic_data_*.csv     ← SUMO 模擬輸出
│           └── handoff/               ← prediction / signals / comparison CSV
├── tools/
│   ├── runtime_pipeline.py            ← 完整流水線排程（每 5 分鐘）
│   ├── grabapi.py                     ← 台北 VD API 抓取
│   ├── selectRoad.py                  ← 篩選北科大周邊路段
│   ├── convertToRou.py                ← 車流資料 → SUMO 路線格式
│   ├── predict_main.py                ← GRU 模型推論
│   ├── traffic_optimizer_signal.py    ← 號誌週期優化
│   ├── export_handoff.py              ← 輸出整理
│   ├── generate_edge_heatmap.py       ← SUMO edgedata → edge_heatmap.json
│   ├── finetune_gemma.py              ← Unsloth Gemma 4 微調
│   ├── export_to_gguf.sh              ← LoRA → GGUF → Ollama
│   └── generate_finetune_dataset.py   ← 微調資料集生成
├── models/                            ← gitignore；本機訓練產生
│   ├── trafficvision-gemma4/          ← Unsloth LoRA adapters
│   └── gguf/                          ← GGUF 量化模型（供 Ollama）
├── train_model.py                     ← GRU 模型訓練
├── gru_traffic_model.pth              ← GRU 模型權重（gitignore）
├── Dockerfile
├── docker-compose.yml                 ← 啟動 serve_api.py + Ollama
├── init_ollama.sh                     ← Ollama 模型初始化
├── requirements.txt                   ← API server 依賴
└── requirements-training.txt          ← Unsloth 微調依賴（GPU）
```

---

## 授權

本專案為學術研究用途。
