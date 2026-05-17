# TrafficVision — NTUT 交通監控系統

台北科技大學周邊路網的智慧交通控制系統。整合台北市 VD 感應器、SUMO 模擬、GRU 神經網路預測、號誌優化，以及本地微調的 Gemma 4 LLM 助理，透過 FastAPI + Docker 提供即時儀表板。

---

## 完成狀態

| 模組 | 狀態 | 說明 |
|------|------|------|
| FastAPI 後端 | ✅ 完成 | 所有 REST 端點 + WebSocket 模擬串流 |
| 前端儀表板 | ✅ 完成 | React + Leaflet 熱力圖 + LLM 聊天面板，`GET /` 直接可用 |
| GRU 預測模型 | ✅ 完成 | 已訓練，`gru_traffic_model.pth`（需手動複製至新電腦） |
| 號誌優化引擎 | ✅ 完成 | 5 策略並行評估，composite score 選最佳 |
| 資料集生成 | ✅ 完成 | 從 100 個 traffic JSON + 20 個 handoff 目錄生成 ~28k 筆 Q&A |
| LLM 微調 | ⏳ **待新電腦執行** | A2000 12GB，約 1–2 小時 |
| GGUF 匯出 | ⏳ **待新電腦執行** | 微調完成後一鍵執行 |
| Docker 部署 | ⏳ **待新電腦執行** | `docker compose up --build` |

---

## 系統架構

```
Taipei VD API ─────────────────────────────────────────────────┐
                                                               ▼
                                             serve_api.py /api/traffic
                                                               │
tools/grabapi.py → convertToRou.py → duarouter                 │
  → VehicleData.py (SUMO × 16 workers)                        │
  → data/simulation_data/*.csv                                 │
  → train_model.py → gru_traffic_model.pth                     │
  → tools/runtime_pipeline.py:                                 │
      1. route 生成                                            │
      2. SUMO 模擬 → CSV                                      │
      3. predict_main.py → prediction CSV + signal XML         │
      4. generate_edge_traffic.py → edge_heatmap.json          │
      → data/runtime_data/<stem>/handoff/                    ──┘
                                                               │
                                                    FastAPI (port 8000)
                                              ┌─────── Ollama (port 11434)
                                              │         Gemma 4 E4B Q8_0
                                              │
                                      GET /  儀表板
                                      GET /api/*  REST 端點
                                      WS  /ws/simulation
```

---

## 新電腦設定步驟（你還需要做的事）

> 按順序執行即可，完成後網站即可正常使用 LLM。

### Step 0 — 確認環境

```powershell
nvidia-smi          # 應顯示 A2000 12GB，右上角 CUDA 版本
docker --version    # Docker Desktop 需開啟 GPU 支援
ollama --version    # 若未安裝：https://ollama.com/download
python --version    # 需要 3.11+
```

### Step 1 — 複製大型檔案

以下檔案被 `.gitignore` 排除，需從舊電腦用 USB / 網路硬碟複製：

| 檔案 / 目錄 | 用途 | 備註 |
|---|---|---|
| `gru_traffic_model.pth` | GRU 預測模型（API pipeline 用） | 必要 |
| `data/simulation_data/` | SUMO 模擬訓練資料 | 只需重訓 GRU 時才複製 |
| `TrafficVision Design System/data/trafficData/` | 歷史 VD 快取 | 選填，API 啟動後會自動補抓 |

> `models/gguf/` 的 GGUF **不用**複製，Step 4 會在新電腦重新生成。

### Step 2 — 安裝微調依賴並生成資料集

```powershell
# 若 CUDA 不是 12.4，先編輯 requirements-training.txt 改為 cu118 或 cu121
pip install -r requirements-training.txt

# 生成 fine-tune 資料集（從 100 個 traffic JSON + 20 個 handoff 目錄）
python tools/generate_finetune_dataset.py
# 完成後：data/finetune_dataset.jsonl（約 28,000 筆 Q&A）
```

### Step 3 — 微調 Gemma 4（約 1–2 小時）

```powershell
python tools/finetune_gemma.py
# A2000 12GB 足夠，不需修改任何超參數
# 完成後：models/trafficvision-gemma4/（LoRA adapters）
```

訓練設定（`tools/finetune_gemma.py` 頂部可調整）：
- 模型：`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`（4B 參數，4-bit NF4）
- LoRA rank：16，epochs：3，learning rate：2e-4
- VRAM 用量：約 8–10GB（A2000 12GB 有餘裕）

### Step 4 — 匯出 GGUF 並匯入 Ollama

```powershell
# 先確認 Ollama 服務正在執行（另開一個終端機）
ollama serve

# 匯出並自動匯入（約 10–20 分鐘）
python tools/export_to_gguf.py
# 完成後：models/gguf/trafficvision-gemma4-q8_0.gguf
# 自動執行：ollama create trafficvision-gemma4
```

驗證：

```powershell
ollama list
# 應顯示 trafficvision-gemma4

ollama run trafficvision-gemma4 "目前北科大周邊交通如何？"
# 確認模型有正常回應再繼續
```

### Step 5 — 啟動 Docker（網站上線）

```powershell
docker compose up --build
# 首次 build 約 3–5 分鐘，之後啟動只需數秒

# 確認服務健康
docker compose ps
```

服務就緒後：

| URL | 說明 |
|-----|------|
| **`http://localhost:8000/`** | **儀表板主頁（由此進入）** |
| `http://localhost:8000/docs` | FastAPI 自動文件 |
| `http://localhost:8000/api/status` | 系統狀態 JSON |
| `http://localhost:8000/api/chat` | LLM 聊天 API（POST） |

### Step 6 — 節省磁碟空間（選填）

微調完成後可刪除：

```powershell
# 合併後的 fp16 模型（約 8GB，GGUF 已不需要它）
Remove-Item -Recurse -Force models\trafficvision-gemma4-merged

# Unsloth 編譯快取（機器相關，可自動重生）
Remove-Item -Recurse -Force unsloth_compiled_cache
```

---

## 日常使用

### 啟動 / 停止

```powershell
docker compose up -d        # 背景執行
docker compose down         # 停止所有服務
docker compose logs -f api  # 查看 API 即時 log
```

### 執行 ML 資料管線（更新預測與號誌）

> 需要安裝 SUMO 並設定環境變數

```powershell
$env:SUMO_HOME = "C:\path\to\sumo"
python tools/runtime_pipeline.py --once          # 執行一次
python tools/runtime_pipeline.py --interval 300  # 每 5 分鐘自動執行
```

### 更新 LLM（資料累積後重新微調）

```powershell
python tools/generate_finetune_dataset.py   # 重新生成資料集
python tools/finetune_gemma.py              # 重新微調
python tools/export_to_gguf.py             # 重新匯出並匯入 Ollama
docker compose restart api                  # 重啟 API（不需 rebuild）
```

---

## 常見問題

**`export_to_gguf.py` 找不到 llama.cpp？**
```powershell
git clone https://github.com/ggerganov/llama.cpp "$HOME\.unsloth\llama.cpp"
pip install -r "$HOME\.unsloth\llama.cpp\requirements.txt"
```

**儀表板地圖是空的？**

`edge_heatmap.json` 由 `runtime_pipeline.py` 生成，首次啟動前不存在。
可以從舊電腦複製 `TrafficVision Design System/data/edge_heatmap.json`，
或啟動後執行一次 `python tools/runtime_pipeline.py --once`。

**`/api/chat` 回應「AI 服務暫時無法使用」？**

確認 `ollama serve` 正在執行且已有 `trafficvision-gemma4` 模型（`ollama list` 確認）。

**CUDA 版本不是 12.4？**

編輯 `requirements-training.txt`，將 `unsloth[cu124]` 改為 `unsloth[cu118]` 或 `unsloth[cu121]`。

---

## 專案結構

```
├── TrafficVision Design System/
│   ├── serve_api.py                         ← FastAPI 主程式（含靜態文件服務）
│   ├── ui_kits/traffic-dashboard/index.html ← 儀表板（GET /）
│   └── data/
│       ├── trafficData/      ← VD 快取 JSON（gitignore，API 自動更新）
│       └── edge_heatmap.json ← 路段熱力圖（gitignore，pipeline 生成）
├── tools/
│   ├── runtime_pipeline.py          ← 主排程器（grab→simulate→predict→handoff）
│   ├── generate_finetune_dataset.py ← 生成 LLM 訓練資料（100 個 JSON）
│   ├── finetune_gemma.py            ← Gemma 4 LoRA 微調（GPU）
│   └── export_to_gguf.py            ← 匯出 GGUF + 自動匯入 Ollama
├── models/
│   ├── trafficvision-gemma4/         ← LoRA adapters（gitignore）
│   ├── trafficvision-gemma4-merged/  ← 合併 fp16（gitignore，微調後可刪）
│   └── gguf/
│       ├── trafficvision-gemma4-q8_0.gguf ← 量化模型（gitignore）
│       └── Modelfile                       ← Ollama 設定（temperature=0.2, ctx=8192）
├── data/
│   ├── simulation_data/  ← SUMO 模擬輸出（gitignore）
│   └── runtime_data/     ← Pipeline 執行結果（gitignore）
├── gru_traffic_model.pth ← GRU 模型（gitignore，需手動複製）
├── train_model.py        ← GRU 離線訓練
├── docker-compose.yml    ← ollama + api 兩個服務
├── Dockerfile            ← API 容器（python:3.11-slim，無 SUMO）
├── requirements.txt      ← API 容器依賴
└── requirements-training.txt ← 微調依賴（GPU 環境）
```
