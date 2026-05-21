# Traffic Dashboard

TrafficVision 的即時交通監控儀表板 — NTUT 周邊路網。

## 架構

整個前端是**單一自包含檔** `dashboard.html`，由 `serve_api.py` 透過 `/ui_kits` 同源送出。
React / ReactDOM / Babel / Leaflet 走 CDN（已加 SRI 完整性檢核），在瀏覽器即時轉譯 JSX。

> 早期的 `*.jsx` 元件原型與 `index.html` 已移除——它們未被載入、且與 `dashboard.html` 的實作不一致（例如舊版是 SVG 示意圖，現版是 Leaflet 真實地圖）。所有元件現都內嵌在 `dashboard.html`。

## 啟動

```bash
python "TrafficVision Design System/serve_api.py"
# 瀏覽器開 http://localhost:8000/  （會自動導向 dashboard.html）
```

## 畫面

- **總覽儀表板** — KPI、路段監控列表、Leaflet 地圖、號誌控制、AI 助理
- **路段監控** — 完整路段資料表
- **號誌控制** — 號誌相位面板 + 地圖
- **流量預測** — GRU 5 分鐘預測表
- **AI 數據助理** — 全高 LLM 對話
- **匯出報告** — handoff 檔案下載

## 地圖模式

| 模式 | 資料來源 |
|------|---------|
| 當前車流 | 真實 VD 路段線（`/api/traffic`）+ 全路網幾何，VD 速度著色 |
| 5分鐘預測 | SUMO no_control 模擬（`/api/edge-heatmap/baseline`）|
| 優化後預測 | SUMO 最佳策略（`/api/edge-heatmap`）|
| 車流熱力圖 | SUMO 最佳策略密度 |

## 資料來源

全部經 `serve_api.py` 的 `/api/*` 端點提供（不直接公開 `data/` 目錄）。
