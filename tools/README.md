# Tools 工具目錄

本目錄包含用於 SUMO 交通模擬系統的各種輔助工具腳本。

---

## 📁 目錄結構

```
tools/
├── config.py              # 集中配置管理 (網路選擇、路徑)
├── __init__.py            # Python 包初始化
│
├── tls_timing/            # 🚦 紅綠燈時制管理
│   ├── update_tls_by_time_v2.py  # 主要時制更新腳本
│   ├── add_tls_to_junction.py    # 新增紅綠燈到路口
│   ├── list_tls_junctions.py     # 列出所有紅綠燈路口
│   ├── timing_schedule.py        # 時制排程定義
│   └── ...
│
├── network_analysis/      # 🗺️ 路網分析與編輯
│   ├── connections_out.py        # 連接分析
│   ├── split_junction.py         # 拆分路口
│   ├── check_junction.py         # 檢查路口狀態
│   └── ...
│
├── api_data/              # 📡 API 資料擷取
│   └── grabapi.py                # 台北市交通 API
│
├── utilities/             # 🔧 輔助工具
│   ├── extract_junction_tls_mapping.py  # 提取 Junction-TLS 對應
│   └── update_mapping_csv.py     # 更新 CSV 對應表
│
└── route_generation/      # 🚗 路線生成 (保留既有)
```

---

## 🚀 快速開始

### 網路選擇
所有腳本支援 `--network` 參數來選擇地圖：

```bash
# 使用 legacy 地圖 (預設)
python tools/tls_timing/list_tls_junctions.py --network legacy

# 使用 real_world 地圖
python tools/tls_timing/list_tls_junctions.py --network real_world
```

### 常用指令

```bash
# 列出所有紅綠燈路口
python tools/tls_timing/list_tls_junctions.py -n legacy

# 為指定路口新增紅綠燈
python tools/tls_timing/add_tls_to_junction.py -n legacy -j 622618108

# 提取 Junction-TLS 對應表
python tools/utilities/extract_junction_tls_mapping.py -n legacy
```

---

## ⚙️ 配置管理 (config.py)

所有路徑由 `tools/config.py` 集中管理：

```python
from tools.config import get_network_config

config = get_network_config("legacy")
# config["net_path"]     -> 網路檔案路徑
# config["mapping_path"] -> 對應表路徑
# config["tls_add_path"] -> 紅綠燈附加檔案路徑
```

---

## 📝 命名規則

### 檔案命名
- 全部小寫
- 使用底線 `_` 分隔單字
- **禁止使用空格、連字號 `-`**
- 範例: `ntut_the_way.net.xml`, `traffic_light.add.xml`

### Python 模組命名
- 全部小寫
- 使用底線分隔
- 範例: `update_tls_by_time.py`, `add_tls_to_junction.py`

### 資料夾命名
- 全部小寫
- 使用底線分隔
- 範例: `tls_timing/`, `network_analysis/`

---

## 📂 資料檔案位置

| 地圖 | 路徑 |
|------|------|
| Legacy (NTUT) | `data/legacy/` |
| Real World | `data/real_world/` |

每個地圖資料夾包含：
- `*.net.xml` - 網路檔案
- `*.sumocfg` - SUMO 配置
- `traffic_light.add.xml` - 紅綠燈時制
- `junction_tls_mapping.csv` - Junction-TLS 對應表
