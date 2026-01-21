# Data 資料目錄

本目錄包含 SUMO 交通模擬所需的所有資料檔案。

---

## 📁 目錄結構

```
data/
├── legacy/                 # 舊地圖 (NTUT 周邊)
│   ├── ntut_network.net.xml
│   ├── ntut_network_split.net.xml  (主要使用)
│   ├── ntut_config.sumocfg
│   ├── ntut_tls.add.xml
│   ├── ntut_routes.rou.xml
│   └── ntut_mapping.csv
│
├── real_world/             # 新地圖
│   ├── realworld_network.net.xml
│   ├── realworld_network_clean.net.xml
│   ├── realworld_config.sumocfg
│   ├── realworld_tls.add.xml
│   ├── realworld_routes.rou.xml
│   └── realworld_poly.poly.xml
│
├── source/                 # 原始資料
│   ├── timing_plan.json
│   ├── map.osm
│   └── ...
│
├── analysis_scripts/       # 分析腳本 (Python)
│
├── logs/                   # 日誌檔案
│
└── sumo_json_mapping_fixed.csv  # API 對應表
```

---

## 📝 命名規則

### 格式
```
<map>_<type>.<ext>

map:  ntut | realworld
type: network | config | routes | tls | mapping | poly
ext:  net.xml | sumocfg | rou.xml | add.xml | csv | poly.xml
```

### 規則
1. **全小寫**
2. **使用底線** `_` 分隔 (無空格、無連字號)
3. **地圖前綴**: `ntut_` 或 `realworld_`
4. **類型標記**: `network`, `config`, `routes`, `tls`, `mapping`, `poly`
5. **變體後綴**: `_split`, `_clean`, `_fixed` (可選)

### 範例
| 用途 | 檔名 |
|------|------|
| NTUT 網路 | `ntut_network.net.xml` |
| NTUT 拆分版 | `ntut_network_split.net.xml` |
| NTUT 配置 | `ntut_config.sumocfg` |
| NTUT 紅綠燈 | `ntut_tls.add.xml` |
| 真實世界網路 | `realworld_network.net.xml` |

---

## 🔧 SUMO 執行

### NTUT 地圖
```bash
cd data/legacy
sumo-gui -c ntut_config.sumocfg
```

### Real World 地圖
```bash
cd data/real_world
sumo-gui -c realworld_config.sumocfg
```

### 輸出檔案
配置中已設定 `output-prefix`，執行後會自動產生：
- `ntut_tripinfos.xml` / `realworld_tripinfos.xml`
- `ntut_stats.xml` / `realworld_stats.xml`

---

## 📊 對應表

### ntut_mapping.csv / realworld_mapping.csv
```csv
junction_id,tls_id,x,y,connections
3086736518,joinedS_3086736518_655375232_655375233,573.15,435.26,47
```

### sumo_json_mapping_fixed.csv (共用)
```csv
junction_id,sumo_id,icid,dist,name
```
對應 SUMO TLS ID 與台北市交通 API 的 ICID。
