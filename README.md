# 1. grabapi.py - 車流數據抓取與解析

## 處理項目：
- **抓取車流數據**：透過 `requests` 模組從遠端 `URL` 抓取 `.xml.gz` 格式的車流資料檔案。
- **解壓縮資料**：將 `.xml.gz` 壓縮檔解壓縮成 `.xml` 格式，便於後續解析。
- **解析 XML 資料**：使用 `xml.etree.ElementTree` 解析解壓後的 XML 檔案，提取路段的車流資訊並儲存至 `Python` 字典中。
- **清理檔案**：解壓縮及解析完後，會刪除壓縮檔和 XML 檔案。

## 總整：

這段程式的目的是從指定的 URL 下載車流數據，並解析該數據以便進行後續分析。它主要用來抓取和處理交通資訊，並將資料以字典格式返回，方便後續進行分析或視覺化。

# 2. searchnetdata.py - 路網結構與範圍搜尋

## 處理項目：
- **解析路網結構資料**：使用 `xml.etree.ElementTree` 解析本地存儲的 `ntut-the way.net.xml` 檔案，並取得 XML 的根元素。
- **獲取地圖範圍**：從 XML 中提取地圖邊界（最左、最下、最右、最上的經緯度）並返回地圖範圍。
- **搜尋特定路段資訊**：搜尋 XML 中的 `<edge>` 標籤，根據路段名稱、ID 和起點/終點資料，將符合條件的路段詳細資訊儲存到字典中。

## 總整：

這段程式主要是處理與路網結構相關的 XML 資料，並提供地圖範圍的提取功能。它還能夠搜尋特定路段的資訊，並將結果以字典形式儲存，便於後續使用。這有助於進行地理範圍內的路網分析。

# 3. select.py - 車流數據篩選與過濾

## 處理項目：
- **篩選車流資料**：利用從 `grabapi.py` 獲取的車流數據（roadInfo），篩選出位於地圖範圍內的路段。
- **範圍檢查**：使用 `inrange()` 函數檢查路段的起點和終點是否在 `searchnetdata.py` 提供的地圖範圍內。
- **儲存篩選結果**：將符合範圍的路段資訊存儲至字典 `temp` 中，並最後輸出符合條件的路段資料。

## 總整：

這段程式碼將車流數據和路網結構結合，根據地理範圍進行篩選，將符合條件的路段資料提取出來並顯示。這對於根據特定地理範圍進行交通數據分析非常有用。

---

# connections_out.py 功能說明

## 主要函式與用途

### 1. extract_traffic_lights(xml_file_path, tree)
- 作用：解析 SUMO XML 網路檔案，提取所有紅綠燈（tlLogic）資訊，包括相位、週期、偏移等。
- 回傳：紅綠燈資訊列表。

### 2. extract_edge_names(tree)
- 作用：提取所有道路（edge）的名稱，建立 edge_id 與 name 的對應。
- 回傳：edge_id 對應名稱的字典。

### 3. decode_traffic_light_state(state_string)
- 作用：將紅綠燈相位的狀態字串（如 G, r, y）解碼為中文說明。
- 回傳：解碼後的狀態描述字串。

### 4. get_current_traffic_light_state(traffic_light, current_time)
- 作用：根據模擬時間計算紅綠燈目前相位與剩餘時間。
- 回傳：目前相位資訊。

### 5. extract_connections(tree, exclude_internal=False)
- 作用：提取所有道路連接（connection），可選擇排除交叉路口內部連接（以 : 開頭的 edge）。
- 回傳：連接資訊列表。

### 6. link_connections_to_traffic_lights(connections, traffic_lights, tree)
- 作用：將連接與紅綠燈控制關聯，標記哪些 connection 受紅綠燈控制。
- 回傳：更新後的連接資訊列表。

### 7. save_traffic_lights_to_csv(traffic_lights, output_file)
- 作用：將所有紅綠燈基本資訊儲存為 CSV。
- 輸出：traffic_lights.csv
- 內容：紅綠燈 id、型態、週期、相位數、各相位描述。

### 8. save_individual_traffic_light_to_csv(traffic_light, output_file)
- 作用：儲存單一紅綠燈的基本資訊。
- 輸出：tl_<id>_info.csv（於個別紅綠燈目錄）
- 內容：同上，但僅單一紅綠燈。

### 9. save_individual_traffic_light_phases_to_csv(traffic_light, output_file)
- 作用：儲存單一紅綠燈的所有相位詳細資訊。
- 輸出：tl_<id>_phases.csv（於個別紅綠燈目錄）
- 內容：每個相位的持續時間、狀態、描述。

### 10. generate_traffic_light_timeline(traffic_lights, duration_seconds=300)
- 作用：根據紅綠燈週期，生成模擬期間（預設 300 秒）每秒的紅綠燈狀態時間軸。
- 回傳：時間軸資料。

### 11. save_timeline_to_csv(timeline, output_file)
- 作用：將所有紅綠燈的時間軸儲存為 CSV。
- 輸出：traffic_light_timeline.csv
- 內容：每秒各紅綠燈的相位、剩餘時間、週期位置。

### 12. save_individual_traffic_light_timeline_to_csv(traffic_light, duration_seconds, output_file)
- 作用：儲存單一紅綠燈的時間軸。
- 輸出：tl_<id>_timeline.csv（於個別紅綠燈目錄）
- 內容：同上，但僅單一紅綠燈。

### 13. save_individual_traffic_light_connections_to_csv(traffic_light_id, connections, output_file)
- 作用：儲存特定紅綠燈控制的所有連接資訊。
- 輸出：tl_<id>_connections.csv（於個別紅綠燈目錄）
- 內容：該紅綠燈控制的 connection 詳細資料。

### 14. filter_connections(connections, from_edge=None, to_edge=None)
- 作用：根據起點或終點過濾連接。
- 回傳：過濾後的連接列表。

### 15. save_connections_to_csv(connections, output_file)
- 作用：儲存所有道路連接資訊。
- 輸出：road_connections.csv
- 內容：所有道路連接（不含交叉路口內部連接）。

---

## 主要輸出檔案說明

- **road_connections.csv**：所有道路連接（不含交叉路口內部連接），包含 from/to edge、車道、方向、是否受紅綠燈控制等。
- **traffic_lights.csv**：所有紅綠燈的基本資訊與相位摘要。
- **traffic_light_timeline.csv**：模擬期間每秒各紅綠燈的相位狀態、剩餘時間、週期位置。
- **紅綠燈<id>檔案/**：每個紅綠燈的個別目錄，包含：
    - tl_<id>_info.csv：該紅綠燈基本資訊
    - tl_<id>_phases.csv：該紅綠燈所有相位詳細資訊
    - tl_<id>_timeline.csv：該紅綠燈模擬期間的時間軸
    - tl_<id>_connections.csv：該紅綠燈控制的所有連接

---

## 使用流程簡述
1. 讀取 SUMO XML 網路檔案，解析紅綠燈與道路連接。
2. 產生紅綠燈、連接、時間軸等 CSV 檔案。
3. 每個紅綠燈自動建立獨立目錄，儲存詳細資訊。