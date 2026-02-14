# VehicleData.py 程式碼解釋文件 (平行處理版)

## Overview
本腳本用於批次處理歷史交通路徑檔 (`.rou.xml`)，利用 **4 個平行進程** 同時驅動 SUMO 模擬，大幅縮短執行時間。

## 流程簡介
1.  **環境與設定**：初始化 `SUMO_HOME`，設定 4 核心並行 (`Pool(4)`)。
2.  **主程式 (Main)**：
    -   讀取路網 (`.net.xml`) 一次，獲取有效路段集合。
    -   搜尋所有 `*.rou.xml` 檔案。
    -   將工作 (檔案路徑 + 有效路段) 分派給 Process Pool。
3.  **單一檔案處理 (Worker)**：
    -   `process_file` 函數由每個 worker 獨立執行。
    -   **過濾**：移除無效路段車輛 (Filter Routes)。
    -   **設定**：建立**唯獨**的暫存設定檔 (檔名加入 Process ID 防止衝突)。
    -   **模擬**：啟動獨立的 SUMO 實例進行模擬。
    -   **存檔**：寫入對應的 CSV。
    -   **清理**：刪除暫存檔。

## 關鍵技術點

### 1. Multiprocessing Pool
```python
with multiprocessing.Pool(processes=4) as pool:
    # 使用 imap_unordered 提升效率，因為我們不需強求輸出順序
    for i, result in enumerate(pool.imap_unordered(process_file_wrapper, tasks), 1):
        ...
```
- 使用 `imap_unordered` 可以讓先完成的工作先回報，讓使用者能即時看到進度，而不是卡在某個慢的檔案。

### 2. 進程隔離與資源
- 每個 Process 都有自己獨立的 `traci` 實例，避免了全域變數衝突。
- `traci.start()` 會自動尋找可用的 port，實現多個 SUMO 同時執行不打架。
- 暫存檔名加入 `os.getpid()` (Process ID)，確保同一時間不同 worker 不會覆蓋到彼此的設定檔。

### 3. Windows 相容性
```python
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
```
- 正確使用 `if __name__ == "__main__":` 區塊，防止 Windows 下產生無限遞迴的 process 生成。

## 效能預期
- 理論上速度將提升接近 4 倍 (視 CPU 與 IO 狀況而定)。
- Console 會顯示即時進度 `[完成數/總數] Result (Time: Xs)`。
