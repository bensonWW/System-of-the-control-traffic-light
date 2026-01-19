import pandas as pd
import os
import ollama

def get_traffic_summary(flow_file):
    """
    使用 pandas 預處理車流數據，算出精簡的統計結果。
    """
    if not os.path.exists(flow_file):
        return f"錯誤: 找不到檔案 {flow_file}"
    
    try:
        df = pd.read_csv(flow_file)
        # 假設欄位名稱為: 時間,路段ID,車輛數,車輛ID列表
        # 我們統計每個路段的平均車流量與出現次數
        summary = df.groupby('路段ID')['車輛數'].agg(['mean', 'max', 'count']).reset_index()
        summary.columns = ['路段ID', '平均車流量', '最大車流量', '出現次數']
        
        # 只取前 20 個最擁擠的路段
        top_20 = summary.sort_values(by='平均車流量', ascending=False).head(20)
        return top_20.to_csv(index=False)
    except Exception as e:
        return f"預處理車流數據時發生錯誤: {e}"

def read_csv_simple(file_path, max_lines=50):
    """
    簡單讀取檔案內容用於參考。
    """
    if not os.path.exists(file_path): return ""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return "".join(f.readlines()[:max_lines])

def analyze_traffic_with_ai(model_name='gemma3:4b'):
    # 檔案路徑
    flow_file = "traffic_data/chinese/traffic_data_chinese.csv"
    conn_file = "紅綠燈個別檔案/aggregated_traffic_lights_info.csv"
    phases_file = "紅綠燈個別檔案/aggregated_traffic_lights_phases.csv"

    print(f"正在預處理大量數據...")
    flow_summary = get_traffic_summary(flow_file)
    
    # 讀取基礎設定（這部分通常不大）
    conn_info = read_csv_simple(conn_file, max_lines=30)
    phase_info = read_csv_simple(phases_file, max_lines=50)

    # 組合精簡後的 Prompt
    prompt = f"""
你是一個交通分析專家。我已經幫你把數萬行的模擬數據整理好了，以下是摘要：

### 1. 前 20 名最擁堵路段統計 (由 Python 預先計算):
{flow_summary}

### 2. 紅綠燈基本資訊參考:
{conn_info}

### 3. 紅綠燈相位設定參考:
{phase_info}

### 分析任務:
1. 指出最嚴重的三個壅塞路段 ID，並從「紅綠燈基本資訊」中推測這可能位於哪個區域。
2. 檢查相位設定中的綠燈時長。如果一個路段的「平均車流量」很高，但其對應紅綠燈的綠燈時間很短，請指出這個矛盾。
3. 給予 2-3 條關於如何調整綠燈秒數或相位順序的建議。

請用中文簡潔回答，直接給出結論。
"""

    print(f"正在調用 Ollama ({model_name}) 進行分析，請稍候...")
    try:
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        print("\n" + "="*50 + "\nAI 分析結果:\n" + "="*50)
        print(response['message']['content'])
        print("="*50)
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    analyze_traffic_with_ai()
