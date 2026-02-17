import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import joblib

# =================================================
# 1. 模型定義 (必須與訓練時一致)
# =================================================
class GRUTraffic(nn.Module):
    def __init__(self, num_edges, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        # Input dim = num_edges (車流量) + 1 (時間特徵)
        self.gru = nn.GRU(
            input_size=num_edges + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_edges)
        self.relu = nn.ReLU() 

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return self.relu(pred)

# =================================================
# 2. 載入模型與資料
# =================================================
def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    # 由於我們儲存了 sklearn 的 scaler，需要設定 weights_only=False
    # 注意：這需要確保載入的檔案是可信來源
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}") # Debug output
    
    config = checkpoint.get("config", {})
    # 修正 num_edges 取得方式：優先從 config 拿，沒有則從 edge_ids 長度推算
    num_edges = config.get("num_edges", len(checkpoint["edge_ids"]))
    input_len = config.get("input_len", 6)
    
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    
    model = GRUTraffic(num_edges, hidden_dim, num_layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, checkpoint["scaler"], checkpoint["edge_ids"], input_len

# 資料前處理 (複製自 train_model.py)
def extract_datetime(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        return m.group(1) + m.group(2)
    return None

def get_time_feature(filename):
    """
    從檔名提取「一天中的時間進度」(0.0 ~ 1.0)
    例如: 12:00:00 -> 0.5
    """
    time_str = extract_datetime(filename) # YYYYMMDDHHMMSS
    if time_str:
        hh = int(time_str[8:10])
        mm = int(time_str[10:12])
        ss = int(time_str[12:14])
        total_seconds = hh * 3600 + mm * 60 + ss
        return total_seconds / 86400.0 # 正規化到 0~1
    return 0.0

def load_sample_data(data_dir, edge_ids, input_len, target_file=None):
    """
    讀取資料來做測試
    target_file: 指定要讀取的單一檔案名稱 (例如: "traffic_data_20260126_121420.csv")
                 若為 None，則預設讀取最後 5 個檔案
    """
    if target_file:
        print(f"Loading specific file: {target_file}...")
        files = [target_file]
    else:
        print("Loading recent data for testing (Default: Last 5 files)...")
        files = [
            f for f in os.listdir(data_dir)
            if f.startswith("traffic_data_") and f.endswith(".csv")
        ]
        files = sorted(files, key=lambda x: extract_datetime(x))
        files = files[-5:] 

    series_list = []
    time_features = []
    
    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            if "時間" in df.columns:
                df = df.rename(columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"})
            
            pivot = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
            
            # 確保欄位與訓練時一致 (補0或移除多餘)
            pivot = pivot.reindex(columns=edge_ids, fill_value=0)
            series_list.append(pivot.values)
            
            # 計算時間特徵
            t_feat = get_time_feature(fname)
            t_feat_array = np.full((len(pivot), 1), t_feat, dtype=np.float32)
            time_features.append(t_feat_array)
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
            
    if not series_list:
        raise ValueError("Cannot load data.")
        
    raw_traffic = np.concatenate(series_list, axis=0)
    raw_time = np.concatenate(time_features, axis=0) # (Total_Length, 1)
    
    # 檢查長度是否足夠
    if len(raw_traffic) <= input_len:
         print(f"Warning: Data length ({len(raw_traffic)}) is too short for input_len ({input_len}).")
         # 若指定單一檔案且長度不足，我們可能無法做預測，或者只能做前面幾筆
         
    return raw_traffic, raw_time, files

# =================================================
# 3. 主程式
# =================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
    MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 指定要測試的檔案 (您可以修改這裡!)
    # TEST_FILE = "traffic_data_20260126_121420.csv" 
    TEST_FILE = None # None = 自動抓最新 5 個檔案

    # 1. 載入模型
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train_model.py first.")
        exit(1)

    try:
        model, scaler, edge_ids, input_len = load_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    # 2. 準備測試資料
    try:
        raw_traffic, raw_time, loaded_files = load_sample_data(DATA_DIR, edge_ids, input_len, target_file=TEST_FILE)
        print(f"Loaded {len(loaded_files)} file(s). Total timesteps: {len(raw_traffic)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 3. 進行預測
    print("\nStarting Prediction Test...")
    print(f"{'Time Step':<10} | {'Edge ID':<15} | {'Real':<5} | {'Pred':<5} | {'Diff':<5}")
    print("-" * 60)

    # 正規化
    scaled_traffic = scaler.transform(raw_traffic)
    
    # 開始預測
    # 我們從 input_len 開始，用前 input_len 筆資料預測第 i 筆
    count = 0
    for i in range(len(scaled_traffic) - input_len):
        # 準備輸入：取 [i : i+input_len]
        input_traffic = scaled_traffic[i : i+input_len] 
        input_time = raw_time[i : i+input_len]
        
        # 合併特徵 -> (input_len, num_edges + 1)
        input_combined = np.hstack([input_traffic, input_time])
        
        input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()[0]
        
        # 反正規化
        pred_real = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        actual_real_seq = raw_traffic[i+input_len]
        
        # 顯示車流最多的前 1 個路段 (避免洗版)
        top_idx = np.argmax(actual_real_seq)
        
        edge_name = edge_ids[top_idx]
        real_val = actual_real_seq[top_idx]
        pred_val = pred_real[top_idx]
        diff = pred_val - real_val
        
        print(f"T+{i:<8} | {edge_name:<15} | {real_val:<5.0f} | {pred_val:<5.1f} | {diff:<5.1f}")
        
        count += 1
        if count >= 20: # 只顯示前 20 筆預測，避免太長
            print("... (Truncated) ...")
            break

    print("-" * 60)
    print("\nTest completed.")
