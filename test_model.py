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
        self.gru = nn.GRU(
            input_size=num_edges,
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

def load_sample_data(data_dir, edge_ids, input_len, num_samples=10):
    """讀取最後幾個檔案來做測試"""
    print("Loading recent data for testing...")
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith("traffic_data_") and f.endswith(".csv")
    ]
    files = sorted(files, key=lambda x: extract_datetime(x))
    
    # 取最後幾個檔案確保資料夠用
    target_files = files[-5:] 
    
    series_list = []
    for fname in target_files:
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            if "時間" in df.columns:
                df = df.rename(columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"})
            
            pivot = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
            pivot = pivot.reindex(columns=edge_ids, fill_value=0)
            series_list.append(pivot.values)
        except:
            continue
            
    if not series_list:
        raise ValueError("Cannot load data.")
        
    raw_data = np.concatenate(series_list, axis=0)
    
    # 我們需要稍微多一點資料來建立 input window
    # 例如：如果要預測 5 個點，且 lookback=6，我們至少需要 11 筆資料
    required_len = input_len + num_samples
    if len(raw_data) < required_len:
        print(f"Warning: Not enough data ({len(raw_data)}) for testing. Need {required_len}.")
        return raw_data
        
    test_data = raw_data[-required_len:]
    
    return test_data

# =================================================
# 3. 主程式
# =================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
    MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # 我們抓取最新的資料來模擬「即時預測」
        raw_data = load_sample_data(DATA_DIR, edge_ids, input_len, num_samples=5)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 3. 進行預測
    print("\nStarting Prediction Test...")
    print(f"{'Time Step':<10} | {'Edge ID':<15} | {'Real':<5} | {'Pred':<5} | {'Diff':<5}")
    print("-" * 60)

    # 正規化
    scaled_data = scaler.transform(raw_data)
    
    # 測試後 5 筆資料
    for i in range(len(scaled_data) - input_len):
        # 準備輸入
        input_seq = scaled_data[i : i+input_len] # (input_len, num_edges)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, input_len, num_edges)
        
        # 預測
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()[0] # (num_edges,)
        
        # 反正規化 (還原成真實車流量)
        # 由於 scaler 是針對 (n_samples, n_features) 設計的，我們需要 reshape
        pred_real = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        actual_real = raw_data[i+input_len]
        
        # 挑選車流最多的前 3 個路段來顯示 (比較有感覺)
        top_indices = np.argsort(actual_real)[-3:][::-1]
        
        for idx in top_indices:
            edge_name = edge_ids[idx]
            real_val = actual_real[idx]
            pred_val = pred_real[idx]
            diff = pred_val - real_val
            
            print(f"T+{i:<8} | {edge_name:<15} | {real_val:<5.0f} | {pred_val:<5.1f} | {diff:<5.1f}")
        print("-" * 60)

    print("\nTest completed.")
