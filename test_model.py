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
        # Input dim = num_edges (車流量)
        # 注意：train_model.py 已移除時間特徵輸入，這裡也要同步
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}") # Debug output
    
    config = checkpoint.get("config", {})
    # 修正 num_edges 取得方式：優先從 config 拿，沒有則從 edge_ids 長度推算
    if "edge_ids" in checkpoint:
        edge_ids = checkpoint["edge_ids"]
    else:
        edge_ids = checkpoint.get("edge_ids_list", [])

    num_edges = len(edge_ids)
    
    input_len = config.get("input_len", 6)
    pred_horizon = config.get("pred_horizon", 15) # Default to 15
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    
    print(f"Model Config: input_len={input_len}, horizon={pred_horizon}, num_edges={num_edges}")
    
    model = GRUTraffic(num_edges, hidden_dim, num_layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, checkpoint["scaler"], edge_ids, input_len, pred_horizon

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

def load_sample_data(data_dir, edge_ids, input_len, target_file=None, num_files=50):
    """
    讀取資料來做測試
    target_file: 指定要讀取的單一檔案名稱 (例如: "traffic_data_20260126_121420.csv")
                 若為 None，則預設讀取最後 num_files 個檔案
    """
    if target_file:
        print(f"Loading specific file: {target_file}...")
        files = [target_file]
    else:
        print(f"Loading recent data for testing (Default: Last {num_files} files)...")
        files = [
            f for f in os.listdir(data_dir)
            if f.startswith("traffic_data_") and f.endswith(".csv")
        ]
        files = sorted(files, key=lambda x: extract_datetime(x))
        if num_files > 0:
            files = files[-num_files:] 

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
    TEST_FILE = None # None = 自動抓最新 N 個檔案
    NUM_TEST_FILES = 50 # 測試最近 50 個檔案

    # 1. 載入模型
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train_model.py first.")
        exit(1)

    try:
        model, scaler, edge_ids, input_len, pred_horizon = load_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    # 2. 準備測試資料
    try:
        raw_traffic, raw_time, loaded_files = load_sample_data(DATA_DIR, edge_ids, input_len, target_file=TEST_FILE, num_files=NUM_TEST_FILES)
        print(f"Loaded {len(loaded_files)} file(s). Total timesteps: {len(raw_traffic)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 正規化
    scaled_traffic = scaler.transform(raw_traffic).astype(np.float32)

    print("\nStarting Prediction Test...")
    print(f"Predicting T+{pred_horizon} (Horizon = {pred_horizon})")
    print(f"{'Time Step':<10} | {'Edge ID':<15} | {'Real':<5} | {'Pred':<5} | {'Diff':<5}")
    print("-" * 60)

    # 收集結果用於計算準確率
    all_real = []
    all_pred = []

    # 3. 進行預測
    # 最後一筆輸入 index 是 len - input_len - pred_horizon
    max_idx = len(scaled_traffic) - input_len - pred_horizon

    # 為了顯示方便，只 print 前 20 筆，但計算用全部
    display_limit = 20
    
    with torch.no_grad():
        for i in range(max_idx + 1):
            # 取出一段輸入 (包含 input_len 個時間步)
            input_traffic = scaled_traffic[i : i+input_len]
            
            # 轉為 Tensor
            input_tensor = torch.tensor(input_traffic, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # 預測
            pred = model(input_tensor) # (1, num_edges)
            
            # 反正規化
            pred_real = scaler.inverse_transform(pred.cpu().numpy())[0] # (num_edges,)
            
            # 取得真實值 (t + input_len + pred_horizon - 1)
            # 假設 horizon=1，target 就是下一個點 (i+input_len)
            # 假設 horizon=15，target 就是 (i+input_len+14)
            target_idx = i + input_len + pred_horizon - 1
            real_val = raw_traffic[target_idx] # (num_edges,)
            
            # 儲存結果
            all_real.append(real_val)
            all_pred.append(pred_real)

            # 只顯示前幾筆
            if i < display_limit:
                # 每個時間步只顯示幾個路段
                for edge_idx in range(min(1, len(edge_ids))):
                     edge_name = edge_ids[edge_idx]
                     r = real_val[edge_idx]
                     p = pred_real[edge_idx]
                     d = p - r
                     print(f"T+{i:<9} | {edge_name:<15} | {r:<5.0f} | {p:<5.1f} | {d:<5.1f}")
    
    print("... (Truncated) ...")
    print("-" * 60)
    
    # ==========================================
    # 4. 計算並顯示準確率報告
    # ==========================================
    if len(all_real) > 0:
        all_real = np.array(all_real) # (N_samples, num_edges)
        all_pred = np.array(all_pred) # (N_samples, num_edges)
        
        # 1. MAE
        diff = np.abs(all_pred - all_real)
        mae = np.mean(diff)
        
        # 2. RMSE
        mse = np.mean((all_pred - all_real) ** 2)
        rmse = np.sqrt(mse)
        
        # 3. Accuracy (Error < 5 vehicles)
        threshold = 5.0
        correct_count = np.sum(diff < threshold)
        total_count = diff.size 
        accuracy = (correct_count / total_count) * 100
        
        print("\n" + "="*60)
        print("       MODEL ACCURACY REPORT       ")
        print("="*60)
        print(f"Prediction Horizon      : {pred_horizon} steps (approx. {pred_horizon * 20 / 60:.1f} mins)")
        print(f"Total Samples Tested    : {len(all_real)}")
        print(f"Total Prediction Points : {total_count} (Samples * {len(edge_ids)} Edges)")
        print("-" * 30)
        print(f"Mean Absolute Error (MAE): {mae:.2f} vehicles")
        print(f"Root Mean Squared Error  : {rmse:.2f} vehicles")
        print(f"Accuracy (Error < {int(threshold)})  : {accuracy:.2f}%")
        print("="*60)

    print("\nTest completed.")
