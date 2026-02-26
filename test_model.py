import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import joblib

# =================================================
# 1. 類別定義 (必須與 train_model.py 一致)
# =================================================
class Log1pScaler:
    def fit(self, data):
        pass 
    def transform(self, data):
        return np.log1p(data)
    def inverse_transform(self, data):
        return np.expm1(data)

class GRUSequence(nn.Module):
    def __init__(self, num_edges, hidden_dim, num_layers, horizon, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.num_edges = num_edges
        
        # Input: Edges + 2 Time Features
        self.gru = nn.GRU(
            input_size=num_edges + 2, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_edges * horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :] 
        pred_flat = self.fc(last)
        pred_seq = pred_flat.view(-1, self.horizon, self.num_edges)
        return pred_seq

# =================================================
# 2. 工具
# =================================================
def extract_datetime(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        return m.group(1) + m.group(2)
    return None

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {})
    if "edge_ids" in checkpoint:
        edge_ids = checkpoint["edge_ids"]
    else:
        edge_ids = checkpoint.get("edge_ids_list", [])

    num_edges = len(edge_ids)
    input_len = config.get("input_len", 15)
    pred_horizon = config.get("pred_horizon", 15)
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    scaler = checkpoint.get("scaler")
    
    print(f"Config: input_len={input_len}, horizon={pred_horizon}, hidden={hidden_dim}")
    print(f"Scaler: {type(scaler)}")

    model = GRUSequence(num_edges, hidden_dim, num_layers, pred_horizon).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, scaler, edge_ids, input_len, pred_horizon

def load_sample_data_generator(data_dir, edge_ids, input_len, num_files=50):
    files = [f for f in os.listdir(data_dir) if f.startswith("traffic_data_") and f.endswith(".csv")]
    files = sorted(files, key=lambda x: extract_datetime(x))
    
    if num_files > 0:
        files = files[-num_files:]

    print(f"Loading {len(files)} files for testing...")
    
    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            if "時間" in df.columns:
                df = df.rename(columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"})
            
            pivot = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
            pivot = pivot.reindex(columns=edge_ids, fill_value=0)
            
            # Time Feature
            t_str = extract_datetime(fname)
            if t_str:
                hh = int(t_str[8:10])
                mm = int(t_str[10:12])
                ss = int(t_str[12:14])
                start_sec = hh * 3600 + mm * 60 + ss
            else:
                start_sec = 0

            time_steps = np.arange(len(pivot)) * 20 + start_sec
            theta = 2 * np.pi * time_steps / (24 * 3600)
            t_arr = np.stack([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)
            
            yield pivot.values.astype(np.float32), t_arr
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

# =================================================
# 3. 主程式
# =================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
    DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_check_data") # Changed per user request
    MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 測試設定 (Test Configuration)
    # ==========================================
    # 設定要測試「幾分鐘後」的預測準確度
    # 例如: 3.0 代表測試 3 分鐘後的預測 (T+3min)
    # 不可超過 PRED_HORIZON * 20秒 (目前 15*20 = 300s = 5min)
    TEST_MINUTES = 3.0 
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        exit(1)

    model, scaler, edge_ids, input_len, pred_horizon = load_model(MODEL_PATH, DEVICE)
    
    # Calculate target step index
    # step 0 = +20s, step 1 = +40s, ..., step k = (k+1)*20s
    # min = (k+1)*20 / 60 => k+1 = min*3 => k = min*3 - 1
    target_step_idx = int(TEST_MINUTES * 60 / 20) - 1
    
    if target_step_idx < 0:
        target_step_idx = 0 # Minimum +20s
    if target_step_idx >= pred_horizon:
        print(f"Warning: TEST_MINUTES {TEST_MINUTES} exceeds model horizon ({pred_horizon*20/60:.1f} min). Using max horizon.")
        target_step_idx = pred_horizon - 1
        
    print(f"\nTarget Verification: +{TEST_MINUTES} min (Step {target_step_idx+1})")
    
    print("\nStarting Test (Lazy Sequence Prediction)...")
    
    all_mae = []
    peak_mae_list = []
    
    # Check specific step metrics
    step_mae = []
    step_peak_mae = []
    
    # Process file by file to avoid memory issues
    data_gen = load_sample_data_generator(DATA_DIR, edge_ids, input_len, num_files=50)
    
    with torch.no_grad():
        for traffic_data, time_data in data_gen:
            # Check length
            T = len(traffic_data)
            if T <= input_len + pred_horizon:
                continue
                
            # Log1p Transform
            scaled_traffic = scaler.transform(traffic_data)
            
            # Test a few samples per file (e.g., stride 5)
            for i in range(0, T - input_len - pred_horizon, 5):
                input_traf = scaled_traffic[i : i+input_len]
                input_time = time_data[i : i+input_len]
                
                input_comb = np.hstack([input_traf, input_time])
                input_tensor = torch.tensor(input_comb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # Predict
                pred_seq_scaled = model(input_tensor).cpu().numpy()[0]
                pred_seq_real = scaler.inverse_transform(pred_seq_scaled)
                
                # Target
                target_seq_real = traffic_data[i+input_len : i+input_len+pred_horizon]
                
                # 1. Overall Sequence Metrics
                abs_diff = np.abs(pred_seq_real - target_seq_real)
                all_mae.append(np.mean(abs_diff))
                
                mask = target_seq_real > 10
                if np.sum(mask) > 0:
                    peak_mae_list.append(np.mean(abs_diff[mask]))
                    
                # 2. Specific Step Metrics (Time Horizon Check)
                pred_step = pred_seq_real[target_step_idx]
                target_step = target_seq_real[target_step_idx]
                
                step_diff = np.abs(pred_step - target_step)
                step_mae.append(np.mean(step_diff))
                
                mask_step = target_step > 10
                if np.sum(mask_step) > 0:
                    step_peak_mae.append(np.mean(step_diff[mask_step]))

    if not all_mae:
        print("No valid samples tested.")
        exit(0)

    mean_mae = np.mean(all_mae)
    mean_peak_mae = np.mean(peak_mae_list) if peak_mae_list else 0.0
    
    mean_step_mae = np.mean(step_mae)
    mean_step_peak_mae = np.mean(step_peak_mae) if step_peak_mae else 0.0
    
    print("\n" + "="*60)
    print("       SEQUENCE MODEL REPORT       ")
    print("="*60)
    print(f"Sequence Horizon : {pred_horizon} steps (Total {pred_horizon*20} sec)")
    print(f"overall Avg MAE  : {mean_mae:.2f}")
    print(f"overall Peak MAE : {mean_peak_mae:.2f} (>10)")
    print("-" * 60)
    print(f"Specific Check   : +{TEST_MINUTES} Minutes (Step {target_step_idx+1})")
    print(f"Step Avg MAE     : {mean_step_mae:.2f}")
    print(f"Step Peak MAE    : {mean_step_peak_mae:.2f} (>10)")
    print("="*60)
    print("Test Complete.")
