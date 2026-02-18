import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import sys

# =================================================
# 0. 設定 (Hyperparameters)
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")

INPUT_LEN = 15          # Lookback
PRED_HORIZON = 15       # Predict Sequence (15 steps)
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
LR = 1e-3
DROPOUT = 0.2

# Optimization: Limit files if needed. None = Load All.
# If still OOM, set this to 1000 or 2000.
MAX_FILES_TO_LOAD = None 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =================================================
# 1. 工具 (Utils: Time Features & Log1p)
# =================================================
def extract_datetime(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        return m.group(1) + m.group(2)
    return None

def load_one_file(path, fname):
    df = pd.read_csv(path)
    if "時間" in df.columns:
        df = df.rename(columns={"時間":"time","路段ID":"edge_id","車輛數":"vehicle_count"})
    pivot = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
    pivot = pivot.sort_index()
    
    # Extract start time from filename
    time_str = extract_datetime(fname)
    if time_str:
        hh = int(time_str[8:10])
        mm = int(time_str[10:12])
        ss = int(time_str[12:14])
        start_seconds = hh * 3600 + mm * 60 + ss
    else:
        start_seconds = 0

    # Create time steps (assuming 20s interval)
    time_steps = np.arange(len(pivot)) * 20 + start_seconds
    seconds_in_day = 24 * 3600
    theta = 2 * np.pi * time_steps / seconds_in_day
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    time_feat = np.stack([sin_t, cos_t], axis=1).astype(np.float32)
    
    return pivot, time_feat

class Log1pScaler:
    def fit(self, data):
        pass 
    def transform(self, data):
        return np.log1p(data)
    def inverse_transform(self, data):
        return np.expm1(data)

# =================================================
# 2. 讀取與前處理 (Store in List, Do NOT Concat)
# =================================================
print("Loading files...")
files = [f for f in os.listdir(DATA_DIR) if f.startswith("traffic_data_") and f.endswith(".csv")]
files = sorted(files, key=lambda x: extract_datetime(x))

if MAX_FILES_TO_LOAD:
    files = files[:MAX_FILES_TO_LOAD]

traffic_pieces = [] # List of numpy arrays
time_pieces = []    # List of numpy arrays
all_edges = None

for fname in files:
    try:
        p, t = load_one_file(os.path.join(DATA_DIR, fname), fname)
        if all_edges is None:
            all_edges = sorted(p.columns.tolist())
        
        # Ensure alignment
        p = p.reindex(columns=all_edges, fill_value=0)
        
        # Log1p Transform Immediately to save memory (float32)
        # Log1p is element-wise, so we can do it file by file
        traf_data = np.log1p(p.values.astype(np.float32))
        
        traffic_pieces.append(traf_data)
        time_pieces.append(t)
    except Exception as e:
        print(f"Skipping error file {fname}: {e}")

num_edges = len(all_edges)
print(f"Loaded {len(traffic_pieces)} files. Total edges: {num_edges}")

# We do NOT run MinMaxScaler. fit is implicit for Log1p.
scaler = Log1pScaler()

# =================================================
# 3. Lazy Dataset (Memory Efficient)
# =================================================
class LazySequenceDataset(Dataset):
    def __init__(self, traffic_list, time_list, input_len, pred_horizon):
        self.traffic_list = traffic_list
        self.time_list = time_list
        self.input_len = input_len
        self.pred_horizon = pred_horizon
        
        # Create Index Map: (file_idx, start_idx)
        # Only valid start indices
        self.index_map = []
        
        print("Building Index Map...")
        # Stride = 3 to reduce data size (User Suggestion)
        stride = 3
        
        # Pre-calculate log1p(5) for filtering
        # 5 vehicles is a good threshold for "active traffic"
        threshold_val = np.log1p(5).astype(np.float32)
        
        for file_i, traf in enumerate(traffic_list):
            T = len(traf)
            
            if T < input_len + pred_horizon:
                continue
                
            max_start = T - input_len - pred_horizon
            
            # Optimization: Check if file has ANY significant traffic
            if np.max(traf) < threshold_val: 
                continue
                
            for i in range(0, max_start + 1, stride):
                # Window Check: Check max in the TARGET sequence
                # We want the model to learn to predict meaningful values
                # If target is all < 5, it might just learn noise/zero.
                
                target_window = traf[i + input_len : i + input_len + pred_horizon]
                
                if np.max(target_window) < threshold_val:
                    continue
                    
                self.index_map.append((file_i, i))
                
        print(f"Dataset ready. Total samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_i, start_i = self.index_map[idx]
        
        # Retrieve full arrays from list (Reference only, cheap)
        full_traf = self.traffic_list[file_i]
        full_time = self.time_list[file_i]
        
        # Slice on-the-fly
        # Input: [start : start + input]
        x_traf = full_traf[start_i : start_i + self.input_len]
        x_time = full_time[start_i : start_i + self.input_len]
        
        # Target: [start + input : start + input + horizon]
        target_start = start_i + self.input_len
        y_target = full_traf[target_start : target_start + self.pred_horizon]
        
        # Combined Input
        x_comb = np.hstack([x_traf, x_time])
        
        return torch.tensor(x_comb, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

dataset = LazySequenceDataset(traffic_pieces, time_pieces, INPUT_LEN, PRED_HORIZON)

if len(dataset) == 0:
    print("Error: Dataset empty.")
    exit(1)

# Split indices for Train/Val (Shuffle indices, not data)
train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.1, shuffle=True, random_state=42)

# Use Subset with original Lazy dataset
train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

# num_workers=0 to avoid pickling overhead on Windows, or use small number
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# =================================================
# 4. Model (Sequence Output)
# =================================================
class GRUSequence(nn.Module):
    def __init__(self, num_edges, hidden_dim, num_layers, horizon, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.num_edges = num_edges
        
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

model = GRUSequence(num_edges, HIDDEN_DIM, NUM_LAYERS, PRED_HORIZON, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =================================================
# 5. Weighted Sequence Loss
# =================================================
def weighted_sequence_loss(pred, target):
    # Log1p of 10 cars ~ 2.4
    threshold = 2.4 
    weights = 1.0 + 4.0 * (target > threshold).float()
    loss = (pred - target) ** 2
    loss = loss * weights
    return loss.mean()

# =================================================
# 6. Training Loop
# =================================================
print(f"Start Training V2 (Lazy Dataset, Log1p)...")
min_val = float("inf")
best_state = None
pat = 0

for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    
    for i, (bx, by) in enumerate(train_loader):
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        pred = model(bx)
        loss = weighted_sequence_loss(pred, by)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * bx.size(0)
        
        if i % 500 == 0: # Print less frequently
            p_mean = pred.mean().item()
            # print(f"  Batch {i}: Loss={loss.item():.4f}")

    tr_loss /= len(train_indices)
    
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            vp = model(vx)
            va_loss += weighted_sequence_loss(vp, vy).item() * vx.size(0)
    va_loss /= len(val_indices)
    
    print(f"Epoch {epoch+1:03d} | Train {tr_loss:.6f} | Val {va_loss:.6f}", end="")
    
    if va_loss < min_val:
        min_val = va_loss
        best_state = {
            "model_state_dict": model.state_dict(),
            "scaler_type": "log1p", 
            "scaler": scaler,
            "edge_ids": all_edges,
            "config": {
                "input_len": INPUT_LEN,
                "pred_horizon": PRED_HORIZON,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "model_type": "gru_sequence_log1p"
            }
        }
        print(" * Best")
        pat = 0
    else:
        print(f" | Pat {pat+1}/{PATIENCE}")
        pat += 1
        if pat >= PATIENCE:
            print("\nEarly stopping.")
            break

if best_state:
    torch.save(best_state, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")
else:
    print("Training failed.")
