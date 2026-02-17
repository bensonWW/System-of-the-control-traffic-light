import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset

# =================================================
# 0. 設定
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")

INPUT_LEN = 15          # 建議 15 (= 5 分鐘 if 20s/step)
PRED_HORIZON = 15       # 建議 15 (= 預測 5 分鐘後)
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
DROPOUT = 0.2

MAX_FILES_TO_LOAD = None  # 測試可設 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =================================================
# 1. 工具
# =================================================
def extract_datetime(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        return m.group(1) + m.group(2)
    return None

def load_one_file(path):
    df = pd.read_csv(path)
    if "時間" in df.columns:
        df = df.rename(columns={"時間":"time","路段ID":"edge_id","車輛數":"vehicle_count"})
    pivot = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
    pivot = pivot.sort_index()
    return pivot

# =================================================
# 2. 讀取檔案清單
# =================================================
files = [f for f in os.listdir(DATA_DIR) if f.startswith("traffic_data_") and f.endswith(".csv")]
files = sorted(files, key=lambda x: extract_datetime(x))

if not files:
    raise FileNotFoundError(f"No traffic_data files found in {DATA_DIR}")

if MAX_FILES_TO_LOAD is not None:
    files = files[:MAX_FILES_TO_LOAD]

print("Total files:", len(files))

# =================================================
# 3. 第一輪：建立 all_edges（全域一致）
# =================================================
all_edges = None
pivots = []

for i, fname in enumerate(files):
    p = load_one_file(os.path.join(DATA_DIR, fname))
    if all_edges is None:
        all_edges = sorted(p.columns.tolist())
    p = p.reindex(columns=all_edges, fill_value=0)
    pivots.append(p)

num_edges = len(all_edges)
print("Total edges:", num_edges)

# =================================================
# 4. 將所有檔案的 traffic 合併後做 scaler（只 fit traffic）
#    注意：這裡先串起來是為了 fit scaler，不用來做 window
# =================================================
raw_traffic_concat = np.concatenate([p.values.astype(np.float32) for p in pivots], axis=0)
scaler = MinMaxScaler()
raw_traffic_concat = scaler.fit_transform(raw_traffic_concat).astype(np.float32)

# 把正規化後的資料再切回每個檔案（避免跨檔案 window）
lengths = [len(p) for p in pivots]
norm_pieces = []
cursor = 0
for L in lengths:
    norm_pieces.append(raw_traffic_concat[cursor:cursor+L])
    cursor += L

# =================================================
# 5. Dataset：每檔案獨立產生 windows（不跨檔案）
# =================================================
class MultiFileTrafficDataset(Dataset):
    def __init__(self, norm_pieces, input_len, pred_horizon):
        self.X = []
        self.y = []
        self.input_len = input_len
        self.pred_horizon = pred_horizon

        for arr in norm_pieces:
            T = len(arr)
            # 需要至少 input_len + pred_horizon 的長度
            if T <= input_len + pred_horizon:
                continue
            for i in range(T - input_len - pred_horizon):
                self.X.append(arr[i:i+input_len])
                self.y.append(arr[i+input_len+pred_horizon])

        self.X = np.array(self.X, dtype=np.float32)  # [N, input_len, num_edges]
        self.y = np.array(self.y, dtype=np.float32)  # [N, num_edges]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

dataset = MultiFileTrafficDataset(norm_pieces, INPUT_LEN, PRED_HORIZON)
print("Total samples:", len(dataset))
if len(dataset) < 10000:
    print("Warning: samples are low; consider longer files or smaller INPUT_LEN/HORIZON.")

# =================================================
# 6. Train/Val split（不 shuffle 時序也可，但這裡以樣本切分；若你要更嚴格可用檔案切分）
# =================================================
indices = np.arange(len(dataset))
train_idx, val_idx = train_test_split(indices, test_size=0.1, shuffle=False)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Train samples:", len(train_ds), "Val samples:", len(val_ds))

# =================================================
# 7. GRU 模型
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

model = GRUTraffic(num_edges, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# =================================================
# 8. Train + Early Stopping
# =================================================
print(f"Start training: input_len={INPUT_LEN}, horizon={PRED_HORIZON} (each step=20s)")
min_val = float("inf")
best_state = None
pat = 0

for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0

    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * bx.size(0)

    tr_loss /= len(train_ds)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            vp = model(vx)
            va_loss += loss_fn(vp, vy).item() * vx.size(0)
    va_loss /= len(val_ds)

    print(f"Epoch {epoch+1:03d} | Train {tr_loss:.6f} | Val {va_loss:.6f}", end="")

    if va_loss < min_val:
        min_val = va_loss
        best_state = model.state_dict()
        pat = 0
        print(" * Best")
    else:
        pat += 1
        print(f" | Patience {pat}/{PATIENCE}")

    if pat >= PATIENCE:
        print("Early stopping.")
        break

if best_state is not None:
    model.load_state_dict(best_state)

# =================================================
# 9. Save model + scaler + metadata
# =================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler": scaler,
    "edge_ids": all_edges,
    "config": {
        "input_len": INPUT_LEN,
        "pred_horizon": PRED_HORIZON,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }
}, MODEL_PATH)

print("Saved to:", MODEL_PATH)
