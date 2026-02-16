import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用於儲存 scaler，或者直接用 torch.save 也可以

# =================================================
# 0. 設定 (Configuration)
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")

# 訓練參數
INPUT_LEN = 6       # 輸入時間步長 (Lookback window)
HIDDEN_DIM = 256    # 隱藏層大小
NUM_LAYERS = 2      # GRU 層數
BATCH_SIZE = 64
EPOCHS = 100        # 設定一個較大的上限，搭配 Early Stopping
PATIENCE = 10       # 若 Val Loss 連續 10 個 Epoch 沒有改善就停止
LR = 1e-3
DROPOUT = 0.2

# 測試用設定：限制讀取檔案數量
# 如果要訓練所有資料，請設為 None
MAX_FILES_TO_LOAD = None 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =================================================
# 1. 建立 GRU 模型 (Model Definition)
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
        # 用於確保輸出非負值 (車流量不可能是負的)
        self.relu = nn.ReLU() 

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_edges)
        out, _ = self.gru(x)
        # 取最後一個時間點的輸出
        last = out[:, -1, :]
        # 全連接層預測
        pred = self.fc(last)
        return self.relu(pred)

# =================================================
# 2. 資料處理工具函式 (Data Utils)
# =================================================
def extract_datetime(filename):
    """從檔名提取時間戳記以便排序"""
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        return m.group(1) + m.group(2)
    return None

def load_and_process_data(data_dir, max_files=None):
    print("Loading data...")
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith("traffic_data_") and f.endswith(".csv")
    ]
    
    # 排序檔案
    files = sorted(files, key=lambda x: extract_datetime(x))
    
    if not files:
        raise FileNotFoundError(f"No traffic data files found in {data_dir}")

    print(f"Total files found: {len(files)}")

    # 限制檔案數量 (測試用)
    if max_files is not None:
        print(f"Testing Mode: Loading only first {max_files} files...")
        files = files[:max_files]
        print(f"Files to load: {files[0]} ... {files[-1]}")
    else:
        print(f"Loading all {len(files)} files...")

    all_edges = None
    series_list = []
    
    # 紀錄時間範圍
    start_time_str = None
    end_time_str = None

    # 讀取每個檔案並轉置
    for i, fname in enumerate(files):
        if (i+1) % 10 == 0:
            print(f"Processing file {i+1}/{len(files)}")
            
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            
            # 欄位名稱映射 (處理中文欄位)
            if "時間" in df.columns:
                df = df.rename(columns={
                    "時間": "time",
                    "路段ID": "edge_id",
                    "車輛數": "vehicle_count"
                })
            
            # 轉置：Index=Time, Columns=EdgeID
            pivot = df.pivot(
                index="time",
                columns="edge_id",
                values="vehicle_count"
            ).fillna(0)
            
            # 排序時間索引
            pivot = pivot.sort_index()

            # 第一次讀取時確定所有路段 ID
            if all_edges is None:
                all_edges = sorted(pivot.columns.tolist())
                # 紀錄第一筆資料的時間
                if not pivot.empty:
                    start_time_str = pivot.index[0]
            
            # 紀錄最後一筆資料的時間
            if not pivot.empty:
                end_time_str = pivot.index[-1]

            # 對齊路段 (Reindex 確保所有 DataFrame 欄位順序一致，缺少的補0)
            pivot = pivot.reindex(columns=all_edges, fill_value=0)
            
            series_list.append(pivot.values)
            
        except Exception as e:
            print(f"Error processing file {fname}: {e}")
            continue

    if not series_list:
        raise ValueError("No valid data loaded.")

    # 合併所有時間序列
    raw_data = np.concatenate(series_list, axis=0) # (Total_Timesteps, Num_Edges)
    print(f"Data loaded. Shape: {raw_data.shape}")
    print(f"Time Range: {start_time_str} -> {end_time_str}")
    
    return raw_data, all_edges

def create_sliding_windows(data, input_len):
    X, y = [], []
    # 預測 t+1 時刻，需要資料 [t-input_len : t]
    for i in range(len(data) - input_len):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len])
    return np.array(X), np.array(y)

# =================================================
# 3. 主程式 (Main Execution)
# =================================================
if __name__ == "__main__":
    # --- 步驟 1: 讀取資料 ---
    raw_data, edge_ids = load_and_process_data(DATA_DIR, MAX_FILES_TO_LOAD)
    num_edges = len(edge_ids)
    print(f"Total Edges: {num_edges}")

    # --- 步驟 2: 資料正規化 (Normalization) ---
    print("Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 對每個路段 (Column) 獨立進行 MinMax Scaling
    scaled_data = scaler.fit_transform(raw_data)

    # --- 步驟 3: 製作訓練資料集 (Sliding Window) ---
    print("Creating datasets...")
    X, y = create_sliding_windows(scaled_data, INPUT_LEN)
    print(f"Dataset shape - X: {X.shape}, y: {y.shape}")

    # 分割訓練集與驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False # 時間序列資料通常不打亂順序，或者只在 Window 層級打亂
    )

    # 轉為 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    # 建立 DataLoader (Optional, 但建議使用 batch training)
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 步驟 4: 初始化模型 ---
    model = GRUTraffic(
        num_edges=num_edges, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- 步驟 5: 訓練迴圈 ---
    print(f"\nStart training for {EPOCHS} epochs (with Early Stopping)...")
    min_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", end="")

        # Early Stopping 機制
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
            # 儲存最佳模型狀態 (暫存)
            best_model_state = model.state_dict()
            print(" * Best Model")
        else:
            patience_counter += 1
            print(f" | Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered! Valid loss has not improved for {PATIENCE} epochs.")
            # 回復最佳模型權重
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
            
    # 如果訓練結束都沒有觸發 Early Stopping (或者觸發後已經載回)，確認最終模型狀態
    if best_model_state is not None and patience_counter < PATIENCE:
         # 確保最後儲存的是最佳模型 (防呆)
         model.load_state_dict(best_model_state)
    
    # --- 步驟 6: 儲存完整模型與變數 ---
    print("\nSaving model and metadata...")
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,           # 儲存 MinMaxScaler 物件
        "edge_ids": edge_ids,       # 儲存路段 ID 對照表
        "config": {
            "input_len": INPUT_LEN,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_edges": num_edges
        }
    }

    torch.save(save_dict, MODEL_SAVE_PATH)
    print(f"All saved to: {MODEL_SAVE_PATH}")
    print("你可以使用這個檔案在另一個腳本中載入並進行預測與紅綠燈控制。")
