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
MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model.pth")

# 訓練參數
INPUT_LEN = 6       # 輸入時間步長 (Lookback window)
HIDDEN_DIM = 256    # 隱藏層大小
NUM_LAYERS = 2      # GRU 層數
BATCH_SIZE = 64
EPOCHS = 20         # 應使用者要求，減少 Epochs 以節省時間
PATIENCE = 5        # 相應調整 Early Stopping 的耐心值
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
        # Input dim = num_edges (車流量) + 1 (時間特徵)
        self.gru = nn.GRU(
            input_size=num_edges + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_edges)
        # 用於確保輸出非負值 (車流量不可能是負的)
        self.relu = nn.ReLU() 

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_edges + 1)
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
    time_features = [] # 儲存對應的時間特徵
    
    # 紀錄時間範圍用於顯示
    start_time_str = None
    end_time_str = None

    # 讀取每個檔案並轉置
    for i, fname in enumerate(files):
        if (i+1) % 50 == 0: # 減少 print 頻率
            print(f"Processing file {i+1}/{len(files)}")
            
        path = os.path.join(data_dir, fname)
        try:
            # 這裡可以考慮使用 usecols 減少讀取量，但若要 pivot 比較難
            df = pd.read_csv(path)
            
            if "時間" in df.columns:
                df = df.rename(columns={
                    "時間": "time",
                    "路段ID": "edge_id",
                    "車輛數": "vehicle_count"
                })
            
            pivot = df.pivot(
                index="time",
                columns="edge_id",
                values="vehicle_count"
            ).fillna(0)
            
            pivot = pivot.sort_index()

            if all_edges is None:
                all_edges = sorted(pivot.columns.tolist())
                if not pivot.empty:
                    start_time_str = pivot.index[0]
            
            if not pivot.empty:
                end_time_str = pivot.index[-1]

            pivot = pivot.reindex(columns=all_edges, fill_value=0)
            
            # 立即轉為 float32 以節省記憶體
            series_list.append(pivot.values.astype(np.float32))
            
            # 計算該檔案的時間特徵 (假設一個檔案內的時間點差異不大，用檔名時間代表)
            # 或者更精確：每個 row 其實有自己的時間，但這裡簡化為檔案時間
            t_feat = get_time_feature(fname)
            # 擴充成與 series 相同長度 (n_steps, 1)
            t_feat_array = np.full((len(pivot), 1), t_feat, dtype=np.float32)
            time_features.append(t_feat_array)

        except Exception as e:
            print(f"Error processing file {fname}: {e}")
            continue

    if not series_list:
        raise ValueError("No valid data loaded.")

    # 合併所有時間序列
    print("Concatenating data...")
    raw_traffic = np.concatenate(series_list, axis=0) # (Total_Timesteps, Num_Edges)
    raw_time = np.concatenate(time_features, axis=0)  # (Total_Timesteps, 1)
    
    # 將車流與時間特徵合併 -> (Total_Timesteps, Num_Edges + 1)
    # raw_data = np.hstack([raw_traffic, raw_time]) 
    # 不要在這裡合併，因為 scaler 只應該對 traffic 做
    
    print(f"Traffic Data Shape: {raw_traffic.shape}")
    print(f"Time Range: {start_time_str} -> {end_time_str}")
    
    return raw_traffic, raw_time, all_edges

# =================================================
# 3. 自定義 Dataset (記憶體優化關鍵)
# =================================================
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, traffic_data, time_data, input_len):
        """
        traffic_data: (T, num_edges)
        time_data: (T, 1)
        """
        # 合併特徵: [Traffic, Time]
        # 注意：Traffic 已經正規化，Time 也已經是 0~1
        self.data = torch.tensor(np.hstack([traffic_data, time_data]), dtype=torch.float32)
        self.input_len = input_len
        self.num_samples = len(traffic_data) - input_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # X: [Traffic + Time] (seq_len, num_edges + 1)
        # y: [Traffic Only]   (num_edges) -> 我們只預測車流量，不預測時間
        
        input_seq = self.data[idx : idx+self.input_len]
        
        # Target 取下一個時間點的 "Traffic" 部分 (排除最後一個 channel)
        # self.data[idx+self.input_len] 是 (num_edges + 1)
        # 我們只要前 num_edges
        target = self.data[idx+self.input_len][:-1] 
        
        return input_seq, target

# =================================================
# 4. 主程式
# =================================================
if __name__ == "__main__":
    if MAX_FILES_TO_LOAD is None:
        print("Running in FULL DATA mode. This might take a while...")
    
    # 載入資料
    try:
        raw_traffic, raw_time, all_edges = load_and_process_data(DATA_DIR, MAX_FILES_TO_LOAD)
        num_edges = len(all_edges)
        print(f"Total Edges: {num_edges}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit(1)

    # 正規化 (只對車流量做)
    print("Normalizing data (MinMaxScaler)...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # 注意：fit_transform 會產生一個新的 copy，如果記憶體還是不夠，可以考慮 inplace 操作或分批
    raw_traffic = scaler.fit_transform(raw_traffic).astype(np.float32)

    # 建立 Dataset 與 DataLoader
    print("Creating datasets (Indices split)...")
    # 我們不切分資料本身，而是切分「索引」
    total_samples = len(raw_traffic) - INPUT_LEN
    indices = np.arange(total_samples)
    
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, shuffle=False # 時間序列通常不 shuffle，或者如果要 shuffle 需小心
    )
    
    # 建立 Dataset 實體 (它們共享同一份 raw_data 記憶體)
    full_dataset = TrafficDataset(raw_traffic, raw_time, INPUT_LEN)
    
    # 使用 Subset 來建立 Train/Val Dataset
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # DataLoader (設定 num_workers=0 以避免 Windows 上多程序的額外記憶體開銷)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Validation 不用 shuffle
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 
    
    # 預先將 Validation Data 轉為 Tensor 放到 GPU (如果有空間) 
    # 若記憶體不足，這裡要改回用 Loop 驗證，不要一次載入
    # 為了安全起見，我們在 Validation Loop 中也是用 Batch 讀取
    
    # 模型初始化
    model = GRUTraffic(num_edges, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
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
        
        # 使用 DataLoader 迭代
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_dataset)

        # Validation (改用 Loader 以節省記憶體)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(DEVICE), val_y.to(DEVICE)
                val_pred = model(val_X)
                val_loss += loss_fn(val_pred, val_y).item() * val_X.size(0)
            val_loss /= len(val_dataset)

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
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,     # 儲存 Scaler 以便反正規化
        "edge_ids": all_edges,# 儲存路段順序
        "config": {           # 儲存模型超參數
            "input_len": INPUT_LEN,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS
        }
    }, MODEL_PATH)

    print(f"All saved to: {MODEL_PATH}")
    print("你可以使用這個檔案在另一個腳本中載入並進行預測與紅綠燈控制。")
