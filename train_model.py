"""
train_model.py — Pair-based 5 分鐘車流預測 GRU 模型

訓練範式：給定一輪 SUMO 模擬暖機後的狀態 (CSV_A 前 15 步, time 60-340s)，
預測 5 分鐘後另一輪 SUMO 模擬的暖機後狀態 (CSV_B 前 15 步)。
Pair 由同日內檔名相鄰、timestamp 差 [3, 15] 分鐘的 CSV 對構成。

支援三種模式 (CLI):
  python train_model.py                                      # 訓練 (預設)
  python train_model.py --mode eval --input A.csv --target B.csv [--gap 5.0]
  python train_model.py --mode eval-batch --dir data/simulation_data/
  python train_model.py --mode eval-batch --dir ... --dry-run    # 只看 pair 統計

新模型存到 gru_traffic_model_pair.pth (不覆寫舊的 gru_traffic_model.pth)，
inference 整合方式詳見 INTEGRATION_PAIR_MODEL.md。
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =================================================
# 0. 設定
# =================================================
# Reproducibility — without this, every training run produces different
# weights, which propagates to different signal plans at inference.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model_pair_v2.pth")
EDGE_UNION_CACHE = os.path.join(BASE_DIR, "data", "edge_ids_union.json")

INPUT_LEN = 15        # 5 分鐘 (20 秒/步)
PRED_HORIZON = 15     # 5 分鐘
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 60
PATIENCE = 8
LR = 3e-4
DROPOUT = 0.2

GAP_MIN_SEC = 180     # pair 最小時間差 (3 分鐘)
GAP_MAX_SEC = 900     # pair 最大時間差 (15 分鐘)

# v2: 2 channels per edge (vehicle_count + avg_speed_kmh)
# v1 用 OUTPUT_CHANNELS=1; 保留變數讓未來方便回到 v1 或擴充更多 channel
OUTPUT_CHANNELS = 2
SPEED_LOSS_WEIGHT = 0.5   # speed channel 在 total loss 內的權重 (count 部分仍有 5× 高流量加權)
# 註: speed loss 用 has-vehicle mask 後,量級從「3450 cells mean」變「~350 cells mean」
# 約大 10×。若仍用 1.0 會 dominate total loss 害到 count 學習;0.5 為折衷起點,
# count_loss vs speed_loss contribution 大致同量級。實證若不夠平衡再 tune。

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================
# 1. 工具 (Utils)
# =================================================
def extract_datetime(filename):
    """從檔名抓 timestamp 字串 (YYYYMMDDHHMMSS)。"""
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        return m.group(1) + m.group(2)
    return None


def parse_timestamp(filename):
    """從檔名解析為 datetime 物件。"""
    ts = extract_datetime(filename)
    if ts is None:
        return None
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S")
    except ValueError:
        return None


class Log1pScaler:
    """log1p 縮放,與舊版相容 (predict_to_csv.py 會 unpickle 此類別)。"""
    def fit(self, data):
        pass

    def transform(self, data):
        return np.log1p(data)

    def inverse_transform(self, data):
        return np.expm1(data)


def build_time_features(filename, num_steps, gap_minutes):
    """
    建構 time features: shape = (num_steps, 3)
        欄 0: sin(time-of-day)
        欄 1: cos(time-of-day)
        欄 2: gap_minutes (廣播到全部 steps)
    """
    time_str = extract_datetime(filename)
    if time_str:
        hh = int(time_str[8:10])
        mm = int(time_str[10:12])
        ss = int(time_str[12:14])
        start_seconds = hh * 3600 + mm * 60 + ss
    else:
        start_seconds = 0

    time_steps = np.arange(num_steps) * 20 + start_seconds
    theta = 2 * np.pi * time_steps / (24 * 3600)
    sin_t = np.sin(theta).astype(np.float32)
    cos_t = np.cos(theta).astype(np.float32)
    gap_col = np.full(num_steps, float(gap_minutes), dtype=np.float32)
    return np.stack([sin_t, cos_t, gap_col], axis=1)


# =================================================
# 2. Pair 選擇
# =================================================
def find_csv_pairs(dir_path, min_gap_sec=GAP_MIN_SEC, max_gap_sec=GAP_MAX_SEC):
    """
    從資料夾找出所有「同日、時間差在範圍內」的 (A, B) CSV pair。

    對每個起點 A,往後掃描所有 B,只要 gap 落在 [min_gap_sec, max_gap_sec]
    都收錄;一旦 gap 超過上限就提早跳出 (因檔案已按時間升序)。

    這對「密集取樣」資料 (例如 2 分鐘一筆) 特別重要 —— 若只取相鄰 pair,
    幾乎所有 gap 都會落在 1-3 分鐘區間而被 min_gap_sec=180 過濾掉。

    Returns:
        list of (path_A, path_B, gap_minutes)
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"找不到資料夾: {dir_path}")

    files = [
        f for f in os.listdir(dir_path)
        if f.startswith("traffic_data_") and f.endswith(".csv")
    ]

    by_date = defaultdict(list)
    for f in files:
        dt = parse_timestamp(f)
        if dt is None:
            continue
        by_date[dt.strftime("%Y%m%d")].append((dt, f))

    pairs = []
    for items in by_date.values():
        items.sort()  # 按 datetime 升序
        for i in range(len(items)):
            dt_a, f_a = items[i]
            for j in range(i + 1, len(items)):
                dt_b, f_b = items[j]
                gap_sec = (dt_b - dt_a).total_seconds()
                if gap_sec > max_gap_sec:
                    break  # 因 items 已排序,後面只會更大,提早終止
                if gap_sec >= min_gap_sec:
                    pairs.append((
                        os.path.join(dir_path, f_a),
                        os.path.join(dir_path, f_b),
                        gap_sec / 60.0,
                    ))
    return pairs


# =================================================
# 3. Edge ID Union (canonical edge list)
# =================================================
def build_edge_union(csv_paths, cache_path=EDGE_UNION_CACHE, rebuild=False):
    """
    掃描所有 CSV 收集 edge_id 聯集,結果寫到 cache。
    再次執行直接讀 cache (除非 rebuild=True)。
    """
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "r", encoding="utf-8") as f:
            edges = json.load(f)
        print(f"Loaded edge union from cache ({len(edges)} edges): {cache_path}")
        return edges

    print(f"Scanning {len(csv_paths)} CSV files for edge_id union ...")
    edge_set = set()
    for p in tqdm(csv_paths, desc="Edge scan"):
        try:
            df = pd.read_csv(p, usecols=["edge_id"])
            edge_set.update(df["edge_id"].unique().tolist())
        except Exception as exc:
            print(f"  skip {os.path.basename(p)}: {exc}")
            continue

    edges = sorted(edge_set)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False)
    print(f"Edge union: {len(edges)} edges saved to {cache_path}")
    return edges


# =================================================
# 4. CSV 載入 (取前 N 步,對齊 edge)
# =================================================
def load_csv_first_n_steps(csv_path, edge_ids, n_steps, output_channels=1):
    """
    讀 CSV → pivot (time, edge_id) → 對齊 edge_ids → 取前 n_steps 步。

    output_channels:
        1 → 只取 vehicle_count            返回 shape = (n_steps, num_edges)
        2 → vehicle_count + avg_speed_kmh 返回 shape = (n_steps, num_edges*2)
            前 num_edges 欄是 count, 後 num_edges 欄是 speed

    Returns:
        (ndarray, fname) 或 None (資料不足/讀取失敗/缺少必要欄位)
    """
    try:
        df = pd.read_csv(csv_path)
        if "時間" in df.columns:
            df = df.rename(columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"})

        pivot_count = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
        pivot_count = pivot_count.sort_index().reindex(columns=edge_ids, fill_value=0)

        if output_channels == 1:
            if len(pivot_count) < n_steps:
                return None
            arr = pivot_count.iloc[:n_steps].values.astype(np.float32)
            return arr, os.path.basename(csv_path)

        # output_channels == 2: 也 pivot speed,對齊到同個 time index 後 concat
        if "avg_speed_kmh" not in df.columns:
            return None  # 舊格式 CSV 不能用於 v2 訓練
        pivot_speed = df.pivot(index="time", columns="edge_id", values="avg_speed_kmh").fillna(0)
        pivot_speed = pivot_speed.sort_index().reindex(columns=edge_ids, fill_value=0)
        # 取兩者共有的 time index 對齊 (理論上 count CSV 寫得齊時兩者 index 一致)
        common_idx = pivot_count.index.intersection(pivot_speed.index)
        pivot_count = pivot_count.loc[common_idx]
        pivot_speed = pivot_speed.loc[common_idx]
        if len(pivot_count) < n_steps:
            return None
        arr_count = pivot_count.iloc[:n_steps].values.astype(np.float32)
        arr_speed = pivot_speed.iloc[:n_steps].values.astype(np.float32)
        arr = np.concatenate([arr_count, arr_speed], axis=1)  # (n_steps, num_edges*2)
        return arr, os.path.basename(csv_path)
    except Exception:
        return None


def preload_csvs(csv_paths, edge_ids, n_steps, output_channels=1):
    """一次性把所有 CSV 的前 n_steps 步載入記憶體,加速訓練。"""
    cache = {}
    skipped = 0
    for p in tqdm(csv_paths, desc="Preloading CSVs"):
        res = load_csv_first_n_steps(p, edge_ids, n_steps, output_channels=output_channels)
        if res is None:
            skipped += 1
            continue
        cache[p] = res[0]
    print(f"Preloaded {len(cache)} CSVs (skipped {skipped})")
    return cache


# =================================================
# 5. PairDataset
# =================================================
class PairDataset(Dataset):
    """
    每個樣本: (input, target)
        output_channels=1:
            input  shape = (INPUT_LEN, num_edges + 3)         ← edges + sin/cos/gap
            target shape = (PRED_HORIZON, num_edges)
        output_channels=2:
            input  shape = (INPUT_LEN, num_edges*2 + 3)       ← count + speed + sin/cos/gap
            target shape = (PRED_HORIZON, num_edges*2)
    """
    def __init__(self, pair_list, edge_ids, scaler, csv_cache, output_channels=1):
        self.edge_ids = edge_ids
        self.num_edges = len(edge_ids)
        self.scaler = scaler
        self.cache = csv_cache
        self.output_channels = output_channels
        # 只留下兩端 CSV 都已成功 preload 的 pair
        self.pair_list = [
            (a, b, g) for a, b, g in pair_list if a in csv_cache and b in csv_cache
        ]
        # 額外過濾: 若 A 的 count 部分 max < log1p(5),代表幾乎無車流,跳過
        # 多 channel 時只判斷 count 部分(前 num_edges 欄),不被 speed 干擾
        threshold = np.log1p(5).astype(np.float32)
        kept = []
        for a, b, g in self.pair_list:
            a_arr = self.cache[a][:INPUT_LEN]
            a_count = a_arr[:, :self.num_edges]
            a_max = float(np.log1p(a_count).max())
            if a_max >= threshold:
                kept.append((a, b, g))
        self.pair_list = kept

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        path_a, path_b, gap = self.pair_list[idx]
        arr_a = self.cache[path_a][:INPUT_LEN]
        arr_b = self.cache[path_b][:PRED_HORIZON]
        # scaler 對全部 channel 一律 log1p (count 和 speed 同 scaler,簡單一致)
        x_traf = self.scaler.transform(arr_a)
        y_target = self.scaler.transform(arr_b)
        x_time = build_time_features(os.path.basename(path_a), INPUT_LEN, gap)
        x_comb = np.hstack([x_traf, x_time])
        return (
            torch.tensor(x_comb, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
        )


# =================================================
# 6. Model
# =================================================
class GRUSequence(nn.Module):
    """
    GRU + Attention + Decoder。

    output_channels=1 (v1):
        input  shape = (B, INPUT_LEN, num_edges + input_extra_features)
        output shape = (B, PRED_HORIZON, num_edges)
    output_channels=2 (v2):
        input  shape = (B, INPUT_LEN, num_edges*2 + input_extra_features)
        output shape = (B, PRED_HORIZON, num_edges*2)
        前 num_edges 欄是 count, 後 num_edges 欄是 speed

    input_extra_features:
        新版 (pair model): 3 (sin/cos/gap)
        舊版 (sliding):    2 (sin/cos)
    """
    def __init__(self, num_edges, hidden_dim, num_layers, horizon, dropout=0.2,
                 input_extra_features=3, output_channels=1):
        super().__init__()
        self.horizon = horizon
        self.num_edges = num_edges
        self.output_channels = output_channels
        self.gru = nn.GRU(
            input_size=num_edges * output_channels + input_extra_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.Linear(hidden_dim, 1)
        mid = hidden_dim // 2
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_edges * horizon * output_channels),
        )

    def forward(self, x):
        out, _ = self.gru(x)                            # (B, T, H)
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        context = (attn_w * out).sum(dim=1)             # (B, H)
        pred_flat = self.decoder(context)
        return pred_flat.view(-1, self.horizon, self.num_edges * self.output_channels)


def weighted_sequence_loss(pred, target, num_edges=None, output_channels=1,
                            speed_loss_weight=SPEED_LOSS_WEIGHT):
    """
    output_channels=1: 沿用 v1 行為,高流量 edge (target > log1p(10) ≈ 2.4) 權重 5×。
    output_channels=2:
        - count 部分: 沿用 v1 weighted MSE (高流量 5×)
        - speed 部分: has-vehicle masked MSE × speed_loss_weight
        - 兩個 loss 相加

    為什麼 speed 用 has-vehicle mask:
        Target speed matrix 是 (T × num_edges) dense,但 SUMO CSV 是 sparse
        (只記 count > 0 的 cell),pivot 後 fillna(0) 把「沒車 edge」補成 0。
        典型 90% cells 是這種 fillna 假 0,只有 10% 是真實 speed 量測。
        若 speed loss 平均整個 matrix,90% 假 0 會稀釋掉真 signal,model
        會 collapse 到「對所有 cells 輸出 ~5 km/h」(loss local minimum)。
        Mask 後 speed loss 只計算 targ_count > 0 的 cells (真量測點),
        強制 model 學「對有車 edge,輸出正確速度」這個真實 mapping。
    """
    if output_channels == 1:
        threshold = 2.4
        weights = 1.0 + 4.0 * (target > threshold).float()
        loss = (pred - target) ** 2
        return (loss * weights).mean()

    # v2: 拆 count vs speed
    assert num_edges is not None, "v2 loss 需傳 num_edges"
    pred_count  = pred[...,  :num_edges]
    pred_speed  = pred[..., num_edges:]
    targ_count  = target[...,  :num_edges]
    targ_speed  = target[..., num_edges:]

    threshold = 2.4
    weights = 1.0 + 4.0 * (targ_count > threshold).float()
    count_loss = ((pred_count - targ_count) ** 2 * weights).mean()

    # has-vehicle mask: speed loss 只在 target count > 0 的 cells 計算,
    # 避免 fillna(0) 的 dense matrix 假 0 稀釋真 signal。+1e-6 防 div-by-zero
    # (理論上 train batch 一定會有 has-vehicle cells,但保險)。
    speed_mask = (targ_count > 0).float()
    speed_sq_err = (pred_speed - targ_speed) ** 2 * speed_mask
    speed_loss = speed_sq_err.sum() / (speed_mask.sum() + 1e-6)

    return count_loss + speed_loss_weight * speed_loss


# =================================================
# 7. Training
# =================================================
def run_training(args):
    print(f"Device: {DEVICE}")
    print(f"Data dir: {args.dir}")

    pairs = find_csv_pairs(args.dir)
    if not pairs:
        print("No valid pairs found. Exiting.")
        return
    print(f"Found {len(pairs)} valid pairs")

    gaps = [g for _, _, g in pairs]
    print(f"  Gap minutes: min={min(gaps):.2f}, max={max(gaps):.2f}, mean={sum(gaps)/len(gaps):.2f}")

    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
        print(f"  Limited to first {len(pairs)} pairs (--max-pairs)")

    # Edge union (canonical)
    all_paths = sorted({p for pa, pb, _ in pairs for p in (pa, pb)})
    edge_ids = build_edge_union(all_paths, rebuild=args.rebuild_cache)
    num_edges = len(edge_ids)

    # 預載入所有 CSV (節省每 epoch 重複 I/O)
    n_steps = max(INPUT_LEN, PRED_HORIZON)
    csv_cache = preload_csvs(all_paths, edge_ids, n_steps, output_channels=OUTPUT_CHANNELS)

    # 切分 (時序: 最後 10% 當 val,避免時序洩漏)
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    scaler = Log1pScaler()
    train_ds = PairDataset(train_pairs, edge_ids, scaler, csv_cache,
                           output_channels=OUTPUT_CHANNELS)
    val_ds = PairDataset(val_pairs, edge_ids, scaler, csv_cache,
                         output_channels=OUTPUT_CHANNELS)
    print(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs (after filtering)")

    if len(train_ds) == 0:
        print("Train set empty after filtering. Exiting.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = GRUSequence(num_edges, HIDDEN_DIM, NUM_LAYERS, PRED_HORIZON, DROPOUT,
                        input_extra_features=3,
                        output_channels=OUTPUT_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    input_size = num_edges * OUTPUT_CHANNELS + 3
    print(f"\nStart training (pair-based, gap feature ON, output_channels={OUTPUT_CHANNELS})")
    print(f"Model: input_size={input_size}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, horizon={PRED_HORIZON}")
    print(f"Hyperparams: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE}\n")

    min_val = float("inf")
    best_state = None
    pat = 0

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        n_train = 0
        for bx, by in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = weighted_sequence_loss(pred, by,
                                          num_edges=num_edges,
                                          output_channels=OUTPUT_CHANNELS)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
            n_train += bx.size(0)
        tr_loss /= max(n_train, 1)

        model.eval()
        va_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                vp = model(vx)
                va_loss += weighted_sequence_loss(vp, vy,
                                                  num_edges=num_edges,
                                                  output_channels=OUTPUT_CHANNELS).item() * vx.size(0)
                n_val += vx.size(0)
        va_loss /= max(n_val, 1)

        print(f"Epoch {epoch + 1:03d} | Train {tr_loss:.6f} | Val {va_loss:.6f}", end="")

        if va_loss < min_val:
            min_val = va_loss
            model_type = "gru_pair_log1p_v2" if OUTPUT_CHANNELS == 2 else "gru_pair_log1p_v1"
            best_state = {
                "model_state_dict": model.state_dict(),
                "scaler_type": "log1p",
                "scaler": scaler,
                "edge_ids": edge_ids,
                "config": {
                    "input_len": INPUT_LEN,
                    "pred_horizon": PRED_HORIZON,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "model_type": model_type,
                    "gap_feature": True,
                    "input_basis": "pair",
                    "output_channels": OUTPUT_CHANNELS,
                    "channel_order": ["vehicle_count", "avg_speed_kmh"][:OUTPUT_CHANNELS],
                    "speed_loss_weight": SPEED_LOSS_WEIGHT if OUTPUT_CHANNELS == 2 else None,
                },
            }
            print(" * Best")
            pat = 0
        else:
            print(f" | Pat {pat + 1}/{PATIENCE}")
            pat += 1
            if pat >= PATIENCE:
                print("\nEarly stopping.")
                break

    if best_state:
        torch.save(best_state, MODEL_PATH)
        print(f"\nSaved best model to {MODEL_PATH}")
        print(f"  Best val loss: {min_val:.6f}")
    else:
        print("\nTraining failed (no best state saved).")


# =================================================
# 8. Evaluation
# =================================================
def _register_legacy_classes():
    """讓 torch.load 能 unpickle 自訂類別。"""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    if not hasattr(main_module, "Log1pScaler"):
        setattr(main_module, "Log1pScaler", Log1pScaler)
    if not hasattr(main_module, "GRUSequence"):
        setattr(main_module, "GRUSequence", GRUSequence)


def load_checkpoint(model_path):
    print(f"Loading model from {model_path} ...")
    _register_legacy_classes()
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})
    edge_ids = ckpt.get("edge_ids") or ckpt.get("edge_ids_list") or []
    num_edges = len(edge_ids)

    input_extra = 3 if config.get("gap_feature", False) else 2
    output_channels = config.get("output_channels", 1)  # 舊 v1 模型沒這欄,預設 1

    model = GRUSequence(
        num_edges,
        config.get("hidden_dim", HIDDEN_DIM),
        config.get("num_layers", NUM_LAYERS),
        config.get("pred_horizon", PRED_HORIZON),
        input_extra_features=input_extra,
        output_channels=output_channels,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = ckpt.get("scaler") or Log1pScaler()
    print(f"  num_edges={num_edges}, model_type={config.get('model_type', 'unknown')}, "
          f"gap_feature={config.get('gap_feature', False)}, output_channels={output_channels}")
    return model, scaler, edge_ids, config


def evaluate_pair(input_csv, target_csv, model_path=None, gap_minutes=None, bundle=None):
    """
    Returns dict:
        input_csv, target_csv, gap_minutes
        overall_mae, overall_rmse, per_step_mae      ← v1 數字 (count 部分)
        v2 額外有:
            count_mae, count_rmse, count_per_step
            speed_mae, speed_rmse, speed_per_step    ← km/h
    """
    if bundle is None:
        bundle = load_checkpoint(model_path)
    model, scaler, edge_ids, config = bundle

    input_len = config.get("input_len", INPUT_LEN)
    pred_horizon = config.get("pred_horizon", PRED_HORIZON)
    output_channels = config.get("output_channels", 1)
    num_edges = len(edge_ids)

    res_a = load_csv_first_n_steps(input_csv, edge_ids, input_len,
                                   output_channels=output_channels)
    res_b = load_csv_first_n_steps(target_csv, edge_ids, pred_horizon,
                                   output_channels=output_channels)
    if res_a is None:
        raise ValueError(f"Input CSV 不足 {input_len} 步或讀取失敗: {input_csv}")
    if res_b is None:
        raise ValueError(f"Target CSV 不足 {pred_horizon} 步或讀取失敗: {target_csv}")
    arr_a, fname_a = res_a
    arr_b, _ = res_b

    # gap_minutes 推算 (若未指定)
    if gap_minutes is None:
        dt_a = parse_timestamp(os.path.basename(input_csv))
        dt_b = parse_timestamp(os.path.basename(target_csv))
        if dt_a and dt_b:
            gap_minutes = abs((dt_b - dt_a).total_seconds()) / 60.0
        else:
            gap_minutes = 5.0

    # 前向
    x_traf = scaler.transform(arr_a)
    if config.get("gap_feature", False):
        x_time = build_time_features(fname_a, input_len, gap_minutes)
    else:
        # legacy 模型: 只用 sin/cos 兩欄
        x_time = build_time_features(fname_a, input_len, gap_minutes)[:, :2]
    x_comb = np.hstack([x_traf, x_time])
    x_tensor = torch.tensor(x_comb, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()[0]
    pred_real = np.clip(scaler.inverse_transform(pred_scaled), 0.0, None)

    abs_err = np.abs(pred_real - arr_b)
    sq_err = (pred_real - arr_b) ** 2

    result = {
        "input_csv": os.path.basename(input_csv),
        "target_csv": os.path.basename(target_csv),
        "gap_minutes": float(gap_minutes),
        "output_channels": output_channels,
    }

    if output_channels == 1:
        # v1: 直接全矩陣統計
        result.update({
            "overall_mae": float(np.mean(abs_err)),
            "overall_rmse": float(np.sqrt(np.mean(sq_err))),
            "per_step_mae": [float(np.mean(abs_err[t])) for t in range(pred_horizon)],
        })
    else:
        # v2: 拆 count vs speed,分別算
        abs_count = abs_err[:, :num_edges]
        abs_speed = abs_err[:, num_edges:]
        sq_count  = sq_err[:,  :num_edges]
        sq_speed  = sq_err[:,  num_edges:]
        result.update({
            "count_mae":      float(np.mean(abs_count)),
            "count_rmse":     float(np.sqrt(np.mean(sq_count))),
            "count_per_step": [float(np.mean(abs_count[t])) for t in range(pred_horizon)],
            "speed_mae":      float(np.mean(abs_speed)),
            "speed_rmse":     float(np.sqrt(np.mean(sq_speed))),
            "speed_per_step": [float(np.mean(abs_speed[t])) for t in range(pred_horizon)],
            # 為了和 v1 介面相容,overall_mae/rmse 仍指 count (主要評估目標)
            "overall_mae":    float(np.mean(abs_count)),
            "overall_rmse":   float(np.sqrt(np.mean(sq_count))),
            "per_step_mae":   [float(np.mean(abs_count[t])) for t in range(pred_horizon)],
        })
    return result


def print_eval_result(res):
    print(f"\n[Pair Eval] {res['input_csv']}  →  {res['target_csv']}")
    print(f"  Gap         : {res['gap_minutes']:.2f} min")
    is_v2 = res.get("output_channels", 1) == 2

    if is_v2:
        print(f"  Count MAE / RMSE : {res['count_mae']:.4f} / {res['count_rmse']:.4f} (輛)")
        print(f"  Speed MAE / RMSE : {res['speed_mae']:.4f} / {res['speed_rmse']:.4f} (km/h)")
        print(f"  Per-step MAE (count | speed, each step = 20s):")
        for i, (c, s) in enumerate(zip(res["count_per_step"], res["speed_per_step"])):
            sec = (i + 1) * 20
            bar = "█" * int(c * 20)
            print(f"    +{sec:>3d}s (step {i + 1:2d}): count={c:.4f}  speed={s:.4f}  {bar}")
    else:
        print(f"  Overall MAE : {res['overall_mae']:.4f}")
        print(f"  Overall RMSE: {res['overall_rmse']:.4f}")
        print(f"  Per-step MAE (each step = 20s):")
        for i, mae in enumerate(res["per_step_mae"]):
            sec = (i + 1) * 20
            bar = "█" * int(mae * 20)
            print(f"    +{sec:>3d}s (step {i + 1:2d}): {mae:.4f}  {bar}")


def evaluate_batch(args):
    pairs = find_csv_pairs(args.dir)
    print(f"Found {len(pairs)} pairs in {args.dir}")
    if not pairs:
        return

    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
        print(f"Limited to first {len(pairs)} pairs (--max-pairs)")

    bundle = load_checkpoint(args.model)
    _, _, _, _config = bundle
    is_v2 = _config.get("output_channels", 1) == 2

    overall_maes = []
    overall_rmses = []
    per_step_acc = np.zeros(PRED_HORIZON, dtype=np.float64)
    # v2 額外收集 speed 指標
    speed_maes = []
    speed_rmses = []
    speed_per_step_acc = np.zeros(PRED_HORIZON, dtype=np.float64)
    n_success = n_fail = 0

    for path_a, path_b, gap in tqdm(pairs, desc="Evaluating"):
        try:
            res = evaluate_pair(path_a, path_b, gap_minutes=gap, bundle=bundle)
            overall_maes.append(res["overall_mae"])
            overall_rmses.append(res["overall_rmse"])
            per_step_acc += np.array(res["per_step_mae"])
            if is_v2:
                speed_maes.append(res["speed_mae"])
                speed_rmses.append(res["speed_rmse"])
                speed_per_step_acc += np.array(res["speed_per_step"])
            n_success += 1
        except Exception:
            n_fail += 1

    if n_success == 0:
        print("All pairs failed evaluation.")
        return

    per_step_mean = per_step_acc / n_success
    print("\n========== Batch Eval Summary ==========")
    print(f"Pairs evaluated: {n_success} (skipped {n_fail})")
    print(f"Model version: {'v2 (count + speed)' if is_v2 else 'v1 (count only)'}")
    label = "Count MAE" if is_v2 else "Overall MAE"
    print(f"{label}    : mean={np.mean(overall_maes):.4f}, "
          f"median={np.median(overall_maes):.4f}, "
          f"p90={np.percentile(overall_maes, 90):.4f}  (輛)")
    print(f"Count RMSE   : mean={np.mean(overall_rmses):.4f}, "
          f"median={np.median(overall_rmses):.4f}  (輛)")
    if is_v2:
        speed_per_step_mean = speed_per_step_acc / n_success
        print(f"Speed MAE    : mean={np.mean(speed_maes):.4f}, "
              f"median={np.median(speed_maes):.4f}, "
              f"p90={np.percentile(speed_maes, 90):.4f}  (km/h)")
        print(f"Speed RMSE   : mean={np.mean(speed_rmses):.4f}, "
              f"median={np.median(speed_rmses):.4f}  (km/h)")
    print(f"Per-step Count MAE:")
    for i, mae in enumerate(per_step_mean):
        sec = (i + 1) * 20
        bar = "█" * int(mae * 20)
        print(f"  +{sec:>3d}s (step {i + 1:2d}): {mae:.4f}  {bar}")
    if is_v2:
        print(f"Per-step Speed MAE (km/h):")
        for i, mae in enumerate(speed_per_step_mean):
            sec = (i + 1) * 20
            bar = "█" * int(mae)  # km/h 量級較大,bar 縮小
            print(f"  +{sec:>3d}s (step {i + 1:2d}): {mae:.4f}  {bar}")
    monotonic = all(per_step_mean[i] <= per_step_mean[i + 1] + 1e-6
                    for i in range(len(per_step_mean) - 1))
    print(f"Monotonic per-step degradation (count): {'YES' if monotonic else 'NO'}")

    # ---------- Step 1 驗收 (對照 INTEGRATION_PAIR_MODEL.md) ----------
    mae_mean = float(np.mean(overall_maes))
    mae_median = float(np.median(overall_maes))
    rmse_mean = float(np.mean(overall_rmses))
    step1_mae = float(per_step_mean[0])
    step_last_mae = float(per_step_mean[-1])
    ratio_last_first = (step_last_mae / step1_mae) if step1_mae > 0 else float("inf")
    total_attempted = n_success + n_fail
    skip_ratio = (n_fail / total_attempted) if total_attempted > 0 else 0.0

    checks = [
        ("整體 MAE mean ≤ 3.0",
         mae_mean <= 3.0,
         f"mean={mae_mean:.4f} (threshold ≤ 3.0)"),
        ("整體 MAE median ≤ 2.5",
         mae_median <= 2.5,
         f"median={mae_median:.4f} (threshold ≤ 2.5)"),
        ("Monotonic per-step degradation",
         monotonic,
         f"{'YES' if monotonic else 'NO'}"),
        ("第 1 步 vs 最末步 MAE 比值 < 2.5",
         ratio_last_first < 2.5,
         f"step1={step1_mae:.4f}, step{len(per_step_mean)}={step_last_mae:.4f}, "
         f"ratio={ratio_last_first:.3f} (threshold < 2.5)"),
        ("整體 RMSE mean ≤ 6.0",
         rmse_mean <= 6.0,
         f"mean={rmse_mean:.4f} (threshold ≤ 6.0)"),
        ("失敗 pair 比例 < 5%",
         skip_ratio < 0.05,
         f"skipped={n_fail}/{total_attempted} = {skip_ratio*100:.2f}% (threshold < 5%)"),
    ]

    print("\n========== Step 1 驗收檢查 ==========")
    n_pass = 0
    for name, passed, detail in checks:
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} {name}")
        print(f"         → {detail}")
        if passed:
            n_pass += 1

    all_pass = n_pass == len(checks)
    print(f"\n通過: {n_pass}/{len(checks)}")
    if all_pass:
        print(">>> 全部通過,可進行 INTEGRATION_PAIR_MODEL.md Step 2 整合 <<<")
    else:
        print(">>> 未全部通過,請依 INTEGRATION_PAIR_MODEL.md 附錄 C 調整 hyperparameters <<<")


# =================================================
# 9. Dry-run (僅統計 pair,不跑模型)
# =================================================
def dry_run(args):
    pairs = find_csv_pairs(args.dir)
    print(f"Found {len(pairs)} pairs in {args.dir}")
    if not pairs:
        return
    gaps = [g for _, _, g in pairs]
    print(f"Gap minutes: min={min(gaps):.2f}, max={max(gaps):.2f}, mean={sum(gaps)/len(gaps):.2f}")
    print("Gap distribution:")
    for lo, hi in [(3, 5), (5, 7), (7, 9), (9, 11), (11, 15)]:
        n = sum(1 for g in gaps if lo <= g < hi)
        pct = 100.0 * n / len(gaps)
        print(f"  [{lo:>2d}, {hi:>2d}) min: {n:>5d} pairs ({pct:5.1f}%)")
    print("First 3 pairs:")
    for pa, pb, g in pairs[:3]:
        print(f"  {os.path.basename(pa)}  →  {os.path.basename(pb)}  ({g:.2f} min)")
    print("Last 3 pairs:")
    for pa, pb, g in pairs[-3:]:
        print(f"  {os.path.basename(pa)}  →  {os.path.basename(pb)}  ({g:.2f} min)")


# =================================================
# 10. CLI
# =================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pair-based GRU 5-minute traffic prediction "
                    "(train / eval / eval-batch).")
    parser.add_argument("--mode", choices=["train", "eval", "eval-batch"], default="train",
                        help="train (預設) | eval (單組 pair) | eval-batch (整個資料夾)")
    parser.add_argument("--input", help="[eval] 輸入 CSV 路徑")
    parser.add_argument("--target", help="[eval] 目標 CSV 路徑")
    parser.add_argument("--gap", type=float, default=None,
                        help="[eval] gap_minutes (未指定則從檔名推算)")
    parser.add_argument("--dir", default=DATA_DIR,
                        help="[train/eval-batch] CSV 資料夾路徑")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="模型 checkpoint 路徑 (預設 gru_traffic_model_pair.pth)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="強制重建 data/edge_ids_union.json")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="限制 pair 數量 (除錯用,0 = 全部)")
    parser.add_argument("--dry-run", action="store_true",
                        help="[eval-batch] 僅統計 pair 不跑模型")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "eval":
        if not args.input or not args.target:
            parser.error("--input 與 --target 為 eval 模式必填")
        res = evaluate_pair(args.input, args.target, model_path=args.model,
                            gap_minutes=args.gap)
        print_eval_result(res)
    elif args.mode == "eval-batch":
        if args.dry_run:
            dry_run(args)
        else:
            evaluate_batch(args)


if __name__ == "__main__":
    main()
