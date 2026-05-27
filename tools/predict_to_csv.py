import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from traffic_light_optimizer import process_prediction_csv


class Log1pScaler:
    def fit(self, data):
        pass

    def transform(self, data):
        return np.log1p(data)

    def inverse_transform(self, data):
        return np.expm1(data)


class GRUSequence(nn.Module):
    def __init__(self, num_edges, hidden_dim, num_layers, horizon, dropout=0.2,
                 input_extra_features=2, output_channels=1):
        """
        input_extra_features:
            新版 pair model    : 3 (sin/cos/gap)
            舊版 sliding model : 2 (sin/cos)

        output_channels:
            1 (v1)             : 只預測 vehicle_count
            2 (v2)             : 預測 vehicle_count + avg_speed_kmh
                                  前 num_edges 欄是 count, 後 num_edges 欄是 speed
        """
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
        out, _ = self.gru(x)                           # (B, T, H)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = (attn_w * out).sum(dim=1)            # (B, H)
        pred_flat = self.decoder(context)
        pred_seq = pred_flat.view(-1, self.horizon,
                                  self.num_edges * self.output_channels)
        return pred_seq


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


TRAFFIC_LIGHT_DEMO_DIR = os.path.join(ROOT_DIR, "data", "traffic_light_demo")
# 預設用 v2 pair-based 模型 (gap_feature=True, model_type=gru_pair_log1p_v2,
# output_channels=2 含速度預測)。Inference 只取 count 部分,speed 由前端另一台機器處理。
# Rollback 路徑:
#   v2 → v1 (count only): gru_traffic_model_pair.pth
#   v2 → 舊 sliding     : gru_traffic_model.pth
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "gru_traffic_model_pair_v2.pth")
PREDICTION_MIN_THRESHOLD = 0.02
PREDICTION_TOP_K_PER_STEP = 40
PREDICTION_FUSION_LAST_WINDOWS = 10

def extract_datetime(filename):
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        return match.group(1) + match.group(2)
    return None


class PredictDataShortageError(ValueError):
    """Input CSV has fewer time bins than the model's input_len.
    Caller (runtime_pipeline) should treat this as a graceful skip, not crash."""
    def __init__(self, available, required):
        self.available = available
        self.required = required
        super().__init__(
            f"資料長度不足，至少需要 {required} 筆，但目前只有 {available} 筆。"
        )


class PredictModelMismatchError(RuntimeError):
    """Model checkpoint and input CSV disagree on edge count or feature width.
    Distinct from data shortage — this means the model is fundamentally
    incompatible with the current SUMO net, not just that the sim was short."""


def register_legacy_checkpoint_classes():
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    if not hasattr(main_module, "Log1pScaler"):
        setattr(main_module, "Log1pScaler", Log1pScaler)
    if not hasattr(main_module, "GRUSequence"):
        setattr(main_module, "GRUSequence", GRUSequence)


def load_model(model_path, device):
    print(f"Loading model from {model_path}...")

    # ── Integrity checks: fail fast with actionable messages ──────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型檔案: {model_path}\n"
            f"提示：runtime_pipeline 預設用 gru_traffic_model_pair.pth；"
            f"請確認檔案存在或用 --model-path 指定其他路徑。"
        )
    size_bytes = os.path.getsize(model_path)
    if size_bytes < 1024:  # < 1 KB = almost certainly empty / write-truncated
        raise ValueError(
            f"模型檔案疑似損毀（大小僅 {size_bytes} bytes）: {model_path}\n"
            f"預期 ~5 MB。請重新下載 / re-train，或從備份還原。"
        )

    register_legacy_checkpoint_classes()
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"無法載入模型 {model_path}（torch.load 失敗）：{exc}\n"
            f"可能原因：檔案損毀、PyTorch 版本不相容、或 pickle 內含的類別在當前環境找不到。"
        ) from exc

    # Required keys — checkpoints saved by train_model.py always have these
    required_keys = ("model_state_dict",)
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(
            f"模型 checkpoint 缺少必要欄位 {missing}\n"
            f"現有 keys: {list(checkpoint.keys())}\n"
            f"這個檔案可能不是 TrafficVision 的 GRU checkpoint。"
        )

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        print(f"  ⚠ 警告: checkpoint 中 config 不是 dict ({type(config).__name__})，使用預設值")
        config = {}

    if "edge_ids" in checkpoint:
        edge_ids = checkpoint["edge_ids"]
    else:
        edge_ids = checkpoint.get("edge_ids_list", [])

    num_edges = len(edge_ids)
    if num_edges == 0:
        raise ValueError(
            "模型 checkpoint 沒有 edge_ids — 無法對齊 inference 時的 edge 順序。"
            "此 checkpoint 可能來自舊版 train_model.py，請重新訓練。"
        )

    input_len = config.get("input_len", 15)
    pred_horizon = config.get("pred_horizon", 15)
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    gap_feature = bool(config.get("gap_feature", False))
    output_channels = int(config.get("output_channels", 1))   # v2: 2, v1/legacy: 1

    # Sanity check: weight shape matches declared num_edges × output_channels + extras
    state = checkpoint["model_state_dict"]
    gru_weight = state.get("gru.weight_ih_l0")
    if gru_weight is not None:
        expected_input = num_edges * output_channels + (3 if gap_feature else 2)
        actual_input = gru_weight.shape[1]
        if actual_input != expected_input:
            raise ValueError(
                f"模型維度不一致：state_dict 的 GRU input = {actual_input}，"
                f"但 edge_ids({num_edges}) × output_channels({output_channels}) + "
                f"features({3 if gap_feature else 2}) = {expected_input}。\n"
                f"可能原因：checkpoint 是用不同的 output_channels / gap_feature 設定訓練的，"
                f"或 edge_ids 在訓練後被人手動改過。"
            )

    scaler = checkpoint.get("scaler")
    if scaler is None:
        scaler = Log1pScaler()

    model_type = config.get("model_type", "gru_sequence_log1p")
    input_extra = 3 if gap_feature else 2

    print(f"Config: input_len={input_len}, horizon={pred_horizon}, hidden={hidden_dim}")
    print(f"  model_type={model_type}, gap_feature={gap_feature}, "
          f"input_extra={input_extra}, output_channels={output_channels}")
    if output_channels == 2:
        print(f"  → v2 模型: 預測 vehicle_count + speed,inference 只取 count")
    print(f"Scaler: {type(scaler)}")

    model = GRUSequence(num_edges, hidden_dim, num_layers, pred_horizon,
                        input_extra_features=input_extra,
                        output_channels=output_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, edge_ids, input_len, pred_horizon, config


def find_demo_input_csv():
    if not os.path.isdir(TRAFFIC_LIGHT_DEMO_DIR):
        raise FileNotFoundError(f"找不到資料夾: {TRAFFIC_LIGHT_DEMO_DIR}")

    csv_files = sorted(
        os.path.join(TRAFFIC_LIGHT_DEMO_DIR, name)
        for name in os.listdir(TRAFFIC_LIGHT_DEMO_DIR)
        if name.lower().endswith(".csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"{TRAFFIC_LIGHT_DEMO_DIR} 裡找不到任何 CSV 檔案。"
        )
    if len(csv_files) != 1:
        raise ValueError(
            f"{TRAFFIC_LIGHT_DEMO_DIR} 應只保留 1 個 CSV，目前找到 {len(csv_files)} 個。"
        )

    return csv_files[0]


def load_demo_csv(file_path, edge_ids, output_channels=1):
    """
    讀 CSV → pivot → 對齊 edge_ids。

    output_channels=1 (v1):  回傳 DataFrame shape = (T, num_edges) 只含 vehicle_count
    output_channels=2 (v2):  回傳 DataFrame shape = (T, num_edges*2),
                              前 num_edges 欄是 count, 後 num_edges 欄是 speed
                              若 CSV 缺 avg_speed_kmh 欄,speed 部分填 0 並印警告
    """
    df = pd.read_csv(file_path)
    if "時間" in df.columns:
        df = df.rename(
            columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"}
        )

    pivot_count = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
    pivot_count = pivot_count.sort_index().reindex(columns=edge_ids, fill_value=0)

    if output_channels == 1:
        return pivot_count

    # v2: 也要 speed
    if "avg_speed_kmh" in df.columns:
        pivot_speed = df.pivot(index="time", columns="edge_id", values="avg_speed_kmh").fillna(0)
        pivot_speed = pivot_speed.sort_index().reindex(columns=edge_ids, fill_value=0)
        # 對齊 time index (理論上 count 和 speed 是同一 CSV 同 time bins,但保險起見)
        common_idx = pivot_count.index.intersection(pivot_speed.index)
        pivot_count = pivot_count.loc[common_idx]
        pivot_speed = pivot_speed.loc[common_idx]
        speed_values = pivot_speed.values
    else:
        print(f"  ⚠ CSV 缺 avg_speed_kmh 欄,v2 模型將以 speed=0 推論 (建議升級 runtime "
              f"或 traffic_optimizer_io 也寫 speed)")
        speed_values = np.zeros_like(pivot_count.values)

    # 包回 DataFrame: 列保持 time index, 欄 = [count edges..., speed edges (重複名稱加前綴)...]
    combined = np.concatenate([pivot_count.values, speed_values], axis=1)
    speed_col_labels = [f"__speed__{e}" for e in edge_ids]
    combined_df = pd.DataFrame(
        combined,
        index=pivot_count.index,
        columns=list(edge_ids) + speed_col_labels,
    )
    return combined_df


def build_time_features(file_name, num_steps, gap_minutes=None):
    """
    Build time features for the model input.

    Args:
        file_name: 用來解析 start time (HHMMSS)
        num_steps: input 步數
        gap_minutes: 若為 None,回傳 2 欄 (sin, cos) — 給舊 sliding model
                     若有值,回傳 3 欄 (sin, cos, gap) — 給新 pair model
    """
    time_str = extract_datetime(file_name)
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
    if gap_minutes is None:
        return np.stack([sin_t, cos_t], axis=1)
    gap_col = np.full(num_steps, float(gap_minutes), dtype=np.float32)
    return np.stack([sin_t, cos_t, gap_col], axis=1)


def _predict_sequence(model, device, input_traf, input_time, scaler, gap_feature=False):
    parts = [input_traf, input_time]
    if gap_feature:
        # Gap feature: fraction of zero-traffic edges per timestep (shape: T x 1)
        gap = (input_traf == 0).mean(axis=1, keepdims=True).astype(np.float32)
        parts.append(gap)
    input_comb = np.hstack(parts)
    input_tensor = torch.tensor(input_comb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_seq_scaled = model(input_tensor).cpu().numpy()[0]
    pred_seq_real = scaler.inverse_transform(pred_seq_scaled)
    return np.clip(pred_seq_real, 0.0, None)


def _select_edge_indices(step_values):
    selected_indices = {
        int(edge_index)
        for edge_index, value in enumerate(step_values)
        if float(value) >= PREDICTION_MIN_THRESHOLD
    }

    positive_ranked_indices = [
        int(index)
        for index in np.argsort(step_values)[::-1]
        if float(step_values[index]) > 0
    ]

    for edge_index in positive_ranked_indices:
        if len(selected_indices) >= PREDICTION_TOP_K_PER_STEP:
            break
        selected_indices.add(edge_index)

    return sorted(
        selected_indices,
        key=lambda index: float(step_values[index]),
        reverse=True,
    )


def _aggregate_predictions(
    model,
    device,
    scaler,
    scaled_traffic,
    time_data,
    time_index,
    input_len,
    pred_horizon,
    max_windows=None,
    gap_feature=False,
):
    """Sliding-window ensemble aggregator.

    Vectorized over windows: instead of N separate batch-size-1 forwards (one
    per sliding window), stack all windows into a single (B, input_len, features)
    batch and do one forward. For a ~280 window run this cuts inference time
    from ~280×forward_latency to ~1×forward_latency.

    Memory is bounded: B × input_len × (num_edges + 2 or 3) × 4 bytes. For
    typical B=300, input_len=15, num_edges=1000 that's ~17 MB — fine.
    """
    aggregated_sum = {}
    aggregated_count = {}

    window_start = input_len
    if max_windows is not None:
        window_start = max(input_len, len(time_index) - int(max_windows) + 1)

    window_ends = list(range(window_start, len(time_index) + 1))
    if not window_ends:
        return aggregated_sum, aggregated_count

    # Build batched input: (B, input_len, num_edges + extras)
    batch_inputs = []
    for window_end in window_ends:
        input_traf = scaled_traffic[window_end - input_len:window_end]
        input_time = time_data[window_end - input_len:window_end]
        parts = [input_traf, input_time]
        if gap_feature:
            gap = (input_traf == 0).mean(axis=1, keepdims=True).astype(np.float32)
            parts.append(gap)
        batch_inputs.append(np.hstack(parts))
    batch_np = np.stack(batch_inputs, axis=0)  # (B, input_len, features)
    batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_scaled = model(batch_tensor).cpu().numpy()  # (B, pred_horizon, num_edges)
    pred_real = scaler.inverse_transform(pred_scaled)
    pred_real = np.clip(pred_real, 0.0, None)

    for batch_idx, window_end in enumerate(window_ends):
        base_time = float(time_index[window_end - 1])
        for step_index in range(pred_horizon):
            current_time = float(base_time + 20 * (step_index + 1))
            step_values = pred_real[batch_idx, step_index]

            if current_time not in aggregated_sum:
                aggregated_sum[current_time] = np.zeros_like(step_values, dtype=np.float64)
                aggregated_count[current_time] = np.zeros_like(step_values, dtype=np.float64)

            aggregated_sum[current_time] += step_values
            aggregated_count[current_time] += 1.0

    return aggregated_sum, aggregated_count


def export_prediction_csv(input_csv, model_path=DEFAULT_MODEL_PATH, output_root=None):
    input_csv = os.path.abspath(input_csv)
    model_path = os.path.abspath(model_path)
    output_root = os.path.abspath(output_root or os.path.dirname(input_csv))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到輸入 CSV: {input_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, edge_ids, input_len, pred_horizon, config = load_model(model_path, device)

    num_edges = len(edge_ids)
    output_channels = int(config.get("output_channels", 1))

    pivot = load_demo_csv(input_csv, edge_ids, output_channels=output_channels)
    if len(pivot) < input_len:
        # Use typed exception so runtime_pipeline can skip Step 3 gracefully
        # without confusing it with model-loading errors or genuine code bugs.
        raise PredictDataShortageError(available=len(pivot), required=input_len)

    traffic_data = pivot.values.astype(np.float32)
    scaled_traffic = scaler.transform(traffic_data)
    time_index = pivot.index.to_numpy(dtype=np.float64)

    # ─── Step 2.3: 推算 gap_for_inference ───
    # gap=5.0 對齊 runtime_pipeline.py --interval 300 (5 分鐘排程週期)
    # 舊模型 (gap_feature=False) 用 None,build_time_features 會回 2 欄;
    # 新模型 (gap_feature=True)  用 5.0,回 3 欄。
    gap_for_inference = 5.0 if config.get("gap_feature", False) else None
    # pair model 涵蓋 v1 (gru_pair_log1p_v1) 和 v2 (gru_pair_log1p_v2)
    is_pair_model = config.get("model_type", "").startswith("gru_pair_log1p")

    source_stem = os.path.splitext(os.path.basename(input_csv))[0]
    prediction_stem = f"{source_stem}_predict"
    work_dir = os.path.join(output_root, prediction_stem)
    os.makedirs(work_dir, exist_ok=True)

    prediction_csv = os.path.join(work_dir, f"{prediction_stem}.csv")
    prediction_full_csv = os.path.join(work_dir, f"{prediction_stem}_full.csv")
    shutil.copy2(input_csv, os.path.join(work_dir, os.path.basename(input_csv)))

    rows = []
    full_rows = []

    if is_pair_model:
        # ═════════════════════════════════════════════════════════════
        # 新模型: 單次預測 (固定用 CSV 前 input_len 步,符合訓練分布)
        #
        # 為何不能滑動視窗? 新模型訓練時只見過「CSV 前 15 步」這種分布,
        # 把 CSV 中段 (如 time 3000-3340s) 當輸入會輸出垃圾 (訓練分布外)。
        #
        # v2 (output_channels=2): 同時預測 speed,inference 流程不變;
        # 下游 signal optimizer 只看 count, 但 prediction CSV 多寫一欄 avg_speed_kmh
        # 讓前端 / LLM / 其他下游可以拿到速度預測。
        # ═════════════════════════════════════════════════════════════
        input_traf = scaled_traffic[:input_len]
        input_time = build_time_features(
            os.path.basename(input_csv), input_len, gap_minutes=gap_for_inference
        )
        pred_seq_real = _predict_sequence(model, device, input_traf, input_time, scaler)
        # pred_seq_real shape:
        #   v1: (pred_horizon, num_edges)
        #   v2: (pred_horizon, num_edges*2)  前半 count, 後半 speed

        # 拆分 count vs speed
        pred_count = pred_seq_real[:, :num_edges]
        pred_speed = (pred_seq_real[:, num_edges:] if output_channels == 2 else None)

        # time 軸: 接續輸入 CSV 之後 (對齊舊模型第一個 window 的輸出)
        last_input_time = float(time_index[input_len - 1])  # 通常 ≈ 340.0
        for step_index in range(pred_horizon):
            current_time = float(last_input_time + 20.0 * (step_index + 1))
            for edge_index, edge_id in enumerate(edge_ids):
                vol = round(float(pred_count[step_index, edge_index]), 4)
                row_full = {
                    "time": current_time,
                    "edge_id": edge_id,
                    "vehicle_count": vol,
                }
                if pred_speed is not None:
                    row_full["avg_speed_kmh"] = round(float(pred_speed[step_index, edge_index]), 2)
                full_rows.append(row_full)
                if vol > PREDICTION_MIN_THRESHOLD:
                    row_ctrl = {
                        "time": current_time,
                        "edge_id": edge_id,
                        "vehicle_count": vol,
                    }
                    if pred_speed is not None:
                        row_ctrl["avg_speed_kmh"] = row_full["avg_speed_kmh"]
                    rows.append(row_ctrl)
        # 若 rows 為空 (極稀疏情況),退而求 top-K
        if not rows:
            for step_index in range(pred_horizon):
                current_time = float(last_input_time + 20.0 * (step_index + 1))
                step_values = pred_count[step_index]
                for edge_index in _select_edge_indices(step_values):
                    vol = float(step_values[int(edge_index)])
                    if vol <= 0:
                        continue
                    row = {
                        "time": current_time,
                        "edge_id": edge_ids[edge_index],
                        "vehicle_count": round(vol, 4),
                    }
                    if pred_speed is not None:
                        row["avg_speed_kmh"] = round(float(pred_speed[step_index, edge_index]), 2)
                    rows.append(row)
    else:
        # ═════════════════════════════════════════════════════════════
        # 舊模型: 滑動視窗 + 多窗融合 (保留原始邏輯)
        # ═════════════════════════════════════════════════════════════
        time_data = build_time_features(
            os.path.basename(input_csv), len(pivot), gap_minutes=gap_for_inference
        )

        control_sum, control_count = _aggregate_predictions(
            model=model,
            device=device,
            scaler=scaler,
            scaled_traffic=scaled_traffic,
            time_data=time_data,
            time_index=time_index,
            input_len=input_len,
            pred_horizon=pred_horizon,
            max_windows=PREDICTION_FUSION_LAST_WINDOWS,
        )

        full_sum, full_count = _aggregate_predictions(
            model=model,
            device=device,
            scaler=scaler,
            scaled_traffic=scaled_traffic,
            time_data=time_data,
            time_index=time_index,
            input_len=input_len,
            pred_horizon=pred_horizon,
            max_windows=None,
        )

        for current_time in sorted(full_sum.keys()):
            sum_values = full_sum[current_time]
            count_values = full_count[current_time]
            step_values = np.divide(
                sum_values,
                np.maximum(count_values, 1e-9),
                out=np.zeros_like(sum_values),
                where=count_values > 0,
            )
            for edge_index, edge_id in enumerate(edge_ids):
                full_rows.append({
                    "time": float(current_time),
                    "edge_id": edge_id,
                    "vehicle_count": round(float(step_values[edge_index]), 4),
                })

        for current_time in sorted(control_sum.keys()):
            sum_values = control_sum[current_time]
            count_values = control_count[current_time]
            step_values = np.divide(
                sum_values,
                np.maximum(count_values, 1e-9),
                out=np.zeros_like(sum_values),
                where=count_values > 0,
            )
            for edge_index in _select_edge_indices(step_values):
                vehicle_count = float(step_values[int(edge_index)])
                if vehicle_count <= 0:
                    continue
                rows.append({
                    "time": float(current_time),
                    "edge_id": edge_ids[edge_index],
                    "vehicle_count": round(vehicle_count, 4),
                })

    # v2 額外有 avg_speed_kmh 欄;v1/legacy 維持 3 欄
    csv_columns = ["time", "edge_id", "vehicle_count"]
    if output_channels == 2 and is_pair_model:
        csv_columns.append("avg_speed_kmh")

    prediction_df = pd.DataFrame(rows, columns=csv_columns)
    prediction_df.to_csv(prediction_csv, index=False)

    prediction_full_df = pd.DataFrame(full_rows, columns=csv_columns)
    prediction_full_df.to_csv(prediction_full_csv, index=False)

    print(f"預測 CSV 已輸出: {prediction_csv}")
    print(f"完整預測 CSV 已輸出: {prediction_full_csv}")
    print(f"  模式: {'pair (單次)' if is_pair_model else 'sliding (多窗)'} | "
          f"output_channels: {output_channels} | "
          f"time bins: {prediction_full_df['time'].nunique()} | "
          f"控制 rows: {len(rows)} | full rows: {len(full_rows)}")
    return {
        "prediction_csv": prediction_csv,
        "prediction_full_csv": prediction_full_csv,
        "work_dir": work_dir,
        "input_len": input_len,
        "pred_horizon": pred_horizon,
        "model_type": config.get("model_type", "unknown"),
        "is_pair_model": is_pair_model,
    }


def run_prediction_pipeline(
    input_csv=None,
    model_path=DEFAULT_MODEL_PATH,
    output_root=None,
    run_signal_control=False,
):
    input_csv = os.path.abspath(input_csv or find_demo_input_csv())

    export_result = export_prediction_csv(
        input_csv=input_csv,
        model_path=model_path,
        output_root=output_root,
    )

    if run_signal_control:
        signal_result = process_prediction_csv(
            export_result["prediction_csv"],
            work_dir=export_result["work_dir"],
            run_simulations=True,
        )
        export_result["signal_control"] = signal_result

    return export_result


if __name__ == "__main__":
    run_prediction_pipeline(
        model_path=DEFAULT_MODEL_PATH,
        run_signal_control=False,
    )