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
    def __init__(self, num_edges, hidden_dim, num_layers, horizon, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.num_edges = num_edges
        self.gru = nn.GRU(
            input_size=num_edges + 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, num_edges * horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred_flat = self.fc(last)
        pred_seq = pred_flat.view(-1, self.horizon, self.num_edges)
        return pred_seq


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


TRAFFIC_LIGHT_DEMO_DIR = os.path.join(ROOT_DIR, "data", "traffic_light_demo")
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "gru_traffic_model.pth")
PREDICTION_MIN_THRESHOLD = 0.02
PREDICTION_TOP_K_PER_STEP = 40
PREDICTION_FUSION_LAST_WINDOWS = 10

def extract_datetime(filename):
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        return match.group(1) + match.group(2)
    return None


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
    register_legacy_checkpoint_classes()
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

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
    if scaler is None:
        scaler = Log1pScaler()

    print(f"Config: input_len={input_len}, horizon={pred_horizon}, hidden={hidden_dim}")
    print(f"Scaler: {type(scaler)}")

    model = GRUSequence(num_edges, hidden_dim, num_layers, pred_horizon).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, edge_ids, input_len, pred_horizon


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


def load_demo_csv(file_path, edge_ids):
    df = pd.read_csv(file_path)
    if "時間" in df.columns:
        df = df.rename(
            columns={"時間": "time", "路段ID": "edge_id", "車輛數": "vehicle_count"}
        )

    pivot = df.pivot(index="time", columns="edge_id", values="vehicle_count").fillna(0)
    pivot = pivot.sort_index()
    pivot = pivot.reindex(columns=edge_ids, fill_value=0)
    return pivot


def build_time_features(file_name, num_steps):
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
    return np.stack([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)


def _predict_sequence(model, device, input_traf, input_time, scaler):
    input_comb = np.hstack([input_traf, input_time])
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
):
    aggregated_sum = {}
    aggregated_count = {}

    window_start = input_len
    if max_windows is not None:
        window_start = max(input_len, len(time_index) - int(max_windows) + 1)

    for window_end in range(window_start, len(time_index) + 1):
        input_traf = scaled_traffic[window_end - input_len:window_end]
        input_time = time_data[window_end - input_len:window_end]
        pred_seq_real = _predict_sequence(model, device, input_traf, input_time, scaler)

        base_time = float(time_index[window_end - 1])
        for step_index in range(pred_horizon):
            current_time = float(base_time + 20 * (step_index + 1))
            step_values = pred_seq_real[step_index]

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
    model, scaler, edge_ids, input_len, pred_horizon = load_model(model_path, device)

    pivot = load_demo_csv(input_csv, edge_ids)
    if len(pivot) < input_len:
        raise ValueError(
            f"資料長度不足，至少需要 {input_len} 筆，但目前只有 {len(pivot)} 筆。"
        )

    traffic_data = pivot.values.astype(np.float32)
    time_data = build_time_features(os.path.basename(input_csv), len(pivot))
    scaled_traffic = scaler.transform(traffic_data)

    time_index = pivot.index.to_numpy(dtype=np.float64)

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

    source_stem = os.path.splitext(os.path.basename(input_csv))[0]
    prediction_stem = f"{source_stem}_predict"
    work_dir = os.path.join(output_root, prediction_stem)
    os.makedirs(work_dir, exist_ok=True)

    prediction_csv = os.path.join(work_dir, f"{prediction_stem}.csv")
    prediction_full_csv = os.path.join(work_dir, f"{prediction_stem}_full.csv")
    shutil.copy2(input_csv, os.path.join(work_dir, os.path.basename(input_csv)))

    rows = []
    full_rows = []
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
            full_rows.append(
                {
                    "time": float(current_time),
                    "edge_id": edge_id,
                    "vehicle_count": round(float(step_values[edge_index]), 4),
                }
            )

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

            rows.append(
                {
                    "time": float(current_time),
                    "edge_id": edge_ids[edge_index],
                    "vehicle_count": round(vehicle_count, 4),
                }
            )

    prediction_df = pd.DataFrame(rows, columns=["time", "edge_id", "vehicle_count"])
    prediction_df.to_csv(prediction_csv, index=False)

    prediction_full_df = pd.DataFrame(full_rows, columns=["time", "edge_id", "vehicle_count"])
    prediction_full_df.to_csv(prediction_full_csv, index=False)

    print(f"預測 CSV 已輸出: {prediction_csv}")
    print(f"完整預測 CSV 已輸出: {prediction_full_csv}")
    return {
        "prediction_csv": prediction_csv,
        "prediction_full_csv": prediction_full_csv,
        "work_dir": work_dir,
        "input_len": input_len,
        "pred_horizon": pred_horizon,
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