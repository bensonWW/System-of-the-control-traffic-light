"""
eval_speed_holdout.py — 補充評估 v2 model 的 speed channel holdout MAE

test_model.py 只測 count channel (L244-254 hard-code 切 [:num_edges])。
這支 script 跑同樣 holdout pairs,但只看 speed 部分。

用途: 確認 v2 重訓後 speed channel 真的有學到,不是 mode collapse to 0。

執行: python eval_speed_holdout.py
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from train_model import (
    Log1pScaler,           # noqa: F401
    GRUSequence,           # noqa: F401
    parse_timestamp,
    build_time_features,
    find_csv_pairs,
    load_csv_first_n_steps,
    load_checkpoint,
    INPUT_LEN,
    PRED_HORIZON,
)
from test_model import (
    collect_dates,
    filter_pairs_excluding_dates,
    predict_with_model,
    predict_zero,
    predict_persist,
)

DEFAULT_TEST_DIR = os.path.join(BASE_DIR, "data", "simulation_data_check")
DEFAULT_TRAIN_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model_pair_v2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--max-pairs", type=int, default=2000,
                        help="speed eval 不需要全部 21K pairs,2000 已夠 representative")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Model:  {args.model}\n")

    model, scaler, edge_ids, config = load_checkpoint(args.model)
    num_edges = len(edge_ids)
    input_len = config.get("input_len", INPUT_LEN)
    pred_horizon = config.get("pred_horizon", PRED_HORIZON)
    output_channels = config.get("output_channels", 1)

    if output_channels != 2:
        print(f"模型不是 v2 (output_channels={output_channels}),沒 speed channel,結束。")
        return

    train_dates = collect_dates(args.train_dir)
    test_dates = collect_dates(args.dir)
    overlap = sorted(train_dates & test_dates)

    pairs = find_csv_pairs(args.dir)
    if overlap:
        pairs = filter_pairs_excluding_dates(pairs, overlap)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Eval pairs: {len(pairs)}\n")

    model_speed_errs = []   # |pred - target| (raw km/h)
    persist_speed_errs = []
    speed_targets = []      # for distribution analysis
    model_speed_preds = []  # for collapse check
    high_speed_model_errs = []  # for target > 20 km/h (busy edges)
    high_speed_persist_errs = []

    for pa, pb, gap in tqdm(pairs, desc="Speed holdout"):
        try:
            res_a = load_csv_first_n_steps(pa, edge_ids, input_len, output_channels=2)
            res_b = load_csv_first_n_steps(pb, edge_ids, pred_horizon, output_channels=2)
            if res_a is None or res_b is None:
                continue
            arr_a, fname_a = res_a
            arr_b, _ = res_b

            pred = predict_with_model(model, scaler, config, arr_a, fname_a, gap)
            # shape: (horizon, num_edges*2)  - first half count, second half speed
            pred_speed = pred[:, num_edges:]
            target_speed = arr_b[:, num_edges:]

            err = np.abs(pred_speed - target_speed)
            model_speed_errs.append(float(err.mean()))
            speed_targets.append(target_speed.flatten())
            model_speed_preds.append(pred_speed.flatten())

            # Persistence baseline for speed
            persist_pred = np.tile(arr_a[-1, num_edges:], (pred_horizon, 1))
            persist_err = np.abs(persist_pred - target_speed)
            persist_speed_errs.append(float(persist_err.mean()))

            # High-speed edges (target > 20 km/h = active roads)
            hi_mask = target_speed > 20
            if hi_mask.any():
                high_speed_model_errs.append(float(err[hi_mask].mean()))
                high_speed_persist_errs.append(float(persist_err[hi_mask].mean()))

        except Exception as e:
            continue

    if not model_speed_errs:
        print("無 valid pair。")
        return

    targets_flat = np.concatenate(speed_targets)
    preds_flat = np.concatenate(model_speed_preds)
    model_arr = np.asarray(model_speed_errs)
    persist_arr = np.asarray(persist_speed_errs)

    print("=" * 70)
    print("                Speed channel Holdout 評估")
    print("=" * 70)
    print(f"Total pairs evaluated: {len(model_speed_errs)}")
    print()

    print("─" * 70)
    print(f"{'method':<22} {'MAE mean':>10} {'MAE median':>12} {'MAE p90':>10}")
    print("─" * 70)
    print(f"{'Model (v2 speed)':<22} {model_arr.mean():>10.3f} {np.median(model_arr):>12.3f} {np.percentile(model_arr, 90):>10.3f}")
    print(f"{'Persistence baseline':<22} {persist_arr.mean():>10.3f} {np.median(persist_arr):>12.3f} {np.percentile(persist_arr, 90):>10.3f}")
    print()

    print("─" * 70)
    print("Speed 分佈 — predictions vs targets (raw km/h)")
    print("─" * 70)
    print(f"{'distribution':<22} {'mean':>10} {'median':>10} {'p90':>10} {'max':>10}")
    print(f"{'targets':<22} {targets_flat.mean():>10.2f} {np.median(targets_flat):>10.2f} {np.percentile(targets_flat, 90):>10.2f} {targets_flat.max():>10.2f}")
    print(f"{'model predictions':<22} {preds_flat.mean():>10.2f} {np.median(preds_flat):>10.2f} {np.percentile(preds_flat, 90):>10.2f} {preds_flat.max():>10.2f}")
    print()

    # Mode collapse check: if more than 80% of predictions are < 1 km/h, model is broken
    pct_near_zero = (preds_flat < 1.0).mean() * 100
    pct_target_near_zero = (targets_flat < 1.0).mean() * 100
    print(f"  Model predictions < 1 km/h: {pct_near_zero:.1f}% (target < 1 km/h: {pct_target_near_zero:.1f}%)")
    print()

    print("─" * 70)
    print("高速度 edge 專項 (target > 20 km/h = active roads)")
    print("─" * 70)
    if high_speed_model_errs:
        m_hi = np.mean(high_speed_model_errs)
        p_hi = np.mean(high_speed_persist_errs)
        print(f"  Model   MAE = {m_hi:.3f} km/h")
        print(f"  Persist MAE = {p_hi:.3f} km/h")
        if p_hi > 0:
            imp = 100 * (1 - m_hi / p_hi)
            tag = "OK 模型有貢獻" if imp > 0 else "FAIL 模型反而較差"
            print(f"  Model vs Persist: {imp:+.1f}%  {tag}")
    else:
        print("  (無高速 edge pair)")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    verdicts = []

    # 1. 不是 mode collapse
    if pct_near_zero < pct_target_near_zero + 20:
        verdicts.append((True, f"預測 < 1 km/h 比例 {pct_near_zero:.1f}% 接近 target {pct_target_near_zero:.1f}% → 沒有 mode collapse"))
    else:
        verdicts.append((False, f"預測 < 1 km/h 比例 {pct_near_zero:.1f}% 遠高於 target {pct_target_near_zero:.1f}% → 仍有 mode collapse"))

    # 2. 比 persistence 好或相當
    if model_arr.mean() < persist_arr.mean() * 0.95:
        diff = 100 * (1 - model_arr.mean() / persist_arr.mean())
        verdicts.append((True, f"Speed MAE {model_arr.mean():.3f} < Persistence {persist_arr.mean():.3f} (好 {diff:.1f}%) → 學到時間演化"))
    elif model_arr.mean() > persist_arr.mean() * 1.05:
        verdicts.append((False, f"Speed MAE {model_arr.mean():.3f} > Persistence {persist_arr.mean():.3f} → 比 baseline 還差"))
    else:
        verdicts.append((None, f"Speed MAE {model_arr.mean():.3f} ≈ Persistence {persist_arr.mean():.3f} → 持平"))

    # 3. 預測分佈合理
    if preds_flat.mean() > 5.0 and preds_flat.mean() < 40.0:
        verdicts.append((True, f"預測 speed mean {preds_flat.mean():.2f} km/h 在合理範圍"))
    else:
        verdicts.append((False, f"預測 speed mean {preds_flat.mean():.2f} km/h 不合理 (target mean {targets_flat.mean():.2f})"))

    for tag, msg in verdicts:
        sym = "OK" if tag is True else "FAIL" if tag is False else "INFO"
        print(f"  [{sym}] {msg}")
    print()

    n_pass = sum(1 for t, _ in verdicts if t is True)
    n_fail = sum(1 for t, _ in verdicts if t is False)
    if n_fail == 0:
        print(f">>> Speed channel 通過 ({n_pass} pass, 0 fail) <<<")
    else:
        print(f">>> Speed channel 未完全通過 ({n_pass} pass, {n_fail} fail) <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
