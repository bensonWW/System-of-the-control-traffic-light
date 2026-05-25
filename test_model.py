"""
test_model.py — Pair-based GRU 模型真實泛化測試 (holdout)

讀取 data/simulation_data_check/ 的測試 CSV,
評估 gru_traffic_model_pair.pth 在「**未見過日期**」上的真實表現。

與 train_model.py --mode eval-batch 的關鍵差別:
  eval-batch  → 跑全部 simulation_data/,等同 in-sample 評估,數字偏樂觀
  test_model  → 只跑 holdout,自動剔除與訓練重疊的日期,反映真實泛化能力

並同時比對三種預測方式,讓你判斷模型有沒有真的學到東西:
  Model     : pair GRU 模型預測
  Persist   : 把 input 最後一步當常數複製 15 步 (naive baseline)
  Zero      : 全部預測為 0 (sanity baseline)

若 Model > Zero    → 模型沒學到「車流非零」
若 Model > Persist → 模型只是學會「下一步 ≈ 當前」,沒學到時間演化

執行:
  python test_model.py
  python test_model.py --max-pairs 500              # 快測 (約 1 分鐘)
  python test_model.py --include-overlap-dates       # 保留與訓練重疊日期
  python test_model.py --model path/to/other.pth
  python test_model.py --dir path/to/test/csvs
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Windows cp950 終端機對 Unicode 符號 (✓✗ℹ★→等) 會 raise UnicodeEncodeError,
# 強制把 stdout/stderr 切到 UTF-8 + replace 模式,避免報告中途 crash。
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 重用 train_model.py 的元件,確保推論邏輯與訓練端一致
from train_model import (
    Log1pScaler,           # noqa: F401  (unpickle 需要)
    GRUSequence,           # noqa: F401  (unpickle 需要)
    parse_timestamp,
    build_time_features,
    find_csv_pairs,
    load_csv_first_n_steps,
    load_checkpoint,
    INPUT_LEN,
    PRED_HORIZON,
)

DEFAULT_TEST_DIR = os.path.join(BASE_DIR, "data", "simulation_data_check")
DEFAULT_TRAIN_DIR = os.path.join(BASE_DIR, "data", "simulation_data")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "gru_traffic_model_pair.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================
# 1. 資料洩漏檢測
# =================================================
def collect_dates(dir_path):
    """從資料夾收集所有 CSV 的日期 (YYYYMMDD set)。"""
    if not os.path.isdir(dir_path):
        return set()
    dates = set()
    for f in os.listdir(dir_path):
        if not (f.startswith("traffic_data_") and f.endswith(".csv")):
            continue
        dt = parse_timestamp(f)
        if dt is not None:
            dates.add(dt.strftime("%Y%m%d"))
    return dates


def filter_pairs_excluding_dates(pairs, exclude_dates):
    """剔除 pair 兩端任一日期在 exclude_dates 內的。"""
    exclude_set = set(exclude_dates)
    kept = []
    for pa, pb, gap in pairs:
        dt_a = parse_timestamp(os.path.basename(pa))
        dt_b = parse_timestamp(os.path.basename(pb))
        if dt_a is None or dt_b is None:
            continue
        if (dt_a.strftime("%Y%m%d") in exclude_set or
                dt_b.strftime("%Y%m%d") in exclude_set):
            continue
        kept.append((pa, pb, gap))
    return kept


# =================================================
# 2. 三種預測方法
# =================================================
def predict_with_model(model, scaler, config, input_arr, fname, gap_minutes):
    """pair GRU 模型預測 (回傳 real value, shape=(horizon, num_edges))。"""
    input_len = config.get("input_len", INPUT_LEN)
    input_traf = scaler.transform(input_arr)
    if config.get("gap_feature", False):
        input_time = build_time_features(fname, input_len, gap_minutes)
    else:
        input_time = build_time_features(fname, input_len, gap_minutes)[:, :2]
    x_comb = np.hstack([input_traf, input_time])
    x_tensor = torch.tensor(x_comb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()[0]
    return np.clip(scaler.inverse_transform(pred_scaled), 0.0, None)


def predict_zero(horizon, num_edges):
    """全零基準。"""
    return np.zeros((horizon, num_edges), dtype=np.float32)


def predict_persist(input_arr, horizon):
    """Persistence: 把 input 最後一步當常數複製到 horizon。"""
    return np.tile(input_arr[-1], (horizon, 1))


# =================================================
# 3. 主流程
# =================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pair-based GRU 模型真實泛化測試 (holdout)"
    )
    parser.add_argument("--dir", default=DEFAULT_TEST_DIR,
                        help=f"測試資料夾 (預設 {DEFAULT_TEST_DIR})")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help=f"模型檔 (預設 {DEFAULT_MODEL_PATH})")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR,
                        help="訓練資料夾,用於偵測日期洩漏 (預設 simulation_data)")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="限制 pair 數量,0=全部 (測試用)")
    parser.add_argument("--include-overlap-dates", action="store_true",
                        help="保留與訓練集重疊的日期 (預設自動剔除)")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Test   : {args.dir}")
    print(f"Model  : {args.model}\n")

    if not os.path.exists(args.model):
        print(f"找不到模型檔: {args.model}")
        sys.exit(1)
    if not os.path.isdir(args.dir):
        print(f"找不到測試資料夾: {args.dir}")
        sys.exit(1)

    # ─── 資料洩漏檢測 ───
    print("=" * 70)
    print("資料洩漏檢測")
    print("=" * 70)
    train_dates = collect_dates(args.train_dir)
    test_dates = collect_dates(args.dir)
    overlap = sorted(train_dates & test_dates)
    pure_holdout = sorted(test_dates - train_dates)
    print(f"  訓練日期數: {len(train_dates)}")
    print(f"  測試日期數: {len(test_dates)}")
    print(f"  重疊日期數: {len(overlap)}{' — ' + ', '.join(overlap) if overlap else ''}")
    print(f"  純 holdout: {len(pure_holdout)} 天")
    if overlap and not args.include_overlap_dates:
        print(f"  → 將自動剔除重疊日期 (使用 --include-overlap-dates 可保留)")
    print()

    # ─── 載入模型 ───
    print("=" * 70)
    print("模型載入")
    print("=" * 70)
    model, scaler, edge_ids, config = load_checkpoint(args.model)
    num_edges = len(edge_ids)
    input_len = config.get("input_len", INPUT_LEN)
    pred_horizon = config.get("pred_horizon", PRED_HORIZON)
    print()

    # ─── 找 pair ───
    print("=" * 70)
    print("Pair 搜尋")
    print("=" * 70)
    pairs = find_csv_pairs(args.dir)
    print(f"  原始 pair 數: {len(pairs)}")
    if overlap and not args.include_overlap_dates:
        pairs = filter_pairs_excluding_dates(pairs, overlap)
        print(f"  剔除重疊後 : {len(pairs)}")
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
        print(f"  max-pairs  : {len(pairs)}")
    if not pairs:
        print("\n無可用 pair,結束。")
        return
    print()

    # ─── 評估迴圈 ───
    n_success = n_fail = 0
    model_maes, zero_maes, persist_maes = [], [], []
    model_rmses = []
    model_step_acc = np.zeros(pred_horizon, dtype=np.float64)
    persist_step_acc = np.zeros(pred_horizon, dtype=np.float64)
    zero_step_acc = np.zeros(pred_horizon, dtype=np.float64)
    model_hi_maes, persist_hi_maes = [], []

    # 按 gap bucket 分桶
    bucket_edges = [3, 5, 7, 9, 11, 15.001]
    gap_bucket_maes = {i: [] for i in range(len(bucket_edges) - 1)}

    def bucket_idx(g):
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] <= g < bucket_edges[i + 1]:
                return i
        return len(bucket_edges) - 2

    for pa, pb, gap in tqdm(pairs, desc="Holdout test"):
        try:
            res_a = load_csv_first_n_steps(pa, edge_ids, input_len)
            res_b = load_csv_first_n_steps(pb, edge_ids, pred_horizon)
            if res_a is None or res_b is None:
                n_fail += 1
                continue
            arr_a, fname_a = res_a
            arr_b, _ = res_b

            pred_m = predict_with_model(model, scaler, config, arr_a, fname_a, gap)
            pred_z = predict_zero(pred_horizon, num_edges)
            pred_p = predict_persist(arr_a, pred_horizon)

            err_m = np.abs(pred_m - arr_b)
            err_z = np.abs(pred_z - arr_b)
            err_p = np.abs(pred_p - arr_b)

            model_maes.append(float(err_m.mean()))
            zero_maes.append(float(err_z.mean()))
            persist_maes.append(float(err_p.mean()))
            model_rmses.append(float(np.sqrt(((pred_m - arr_b) ** 2).mean())))

            model_step_acc += err_m.mean(axis=1)
            zero_step_acc += err_z.mean(axis=1)
            persist_step_acc += err_p.mean(axis=1)

            hi_mask = arr_b > 10
            if hi_mask.any():
                model_hi_maes.append(float(err_m[hi_mask].mean()))
                persist_hi_maes.append(float(err_p[hi_mask].mean()))

            gap_bucket_maes[bucket_idx(gap)].append(float(err_m.mean()))

            n_success += 1
        except Exception:
            n_fail += 1

    if n_success == 0:
        print("\n所有 pair 都失敗,無法計算統計。")
        return

    model_step_mean = model_step_acc / n_success
    zero_step_mean = zero_step_acc / n_success
    persist_step_mean = persist_step_acc / n_success

    # =================================================
    # 4. 報告
    # =================================================
    print()
    print("=" * 70)
    print("                Holdout 真實泛化測試報告")
    print("=" * 70)
    print(f"成功 pair: {n_success} / 失敗 pair: {n_fail} "
          f"({100*n_fail/(n_success+n_fail):.2f}%)")
    print(f"模型: {os.path.basename(args.model)} | "
          f"gap_feature={config.get('gap_feature', False)} | "
          f"model_type={config.get('model_type', 'unknown')}")
    print(f"input_len={input_len}, pred_horizon={pred_horizon}, num_edges={num_edges}")
    print()

    # 4.1 三方法總覽
    print("─" * 70)
    print(f"{'方法':<22} {'MAE mean':>10} {'MAE median':>12} {'MAE p90':>10} {'MAE p99':>10}")
    print("─" * 70)
    for vals, label in [
        (model_maes, "Model (pair GRU)"),
        (persist_maes, "Persistence baseline"),
        (zero_maes, "Zero baseline"),
    ]:
        arr = np.asarray(vals)
        print(f"{label:<22} {arr.mean():>10.4f} {np.median(arr):>12.4f} "
              f"{np.percentile(arr, 90):>10.4f} {np.percentile(arr, 99):>10.4f}")
    print(f"\nModel RMSE mean: {np.mean(model_rmses):.4f}")
    print()

    # 4.2 Per-step
    print("─" * 70)
    print("Per-step MAE (每步 20 秒,Step 1 = +20s, Step 15 = +300s)")
    print("─" * 70)
    print(f"{'Step':<6}{'+sec':<8}{'Model':>10}{'Persist':>12}{'Zero':>10}{'Δ':>10}  Winner")
    for i in range(pred_horizon):
        m, p, z = model_step_mean[i], persist_step_mean[i], zero_step_mean[i]
        delta = p - m
        winner = "★ Model" if m < p else "  Persist"
        print(f"{i+1:<6}+{(i+1)*20:<7}{m:>10.4f}{p:>12.4f}{z:>10.4f}{delta:>+10.4f}  {winner}")
    print()

    # 4.3 高流量 edge
    print("─" * 70)
    print("高流量 edge 專項 (target > 10 輛/step)")
    print("─" * 70)
    if model_hi_maes:
        m_hi = np.mean(model_hi_maes)
        p_hi = np.mean(persist_hi_maes)
        print(f"Model   MAE = {m_hi:.4f}")
        print(f"Persist MAE = {p_hi:.4f}")
        if p_hi > 0:
            imp = 100 * (1 - m_hi / p_hi)
            tag = "★ 模型有貢獻" if imp > 0 else "✗ 模型反而較差"
            print(f"Model 比 Persist 改善: {imp:+.1f}%  {tag}")
    else:
        print("無高流量 pair (target 全 ≤ 10)")
    print()

    # 4.4 按 gap 分桶
    print("─" * 70)
    print("按 gap 分桶 (檢查 gap_minutes feature 是否有效)")
    print("─" * 70)
    bucket_labels = [
        "[ 3,  5) min", "[ 5,  7) min", "[ 7,  9) min",
        "[ 9, 11) min", "[11, 15] min",
    ]
    for i, label in enumerate(bucket_labels):
        vs = gap_bucket_maes[i]
        if vs:
            print(f"  {label}: n={len(vs):>5}, MAE mean={np.mean(vs):.4f}, median={np.median(vs):.4f}")
        else:
            print(f"  {label}: 無樣本")
    print()

    # =================================================
    # 5. 關鍵判斷 (Verdict)
    # =================================================
    print("=" * 70)
    print("關鍵判斷")
    print("=" * 70)
    model_mean = float(np.mean(model_maes))
    zero_mean = float(np.mean(zero_maes))
    persist_mean = float(np.mean(persist_maes))

    verdicts = []

    # 1. Model vs Zero
    if model_mean < zero_mean:
        r = zero_mean / model_mean
        verdicts.append((True,
            f"Model ({model_mean:.4f}) 比 Zero baseline ({zero_mean:.4f}) 好 {r:.1f}x → 模型確實學到「車流非零」"))
    else:
        verdicts.append((False,
            f"Model ({model_mean:.4f}) ≥ Zero baseline ({zero_mean:.4f}) → 模型沒學到任何有用訊號"))

    # 2. Model vs Persistence
    diff_pct = 100 * (model_mean / persist_mean - 1) if persist_mean > 0 else 0.0
    if model_mean < persist_mean * 0.95:
        verdicts.append((True,
            f"Model 比 Persistence ({persist_mean:.4f}) 好 {-diff_pct:.1f}% → 模型有學到「時間演化」"))
    elif model_mean > persist_mean * 1.05:
        verdicts.append((False,
            f"Model 比 Persistence 差 {diff_pct:.1f}% → 模型可能只學到平凡解"))
    else:
        verdicts.append((None,
            f"Model 與 Persistence 相當 (差 {diff_pct:+.1f}%) → 「時間演化」訊號弱,但高流量區可能仍有貢獻 (看上面)"))

    # 3. Step 1 驗收門檻
    if model_mean <= 3.0:
        verdicts.append((True,
            f"Holdout MAE mean = {model_mean:.4f} ≤ 3.0 (符合 Step 1 驗收)"))
    else:
        verdicts.append((False,
            f"Holdout MAE mean = {model_mean:.4f} > 3.0 (不符合 Step 1 驗收)"))

    # 4. 與 in-sample 對比 (若使用者跑過 eval-batch)
    # (這邊不存實際 in-sample 值,僅提示)
    verdicts.append((None,
        f"若 in-sample (eval-batch) MAE 顯著低於 holdout MAE = {model_mean:.4f},"
        "代表存在時序切分洩漏,以 holdout 數字為準"))

    for tag, msg in verdicts:
        sym = "✓" if tag is True else "✗" if tag is False else "ℹ"
        print(f"  {sym} {msg}")
    print()

    # 整體結論
    n_pass = sum(1 for t, _ in verdicts if t is True)
    n_fail = sum(1 for t, _ in verdicts if t is False)
    print("─" * 70)
    if n_fail == 0:
        print(f">>> 模型通過 holdout 測試 ({n_pass} pass, 0 fail) <<<")
        print(">>> 可以進行 INTEGRATION_PAIR_MODEL.md Step 2 整合 <<<")
    elif n_fail >= 2:
        print(f">>> 模型未通過 holdout 測試 ({n_pass} pass, {n_fail} fail) <<<")
        print(">>> 建議檢查 INTEGRATION_PAIR_MODEL.md 附錄 C 調整超參數重訓 <<<")
    else:
        print(f">>> 模型部分通過 ({n_pass} pass, {n_fail} fail),整合風險中等 <<<")
        print(">>> 建議: 直接整合並用 INTEGRATION Step 5 (SUMO 回歸測試) 做最終把關 <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
