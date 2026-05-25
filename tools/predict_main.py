import os

from export_handoff import export_handoff_outputs
from predict_to_csv import export_prediction_csv
from traffic_light_optimizer import run_prediction_driven_strategy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TRAFFIC_LIGHT_DEMO_DIR = os.path.join(ROOT_DIR, "data", "traffic_light_demo")
# 預設用 pair-based 模型 (gap_feature=True, model_type=gru_pair_log1p_v1)
# Rollback: 改回 "gru_traffic_model.pth" 即可
MODEL_PATH = os.path.join(ROOT_DIR, "gru_traffic_model_pair.pth")


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


def run_full_pipeline(
    input_csv=None,
    model_path=MODEL_PATH,
    work_dir=None,
    handoff_dir=None,
    route_xml_dir=None,
):
    input_csv = os.path.abspath(input_csv or find_demo_input_csv())
    model_path = os.path.abspath(model_path or MODEL_PATH)
    output_root = os.path.dirname(input_csv)

    prediction_result = export_prediction_csv(
        input_csv=input_csv,
        model_path=model_path,
        output_root=output_root,
    )

    strategy_result = run_prediction_driven_strategy(
        prediction_csv=prediction_result["prediction_csv"],
        work_dir=work_dir or prediction_result["work_dir"],
        route_xml_dir=route_xml_dir,
    )

    handoff_result = export_handoff_outputs(
        prediction_result=prediction_result,
        strategy_result=strategy_result,
        handoff_dir=handoff_dir,
    )

    return {
        "prediction": prediction_result,
        "signal_control": strategy_result["best_result"],
        "strategy_selection": strategy_result,
        "handoff": handoff_result,
    }


if __name__ == "__main__":
    result = run_full_pipeline()
    print(f"交付檔案已輸出: {result['handoff']['handoff_dir']}")