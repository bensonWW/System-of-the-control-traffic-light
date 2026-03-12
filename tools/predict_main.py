import os

from predict_to_csv import export_prediction_csv
from traffic_light_optimizer import run_prediction_driven_strategy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TRAFFIC_LIGHT_DEMO_DIR = os.path.join(ROOT_DIR, "data", "traffic_light_demo")
MODEL_PATH = os.path.join(ROOT_DIR, "gru_traffic_model.pth")


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


def run_full_pipeline(input_csv=None, model_path=MODEL_PATH, work_dir=None):
    input_csv = os.path.abspath(input_csv or find_demo_input_csv())
    output_root = os.path.dirname(input_csv)

    prediction_result = export_prediction_csv(
        input_csv=input_csv,
        model_path=model_path,
        output_root=output_root,
    )

    automation_result = run_prediction_driven_strategy(
        prediction_csv=prediction_result["prediction_csv"],
        work_dir=work_dir or prediction_result["work_dir"],
    )

    return {
        "prediction": prediction_result,
        "signal_control": automation_result["best_result"],
        "automation": automation_result,
    }


if __name__ == "__main__":
    run_full_pipeline()