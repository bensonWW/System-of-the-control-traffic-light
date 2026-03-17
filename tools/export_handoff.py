import os
import shutil

import pandas as pd


def export_handoff_outputs(prediction_result, strategy_result, handoff_dir=None):
    best_result = strategy_result["best_result"]
    best_strategy = strategy_result.get("best_strategy", {})
    base_work_dir = strategy_result["base_work_dir"]

    handoff_dir = os.path.abspath(handoff_dir or os.path.join(base_work_dir, "handoff"))
    if os.path.isdir(handoff_dir):
        for name in os.listdir(handoff_dir):
            path = os.path.join(handoff_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    else:
        os.makedirs(handoff_dir, exist_ok=True)

    source_files = {
        "prediction_csv": prediction_result.get("prediction_csv"),
        "prediction_full_csv": prediction_result.get("prediction_full_csv"),
        "best_strategy_csv": strategy_result.get("strategy_summary_csv"),
        "comparison_summary_csv": best_result.get("comparison_summary_csv"),
        "signal_plan_summary_csv": best_result.get("signal_plan_summary_csv"),
        "signal_change_detail_csv": best_result.get("signal_change_detail_csv"),
        "override_xml": best_result.get("override_xml"),
    }

    moved_files = {}
    for key, src_path in source_files.items():
        if not src_path or not os.path.exists(src_path):
            continue
        src_abs = os.path.abspath(src_path)
        dst_path = os.path.join(handoff_dir, os.path.basename(src_abs))

        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except OSError:
                pass

        if os.path.abspath(src_abs) != os.path.abspath(dst_path):
            shutil.move(src_abs, dst_path)

        moved_files[key] = os.path.abspath(dst_path)

    best_summary_csv = os.path.join(handoff_dir, "best_result_summary.csv")
    pd.DataFrame([best_strategy]).to_csv(best_summary_csv, index=False, encoding="utf-8-sig")
    moved_files["best_result_summary_csv"] = best_summary_csv

    manifest_csv = os.path.join(handoff_dir, "handoff_manifest.csv")
    pd.DataFrame(
        [{"name": key, "path": path} for key, path in sorted(moved_files.items())]
    ).to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    best_work_dir = best_result.get("work_dir")
    if best_work_dir:
        best_work_dir = os.path.abspath(best_work_dir)
        if os.path.isdir(best_work_dir) and os.path.abspath(best_work_dir) != os.path.abspath(handoff_dir):
            shutil.rmtree(best_work_dir, ignore_errors=True)

    return {
        "handoff_dir": handoff_dir,
        "manifest_csv": manifest_csv,
        "files": moved_files,
    }
