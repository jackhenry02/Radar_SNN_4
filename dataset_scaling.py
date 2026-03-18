from __future__ import annotations

import gc
import os
from pathlib import Path

from stages.base import StageContext
from stages.improvement import DATASET_MODE_SPECS, Model6PathwaySplit, _evaluate_dataset_bundle, _load_json, _prepare_dataset_bundle
from utils.common import (
    GlobalConfig,
    OutputPaths,
    get_device,
    limit_backend_resources,
    save_error_histogram,
    save_grouped_bar_chart,
    save_json,
    save_loss_curve,
    save_prediction_scatter,
    save_text_figure,
    seed_everything,
)


def _reference_params(outputs: OutputPaths) -> tuple[dict[str, float | int | str], str]:
    model7_path = outputs.logs / "model7_enhanced_optuna.json"
    if model7_path.exists():
        model7_payload = _load_json(model7_path)
        best_params = model7_payload.get("best_metrics", {}).get("best_params")
        if best_params:
            selected_trial = model7_payload["best_metrics"].get("selected_trial", "unknown")
            study_name = model7_payload["best_metrics"].get("study_name", "unknown")
            return best_params, f"{study_name} trial {selected_trial}"
    return Model6PathwaySplit().attempt_settings()[1], "model6_pathway_split attempt 2"


def _mode_sequence() -> list[str]:
    modes = ["dev", "stable", "final"]
    explicit_modes = os.environ.get("RADAR_SNN_DATASET_MODES", "").strip()
    if explicit_modes:
        return [mode for mode in (item.strip().lower() for item in explicit_modes.split(",")) if mode in modes]
    start_mode = os.environ.get("RADAR_SNN_DATASET_START_MODE", "").strip()
    if not start_mode:
        return modes
    if start_mode not in modes:
        return modes
    return modes[modes.index(start_mode) :]


def _mode_metrics_payload(mode: str, evaluation: dict[str, object]) -> dict[str, object]:
    return {
        "mode": mode,
        "counts": DATASET_MODE_SPECS[mode],
        "val_distance_mae_m": evaluation["distance_mae_m"],
        "val_azimuth_mae_deg": evaluation["azimuth_mae_deg"],
        "val_elevation_mae_deg": evaluation["elevation_mae_deg"],
        "val_combined_error": evaluation["combined_error"],
        "val_spike_rate": evaluation["mean_spike_rate"],
        "test_distance_mae_m": evaluation["test_distance_mae_m"],
        "test_azimuth_mae_deg": evaluation["test_azimuth_mae_deg"],
        "test_elevation_mae_deg": evaluation["test_elevation_mae_deg"],
        "test_combined_error": evaluation["test_combined_error"],
        "test_spike_rate": evaluation["test_mean_spike_rate"],
        "best_epoch": evaluation["training"].best_epoch,
        "best_val_loss": evaluation["training"].best_loss,
    }


def _save_mode_outputs(stage_dir: Path, mode: str, evaluation: dict[str, object]) -> None:
    mode_dir = stage_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    save_loss_curve(
        evaluation["training"].train_loss,
        evaluation["training"].val_loss,
        mode_dir / "loss.png",
        f"Dataset Scaling {mode.upper()} Loss Curve",
    )
    save_prediction_scatter(
        evaluation["test_target_distance"],
        evaluation["test_predicted_distance"],
        mode_dir / "test_distance_prediction.png",
        f"{mode.upper()} Test Distance Prediction",
        xlabel="True Distance (m)",
        ylabel="Predicted Distance (m)",
    )
    save_prediction_scatter(
        evaluation["test_target_azimuth"],
        evaluation["test_predicted_azimuth"],
        mode_dir / "test_azimuth_prediction.png",
        f"{mode.upper()} Test Azimuth Prediction",
        xlabel="True Azimuth (deg)",
        ylabel="Predicted Azimuth (deg)",
    )
    save_prediction_scatter(
        evaluation["test_target_elevation"],
        evaluation["test_predicted_elevation"],
        mode_dir / "test_elevation_prediction.png",
        f"{mode.upper()} Test Elevation Prediction",
        xlabel="True Elevation (deg)",
        ylabel="Predicted Elevation (deg)",
    )
    save_error_histogram(
        evaluation["test_predicted_elevation"] - evaluation["test_target_elevation"],
        mode_dir / "test_elevation_error.png",
        f"{mode.upper()} Test Elevation Error",
        xlabel="Elevation Error (deg)",
    )
    save_text_figure(
        [
            f"mode: {mode}",
            f"train: {DATASET_MODE_SPECS[mode]['train']}",
            f"val: {DATASET_MODE_SPECS[mode]['val']}",
            f"test: {DATASET_MODE_SPECS[mode]['test']}",
            f"val_combined_error: {evaluation['combined_error']:.4f}",
            f"test_combined_error: {evaluation['test_combined_error']:.4f}",
            f"test_distance_mae_m: {evaluation['test_distance_mae_m']:.4f}",
            f"test_azimuth_mae_deg: {evaluation['test_azimuth_mae_deg']:.4f}",
            f"test_elevation_mae_deg: {evaluation['test_elevation_mae_deg']:.4f}",
            f"best_epoch: {evaluation['training'].best_epoch}",
            f"best_val_loss: {evaluation['training'].best_loss:.4f}",
        ],
        mode_dir / "summary.png",
        f"{mode.upper()} Dataset Summary",
    )


def _save_overall_outputs(outputs: OutputPaths, results: list[dict[str, object]], reference_label: str) -> Path:
    stage_dir = outputs.stage_dir("dataset_scaling")
    mode_labels = [str(item["mode"]).upper() for item in results]

    save_grouped_bar_chart(
        mode_labels,
        {
            "Validation": [float(item["val_combined_error"]) for item in results],
            "Test": [float(item["test_combined_error"]) for item in results],
        },
        stage_dir / "combined_error_comparison.png",
        "Dataset Scaling Combined Error",
        ylabel="Combined Error",
    )
    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
        {
            str(item["mode"]).upper(): [
                float(item["test_distance_mae_m"]),
                float(item["test_azimuth_mae_deg"]),
                float(item["test_elevation_mae_deg"]),
                float(item["test_combined_error"]),
            ]
            for item in results
        },
        stage_dir / "test_metric_comparison.png",
        "Dataset Scaling Test Metrics",
        ylabel="Error",
    )

    baseline = float(results[0]["test_combined_error"])
    lines = [
        "# Dataset Scaling Study",
        "",
        f"Reference configuration: `{reference_label}`",
        "",
        "| Mode | Train | Val | Test | Val Combined | Test Combined | Test Distance MAE | Test Azimuth MAE | Test Elevation MAE | Improvement vs Dev |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        improvement = (baseline - float(item["test_combined_error"])) / max(baseline, 1e-6)
        lines.append(
            f"| {str(item['mode']).upper()} | {DATASET_MODE_SPECS[str(item['mode'])]['train']} | "
            f"{DATASET_MODE_SPECS[str(item['mode'])]['val']} | {DATASET_MODE_SPECS[str(item['mode'])]['test']} | "
            f"{float(item['val_combined_error']):.4f} | {float(item['test_combined_error']):.4f} | "
            f"{float(item['test_distance_mae_m']):.4f} | {float(item['test_azimuth_mae_deg']):.4f} | "
            f"{float(item['test_elevation_mae_deg']):.4f} | {improvement:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Combined error comparison](figures/dataset_scaling/combined_error_comparison.png)",
            "![Test metric comparison](figures/dataset_scaling/test_metric_comparison.png)",
        ]
    )
    report_path = outputs.root / "dataset_scaling.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    if "RADAR_SNN_DEVICE" not in os.environ:
        os.environ["RADAR_SNN_DEVICE"] = "cpu"
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    context = StageContext(config=config, device=get_device(), outputs=outputs)
    params, reference_label = _reference_params(outputs)

    stage_dir = outputs.stage_dir("dataset_scaling")
    log_path = outputs.logs / "dataset_scaling.json"
    existing_results: dict[str, dict[str, object]] = {}
    if log_path.exists():
        previous_payload = _load_json(log_path)
        for item in previous_payload.get("results", []):
            existing_results[str(item["mode"])] = item

    for index, mode in enumerate(_mode_sequence()):
        print(f"Running dataset scaling mode={mode} counts={DATASET_MODE_SPECS[mode]} device={context.device}", flush=True)
        bundle = _prepare_dataset_bundle(context, mode)
        evaluation = _evaluate_dataset_bundle(
            context,
            params,
            bundle,
            include_artifacts=False,
            seed=config.seed + 20_000 + index,
        )
        _save_mode_outputs(stage_dir, mode, evaluation)
        existing_results[mode] = _mode_metrics_payload(mode, evaluation)
        ordered_results = [existing_results[item] for item in ("dev", "stable", "final") if item in existing_results]
        save_json(
            log_path,
            {
                "max_threads": max_threads,
                "device": str(context.device),
                "reference_label": reference_label,
                "reference_params": params,
                "results": ordered_results,
            },
        )
        context.shared.pop(f"dataset_bundle::{mode}", None)
        gc.collect()

    results = [existing_results[item] for item in ("dev", "stable", "final") if item in existing_results]
    report_path = _save_overall_outputs(outputs, results, reference_label)
    save_json(
        outputs.root / "dataset_scaling_summary.json",
        {
            "max_threads": max_threads,
            "device": str(context.device),
            "reference_label": reference_label,
            "reference_params": params,
            "results": results,
            "report_path": str(report_path),
        },
    )
    print(f"Dataset scaling complete with max_threads={max_threads}, device={context.device}.", flush=True)


if __name__ == "__main__":
    main()
