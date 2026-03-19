from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from stages.base import StageContext
from stages.experiments import _baseline_reference_params, _metrics_delta, _prepare_experiment_data
from stages.improved_experiments import (
    ImprovedExperimentSpec,
    _evaluate_improved_model,
    _improved_loss_components,
    _instantiate_improved_model,
    _is_accepted,
    _prepare_target_bundle,
    _save_improved_outputs,
)
from stages.training_improved_experiments import (
    EnhancedTrainingConfig,
    _run_device_baseline,
    _train_improved_model_with_training_improvements,
)
from utils.common import format_float, save_grouped_bar_chart, save_json, save_text_figure


def _combined_spec() -> ImprovedExperimentSpec:
    return ImprovedExperimentSpec(
        name="combined_experiment_1235",
        title="Combined Experiment: Residual Elevation + Corrected Uncertainty + Elevation SConv",
        description=(
            "Combine the accepted architectural changes from Experiments 1 and 5 inside the elevation pathway and "
            "train them with the corrected per-task objective from Experiments 2 and 3."
        ),
        rationale=(
            "Experiments 1 and 5 each improved elevation while preserving the baseline distance and azimuth inductive "
            "bias, and Experiments 2 and 3 improved task balance through better-scaled losses. This run tests whether "
            "those gains stack when they are applied together under the same long-training regime."
        ),
        implemented_steps=[
            "Step 1: keep the handcrafted distance and azimuth pathways unchanged so the strong timing and binaural cues remain intact.",
            "Step 2: add the residual learned spectral CNN from Experiment 1 inside the elevation pathway.",
            "Step 3: add the residual elevation SConv2dLSTM context branch from Experiment 5 in parallel with the spectral CNN.",
            "Step 4: fuse both residual elevation corrections back into the baseline elevation latent with small learned gains.",
            "Step 5: train with corrected per-task normalization from Experiment 2 and uncertainty weighting with warm-up from Experiment 3.",
        ],
        remaining_steps=[
            "Ablate the CNN and SConv residual gains separately after training to measure which elevation correction carries the improvement.",
            "Retry the same combined variant on MPS only after the cochlea front-end avoids unsupported torch.logspace operations.",
            "Promote the combined model to a larger confirmation run only if it clearly beats the fixed training-improved baseline.",
        ],
        variant="combined_residual_elevation",
        loss_mode="corrected_uncertainty",
        training_overrides={"learning_rate_scale": 0.85, "batch_size": 16, "uncertainty_warmup_epochs": 4},
    )


def _combined_short_spec() -> ImprovedExperimentSpec:
    base = _combined_spec()
    return ImprovedExperimentSpec(
        name="combined_experiment_1235_small_data",
        title="Combined Experiment: Small-Data Check",
        description=(
            "Run the same combined model on a reduced 1000-sample dataset with a 700 / 150 / 150 split and only 10 epochs "
            "to measure speed and accuracy trade-offs."
        ),
        rationale=(
            "This isolates how much the accepted combined architecture depends on the larger training set and longer "
            "training horizon used in the main run."
        ),
        implemented_steps=[
            "Step 1: keep the combined architecture from the accepted long-training run unchanged.",
            "Step 2: reduce the dataset to 700 train, 150 validation, and 150 test samples.",
            "Step 3: cap training at 10 epochs while keeping the same optimizer family and scheduler logic.",
            "Step 4: compare the short run directly against the saved long-training combined result rather than retraining that reference.",
        ],
        remaining_steps=[
            "Repeat the same reduced-data check with 20 epochs to separate dataset-size effects from epoch-budget effects.",
            "Test a cached-feature path if data preparation dominates the short-run wall time.",
        ],
        variant=base.variant,
        loss_mode=base.loss_mode,
        training_overrides=dict(base.training_overrides),
    )


def _load_or_build_cpu_baseline(
    config: Any,
    outputs: Any,
    training_config: EnhancedTrainingConfig,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    summary_path = outputs.root / "training_improved_experiments_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cpu_result = payload.get("cpu")
        if cpu_result and cpu_result.get("status") == "success":
            return (
                str(payload.get("baseline_label", cpu_result.get("baseline_label", "unknown baseline"))),
                cpu_result,
                payload.get("mps", {}),
                "reused existing training-improved baseline",
            )

    cpu_context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    params, baseline_label = _baseline_reference_params(cpu_context)
    cpu_result = _run_device_baseline(cpu_context, params, baseline_label, training_config, device_name="cpu")
    return baseline_label, cpu_result, {"status": "not_run"}, "reran CPU baseline because no saved training-improved baseline was available"


def _baseline_metrics(cpu_result: dict[str, Any]) -> dict[str, float]:
    test_metrics = cpu_result["test_metrics"]
    return {
        "distance_mae_m": float(test_metrics["distance_mae_m"]),
        "azimuth_mae_deg": float(test_metrics["azimuth_mae_deg"]),
        "elevation_mae_deg": float(test_metrics["elevation_mae_deg"]),
        "combined_error": float(test_metrics["combined_error"]),
        "mean_spike_rate": float(test_metrics["mean_spike_rate"]),
    }


def _load_saved_combined_result(outputs_root: Path) -> dict[str, Any]:
    result_path = outputs_root / "combined_experiment" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError("Saved long-training combined result was not found at outputs/combined_experiment/result.json.")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_combined_report(
    outputs_root: Path,
    baseline_label: str,
    baseline_source: str,
    baseline_metrics: dict[str, float],
    mps_result: dict[str, Any],
    training_config: EnhancedTrainingConfig,
    spec: ImprovedExperimentSpec,
    result: dict[str, Any],
    short_run_result: dict[str, Any] | None = None,
) -> Path:
    report_path = outputs_root / "combined_experiment_report.md"
    relative_root = Path("combined_experiment")
    experiment_dir = relative_root / spec.name
    mps_status = str(mps_result.get("status", "unknown")).upper()
    mps_note = ""
    if mps_result.get("status") == "failed":
        mps_note = f"MPS was not retried here because the previous long-training baseline already failed on an unsupported op: `{mps_result.get('error', '').strip()}`."
    elif mps_result.get("status") == "unavailable":
        mps_note = "MPS was not retried here because it was unavailable in the saved training-improved baseline run."
    elif mps_result.get("status") == "success":
        mps_note = "MPS support exists at the environment level, but this combined run stayed on CPU so it remained directly comparable with the fixed CPU baseline."
    else:
        mps_note = "MPS was not used for this combined run."

    lines = [
        "# Combined Experiment Report",
        "",
        "## Scope",
        "",
        f"- Baseline reference: `{baseline_label}`",
        f"- Baseline source: {baseline_source}",
        f"- Dataset split: `3500 / 750 / 750` synthetic scenes (`70% / 15% / 15%` of 5000 total)",
        f"- Max epochs: `{training_config.max_epochs}`",
        f"- Early stopping patience: `{training_config.early_stopping_patience}`",
        f"- Scheduler: `ReduceLROnPlateau` with patience `{training_config.scheduler_patience}` and factor `{training_config.scheduler_factor}`",
        "- Backend threads: `1`",
        "- Run device: `cpu`",
        f"- Previous MPS baseline status: `{mps_status}`",
        "",
        mps_note,
        "",
        "## Baseline Reference",
        "",
        f"- Combined error: `{baseline_metrics['combined_error']:.4f}`",
        f"- Distance MAE: `{baseline_metrics['distance_mae_m']:.4f} m`",
        f"- Azimuth MAE: `{baseline_metrics['azimuth_mae_deg']:.4f} deg`",
        f"- Elevation MAE: `{baseline_metrics['elevation_mae_deg']:.4f} deg`",
        "",
        "![Baseline loss](training_improved_experiments/cpu/baseline/loss.png)",
        "",
        "## Combined Design",
        "",
        f"- Change: {spec.description}",
        f"- Rationale: {spec.rationale}",
        "",
        "Implemented steps:",
    ]
    lines.extend([f"- {step}" for step in spec.implemented_steps])
    lines.extend(
        [
            "",
            "This combines the accepted pieces as follows:",
            "- Experiment 1 contribution: residual learned spectral CNN in the elevation branch.",
            "- Experiment 2 contribution: corrected per-task normalization in the localisation loss.",
            "- Experiment 3 contribution: uncertainty-weighted task balancing with warm-up and manual-weight initialization.",
            "- Experiment 5 contribution: residual elevation SConv2dLSTM branch for spectral-temporal context.",
            "",
            "## Result",
            "",
            f"- Decision: `{result['decision']}`",
            f"- Accepted under fixed-baseline rule: `{result['accepted']}`",
            f"- Executed epochs: `{result['training']['executed_epochs']}`",
            f"- Best epoch: `{result['training']['best_epoch']}`",
            f"- Early stopped: `{result['training']['stopped_early']}`",
            f"- Initial learning rate: `{result['training']['initial_learning_rate']:.6f}`",
            f"- Final learning rate: `{result['training']['final_learning_rate']:.6f}`",
            f"- Data preparation time: `{result['timings']['data_prep_seconds']:.2f} s`",
            f"- Training time: `{result['timings']['training_seconds']:.2f} s`",
            f"- Evaluation time: `{result['timings']['evaluation_seconds']:.2f} s`",
            f"- Total runtime: `{result['timings']['total_seconds']:.2f} s`",
            "",
            f"- Test combined error: `{result['test_metrics']['combined_error']:.4f}`",
            f"- Test distance MAE: `{result['test_metrics']['distance_mae_m']:.4f} m`",
            f"- Test azimuth MAE: `{result['test_metrics']['azimuth_mae_deg']:.4f} deg`",
            f"- Test elevation MAE: `{result['test_metrics']['elevation_mae_deg']:.4f} deg`",
            f"- Combined error delta vs baseline: `{result['comparison']['combined_error_delta']:.4f}`",
            f"- Distance delta vs baseline: `{result['comparison']['distance_mae_delta']:.4f}`",
            f"- Azimuth delta vs baseline: `{result['comparison']['azimuth_mae_delta']:.4f}`",
            f"- Elevation delta vs baseline: `{result['comparison']['elevation_mae_delta']:.4f}`",
        ]
    )

    sigma_values = result.get("learned_sigmas")
    if sigma_values is not None:
        lines.extend(
            [
                f"- Learned sigma distance: `{sigma_values['distance']:.4f}`",
                f"- Learned sigma azimuth: `{sigma_values['azimuth']:.4f}`",
                f"- Learned sigma elevation: `{sigma_values['elevation']:.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            f"![Combined metrics](combined_experiment/baseline_vs_combined.png)",
            f"![Combined loss]({experiment_dir.as_posix()}/loss.png)",
            f"![Combined summary]({experiment_dir.as_posix()}/summary.png)",
            f"![Combined distance]({experiment_dir.as_posix()}/test_distance_prediction.png)",
            f"![Combined azimuth]({experiment_dir.as_posix()}/test_azimuth_prediction.png)",
            f"![Combined elevation]({experiment_dir.as_posix()}/test_elevation_prediction.png)",
            "",
            "## Interpretation",
            "",
            "- This run tests whether the two accepted elevation-pathway changes stack while the loss correction keeps distance and angle training balanced.",
            "- Because the distance and azimuth branches stayed handcrafted, any gain here should be attributable mainly to the combined elevation augmentation and the corrected task weighting.",
            "- Acceptance still requires beating the same long-training CPU baseline on combined error and at least one individual metric.",
            "",
        ]
    )

    if short_run_result is not None:
        short_dir = relative_root / short_run_result["name"]
        speedup = float(result["timings"]["total_seconds"]) / max(float(short_run_result["timings"]["total_seconds"]), 1e-6)
        lines.extend(
            [
                "## Reduced Data Check",
                "",
                "This section reuses the saved long-training result above and compares it against a smaller run of the same combined model.",
                f"- Reduced-data split: `{short_run_result['dataset_counts']['train']} / {short_run_result['dataset_counts']['val']} / {short_run_result['dataset_counts']['test']}`",
                f"- Reduced-data max epochs: `{short_run_result['training_config']['max_epochs']}`",
                f"- Reduced-data decision: `{short_run_result['decision']}`",
                f"- Reduced-data executed epochs: `{short_run_result['training']['executed_epochs']}`",
                f"- Reduced-data best epoch: `{short_run_result['training']['best_epoch']}`",
                f"- Reduced-data early stopped: `{short_run_result['training']['stopped_early']}`",
                f"- Reduced-data total runtime: `{short_run_result['timings']['total_seconds']:.2f} s`",
                f"- Reduced-data training time: `{short_run_result['timings']['training_seconds']:.2f} s`",
                f"- Relative speedup vs long training: `{speedup:.2f}x`",
                "",
                f"- Reduced-data combined error: `{short_run_result['test_metrics']['combined_error']:.4f}`",
                f"- Reduced-data distance MAE: `{short_run_result['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Reduced-data azimuth MAE: `{short_run_result['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Reduced-data elevation MAE: `{short_run_result['test_metrics']['elevation_mae_deg']:.4f} deg`",
                f"- Combined error delta vs long training: `{short_run_result['comparison_vs_full']['combined_error_delta']:.4f}`",
                f"- Distance delta vs long training: `{short_run_result['comparison_vs_full']['distance_mae_delta']:.4f}`",
                f"- Azimuth delta vs long training: `{short_run_result['comparison_vs_full']['azimuth_mae_delta']:.4f}`",
                f"- Elevation delta vs long training: `{short_run_result['comparison_vs_full']['elevation_mae_delta']:.4f}`",
                "",
                "![Short-vs-long metrics](combined_experiment/short_data_vs_full.png)",
                f"![Short-data loss]({short_dir.as_posix()}/loss.png)",
                f"![Short-data distance]({short_dir.as_posix()}/test_distance_prediction.png)",
                f"![Short-data azimuth]({short_dir.as_posix()}/test_azimuth_prediction.png)",
                f"![Short-data elevation]({short_dir.as_posix()}/test_elevation_prediction.png)",
                "",
            ]
        )

    lines.extend(["## Remaining Follow-Up", ""])
    lines.extend([f"- {step}" for step in spec.remaining_steps])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_combined_experiment(config: Any, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig()
    baseline_label, cpu_baseline, mps_result, baseline_source = _load_or_build_cpu_baseline(config, outputs, training_config)
    if cpu_baseline.get("status") != "success":
        raise RuntimeError("CPU baseline is required before running the combined experiment.")

    context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    params, _ = _baseline_reference_params(context)
    spec = _combined_spec()
    baseline_metrics = _baseline_metrics(cpu_baseline)
    output_root = outputs.root / "combined_experiment"
    output_root.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    prep_start = time.perf_counter()
    print(f"[combined_experiment] preparing dataset on cpu with dataset_mode={training_config.dataset_mode}", flush=True)
    data = _prepare_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    data_prep_seconds = time.perf_counter() - prep_start

    print("[combined_experiment] training combined model on cpu", flush=True)
    model = _instantiate_improved_model(data, spec)
    training_start = time.perf_counter()
    train_result, uncertainty_module = _train_improved_model_with_training_improvements(
        model,
        data,
        target_bundle,
        spec,
        training_config,
    )
    training_seconds = time.perf_counter() - training_start

    model.load_state_dict(train_result.best_state)
    learned_sigmas = None
    if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
        uncertainty_module.load_state_dict(train_result.best_auxiliary_state)
        sigma = torch.exp(uncertainty_module.log_sigma.detach().clamp(-3.0, 2.0))
        learned_sigmas = {
            "distance": float(sigma[0].item()),
            "azimuth": float(sigma[1].item()),
            "elevation": float(sigma[2].item()),
        }

    evaluation_start = time.perf_counter()
    val_eval = _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
    test_eval = _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    comparison = _metrics_delta(test_eval.metrics, baseline_metrics)
    accepted = _is_accepted(test_eval.metrics, baseline_metrics)
    artifacts = _save_improved_outputs(output_root, spec, train_result, test_eval, baseline_metrics, model)

    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
        {
            "Baseline": [
                float(baseline_metrics["distance_mae_m"]),
                float(baseline_metrics["azimuth_mae_deg"]),
                float(baseline_metrics["elevation_mae_deg"]),
                float(baseline_metrics["combined_error"]),
            ],
            "Combined": [
                float(test_eval.metrics["distance_mae_m"]),
                float(test_eval.metrics["azimuth_mae_deg"]),
                float(test_eval.metrics["elevation_mae_deg"]),
                float(test_eval.metrics["combined_error"]),
            ],
        },
        output_root / "baseline_vs_combined.png",
        "Combined Experiment vs Training-Improved Baseline",
        ylabel="Error",
    )
    save_text_figure(
        [
            f"baseline_label: {baseline_label}",
            f"decision: {'ACCEPTED' if accepted else 'REJECTED'}",
            f"combined_error: {test_eval.metrics['combined_error']:.4f}",
            f"distance_mae_m: {test_eval.metrics['distance_mae_m']:.4f}",
            f"azimuth_mae_deg: {test_eval.metrics['azimuth_mae_deg']:.4f}",
            f"elevation_mae_deg: {test_eval.metrics['elevation_mae_deg']:.4f}",
            f"data_prep_seconds: {data_prep_seconds:.2f}",
            f"training_seconds: {training_seconds:.2f}",
            f"evaluation_seconds: {evaluation_seconds:.2f}",
            f"total_seconds: {total_seconds:.2f}",
            f"best_epoch: {train_result.best_epoch + 1}",
            f"executed_epochs: {train_result.executed_epochs}",
            f"stopped_early: {train_result.stopped_early}",
        ],
        output_root / "run_summary.png",
        "Combined Experiment Summary",
    )

    result = {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "rationale": spec.rationale,
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "accepted": accepted,
        "baseline_label": baseline_label,
        "baseline_source": baseline_source,
        "dataset_mode": training_config.dataset_mode,
        "dataset_counts": {"train": 3500, "val": 750, "test": 750},
        "training_config": {
            "max_epochs": training_config.max_epochs,
            "early_stopping_patience": training_config.early_stopping_patience,
            "scheduler_patience": training_config.scheduler_patience,
            "scheduler_factor": training_config.scheduler_factor,
            "learning_rate_scale": spec.training_overrides["learning_rate_scale"],
            "batch_size": spec.training_overrides["batch_size"],
            "uncertainty_warmup_epochs": spec.training_overrides["uncertainty_warmup_epochs"],
        },
        "training": {
            "executed_epochs": train_result.executed_epochs,
            "best_epoch": train_result.best_epoch + 1,
            "stopped_early": train_result.stopped_early,
            "best_val_loss": format_float(train_result.best_loss),
            "best_val_combined_error": format_float(train_result.best_combined_error),
            "initial_learning_rate": format_float(
                float(params["learning_rate"]) * float(spec.training_overrides["learning_rate_scale"]),
                digits=6,
            ),
            "final_learning_rate": format_float(train_result.lr_history[-1], digits=6),
        },
        "timings": {
            "data_prep_seconds": format_float(data_prep_seconds),
            "training_seconds": format_float(training_seconds),
            "evaluation_seconds": format_float(evaluation_seconds),
            "total_seconds": format_float(total_seconds),
        },
        "baseline_metrics": {key: format_float(value) for key, value in baseline_metrics.items()},
        "val_metrics": {key: format_float(value) for key, value in val_eval.metrics.items()},
        "test_metrics": {key: format_float(value) for key, value in test_eval.metrics.items()},
        "comparison": {key: format_float(value) for key, value in comparison.items()},
        "learned_sigmas": None if learned_sigmas is None else {key: format_float(value) for key, value in learned_sigmas.items()},
        "artifacts": {
            **artifacts,
            "baseline_vs_combined": str(output_root / "baseline_vs_combined.png"),
            "run_summary": str(output_root / "run_summary.png"),
        },
    }
    save_json(output_root / "result.json", result)
    report_path = _write_combined_report(
        outputs.root,
        baseline_label,
        baseline_source,
        baseline_metrics,
        mps_result,
        training_config,
        spec,
        result,
    )

    summary = {
        "baseline_label": baseline_label,
        "baseline_source": baseline_source,
        "dataset_mode": training_config.dataset_mode,
        "mps_reference_status": mps_result.get("status", "unknown"),
        "result": result,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "combined_experiment_summary.json", summary)
    return summary


def run_combined_small_data_test(config: Any, outputs: Any) -> dict[str, Any]:
    long_result = _load_saved_combined_result(outputs.root)
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=10,
        scheduler_patience=3,
    )
    baseline_label, cpu_baseline, mps_result, baseline_source = _load_or_build_cpu_baseline(
        config,
        outputs,
        EnhancedTrainingConfig(),
    )
    if cpu_baseline.get("status") != "success":
        raise RuntimeError("CPU baseline is required before running the reduced-data combined test.")

    context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    params, _ = _baseline_reference_params(context)
    spec = _combined_short_spec()
    output_root = outputs.root / "combined_experiment"
    output_root.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    prep_start = time.perf_counter()
    print(f"[combined_experiment] preparing reduced dataset on cpu with dataset_mode={training_config.dataset_mode}", flush=True)
    data = _prepare_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    data_prep_seconds = time.perf_counter() - prep_start

    print("[combined_experiment] training reduced-data combined model on cpu", flush=True)
    model = _instantiate_improved_model(data, spec)
    training_start = time.perf_counter()
    train_result, uncertainty_module = _train_improved_model_with_training_improvements(
        model,
        data,
        target_bundle,
        spec,
        training_config,
    )
    training_seconds = time.perf_counter() - training_start

    model.load_state_dict(train_result.best_state)
    learned_sigmas = None
    if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
        uncertainty_module.load_state_dict(train_result.best_auxiliary_state)
        sigma = torch.exp(uncertainty_module.log_sigma.detach().clamp(-3.0, 2.0))
        learned_sigmas = {
            "distance": float(sigma[0].item()),
            "azimuth": float(sigma[1].item()),
            "elevation": float(sigma[2].item()),
        }

    evaluation_start = time.perf_counter()
    val_eval = _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
    test_eval = _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    full_metrics = {
        "distance_mae_m": float(long_result["test_metrics"]["distance_mae_m"]),
        "azimuth_mae_deg": float(long_result["test_metrics"]["azimuth_mae_deg"]),
        "elevation_mae_deg": float(long_result["test_metrics"]["elevation_mae_deg"]),
        "combined_error": float(long_result["test_metrics"]["combined_error"]),
        "mean_spike_rate": float(long_result["test_metrics"]["mean_spike_rate"]),
    }
    comparison_vs_full = _metrics_delta(test_eval.metrics, full_metrics)
    accepted_vs_full = _is_accepted(test_eval.metrics, full_metrics)
    artifacts = _save_improved_outputs(output_root, spec, train_result, test_eval, full_metrics, model)

    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
        {
            "Full Combined": [
                float(full_metrics["distance_mae_m"]),
                float(full_metrics["azimuth_mae_deg"]),
                float(full_metrics["elevation_mae_deg"]),
                float(full_metrics["combined_error"]),
            ],
            "Small Data": [
                float(test_eval.metrics["distance_mae_m"]),
                float(test_eval.metrics["azimuth_mae_deg"]),
                float(test_eval.metrics["elevation_mae_deg"]),
                float(test_eval.metrics["combined_error"]),
            ],
        },
        output_root / "short_data_vs_full.png",
        "Reduced-Data Combined Model vs Full Combined Model",
        ylabel="Error",
    )

    result = {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "rationale": spec.rationale,
        "decision": "ACCEPTED" if accepted_vs_full else "REJECTED",
        "accepted": accepted_vs_full,
        "comparison_reference": "saved long-training combined result",
        "baseline_label": baseline_label,
        "baseline_source": baseline_source,
        "dataset_mode": training_config.dataset_mode,
        "dataset_counts": {"train": 700, "val": 150, "test": 150},
        "training_config": {
            "max_epochs": training_config.max_epochs,
            "early_stopping_patience": training_config.early_stopping_patience,
            "scheduler_patience": training_config.scheduler_patience,
            "scheduler_factor": training_config.scheduler_factor,
            "learning_rate_scale": spec.training_overrides["learning_rate_scale"],
            "batch_size": spec.training_overrides["batch_size"],
            "uncertainty_warmup_epochs": spec.training_overrides["uncertainty_warmup_epochs"],
        },
        "training": {
            "executed_epochs": train_result.executed_epochs,
            "best_epoch": train_result.best_epoch + 1,
            "stopped_early": train_result.stopped_early,
            "best_val_loss": format_float(train_result.best_loss),
            "best_val_combined_error": format_float(train_result.best_combined_error),
            "initial_learning_rate": format_float(
                float(params["learning_rate"]) * float(spec.training_overrides["learning_rate_scale"]),
                digits=6,
            ),
            "final_learning_rate": format_float(train_result.lr_history[-1], digits=6),
        },
        "timings": {
            "data_prep_seconds": format_float(data_prep_seconds),
            "training_seconds": format_float(training_seconds),
            "evaluation_seconds": format_float(evaluation_seconds),
            "total_seconds": format_float(total_seconds),
        },
        "reference_full_metrics": {key: format_float(value) for key, value in full_metrics.items()},
        "val_metrics": {key: format_float(value) for key, value in val_eval.metrics.items()},
        "test_metrics": {key: format_float(value) for key, value in test_eval.metrics.items()},
        "comparison_vs_full": {key: format_float(value) for key, value in comparison_vs_full.items()},
        "learned_sigmas": None if learned_sigmas is None else {key: format_float(value) for key, value in learned_sigmas.items()},
        "artifacts": {
            **artifacts,
            "short_data_vs_full": str(output_root / "short_data_vs_full.png"),
        },
    }
    save_json(output_root / "short_data_1000_result.json", result)

    baseline_metrics = _baseline_metrics(cpu_baseline)
    report_path = _write_combined_report(
        outputs.root,
        baseline_label,
        baseline_source,
        baseline_metrics,
        mps_result,
        EnhancedTrainingConfig(),
        _combined_spec(),
        long_result,
        short_run_result=result,
    )

    summary_path = outputs.root / "combined_experiment_summary.json"
    existing_summary = {}
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = {
        **existing_summary,
        "baseline_label": baseline_label,
        "baseline_source": baseline_source,
        "dataset_mode": EnhancedTrainingConfig().dataset_mode,
        "mps_reference_status": mps_result.get("status", "unknown"),
        "result": long_result,
        "short_data_test": result,
        "report_path": str(report_path),
    }
    save_json(summary_path, summary)
    return summary
