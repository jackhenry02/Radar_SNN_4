from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from models.round2_variants import AllRound2CombinedModel, AllRound2Encoder
from stages.base import StageContext
from stages.combined_experiment import _save_coordinate_error_profiles, _save_prediction_cache
from stages.experiments import _baseline_reference_params, _metrics_delta, _prepare_experiment_data
from stages.improved_experiments import _evaluate_improved_model, _prepare_target_bundle, _save_improved_outputs
from stages.improvement import _distance_candidates, _itd_candidates
from stages.round_2_experiments import (
    Round2ExperimentSpec,
    _augment_with_cartesian_metrics,
    _load_short_combined_baseline,
    _save_cartesian_outputs,
    _save_variant_artifacts,
    _train_round2_model,
)
from stages.training_improved_experiments import EnhancedTrainingConfig
from utils.common import format_float, save_grouped_bar_chart, save_json


def _combined_all_spec() -> Round2ExperimentSpec:
    return Round2ExperimentSpec(
        name="round_2_combined_all",
        title="Round 2 Combined-All Model",
        description=(
            "Combine the adaptive fixed-cue tuning, resonant branches, pre-pathway LIF residual, post-pathway LIF "
            "residual, and the mixed Cartesian-plus-polar loss into a single short-data experiment."
        ),
        rationale=(
            "The short-data round showed that each individual idea helped against the saved combined-small baseline. "
            "This run checks whether those gains stack constructively when all architectural additions are active together."
        ),
        implemented_steps=[
            "Start from the accepted combined model used in the short-data control.",
            "Add constrained adaptive delay, ITD, and spectral offsets/gains as in Experiment 1.",
            "Use one shared corollary-discharge resonance bank and route it both into pathway residuals and the final fusion stage.",
            "Add a pre-pathway LIF residual branch that rebuilds pathway features from processed spikes.",
            "Add post-pathway branch-specific LIF residual blocks before final fusion.",
            "Train the combined architecture with the mixed Cartesian-plus-polar loss from Experiment 5B.",
        ],
        analysis_focus=[
            "Whether the all-in model beats the fixed short-data baseline.",
            "Whether it also beats the best individual round-2 variant, or whether the features interfere.",
            "Which metrics improve and which regress when all additions are active together.",
        ],
        variant="combined_all",
        loss_mode="mixed_cartesian",
        training_overrides={"learning_rate_scale": 0.85, "batch_size": 12, "cartesian_mix_weight": 0.5},
    )


def _load_best_round2_result(outputs_root: Path) -> dict[str, Any]:
    results_path = outputs_root / "round_2_experiments" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError("Round 2 experiment results were not found.")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in round 2 results.")
    best = min(experiments, key=lambda item: float(item["test_metrics"]["combined_error"]))
    return best


def _instantiate_combined_all_model(data: Any, spec: Round2ExperimentSpec) -> torch.nn.Module:
    params = data.params
    branch_hidden_dim = int(params["branch_hidden_dim"])
    hidden_dim = int(params["hidden_dim"])
    num_steps = int(params["num_steps"])
    beta = float(params["membrane_beta"])
    threshold = float(params["fusion_threshold"])
    reset_mechanism = str(params["reset_mechanism"])
    num_frequency_channels = int(params["num_frequency_channels"])
    num_delay_lines = int(params["num_delay_lines"])

    distance_candidates = _distance_candidates(data.local_config, data.train_targets_raw.device, num_delay_lines)
    itd_candidates = _itd_candidates(data.local_config, data.train_targets_raw.device, num_delay_lines)
    encoder = AllRound2Encoder(
        distance_dim=data.train_batch.pathway.distance.shape[-1],
        azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
        elevation_dim=data.train_batch.pathway.elevation.shape[-1],
        branch_hidden_dim=branch_hidden_dim,
        num_frequency_channels=num_frequency_channels,
        num_delay_lines=num_delay_lines,
        distance_candidates=distance_candidates,
        itd_candidates=itd_candidates,
        beta=beta,
        threshold=threshold,
        num_steps=num_steps,
    )
    return AllRound2CombinedModel(
        encoder=encoder,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        reset_mechanism=reset_mechanism,
    ).to(data.train_targets_raw.device)


def _write_report(
    outputs_root: Path,
    baseline: dict[str, Any],
    best_round2: dict[str, Any],
    result: dict[str, Any],
    training_config: EnhancedTrainingConfig,
) -> Path:
    lines = [
        "# Round 2 Combined-All Experiment",
        "",
        "## Overview",
        "",
        "This run combines all of the short-data architectural additions into one model and uses the mixed Cartesian-plus-polar loss. It is intentionally a stress test: if performance improves, the features are stacking constructively; if it degrades, the gains from the individual experiments are not additive.",
        "",
        "## Fixed Protocol",
        "",
        f"- Dataset mode: `{training_config.dataset_mode}`",
        "- Split: `700 train / 150 validation / 150 test`",
        f"- Max epochs: `{training_config.max_epochs}`",
        f"- Scheduler: `ReduceLROnPlateau` with patience `{training_config.scheduler_patience}` and factor `{training_config.scheduler_factor}`",
        "- Device: `cpu`",
        "- Thread cap: `1`",
        "",
        "## Reference Models",
        "",
        f"- Fixed short-data combined baseline: combined `{baseline['metrics']['combined_error']:.4f}`, Euclidean `{baseline['metrics']['euclidean_error_m']:.4f} m`",
        f"- Best individual round-2 model: `{best_round2['title']}` with combined `{float(best_round2['test_metrics']['combined_error']):.4f}` and Euclidean `{float(best_round2['test_metrics']['euclidean_error_m']):.4f} m`",
        "",
        "## Combined-All Design",
        "",
        "Architectural additions active together:",
        "- Adaptive cue tuning from Experiment 1",
        "- Shared corollary-discharge resonance routed both per-pathway and at fusion from Experiments 2A and 2B",
        "- Pre-pathway LIF residual from Experiment 3",
        "- Post-pathway LIF residual from Experiment 4",
        "- Mixed Cartesian-plus-polar loss from Experiment 5B",
        "",
        "## Result",
        "",
        f"- Decision vs fixed short-data baseline: `{'ACCEPTED' if result['accepted_vs_baseline'] else 'REJECTED'}`",
        f"- Better than best individual round-2 model: `{'YES' if result['beats_best_round2'] else 'NO'}`",
        "",
        "Polar metrics:",
        f"- Combined error: `{result['test_metrics']['combined_error']:.4f}`",
        f"- Distance MAE: `{result['test_metrics']['distance_mae_m']:.4f} m`",
        f"- Azimuth MAE: `{result['test_metrics']['azimuth_mae_deg']:.4f} deg`",
        f"- Elevation MAE: `{result['test_metrics']['elevation_mae_deg']:.4f} deg`",
        "",
        "Cartesian metrics:",
        f"- Euclidean error: `{result['test_metrics']['euclidean_error_m']:.4f} m`",
        f"- X / Y / Z MAE: `{result['test_metrics']['x_mae_m']:.4f}`, `{result['test_metrics']['y_mae_m']:.4f}`, `{result['test_metrics']['z_mae_m']:.4f} m`",
        "",
        "Delta vs fixed short-data baseline:",
        f"- Combined error delta: `{result['comparison_vs_baseline']['combined_error_delta']:.4f}`",
        f"- Distance MAE delta: `{result['comparison_vs_baseline']['distance_mae_delta']:.4f}`",
        f"- Azimuth MAE delta: `{result['comparison_vs_baseline']['azimuth_mae_delta']:.4f}`",
        f"- Elevation MAE delta: `{result['comparison_vs_baseline']['elevation_mae_delta']:.4f}`",
        f"- Euclidean error delta: `{result['cartesian_delta_vs_baseline']['euclidean_error_delta']:.4f} m`",
        "",
        "Delta vs best individual round-2 model:",
        f"- Combined error delta: `{result['comparison_vs_best_round2']['combined_error_delta']:.4f}`",
        f"- Distance MAE delta: `{result['comparison_vs_best_round2']['distance_mae_delta']:.4f}`",
        f"- Azimuth MAE delta: `{result['comparison_vs_best_round2']['azimuth_mae_delta']:.4f}`",
        f"- Elevation MAE delta: `{result['comparison_vs_best_round2']['elevation_mae_delta']:.4f}`",
        f"- Euclidean error delta: `{result['cartesian_delta_vs_best_round2']['euclidean_error_delta']:.4f} m`",
        "",
        "Timing:",
        f"- Data prep: `{result['timings']['data_prep_seconds']:.2f} s`",
        f"- Training: `{result['timings']['training_seconds']:.2f} s`",
        f"- Evaluation: `{result['timings']['evaluation_seconds']:.2f} s`",
        f"- Total: `{result['timings']['total_seconds']:.2f} s`",
        "",
        "## Plots",
        "",
        "![Combined-all distance](round_2_combined_all/test_distance_prediction.png)",
        "![Combined-all comparison](round_2_combined_all/comparison.png)",
        "![Combined-all cartesian comparison](round_2_combined_all/cartesian_comparison.png)",
        "![Combined-all coordinate profile](round_2_combined_all/coordinate_error_profile.png)",
        "![Combined-all adaptive delays](round_2_combined_all/adaptive_delay_offsets.png)",
        "![Combined-all adaptive gains](round_2_combined_all/adaptive_gains.png)",
        "![Combined-all resonant tuning](round_2_combined_all/resonant_tuning.png)",
        "![Combined-all resonant spikes](round_2_combined_all/resonant_spikes.png)",
        "![Combined-all pre-pathway spikes](round_2_combined_all/pre_pathway_left_spikes.png)",
        "![Combined-all post-pathway spikes](round_2_combined_all/post_pathway_distance_spikes.png)",
        "",
        "## Interpretation",
        "",
        "If this model improves on the best individual result, the round-2 changes are largely complementary. If it only beats the fixed baseline but not the best individual variant, then the additions help in isolation but partly compete when stacked. If it loses to both, the short-data improvements are not additive and the combined model is over-complex for this regime.",
    ]
    report_path = outputs_root / "round_2_combined_all_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_round_2_combined_all(config: Any, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=10,
        scheduler_patience=3,
    )
    baseline = _load_short_combined_baseline(config, outputs)
    best_round2 = _load_best_round2_result(outputs.root)
    context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    params, baseline_label = _baseline_reference_params(context)

    total_start = time.perf_counter()
    prep_start = time.perf_counter()
    data = _prepare_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    data_prep_seconds = time.perf_counter() - prep_start

    spec = _combined_all_spec()
    output_root = outputs.root / "round_2_combined_all"
    output_root.mkdir(parents=True, exist_ok=True)

    model = _instantiate_combined_all_model(data, spec)
    training_start = time.perf_counter()
    train_result, uncertainty_module = _train_round2_model(model, data, target_bundle, spec, training_config)
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
    val_eval = _augment_with_cartesian_metrics(
        _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
    )
    test_eval = _augment_with_cartesian_metrics(
        _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    comparison_vs_baseline = _metrics_delta(test_eval.metrics, baseline["metrics"])
    best_round2_metrics = best_round2["test_metrics"]
    comparison_vs_best_round2 = _metrics_delta(test_eval.metrics, best_round2_metrics)
    cartesian_delta_vs_baseline = {
        "x_mae_delta": float(test_eval.metrics["x_mae_m"] - baseline["metrics"]["x_mae_m"]),
        "y_mae_delta": float(test_eval.metrics["y_mae_m"] - baseline["metrics"]["y_mae_m"]),
        "z_mae_delta": float(test_eval.metrics["z_mae_m"] - baseline["metrics"]["z_mae_m"]),
        "euclidean_error_delta": float(test_eval.metrics["euclidean_error_m"] - baseline["metrics"]["euclidean_error_m"]),
    }
    cartesian_delta_vs_best_round2 = {
        "x_mae_delta": float(test_eval.metrics["x_mae_m"] - float(best_round2_metrics["x_mae_m"])),
        "y_mae_delta": float(test_eval.metrics["y_mae_m"] - float(best_round2_metrics["y_mae_m"])),
        "z_mae_delta": float(test_eval.metrics["z_mae_m"] - float(best_round2_metrics["z_mae_m"])),
        "euclidean_error_delta": float(test_eval.metrics["euclidean_error_m"] - float(best_round2_metrics["euclidean_error_m"])),
    }
    accepted_vs_baseline = (
        float(test_eval.metrics["combined_error"]) < float(baseline["metrics"]["combined_error"]) - 1e-6
        and any(
            float(test_eval.metrics[key]) < float(baseline["metrics"][key]) - 1e-6
            for key in ("distance_mae_m", "azimuth_mae_deg", "elevation_mae_deg")
        )
    )
    beats_best_round2 = float(test_eval.metrics["combined_error"]) < float(best_round2_metrics["combined_error"]) - 1e-6

    artifacts = _save_improved_outputs(output_root.parent, spec, train_result, test_eval, baseline["metrics"], model)
    prediction_cache = _save_prediction_cache(output_root, test_eval.predictions, data.test_targets_raw)
    coordinate_profile = _save_coordinate_error_profiles(
        Path(prediction_cache),
        output_root / "coordinate_error_profile.png",
        f"{spec.title} Coordinate Error Profile",
    )
    cartesian_artifacts = _save_cartesian_outputs(output_root, spec.title, test_eval, baseline["metrics"])
    variant_artifacts = _save_variant_artifacts(output_root, model, test_eval.diagnostics)

    save_grouped_bar_chart(
        ["Combined", "Distance", "Azimuth", "Elevation"],
        {
            "Short Baseline": [
                float(baseline["metrics"]["combined_error"]),
                float(baseline["metrics"]["distance_mae_m"]),
                float(baseline["metrics"]["azimuth_mae_deg"]),
                float(baseline["metrics"]["elevation_mae_deg"]),
            ],
            "Best Round 2": [
                float(best_round2_metrics["combined_error"]),
                float(best_round2_metrics["distance_mae_m"]),
                float(best_round2_metrics["azimuth_mae_deg"]),
                float(best_round2_metrics["elevation_mae_deg"]),
            ],
            "Combined All": [
                float(test_eval.metrics["combined_error"]),
                float(test_eval.metrics["distance_mae_m"]),
                float(test_eval.metrics["azimuth_mae_deg"]),
                float(test_eval.metrics["elevation_mae_deg"]),
            ],
        },
        output_root / "triple_comparison.png",
        "Combined-All vs Baselines",
        ylabel="Error",
    )

    result = {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "rationale": spec.rationale,
        "implemented_steps": spec.implemented_steps,
        "analysis_focus": spec.analysis_focus,
        "loss_mode": spec.loss_mode,
        "dataset_mode": training_config.dataset_mode,
        "baseline_label": baseline_label,
        "accepted_vs_baseline": accepted_vs_baseline,
        "beats_best_round2": beats_best_round2,
        "training_config": {
            "max_epochs": training_config.max_epochs,
            "early_stopping_patience": training_config.early_stopping_patience,
            "scheduler_patience": training_config.scheduler_patience,
            "scheduler_factor": training_config.scheduler_factor,
            **spec.training_overrides,
        },
        "training": {
            "executed_epochs": train_result.executed_epochs,
            "best_epoch": train_result.best_epoch + 1,
            "stopped_early": train_result.stopped_early,
            "best_val_loss": format_float(train_result.best_loss),
            "best_val_combined_error": format_float(train_result.best_combined_error),
            "initial_learning_rate": format_float(
                float(params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0)),
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
        "val_metrics": {key: format_float(value) for key, value in val_eval.metrics.items()},
        "test_metrics": {key: format_float(value) for key, value in test_eval.metrics.items()},
        "comparison_vs_baseline": {key: format_float(value) for key, value in comparison_vs_baseline.items()},
        "comparison_vs_best_round2": {key: format_float(value) for key, value in comparison_vs_best_round2.items()},
        "cartesian_delta_vs_baseline": {key: format_float(value) for key, value in cartesian_delta_vs_baseline.items()},
        "cartesian_delta_vs_best_round2": {key: format_float(value) for key, value in cartesian_delta_vs_best_round2.items()},
        "learned_sigmas": None if learned_sigmas is None else {key: format_float(value) for key, value in learned_sigmas.items()},
        "artifacts": {
            **artifacts,
            **cartesian_artifacts,
            **variant_artifacts,
            "prediction_cache": prediction_cache,
            "coordinate_error_profile": coordinate_profile,
            "triple_comparison": str(output_root / "triple_comparison.png"),
        },
    }
    save_json(output_root / "result.json", result)
    report_path = _write_report(outputs.root, baseline, best_round2, result, training_config)
    summary = {
        "baseline": baseline["result"],
        "best_round2": best_round2,
        "result": result,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_2_combined_all_summary.json", summary)
    return summary
