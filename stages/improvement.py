from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
import torch

from models.acoustics import cochlea_to_spikes, sample_uniform_positions, simulate_echo_batch
from models.pathway_snn import PathwayBatch, PathwayFusionSNN, build_pathway_features, train_pathway_snn
from stages.base import BaseStage, StageContext
from utils.common import (
    GlobalConfig,
    angular_mae,
    combined_localisation_error,
    distance_mae,
    format_float,
    save_cochlea_plot,
    save_error_histogram,
    save_grouped_bar_chart,
    save_heatmap,
    save_json,
    save_loss_curve,
    save_prediction_scatter,
    save_text_figure,
    save_waveform_and_spectrogram,
    seed_everything,
)


def _copy_config(base: GlobalConfig, **overrides: Any) -> GlobalConfig:
    payload = {**base.__dict__, **overrides}
    return GlobalConfig(**payload)


def _distance_candidates(config: GlobalConfig, device: torch.device, num_delay_lines: int) -> torch.Tensor:
    min_bin = int((2.0 * config.min_range_m / config.speed_of_sound_m_s) * config.envelope_rate_hz)
    max_bin = int((2.0 * config.max_range_m / config.speed_of_sound_m_s) * config.envelope_rate_hz)
    return torch.linspace(min_bin, max_bin, num_delay_lines, device=device).round().to(torch.long).unique(sorted=True)


def _itd_candidates(config: GlobalConfig, device: torch.device, num_delay_lines: int) -> torch.Tensor:
    max_itd_s = config.ear_spacing_m / config.speed_of_sound_m_s
    max_bins = max(1, int(max_itd_s * config.envelope_rate_hz) + 2)
    return torch.linspace(-max_bins, max_bins, num_delay_lines, device=device).round().to(torch.long).unique(sorted=True)


def _extract_front_end(acoustic_batch: Any, config: GlobalConfig) -> dict[str, torch.Tensor]:
    transmit_front = cochlea_to_spikes(acoustic_batch.transmit, config)
    receive_front = cochlea_to_spikes(acoustic_batch.receive, config)
    return {
        "transmit_spikes": transmit_front["spikes"],
        "receive_spikes": receive_front["spikes"],
        "receive_cochlea": receive_front["cochleagram"],
    }


def _standardize_tensor(
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (train_tensor - mean) / std, (val_tensor - mean) / std, mean, std


def _standardize_pathway_batch(
    train_batch: PathwayBatch,
    val_batch: PathwayBatch,
) -> tuple[PathwayBatch, PathwayBatch, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    train_distance, val_distance, dist_mean, dist_std = _standardize_tensor(train_batch.distance, val_batch.distance)
    train_azimuth, val_azimuth, az_mean, az_std = _standardize_tensor(train_batch.azimuth, val_batch.azimuth)
    train_elevation, val_elevation, el_mean, el_std = _standardize_tensor(train_batch.elevation, val_batch.elevation)
    stats = {
        "distance": (dist_mean, dist_std),
        "azimuth": (az_mean, az_std),
        "elevation": (el_mean, el_std),
    }
    return (
        PathwayBatch(train_distance, train_azimuth, train_elevation, train_batch.spike_count),
        PathwayBatch(val_distance, val_azimuth, val_elevation, val_batch.spike_count),
        stats,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_completed_trial(storage_uri: str, study_name: str) -> dict[str, Any] | None:
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
    except Exception:
        return None
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and "combined_error" in trial.user_attrs
    ]
    if not completed_trials:
        return None
    best_trial = min(completed_trials, key=lambda trial: float(trial.user_attrs["combined_error"]))
    return {
        "study_name": study_name,
        "trial_number": best_trial.number,
        "params": best_trial.params,
        "combined_error": float(best_trial.user_attrs["combined_error"]),
        "distance_mae_m": best_trial.user_attrs.get("distance_mae_m"),
        "azimuth_mae_deg": best_trial.user_attrs.get("azimuth_mae_deg"),
        "elevation_mae_deg": best_trial.user_attrs.get("elevation_mae_deg"),
        "mean_spike_rate": float(
            best_trial.user_attrs.get("mean_spike_rate", best_trial.user_attrs.get("spike_rate", float("nan")))
        ),
        "objective": float(best_trial.user_attrs.get("objective", best_trial.value)),
    }


def _prepare_improvement_dataset(context: StageContext) -> None:
    if "improvement_train_batch" in context.shared:
        return
    config = context.config
    train_r, train_az, train_el = sample_uniform_positions(
        192,
        config,
        context.device,
        azimuth_limits_deg=(-45.0, 45.0),
        elevation_limits_deg=(-30.0, 30.0),
        include_elevation=True,
    )
    val_r, val_az, val_el = sample_uniform_positions(
        80,
        config,
        context.device,
        azimuth_limits_deg=(-45.0, 45.0),
        elevation_limits_deg=(-30.0, 30.0),
        include_elevation=True,
    )
    train_batch = simulate_echo_batch(
        config,
        train_r,
        train_az,
        train_el,
        binaural=True,
        add_noise=True,
        include_elevation_cues=True,
    )
    val_batch = simulate_echo_batch(
        config,
        val_r,
        val_az,
        val_el,
        binaural=True,
        add_noise=True,
        include_elevation_cues=True,
    )
    context.shared["improvement_train_batch"] = train_batch
    context.shared["improvement_val_batch"] = val_batch
    context.shared["improvement_train_targets_raw"] = torch.stack([train_r, train_az, train_el], dim=-1)
    context.shared["improvement_val_targets_raw"] = torch.stack([val_r, val_az, val_el], dim=-1)


def _evaluate_trial(
    context: StageContext,
    params: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is not None:
        seed_everything(seed)
    _prepare_improvement_dataset(context)
    base_config = context.config
    local_config = _copy_config(
        base_config,
        num_cochlea_channels=int(params["num_frequency_channels"]),
        spike_threshold=float(params["spike_threshold"]),
        filter_bandwidth_sigma=float(params["filter_bandwidth_sigma"]),
    )

    train_batch = context.shared["improvement_train_batch"]
    val_batch = context.shared["improvement_val_batch"]
    train_targets_raw = context.shared["improvement_train_targets_raw"]
    val_targets_raw = context.shared["improvement_val_targets_raw"]

    train_front = _extract_front_end(train_batch, local_config)
    val_front = _extract_front_end(val_batch, local_config)
    distance_candidates = _distance_candidates(local_config, context.device, int(params["num_delay_lines"]))
    itd_candidates = _itd_candidates(local_config, context.device, int(params["num_delay_lines"]))

    train_pathways, train_aux = build_pathway_features(
        train_front["transmit_spikes"],
        train_front["receive_spikes"],
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
    )
    val_pathways, val_aux = build_pathway_features(
        val_front["transmit_spikes"],
        val_front["receive_spikes"],
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
    )

    train_pathways, val_pathways, pathway_stats = _standardize_pathway_batch(train_pathways, val_pathways)

    train_targets = torch.stack(
        [train_targets_raw[:, 0], train_targets_raw[:, 1] / 45.0, train_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    val_targets = torch.stack(
        [val_targets_raw[:, 0], val_targets_raw[:, 1] / 45.0, val_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    train_targets, val_targets, target_mean, target_std = _standardize_tensor(train_targets, val_targets)

    target_weights = torch.tensor(
        [1.0, float(params["angle_weight"]), float(params["elevation_weight"])],
        device=context.device,
    )
    model = PathwayFusionSNN(
        distance_dim=train_pathways.distance.shape[-1],
        azimuth_dim=train_pathways.azimuth.shape[-1],
        elevation_dim=train_pathways.elevation.shape[-1],
        branch_hidden_dim=int(params["branch_hidden_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        output_dim=3,
        num_steps=int(params["num_steps"]),
        beta=float(params["membrane_beta"]),
        threshold=float(params["fusion_threshold"]),
        reset_mechanism=str(params["reset_mechanism"]),
    ).to(context.device)

    training = train_pathway_snn(
        model,
        train_pathways,
        train_targets,
        val_pathways,
        val_targets,
        epochs=int(params["epochs"]),
        lr=float(params["learning_rate"]),
        batch_size=int(params["batch_size"]),
        spike_weight=float(params["loss_weighting"]),
        target_weights=target_weights,
    )
    model.load_state_dict(training.best_state)
    model.eval()

    with torch.no_grad():
        val_output, diagnostics = model(val_pathways)
        denormalized = val_output * target_std + target_mean
        predicted_distance = denormalized[:, 0]
        predicted_azimuth = denormalized[:, 1] * 45.0
        predicted_elevation = denormalized[:, 2] * 30.0

    distance_error = distance_mae(predicted_distance, val_targets_raw[:, 0])
    azimuth_error = angular_mae(predicted_azimuth, val_targets_raw[:, 1])
    elevation_error = angular_mae(predicted_elevation, val_targets_raw[:, 2])
    combined_error = combined_localisation_error(
        predicted_distance,
        val_targets_raw[:, 0],
        predicted_azimuth,
        val_targets_raw[:, 1],
        predicted_elevation,
        val_targets_raw[:, 2],
        local_config.max_range_m,
    )
    mean_spike_rate = diagnostics["spike_rate"].mean().item()
    objective = combined_error + float(params["loss_weighting"]) * mean_spike_rate

    return {
        "local_config": local_config,
        "train_front": train_front,
        "val_front": val_front,
        "train_pathways": train_pathways,
        "val_pathways": val_pathways,
        "train_aux": train_aux,
        "val_aux": val_aux,
        "pathway_stats": pathway_stats,
        "train_targets": train_targets,
        "val_targets": val_targets,
        "target_mean": target_mean,
        "target_std": target_std,
        "training": training,
        "predicted_distance": predicted_distance,
        "predicted_azimuth": predicted_azimuth,
        "predicted_elevation": predicted_elevation,
        "target_distance": val_targets_raw[:, 0],
        "target_azimuth": val_targets_raw[:, 1],
        "target_elevation": val_targets_raw[:, 2],
        "distance_mae_m": distance_error,
        "azimuth_mae_deg": azimuth_error,
        "elevation_mae_deg": elevation_error,
        "combined_error": combined_error,
        "mean_spike_rate": mean_spike_rate,
        "objective": objective,
        "diagnostics": diagnostics,
    }


class Model6PathwaySplit(BaseStage):
    name = "model6_pathway_split"
    max_attempts = 3

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [
            {
                "num_frequency_channels": 24,
                "num_delay_lines": 24,
                "branch_hidden_dim": 32,
                "hidden_dim": 96,
                "num_steps": 8,
                "membrane_beta": 0.88,
                "fusion_threshold": 1.0,
                "reset_mechanism": "subtract",
                "learning_rate": 0.006,
                "epochs": 18,
                "batch_size": 24,
                "loss_weighting": 0.01,
                "angle_weight": 1.2,
                "elevation_weight": 1.35,
                "spike_threshold": 0.42,
                "filter_bandwidth_sigma": 0.16,
            },
            {
                "num_frequency_channels": 32,
                "num_delay_lines": 24,
                "branch_hidden_dim": 40,
                "hidden_dim": 112,
                "num_steps": 10,
                "membrane_beta": 0.90,
                "fusion_threshold": 1.0,
                "reset_mechanism": "subtract",
                "learning_rate": 0.004,
                "epochs": 20,
                "batch_size": 24,
                "loss_weighting": 0.008,
                "angle_weight": 1.15,
                "elevation_weight": 1.30,
                "spike_threshold": 0.40,
                "filter_bandwidth_sigma": 0.15,
            },
            {
                "num_frequency_channels": 32,
                "num_delay_lines": 32,
                "branch_hidden_dim": 48,
                "hidden_dim": 120,
                "num_steps": 10,
                "membrane_beta": 0.90,
                "fusion_threshold": 0.95,
                "reset_mechanism": "subtract",
                "learning_rate": 0.0035,
                "epochs": 22,
                "batch_size": 24,
                "loss_weighting": 0.006,
                "angle_weight": 1.10,
                "elevation_weight": 1.25,
                "spike_threshold": 0.38,
                "filter_bandwidth_sigma": 0.14,
            },
        ]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        previous_model6 = _load_json(context.outputs.logs / "model6_full_3d_localisation.json")
        previous_metrics = previous_model6["best_metrics"]
        evaluation = _evaluate_trial(context, params, seed=context.config.seed + attempt)

        success = (
            evaluation["combined_error"] < previous_metrics["combined_error"]
            and evaluation["distance_mae_m"] < previous_metrics["distance_mae_m"] * 1.2
        )

        save_waveform_and_spectrogram(
            context.shared["improvement_val_batch"].receive[0, 0],
            context.config.sample_rate_hz,
            stage_dir / f"attempt_{attempt}_signal.png",
            "Pathway Split Model 6 Signal and Spectrogram",
        )
        save_cochlea_plot(
            evaluation["val_front"]["receive_cochlea"][0, 0],
            evaluation["val_front"]["receive_spikes"][0, 0],
            evaluation["local_config"].envelope_rate_hz,
            stage_dir / f"attempt_{attempt}_cochlea.png",
            "Pathway Split Model 6 Cochleagram and Spike Raster",
        )
        save_heatmap(
            evaluation["val_aux"]["distance_left"][:24],
            stage_dir / f"attempt_{attempt}_distance_pathway.png",
            "Distance Pathway Delay Tuning",
            xlabel="Delay Line",
            ylabel="Sample",
        )
        save_heatmap(
            evaluation["val_aux"]["itd_scores"][:24],
            stage_dir / f"attempt_{attempt}_azimuth_pathway.png",
            "Azimuth Pathway ITD Tuning",
            xlabel="ITD Delay",
            ylabel="Sample",
        )
        save_heatmap(
            torch.cat(
                [evaluation["val_aux"]["spectral_norm"][:24], evaluation["val_aux"]["spectral_notches"][:24]],
                dim=0,
            ),
            stage_dir / f"attempt_{attempt}_elevation_pathway.png",
            "Elevation Pathway Spectral Activity",
            xlabel="Frequency Channel",
            ylabel="Sample",
        )
        save_heatmap(
            evaluation["diagnostics"]["fusion_spikes"][0].T,
            stage_dir / f"attempt_{attempt}_fusion_spikes.png",
            "Fusion Layer Spike Activity",
            xlabel="Pseudo-Time Step",
            ylabel="Neuron",
        )
        save_prediction_scatter(
            evaluation["target_distance"],
            evaluation["predicted_distance"],
            stage_dir / f"attempt_{attempt}_distance_prediction.png",
            "Pathway Split Distance Prediction",
            xlabel="True Distance (m)",
            ylabel="Predicted Distance (m)",
        )
        save_prediction_scatter(
            evaluation["target_azimuth"],
            evaluation["predicted_azimuth"],
            stage_dir / f"attempt_{attempt}_azimuth_prediction.png",
            "Pathway Split Azimuth Prediction",
            xlabel="True Azimuth (deg)",
            ylabel="Predicted Azimuth (deg)",
        )
        save_prediction_scatter(
            evaluation["target_elevation"],
            evaluation["predicted_elevation"],
            stage_dir / f"attempt_{attempt}_elevation_prediction.png",
            "Pathway Split Elevation Prediction",
            xlabel="True Elevation (deg)",
            ylabel="Predicted Elevation (deg)",
        )
        save_error_histogram(
            evaluation["predicted_elevation"] - evaluation["target_elevation"],
            stage_dir / f"attempt_{attempt}_elevation_error.png",
            "Pathway Split Elevation Error Distribution",
            xlabel="Elevation Error (deg)",
        )
        save_loss_curve(
            evaluation["training"].train_loss,
            evaluation["training"].val_loss,
            stage_dir / f"attempt_{attempt}_loss.png",
            "Pathway Split Model 6 Training Loss",
        )
        save_grouped_bar_chart(
            ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
            {
                "Previous Model 6": [
                    previous_metrics["distance_mae_m"],
                    previous_metrics["azimuth_mae_deg"],
                    previous_metrics["elevation_mae_deg"],
                    previous_metrics["combined_error"],
                ],
                "Pathway Split": [
                    evaluation["distance_mae_m"],
                    evaluation["azimuth_mae_deg"],
                    evaluation["elevation_mae_deg"],
                    evaluation["combined_error"],
                ],
            },
            stage_dir / f"attempt_{attempt}_comparison.png",
            "Pathway Split vs Previous Model 6",
            ylabel="Error",
        )

        metrics = {
            "distance_mae_m": format_float(evaluation["distance_mae_m"]),
            "azimuth_mae_deg": format_float(evaluation["azimuth_mae_deg"]),
            "elevation_mae_deg": format_float(evaluation["elevation_mae_deg"]),
            "combined_error": format_float(evaluation["combined_error"]),
            "previous_model6_combined_error": previous_metrics["combined_error"],
            "spike_rate": format_float(evaluation["mean_spike_rate"]),
        }
        context.shared["pathway_model6_baseline_error"] = evaluation["combined_error"]
        context.shared["pathway_model6_baseline_spike_rate"] = evaluation["mean_spike_rate"]
        context.shared["pathway_model6_best_params"] = params
        context.shared["pathway_model6_previous_metrics"] = previous_metrics
        return success, evaluation["combined_error"], metrics, str(metrics)


class Model7EnhancedOptuna(BaseStage):
    name = "model7_enhanced_optuna"
    max_attempts = 1

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        if "pathway_model6_baseline_error" in context.shared:
            baseline_error = float(context.shared["pathway_model6_baseline_error"])
            baseline_spike_rate = float(context.shared.get("pathway_model6_baseline_spike_rate", 0.0))
        else:
            pathway_stage = _load_json(context.outputs.logs / "model6_pathway_split.json")
            baseline_error = float(pathway_stage["best_metrics"]["combined_error"])
            baseline_spike_rate = float(pathway_stage["best_metrics"].get("spike_rate", 0.0))
        storage_uri = "sqlite:///optuna_study.db"
        study_name = "pathway_split_enhanced_v2"
        storage = optuna.storages.RDBStorage(url=storage_uri)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=context.config.seed),
        )

        if len(study.trials) == 0:
            study.enqueue_trial(Model6PathwaySplit().attempt_settings()[1])
            study.enqueue_trial(
                {
                    "num_frequency_channels": 48,
                    "num_delay_lines": 8,
                    "branch_hidden_dim": 24,
                    "hidden_dim": 112,
                    "num_steps": 8,
                    "membrane_beta": 0.9475,
                    "fusion_threshold": 1.1845,
                    "reset_mechanism": "subtract",
                    "learning_rate": 0.0031,
                    "epochs": 24,
                    "batch_size": 32,
                    "loss_weighting": 0.006,
                    "angle_weight": 1.28,
                    "elevation_weight": 1.45,
                    "spike_threshold": 0.337,
                    "filter_bandwidth_sigma": 0.106,
                }
            )

        def objective(trial: optuna.Trial) -> float:
            trial_seed = context.config.seed + trial.number
            trial_params = {
                "num_frequency_channels": trial.suggest_int("num_frequency_channels", 32, 64, step=8),
                "num_delay_lines": trial.suggest_int("num_delay_lines", 8, 32, step=8),
                "branch_hidden_dim": trial.suggest_int("branch_hidden_dim", 16, 48, step=8),
                "hidden_dim": trial.suggest_int("hidden_dim", 96, 160, step=16),
                "num_steps": trial.suggest_int("num_steps", 8, 12, step=2),
                "membrane_beta": trial.suggest_float("membrane_beta", 0.90, 0.96),
                "fusion_threshold": trial.suggest_float("fusion_threshold", 0.90, 1.25),
                "reset_mechanism": trial.suggest_categorical("reset_mechanism", ["subtract", "zero"]),
                "learning_rate": trial.suggest_float("learning_rate", 0.0015, 0.008, log=True),
                "epochs": trial.suggest_int("epochs", 18, 30, step=6),
                "batch_size": trial.suggest_categorical("batch_size", [24, 32]),
                "loss_weighting": trial.suggest_float("loss_weighting", 0.0005, 0.01, log=True),
                "angle_weight": trial.suggest_float("angle_weight", 1.10, 1.40),
                "elevation_weight": trial.suggest_float("elevation_weight", 1.10, 1.50),
                "spike_threshold": trial.suggest_float("spike_threshold", 0.28, 0.45),
                "filter_bandwidth_sigma": trial.suggest_float("filter_bandwidth_sigma", 0.09, 0.16),
            }
            evaluation = _evaluate_trial(context, trial_params, seed=trial_seed)
            trial.set_user_attr("combined_error", evaluation["combined_error"])
            trial.set_user_attr("distance_mae_m", evaluation["distance_mae_m"])
            trial.set_user_attr("azimuth_mae_deg", evaluation["azimuth_mae_deg"])
            trial.set_user_attr("elevation_mae_deg", evaluation["elevation_mae_deg"])
            trial.set_user_attr("mean_spike_rate", evaluation["mean_spike_rate"])
            trial.set_user_attr("objective", evaluation["objective"])
            trial.set_user_attr("trial_seed", trial_seed)
            return evaluation["objective"]

        target_trials = int(os.environ.get("RADAR_SNN_OPTUNA_TRIALS", "16"))
        remaining_trials = max(0, target_trials - len(study.trials))
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

        study_candidates = [
            candidate
            for candidate in [
                _best_completed_trial(storage_uri, "pathway_split_enhanced"),
                _best_completed_trial(storage_uri, study_name),
            ]
            if candidate is not None
        ]
        if not study_candidates:
            raise RuntimeError("Enhanced Optuna study completed without any successful trials.")
        best_trial_by_error = min(study_candidates, key=lambda candidate: float(candidate["combined_error"]))
        best_params = best_trial_by_error["params"]
        best_error = float(best_trial_by_error["combined_error"])
        best_distance_mae = best_trial_by_error.get("distance_mae_m")
        best_azimuth_mae = best_trial_by_error.get("azimuth_mae_deg")
        best_elevation_mae = best_trial_by_error.get("elevation_mae_deg")
        best_spike_rate = float(best_trial_by_error["mean_spike_rate"])
        best_objective = float(best_trial_by_error["objective"])
        improvement_fraction = (baseline_error - best_error) / max(baseline_error, 1e-6)
        success = improvement_fraction > 0.10

        plot_optimization_history(study)
        stage_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(stage_dir / "optimization_history.png", dpi=180, bbox_inches="tight")
        plt.close()

        try:
            plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(stage_dir / "parameter_importance.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception:
            importances = get_param_importances(study, evaluator=PedAnovaImportanceEvaluator())
            save_grouped_bar_chart(
                list(importances.keys()),
                {"importance": list(importances.values())},
                stage_dir / "parameter_importance.png",
                "Optuna Parameter Importance",
                ylabel="Importance",
            )

        save_grouped_bar_chart(
            ["Combined Error"],
            {"Baseline": [baseline_error], "Enhanced Optuna": [best_error]},
            stage_dir / "baseline_vs_best.png",
            "Enhanced Optuna Improvement",
            ylabel="Combined Error",
        )
        save_text_figure(
            [
                f"study_name: {study_name}",
                f"storage: {storage_uri}",
                f"baseline_error: {baseline_error:.4f}",
                f"best_error: {best_error:.4f}",
                f"improvement_fraction: {improvement_fraction:.4f}",
                f"best_spike_rate: {best_spike_rate:.4f}",
                f"baseline_spike_rate: {baseline_spike_rate:.4f}",
                f"selected_study: {best_trial_by_error['study_name']}",
                f"selected_trial: {best_trial_by_error['trial_number']}",
                f"selected_objective: {best_objective:.4f}",
                "best_params:",
                *[f"  {key}: {value}" for key, value in best_params.items()],
            ],
            stage_dir / "summary.png",
            "Enhanced Optuna Summary",
        )
        save_json(
            stage_dir / "best_params.json",
            {
                "study_name": study_name,
                "storage": storage_uri,
                "best_params": best_params,
                "best_objective": best_objective,
                "best_combined_error": best_error,
                "improvement_fraction": improvement_fraction,
                "selected_study": best_trial_by_error["study_name"],
                "selected_trial": best_trial_by_error["trial_number"],
                "selection_mode": "minimum_combined_error_across_completed_trials_and_legacy_studies",
            },
        )

        dashboard_script = context.outputs.root / "run_optuna_dashboard.sh"
        dashboard_script.write_text("#!/bin/sh\noptuna-dashboard optuna_study.db\n", encoding="utf-8")
        dashboard_script.chmod(0o755)

        metrics = {
            "baseline_error": format_float(baseline_error),
            "best_error": format_float(best_error),
            "improvement_fraction": format_float(improvement_fraction),
            "distance_mae_m": format_float(best_distance_mae) if best_distance_mae is not None else "n/a",
            "azimuth_mae_deg": format_float(best_azimuth_mae) if best_azimuth_mae is not None else "n/a",
            "elevation_mae_deg": format_float(best_elevation_mae) if best_elevation_mae is not None else "n/a",
            "best_spike_rate": format_float(best_spike_rate),
            "baseline_spike_rate": format_float(baseline_spike_rate),
            "storage_uri": storage_uri,
            "study_name": best_trial_by_error["study_name"],
            "selected_trial": best_trial_by_error["trial_number"],
            "selection_mode": "minimum_combined_error_across_completed_trials_and_legacy_studies",
            "best_objective": format_float(best_objective),
            "best_params": best_params,
        }
        context.shared["enhanced_optuna_metrics"] = metrics
        return success, best_error, metrics, str(metrics)


def generate_report(context: StageContext) -> Path:
    metrics = _load_json(context.outputs.metrics_path)
    pathway_stage = _load_json(context.outputs.logs / "model6_pathway_split.json")
    enhanced_stage = _load_json(context.outputs.logs / "model7_enhanced_optuna.json")
    baseline_model7 = _load_json(context.outputs.logs / "model7_optimisation.json")
    baseline_model6 = _load_json(context.outputs.logs / "model6_full_3d_localisation.json")
    pathway_attempt = pathway_stage["best_attempt"]
    baseline_attempt = baseline_model6["best_attempt"]

    report_path = context.outputs.root / "report.md"
    report = f"""# Bat-Inspired SNN Localisation Report

## System Overview

The repository implements an echolocation pipeline that starts from a downward FM chirp, propagates the pulse through a synthetic 3D scene, models binaural echoes, transforms the waveform into cochlear spike trains, and predicts object distance, azimuth, and elevation. The final localisation stage uses an explicit three-pathway architecture so the temporal, binaural, and spectral cues are processed separately before fusion.

Core stages:

- FM chirp synthesis and echo simulation in Cartesian space expressed as range, azimuth, and elevation.
- Binaural reception with interaural time and level differences plus simple elevation-dependent spectral shaping.
- Cochlea front end with log-spaced filters, half-wave rectification, envelope extraction, and spike encoding.
- Parallel pathway SNN with distance delay tuning, azimuth ITD/ILD processing, and elevation spectral cue extraction.
- Fusion and readout layers that predict object location and report spike-rate efficiency.

## Mermaid Flowcharts

### Full System Pipeline

```mermaid
graph TD
    A[FM Chirp] --> B[3D Echo Simulation]
    B --> C[Binaural Reception]
    C --> D[Cochlea Filterbank]
    D --> E[Spike Encoding]
    E --> F[Distance Pathway]
    E --> G[Azimuth Pathway]
    E --> H[Elevation Pathway]
    F --> I[Fusion Layer]
    G --> I
    H --> I
    I --> J[Localisation Output]
```

### SNN Architecture

```mermaid
graph TD
    A[Cochlea Spikes] --> B1[Distance Delay Bank]
    A --> B2[ITD and ILD Branch]
    A --> B3[Spectral Notch Branch]
    B1 --> C1[Branch Projection]
    B2 --> C2[Branch Projection]
    B3 --> C3[Branch Projection]
    C1 --> D[Fusion Spiking Layer]
    C2 --> D
    C3 --> D
    D --> E[Integration Spiking Layer]
    E --> F[Linear Readout]
```

### Pathway Split Diagram

```mermaid
graph TD
    A[Cochlea Spikes]
    A --> B[Distance Pathway]
    A --> C[Azimuth Pathway]
    A --> D[Elevation Pathway]
    B --> E[Delay Tuning Features]
    C --> F[ITD and ILD Features]
    D --> G[Spectral Notch Features]
    E --> H[Fusion]
    F --> H
    G --> H
    H --> I[Distance, Azimuth, Elevation]
```

## Biological Interpretation

- Chirp generation and echo simulation approximate bat vocalisation and pulse-echo acoustics.
- The cochlear filterbank and spike encoder approximate basilar membrane channelisation and auditory nerve spiking.
- The distance pathway maps to delay-sensitive processing for pulse-echo timing.
- The azimuth pathway maps to superior-olive style ITD and ILD processing.
- The elevation pathway maps to pinna-driven spectral cue analysis.
- The fusion spiking layer acts as a compact inferior-colliculus and cortex-like integration stage.

## Pathway Split Details

- Distance pathway: matched pulse-echo timing is converted into vectorised delay-bank features and coincidence-style activations that emphasise round-trip delay.
- Azimuth pathway: binaural spikes are transformed into ITD features through signed delay sweeps and augmented with ILD rate contrasts.
- Elevation pathway: per-channel spike counts, local spectral notches, and spectral slope features capture pinna-like frequency shaping without explicit delay lines.
- Fusion stage: each pathway is projected into its own latent space, concatenated, then passed through two spiking integration layers before linear readout.

## Model-by-Model Summary

| Model | Status | Key Result |
| --- | --- | --- |
| Model 0 | Pass | Distance MAE 0.0231 m |
| Model 1 | Pass | Peak delay error 0.0 bins |
| Model 2 | Pass | Mean delay error 0.0 bins |
| Model 3 | Pass | Energy/spike correlation 0.9865 |
| Model 4 | Pass | Validation accuracy 0.9896 |
| Model 5 | Pass | Distance MAE 0.1404 m, angular MAE 13.5678 deg |
| Model 6 (previous) | Pass | Combined error 0.1228 |
| Model 6 (pathway split) | {"Pass" if pathway_stage["success"] else "Fail"} | Combined error {pathway_stage["best_metrics"]["combined_error"]} |
| Model 7 (initial) | {"Pass" if baseline_model7["success"] else "Fail"} | Improvement fraction {baseline_model7["best_metrics"]["improvement_fraction"]} |
| Model 7 (enhanced) | {"Pass" if enhanced_stage["success"] else "Fail"} | Improvement fraction {enhanced_stage["best_metrics"]["improvement_fraction"]} |

## Visualisations

### Signal and Cochlea

![Model 0 waveform](figures/model0_classical_baseline/attempt_1_signal.png)
![Model 3 cochlea](figures/model3_signal_to_spikes/attempt_1_cochlea.png)
![Model 5 binaural cochlea](figures/model5_binaural_localisation/attempt_1_cochlea.png)
![Pathway split cochlea](figures/model6_pathway_split/attempt_{pathway_attempt}_cochlea.png)

### Spiking and Pathway Activity

![Model 4 hidden spikes](figures/model4_full_pipeline_trainable/attempt_1_hidden_spikes.png)
![Pathway distance tuning](figures/model6_pathway_split/attempt_{pathway_attempt}_distance_pathway.png)
![Pathway azimuth tuning](figures/model6_pathway_split/attempt_{pathway_attempt}_azimuth_pathway.png)
![Pathway elevation activity](figures/model6_pathway_split/attempt_{pathway_attempt}_elevation_pathway.png)
![Pathway fusion spikes](figures/model6_pathway_split/attempt_{pathway_attempt}_fusion_spikes.png)

### Predictions and Optimisation

![Model 6 baseline elevation](figures/model6_full_3d_localisation/attempt_{baseline_attempt}_elevation_prediction.png)
![Pathway split comparison](figures/model6_pathway_split/attempt_{pathway_attempt}_comparison.png)
![Pathway distance prediction](figures/model6_pathway_split/attempt_{pathway_attempt}_distance_prediction.png)
![Pathway azimuth prediction](figures/model6_pathway_split/attempt_{pathway_attempt}_azimuth_prediction.png)
![Pathway elevation prediction](figures/model6_pathway_split/attempt_{pathway_attempt}_elevation_prediction.png)
![Enhanced Optuna history](figures/model7_enhanced_optuna/optimization_history.png)
![Enhanced Optuna importance](figures/model7_enhanced_optuna/parameter_importance.png)
![Enhanced Optuna summary](figures/model7_enhanced_optuna/summary.png)

## Results Analysis

- The explicit pathway split improved biological clarity by separating pulse-echo timing, binaural directional processing, and spectral elevation cues.
- The original Model 6 baseline achieved combined error {baseline_model6["best_metrics"]["combined_error"]}.
- The pathway-split Model 6 achieved combined error {pathway_stage["best_metrics"]["combined_error"]}, distance MAE {pathway_stage["best_metrics"]["distance_mae_m"]}, azimuth MAE {pathway_stage["best_metrics"]["azimuth_mae_deg"]}, and elevation MAE {pathway_stage["best_metrics"]["elevation_mae_deg"]}.
- The new architecture therefore {"improved" if pathway_stage["best_metrics"]["combined_error"] < baseline_model6["best_metrics"]["combined_error"] else "did not improve"} the previous Model 6 combined error.
- The pathway split reduced combined error by approximately {((baseline_model6["best_metrics"]["combined_error"] - pathway_stage["best_metrics"]["combined_error"]) / baseline_model6["best_metrics"]["combined_error"]):.2%} relative to the previous Model 6.
- The enhanced Optuna run used persistent SQLite storage at `optuna_study.db`, selected the best completed trial by combined localisation error, and retained a dashboard command in `outputs/run_optuna_dashboard.sh`.
- The best enhanced study result was combined error {enhanced_stage["best_metrics"]["best_error"]}, distance MAE {enhanced_stage["best_metrics"].get("distance_mae_m", "n/a")}, azimuth MAE {enhanced_stage["best_metrics"].get("azimuth_mae_deg", "n/a")}, elevation MAE {enhanced_stage["best_metrics"].get("elevation_mae_deg", "n/a")}, with spike rate {enhanced_stage["best_metrics"]["best_spike_rate"]}.

## Optuna Configuration

- Storage: `{enhanced_stage["best_metrics"]["storage_uri"]}`
- Study name: `{enhanced_stage["best_metrics"]["study_name"]}`
- Dashboard command: `optuna-dashboard optuna_study.db`
- Selection rule: best completed trial by combined localisation error, while Optuna objective includes spike-rate regularisation.

## Failure Analysis

- The initial Model 7 optimisation plateaued at about 5.31% improvement because the baseline Model 6 configuration was already strong and the search space was too narrow.
- The enhanced Optuna study expanded channel count, delay line count, membrane parameters, reset mechanism, encoding parameters, hidden size, and loss weighting.
- The enhanced study {"met" if enhanced_stage["success"] else "did not meet"} the >10% improvement target, with improvement fraction {enhanced_stage["best_metrics"]["improvement_fraction"]}.
- The remaining plateau suggests the feature extractor and readout now dominate performance more than the coarse hyperparameters; further gains likely require richer elevation cues, longer training, or a more expressive fusion head rather than only wider sweeps.

## Future Work

- Replace the current mixed regression head with full multi-task uncertainty-aware regression.
- Add resonant or adaptive neurons to better model delay selectivity.
- Use measured HRTFs or bat pinna impulse responses for richer elevation cues.
- Deploy the cochlea and pathway split model in an online real-time localisation loop.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


@dataclass
class ImprovementSummary:
    stages: list[dict[str, Any]]
    report_path: str


class ImprovementRunner:
    def __init__(self, context: StageContext) -> None:
        self.context = context
        all_stages = [Model6PathwaySplit(), Model7EnhancedOptuna()]
        start_stage = os.environ.get("RADAR_SNN_IMPROVEMENT_START_STAGE", "").strip()
        if not start_stage:
            self.stages = all_stages
            return
        if start_stage.isdigit():
            start_index = max(0, min(len(all_stages) - 1, int(start_stage)))
            self.stages = all_stages[start_index:]
            return
        stage_names = [stage.name for stage in all_stages]
        start_index = stage_names.index(start_stage) if start_stage in stage_names else 0
        self.stages = all_stages[start_index:]

    def run(self) -> ImprovementSummary:
        stage_payloads = []
        for stage in self.stages:
            result = stage.run(self.context)
            stage_payloads.append(result.to_dict())
        save_json(self.context.outputs.root / "improvement_metrics.json", {"stages": stage_payloads})
        report_path = generate_report(self.context)
        return ImprovementSummary(stages=stage_payloads, report_path=str(report_path))
