from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import optuna
from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
import torch
import torch.nn as nn

from models.experimental_variants import ExperimentBatch
from models.pathway_snn import PathwayBatch, build_pathway_features
from models.round2_variants import AllRound2CombinedModel, AllRound2Encoder
from stages.base import StageContext
from stages.combined_experiment import _save_coordinate_error_profiles, _save_prediction_cache
from stages.experiments import (
    PreparedExperimentData,
    TaskUncertaintyWeights,
    _baseline_reference_params,
    _metrics_delta,
)
from stages.improved_experiments import _evaluate_improved_model, _prepare_target_bundle, _save_improved_outputs
from stages.improvement import (
    _apply_standardization,
    _copy_config,
    _distance_candidates,
    _extract_front_end,
    _fit_standardization,
    _itd_candidates,
    _prepare_dataset_bundle,
    _slice_acoustic_batch,
)
from stages.round_2_experiments import (
    Round2ExperimentSpec,
    _augment_with_cartesian_metrics,
    _load_short_combined_baseline,
    _round2_loss_components,
    _save_cartesian_outputs,
    _save_variant_artifacts,
)
from stages.training_improved_experiments import EnhancedPathwayTrainingResult, EnhancedTrainingConfig
from utils.common import format_float, save_grouped_bar_chart, save_json, seed_everything


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
    return min(experiments, key=lambda item: float(item["test_metrics"]["combined_error"]))


def _instantiate_combined_all_model(
    data: PreparedExperimentData,
    spec: Round2ExperimentSpec,
    params_override: dict[str, Any] | None = None,
) -> torch.nn.Module:
    params = data.params if params_override is None else params_override
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


def _timed_build_experiment_split(
    acoustic_batch: Any,
    local_config: Any,
    distance_candidates: torch.Tensor,
    itd_candidates: torch.Tensor,
    *,
    num_delay_lines: int,
    num_frequency_channels: int,
    chunk_size: int,
) -> tuple[ExperimentBatch, dict[str, Any]]:
    transmit_spike_chunks: list[torch.Tensor] = []
    receive_spike_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    azimuth_chunks: list[torch.Tensor] = []
    elevation_chunks: list[torch.Tensor] = []
    spike_count_chunks: list[torch.Tensor] = []

    timing = {
        "samples": int(acoustic_batch.receive.shape[0]),
        "chunks": 0,
        "front_end_seconds": 0.0,
        "pathway_seconds": 0.0,
        "concatenate_seconds": 0.0,
    }

    for start in range(0, acoustic_batch.receive.shape[0], chunk_size):
        stop = min(acoustic_batch.receive.shape[0], start + chunk_size)
        chunk_batch = _slice_acoustic_batch(acoustic_batch, slice(start, stop))
        timing["chunks"] += 1

        front_start = time.perf_counter()
        front = _extract_front_end(chunk_batch, local_config, include_cochlea=False)
        timing["front_end_seconds"] += time.perf_counter() - front_start

        pathway_start = time.perf_counter()
        pathways, _ = build_pathway_features(
            front["transmit_spikes"],
            front["receive_spikes"],
            distance_candidates,
            itd_candidates,
            num_delay_lines=num_delay_lines,
            num_frequency_channels=num_frequency_channels,
        )
        timing["pathway_seconds"] += time.perf_counter() - pathway_start

        transmit_spike_chunks.append(front["transmit_spikes"].to(torch.bool))
        receive_spike_chunks.append(front["receive_spikes"].to(torch.bool))
        distance_chunks.append(pathways.distance)
        azimuth_chunks.append(pathways.azimuth)
        elevation_chunks.append(pathways.elevation)
        spike_count_chunks.append(pathways.spike_count)

    concat_start = time.perf_counter()
    pathway_batch = PathwayBatch(
        distance=torch.cat(distance_chunks, dim=0),
        azimuth=torch.cat(azimuth_chunks, dim=0),
        elevation=torch.cat(elevation_chunks, dim=0),
        spike_count=torch.cat(spike_count_chunks, dim=0),
    )
    experiment_batch = ExperimentBatch(
        transmit_wave=acoustic_batch.transmit,
        receive_wave=acoustic_batch.receive,
        pathway=pathway_batch,
        transmit_spikes=torch.cat(transmit_spike_chunks, dim=0),
        receive_spikes=torch.cat(receive_spike_chunks, dim=0),
        spike_count=pathway_batch.spike_count,
    )
    timing["concatenate_seconds"] = time.perf_counter() - concat_start
    return experiment_batch, timing


def _prepare_profiled_experiment_data(
    context: StageContext,
    params: dict[str, Any],
    dataset_mode: str,
) -> tuple[PreparedExperimentData, dict[str, Any]]:
    prepared_key = f"experiment_data::{dataset_mode}::{json.dumps(params, sort_keys=True)}"
    profile_key = f"profile::{prepared_key}"
    if prepared_key in context.shared and profile_key in context.shared:
        return context.shared[prepared_key], context.shared[profile_key]

    prep_start = time.perf_counter()
    dataset_start = time.perf_counter()
    dataset_bundle = _prepare_dataset_bundle(context, dataset_mode)
    dataset_seconds = time.perf_counter() - dataset_start

    local_config = _copy_config(
        context.config,
        num_cochlea_channels=int(params["num_frequency_channels"]),
        spike_threshold=float(params["spike_threshold"]),
        filter_bandwidth_sigma=float(params["filter_bandwidth_sigma"]),
    )

    candidate_start = time.perf_counter()
    distance_candidates = _distance_candidates(local_config, context.device, int(params["num_delay_lines"]))
    itd_candidates = _itd_candidates(local_config, context.device, int(params["num_delay_lines"]))
    candidate_seconds = time.perf_counter() - candidate_start

    chunk_size = int(os.environ.get("RADAR_SNN_FEATURE_CHUNK_SIZE", "64"))
    if chunk_size <= 0:
        chunk_size = 64

    train_batch, train_timing = _timed_build_experiment_split(
        dataset_bundle.train_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )
    val_batch, val_timing = _timed_build_experiment_split(
        dataset_bundle.val_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )
    test_batch, test_timing = _timed_build_experiment_split(
        dataset_bundle.test_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )

    standardization_start = time.perf_counter()
    distance_mean, distance_std = _fit_standardization(train_batch.pathway.distance)
    azimuth_mean, azimuth_std = _fit_standardization(train_batch.pathway.azimuth)
    elevation_mean, elevation_std = _fit_standardization(train_batch.pathway.elevation)

    train_batch.pathway = PathwayBatch(
        distance=_apply_standardization(train_batch.pathway.distance, distance_mean, distance_std),
        azimuth=_apply_standardization(train_batch.pathway.azimuth, azimuth_mean, azimuth_std),
        elevation=_apply_standardization(train_batch.pathway.elevation, elevation_mean, elevation_std),
        spike_count=train_batch.pathway.spike_count,
    )
    val_batch.pathway = PathwayBatch(
        distance=_apply_standardization(val_batch.pathway.distance, distance_mean, distance_std),
        azimuth=_apply_standardization(val_batch.pathway.azimuth, azimuth_mean, azimuth_std),
        elevation=_apply_standardization(val_batch.pathway.elevation, elevation_mean, elevation_std),
        spike_count=val_batch.pathway.spike_count,
    )
    test_batch.pathway = PathwayBatch(
        distance=_apply_standardization(test_batch.pathway.distance, distance_mean, distance_std),
        azimuth=_apply_standardization(test_batch.pathway.azimuth, azimuth_mean, azimuth_std),
        elevation=_apply_standardization(test_batch.pathway.elevation, elevation_mean, elevation_std),
        spike_count=test_batch.pathway.spike_count,
    )
    standardization_seconds = time.perf_counter() - standardization_start

    prepared = PreparedExperimentData(
        mode=dataset_mode,
        local_config=local_config,
        params=params,
        train_batch=train_batch,
        val_batch=val_batch,
        test_batch=test_batch,
        train_targets_raw=dataset_bundle.train_targets_raw,
        val_targets_raw=dataset_bundle.val_targets_raw,
        test_targets_raw=dataset_bundle.test_targets_raw,
    )
    profile = {
        "dataset_generation_seconds": dataset_seconds,
        "candidate_setup_seconds": candidate_seconds,
        "standardization_seconds": standardization_seconds,
        "split_timings": {
            "train": train_timing,
            "val": val_timing,
            "test": test_timing,
        },
        "front_end_seconds_total": float(
            train_timing["front_end_seconds"] + val_timing["front_end_seconds"] + test_timing["front_end_seconds"]
        ),
        "pathway_seconds_total": float(
            train_timing["pathway_seconds"] + val_timing["pathway_seconds"] + test_timing["pathway_seconds"]
        ),
        "concatenate_seconds_total": float(
            train_timing["concatenate_seconds"] + val_timing["concatenate_seconds"] + test_timing["concatenate_seconds"]
        ),
        "total_profiled_seconds": float(time.perf_counter() - prep_start),
        "chunk_size": chunk_size,
    }
    context.shared[prepared_key] = prepared
    context.shared[profile_key] = profile
    return prepared, profile


def _train_combined_all_model(
    model: nn.Module,
    data: PreparedExperimentData,
    target_bundle: Any,
    spec: Round2ExperimentSpec,
    training_config: EnhancedTrainingConfig,
    model_params: dict[str, Any],
    training_overrides: dict[str, Any],
    *,
    trial: optuna.Trial | None = None,
) -> tuple[EnhancedPathwayTrainingResult, nn.Module | None, dict[str, Any]]:
    task_weights = torch.tensor(
        [1.0, float(model_params["angle_weight"]), float(model_params["elevation_weight"])],
        device=data.train_targets_raw.device,
    )
    uncertainty_module = None
    if spec.loss_mode == "corrected_uncertainty":
        uncertainty_module = TaskUncertaintyWeights().to(data.train_targets_raw.device)
        sigma_init = torch.tensor(
            [
                1.0,
                float((1.0 / max(float(model_params["angle_weight"]), 1e-6)) ** 0.5),
                float((1.0 / max(float(model_params["elevation_weight"]), 1e-6)) ** 0.5),
            ],
            device=data.train_targets_raw.device,
        )
        with torch.no_grad():
            uncertainty_module.log_sigma.copy_(torch.log(sigma_init))

    learning_rate = float(model_params["learning_rate"]) * float(training_overrides.get("learning_rate_scale", 1.0))
    batch_size = int(training_overrides.get("batch_size", 16))
    spike_weight = float(model_params["loss_weighting"]) * float(training_overrides.get("spike_weight_scale", 1.0))
    uncertainty_warmup_epochs = int(training_overrides.get("uncertainty_warmup_epochs", 0))
    cartesian_mix_weight = float(training_overrides.get("cartesian_mix_weight", 0.5))

    parameters = list(model.parameters()) + ([] if uncertainty_module is None else list(uncertainty_module.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        threshold=training_config.scheduler_threshold,
        min_lr=training_config.scheduler_min_lr,
    )

    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    best_auxiliary_state = None if uncertainty_module is None else copy.deepcopy(uncertainty_module.state_dict())
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_combined_history: list[float] = []
    lr_history: list[float] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_combined = float("inf")
    best_diagnostics: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    stopped_early = False

    timing_profile = {
        "forward_seconds": 0.0,
        "loss_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "val_forward_seconds": 0.0,
        "val_loss_seconds": 0.0,
        "val_metrics_seconds": 0.0,
        "batch_count": 0,
        "epoch_seconds": [],
    }

    def batch_iterator() -> tuple[ExperimentBatch, torch.Tensor, torch.Tensor]:
        permutation = torch.randperm(target_bundle.train_model.shape[0], device=target_bundle.train_model.device)
        for start in range(0, target_bundle.train_model.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            yield data.train_batch.index_select(indices), target_bundle.train_model[indices], data.train_targets_raw[indices]

    for epoch in range(training_config.max_epochs):
        epoch_start = time.perf_counter()
        model.train()
        if uncertainty_module is not None:
            uncertainty_module.train()
            uncertainty_module.log_sigma.requires_grad_(epoch >= uncertainty_warmup_epochs)
        batch_losses: list[float] = []

        for batch_features, batch_targets_model, batch_targets_raw in batch_iterator():
            optimizer.zero_grad(set_to_none=True)

            forward_start = time.perf_counter()
            output_model, diagnostics = model(batch_features)
            timing_profile["forward_seconds"] += time.perf_counter() - forward_start

            loss_start = time.perf_counter()
            loss, _ = _round2_loss_components(
                output_model,
                batch_targets_model,
                batch_targets_raw,
                target_bundle,
                diagnostics,
                data.local_config,
                spec.loss_mode,
                task_weights,
                spike_weight,
                uncertainty_module,
                cartesian_mix_weight=cartesian_mix_weight,
            )
            timing_profile["loss_seconds"] += time.perf_counter() - loss_start

            backward_start = time.perf_counter()
            loss.backward()
            timing_profile["backward_seconds"] += time.perf_counter() - backward_start

            step_start = time.perf_counter()
            optimizer.step()
            timing_profile["optimizer_seconds"] += time.perf_counter() - step_start

            if uncertainty_module is not None:
                uncertainty_module.log_sigma.data.clamp_(-3.0, 2.0)
            batch_losses.append(float(loss.item()))
            timing_profile["batch_count"] += 1

        train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))

        model.eval()
        if uncertainty_module is not None:
            uncertainty_module.eval()
        with torch.no_grad():
            val_forward_start = time.perf_counter()
            val_output_model, val_diagnostics = model(data.val_batch)
            timing_profile["val_forward_seconds"] += time.perf_counter() - val_forward_start

            val_loss_start = time.perf_counter()
            val_loss, _ = _round2_loss_components(
                val_output_model,
                target_bundle.val_model,
                data.val_targets_raw,
                target_bundle,
                val_diagnostics,
                data.local_config,
                spec.loss_mode,
                task_weights,
                spike_weight,
                uncertainty_module,
                cartesian_mix_weight=cartesian_mix_weight,
            )
            timing_profile["val_loss_seconds"] += time.perf_counter() - val_loss_start

            val_metrics_start = time.perf_counter()
            val_eval = _augment_with_cartesian_metrics(
                _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
            )
            timing_profile["val_metrics_seconds"] += time.perf_counter() - val_metrics_start

        val_loss_value = float(val_loss.item())
        val_combined_value = float(val_eval.metrics["combined_error"])
        scheduler.step(val_combined_value)
        val_loss_history.append(val_loss_value)
        val_combined_history.append(val_combined_value)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))

        if val_combined_value < best_val_combined - training_config.early_stopping_min_delta:
            best_epoch = epoch
            best_val_loss = val_loss_value
            best_val_combined = val_combined_value
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            best_auxiliary_state = None if uncertainty_module is None else copy.deepcopy(uncertainty_module.state_dict())
            best_diagnostics = {key: value.detach().clone() for key, value in val_diagnostics.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start
        timing_profile["epoch_seconds"].append(epoch_seconds)

        if trial is not None:
            trial.report(val_combined_value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if epochs_without_improvement >= training_config.early_stopping_patience:
            stopped_early = True
            break

    timing_profile["total_training_loop_seconds"] = float(sum(timing_profile["epoch_seconds"]))
    timing_profile["mean_epoch_seconds"] = float(
        timing_profile["total_training_loop_seconds"] / max(1, len(timing_profile["epoch_seconds"]))
    )
    return (
        EnhancedPathwayTrainingResult(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            val_combined_error=val_combined_history,
            lr_history=lr_history,
            best_state=best_state,
            best_epoch=best_epoch,
            best_loss=best_val_loss,
            best_combined_error=best_val_combined,
            diagnostics=best_diagnostics,
            stopped_early=stopped_early,
            executed_epochs=len(train_loss_history),
            best_auxiliary_state=best_auxiliary_state,
        ),
        uncertainty_module,
        timing_profile,
    )


def _interface_summary(data: PreparedExperimentData) -> dict[str, Any]:
    transmit_shape = list(data.train_batch.transmit_spikes.shape) if data.train_batch.transmit_spikes is not None else None
    receive_shape = list(data.train_batch.receive_spikes.shape) if data.train_batch.receive_spikes is not None else None
    return {
        "boundary_stage": "cochlea_to_spikes -> build_pathway_features",
        "upstream_output": {
            "transmit_spikes_shape": transmit_shape,
            "receive_spikes_shape": receive_shape,
            "time_base_hz": float(data.local_config.envelope_rate_hz),
            "num_frequency_channels": int(data.local_config.num_cochlea_channels),
        },
        "downstream_input": {
            "distance_feature_shape": list(data.train_batch.pathway.distance.shape),
            "azimuth_feature_shape": list(data.train_batch.pathway.azimuth.shape),
            "elevation_feature_shape": list(data.train_batch.pathway.elevation.shape),
        },
        "swap_difficulty": "moderate",
        "swap_condition": (
            "The easiest replacement keeps the same spike tensor contract: transmit spikes shaped [batch, channel, time] "
            "and receive spikes shaped [batch, ear, channel, time] on the same envelope-rate time base."
        ),
        "what_breaks_if_changed": (
            "If an alternative cochlea changes channel count, time resolution, or stops emitting spikes, then "
            "build_pathway_features and the pre-pathway residual branch need an adapter or a rewrite."
        ),
    }


def _combined_all_baseline_trial_params(params: dict[str, Any], spec: Round2ExperimentSpec) -> dict[str, Any]:
    return {
        "hidden_dim": int(params["hidden_dim"]),
        "num_steps": int(params["num_steps"]),
        "membrane_beta": float(params["membrane_beta"]),
        "fusion_threshold": float(params["fusion_threshold"]),
        "learning_rate_scale": float(spec.training_overrides.get("learning_rate_scale", 1.0)),
        "batch_size": int(spec.training_overrides.get("batch_size", 16)),
        "cartesian_mix_weight": float(spec.training_overrides.get("cartesian_mix_weight", 0.5)),
        "spike_weight_scale": float(spec.training_overrides.get("spike_weight_scale", 1.0)),
    }


def _trial_configuration(
    trial: optuna.Trial,
    params: dict[str, Any],
    spec: Round2ExperimentSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tuned_params = dict(params)
    tuned_overrides = dict(spec.training_overrides)

    hidden_base = int(params["hidden_dim"])
    hidden_options = sorted({max(64, hidden_base - 16), hidden_base, hidden_base + 16, hidden_base + 32})
    tuned_params["hidden_dim"] = int(trial.suggest_categorical("hidden_dim", hidden_options))
    tuned_params["num_steps"] = int(trial.suggest_categorical("num_steps", [6, 8, 10, 12]))
    tuned_params["membrane_beta"] = float(
        trial.suggest_float("membrane_beta", max(0.88, float(params["membrane_beta"]) - 0.03), min(0.99, float(params["membrane_beta"]) + 0.03))
    )
    tuned_params["fusion_threshold"] = float(
        trial.suggest_float("fusion_threshold", 0.90 * float(params["fusion_threshold"]), 1.15 * float(params["fusion_threshold"]))
    )
    tuned_overrides["learning_rate_scale"] = float(trial.suggest_float("learning_rate_scale", 0.60, 1.20, log=True))
    tuned_overrides["batch_size"] = int(trial.suggest_categorical("batch_size", [8, 12, 16]))
    tuned_overrides["cartesian_mix_weight"] = float(trial.suggest_float("cartesian_mix_weight", 0.30, 0.75))
    tuned_overrides["spike_weight_scale"] = float(trial.suggest_float("spike_weight_scale", 0.75, 1.25))
    return tuned_params, tuned_overrides


def _run_combined_all_optuna(
    context: StageContext,
    data: PreparedExperimentData,
    prep_profile: dict[str, Any],
    target_bundle: Any,
    baseline_result: dict[str, Any],
    spec: Round2ExperimentSpec,
    training_config: EnhancedTrainingConfig,
    params: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    storage_path = output_root / "optuna_study.db"
    storage_uri = f"sqlite:///{storage_path}"
    study_name = "round2_combined_all_short_v1"
    target_trials = int(os.environ.get("RADAR_SNN_ROUND2_COMBINED_OPTUNA_TRIALS", "6"))
    sampler = optuna.samplers.TPESampler(seed=context.config.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    if len(study.trials) == 0:
        study.enqueue_trial(_combined_all_baseline_trial_params(params, spec))

    def objective(trial: optuna.Trial) -> float:
        tuned_params, tuned_overrides = _trial_configuration(trial, params, spec)
        seed_everything(context.config.seed + 200)
        model = _instantiate_combined_all_model(data, spec, tuned_params)
        train_result, uncertainty_module, training_breakdown = _train_combined_all_model(
            model,
            data,
            target_bundle,
            spec,
            training_config,
            tuned_params,
            tuned_overrides,
            trial=trial,
        )
        model.load_state_dict(train_result.best_state)
        if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
            uncertainty_module.load_state_dict(train_result.best_auxiliary_state)

        val_eval = _augment_with_cartesian_metrics(
            _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
        )
        test_eval = _augment_with_cartesian_metrics(
            _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
        )
        objective_value = float(val_eval.metrics["combined_error"])

        trial.set_user_attr("combined_error", float(test_eval.metrics["combined_error"]))
        trial.set_user_attr("distance_mae_m", float(test_eval.metrics["distance_mae_m"]))
        trial.set_user_attr("azimuth_mae_deg", float(test_eval.metrics["azimuth_mae_deg"]))
        trial.set_user_attr("elevation_mae_deg", float(test_eval.metrics["elevation_mae_deg"]))
        trial.set_user_attr("euclidean_error_m", float(test_eval.metrics["euclidean_error_m"]))
        trial.set_user_attr("val_combined_error", float(val_eval.metrics["combined_error"]))
        trial.set_user_attr("objective", objective_value)
        trial.set_user_attr("training_breakdown", training_breakdown)
        trial.set_user_attr(
            "training_config",
            {
                "max_epochs": training_config.max_epochs,
                "early_stopping_patience": training_config.early_stopping_patience,
                **tuned_overrides,
            },
        )
        trial.set_user_attr("prep_cache_reused", True)
        trial.set_user_attr("prep_seconds_reference", float(prep_profile["total_profiled_seconds"]))
        return objective_value

    remaining_trials = max(0, target_trials - len(study.trials))
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("Round 2 combined-all Optuna study produced no completed trials.")
    best_trial = min(completed_trials, key=lambda trial: float(trial.value))
    baseline_trial = None
    baseline_signature = _combined_all_baseline_trial_params(params, spec)
    for trial in completed_trials:
        if all(trial.params.get(key) == value for key, value in baseline_signature.items()):
            baseline_trial = trial
            break

    import matplotlib.pyplot as plt

    plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(output_root / "optuna_history.png", dpi=180, bbox_inches="tight")
    plt.close()
    try:
        plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(output_root / "optuna_importance.png", dpi=180, bbox_inches="tight")
        plt.close()
        importances = get_param_importances(study, evaluator=PedAnovaImportanceEvaluator())
    except Exception:
        importances = get_param_importances(study, evaluator=PedAnovaImportanceEvaluator())
        save_grouped_bar_chart(
            list(importances.keys()),
            {"importance": list(importances.values())},
            output_root / "optuna_importance.png",
            "Round 2 Combined-All Optuna Importance",
            ylabel="Importance",
        )

    save_grouped_bar_chart(
        ["Combined Error", "Euclidean Error"],
        {
            "Saved Combined-All": [
                float(baseline_result["test_metrics"]["combined_error"]),
                float(baseline_result["test_metrics"]["euclidean_error_m"]),
            ],
            "Optuna Best": [
                float(best_trial.user_attrs["combined_error"]),
                float(best_trial.user_attrs["euclidean_error_m"]),
            ],
        },
        output_root / "optuna_vs_baseline.png",
        "Combined-All Optuna Best vs Saved Combined-All",
        ylabel="Error",
    )

    dashboard_script = output_root / "run_optuna_dashboard.sh"
    dashboard_script.write_text(
        f"#!/bin/sh\noptuna-dashboard {storage_path}\n",
        encoding="utf-8",
    )

    trial_rows = []
    for trial in completed_trials:
        trial_rows.append(
            {
                "number": trial.number,
                "objective": float(trial.value),
                "params": trial.params,
                "combined_error": float(trial.user_attrs.get("combined_error", float("nan"))),
                "euclidean_error_m": float(trial.user_attrs.get("euclidean_error_m", float("nan"))),
            }
        )
    save_json(output_root / "optuna_trials.json", {"trials": trial_rows})

    summary = {
        "study_name": study_name,
        "storage_uri": storage_uri,
        "trial_count": len(study.trials),
        "completed_trials": len(completed_trials),
        "best_trial_number": best_trial.number,
        "best_params": best_trial.params,
        "best_objective": float(best_trial.value),
        "best_metrics": {
            "combined_error": float(best_trial.user_attrs["combined_error"]),
            "distance_mae_m": float(best_trial.user_attrs["distance_mae_m"]),
            "azimuth_mae_deg": float(best_trial.user_attrs["azimuth_mae_deg"]),
            "elevation_mae_deg": float(best_trial.user_attrs["elevation_mae_deg"]),
            "euclidean_error_m": float(best_trial.user_attrs["euclidean_error_m"]),
        },
        "improvement_vs_saved_combined_all": {
            "combined_error_delta": float(best_trial.user_attrs["combined_error"] - baseline_result["test_metrics"]["combined_error"]),
            "euclidean_error_delta": float(best_trial.user_attrs["euclidean_error_m"] - baseline_result["test_metrics"]["euclidean_error_m"]),
        },
        "parameter_importances": {key: float(value) for key, value in importances.items()},
        "artifacts": {
            "history_plot": str(output_root / "optuna_history.png"),
            "importance_plot": str(output_root / "optuna_importance.png"),
            "comparison_plot": str(output_root / "optuna_vs_baseline.png"),
            "dashboard_script": str(dashboard_script),
            "trials_json": str(output_root / "optuna_trials.json"),
        },
        "baseline_trial": None,
    }
    if baseline_trial is not None:
        summary["baseline_trial"] = {
            "number": baseline_trial.number,
            "objective": float(baseline_trial.value),
            "params": baseline_trial.params,
            "metrics": {
                "combined_error": float(baseline_trial.user_attrs["combined_error"]),
                "euclidean_error_m": float(baseline_trial.user_attrs["euclidean_error_m"]),
            },
            "training_breakdown": baseline_trial.user_attrs.get("training_breakdown"),
        }
    save_json(output_root / "optuna_summary.json", summary)
    return summary


def _load_or_run_base_result(
    context: StageContext,
    outputs: Any,
    baseline: dict[str, Any],
    best_round2: dict[str, Any],
    spec: Round2ExperimentSpec,
    training_config: EnhancedTrainingConfig,
    params: dict[str, Any],
    data: PreparedExperimentData,
    target_bundle: Any,
    output_root: Path,
) -> dict[str, Any]:
    result_path = output_root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    total_start = time.perf_counter()
    model = _instantiate_combined_all_model(data, spec, params)
    training_start = time.perf_counter()
    train_result, uncertainty_module, _ = _train_combined_all_model(
        model,
        data,
        target_bundle,
        spec,
        training_config,
        params,
        dict(spec.training_overrides),
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
        "baseline_label": baseline["result"].get("baseline_label", "combined_small"),
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
            "data_prep_seconds": format_float(0.0),
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
    save_json(result_path, result)
    return result


def _write_report(
    outputs_root: Path,
    baseline: dict[str, Any],
    best_round2: dict[str, Any],
    result: dict[str, Any],
    training_config: EnhancedTrainingConfig,
    prep_profile: dict[str, Any],
    interface_summary: dict[str, Any],
    optuna_summary: dict[str, Any],
) -> Path:
    total_seconds = float(result["timings"]["total_seconds"])
    saved_training_seconds = float(result["timings"]["training_seconds"])
    saved_prep_seconds = float(result["timings"]["data_prep_seconds"])
    baseline_optuna = optuna_summary.get("baseline_trial")
    baseline_training_breakdown = baseline_optuna.get("training_breakdown") if baseline_optuna is not None else None

    def pct(value: float, denom: float) -> str:
        return f"{100.0 * value / max(denom, 1e-6):.1f}%"

    split_lines = []
    for split_name, split_timing in prep_profile["split_timings"].items():
        split_total = (
            float(split_timing["front_end_seconds"])
            + float(split_timing["pathway_seconds"])
            + float(split_timing["concatenate_seconds"])
        )
        split_lines.extend(
            [
                f"- {split_name.title()} split: `{split_timing['samples']}` samples over `{split_timing['chunks']}` chunks, "
                f"`{split_total:.2f} s` total",
                f"  front end `{split_timing['front_end_seconds']:.2f} s`, pathway `{split_timing['pathway_seconds']:.2f} s`, "
                f"concat `{split_timing['concatenate_seconds']:.2f} s`",
            ]
        )

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
        "## Timing Breakdown",
        "",
        f"- Saved combined-all wall clock: `{total_seconds:.2f} s` total",
        f"- Data prep in saved run: `{saved_prep_seconds:.2f} s` ({pct(saved_prep_seconds, total_seconds)})",
        f"- Training in saved run: `{saved_training_seconds:.2f} s` ({pct(saved_training_seconds, total_seconds)})",
        f"- Evaluation in saved run: `{float(result['timings']['evaluation_seconds']):.2f} s` ({pct(float(result['timings']['evaluation_seconds']), total_seconds)})",
        "",
        "Profiled data-preparation breakdown:",
        f"- Scene synthesis: `{prep_profile['dataset_generation_seconds']:.2f} s` ({pct(float(prep_profile['dataset_generation_seconds']), prep_profile['total_profiled_seconds'])} of profiled prep)",
        f"- Candidate setup: `{prep_profile['candidate_setup_seconds']:.2f} s`",
        f"- Cochlea + spike conversion: `{prep_profile['front_end_seconds_total']:.2f} s` ({pct(float(prep_profile['front_end_seconds_total']), prep_profile['total_profiled_seconds'])} of profiled prep)",
        f"- Pathway feature construction: `{prep_profile['pathway_seconds_total']:.2f} s` ({pct(float(prep_profile['pathway_seconds_total']), prep_profile['total_profiled_seconds'])} of profiled prep)",
        f"- Tensor concatenation: `{prep_profile['concatenate_seconds_total']:.2f} s`",
        f"- Standardization: `{prep_profile['standardization_seconds']:.2f} s`",
        f"- Profiled prep total: `{prep_profile['total_profiled_seconds']:.2f} s` with chunk size `{prep_profile['chunk_size']}`",
        "",
        *split_lines,
        "",
        "Training-loop breakdown:",
    ]
    if baseline_training_breakdown is None:
        lines.append("- No baseline-config Optuna trial was available, so only the saved coarse training time is available.")
    else:
        loop_total = float(baseline_training_breakdown["total_training_loop_seconds"])
        mean_epoch = float(baseline_training_breakdown["mean_epoch_seconds"])
        batch_count = int(baseline_training_breakdown["batch_count"])
        lines.extend(
            [
                f"- Baseline-config Optuna trial loop total: `{loop_total:.2f} s` across `{len(baseline_training_breakdown['epoch_seconds'])}` epochs",
                f"- Mean epoch time: `{mean_epoch:.2f} s`",
                f"- Forward pass: `{baseline_training_breakdown['forward_seconds']:.2f} s`",
                f"- Loss assembly: `{baseline_training_breakdown['loss_seconds']:.2f} s`",
                f"- Backward pass: `{baseline_training_breakdown['backward_seconds']:.2f} s`",
                f"- Optimizer step: `{baseline_training_breakdown['optimizer_seconds']:.2f} s`",
                f"- Validation forward: `{baseline_training_breakdown['val_forward_seconds']:.2f} s`",
                f"- Validation loss: `{baseline_training_breakdown['val_loss_seconds']:.2f} s`",
                f"- Validation metrics pass: `{baseline_training_breakdown['val_metrics_seconds']:.2f} s`",
                f"- Total training batches processed: `{batch_count}`",
                "- Interpretation: the main barrier is not the final linear readout; it is repeated spike-domain model forward/backward through the added residual and resonance branches, plus a second full validation forward used for metric computation each epoch.",
            ]
        )

    lines.extend(
        [
            "",
            "## Cochlea / Spike Boundary",
            "",
            f"- Boundary: `{interface_summary['boundary_stage']}`",
            f"- Upstream spike contract: transmit `{interface_summary['upstream_output']['transmit_spikes_shape']}`, receive `{interface_summary['upstream_output']['receive_spikes_shape']}` at `{interface_summary['upstream_output']['time_base_hz']:.1f} Hz` envelope rate",
            f"- Downstream pathway tensors: distance `{interface_summary['downstream_input']['distance_feature_shape']}`, azimuth `{interface_summary['downstream_input']['azimuth_feature_shape']}`, elevation `{interface_summary['downstream_input']['elevation_feature_shape']}`",
            "- What the boundary means:",
            "  the cochlea side is responsible for taking waveforms to per-channel spike trains, while the rest of the system assumes those spike tensors already exist and builds delay, ITD/ILD, spectral, resonance, and fusion features from them.",
            "- How easy it is to swap:",
            f"  `{interface_summary['swap_difficulty']}`. {interface_summary['swap_condition']}",
            "- What must change if the alternative cochlea differs:",
            f"  {interface_summary['what_breaks_if_changed']}",
            "- Practical conclusion:",
            "  the clean replacement point is `_extract_front_end()` / `cochlea_to_spikes()`. If a new cochlea preserves the spike tensor shapes, the rest of the combined-all model can stay unchanged.",
            "",
            "## Optuna Tuning",
            "",
            f"- Study name: `{optuna_summary['study_name']}`",
            f"- Storage: `{optuna_summary['storage_uri']}`",
            f"- Trials requested/completed: `{optuna_summary['trial_count']}` total, `{optuna_summary['completed_trials']}` completed",
            "- Cache strategy: one short-data prepared bundle was built once, then every Optuna trial reused the same synthetic scenes, cochlea spikes, and pathway tensors. The search space only changed downstream model and training parameters, so no front-end rebuild was needed per trial.",
            "",
            "Best Optuna trial:",
            f"- Trial number: `{optuna_summary['best_trial_number']}`",
            f"- Validation objective: `{optuna_summary['best_objective']:.4f}`",
            f"- Test combined error: `{optuna_summary['best_metrics']['combined_error']:.4f}`",
            f"- Test Euclidean error: `{optuna_summary['best_metrics']['euclidean_error_m']:.4f} m`",
            f"- Distance / azimuth / elevation: `{optuna_summary['best_metrics']['distance_mae_m']:.4f} m`, `{optuna_summary['best_metrics']['azimuth_mae_deg']:.4f} deg`, `{optuna_summary['best_metrics']['elevation_mae_deg']:.4f} deg`",
            f"- Delta vs saved combined-all: combined `{optuna_summary['improvement_vs_saved_combined_all']['combined_error_delta']:.4f}`, Euclidean `{optuna_summary['improvement_vs_saved_combined_all']['euclidean_error_delta']:.4f} m`",
            "",
            "Best parameters:",
        ]
    )
    for key, value in optuna_summary["best_params"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "Parameter importance:",
        ]
    )
    for key, value in optuna_summary["parameter_importances"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")

    lines.extend(
        [
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
            "![Combined-all Optuna history](round_2_combined_all/optuna_history.png)",
            "![Combined-all Optuna importance](round_2_combined_all/optuna_importance.png)",
            "![Combined-all Optuna comparison](round_2_combined_all/optuna_vs_baseline.png)",
            "",
            "## Interpretation",
            "",
            "If this model improves on the best individual result, the round-2 changes are largely complementary. If it only beats the fixed baseline but not the best individual variant, then the additions help in isolation but partly compete when stacked. If it loses to both, the short-data improvements are not additive and the combined model is over-complex for this regime.",
            "",
            "The new timing analysis shows that the combined-all short-data regime is dominated by two areas: front-end spike construction during preparation, and repeated spike-domain forward/backward passes through the resonance plus residual branches during training. The cochlea boundary is explicit and reasonably clean, so an alternative cochlea is practical as long as it preserves the spike-tensor contract or provides an adapter before `build_pathway_features()`.",
        ]
    )

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

    spec = _combined_all_spec()
    output_root = outputs.root / "round_2_combined_all"
    output_root.mkdir(parents=True, exist_ok=True)

    data, prep_profile = _prepare_profiled_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    interface_summary = _interface_summary(data)

    result = _load_or_run_base_result(
        context,
        outputs,
        baseline,
        best_round2,
        spec,
        training_config,
        params,
        data,
        target_bundle,
        output_root,
    )
    result["baseline_label"] = baseline_label
    result["timings"]["data_prep_seconds"] = format_float(prep_profile["total_profiled_seconds"])
    result["timings"]["total_seconds"] = format_float(
        float(result["timings"]["training_seconds"])
        + float(result["timings"]["evaluation_seconds"])
        + float(prep_profile["total_profiled_seconds"])
    )
    save_json(output_root / "result.json", result)

    optuna_summary = _run_combined_all_optuna(
        context,
        data,
        prep_profile,
        target_bundle,
        result,
        spec,
        training_config,
        params,
        output_root,
    )
    report_path = _write_report(
        outputs.root,
        baseline,
        best_round2,
        result,
        training_config,
        prep_profile,
        interface_summary,
        optuna_summary,
    )
    summary = {
        "baseline": baseline["result"],
        "best_round2": best_round2,
        "result": result,
        "prep_profile": prep_profile,
        "interface_summary": interface_summary,
        "optuna": optuna_summary,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_2_combined_all_summary.json", summary)
    return summary
