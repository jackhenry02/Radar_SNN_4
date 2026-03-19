from __future__ import annotations

import copy
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.pathway_snn import PathwayBatch, PathwayFusionSNN
from stages.base import StageContext
from stages.experiments import _baseline_reference_params, _metrics_delta, _prepare_experiment_data, _save_baseline_outputs
from stages.improved_experiments import (
    _evaluate_improved_model,
    _improved_experiment_specs,
    _improved_loss_components,
    _instantiate_improved_model,
    _is_accepted,
    _prepare_target_bundle,
    _save_improved_outputs,
)
from stages.improvement import (
    SplitPrediction,
    _apply_standardization,
    _build_pathway_batch_from_acoustic,
    _copy_config,
    _distance_candidates,
    _fit_standardization,
    _itd_candidates,
    _predict_split,
    _prepare_dataset_bundle,
    _split_metrics,
    _standardize_pathway_triplet,
)
from utils.common import format_float, save_grouped_bar_chart, save_json, save_text_figure, seed_everything


@dataclass
class EnhancedTrainingConfig:
    dataset_mode: str = "training_improved"
    max_epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    scheduler_threshold: float = 1e-4
    scheduler_min_lr: float = 1e-5


@dataclass
class EnhancedPathwayTrainingResult:
    train_loss: list[float]
    val_loss: list[float]
    val_combined_error: list[float]
    lr_history: list[float]
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_loss: float
    best_combined_error: float
    diagnostics: dict[str, torch.Tensor]
    stopped_early: bool
    executed_epochs: int
    best_auxiliary_state: dict[str, Any] | None = None


def _batch_iterator(
    pathway_batch: PathwayBatch,
    targets: torch.Tensor,
    batch_size: int,
) -> tuple[PathwayBatch, torch.Tensor]:
    permutation = torch.randperm(targets.shape[0], device=targets.device)
    for start in range(0, targets.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield pathway_batch.index_select(indices), targets[indices]


def _train_pathway_snn_with_training_improvements(
    model: PathwayFusionSNN,
    train_batch: PathwayBatch,
    train_targets: torch.Tensor,
    val_batch: PathwayBatch,
    val_targets: torch.Tensor,
    val_targets_raw: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_weights: torch.Tensor,
    local_config: Any,
    *,
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
    spike_weight: float,
    scheduler_patience: int,
    scheduler_factor: float,
    scheduler_threshold: float,
    scheduler_min_lr: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> EnhancedPathwayTrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_min_lr,
    )
    criterion = nn.SmoothL1Loss(reduction="none")

    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    best_epoch = 0
    best_loss = float("inf")
    best_combined_error = float("inf")
    best_diagnostics: dict[str, torch.Tensor] = {}
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_combined_history: list[float] = []
    lr_history: list[float] = []
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch_features, batch_targets in _batch_iterator(train_batch, train_targets, batch_size):
            optimizer.zero_grad(set_to_none=True)
            output, diagnostics = model(batch_features)
            localisation_loss = (criterion(output, batch_targets) * target_weights.view(1, -1)).mean()
            spike_penalty = diagnostics["spike_rate"].mean()
            loss = localisation_loss + spike_weight * spike_penalty
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss_history.append(float(sum(epoch_losses) / max(1, len(epoch_losses))))

        model.eval()
        with torch.no_grad():
            val_output, val_diagnostics = model(val_batch)
            val_localisation_loss = (criterion(val_output, val_targets) * target_weights.view(1, -1)).mean()
            val_loss = val_localisation_loss + spike_weight * val_diagnostics["spike_rate"].mean()
            denormalized = val_output * target_std + target_mean
            val_prediction = SplitPrediction(
                predicted_distance=denormalized[:, 0],
                predicted_azimuth=denormalized[:, 1] * 45.0,
                predicted_elevation=denormalized[:, 2] * 30.0,
                diagnostics=val_diagnostics,
            )
            val_metrics = _split_metrics(local_config, val_prediction, val_targets_raw)

        val_loss_value = float(val_loss.item())
        val_combined_value = float(val_metrics["combined_error"])
        scheduler.step(val_combined_value)

        val_loss_history.append(val_loss_value)
        val_combined_history.append(val_combined_value)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))

        if val_combined_value < best_combined_error - early_stopping_min_delta:
            best_epoch = epoch
            best_loss = val_loss_value
            best_combined_error = val_combined_value
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            best_diagnostics = {key: value.detach().clone() for key, value in val_diagnostics.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    return EnhancedPathwayTrainingResult(
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        val_combined_error=val_combined_history,
        lr_history=lr_history,
        best_state=best_state,
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_combined_error=best_combined_error,
        diagnostics=best_diagnostics,
        stopped_early=stopped_early,
        executed_epochs=len(train_loss_history),
    )


def _train_improved_model_with_training_improvements(
    model: nn.Module,
    data: Any,
    target_bundle: Any,
    spec: Any,
    training_config: EnhancedTrainingConfig,
) -> tuple[EnhancedPathwayTrainingResult, nn.Module | None]:
    params = data.params
    task_weights = torch.tensor(
        [1.0, float(params["angle_weight"]), float(params["elevation_weight"])],
        device=data.train_targets_raw.device,
    )
    uncertainty_module = None
    if spec.loss_mode == "corrected_uncertainty":
        from stages.experiments import TaskUncertaintyWeights

        uncertainty_module = TaskUncertaintyWeights().to(data.train_targets_raw.device)
        sigma_init = torch.tensor(
            [
                1.0,
                float((1.0 / max(float(params["angle_weight"]), 1e-6)) ** 0.5),
                float((1.0 / max(float(params["elevation_weight"]), 1e-6)) ** 0.5),
            ],
            device=data.train_targets_raw.device,
        )
        with torch.no_grad():
            uncertainty_module.log_sigma.copy_(torch.log(sigma_init))

    learning_rate = float(params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0))
    batch_size = int(spec.training_overrides.get("batch_size", int(params["batch_size"])))
    spike_weight = float(params["loss_weighting"]) * float(spec.training_overrides.get("spike_weight_scale", 1.0))
    uncertainty_warmup_epochs = int(spec.training_overrides.get("uncertainty_warmup_epochs", 0))

    trainables = list(model.parameters()) + ([] if uncertainty_module is None else list(uncertainty_module.parameters()))
    optimizer = torch.optim.Adam(trainables, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        threshold=training_config.scheduler_threshold,
        min_lr=training_config.scheduler_min_lr,
    )

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_combined_history: list[float] = []
    lr_history: list[float] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_combined = float("inf")
    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    best_auxiliary_state = None if uncertainty_module is None else copy.deepcopy(uncertainty_module.state_dict())
    final_diagnostics: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    stopped_early = False

    def batch_iterator_with_raw():
        permutation = torch.randperm(target_bundle.train_model.shape[0], device=target_bundle.train_model.device)
        for start in range(0, target_bundle.train_model.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            yield (
                data.train_batch.index_select(indices),
                target_bundle.train_model[indices],
                data.train_targets_raw[indices],
            )

    for epoch in range(training_config.max_epochs):
        model.train()
        if uncertainty_module is not None:
            uncertainty_module.train()
            requires_grad = epoch >= uncertainty_warmup_epochs
            uncertainty_module.log_sigma.requires_grad_(requires_grad)

        batch_losses: list[float] = []
        for batch_features, batch_targets_model, batch_targets_raw in batch_iterator_with_raw():
            optimizer.zero_grad(set_to_none=True)
            output_model, diagnostics = model(batch_features)
            loss, _ = _improved_loss_components(
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
            )
            loss.backward()
            optimizer.step()
            if uncertainty_module is not None:
                uncertainty_module.log_sigma.data.clamp_(-3.0, 2.0)
            batch_losses.append(float(loss.item()))

        train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))

        model.eval()
        if uncertainty_module is not None:
            uncertainty_module.eval()
        with torch.no_grad():
            val_output_model, val_diagnostics = model(data.val_batch)
            val_loss, _ = _improved_loss_components(
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
            )
            val_eval = _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)

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
            final_diagnostics = {key: value.detach().clone() for key, value in val_eval.diagnostics.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.early_stopping_patience:
            stopped_early = True
            break

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
            diagnostics=final_diagnostics,
            stopped_early=stopped_early,
            executed_epochs=len(train_loss_history),
            best_auxiliary_state=best_auxiliary_state,
        ),
        uncertainty_module,
    )


def _load_previous_improved_results(outputs_root: Path) -> dict[str, dict[str, Any]]:
    previous_path = outputs_root / "improved_experiments" / "results.json"
    if not previous_path.exists():
        return {}
    import json

    with previous_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(item["name"]): item for item in payload.get("experiments", [])}


def _device_run_dir(outputs_root: Path, device_name: str) -> Path:
    run_dir = outputs_root / "training_improved_experiments" / device_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_device_baseline(
    context: StageContext,
    params: dict[str, Any],
    baseline_label: str,
    training_config: EnhancedTrainingConfig,
    *,
    device_name: str,
) -> dict[str, Any]:
    run_dir = _device_run_dir(context.outputs.root, device_name)
    timings: dict[str, float] = {}
    seed_everything(context.config.seed)
    print(
        f"[training_improved] running baseline on {device_name} with dataset_mode={training_config.dataset_mode}",
        flush=True,
    )

    total_start = time.perf_counter()
    dataset_start = time.perf_counter()
    dataset_bundle = _prepare_dataset_bundle(context, training_config.dataset_mode)
    timings["dataset_prep_seconds"] = time.perf_counter() - dataset_start

    local_config = _copy_config(
        context.config,
        num_cochlea_channels=int(params["num_frequency_channels"]),
        spike_threshold=float(params["spike_threshold"]),
        filter_bandwidth_sigma=float(params["filter_bandwidth_sigma"]),
    )

    distance_candidates = _distance_candidates(local_config, context.device, int(params["num_delay_lines"]))
    itd_candidates = _itd_candidates(local_config, context.device, int(params["num_delay_lines"]))
    chunk_size = int(os.environ.get("RADAR_SNN_FEATURE_CHUNK_SIZE", "64"))
    if chunk_size <= 0:
        chunk_size = 64

    feature_start = time.perf_counter()
    train_pathways, _, _ = _build_pathway_batch_from_acoustic(
        dataset_bundle.train_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
        include_artifacts=False,
    )
    val_pathways, _, _ = _build_pathway_batch_from_acoustic(
        dataset_bundle.val_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
        include_artifacts=False,
    )
    test_pathways, _, _ = _build_pathway_batch_from_acoustic(
        dataset_bundle.test_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
        include_artifacts=False,
    )
    train_pathways, val_pathways, test_pathways, pathway_stats = _standardize_pathway_triplet(
        train_pathways,
        val_pathways,
        test_pathways,
    )
    timings["feature_prep_seconds"] = time.perf_counter() - feature_start

    train_targets = torch.stack(
        [
            dataset_bundle.train_targets_raw[:, 0],
            dataset_bundle.train_targets_raw[:, 1] / 45.0,
            dataset_bundle.train_targets_raw[:, 2] / 30.0,
        ],
        dim=-1,
    )
    val_targets = torch.stack(
        [
            dataset_bundle.val_targets_raw[:, 0],
            dataset_bundle.val_targets_raw[:, 1] / 45.0,
            dataset_bundle.val_targets_raw[:, 2] / 30.0,
        ],
        dim=-1,
    )
    target_mean, target_std = _fit_standardization(train_targets)
    train_targets = _apply_standardization(train_targets, target_mean, target_std)
    val_targets = _apply_standardization(val_targets, target_mean, target_std)

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

    training_start = time.perf_counter()
    print(f"[training_improved] training start on {device_name}", flush=True)
    training = _train_pathway_snn_with_training_improvements(
        model,
        train_pathways,
        train_targets,
        val_pathways,
        val_targets,
        dataset_bundle.val_targets_raw,
        target_mean,
        target_std,
        target_weights,
        local_config,
        max_epochs=training_config.max_epochs,
        learning_rate=float(params["learning_rate"]),
        batch_size=int(params["batch_size"]),
        spike_weight=float(params["loss_weighting"]),
        scheduler_patience=training_config.scheduler_patience,
        scheduler_factor=training_config.scheduler_factor,
        scheduler_threshold=training_config.scheduler_threshold,
        scheduler_min_lr=training_config.scheduler_min_lr,
        early_stopping_patience=training_config.early_stopping_patience,
        early_stopping_min_delta=training_config.early_stopping_min_delta,
    )
    timings["training_seconds"] = time.perf_counter() - training_start
    print(
        f"[training_improved] training finished on {device_name} after {training.executed_epochs} epochs",
        flush=True,
    )

    model.load_state_dict(training.best_state)
    model.eval()

    evaluation_start = time.perf_counter()
    val_prediction = _predict_split(model, val_pathways, target_mean, target_std)
    test_prediction = _predict_split(model, test_pathways, target_mean, target_std)
    val_metrics = _split_metrics(local_config, val_prediction, dataset_bundle.val_targets_raw)
    test_metrics = _split_metrics(local_config, test_prediction, dataset_bundle.test_targets_raw)
    timings["evaluation_seconds"] = time.perf_counter() - evaluation_start
    timings["total_seconds"] = time.perf_counter() - total_start

    evaluation = {
        "dataset_mode": training_config.dataset_mode,
        "dataset_counts": dataset_bundle.counts,
        "training": training,
        "combined_error": val_metrics["combined_error"],
        "distance_mae_m": val_metrics["distance_mae_m"],
        "azimuth_mae_deg": val_metrics["azimuth_mae_deg"],
        "elevation_mae_deg": val_metrics["elevation_mae_deg"],
        "mean_spike_rate": val_metrics["mean_spike_rate"],
        "diagnostics": val_metrics["diagnostics"],
        "predicted_distance": val_metrics["predicted_distance"],
        "predicted_azimuth": val_metrics["predicted_azimuth"],
        "predicted_elevation": val_metrics["predicted_elevation"],
        "target_distance": val_metrics["target_distance"],
        "target_azimuth": val_metrics["target_azimuth"],
        "target_elevation": val_metrics["target_elevation"],
        "test_predicted_distance": test_metrics["predicted_distance"],
        "test_predicted_azimuth": test_metrics["predicted_azimuth"],
        "test_predicted_elevation": test_metrics["predicted_elevation"],
        "test_target_distance": test_metrics["target_distance"],
        "test_target_azimuth": test_metrics["target_azimuth"],
        "test_target_elevation": test_metrics["target_elevation"],
        "test_distance_mae_m": test_metrics["distance_mae_m"],
        "test_azimuth_mae_deg": test_metrics["azimuth_mae_deg"],
        "test_elevation_mae_deg": test_metrics["elevation_mae_deg"],
        "test_combined_error": test_metrics["combined_error"],
        "test_mean_spike_rate": test_metrics["mean_spike_rate"],
        "test_diagnostics": test_metrics["diagnostics"],
    }
    artifacts = _save_baseline_outputs(run_dir, evaluation)

    save_text_figure(
        [
            f"baseline_label: {baseline_label}",
            f"device: {device_name}",
            f"dataset_mode: {training_config.dataset_mode}",
            f"dataset_counts: {dataset_bundle.counts}",
            f"max_epochs: {training_config.max_epochs}",
            f"executed_epochs: {training.executed_epochs}",
            f"best_epoch: {training.best_epoch + 1}",
            f"stopped_early: {training.stopped_early}",
            f"best_val_combined_error: {training.best_combined_error:.4f}",
            f"best_val_loss: {training.best_loss:.4f}",
            f"final_learning_rate: {training.lr_history[-1]:.6f}",
            f"total_seconds: {timings['total_seconds']:.2f}",
            f"training_seconds: {timings['training_seconds']:.2f}",
        ],
        run_dir / "run_summary.png",
        f"{device_name.upper()} Improved Training Summary",
    )

    result = {
        "status": "success",
        "device": device_name,
        "baseline_label": baseline_label,
        "dataset_mode": training_config.dataset_mode,
        "dataset_counts": dataset_bundle.counts,
        "training_config": {
            "max_epochs": training_config.max_epochs,
            "early_stopping_patience": training_config.early_stopping_patience,
            "early_stopping_min_delta": training_config.early_stopping_min_delta,
            "scheduler_patience": training_config.scheduler_patience,
            "scheduler_factor": training_config.scheduler_factor,
            "scheduler_threshold": training_config.scheduler_threshold,
            "scheduler_min_lr": training_config.scheduler_min_lr,
            "learning_rate": float(params["learning_rate"]),
            "batch_size": int(params["batch_size"]),
        },
        "timings": {key: format_float(value) for key, value in timings.items()},
        "artifacts": {**artifacts, "run_summary": str(run_dir / "run_summary.png")},
        "training": {
            "executed_epochs": training.executed_epochs,
            "best_epoch": training.best_epoch + 1,
            "stopped_early": training.stopped_early,
            "best_val_loss": format_float(training.best_loss),
            "best_val_combined_error": format_float(training.best_combined_error),
            "initial_learning_rate": format_float(float(params["learning_rate"]), digits=6),
            "final_learning_rate": format_float(training.lr_history[-1], digits=6),
        },
        "val_metrics": {
            "distance_mae_m": format_float(val_metrics["distance_mae_m"]),
            "azimuth_mae_deg": format_float(val_metrics["azimuth_mae_deg"]),
            "elevation_mae_deg": format_float(val_metrics["elevation_mae_deg"]),
            "combined_error": format_float(val_metrics["combined_error"]),
            "mean_spike_rate": format_float(val_metrics["mean_spike_rate"]),
        },
        "test_metrics": {
            "distance_mae_m": format_float(test_metrics["distance_mae_m"]),
            "azimuth_mae_deg": format_float(test_metrics["azimuth_mae_deg"]),
            "elevation_mae_deg": format_float(test_metrics["elevation_mae_deg"]),
            "combined_error": format_float(test_metrics["combined_error"]),
            "mean_spike_rate": format_float(test_metrics["mean_spike_rate"]),
        },
    }
    save_json(run_dir / "result.json", result)
    return result


def _capture_failed_device_run(
    outputs_root: Path,
    device_name: str,
    baseline_label: str,
    training_config: EnhancedTrainingConfig,
    error: str,
) -> dict[str, Any]:
    run_dir = _device_run_dir(outputs_root, device_name)
    save_text_figure(
        [
            f"baseline_label: {baseline_label}",
            f"device: {device_name}",
            f"dataset_mode: {training_config.dataset_mode}",
            "status: failed",
            "",
            error,
        ],
        run_dir / "failure.png",
        f"{device_name.upper()} Improved Training Failure",
    )
    result = {
        "status": "failed",
        "device": device_name,
        "baseline_label": baseline_label,
        "dataset_mode": training_config.dataset_mode,
        "error": error,
        "artifacts": {"failure": str(run_dir / "failure.png")},
    }
    save_json(run_dir / "result.json", result)
    return result


def _overall_artifacts(outputs_root: Path, cpu_result: dict[str, Any], mps_result: dict[str, Any]) -> dict[str, str]:
    stage_root = outputs_root / "training_improved_experiments"
    stage_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    runtime_series: dict[str, list[float]] = {}
    metric_series: dict[str, list[float]] = {}
    for result in (cpu_result, mps_result):
        if result.get("status") != "success":
            continue
        runtime_series[result["device"].upper()] = [
            float(result["timings"]["dataset_prep_seconds"]),
            float(result["timings"]["feature_prep_seconds"]),
            float(result["timings"]["training_seconds"]),
            float(result["timings"]["evaluation_seconds"]),
            float(result["timings"]["total_seconds"]),
        ]
        metric_series[result["device"].upper()] = [
            float(result["test_metrics"]["distance_mae_m"]),
            float(result["test_metrics"]["azimuth_mae_deg"]),
            float(result["test_metrics"]["elevation_mae_deg"]),
            float(result["test_metrics"]["combined_error"]),
        ]

    if runtime_series:
        save_grouped_bar_chart(
            ["Dataset", "Features", "Training", "Evaluation", "Total"],
            runtime_series,
            stage_root / "runtime_comparison.png",
            "Improved Training Runtime Comparison",
            ylabel="Seconds",
        )
        artifacts["runtime_comparison"] = str(stage_root / "runtime_comparison.png")

    if metric_series:
        save_grouped_bar_chart(
            ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
            metric_series,
            stage_root / "metric_comparison.png",
            "Improved Training Metric Comparison",
            ylabel="Error",
        )
        artifacts["metric_comparison"] = str(stage_root / "metric_comparison.png")

    return artifacts


def _overall_experiment_artifacts(
    outputs_root: Path,
    baseline_metrics: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, str]:
    stage_root = outputs_root / "training_improved_experiments" / "cpu"
    stage_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    if results:
        save_grouped_bar_chart(
            ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
            {
                "Baseline": [
                    float(baseline_metrics["distance_mae_m"]),
                    float(baseline_metrics["azimuth_mae_deg"]),
                    float(baseline_metrics["elevation_mae_deg"]),
                    float(baseline_metrics["combined_error"]),
                ],
                **{
                    item["title"].replace("Improved Experiment ", "Exp "): [
                        float(item["test_metrics"]["distance_mae_m"]),
                        float(item["test_metrics"]["azimuth_mae_deg"]),
                        float(item["test_metrics"]["elevation_mae_deg"]),
                        float(item["test_metrics"]["combined_error"]),
                    ]
                    for item in results
                },
            },
            stage_root / "overall_experiment_comparison.png",
            "Training-Improved Experiments vs CPU Baseline",
            ylabel="Error",
        )
        artifacts["overall_experiment_comparison"] = str(stage_root / "overall_experiment_comparison.png")

        save_grouped_bar_chart(
            [item["title"].replace("Improved Experiment ", "Exp ") for item in results],
            {
                "Training Seconds": [float(item["training"]["training_seconds"]) for item in results],
                "Executed Epochs": [float(item["training"]["executed_epochs"]) for item in results],
            },
            stage_root / "experiment_training_runtime.png",
            "Training-Improved Experiment Runtime",
            ylabel="Value",
        )
        artifacts["experiment_training_runtime"] = str(stage_root / "experiment_training_runtime.png")

    return artifacts


def _run_cpu_training_improved_experiments(
    config: Any,
    outputs: Any,
    params: dict[str, Any],
    baseline_label: str,
    cpu_result: dict[str, Any],
    training_config: EnhancedTrainingConfig,
) -> dict[str, Any]:
    context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    print(
        f"[training_improved] preparing improved experiments on cpu with dataset_mode={training_config.dataset_mode}",
        flush=True,
    )
    data = _prepare_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    stage_root = outputs.root / "training_improved_experiments" / "cpu"
    stage_root.mkdir(parents=True, exist_ok=True)

    baseline_metrics = {
        "distance_mae_m": float(cpu_result["test_metrics"]["distance_mae_m"]),
        "azimuth_mae_deg": float(cpu_result["test_metrics"]["azimuth_mae_deg"]),
        "elevation_mae_deg": float(cpu_result["test_metrics"]["elevation_mae_deg"]),
        "combined_error": float(cpu_result["test_metrics"]["combined_error"]),
        "mean_spike_rate": float(cpu_result["test_metrics"]["mean_spike_rate"]),
    }
    previous_results = _load_previous_improved_results(outputs.root)
    experiment_summaries: list[dict[str, Any]] = []

    for index, spec in enumerate(_improved_experiment_specs(), start=1):
        print(f"[training_improved] running {spec.name} on cpu ({index}/5)", flush=True)
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
        if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
            uncertainty_module.load_state_dict(train_result.best_auxiliary_state)

        val_eval = _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
        test_eval = _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
        accepted = _is_accepted(test_eval.metrics, baseline_metrics)
        artifacts = _save_improved_outputs(stage_root, spec, train_result, test_eval, baseline_metrics, model)
        previous_item = previous_results.get(spec.name, {})
        previous_test_metrics = previous_item.get("test_metrics", {})

        experiment_summaries.append(
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "rationale": spec.rationale,
                "implemented_steps": spec.implemented_steps,
                "remaining_steps": spec.remaining_steps,
                "accepted": accepted,
                "decision": "ACCEPTED" if accepted else "REJECTED",
                "variant": spec.variant,
                "loss_mode": spec.loss_mode,
                "training_config": {
                    "learning_rate": float(params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0)),
                    "batch_size": int(spec.training_overrides.get("batch_size", int(params["batch_size"]))),
                    "max_epochs": training_config.max_epochs,
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
                    "training_seconds": format_float(training_seconds),
                },
                "val_metrics": val_eval.metrics,
                "test_metrics": test_eval.metrics,
                "comparison": _metrics_delta(test_eval.metrics, baseline_metrics),
                "previous_combined_error": previous_test_metrics.get("combined_error"),
                "artifacts": artifacts,
            }
        )

    experiment_artifacts = _overall_experiment_artifacts(outputs.root, baseline_metrics, experiment_summaries)
    payload = {
        "baseline_label": baseline_label,
        "dataset_mode": training_config.dataset_mode,
        "experiments": [
            {
                **{key: value for key, value in item.items() if key not in {"val_metrics", "test_metrics", "comparison"}},
                "val_metrics": {key: format_float(value) for key, value in item["val_metrics"].items()},
                "test_metrics": {key: format_float(value) for key, value in item["test_metrics"].items()},
                "comparison": {key: format_float(value) for key, value in item["comparison"].items()},
            }
            for item in experiment_summaries
        ],
        "artifacts": experiment_artifacts,
    }
    save_json(outputs.root / "training_improved_experiments" / "experiments_results.json", payload)
    return payload


def _write_report(
    outputs_root: Path,
    baseline_label: str,
    training_config: EnhancedTrainingConfig,
    cpu_result: dict[str, Any],
    mps_result: dict[str, Any],
    overall_artifacts: dict[str, str],
    experiment_payload: dict[str, Any] | None = None,
) -> Path:
    report_path = outputs_root / "Training_improved_experiments_report.md"
    lines = [
        "# Training Improved Experiments Report",
        "",
        "## Scope",
        "",
        "This report covers the baseline-only training-improvement pass requested before rerunning any non-baseline experiments.",
        f"- Baseline model: `{baseline_label}`",
        f"- Dataset mode: `{training_config.dataset_mode}`",
        "- Dataset split: `3500 / 750 / 750` synthetic scenes (`70% / 15% / 15%` of 5000 total)",
        f"- Max epochs: `{training_config.max_epochs}`",
        f"- Early stopping patience: `{training_config.early_stopping_patience}` epochs",
        "- Scheduler: `ReduceLROnPlateau` on validation combined error",
        f"- Scheduler patience/factor: `{training_config.scheduler_patience}` / `{training_config.scheduler_factor}`",
        "- Backend threads: `1`",
        "",
        "The CPU baseline remains the fixed reference for every improved experiment in this report.",
        "",
        "## CPU Baseline",
        "",
    ]

    if cpu_result.get("status") == "success":
        lines.extend(
            [
                f"- Status: `SUCCESS`",
                f"- Executed epochs: `{cpu_result['training']['executed_epochs']}`",
                f"- Early stopped: `{cpu_result['training']['stopped_early']}`",
                f"- Best epoch: `{cpu_result['training']['best_epoch']}`",
                f"- Test combined error: `{cpu_result['test_metrics']['combined_error']:.4f}`",
                f"- Test distance MAE: `{cpu_result['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Test azimuth MAE: `{cpu_result['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Test elevation MAE: `{cpu_result['test_metrics']['elevation_mae_deg']:.4f} deg`",
                f"- Total runtime: `{cpu_result['timings']['total_seconds']:.2f} s`",
                f"- Training runtime: `{cpu_result['timings']['training_seconds']:.2f} s`",
                "",
                "![CPU loss](training_improved_experiments/cpu/baseline/loss.png)",
                "![CPU summary](training_improved_experiments/cpu/run_summary.png)",
                "",
            ]
        )
    else:
        lines.extend([f"- Status: `FAILED`", "", "```text", cpu_result["error"], "```", ""])

    lines.extend(["## MPS Baseline", ""])
    if mps_result.get("status") == "success":
        lines.extend(
            [
                f"- Status: `SUCCESS`",
                f"- Executed epochs: `{mps_result['training']['executed_epochs']}`",
                f"- Early stopped: `{mps_result['training']['stopped_early']}`",
                f"- Best epoch: `{mps_result['training']['best_epoch']}`",
                f"- Test combined error: `{mps_result['test_metrics']['combined_error']:.4f}`",
                f"- Test distance MAE: `{mps_result['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Test azimuth MAE: `{mps_result['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Test elevation MAE: `{mps_result['test_metrics']['elevation_mae_deg']:.4f} deg`",
                f"- Total runtime: `{mps_result['timings']['total_seconds']:.2f} s`",
                f"- Training runtime: `{mps_result['timings']['training_seconds']:.2f} s`",
                "",
                "![MPS loss](training_improved_experiments/mps/baseline/loss.png)",
                "![MPS summary](training_improved_experiments/mps/run_summary.png)",
                "",
            ]
        )
    elif mps_result.get("status") == "unavailable":
        lines.extend(
            [
                "- Status: `UNAVAILABLE`",
                "- `torch.backends.mps.is_available()` was false in this environment, so no MPS baseline run was possible.",
                "",
            ]
        )
    else:
        lines.extend([f"- Status: `FAILED`", "", "```text", mps_result["error"], "```", ""])

    lines.extend(["## Comparison", ""])
    if "runtime_comparison" in overall_artifacts:
        lines.append("![Runtime comparison](training_improved_experiments/runtime_comparison.png)")
        lines.append("")
    if "metric_comparison" in overall_artifacts:
        lines.append("![Metric comparison](training_improved_experiments/metric_comparison.png)")
        lines.append("")

    if cpu_result.get("status") == "success" and mps_result.get("status") == "success":
        cpu_total = float(cpu_result["timings"]["total_seconds"])
        mps_total = float(mps_result["timings"]["total_seconds"])
        speed_ratio = cpu_total / max(mps_total, 1e-6)
        lines.extend(
            [
                f"- CPU total runtime: `{cpu_total:.2f} s`",
                f"- MPS total runtime: `{mps_total:.2f} s`",
                f"- Relative speedup (CPU / MPS): `{speed_ratio:.2f}x`",
                f"- CPU test combined error: `{cpu_result['test_metrics']['combined_error']:.4f}`",
                f"- MPS test combined error: `{mps_result['test_metrics']['combined_error']:.4f}`",
                "",
            ]
        )
    elif mps_result.get("status") == "failed":
        lines.extend(
            [
                "- The MPS pass did not complete, so all non-baseline experiments in this report remain CPU-only comparisons against the CPU baseline.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- The MPS pass was not available in this environment, so this report can only establish the new CPU baseline.",
                "",
            ]
        )

    if experiment_payload and experiment_payload.get("experiments"):
        lines.extend(
            [
                "## CPU Improved Experiments",
                "",
                "Each experiment below used the same `3500 / 750 / 750` split, the same `50`-epoch maximum, early stopping, and `ReduceLROnPlateau`, and was judged only against the fixed CPU baseline above.",
                "",
                "| Experiment | Combined Error | Distance MAE | Azimuth MAE | Elevation MAE | Epochs | Accepted |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in experiment_payload["experiments"]:
            lines.append(
                f"| {item['title']} | {item['test_metrics']['combined_error']:.4f} | "
                f"{item['test_metrics']['distance_mae_m']:.4f} | "
                f"{item['test_metrics']['azimuth_mae_deg']:.4f} | "
                f"{item['test_metrics']['elevation_mae_deg']:.4f} | "
                f"{item['training']['executed_epochs']} | "
                f"{'Yes' if item['accepted'] else 'No'} |"
            )

        lines.append("")
        if experiment_payload.get("artifacts", {}).get("overall_experiment_comparison"):
            lines.append("![CPU experiment comparison](training_improved_experiments/cpu/overall_experiment_comparison.png)")
            lines.append("")
        if experiment_payload.get("artifacts", {}).get("experiment_training_runtime"):
            lines.append("![CPU experiment runtime](training_improved_experiments/cpu/experiment_training_runtime.png)")
            lines.append("")

        lines.extend(["## Experiment Details", ""])
        for item in experiment_payload["experiments"]:
            previous_combined = item.get("previous_combined_error")
            lines.extend(
                [
                    f"### {item['title']}",
                    "",
                    f"- Decision: `{item['decision']}`",
                    f"- Change: {item['description']}",
                    f"- Rationale: {item['rationale']}",
                    f"- Executed epochs: `{item['training']['executed_epochs']}`",
                    f"- Best epoch: `{item['training']['best_epoch']}`",
                    f"- Early stopped: `{item['training']['stopped_early']}`",
                    f"- Training time: `{item['training']['training_seconds']:.2f} s`",
                    f"- Initial learning rate: `{item['training']['initial_learning_rate']:.6f}`",
                    f"- Final learning rate: `{item['training']['final_learning_rate']:.6f}`",
                    f"- Test combined error: `{item['test_metrics']['combined_error']:.4f}`",
                    f"- Distance MAE: `{item['test_metrics']['distance_mae_m']:.4f} m`",
                    f"- Azimuth MAE: `{item['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                    f"- Elevation MAE: `{item['test_metrics']['elevation_mae_deg']:.4f} deg`",
                    f"- Delta vs CPU baseline: `{item['comparison']['combined_error_delta']:.4f}`",
                ]
            )
            if previous_combined is not None:
                lines.append(f"- Delta vs previous improved-experiment run: `{item['test_metrics']['combined_error'] - previous_combined:.4f}`")
            lines.extend(["", "Implemented steps:"])
            lines.extend([f"- {step}" for step in item["implemented_steps"]])
            lines.extend(["", "Remaining follow-up steps:"])
            lines.extend([f"- {step}" for step in item["remaining_steps"]])
            lines.extend(
                [
                    "",
                    f"![{item['title']} loss](training_improved_experiments/cpu/{item['name']}/loss.png)",
                    f"![{item['title']} comparison](training_improved_experiments/cpu/{item['name']}/comparison.png)",
                    f"![{item['title']} azimuth](training_improved_experiments/cpu/{item['name']}/test_azimuth_prediction.png)",
                    f"![{item['title']} elevation](training_improved_experiments/cpu/{item['name']}/test_elevation_prediction.png)",
                    "",
                ]
            )

        accepted = [item["title"] for item in experiment_payload["experiments"] if item["accepted"]]
        lines.extend(
            [
                "## Experiment Summary",
                "",
                f"- Accepted experiments under the training-improved regime: {', '.join(accepted) if accepted else 'none'}",
                "- The baseline remains the best reference unless an experiment beats it on combined error and at least one individual metric.",
                "",
            ]
        )

    lines.extend(
        [
            "## Next Step",
            "",
            "This report now contains the training-improved baseline and the training-improved CPU experiment results. The next step is to decide whether any of the rejected variants are worth another architectural fix pass.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_training_improved_suite(config: Any, outputs: Any, *, include_experiments: bool = False) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig()
    cpu_context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    params, baseline_label = _baseline_reference_params(cpu_context)

    summary_path = outputs.root / "training_improved_experiments_summary.json"
    if summary_path.exists():
        import json

        existing = json.loads(summary_path.read_text())
    else:
        existing = {}

    cpu_result = existing.get("cpu")
    if cpu_result is None:
        cpu_result = _run_device_baseline(cpu_context, params, baseline_label, training_config, device_name="cpu")

    if existing.get("mps") is not None:
        mps_result = existing["mps"]
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        mps_context = StageContext(config=config, device=torch.device("mps"), outputs=outputs)
        try:
            mps_result = _run_device_baseline(mps_context, params, baseline_label, training_config, device_name="mps")
        except Exception as exc:  # pragma: no cover - device dependent
            mps_result = _capture_failed_device_run(
                outputs.root,
                "mps",
                baseline_label,
                training_config,
                "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
    else:
        print("[training_improved] mps backend unavailable in this environment", flush=True)
        mps_result = {
            "status": "unavailable",
            "device": "mps",
            "baseline_label": baseline_label,
            "dataset_mode": training_config.dataset_mode,
        }

    overall_artifacts = _overall_artifacts(outputs.root, cpu_result, mps_result)
    experiment_payload = existing.get("experiments")
    if include_experiments:
        experiment_payload = _run_cpu_training_improved_experiments(
            config,
            outputs,
            params,
            baseline_label,
            cpu_result,
            training_config,
        )
    report_path = _write_report(
        outputs.root,
        baseline_label,
        training_config,
        cpu_result,
        mps_result,
        overall_artifacts,
        experiment_payload,
    )

    summary = {
        "baseline_label": baseline_label,
        "dataset_mode": training_config.dataset_mode,
        "dataset_counts": {"train": 3500, "val": 750, "test": 750},
        "training_config": {
            "max_epochs": training_config.max_epochs,
            "early_stopping_patience": training_config.early_stopping_patience,
            "early_stopping_min_delta": training_config.early_stopping_min_delta,
            "scheduler_patience": training_config.scheduler_patience,
            "scheduler_factor": training_config.scheduler_factor,
            "scheduler_threshold": training_config.scheduler_threshold,
            "scheduler_min_lr": training_config.scheduler_min_lr,
        },
        "cpu": cpu_result,
        "mps": mps_result,
        "experiments": experiment_payload,
        "overall_artifacts": overall_artifacts,
        "report_path": str(report_path),
    }
    save_json(summary_path, summary)
    return summary


def run_training_improved_baseline_suite(config: Any, outputs: Any) -> dict[str, Any]:
    return run_training_improved_suite(config, outputs, include_experiments=False)
