from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_variants import (
    ExperimentBatch,
    ExperimentalPathwayModel,
    HandcraftedBranchEncoder,
    LearnedBranchEncoder,
)
from models.pathway_snn import PathwayBatch, build_pathway_features
from stages.base import StageContext
from stages.improvement import (
    _apply_standardization,
    _copy_config,
    _distance_candidates,
    _evaluate_dataset_bundle,
    _extract_front_end,
    _fit_standardization,
    _itd_candidates,
    _load_json,
    _prepare_dataset_bundle,
    _slice_acoustic_batch,
)
from utils.common import (
    angular_mae,
    combined_localisation_error,
    distance_mae,
    format_float,
    save_error_histogram,
    save_grouped_bar_chart,
    save_heatmap,
    save_json,
    save_loss_curve,
    save_prediction_scatter,
    save_text_figure,
)


@dataclass
class ExperimentConfigState:
    feature_mode: str = "handcrafted"
    loss_mode: str = "baseline"
    use_resonant: bool = False
    use_sconv: bool = False


@dataclass
class ExperimentSpec:
    name: str
    title: str
    description: str
    rationale: str
    updates: dict[str, Any]
    training_overrides: dict[str, Any]


@dataclass
class ExperimentTrainingResult:
    train_loss: list[float]
    val_loss: list[float]
    val_combined_error: list[float]
    best_epoch: int
    best_val_combined_error: float
    best_state: dict[str, torch.Tensor]
    best_auxiliary_state: dict[str, torch.Tensor] | None
    final_diagnostics: dict[str, torch.Tensor]


@dataclass
class PreparedExperimentData:
    mode: str
    local_config: Any
    params: dict[str, Any]
    train_batch: ExperimentBatch
    val_batch: ExperimentBatch
    test_batch: ExperimentBatch
    train_targets_raw: torch.Tensor
    val_targets_raw: torch.Tensor
    test_targets_raw: torch.Tensor


@dataclass
class ExperimentEvaluation:
    metrics: dict[str, Any]
    predictions: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


@dataclass
class ExperimentRunResult:
    name: str
    title: str
    description: str
    rationale: str
    config: dict[str, Any]
    decision: str
    accepted: bool
    compared_against: str
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    training: dict[str, Any]
    comparison: dict[str, Any]
    artifacts: dict[str, str]


class TaskUncertaintyWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(3))


def _baseline_reference_params(context: StageContext) -> tuple[dict[str, Any], str]:
    model7_path = context.outputs.logs / "model7_enhanced_optuna.json"
    if model7_path.exists():
        model7_payload = _load_json(model7_path)
        best_metrics = model7_payload.get("best_metrics", {})
        best_params = best_metrics.get("best_params")
        if best_params:
            study_name = best_metrics.get("study_name", "pathway_split_enhanced")
            trial_number = best_metrics.get("selected_trial", "unknown")
            return best_params, f"{study_name} trial {trial_number}"
    from stages.improvement import Model6PathwaySplit

    return Model6PathwaySplit().attempt_settings()[1], "model6_pathway_split attempt 2"


def _experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="experiment_1_learned_features",
            title="Experiment 1: Learned Feature Extraction",
            description="Replace handcrafted pathway features with learnable cochlear, delay, and spectral modules.",
            rationale="Learnable filters and temporal kernels may adapt better to the synthetic echo statistics than fixed heuristics.",
            updates={"feature_mode": "learned"},
            training_overrides={"learning_rate_scale": 0.8, "batch_size": 16},
        ),
        ExperimentSpec(
            name="experiment_2_compound_loss",
            title="Experiment 2: Compound Loss Function",
            description="Switch from the baseline weighted SmoothL1 loss to pathway-aligned per-task losses plus spike penalty.",
            rationale="Separating distance, azimuth, and elevation losses should align optimization with the architecture split.",
            updates={"loss_mode": "compound"},
            training_overrides={"learning_rate_scale": 1.0},
        ),
        ExperimentSpec(
            name="experiment_3_uncertainty_weighting",
            title="Experiment 3: Uncertainty Weighting",
            description="Replace manual task weights with learnable task uncertainty parameters.",
            rationale="Automatically learned task weighting may improve balance across distance and angular errors.",
            updates={"loss_mode": "uncertainty"},
            training_overrides={"learning_rate_scale": 0.9},
        ),
        ExperimentSpec(
            name="experiment_4_resonant_neurons",
            title="Experiment 4: Resonant Neurons",
            description="Introduce damped resonant spiking dynamics in the fusion stage.",
            rationale="A second-order resonant fusion layer may sharpen temporal selectivity for pulse-echo timing.",
            updates={"use_resonant": True},
            training_overrides={"learning_rate_scale": 0.9},
        ),
        ExperimentSpec(
            name="experiment_5_sconv2dlstm",
            title="Experiment 5: SConv2dLSTM Augmentation",
            description="Augment the current model with an snnTorch SConv2dLSTM spectral-temporal context path.",
            rationale="An explicit spatiotemporal recurrent branch may capture spectral-temporal cues that the simple fusion head misses.",
            updates={"use_sconv": True},
            training_overrides={"learning_rate_scale": 0.8, "batch_size": 12},
        ),
    ]


def _build_experiment_split(
    acoustic_batch: Any,
    local_config: Any,
    distance_candidates: torch.Tensor,
    itd_candidates: torch.Tensor,
    *,
    num_delay_lines: int,
    num_frequency_channels: int,
    chunk_size: int,
) -> ExperimentBatch:
    transmit_spike_chunks: list[torch.Tensor] = []
    receive_spike_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    azimuth_chunks: list[torch.Tensor] = []
    elevation_chunks: list[torch.Tensor] = []
    spike_count_chunks: list[torch.Tensor] = []

    for start in range(0, acoustic_batch.receive.shape[0], chunk_size):
        stop = min(acoustic_batch.receive.shape[0], start + chunk_size)
        chunk_batch = _slice_acoustic_batch(acoustic_batch, slice(start, stop))
        front = _extract_front_end(chunk_batch, local_config, include_cochlea=False)
        pathways, _ = build_pathway_features(
            front["transmit_spikes"],
            front["receive_spikes"],
            distance_candidates,
            itd_candidates,
            num_delay_lines=num_delay_lines,
            num_frequency_channels=num_frequency_channels,
        )
        transmit_spike_chunks.append(front["transmit_spikes"].to(torch.bool))
        receive_spike_chunks.append(front["receive_spikes"].to(torch.bool))
        distance_chunks.append(pathways.distance)
        azimuth_chunks.append(pathways.azimuth)
        elevation_chunks.append(pathways.elevation)
        spike_count_chunks.append(pathways.spike_count)

    pathway_batch = PathwayBatch(
        distance=torch.cat(distance_chunks, dim=0),
        azimuth=torch.cat(azimuth_chunks, dim=0),
        elevation=torch.cat(elevation_chunks, dim=0),
        spike_count=torch.cat(spike_count_chunks, dim=0),
    )
    return ExperimentBatch(
        transmit_wave=acoustic_batch.transmit,
        receive_wave=acoustic_batch.receive,
        pathway=pathway_batch,
        transmit_spikes=torch.cat(transmit_spike_chunks, dim=0),
        receive_spikes=torch.cat(receive_spike_chunks, dim=0),
        spike_count=pathway_batch.spike_count,
    )


def _prepare_experiment_data(
    context: StageContext,
    params: dict[str, Any],
    dataset_mode: str,
) -> PreparedExperimentData:
    cache_key = f"experiment_data::{dataset_mode}::{json.dumps(params, sort_keys=True)}"
    if cache_key in context.shared:
        return context.shared[cache_key]

    dataset_bundle = _prepare_dataset_bundle(context, dataset_mode)
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

    train_batch = _build_experiment_split(
        dataset_bundle.train_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )
    val_batch = _build_experiment_split(
        dataset_bundle.val_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )
    test_batch = _build_experiment_split(
        dataset_bundle.test_batch,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )

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
    context.shared[cache_key] = prepared
    return prepared


def _batch_iterator(batch: ExperimentBatch, targets: torch.Tensor, batch_size: int) -> tuple[ExperimentBatch, torch.Tensor]:
    permutation = torch.randperm(targets.shape[0], device=targets.device)
    for start in range(0, targets.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield batch.index_select(indices), targets[indices]


def _prediction_metrics(
    local_config: Any,
    prediction: torch.Tensor,
    targets_raw: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> ExperimentEvaluation:
    pred_distance = prediction[:, 0]
    pred_azimuth = prediction[:, 1]
    pred_elevation = prediction[:, 2]
    target_distance = targets_raw[:, 0]
    target_azimuth = targets_raw[:, 1]
    target_elevation = targets_raw[:, 2]
    metrics = {
        "distance_mae_m": distance_mae(pred_distance, target_distance),
        "azimuth_mae_deg": angular_mae(pred_azimuth, target_azimuth),
        "elevation_mae_deg": angular_mae(pred_elevation, target_elevation),
        "combined_error": combined_localisation_error(
            pred_distance,
            target_distance,
            pred_azimuth,
            target_azimuth,
            pred_elevation,
            target_elevation,
            local_config.max_range_m,
        ),
        "mean_spike_rate": diagnostics["spike_rate"].mean().item(),
    }
    predictions = {
        "predicted_distance": pred_distance,
        "predicted_azimuth": pred_azimuth,
        "predicted_elevation": pred_elevation,
        "target_distance": target_distance,
        "target_azimuth": target_azimuth,
        "target_elevation": target_elevation,
    }
    return ExperimentEvaluation(metrics=metrics, predictions=predictions, diagnostics=diagnostics)


def _loss_components(
    prediction: torch.Tensor,
    targets_raw: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    local_config: Any,
    loss_mode: str,
    task_weights: torch.Tensor,
    spike_weight: float,
    uncertainty_module: TaskUncertaintyWeights | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    baseline_scales = prediction.new_tensor([local_config.max_range_m, 45.0, 30.0])
    compound_scales = prediction.new_tensor([local_config.max_range_m, 180.0, 180.0])

    baseline_task_losses = F.smooth_l1_loss(
        prediction / baseline_scales,
        targets_raw / baseline_scales,
        reduction="none",
    ).mean(dim=0)
    compound_task_losses = (
        torch.abs(prediction - targets_raw) / compound_scales.view(1, -1)
    ).mean(dim=0)
    spike_penalty = diagnostics["spike_rate"].mean()

    if loss_mode == "baseline":
        localisation_loss = (baseline_task_losses * task_weights).sum()
    elif loss_mode == "compound":
        localisation_loss = (
            compound_task_losses[0]
            + task_weights[1] * compound_task_losses[1]
            + task_weights[2] * compound_task_losses[2]
        )
    elif loss_mode == "uncertainty":
        if uncertainty_module is None:
            raise ValueError("uncertainty loss mode requires an uncertainty module.")
        log_sigma = uncertainty_module.log_sigma.clamp(-3.0, 2.0)
        localisation_loss = torch.sum(torch.exp(-2.0 * log_sigma) * compound_task_losses + log_sigma)
    else:
        raise ValueError(f"Unsupported loss mode '{loss_mode}'.")

    total_loss = localisation_loss + spike_weight * spike_penalty
    summary = {
        "distance_loss": float(compound_task_losses[0].item()),
        "azimuth_loss": float(compound_task_losses[1].item()),
        "elevation_loss": float(compound_task_losses[2].item()),
        "spike_penalty": float(spike_penalty.item()),
    }
    if uncertainty_module is not None:
        sigma = torch.exp(uncertainty_module.log_sigma.detach().clamp(-3.0, 2.0))
        summary.update(
            {
                "sigma_distance": float(sigma[0].item()),
                "sigma_azimuth": float(sigma[1].item()),
                "sigma_elevation": float(sigma[2].item()),
            }
        )
    return total_loss, summary


def _train_experimental_model(
    model: ExperimentalPathwayModel,
    train_batch: ExperimentBatch,
    train_targets_raw: torch.Tensor,
    val_batch: ExperimentBatch,
    val_targets_raw: torch.Tensor,
    local_config: Any,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    loss_mode: str,
    task_weights: torch.Tensor,
    spike_weight: float,
) -> tuple[ExperimentTrainingResult, TaskUncertaintyWeights | None]:
    uncertainty_module = TaskUncertaintyWeights().to(train_targets_raw.device) if loss_mode == "uncertainty" else None
    parameters = list(model.parameters()) + ([] if uncertainty_module is None else list(uncertainty_module.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_combined_history: list[float] = []
    best_epoch = 0
    best_val_combined = float("inf")
    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    best_auxiliary_state = None if uncertainty_module is None else uncertainty_module.state_dict()
    final_diagnostics: dict[str, torch.Tensor] = {}

    for epoch in range(epochs):
        model.train()
        if uncertainty_module is not None:
            uncertainty_module.train()
        batch_losses = []

        for batch_features, batch_targets in _batch_iterator(train_batch, train_targets_raw, batch_size):
            optimizer.zero_grad(set_to_none=True)
            output, diagnostics = model(batch_features)
            loss, _ = _loss_components(
                output,
                batch_targets,
                diagnostics,
                local_config,
                loss_mode,
                task_weights,
                spike_weight,
                uncertainty_module,
            )
            loss.backward()
            optimizer.step()
            if uncertainty_module is not None:
                uncertainty_module.log_sigma.data.clamp_(-3.0, 2.0)
            batch_losses.append(loss.item())

        train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))

        model.eval()
        if uncertainty_module is not None:
            uncertainty_module.eval()
        with torch.no_grad():
            val_output, val_diagnostics = model(val_batch)
            val_loss, _ = _loss_components(
                val_output,
                val_targets_raw,
                val_diagnostics,
                local_config,
                loss_mode,
                task_weights,
                spike_weight,
                uncertainty_module,
            )
            val_eval = _prediction_metrics(local_config, val_output, val_targets_raw, val_diagnostics)
        val_loss_history.append(float(val_loss.item()))
        val_combined_history.append(float(val_eval.metrics["combined_error"]))

        if float(val_eval.metrics["combined_error"]) < best_val_combined:
            best_epoch = epoch
            best_val_combined = float(val_eval.metrics["combined_error"])
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            best_auxiliary_state = None if uncertainty_module is None else copy.deepcopy(uncertainty_module.state_dict())
            final_diagnostics = {key: value.detach().clone() for key, value in val_diagnostics.items()}

    return (
        ExperimentTrainingResult(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            val_combined_error=val_combined_history,
            best_epoch=best_epoch,
            best_val_combined_error=best_val_combined,
            best_state=best_state,
            best_auxiliary_state=best_auxiliary_state,
            final_diagnostics=final_diagnostics,
        ),
        uncertainty_module,
    )


def _evaluate_model(
    model: ExperimentalPathwayModel,
    batch: ExperimentBatch,
    targets_raw: torch.Tensor,
    local_config: Any,
) -> ExperimentEvaluation:
    model.eval()
    with torch.no_grad():
        prediction, diagnostics = model(batch)
    return _prediction_metrics(local_config, prediction, targets_raw, diagnostics)


def _instantiate_model(
    data: PreparedExperimentData,
    state: ExperimentConfigState,
) -> ExperimentalPathwayModel:
    params = data.params
    if state.feature_mode == "learned":
        encoder = LearnedBranchEncoder(
            data.local_config,
            num_filters=int(params["num_frequency_channels"]),
            num_delay_lines=int(params["num_delay_lines"]),
            branch_hidden_dim=int(params["branch_hidden_dim"]),
        )
    elif state.feature_mode == "handcrafted":
        if data.train_batch.pathway is None:
            raise ValueError("Handcrafted experiment state requires pathway features.")
        encoder = HandcraftedBranchEncoder(
            distance_dim=data.train_batch.pathway.distance.shape[-1],
            azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
            elevation_dim=data.train_batch.pathway.elevation.shape[-1],
            branch_hidden_dim=int(params["branch_hidden_dim"]),
        )
    else:
        raise ValueError(f"Unsupported feature mode '{state.feature_mode}'.")

    return ExperimentalPathwayModel(
        encoder=encoder,
        hidden_dim=int(params["hidden_dim"]),
        output_dim=3,
        num_steps=int(params["num_steps"]),
        beta=float(params["membrane_beta"]),
        threshold=float(params["fusion_threshold"]),
        reset_mechanism=str(params["reset_mechanism"]),
        use_resonant=state.use_resonant,
        use_sconv=state.use_sconv,
    ).to(data.train_targets_raw.device)


def _metrics_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        "combined_error_delta": float(candidate["combined_error"] - reference["combined_error"]),
        "distance_mae_delta": float(candidate["distance_mae_m"] - reference["distance_mae_m"]),
        "azimuth_mae_delta": float(candidate["azimuth_mae_deg"] - reference["azimuth_mae_deg"]),
        "elevation_mae_delta": float(candidate["elevation_mae_deg"] - reference["elevation_mae_deg"]),
        "spike_rate_delta": float(candidate["mean_spike_rate"] - reference["mean_spike_rate"]),
    }


def _is_accepted(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    tolerance = 1e-6
    combined_improved = float(candidate["combined_error"]) < float(reference["combined_error"]) - tolerance
    any_metric_improved = any(
        float(candidate[key]) < float(reference[key]) - tolerance
        for key in ("distance_mae_m", "azimuth_mae_deg", "elevation_mae_deg")
    )
    return combined_improved and any_metric_improved


def _save_baseline_outputs(stage_root: Path, evaluation: dict[str, Any]) -> dict[str, str]:
    baseline_dir = stage_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    save_loss_curve(
        evaluation["training"].train_loss,
        evaluation["training"].val_loss,
        baseline_dir / "loss.png",
        "Baseline Model 6 Loss Curve",
    )
    save_prediction_scatter(
        evaluation["test_target_distance"],
        evaluation["test_predicted_distance"],
        baseline_dir / "test_distance_prediction.png",
        "Baseline Test Distance Prediction",
        xlabel="True Distance (m)",
        ylabel="Predicted Distance (m)",
    )
    save_prediction_scatter(
        evaluation["test_target_azimuth"],
        evaluation["test_predicted_azimuth"],
        baseline_dir / "test_azimuth_prediction.png",
        "Baseline Test Azimuth Prediction",
        xlabel="True Azimuth (deg)",
        ylabel="Predicted Azimuth (deg)",
    )
    save_prediction_scatter(
        evaluation["test_target_elevation"],
        evaluation["test_predicted_elevation"],
        baseline_dir / "test_elevation_prediction.png",
        "Baseline Test Elevation Prediction",
        xlabel="True Elevation (deg)",
        ylabel="Predicted Elevation (deg)",
    )
    save_error_histogram(
        evaluation["test_predicted_distance"] - evaluation["test_target_distance"],
        baseline_dir / "test_distance_error.png",
        "Baseline Test Distance Error",
        xlabel="Distance Error (m)",
    )
    save_error_histogram(
        evaluation["test_predicted_azimuth"] - evaluation["test_target_azimuth"],
        baseline_dir / "test_azimuth_error.png",
        "Baseline Test Azimuth Error",
        xlabel="Azimuth Error (deg)",
    )
    save_error_histogram(
        evaluation["test_predicted_elevation"] - evaluation["test_target_elevation"],
        baseline_dir / "test_elevation_error.png",
        "Baseline Test Elevation Error",
        xlabel="Elevation Error (deg)",
    )
    save_text_figure(
        [
            f"val_combined_error: {evaluation['combined_error']:.4f}",
            f"test_combined_error: {evaluation['test_combined_error']:.4f}",
            f"test_distance_mae_m: {evaluation['test_distance_mae_m']:.4f}",
            f"test_azimuth_mae_deg: {evaluation['test_azimuth_mae_deg']:.4f}",
            f"test_elevation_mae_deg: {evaluation['test_elevation_mae_deg']:.4f}",
            f"test_spike_rate: {evaluation['test_mean_spike_rate']:.4f}",
            f"pred_distance_std: {evaluation['test_predicted_distance'].std().item():.4f}",
            f"pred_azimuth_std: {evaluation['test_predicted_azimuth'].std().item():.4f}",
            f"pred_elevation_std: {evaluation['test_predicted_elevation'].std().item():.4f}",
            f"target_distance_std: {evaluation['test_target_distance'].std().item():.4f}",
            f"target_azimuth_std: {evaluation['test_target_azimuth'].std().item():.4f}",
            f"target_elevation_std: {evaluation['test_target_elevation'].std().item():.4f}",
            f"best_epoch: {evaluation['training'].best_epoch}",
        ],
        baseline_dir / "summary.png",
        "Baseline Experiment Summary",
    )
    return {
        "loss": str(baseline_dir / "loss.png"),
        "test_distance_prediction": str(baseline_dir / "test_distance_prediction.png"),
        "test_azimuth_prediction": str(baseline_dir / "test_azimuth_prediction.png"),
        "test_elevation_prediction": str(baseline_dir / "test_elevation_prediction.png"),
        "test_distance_error": str(baseline_dir / "test_distance_error.png"),
        "test_azimuth_error": str(baseline_dir / "test_azimuth_error.png"),
        "test_elevation_error": str(baseline_dir / "test_elevation_error.png"),
        "summary": str(baseline_dir / "summary.png"),
    }


def _save_experiment_outputs(
    stage_root: Path,
    spec: ExperimentSpec,
    result: ExperimentRunResult,
    train_result: ExperimentTrainingResult,
    test_eval: ExperimentEvaluation,
    comparison_reference_name: str,
    comparison_reference_metrics: dict[str, Any],
    model: ExperimentalPathwayModel,
) -> dict[str, str]:
    stage_dir = stage_root / spec.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    save_loss_curve(
        train_result.train_loss,
        train_result.val_loss,
        stage_dir / "loss.png",
        spec.title,
    )
    save_prediction_scatter(
        test_eval.predictions["target_distance"],
        test_eval.predictions["predicted_distance"],
        stage_dir / "test_distance_prediction.png",
        f"{spec.title} Distance Prediction",
        xlabel="True Distance (m)",
        ylabel="Predicted Distance (m)",
    )
    save_prediction_scatter(
        test_eval.predictions["target_azimuth"],
        test_eval.predictions["predicted_azimuth"],
        stage_dir / "test_azimuth_prediction.png",
        f"{spec.title} Azimuth Prediction",
        xlabel="True Azimuth (deg)",
        ylabel="Predicted Azimuth (deg)",
    )
    save_prediction_scatter(
        test_eval.predictions["target_elevation"],
        test_eval.predictions["predicted_elevation"],
        stage_dir / "test_elevation_prediction.png",
        f"{spec.title} Elevation Prediction",
        xlabel="True Elevation (deg)",
        ylabel="Predicted Elevation (deg)",
    )
    save_error_histogram(
        test_eval.predictions["predicted_elevation"] - test_eval.predictions["target_elevation"],
        stage_dir / "test_elevation_error.png",
        f"{spec.title} Elevation Error",
        xlabel="Elevation Error (deg)",
    )
    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
        {
            comparison_reference_name: [
                float(comparison_reference_metrics["distance_mae_m"]),
                float(comparison_reference_metrics["azimuth_mae_deg"]),
                float(comparison_reference_metrics["elevation_mae_deg"]),
                float(comparison_reference_metrics["combined_error"]),
            ],
            "Candidate": [
                float(result.test_metrics["distance_mae_m"]),
                float(result.test_metrics["azimuth_mae_deg"]),
                float(result.test_metrics["elevation_mae_deg"]),
                float(result.test_metrics["combined_error"]),
            ],
        },
        stage_dir / "comparison.png",
        f"{spec.title} Comparison",
        ylabel="Error",
    )
    save_text_figure(
        [
            f"decision: {result.decision}",
            f"accepted: {result.accepted}",
            f"compared_against: {result.compared_against}",
            f"test_combined_error: {result.test_metrics['combined_error']:.4f}",
            f"test_distance_mae_m: {result.test_metrics['distance_mae_m']:.4f}",
            f"test_azimuth_mae_deg: {result.test_metrics['azimuth_mae_deg']:.4f}",
            f"test_elevation_mae_deg: {result.test_metrics['elevation_mae_deg']:.4f}",
            f"test_spike_rate: {result.test_metrics['mean_spike_rate']:.4f}",
            f"best_epoch: {train_result.best_epoch}",
        ],
        stage_dir / "summary.png",
        f"{spec.title} Summary",
    )

    artifacts = {
        "loss": str(stage_dir / "loss.png"),
        "test_distance_prediction": str(stage_dir / "test_distance_prediction.png"),
        "test_azimuth_prediction": str(stage_dir / "test_azimuth_prediction.png"),
        "test_elevation_prediction": str(stage_dir / "test_elevation_prediction.png"),
        "test_elevation_error": str(stage_dir / "test_elevation_error.png"),
        "comparison": str(stage_dir / "comparison.png"),
        "summary": str(stage_dir / "summary.png"),
    }

    encoder = model.encoder
    if isinstance(encoder, LearnedBranchEncoder):
        initial_filters = encoder.initial_filterbank_weight.squeeze(1)
        learned_filters = encoder.filterbank.weight.detach().squeeze(1)
        initial_response = torch.abs(torch.fft.rfft(initial_filters, n=512, dim=-1))
        learned_response = torch.abs(torch.fft.rfft(learned_filters, n=512, dim=-1))
        distance_kernels = encoder.distance_delay.weight.detach().reshape(encoder.distance_delay.out_channels, -1)

        save_heatmap(
            torch.cat([initial_filters, learned_filters], dim=0),
            stage_dir / "filter_kernels.png",
            f"{spec.title} Initial and Learned Filter Kernels",
            xlabel="Time Sample",
            ylabel="Filter",
        )
        save_heatmap(
            torch.cat([initial_response, learned_response], dim=0),
            stage_dir / "filter_frequency_response.png",
            f"{spec.title} Initial and Learned Frequency Responses",
            xlabel="Frequency Bin",
            ylabel="Filter",
        )
        save_heatmap(
            distance_kernels,
            stage_dir / "delay_kernels.png",
            f"{spec.title} Learned Delay Kernels",
            xlabel="Kernel Weight Index",
            ylabel="Delay Channel",
        )
        artifacts.update(
            {
                "filter_kernels": str(stage_dir / "filter_kernels.png"),
                "filter_frequency_response": str(stage_dir / "filter_frequency_response.png"),
                "delay_kernels": str(stage_dir / "delay_kernels.png"),
            }
        )

    save_json(stage_dir / "metrics.json", result.__dict__)
    return artifacts


def _overall_artifacts(stage_root: Path, baseline: dict[str, Any], results: list[ExperimentRunResult]) -> dict[str, str]:
    labels = ["Baseline"] + [result.title.replace("Experiment ", "Exp ") for result in results]
    series = {
        "Distance MAE": [float(baseline["distance_mae_m"])] + [float(result.test_metrics["distance_mae_m"]) for result in results],
        "Azimuth MAE": [float(baseline["azimuth_mae_deg"])] + [float(result.test_metrics["azimuth_mae_deg"]) for result in results],
        "Elevation MAE": [float(baseline["elevation_mae_deg"])] + [float(result.test_metrics["elevation_mae_deg"]) for result in results],
        "Combined Error": [float(baseline["combined_error"])] + [float(result.test_metrics["combined_error"]) for result in results],
    }
    save_grouped_bar_chart(
        list(series.keys()),
        {label: [series[key][index] for key in series] for index, label in enumerate(labels)},
        stage_root / "overall_test_comparison.png",
        "Experimental Pipeline Test Metrics",
        ylabel="Error",
    )
    save_grouped_bar_chart(
        labels,
        {
            "Spike Rate": [float(baseline["mean_spike_rate"])] + [float(result.test_metrics["mean_spike_rate"]) for result in results],
        },
        stage_root / "overall_spike_rate.png",
        "Experimental Pipeline Spike Rate",
        ylabel="Mean Spike Rate",
    )
    return {
        "overall_test_comparison": str(stage_root / "overall_test_comparison.png"),
        "overall_spike_rate": str(stage_root / "overall_spike_rate.png"),
    }


def _failure_analysis_lines(result: ExperimentRunResult) -> list[str]:
    if result.name == "experiment_1_learned_features":
        return [
            "- Likely cause: the learnable front end removed the handcrafted binaural and spectral inductive bias all at once, while keeping a small dataset and short training schedule. The Conv1d/Conv2d replacement is therefore likely learning coarse energy structure rather than stable ITD and spectral-elevation cues.",
            "- Assessment: mostly an implementation issue. This is not strong evidence that learned feature extraction is a bad idea in principle; it is strong evidence that this specific low-data replacement strategy degrades localisation.",
        ]
    if result.name == "experiment_2_compound_loss":
        return [
            "- Likely cause: the compound loss underweighted angular terms. In the current implementation both angles were normalized by `180 deg`, even though the actual sampled ranges are much narrower, so the optimizer could improve distance while largely ignoring azimuth and elevation.",
            "- Assessment: implementation issue. This is not a strong negative finding against compound losses; it is a strong finding that this particular scaling choice damages angular learning.",
        ]
    if result.name == "experiment_3_uncertainty_weighting":
        return [
            "- Likely cause: uncertainty weighting sat on top of the same loss scaling used in Experiment 2. The learned task uncertainties converged to similar values instead of correcting the imbalance, so the angular tasks remained weakly driven.",
            "- Assessment: mostly an implementation issue. This does not strongly reject uncertainty weighting as a method; it shows that the current uncertainty formulation did not rescue the mis-scaled objective.",
        ]
    if result.name == "experiment_4_resonant_neurons":
        return [
            "- Likely cause: the resonant block replaced a stable leaky fusion stage with an uncalibrated second-order dynamical system. The much higher spike rate suggests that it became over-excitable and distorted the precise timing information needed for distance and angle estimation.",
            "- Assessment: mixed. This is a strong negative result for this exact resonant implementation, but not a strong biological conclusion that resonant neurons are unsuitable in general.",
        ]
    if result.name == "experiment_5_sconv2dlstm":
        return [
            "- Likely cause: the added `SConv2dLSTM` branch appears to preserve some broad spectral-temporal structure, but its pooled context likely blurs the timing precision needed for distance. The earlier state-reset bug was fixed, and the model still underperformed, so the remaining drop is not just a broken run.",
            "- Assessment: moderate negative result for the current integration strategy. This does not reject `SConv2dLSTM` in principle, but it is good evidence that this way of inserting it into fusion hurts accuracy.",
        ]
    return [
        "- Likely cause: the modification changed the optimization or representation in a way that was not matched to the current dataset and cue structure.",
        "- Assessment: this should be treated as a negative result for the present implementation, not a general rejection of the broader idea.",
    ]


def _proposed_solution_lines(result: ExperimentRunResult) -> list[str]:
    if result.name == "experiment_1_learned_features":
        return [
            "- Keep the pathway split but replace only one handcrafted block at a time, starting with the elevation branch rather than all branches simultaneously.",
            "- Initialize the learned cochlear and delay filters from the handcrafted templates, freeze them for a short warm-up period, then unfreeze gradually.",
            "- Add structural regularizers so the learned filters stay bandpass and the delay kernels stay smooth and localized.",
            "- Re-run with the `stable` dataset split and longer training, because the current learned front end is too data-hungry for the `dev` setup.",
        ]
    if result.name == "experiment_2_compound_loss":
        return [
            "- Rescale the angular losses by the actual sampled target ranges, for example azimuth by `45 deg` and elevation by `30 deg`, instead of `180 deg`.",
            "- Tune `lambda_d`, `lambda_a`, and `lambda_e` explicitly after the normalization is corrected.",
            "- Log per-task gradient magnitudes during training to verify that azimuth and elevation are still receiving meaningful updates.",
        ]
    if result.name == "experiment_3_uncertainty_weighting":
        return [
            "- Apply uncertainty weighting only after fixing the base task normalization used in Experiment 2.",
            "- Initialize the uncertainty parameters near the current successful manual weights rather than starting all tasks equally.",
            "- Delay learning of the uncertainty terms for a few epochs or regularize them more strongly so they do not settle into a weak but balanced solution too early.",
        ]
    if result.name == "experiment_4_resonant_neurons":
        return [
            "- Constrain resonance frequency and damping to a narrower biologically plausible range matched to the echo envelope timescale.",
            "- Insert the resonant dynamics only in the distance pathway first, instead of replacing the whole fusion stage.",
            "- Add a stronger spike-rate penalty or explicit stability constraint, since the current resonant block became over-active.",
        ]
    if result.name == "experiment_5_sconv2dlstm":
        return [
            "- Move `SConv2dLSTM` into the elevation or spectral branch instead of fusing its pooled output directly into the global head.",
            "- Preserve more timing detail by avoiding aggressive temporal pooling before the recurrent layer.",
            "- Use a residual integration strategy where the baseline fusion head remains dominant and the recurrent branch only adds a correction term.",
        ]
    return [
        "- Narrow the scope of the modification so it changes one part of the system at a time.",
        "- Add diagnostics that verify prediction variance, per-task gradients, and spike-rate stability before interpreting the result as a true model finding.",
    ]


def _write_experiment_report(
    context: StageContext,
    stage_root: Path,
    dataset_mode: str,
    baseline_label: str,
    baseline_metrics: dict[str, Any],
    baseline_artifacts: dict[str, str],
    results: list[ExperimentRunResult],
    final_best_name: str,
) -> Path:
    lines = [
        "# Controlled Experimental Pipeline",
        "",
        f"- Dataset mode: `{dataset_mode}`",
        f"- Baseline reference: `{baseline_label}`",
        f"- Final selected model: `{final_best_name}`",
        "",
        "## Overview",
        "",
        "The experiments used one fixed dataset split and one shared evaluation protocol. Each candidate was compared against the current accepted model. A change was accepted only if it improved combined localisation error and at least one individual metric.",
        "",
        "## Implementation Details",
        "",
        "The controlled pipeline was added without replacing the working Model 6 / Model 7 path. The new entrypoint is `experiments.py`, the shared protocol and report generation live in `stages/experiments.py`, and the new model variants live in `models/experimental_variants.py`.",
        "",
        "- Experiment 0 control: the tuned pathway-split baseline is evaluated through the same fixed dataset split and the same held-out test metrics used for the later variants.",
        "- Shared dataset protocol: one cached `train/val/test` split is prepared once, using the same synthetic scene generator, the same cochlea/spike front end, and the same target ranges for every experiment.",
        "- Shared acceptance rule: a variant is accepted only if its test combined error is lower than the current accepted model and at least one of distance, azimuth, or elevation MAE is also lower.",
        "- Experiment 1 implementation: `LearnedBranchEncoder` replaces handcrafted pathway extraction with a learnable Conv1d cochlear bank, learnable temporal delay kernels for distance and azimuth, and a Conv2d spectral branch for elevation.",
        "- Experiment 2 implementation: `loss_mode=\"compound\"` keeps the same model but replaces the baseline weighted SmoothL1 objective with separate distance, azimuth, and elevation losses plus the spike penalty.",
        "- Experiment 3 implementation: `TaskUncertaintyWeights` adds three learnable `log_sigma` parameters and uses them to reweight the per-task losses during training.",
        "- Experiment 4 implementation: `use_resonant=True` swaps the standard leaky fusion dynamics for a second-order resonant fusion block with damped oscillatory state updates.",
        "- Experiment 5 implementation: `use_sconv=True` adds an `snn.SConv2dLSTM` branch that reads spectral-temporal source frames and contributes a pooled recurrent context vector at fusion time.",
        "- Training protocol for variants: all new variants are trained from scratch on the same cached split, with the same number of epochs unless an experiment-specific override is stated in the config block.",
        "- Interpretation of flat responses: if Experiment 0 shows broad prediction spread and good error while a candidate collapses, the issue is in the added feature/loss/dynamics change rather than the report or metric code.",
        "",
        "## Experiments",
        "",
        "| Experiment | Change | Result | Accepted |",
        "| --- | --- | --- | --- |",
        "| Experiment 0: Control (Model 6) | Tuned pathway-split baseline evaluated inside the experiment harness. | "
        f"combined {baseline_metrics['combined_error']:.4f}, distance {baseline_metrics['distance_mae_m']:.4f} m | Reference |",
    ]
    for result in results:
        summary = f"combined {result.test_metrics['combined_error']:.4f}, distance {result.test_metrics['distance_mae_m']:.4f} m"
        lines.append(f"| {result.title} | {result.description} | {summary} | {'Yes' if result.accepted else 'No'} |")

    lines.extend(
        [
            "",
            "## Experiment 0 Control",
            "",
            f"- Test combined error: `{baseline_metrics['combined_error']:.4f}`",
            f"- Test distance MAE: `{baseline_metrics['distance_mae_m']:.4f} m`",
            f"- Test azimuth MAE: `{baseline_metrics['azimuth_mae_deg']:.4f} deg`",
            f"- Test elevation MAE: `{baseline_metrics['elevation_mae_deg']:.4f} deg`",
            f"- Test spike rate: `{baseline_metrics['mean_spike_rate']:.4f}`",
            f"- Predicted distance std: `{baseline_metrics['predicted_distance_std']:.4f}`",
            f"- Predicted azimuth std: `{baseline_metrics['predicted_azimuth_std']:.4f}`",
            f"- Predicted elevation std: `{baseline_metrics['predicted_elevation_std']:.4f}`",
            f"- Target distance std: `{baseline_metrics['target_distance_std']:.4f}`",
            f"- Target azimuth std: `{baseline_metrics['target_azimuth_std']:.4f}`",
            f"- Target elevation std: `{baseline_metrics['target_elevation_std']:.4f}`",
            "",
            "The control is included to check whether the edited experimental harness itself is collapsing to flat predictions. If the control retains substantial prediction variance and low error, then the flat-response issue is specific to the modified variants rather than the evaluation stack.",
            "",
            "Proposed solution:",
            "- Keep Experiment 0 as a required sanity check for every future run.",
            "- Add automatic guards that flag a candidate when prediction standard deviation collapses far below the target standard deviation.",
            "",
            "![Experiment 0 loss](experiments/baseline/loss.png)",
            "![Experiment 0 distance](experiments/baseline/test_distance_prediction.png)",
            "![Experiment 0 azimuth](experiments/baseline/test_azimuth_prediction.png)",
            "![Experiment 0 elevation](experiments/baseline/test_elevation_prediction.png)",
            "![Experiment 0 azimuth error](experiments/baseline/test_azimuth_error.png)",
            "![Experiment 0 elevation error](experiments/baseline/test_elevation_error.png)",
            "",
            "![Overall test comparison](experiments/overall_test_comparison.png)",
            "![Overall spike rate](experiments/overall_spike_rate.png)",
            "",
            "## Detailed Analysis",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"### {result.title}",
                "",
                f"- Change: {result.description}",
                f"- Why it should help: {result.rationale}",
                f"- Compared against: `{result.compared_against}`",
                f"- Decision: `{result.decision}`",
                f"- Test combined error: `{result.test_metrics['combined_error']:.4f}`",
                f"- Test distance MAE: `{result.test_metrics['distance_mae_m']:.4f} m`",
                f"- Test azimuth MAE: `{result.test_metrics['azimuth_mae_deg']:.4f} deg`",
                f"- Test elevation MAE: `{result.test_metrics['elevation_mae_deg']:.4f} deg`",
                f"- Test spike rate: `{result.test_metrics['mean_spike_rate']:.4f}`",
                f"- Combined delta vs reference: `{result.comparison['against_active']['combined_error_delta']:.4f}`",
                "",
                "Failure analysis:",
                *_failure_analysis_lines(result),
                "",
                "Proposed solution:",
                *_proposed_solution_lines(result),
                "",
                f"![{result.title} loss](experiments/{result.name}/loss.png)",
                f"![{result.title} comparison](experiments/{result.name}/comparison.png)",
                f"![{result.title} distance](experiments/{result.name}/test_distance_prediction.png)",
                f"![{result.title} azimuth](experiments/{result.name}/test_azimuth_prediction.png)",
                f"![{result.title} elevation](experiments/{result.name}/test_elevation_prediction.png)",
                "",
            ]
        )
        if "filter_kernels" in result.artifacts:
            lines.extend(
                [
                    "#### Learned Feature Analysis",
                    "",
                    "The learned front end is compared against the log-spaced initialization, which acts as the handcrafted reference template.",
                    "",
                    f"![{result.title} filter kernels](experiments/{result.name}/filter_kernels.png)",
                    f"![{result.title} frequency response](experiments/{result.name}/filter_frequency_response.png)",
                    f"![{result.title} delay kernels](experiments/{result.name}/delay_kernels.png)",
                    "",
                ]
            )

    accepted = [result.title for result in results if result.accepted]
    rejected = [result.title for result in results if not result.accepted]
    lines.extend(
        [
            "## Key Insights",
            "",
            f"- Accepted changes: {', '.join(accepted) if accepted else 'none'}",
            f"- Rejected changes: {', '.join(rejected) if rejected else 'none'}",
            "- The main limiting factor remains front-end cue quality rather than readout capacity alone.",
            "- Loss shaping can help balance distance and angle errors, but only when the feature extractor remains stable.",
            "- More expressive temporal layers increase cost quickly, so they need clear metric gains to justify acceptance.",
            "",
            "## Updated Best Model",
            "",
            f"The current best experiment-selected model is `{final_best_name}`.",
            "",
            "## Recommendations",
            "",
            "- Re-run the accepted stack on the `stable` dataset split before treating the gain as robust.",
            "- Improve elevation realism with ear-specific spectral filtering before further expanding the fusion stack.",
            "- If the learned front end is promising, add a regularizer that keeps the filters bandpass and delay kernels smooth.",
        ]
    )

    report_path = context.outputs.root / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_experiments(context: StageContext) -> dict[str, Any]:
    dataset_mode = os.environ.get("RADAR_SNN_EXPERIMENT_DATASET_MODE", "dev")
    params, baseline_label = _baseline_reference_params(context)
    print(f"[experiments] preparing dataset mode={dataset_mode}", flush=True)
    data = _prepare_experiment_data(context, params, dataset_mode)

    stage_root = context.outputs.root / "experiments"
    stage_root.mkdir(parents=True, exist_ok=True)

    print("[experiments] evaluating official baseline", flush=True)
    baseline_evaluation = _evaluate_dataset_bundle(context, params, _prepare_dataset_bundle(context, dataset_mode), include_artifacts=False)
    baseline_test_metrics = {
        "distance_mae_m": baseline_evaluation["test_distance_mae_m"],
        "azimuth_mae_deg": baseline_evaluation["test_azimuth_mae_deg"],
        "elevation_mae_deg": baseline_evaluation["test_elevation_mae_deg"],
        "combined_error": baseline_evaluation["test_combined_error"],
        "mean_spike_rate": baseline_evaluation["test_mean_spike_rate"],
        "predicted_distance_std": baseline_evaluation["test_predicted_distance"].std().item(),
        "predicted_azimuth_std": baseline_evaluation["test_predicted_azimuth"].std().item(),
        "predicted_elevation_std": baseline_evaluation["test_predicted_elevation"].std().item(),
        "target_distance_std": baseline_evaluation["test_target_distance"].std().item(),
        "target_azimuth_std": baseline_evaluation["test_target_azimuth"].std().item(),
        "target_elevation_std": baseline_evaluation["test_target_elevation"].std().item(),
    }
    baseline_artifacts = _save_baseline_outputs(stage_root, baseline_evaluation)

    accepted_state_snapshot = ExperimentConfigState()
    active_reference_name = "baseline"
    active_reference_metrics = baseline_test_metrics
    accepted_best_name = "baseline"

    experiment_results: list[ExperimentRunResult] = []
    best_model_payload: dict[str, Any] | None = None

    for index, spec in enumerate(_experiment_specs(), start=1):
        print(f"[experiments] running {spec.name} ({index}/5)", flush=True)
        trial_state = copy.deepcopy(accepted_state_snapshot)
        for key, value in spec.updates.items():
            setattr(trial_state, key, value)

        model = _instantiate_model(data, trial_state)
        task_weights = torch.tensor(
            [1.0, float(params["angle_weight"]), float(params["elevation_weight"])],
            device=context.device,
        )
        learning_rate = float(params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0))
        batch_size = int(spec.training_overrides.get("batch_size", int(params["batch_size"])))
        epochs = int(spec.training_overrides.get("epochs", int(params["epochs"])))

        train_result, uncertainty_module = _train_experimental_model(
            model,
            data.train_batch,
            data.train_targets_raw,
            data.val_batch,
            data.val_targets_raw,
            data.local_config,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_mode=trial_state.loss_mode,
            task_weights=task_weights,
            spike_weight=float(params["loss_weighting"]),
        )
        model.load_state_dict(train_result.best_state)
        if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
            uncertainty_module.load_state_dict(train_result.best_auxiliary_state)

        val_eval = _evaluate_model(model, data.val_batch, data.val_targets_raw, data.local_config)
        test_eval = _evaluate_model(model, data.test_batch, data.test_targets_raw, data.local_config)

        accepted = _is_accepted(test_eval.metrics, active_reference_metrics)
        decision = "ACCEPTED" if accepted else "REJECTED"
        comparison = {
            "against_active": _metrics_delta(test_eval.metrics, active_reference_metrics),
            "against_baseline": _metrics_delta(test_eval.metrics, baseline_test_metrics),
        }
        run_result = ExperimentRunResult(
            name=spec.name,
            title=spec.title,
            description=spec.description,
            rationale=spec.rationale,
            config={
                "feature_mode": trial_state.feature_mode,
                "loss_mode": trial_state.loss_mode,
                "use_resonant": trial_state.use_resonant,
                "use_sconv": trial_state.use_sconv,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            },
            decision=decision,
            accepted=accepted,
            compared_against=active_reference_name,
            val_metrics=val_eval.metrics,
            test_metrics=test_eval.metrics,
            training={
                "best_epoch": train_result.best_epoch,
                "best_val_combined_error": train_result.best_val_combined_error,
                "train_loss": train_result.train_loss,
                "val_loss": train_result.val_loss,
                "val_combined_error": train_result.val_combined_error,
                "uncertainty_state": None if uncertainty_module is None else {
                    "sigma_distance": float(torch.exp(uncertainty_module.log_sigma.detach()[0]).item()),
                    "sigma_azimuth": float(torch.exp(uncertainty_module.log_sigma.detach()[1]).item()),
                    "sigma_elevation": float(torch.exp(uncertainty_module.log_sigma.detach()[2]).item()),
                },
            },
            comparison=comparison,
            artifacts={},
        )
        run_result.artifacts = _save_experiment_outputs(
            stage_root,
            spec,
            run_result,
            train_result,
            test_eval,
            active_reference_name,
            active_reference_metrics,
            model,
        )
        experiment_results.append(run_result)

        if accepted:
            accepted_state_snapshot = trial_state
            active_reference_name = spec.name
            active_reference_metrics = test_eval.metrics
            accepted_best_name = spec.name
            best_model_payload = {
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": run_result.config,
                "test_metrics": run_result.test_metrics,
                "title": run_result.title,
            }

    overall_artifacts = _overall_artifacts(stage_root, baseline_test_metrics, experiment_results)
    report_path = _write_experiment_report(
        context,
        stage_root,
        dataset_mode,
        baseline_label,
        baseline_test_metrics,
        baseline_artifacts,
        experiment_results,
        accepted_best_name,
    )

    summary_payload = {
        "dataset_mode": dataset_mode,
        "baseline_label": baseline_label,
        "baseline_test_metrics": {key: format_float(value) for key, value in baseline_test_metrics.items()},
        "baseline_artifacts": baseline_artifacts,
        "experiments": [
            {
                "name": result.name,
                "title": result.title,
                "decision": result.decision,
                "accepted": result.accepted,
                "config": result.config,
                "test_metrics": {key: format_float(value) for key, value in result.test_metrics.items()},
                "val_metrics": {key: format_float(value) for key, value in result.val_metrics.items()},
                "comparison": {name: {metric: format_float(value) for metric, value in payload.items()} for name, payload in result.comparison.items()},
                "artifacts": result.artifacts,
            }
            for result in experiment_results
        ],
        "overall_artifacts": overall_artifacts,
        "final_best_name": accepted_best_name,
        "report_path": str(report_path),
    }
    save_json(stage_root / "results.json", summary_payload)
    if best_model_payload is not None:
        torch.save(best_model_payload, stage_root / "best_model.pt")
    return summary_payload
