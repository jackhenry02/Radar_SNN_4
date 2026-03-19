from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_variants import (
    CombinedElevationEncoder,
    DistanceResonanceModel,
    ElevationSConvResidualEncoder,
    ExperimentalPathwayModel,
    HandcraftedBranchEncoder,
    ResidualElevationEncoder,
)
from stages.base import StageContext
from stages.experiments import (
    ExperimentEvaluation,
    ExperimentTrainingResult,
    TaskUncertaintyWeights,
    _baseline_reference_params,
    _batch_iterator,
    _metrics_delta,
    _prediction_metrics,
    _prepare_experiment_data,
    _save_baseline_outputs,
)
from stages.improvement import _apply_standardization, _evaluate_dataset_bundle, _fit_standardization, _prepare_dataset_bundle
from utils.common import (
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
class ImprovedExperimentSpec:
    name: str
    title: str
    description: str
    rationale: str
    implemented_steps: list[str]
    remaining_steps: list[str]
    variant: str
    loss_mode: str
    training_overrides: dict[str, Any]


@dataclass
class TargetBundle:
    train_model: torch.Tensor
    val_model: torch.Tensor
    test_model: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def _improved_experiment_specs() -> list[ImprovedExperimentSpec]:
    return [
        ImprovedExperimentSpec(
            name="improved_experiment_1_learned_features",
            title="Improved Experiment 1: Residual Learned Elevation",
            description="Keep handcrafted distance/azimuth features and replace only the elevation branch with a residual learned spectral module.",
            rationale="This targets the weakest cue branch first while preserving the baseline timing and binaural inductive bias.",
            implemented_steps=[
                "Step 1: keep handcrafted distance and azimuth pathways unchanged.",
                "Step 2: replace only the elevation branch with a learned residual spectral CNN.",
                "Step 3: keep the learned residual contribution small at initialization so the baseline pathway still dominates early training.",
            ],
            remaining_steps=[
                "Add explicit bandpass and smoothness regularization to the learned spectral filters.",
                "Warm-start from the stable dataset split if the dev split is still too small.",
            ],
            variant="residual_elevation",
            loss_mode="baseline_standardized",
            training_overrides={"learning_rate_scale": 0.9, "batch_size": 24},
        ),
        ImprovedExperimentSpec(
            name="improved_experiment_2_compound_loss",
            title="Improved Experiment 2: Corrected Compound Loss",
            description="Use a compound loss with distance, azimuth, and elevation normalized by their actual sampled ranges rather than by global angular bounds.",
            rationale="The previous compound loss likely collapsed angle learning because its angular terms were too weak relative to distance.",
            implemented_steps=[
                "Step 1: keep the baseline architecture unchanged.",
                "Step 2: normalize distance by max range, azimuth by 45 deg, and elevation by 30 deg.",
                "Step 3: reuse the manual task weights only after correcting the term scales.",
            ],
            remaining_steps=[
                "Tune the task weights after confirming that the corrected scaling no longer collapses angles.",
                "Log per-task gradient norms to verify balance directly.",
            ],
            variant="baseline",
            loss_mode="corrected_compound",
            training_overrides={"learning_rate_scale": 1.0, "batch_size": 32},
        ),
        ImprovedExperimentSpec(
            name="improved_experiment_3_uncertainty_weighting",
            title="Improved Experiment 3: Corrected Uncertainty Weighting",
            description="Apply uncertainty weighting only after correcting the task normalization and initialize the uncertainty terms near the successful manual weighting.",
            rationale="This tests uncertainty weighting under a fairer objective instead of asking it to rescue a mis-scaled loss.",
            implemented_steps=[
                "Step 1: use the corrected per-task normalization from Improved Experiment 2.",
                "Step 2: initialize the uncertainty parameters from the baseline manual weights.",
                "Step 3: freeze the uncertainty parameters during a short warm-up so the task head stabilizes first.",
            ],
            remaining_steps=[
                "Add regularization toward the manual weighting if the learned sigmas drift too early.",
                "Test a longer warm-up period on the stable split.",
            ],
            variant="baseline",
            loss_mode="corrected_uncertainty",
            training_overrides={"learning_rate_scale": 0.9, "batch_size": 32, "uncertainty_warmup_epochs": 4},
        ),
        ImprovedExperimentSpec(
            name="improved_experiment_4_resonant_neurons",
            title="Improved Experiment 4: Distance-Only Resonance",
            description="Confine resonant dynamics to the distance pathway and keep the baseline fusion stage for azimuth and elevation.",
            rationale="The original resonant experiment changed too much at once and likely destabilized all three outputs.",
            implemented_steps=[
                "Step 1: keep the handcrafted branch encoder and the baseline fusion head.",
                "Step 2: apply resonant dynamics only to the distance branch before fusion.",
                "Step 3: constrain resonance frequency and damping to a narrow range and increase the spike penalty slightly.",
            ],
            remaining_steps=[
                "Sweep the resonance band against the echo envelope timescale rather than using a single constrained range.",
                "Test the resonant block in the distance pathway only on the stable split.",
            ],
            variant="distance_resonance",
            loss_mode="baseline_standardized",
            training_overrides={"learning_rate_scale": 0.9, "batch_size": 24, "spike_weight_scale": 1.5},
        ),
        ImprovedExperimentSpec(
            name="improved_experiment_5_sconv2dlstm",
            title="Improved Experiment 5: Elevation SConv Residual",
            description="Move SConv2dLSTM into the elevation pathway as a residual spectral-temporal correction instead of a global fusion addition.",
            rationale="This keeps the baseline timing path intact while giving the elevation branch extra spectral-temporal capacity.",
            implemented_steps=[
                "Step 1: keep the baseline handcrafted pathway encoder as the main route.",
                "Step 2: add an SConv2dLSTM branch only inside the elevation pathway.",
                "Step 3: inject the recurrent output as a small residual correction rather than as a new dominant fusion feature.",
            ],
            remaining_steps=[
                "Reduce temporal pooling further if CPU budget allows and timing detail is still being blurred.",
                "Try an ear-specific spectral branch before the recurrent block for stronger elevation cues.",
            ],
            variant="elevation_sconv_residual",
            loss_mode="baseline_standardized",
            training_overrides={"learning_rate_scale": 0.85, "batch_size": 16},
        ),
    ]


def _prepare_target_bundle(data: Any) -> TargetBundle:
    train = torch.stack(
        [data.train_targets_raw[:, 0], data.train_targets_raw[:, 1] / 45.0, data.train_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    val = torch.stack(
        [data.val_targets_raw[:, 0], data.val_targets_raw[:, 1] / 45.0, data.val_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    test = torch.stack(
        [data.test_targets_raw[:, 0], data.test_targets_raw[:, 1] / 45.0, data.test_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    mean, std = _fit_standardization(train)
    return TargetBundle(
        train_model=_apply_standardization(train, mean, std),
        val_model=_apply_standardization(val, mean, std),
        test_model=_apply_standardization(test, mean, std),
        mean=mean,
        std=std,
    )


def _decode_model_output(output_model: torch.Tensor, target_bundle: TargetBundle) -> torch.Tensor:
    denormalized = output_model * target_bundle.std + target_bundle.mean
    return torch.stack(
        [denormalized[:, 0], denormalized[:, 1] * 45.0, denormalized[:, 2] * 30.0],
        dim=-1,
    )


def _improved_loss_components(
    output_model: torch.Tensor,
    target_model: torch.Tensor,
    target_raw: torch.Tensor,
    target_bundle: TargetBundle,
    diagnostics: dict[str, torch.Tensor],
    local_config: Any,
    loss_mode: str,
    task_weights: torch.Tensor,
    spike_weight: float,
    uncertainty_module: TaskUncertaintyWeights | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    spike_penalty = diagnostics["spike_rate"].mean()
    raw_prediction = _decode_model_output(output_model, target_bundle)
    corrected_scales = raw_prediction.new_tensor([local_config.max_range_m, 45.0, 30.0]).view(1, -1)

    if loss_mode == "baseline_standardized":
        standard_loss = F.smooth_l1_loss(output_model, target_model, reduction="none")
        localisation_loss = (standard_loss * task_weights.view(1, -1)).mean()
        task_terms = torch.abs(raw_prediction - target_raw) / corrected_scales
    elif loss_mode == "corrected_compound":
        task_terms = torch.abs(raw_prediction - target_raw) / corrected_scales
        localisation_loss = (task_terms * task_weights.view(1, -1)).mean()
    elif loss_mode == "corrected_uncertainty":
        if uncertainty_module is None:
            raise ValueError("corrected_uncertainty requires an uncertainty module.")
        task_terms = (torch.abs(raw_prediction - target_raw) / corrected_scales).mean(dim=0)
        log_sigma = uncertainty_module.log_sigma.clamp(-3.0, 2.0)
        localisation_loss = torch.sum(torch.exp(-2.0 * log_sigma) * task_terms + log_sigma)
        task_terms = torch.abs(raw_prediction - target_raw) / corrected_scales
    else:
        raise ValueError(f"Unsupported improved loss mode '{loss_mode}'.")

    loss = localisation_loss + spike_weight * spike_penalty
    summary = {
        "distance_loss": float(task_terms[:, 0].mean().item()),
        "azimuth_loss": float(task_terms[:, 1].mean().item()),
        "elevation_loss": float(task_terms[:, 2].mean().item()),
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
    return loss, summary


def _instantiate_improved_model(data: Any, spec: ImprovedExperimentSpec) -> nn.Module:
    params = data.params
    branch_hidden_dim = int(params["branch_hidden_dim"])

    if spec.variant == "baseline":
        encoder = HandcraftedBranchEncoder(
            distance_dim=data.train_batch.pathway.distance.shape[-1],
            azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
            elevation_dim=data.train_batch.pathway.elevation.shape[-1],
            branch_hidden_dim=branch_hidden_dim,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=3,
            num_steps=int(params["num_steps"]),
            beta=float(params["membrane_beta"]),
            threshold=float(params["fusion_threshold"]),
            reset_mechanism=str(params["reset_mechanism"]),
        ).to(data.train_targets_raw.device)

    if spec.variant == "residual_elevation":
        encoder = ResidualElevationEncoder(
            distance_dim=data.train_batch.pathway.distance.shape[-1],
            azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
            elevation_dim=data.train_batch.pathway.elevation.shape[-1],
            branch_hidden_dim=branch_hidden_dim,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=3,
            num_steps=int(params["num_steps"]),
            beta=float(params["membrane_beta"]),
            threshold=float(params["fusion_threshold"]),
            reset_mechanism=str(params["reset_mechanism"]),
        ).to(data.train_targets_raw.device)

    if spec.variant == "distance_resonance":
        encoder = HandcraftedBranchEncoder(
            distance_dim=data.train_batch.pathway.distance.shape[-1],
            azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
            elevation_dim=data.train_batch.pathway.elevation.shape[-1],
            branch_hidden_dim=branch_hidden_dim,
        )
        return DistanceResonanceModel(
            encoder=encoder,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=3,
            num_steps=int(params["num_steps"]),
            beta=float(params["membrane_beta"]),
            threshold=float(params["fusion_threshold"]),
            reset_mechanism=str(params["reset_mechanism"]),
        ).to(data.train_targets_raw.device)

    if spec.variant == "elevation_sconv_residual":
        encoder = ElevationSConvResidualEncoder(
            distance_dim=data.train_batch.pathway.distance.shape[-1],
            azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
            elevation_dim=data.train_batch.pathway.elevation.shape[-1],
            branch_hidden_dim=branch_hidden_dim,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=3,
            num_steps=int(params["num_steps"]),
            beta=float(params["membrane_beta"]),
            threshold=float(params["fusion_threshold"]),
            reset_mechanism=str(params["reset_mechanism"]),
        ).to(data.train_targets_raw.device)

    if spec.variant == "combined_residual_elevation":
        encoder = CombinedElevationEncoder(
            distance_dim=data.train_batch.pathway.distance.shape[-1],
            azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
            elevation_dim=data.train_batch.pathway.elevation.shape[-1],
            branch_hidden_dim=branch_hidden_dim,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=3,
            num_steps=int(params["num_steps"]),
            beta=float(params["membrane_beta"]),
            threshold=float(params["fusion_threshold"]),
            reset_mechanism=str(params["reset_mechanism"]),
        ).to(data.train_targets_raw.device)

    raise ValueError(f"Unsupported improved variant '{spec.variant}'.")


def _train_improved_model(
    model: nn.Module,
    data: Any,
    target_bundle: TargetBundle,
    spec: ImprovedExperimentSpec,
) -> tuple[ExperimentTrainingResult, TaskUncertaintyWeights | None]:
    params = data.params
    task_weights = torch.tensor(
        [1.0, float(params["angle_weight"]), float(params["elevation_weight"])],
        device=data.train_targets_raw.device,
    )
    uncertainty_module = TaskUncertaintyWeights().to(data.train_targets_raw.device) if spec.loss_mode == "corrected_uncertainty" else None
    if uncertainty_module is not None:
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
    epochs = int(spec.training_overrides.get("epochs", int(params["epochs"])))
    spike_weight = float(params["loss_weighting"]) * float(spec.training_overrides.get("spike_weight_scale", 1.0))
    uncertainty_warmup_epochs = int(spec.training_overrides.get("uncertainty_warmup_epochs", 0))

    trainables = list(model.parameters()) + ([] if uncertainty_module is None else list(uncertainty_module.parameters()))
    optimizer = torch.optim.Adam(trainables, lr=learning_rate)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_combined_history: list[float] = []
    best_epoch = 0
    best_val_combined = float("inf")
    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    best_auxiliary_state = None if uncertainty_module is None else copy.deepcopy(uncertainty_module.state_dict())
    final_diagnostics: dict[str, torch.Tensor] = {}

    def batch_iterator_with_raw():
        permutation = torch.randperm(target_bundle.train_model.shape[0], device=target_bundle.train_model.device)
        for start in range(0, target_bundle.train_model.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            yield (
                data.train_batch.index_select(indices),
                target_bundle.train_model[indices],
                data.train_targets_raw[indices],
            )

    for epoch in range(epochs):
        model.train()
        if uncertainty_module is not None:
            uncertainty_module.train()
            requires_grad = epoch >= uncertainty_warmup_epochs
            uncertainty_module.log_sigma.requires_grad_(requires_grad)

        batch_losses = []
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
            val_prediction = _decode_model_output(val_output_model, target_bundle)
            val_eval = _prediction_metrics(data.local_config, val_prediction, data.val_targets_raw, val_diagnostics)

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


def _evaluate_improved_model(
    model: nn.Module,
    batch: Any,
    targets_raw: torch.Tensor,
    target_bundle: TargetBundle,
    local_config: Any,
) -> ExperimentEvaluation:
    model.eval()
    with torch.no_grad():
        output_model, diagnostics = model(batch)
        raw_prediction = _decode_model_output(output_model, target_bundle)
    evaluation = _prediction_metrics(local_config, raw_prediction, targets_raw, diagnostics)
    evaluation.metrics.update(
        {
            "predicted_distance_std": float(raw_prediction[:, 0].std().item()),
            "predicted_azimuth_std": float(raw_prediction[:, 1].std().item()),
            "predicted_elevation_std": float(raw_prediction[:, 2].std().item()),
            "target_distance_std": float(targets_raw[:, 0].std().item()),
            "target_azimuth_std": float(targets_raw[:, 1].std().item()),
            "target_elevation_std": float(targets_raw[:, 2].std().item()),
        }
    )
    return evaluation


def _is_accepted(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    tolerance = 1e-6
    combined_improved = float(candidate["combined_error"]) < float(baseline["combined_error"]) - tolerance
    any_metric_improved = any(
        float(candidate[key]) < float(baseline[key]) - tolerance
        for key in ("distance_mae_m", "azimuth_mae_deg", "elevation_mae_deg")
    )
    return combined_improved and any_metric_improved


def _load_previous_results(outputs_root: Path) -> dict[str, dict[str, Any]]:
    previous_path = outputs_root / "experiments" / "results.json"
    if not previous_path.exists():
        return {}
    with previous_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(item["name"]): item for item in payload.get("experiments", [])}


def _save_improved_outputs(
    stage_root: Path,
    spec: ImprovedExperimentSpec,
    train_result: ExperimentTrainingResult,
    test_eval: ExperimentEvaluation,
    baseline_metrics: dict[str, Any],
    model: nn.Module,
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
            "Baseline": [
                float(baseline_metrics["distance_mae_m"]),
                float(baseline_metrics["azimuth_mae_deg"]),
                float(baseline_metrics["elevation_mae_deg"]),
                float(baseline_metrics["combined_error"]),
            ],
            "Improved": [
                float(test_eval.metrics["distance_mae_m"]),
                float(test_eval.metrics["azimuth_mae_deg"]),
                float(test_eval.metrics["elevation_mae_deg"]),
                float(test_eval.metrics["combined_error"]),
            ],
        },
        stage_dir / "comparison.png",
        f"{spec.title} vs Baseline",
        ylabel="Error",
    )
    save_text_figure(
        [
            f"combined_error: {test_eval.metrics['combined_error']:.4f}",
            f"distance_mae_m: {test_eval.metrics['distance_mae_m']:.4f}",
            f"azimuth_mae_deg: {test_eval.metrics['azimuth_mae_deg']:.4f}",
            f"elevation_mae_deg: {test_eval.metrics['elevation_mae_deg']:.4f}",
            f"spike_rate: {test_eval.metrics['mean_spike_rate']:.4f}",
            f"predicted_azimuth_std: {test_eval.metrics['predicted_azimuth_std']:.4f}",
            f"predicted_elevation_std: {test_eval.metrics['predicted_elevation_std']:.4f}",
            f"best_epoch: {train_result.best_epoch}",
        ],
        stage_dir / "summary.png",
        f"{spec.title} Summary",
    )

    diagnostics = test_eval.diagnostics
    if "elevation_map" in diagnostics:
        save_heatmap(
            diagnostics["elevation_map"][0].flatten(0, 1),
            stage_dir / "elevation_map.png",
            f"{spec.title} Elevation Map",
            xlabel="Time",
            ylabel="Feature",
        )
    if "distance_resonance_spikes" in diagnostics:
        save_heatmap(
            diagnostics["distance_resonance_spikes"][0].T,
            stage_dir / "distance_resonance_spikes.png",
            f"{spec.title} Distance Resonance Spikes",
            xlabel="Pseudo-Time",
            ylabel="Neuron",
        )
    if "elevation_sconv_spikes" in diagnostics:
        sconv_matrix = diagnostics["elevation_sconv_spikes"][0].permute(1, 2, 3, 0).reshape(
            -1, diagnostics["elevation_sconv_spikes"].shape[1]
        )
        save_heatmap(
            sconv_matrix,
            stage_dir / "elevation_sconv_spikes.png",
            f"{spec.title} Elevation SConv Spikes",
            xlabel="Pseudo-Time",
            ylabel="Unit",
        )

    return {
        "loss": str(stage_dir / "loss.png"),
        "test_distance_prediction": str(stage_dir / "test_distance_prediction.png"),
        "test_azimuth_prediction": str(stage_dir / "test_azimuth_prediction.png"),
        "test_elevation_prediction": str(stage_dir / "test_elevation_prediction.png"),
        "test_elevation_error": str(stage_dir / "test_elevation_error.png"),
        "comparison": str(stage_dir / "comparison.png"),
        "summary": str(stage_dir / "summary.png"),
    }


def _overall_plots(stage_root: Path, baseline_metrics: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, str]:
    labels = ["Baseline"] + [item["title"].replace("Improved ", "") for item in results]
    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error"],
        {
            label: values
            for label, values in zip(
                labels,
                [
                    [
                        float(baseline_metrics["distance_mae_m"]),
                        float(baseline_metrics["azimuth_mae_deg"]),
                        float(baseline_metrics["elevation_mae_deg"]),
                        float(baseline_metrics["combined_error"]),
                    ]
                ]
                + [
                    [
                        float(item["test_metrics"]["distance_mae_m"]),
                        float(item["test_metrics"]["azimuth_mae_deg"]),
                        float(item["test_metrics"]["elevation_mae_deg"]),
                        float(item["test_metrics"]["combined_error"]),
                    ]
                    for item in results
                ],
            )
        },
        stage_root / "overall_test_comparison.png",
        "Improved Experiments vs Baseline",
        ylabel="Error",
    )
    save_grouped_bar_chart(
        labels,
        {
            "Predicted Azimuth Std": [float(baseline_metrics["predicted_azimuth_std"])]
            + [float(item["test_metrics"]["predicted_azimuth_std"]) for item in results],
            "Predicted Elevation Std": [float(baseline_metrics["predicted_elevation_std"])]
            + [float(item["test_metrics"]["predicted_elevation_std"]) for item in results],
        },
        stage_root / "prediction_spread_comparison.png",
        "Improved Experiment Prediction Spread",
        ylabel="Std",
    )
    return {
        "overall_test_comparison": str(stage_root / "overall_test_comparison.png"),
        "prediction_spread_comparison": str(stage_root / "prediction_spread_comparison.png"),
    }


def _write_improved_report(
    outputs_root: Path,
    dataset_mode: str,
    baseline_label: str,
    baseline_metrics: dict[str, Any],
    results: list[dict[str, Any]],
    previous_results: dict[str, dict[str, Any]],
) -> Path:
    lines = [
        "# Improved Experimental Pipeline",
        "",
        f"- Dataset mode: `{dataset_mode}`",
        f"- Fixed baseline reference for every experiment: `{baseline_label}`",
        "",
        "## Protocol",
        "",
        "Every improved experiment was compared directly against the same baseline control rather than against the previous improved experiment. Acceptance therefore means that the modified variant beat the fixed baseline on combined error and on at least one individual metric.",
        "",
        "## Experiment 0 Control",
        "",
        f"- Combined error: `{baseline_metrics['combined_error']:.4f}`",
        f"- Distance MAE: `{baseline_metrics['distance_mae_m']:.4f} m`",
        f"- Azimuth MAE: `{baseline_metrics['azimuth_mae_deg']:.4f} deg`",
        f"- Elevation MAE: `{baseline_metrics['elevation_mae_deg']:.4f} deg`",
        f"- Predicted azimuth std: `{baseline_metrics['predicted_azimuth_std']:.4f}`",
        f"- Predicted elevation std: `{baseline_metrics['predicted_elevation_std']:.4f}`",
        "",
        "![Improved baseline comparison](improved_experiments/overall_test_comparison.png)",
        "![Prediction spread comparison](improved_experiments/prediction_spread_comparison.png)",
        "",
        "## Results Table",
        "",
        "| Experiment | Combined Error | Distance MAE | Azimuth MAE | Elevation MAE | Accepted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['title']} | {item['test_metrics']['combined_error']:.4f} | "
            f"{item['test_metrics']['distance_mae_m']:.4f} | {item['test_metrics']['azimuth_mae_deg']:.4f} | "
            f"{item['test_metrics']['elevation_mae_deg']:.4f} | {'Yes' if item['accepted'] else 'No'} |"
        )

    lines.extend(["", "## Detailed Analysis", ""])

    previous_name_map = {
        "improved_experiment_1_learned_features": "experiment_1_learned_features",
        "improved_experiment_2_compound_loss": "experiment_2_compound_loss",
        "improved_experiment_3_uncertainty_weighting": "experiment_3_uncertainty_weighting",
        "improved_experiment_4_resonant_neurons": "experiment_4_resonant_neurons",
        "improved_experiment_5_sconv2dlstm": "experiment_5_sconv2dlstm",
    }

    for item in results:
        previous_metrics = previous_results.get(previous_name_map[item["name"]], {})
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Change: {item['description']}",
                f"- Rationale: {item['rationale']}",
                f"- Decision against fixed baseline: `{'ACCEPTED' if item['accepted'] else 'REJECTED'}`",
                f"- Combined error: `{item['test_metrics']['combined_error']:.4f}`",
                f"- Distance MAE: `{item['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Azimuth MAE: `{item['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Elevation MAE: `{item['test_metrics']['elevation_mae_deg']:.4f} deg`",
                f"- Predicted azimuth std: `{item['test_metrics']['predicted_azimuth_std']:.4f}`",
                f"- Predicted elevation std: `{item['test_metrics']['predicted_elevation_std']:.4f}`",
                f"- Delta vs fixed baseline: `{item['comparison']['combined_error_delta']:.4f}`",
            ]
        )
        if previous_metrics:
            lines.append(
                f"- Delta vs original failed version: `{item['test_metrics']['combined_error'] - previous_metrics['test_metrics']['combined_error']:.4f}`"
            )
        lines.extend(["", "Implemented steps:"])
        lines.extend([f"- {step}" for step in item["implemented_steps"]])
        lines.extend(["", "Remaining follow-up steps:"])
        lines.extend([f"- {step}" for step in item["remaining_steps"]])
        lines.extend(
            [
                "",
                f"![{item['title']} loss](improved_experiments/{item['name']}/loss.png)",
                f"![{item['title']} comparison](improved_experiments/{item['name']}/comparison.png)",
                f"![{item['title']} azimuth](improved_experiments/{item['name']}/test_azimuth_prediction.png)",
                f"![{item['title']} elevation](improved_experiments/{item['name']}/test_elevation_prediction.png)",
                "",
            ]
        )

    accepted = [item["title"] for item in results if item["accepted"]]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Accepted improved experiments: {', '.join(accepted) if accepted else 'none'}",
            "- Because every comparison used the same fixed baseline, the acceptance decisions are directly comparable across experiments.",
            "- If an experiment improved on its previous failed version but still lost to baseline, it should still be treated as a rejected baseline replacement and a useful partial fix only.",
        ]
    )

    report_path = outputs_root / "improved_experiments_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_improved_experiments(context: StageContext) -> dict[str, Any]:
    dataset_mode = os.environ.get("RADAR_SNN_IMPROVED_EXPERIMENT_DATASET_MODE", "dev")
    params, baseline_label = _baseline_reference_params(context)
    print(f"[improved_experiments] preparing dataset mode={dataset_mode}", flush=True)
    data = _prepare_experiment_data(context, params, dataset_mode)
    target_bundle = _prepare_target_bundle(data)

    stage_root = context.outputs.root / "improved_experiments"
    stage_root.mkdir(parents=True, exist_ok=True)

    print("[improved_experiments] evaluating fixed baseline", flush=True)
    baseline_evaluation = _evaluate_dataset_bundle(
        context,
        params,
        _prepare_dataset_bundle(context, dataset_mode),
        include_artifacts=False,
    )
    baseline_metrics = {
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

    previous_results = _load_previous_results(context.outputs.root)
    experiment_summaries: list[dict[str, Any]] = []

    for index, spec in enumerate(_improved_experiment_specs(), start=1):
        print(f"[improved_experiments] running {spec.name} ({index}/5)", flush=True)
        model = _instantiate_improved_model(data, spec)
        train_result, uncertainty_module = _train_improved_model(model, data, target_bundle, spec)
        model.load_state_dict(train_result.best_state)
        if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
            uncertainty_module.load_state_dict(train_result.best_auxiliary_state)

        val_eval = _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
        test_eval = _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
        accepted = _is_accepted(test_eval.metrics, baseline_metrics)
        artifacts = _save_improved_outputs(stage_root, spec, train_result, test_eval, baseline_metrics, model)

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
                    "learning_rate": float(data.params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0)),
                    "batch_size": int(spec.training_overrides.get("batch_size", int(data.params["batch_size"]))),
                    "epochs": int(spec.training_overrides.get("epochs", int(data.params["epochs"]))),
                },
                "val_metrics": val_eval.metrics,
                "test_metrics": test_eval.metrics,
                "comparison": _metrics_delta(test_eval.metrics, baseline_metrics),
                "artifacts": artifacts,
            }
        )

    overall_artifacts = _overall_plots(stage_root, baseline_metrics, experiment_summaries)
    report_path = _write_improved_report(
        context.outputs.root,
        dataset_mode,
        baseline_label,
        baseline_metrics,
        experiment_summaries,
        previous_results,
    )

    payload = {
        "dataset_mode": dataset_mode,
        "baseline_label": baseline_label,
        "baseline_metrics": {key: format_float(value) for key, value in baseline_metrics.items()},
        "baseline_artifacts": baseline_artifacts,
        "experiments": [
            {
                **{key: value for key, value in item.items() if key not in {"val_metrics", "test_metrics", "comparison"}},
                "val_metrics": {key: format_float(value) for key, value in item["val_metrics"].items()},
                "test_metrics": {key: format_float(value) for key, value in item["test_metrics"].items()},
                "comparison": {key: format_float(value) for key, value in item["comparison"].items()},
            }
            for item in experiment_summaries
        ],
        "overall_artifacts": overall_artifacts,
        "report_path": str(report_path),
    }
    save_json(stage_root / "results.json", payload)
    return payload
