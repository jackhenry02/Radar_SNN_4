from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.experimental_variants import CombinedElevationEncoder, ExperimentalPathwayModel
from models.round2_variants import (
    AdaptiveResidualCombinedEncoder,
    FusionResonanceAugmentedModel,
    PerPathwayResonanceEncoder,
    PostPathwayResidualEncoder,
    PrePathwayResidualEncoder,
)
from stages.base import StageContext
from stages.combined_experiment import _save_coordinate_error_profiles, _save_prediction_cache
from stages.experiments import _baseline_reference_params, _metrics_delta, _prepare_experiment_data
from stages.improved_experiments import (
    TargetBundle,
    _decode_model_output,
    _evaluate_improved_model,
    _is_accepted,
    _prepare_target_bundle,
    _save_improved_outputs,
)
from stages.improvement import _distance_candidates, _itd_candidates
from stages.training_improved_experiments import EnhancedPathwayTrainingResult, EnhancedTrainingConfig
from utils.common import (
    degrees_to_radians,
    format_float,
    save_grouped_bar_chart,
    save_heatmap,
    save_json,
    save_prediction_scatter,
    seed_everything,
)


@dataclass
class Round2ExperimentSpec:
    name: str
    title: str
    description: str
    rationale: str
    implemented_steps: list[str]
    analysis_focus: list[str]
    variant: str
    loss_mode: str
    training_overrides: dict[str, Any]


def _round2_specs() -> list[Round2ExperimentSpec]:
    return [
        Round2ExperimentSpec(
            name="round2_experiment_1_adaptive_filters_delays",
            title="Round 2 Experiment 1: Adaptive Delay And Spectral Filters",
            description=(
                "Keep the accepted combined model intact and add constrained trainable delay offsets, delay weights, "
                "and spectral-channel gains/offsets as small residual corrections."
            ),
            rationale=(
                "This tests the recommendation to learn small deviations around the fixed biologically motivated "
                "delays and spectral channels rather than replacing them with fully free filters."
            ),
            implemented_steps=[
                "Use the combined model as the base path.",
                "Add learnable offset-and-gain transforms to the fixed distance, ITD, and spectral feature axes.",
                "Keep all residual gains small at initialization so the fixed cue geometry still dominates early training.",
                "Log the learned offsets and gains so they can be compared directly against the original fixed values.",
            ],
            analysis_focus=[
                "Whether learned delay offsets stay near zero or move substantially.",
                "Whether spectral channel gains become strongly non-uniform.",
                "Whether the constrained adaptive version improves on the saved short-data combined baseline.",
            ],
            variant="adaptive_filters_delays",
            loss_mode="corrected_uncertainty",
            training_overrides={"learning_rate_scale": 0.9, "batch_size": 16, "uncertainty_warmup_epochs": 4},
        ),
        Round2ExperimentSpec(
            name="round2_experiment_2a_resonance_fusion",
            title="Round 2 Experiment 2A: Resonant Branch At Fusion",
            description=(
                "Add a corollary-discharge resonant branch in parallel with the current distance/azimuth/elevation "
                "latents and fuse it only at the final fusion stage."
            ),
            rationale=(
                "This is the least invasive resonance test: the existing pathways stay unchanged while the SNN gets an "
                "extra resonant timing summary built from negative transmit drive plus positive echo drive."
            ),
            implemented_steps=[
                "Build signed resonance drives using positive receive activity and negative transmit activity.",
                "Run the signed drives through a learnable bank of damped oscillatory resonators.",
                "Project the pooled resonant spikes into one extra latent block and concatenate that at fusion.",
            ],
            analysis_focus=[
                "Whether the resonant branch improves distance or angular metrics without destabilising the existing latents.",
                "Whether the learned resonance frequencies cluster in a useful range rather than saturating.",
            ],
            variant="resonance_fusion",
            loss_mode="corrected_uncertainty",
            training_overrides={"learning_rate_scale": 0.85, "batch_size": 12, "uncertainty_warmup_epochs": 4},
        ),
        Round2ExperimentSpec(
            name="round2_experiment_2b_resonance_per_pathway",
            title="Round 2 Experiment 2B: Resonant Branch Per Pathway",
            description=(
                "Add one shared resonant bank but project its output separately into residual corrections for the "
                "distance, azimuth, and elevation pathways."
            ),
            rationale=(
                "This tests whether the resonance features are more useful when each pathway can consume them with its "
                "own learned projection rather than only at the final fusion layer."
            ),
            implemented_steps=[
                "Reuse the same signed resonant input construction as Experiment 2A.",
                "Project the pooled resonant spikes separately into distance, azimuth, and elevation residuals.",
                "Inject the resonant pathway residuals with small learned gains.",
            ],
            analysis_focus=[
                "Whether pathway-specific resonance routing is better than fusion-only resonance.",
                "Whether any one pathway benefits disproportionately from the resonant information.",
            ],
            variant="resonance_per_pathway",
            loss_mode="corrected_uncertainty",
            training_overrides={"learning_rate_scale": 0.85, "batch_size": 16, "uncertainty_warmup_epochs": 4},
        ),
        Round2ExperimentSpec(
            name="round2_experiment_3_pre_pathway_lif",
            title="Round 2 Experiment 3: Pre-Pathway LIF Residual",
            description=(
                "Insert an extra learnable LIF preprocessing stage on the spike trains before rebuilding the fixed "
                "pathway features, then add the resulting features back as small residuals."
            ),
            rationale=(
                "This tests whether another spiking preprocessing step can improve feature extraction without replacing "
                "the strong handcrafted cue pathways."
            ),
            implemented_steps=[
                "Process transmit and receive spike trains with learned 1x1 channel mixing and LIF neurons.",
                "Rebuild distance, azimuth, and elevation pathway features from the processed spikes.",
                "Project the rebuilt features into residual pathway corrections rather than replacing the baseline pathways.",
            ],
            analysis_focus=[
                "Whether early spiking preprocessing sharpens the pathways or blurs their timing structure.",
                "Whether the learned residual stays small or dominates the baseline cues.",
            ],
            variant="pre_pathway_lif",
            loss_mode="corrected_uncertainty",
            training_overrides={"learning_rate_scale": 0.8, "batch_size": 12, "uncertainty_warmup_epochs": 4},
        ),
        Round2ExperimentSpec(
            name="round2_experiment_4_post_pathway_lif",
            title="Round 2 Experiment 4: Post-Pathway LIF Residual",
            description=(
                "Add one extra LIF processing block on each pathway latent after the current branch encoders and before "
                "fusion so the model can do deeper branch-specific processing."
            ),
            rationale=(
                "This is the safer depth experiment because it preserves the current pathway feature extraction and only "
                "adds extra processing after those cues already exist."
            ),
            implemented_steps=[
                "Keep the accepted combined encoder untouched.",
                "Add a branch-specific linear-plus-LIF residual block after each pathway latent.",
                "Feed the updated latents into the existing fusion SNN head.",
            ],
            analysis_focus=[
                "Whether extra branch-specific processing helps without flattening the learned representation.",
                "Whether post-pathway depth is safer than pre-pathway depth on the short run.",
            ],
            variant="post_pathway_lif",
            loss_mode="corrected_uncertainty",
            training_overrides={"learning_rate_scale": 0.9, "batch_size": 16, "uncertainty_warmup_epochs": 4},
        ),
        Round2ExperimentSpec(
            name="round2_experiment_5a_cartesian_loss",
            title="Round 2 Experiment 5A: Pure Cartesian Loss",
            description=(
                "Keep the accepted combined architecture fixed and train it with a pure Cartesian-position loss plus "
                "the usual spike penalty."
            ),
            rationale=(
                "This tests whether optimizing directly for physical position improves localization in Euclidean space "
                "even if the polar-coordinate metrics shift differently."
            ),
            implemented_steps=[
                "Decode the model output to polar coordinates as usual.",
                "Convert both prediction and target to Cartesian coordinates.",
                "Optimize only the normalized Cartesian error and the spike penalty.",
                "Still report both Cartesian and polar metrics at evaluation time.",
            ],
            analysis_focus=[
                "Whether Euclidean position error improves relative to the short combined baseline.",
                "Whether pure Cartesian optimization hurts the angular metrics even if position improves.",
            ],
            variant="combined_baseline",
            loss_mode="pure_cartesian",
            training_overrides={"learning_rate_scale": 0.85, "batch_size": 16},
        ),
        Round2ExperimentSpec(
            name="round2_experiment_5b_mixed_cartesian_loss",
            title="Round 2 Experiment 5B: Mixed Cartesian And Polar Loss",
            description=(
                "Keep the accepted combined architecture fixed and train with a mixed loss made of Cartesian error, "
                "polar error regularization, and the spike penalty."
            ),
            rationale=(
                "This checks whether a mixed objective can improve physical position without giving up the stable "
                "distance, azimuth, and elevation behavior learned by the current polar formulation."
            ),
            implemented_steps=[
                "Compute the same normalized Cartesian error as Experiment 5A.",
                "Add a smaller corrected polar loss term as a regularizer.",
                "Keep all evaluation outputs in both coordinate systems for a fair comparison to the pure Cartesian run.",
            ],
            analysis_focus=[
                "Whether mixed optimization is a safer compromise than pure Cartesian optimization.",
                "Whether the mixed loss retains the baseline angular behavior better than the pure Cartesian loss.",
            ],
            variant="combined_baseline",
            loss_mode="mixed_cartesian",
            training_overrides={"learning_rate_scale": 0.85, "batch_size": 16, "cartesian_mix_weight": 0.5},
        ),
    ]


def _polar_to_cartesian(
    distance_m: torch.Tensor,
    azimuth_deg: torch.Tensor,
    elevation_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    azimuth = degrees_to_radians(azimuth_deg)
    elevation = degrees_to_radians(elevation_deg)
    x = distance_m * torch.cos(elevation) * torch.cos(azimuth)
    y = distance_m * torch.cos(elevation) * torch.sin(azimuth)
    z = distance_m * torch.sin(elevation)
    return x, y, z


def _cartesian_metrics_from_predictions(
    predicted_distance: torch.Tensor,
    predicted_azimuth: torch.Tensor,
    predicted_elevation: torch.Tensor,
    target_distance: torch.Tensor,
    target_azimuth: torch.Tensor,
    target_elevation: torch.Tensor,
) -> dict[str, float]:
    pred_x, pred_y, pred_z = _polar_to_cartesian(predicted_distance, predicted_azimuth, predicted_elevation)
    true_x, true_y, true_z = _polar_to_cartesian(target_distance, target_azimuth, target_elevation)
    euclidean = torch.sqrt((pred_x - true_x).square() + (pred_y - true_y).square() + (pred_z - true_z).square())
    return {
        "x_mae_m": float(torch.mean(torch.abs(pred_x - true_x)).item()),
        "y_mae_m": float(torch.mean(torch.abs(pred_y - true_y)).item()),
        "z_mae_m": float(torch.mean(torch.abs(pred_z - true_z)).item()),
        "euclidean_error_m": float(torch.mean(euclidean).item()),
    }


def _load_short_combined_baseline(config: Any, outputs: Any) -> dict[str, Any]:
    result_path = outputs.root / "combined_experiment" / "short_data_1000_result.json"
    if not result_path.exists():
        from stages.combined_experiment import run_combined_small_data_test

        run_combined_small_data_test(config, outputs)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    cache_path = Path(payload["artifacts"]["prediction_cache"])
    predictions = np.load(cache_path)
    baseline_cartesian = _cartesian_metrics_from_predictions(
        torch.from_numpy(predictions["predicted_distance"]),
        torch.from_numpy(predictions["predicted_azimuth"]),
        torch.from_numpy(predictions["predicted_elevation"]),
        torch.from_numpy(predictions["target_distance"]),
        torch.from_numpy(predictions["target_azimuth"]),
        torch.from_numpy(predictions["target_elevation"]),
    )
    metrics = {**payload["test_metrics"], **baseline_cartesian}
    return {
        "result": payload,
        "metrics": metrics,
        "prediction_cache": str(cache_path),
    }


def _augment_with_cartesian_metrics(evaluation: Any) -> Any:
    cartesian = _cartesian_metrics_from_predictions(
        evaluation.predictions["predicted_distance"],
        evaluation.predictions["predicted_azimuth"],
        evaluation.predictions["predicted_elevation"],
        evaluation.predictions["target_distance"],
        evaluation.predictions["target_azimuth"],
        evaluation.predictions["target_elevation"],
    )
    evaluation.metrics.update(cartesian)
    return evaluation


def _round2_loss_components(
    output_model: torch.Tensor,
    target_model: torch.Tensor,
    target_raw: torch.Tensor,
    target_bundle: TargetBundle,
    diagnostics: dict[str, torch.Tensor],
    local_config: Any,
    loss_mode: str,
    task_weights: torch.Tensor,
    spike_weight: float,
    uncertainty_module: nn.Module | None,
    *,
    cartesian_mix_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_prediction = _decode_model_output(output_model, target_bundle)
    polar_scales = raw_prediction.new_tensor([local_config.max_range_m, 45.0, 30.0]).view(1, -1)
    spike_penalty = diagnostics["spike_rate"].mean()

    polar_terms = torch.abs(raw_prediction - target_raw) / polar_scales
    pred_x, pred_y, pred_z = _polar_to_cartesian(raw_prediction[:, 0], raw_prediction[:, 1], raw_prediction[:, 2])
    true_x, true_y, true_z = _polar_to_cartesian(target_raw[:, 0], target_raw[:, 1], target_raw[:, 2])
    cartesian_prediction = torch.stack([pred_x, pred_y, pred_z], dim=-1)
    cartesian_target = torch.stack([true_x, true_y, true_z], dim=-1)
    cartesian_scales = cartesian_prediction.new_full((1, 3), float(local_config.max_range_m))
    cartesian_terms = torch.abs(cartesian_prediction - cartesian_target) / cartesian_scales

    if loss_mode == "corrected_uncertainty":
        if uncertainty_module is None:
            raise ValueError("corrected_uncertainty requires an uncertainty module.")
        task_term_mean = polar_terms.mean(dim=0)
        log_sigma = uncertainty_module.log_sigma.clamp(-3.0, 2.0)
        localisation_loss = torch.sum(torch.exp(-2.0 * log_sigma) * task_term_mean + log_sigma)
    elif loss_mode == "pure_cartesian":
        localisation_loss = cartesian_terms.mean()
    elif loss_mode == "mixed_cartesian":
        localisation_loss = cartesian_terms.mean() + cartesian_mix_weight * (polar_terms * task_weights.view(1, -1)).mean()
    else:
        raise ValueError(f"Unsupported round-2 loss mode '{loss_mode}'.")

    loss = localisation_loss + spike_weight * spike_penalty
    summary = {
        "distance_loss": float(polar_terms[:, 0].mean().item()),
        "azimuth_loss": float(polar_terms[:, 1].mean().item()),
        "elevation_loss": float(polar_terms[:, 2].mean().item()),
        "x_loss": float(cartesian_terms[:, 0].mean().item()),
        "y_loss": float(cartesian_terms[:, 1].mean().item()),
        "z_loss": float(cartesian_terms[:, 2].mean().item()),
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


def _instantiate_round2_model(data: Any, spec: Round2ExperimentSpec) -> nn.Module:
    params = data.params
    branch_hidden_dim = int(params["branch_hidden_dim"])
    hidden_dim = int(params["hidden_dim"])
    num_steps = int(params["num_steps"])
    beta = float(params["membrane_beta"])
    threshold = float(params["fusion_threshold"])
    reset_mechanism = str(params["reset_mechanism"])
    distance_dim = data.train_batch.pathway.distance.shape[-1]
    azimuth_dim = data.train_batch.pathway.azimuth.shape[-1]
    elevation_dim = data.train_batch.pathway.elevation.shape[-1]
    num_frequency_channels = int(params["num_frequency_channels"])
    num_delay_lines = int(params["num_delay_lines"])

    if spec.variant == "combined_baseline":
        encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        ).to(data.train_targets_raw.device)

    if spec.variant == "adaptive_filters_delays":
        encoder = AdaptiveResidualCombinedEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
            num_frequency_channels=num_frequency_channels,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        ).to(data.train_targets_raw.device)

    if spec.variant == "resonance_fusion":
        return FusionResonanceAugmentedModel(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            num_frequency_channels=num_frequency_channels,
        ).to(data.train_targets_raw.device)

    if spec.variant == "resonance_per_pathway":
        encoder = PerPathwayResonanceEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
            num_frequency_channels=num_frequency_channels,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        ).to(data.train_targets_raw.device)

    if spec.variant == "pre_pathway_lif":
        distance_candidates = _distance_candidates(data.local_config, data.train_targets_raw.device, num_delay_lines)
        itd_candidates = _itd_candidates(data.local_config, data.train_targets_raw.device, num_delay_lines)
        encoder = PrePathwayResidualEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
            num_frequency_channels=num_frequency_channels,
            num_delay_lines=num_delay_lines,
            distance_candidates=distance_candidates,
            itd_candidates=itd_candidates,
            beta=0.92,
            threshold=0.75,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        ).to(data.train_targets_raw.device)

    if spec.variant == "post_pathway_lif":
        encoder = PostPathwayResidualEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
            beta=beta,
            threshold=0.85 * threshold,
            num_steps=num_steps,
        )
        return ExperimentalPathwayModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        ).to(data.train_targets_raw.device)

    raise ValueError(f"Unsupported round-2 variant '{spec.variant}'.")


def _train_round2_model(
    model: nn.Module,
    data: Any,
    target_bundle: TargetBundle,
    spec: Round2ExperimentSpec,
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
    batch_size = int(spec.training_overrides.get("batch_size", 16))
    spike_weight = float(params["loss_weighting"]) * float(spec.training_overrides.get("spike_weight_scale", 1.0))
    uncertainty_warmup_epochs = int(spec.training_overrides.get("uncertainty_warmup_epochs", 0))
    cartesian_mix_weight = float(spec.training_overrides.get("cartesian_mix_weight", 0.5))

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

    def batch_iterator():
        permutation = torch.randperm(target_bundle.train_model.shape[0], device=target_bundle.train_model.device)
        for start in range(0, target_bundle.train_model.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            yield data.train_batch.index_select(indices), target_bundle.train_model[indices], data.train_targets_raw[indices]

    for epoch in range(training_config.max_epochs):
        model.train()
        if uncertainty_module is not None:
            uncertainty_module.train()
            uncertainty_module.log_sigma.requires_grad_(epoch >= uncertainty_warmup_epochs)
        batch_losses: list[float] = []

        for batch_features, batch_targets_model, batch_targets_raw in batch_iterator():
            optimizer.zero_grad(set_to_none=True)
            output_model, diagnostics = model(batch_features)
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
            val_eval = _augment_with_cartesian_metrics(
                _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
            )

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
            diagnostics=best_diagnostics,
            stopped_early=stopped_early,
            executed_epochs=len(train_loss_history),
            best_auxiliary_state=best_auxiliary_state,
        ),
        uncertainty_module,
    )


def _save_line_plot(series: dict[str, np.ndarray], path: Path, title: str, ylabel: str, baseline: float | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in series.items():
        ax.plot(values, linewidth=2.0, label=label)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_cartesian_outputs(stage_dir: Path, title: str, evaluation: Any, baseline_metrics: dict[str, Any]) -> dict[str, str]:
    pred_x, pred_y, pred_z = _polar_to_cartesian(
        evaluation.predictions["predicted_distance"],
        evaluation.predictions["predicted_azimuth"],
        evaluation.predictions["predicted_elevation"],
    )
    true_x, true_y, true_z = _polar_to_cartesian(
        evaluation.predictions["target_distance"],
        evaluation.predictions["target_azimuth"],
        evaluation.predictions["target_elevation"],
    )
    save_prediction_scatter(true_x, pred_x, stage_dir / "test_x_prediction.png", f"{title} X Prediction", "True X (m)", "Predicted X (m)")
    save_prediction_scatter(true_y, pred_y, stage_dir / "test_y_prediction.png", f"{title} Y Prediction", "True Y (m)", "Predicted Y (m)")
    save_prediction_scatter(true_z, pred_z, stage_dir / "test_z_prediction.png", f"{title} Z Prediction", "True Z (m)", "Predicted Z (m)")
    save_grouped_bar_chart(
        ["X MAE", "Y MAE", "Z MAE", "Euclidean"],
        {
            "Baseline": [
                float(baseline_metrics["x_mae_m"]),
                float(baseline_metrics["y_mae_m"]),
                float(baseline_metrics["z_mae_m"]),
                float(baseline_metrics["euclidean_error_m"]),
            ],
            "Experiment": [
                float(evaluation.metrics["x_mae_m"]),
                float(evaluation.metrics["y_mae_m"]),
                float(evaluation.metrics["z_mae_m"]),
                float(evaluation.metrics["euclidean_error_m"]),
            ],
        },
        stage_dir / "cartesian_comparison.png",
        f"{title} Cartesian Comparison",
        ylabel="Error (m)",
    )
    return {
        "test_x_prediction": str(stage_dir / "test_x_prediction.png"),
        "test_y_prediction": str(stage_dir / "test_y_prediction.png"),
        "test_z_prediction": str(stage_dir / "test_z_prediction.png"),
        "cartesian_comparison": str(stage_dir / "cartesian_comparison.png"),
    }


def _save_variant_artifacts(stage_dir: Path, model: nn.Module, diagnostics: dict[str, torch.Tensor]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    encoder = getattr(model, "encoder", None)

    if encoder is not None and hasattr(encoder, "distance_left_offsets"):
        artifacts["adaptive_delay_offsets"] = _save_line_plot(
            {
                "Distance Left": torch.tanh(encoder.distance_left_offsets).detach().cpu().numpy(),
                "Distance Right": encoder.distance_right_offsets.detach().cpu().tanh().numpy(),
                "ITD": encoder.azimuth_itd_offsets.detach().cpu().tanh().numpy(),
            },
            stage_dir / "adaptive_delay_offsets.png",
            "Adaptive Delay Offsets",
            "Offset (bins, normalized)",
            baseline=0.0,
        )
        artifacts["adaptive_spectral_offsets"] = _save_line_plot(
            {"Spectral": encoder.spectral_offsets.detach().cpu().tanh().numpy()},
            stage_dir / "adaptive_spectral_offsets.png",
            "Adaptive Spectral Offsets",
            "Offset (channels, normalized)",
            baseline=0.0,
        )
        artifacts["adaptive_gains"] = _save_line_plot(
            {
                "Distance Left": (1.0 + 0.35 * torch.tanh(encoder.distance_left_gain)).detach().cpu().numpy(),
                "Distance Right": (1.0 + 0.35 * torch.tanh(encoder.distance_right_gain)).detach().cpu().numpy(),
                "Spectral": (1.0 + 0.35 * torch.tanh(encoder.spectral_gain)).detach().cpu().numpy(),
            },
            stage_dir / "adaptive_gains.png",
            "Adaptive Gains",
            "Gain",
            baseline=1.0,
        )

    resonant_spikes = diagnostics.get("resonant_spikes")
    resonant_frequency = diagnostics.get("resonant_frequency")
    resonant_decay = diagnostics.get("resonant_decay")
    if resonant_frequency is not None:
        artifacts["resonant_tuning"] = _save_line_plot(
            {
                "Frequency": resonant_frequency.flatten().detach().cpu().numpy(),
                "Decay": resonant_decay.flatten().detach().cpu().numpy() if resonant_decay is not None else np.zeros_like(resonant_frequency.flatten().detach().cpu().numpy()),
            },
            stage_dir / "resonant_tuning.png",
            "Resonant Parameter Profile",
            "Value",
        )
    if resonant_spikes is not None:
        save_heatmap(
            resonant_spikes[0].T,
            stage_dir / "resonant_spikes.png",
            "Resonant Spike Raster",
            xlabel="Pseudo-Time",
            ylabel="Resonator",
        )
        artifacts["resonant_spikes"] = str(stage_dir / "resonant_spikes.png")

    if "pre_pathway_left_spikes" in diagnostics:
        save_heatmap(
            diagnostics["pre_pathway_left_spikes"][0],
            stage_dir / "pre_pathway_left_spikes.png",
            "Pre-Pathway Left Spike Trace",
            xlabel="Time",
            ylabel="Channel",
        )
        artifacts["pre_pathway_left_spikes"] = str(stage_dir / "pre_pathway_left_spikes.png")

    if "post_pathway_distance_spikes" in diagnostics:
        save_heatmap(
            diagnostics["post_pathway_distance_spikes"][0].T,
            stage_dir / "post_pathway_distance_spikes.png",
            "Post-Pathway Distance Spikes",
            xlabel="Pseudo-Time",
            ylabel="Neuron",
        )
        artifacts["post_pathway_distance_spikes"] = str(stage_dir / "post_pathway_distance_spikes.png")

    return artifacts


def _round2_report(
    outputs_root: Path,
    baseline: dict[str, Any],
    training_config: EnhancedTrainingConfig,
    results: list[dict[str, Any]],
) -> Path:
    lines = [
        "# Round 2 Experiments",
        "",
        "## Overview",
        "",
        "This report tests new ideas against the accepted combined model using only the short-data protocol. The combined small-data run is treated as the fixed control for every comparison, so all acceptance decisions are against the same reference model under the same reduced-data budget.",
        "",
        "## Fixed Protocol",
        "",
        f"- Dataset mode: `{training_config.dataset_mode}`",
        "- Split: `700 train / 150 validation / 150 test`",
        f"- Max epochs: `{training_config.max_epochs}`",
        f"- Early stopping patience: `{training_config.early_stopping_patience}`",
        f"- Scheduler: `ReduceLROnPlateau` with patience `{training_config.scheduler_patience}` and factor `{training_config.scheduler_factor}`",
        "- Device: `cpu`",
        "- Thread cap: `1`",
        "",
        "## Experiment 0 Control",
        "",
        f"- Source: saved short-data combined run at `outputs/combined_experiment/short_data_1000_result.json`",
        f"- Polar combined error: `{baseline['metrics']['combined_error']:.4f}`",
        f"- Distance MAE: `{baseline['metrics']['distance_mae_m']:.4f} m`",
        f"- Azimuth MAE: `{baseline['metrics']['azimuth_mae_deg']:.4f} deg`",
        f"- Elevation MAE: `{baseline['metrics']['elevation_mae_deg']:.4f} deg`",
        f"- Cartesian Euclidean error: `{baseline['metrics']['euclidean_error_m']:.4f} m`",
        f"- X / Y / Z MAE: `{baseline['metrics']['x_mae_m']:.4f}`, `{baseline['metrics']['y_mae_m']:.4f}`, `{baseline['metrics']['z_mae_m']:.4f} m`",
        f"- Runtime: `{baseline['result']['timings']['total_seconds']:.2f} s` total, `{baseline['result']['timings']['training_seconds']:.2f} s` training",
        "",
        "![Baseline distance](combined_experiment/combined_experiment_1235_small_data/test_distance_prediction.png)",
        "![Baseline coordinate error profile](combined_experiment/combined_experiment_1235_small_data/coordinate_error_profile.png)",
        "",
        "## Results Table",
        "",
        "| Experiment | Combined Error | Euclidean Error (m) | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Accepted | Cartesian Improved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['title']} | {item['test_metrics']['combined_error']:.4f} | {item['test_metrics']['euclidean_error_m']:.4f} | "
            f"{item['test_metrics']['distance_mae_m']:.4f} | {item['test_metrics']['azimuth_mae_deg']:.4f} | "
            f"{item['test_metrics']['elevation_mae_deg']:.4f} | {'Yes' if item['accepted'] else 'No'} | "
            f"{'Yes' if item['cartesian_improved'] else 'No'} |"
        )

    lines.extend(["", "## Detailed Experiments", ""])

    for item in results:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Change: {item['description']}",
                f"- Rationale: {item['rationale']}",
                f"- Loss mode: `{item['loss_mode']}`",
                f"- Decision against fixed short-data baseline: `{'ACCEPTED' if item['accepted'] else 'REJECTED'}`",
                f"- Cartesian improvement: `{'YES' if item['cartesian_improved'] else 'NO'}`",
                "",
                "Implementation details:",
            ]
        )
        lines.extend([f"- {step}" for step in item["implemented_steps"]])
        lines.extend(
            [
                "",
                "Analysis focus:",
            ]
        )
        lines.extend([f"- {point}" for point in item["analysis_focus"]])
        lines.extend(
            [
                "",
                "Polar metrics:",
                f"- Combined error: `{item['test_metrics']['combined_error']:.4f}`",
                f"- Distance MAE: `{item['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Azimuth MAE: `{item['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Elevation MAE: `{item['test_metrics']['elevation_mae_deg']:.4f} deg`",
                "",
                "Cartesian metrics:",
                f"- Euclidean error: `{item['test_metrics']['euclidean_error_m']:.4f} m`",
                f"- X / Y / Z MAE: `{item['test_metrics']['x_mae_m']:.4f}`, `{item['test_metrics']['y_mae_m']:.4f}`, `{item['test_metrics']['z_mae_m']:.4f} m`",
                "",
                "Delta vs fixed short-data baseline:",
                f"- Combined error delta: `{item['comparison']['combined_error_delta']:.4f}`",
                f"- Distance MAE delta: `{item['comparison']['distance_mae_delta']:.4f}`",
                f"- Azimuth MAE delta: `{item['comparison']['azimuth_mae_delta']:.4f}`",
                f"- Elevation MAE delta: `{item['comparison']['elevation_mae_delta']:.4f}`",
                f"- Euclidean error delta: `{item['cartesian_delta']['euclidean_error_delta']:.4f} m`",
                "",
                "Timing:",
                f"- Data prep: `{item['timings']['data_prep_seconds']:.2f} s`",
                f"- Training: `{item['timings']['training_seconds']:.2f} s`",
                f"- Evaluation: `{item['timings']['evaluation_seconds']:.2f} s`",
                f"- Total: `{item['timings']['total_seconds']:.2f} s`",
                "",
                f"![{item['title']} loss](round_2_experiments/{item['name']}/loss.png)",
                f"![{item['title']} polar comparison](round_2_experiments/{item['name']}/comparison.png)",
                f"![{item['title']} cartesian comparison](round_2_experiments/{item['name']}/cartesian_comparison.png)",
                f"![{item['title']} distance](round_2_experiments/{item['name']}/test_distance_prediction.png)",
                f"![{item['title']} coordinate profile](round_2_experiments/{item['name']}/coordinate_error_profile.png)",
            ]
        )
        if item["artifacts"].get("adaptive_delay_offsets"):
            lines.append(f"![{item['title']} adaptive delays](round_2_experiments/{item['name']}/adaptive_delay_offsets.png)")
        if item["artifacts"].get("adaptive_spectral_offsets"):
            lines.append(f"![{item['title']} adaptive spectral offsets](round_2_experiments/{item['name']}/adaptive_spectral_offsets.png)")
        if item["artifacts"].get("adaptive_gains"):
            lines.append(f"![{item['title']} adaptive gains](round_2_experiments/{item['name']}/adaptive_gains.png)")
        if item["artifacts"].get("resonant_tuning"):
            lines.append(f"![{item['title']} resonant tuning](round_2_experiments/{item['name']}/resonant_tuning.png)")
        if item["artifacts"].get("resonant_spikes"):
            lines.append(f"![{item['title']} resonant spikes](round_2_experiments/{item['name']}/resonant_spikes.png)")
        if item["artifacts"].get("pre_pathway_left_spikes"):
            lines.append(f"![{item['title']} pre-pathway spikes](round_2_experiments/{item['name']}/pre_pathway_left_spikes.png)")
        if item["artifacts"].get("post_pathway_distance_spikes"):
            lines.append(f"![{item['title']} post-pathway spikes](round_2_experiments/{item['name']}/post_pathway_distance_spikes.png)")
        lines.append("")

    accepted = [item["title"] for item in results if item["accepted"]]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Accepted experiments by the existing polar acceptance rule: {', '.join(accepted) if accepted else 'none'}",
            "- Cartesian-only improvements are reported separately so the Cartesian-loss runs can be interpreted even if they do not beat the baseline on the original polar rule.",
            "- This round uses the short-data protocol only. Any promising experiment should be rerun later with the longer training regime before it is treated as a genuine model replacement.",
        ]
    )

    report_path = outputs_root / "round_2_experiments_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_round_2_experiments(config: Any, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=10,
        scheduler_patience=3,
    )
    baseline = _load_short_combined_baseline(config, outputs)
    context = StageContext(config=config, device=torch.device("cpu"), outputs=outputs)
    params, baseline_label = _baseline_reference_params(context)

    prep_start = time.perf_counter()
    data = _prepare_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    data_prep_seconds = time.perf_counter() - prep_start

    stage_root = outputs.root / "round_2_experiments"
    stage_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for index, spec in enumerate(_round2_specs(), start=1):
        seed_everything(config.seed + index)
        print(f"[round_2] running {spec.name} on cpu", flush=True)
        model = _instantiate_round2_model(data, spec)
        total_start = time.perf_counter()
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

        comparison = _metrics_delta(test_eval.metrics, baseline["metrics"])
        cartesian_delta = {
            "x_mae_delta": float(test_eval.metrics["x_mae_m"] - baseline["metrics"]["x_mae_m"]),
            "y_mae_delta": float(test_eval.metrics["y_mae_m"] - baseline["metrics"]["y_mae_m"]),
            "z_mae_delta": float(test_eval.metrics["z_mae_m"] - baseline["metrics"]["z_mae_m"]),
            "euclidean_error_delta": float(test_eval.metrics["euclidean_error_m"] - baseline["metrics"]["euclidean_error_m"]),
        }
        accepted = _is_accepted(test_eval.metrics, baseline["metrics"])
        cartesian_improved = float(test_eval.metrics["euclidean_error_m"]) < float(baseline["metrics"]["euclidean_error_m"]) - 1e-6
        artifacts = _save_improved_outputs(stage_root, spec, train_result, test_eval, baseline["metrics"], model)
        stage_dir = stage_root / spec.name
        prediction_cache = _save_prediction_cache(stage_dir, test_eval.predictions, data.test_targets_raw)
        coordinate_profile = _save_coordinate_error_profiles(
            Path(prediction_cache),
            stage_dir / "coordinate_error_profile.png",
            f"{spec.title} Coordinate Error Profile",
        )
        cartesian_artifacts = _save_cartesian_outputs(stage_dir, spec.title, test_eval, baseline["metrics"])
        variant_artifacts = _save_variant_artifacts(stage_dir, model, test_eval.diagnostics)

        result = {
            "name": spec.name,
            "title": spec.title,
            "description": spec.description,
            "rationale": spec.rationale,
            "implemented_steps": spec.implemented_steps,
            "analysis_focus": spec.analysis_focus,
            "loss_mode": spec.loss_mode,
            "accepted": accepted,
            "cartesian_improved": cartesian_improved,
            "decision": "ACCEPTED" if accepted else "REJECTED",
            "baseline_label": baseline_label,
            "dataset_mode": training_config.dataset_mode,
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
            "comparison": {key: format_float(value) for key, value in comparison.items()},
            "cartesian_delta": {key: format_float(value) for key, value in cartesian_delta.items()},
            "learned_sigmas": None if learned_sigmas is None else {key: format_float(value) for key, value in learned_sigmas.items()},
            "artifacts": {
                **artifacts,
                **cartesian_artifacts,
                **variant_artifacts,
                "prediction_cache": prediction_cache,
                "coordinate_error_profile": coordinate_profile,
            },
        }
        save_json(stage_dir / "result.json", result)
        results.append(result)

    save_grouped_bar_chart(
        ["Combined", "Distance", "Azimuth", "Elevation"],
        {
            "Baseline": [
                float(baseline["metrics"]["combined_error"]),
                float(baseline["metrics"]["distance_mae_m"]),
                float(baseline["metrics"]["azimuth_mae_deg"]),
                float(baseline["metrics"]["elevation_mae_deg"]),
            ],
            **{
                item["title"]: [
                    float(item["test_metrics"]["combined_error"]),
                    float(item["test_metrics"]["distance_mae_m"]),
                    float(item["test_metrics"]["azimuth_mae_deg"]),
                    float(item["test_metrics"]["elevation_mae_deg"]),
                ]
                for item in results
            },
        },
        stage_root / "overall_polar_comparison.png",
        "Round 2 Polar Metric Comparison",
        ylabel="Error",
    )
    save_grouped_bar_chart(
        ["Euclidean", "X", "Y", "Z"],
        {
            "Baseline": [
                float(baseline["metrics"]["euclidean_error_m"]),
                float(baseline["metrics"]["x_mae_m"]),
                float(baseline["metrics"]["y_mae_m"]),
                float(baseline["metrics"]["z_mae_m"]),
            ],
            **{
                item["title"]: [
                    float(item["test_metrics"]["euclidean_error_m"]),
                    float(item["test_metrics"]["x_mae_m"]),
                    float(item["test_metrics"]["y_mae_m"]),
                    float(item["test_metrics"]["z_mae_m"]),
                ]
                for item in results
            },
        },
        stage_root / "overall_cartesian_comparison.png",
        "Round 2 Cartesian Metric Comparison",
        ylabel="Error (m)",
    )

    report_path = _round2_report(outputs.root, baseline, training_config, results)
    summary = {
        "baseline_label": baseline_label,
        "dataset_mode": training_config.dataset_mode,
        "baseline": baseline["result"],
        "results": results,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_2_experiments_summary.json", summary)
    save_json(stage_root / "results.json", {"baseline": baseline["result"], "experiments": results})
    return summary
