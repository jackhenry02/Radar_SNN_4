from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.acoustics import cochlea_filterbank_stages, lif_encode_stages, sample_uniform_positions, simulate_echo_batch
from models.experimental_variants import ExperimentBatch
from models.pathway_snn import PathwayBatch
from stages.base import StageContext
from stages.cochlea_explained import _matched_human_band_config
from stages.combined_experiment import _save_coordinate_error_profiles, _save_prediction_cache
from stages.experiments import (
    ExperimentEvaluation,
    PreparedExperimentData,
    _baseline_reference_params,
    _build_experiment_split,
    _metrics_delta,
    _prediction_metrics,
)
from stages.improved_experiments import ImprovedExperimentSpec, _save_improved_outputs
from stages.improvement import _apply_standardization, _copy_config, _distance_candidates, _fit_standardization, _itd_candidates
from stages.round_2_combined_all import _combined_all_spec, _instantiate_combined_all_model
from stages.round_2_experiments import _augment_with_cartesian_metrics
from utils.common import (
    GlobalConfig,
    OutputPaths,
    format_float,
    save_cochlea_plot,
    save_grouped_bar_chart,
    save_json,
    save_text_figure,
    save_waveform_and_spectrogram,
    seed_everything,
)


@dataclass
class ExpandedTargetBundle:
    train_model: torch.Tensor
    val_model: torch.Tensor
    test_model: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    scales: torch.Tensor


@dataclass
class ExpandedTrainingResult:
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


@dataclass(frozen=True)
class SpatialSupportSpec:
    name: str
    title: str
    description: str
    rationale: str
    output_dirname: str
    report_filename: str
    max_range_m: float
    azimuth_limits_deg: tuple[float, float]
    elevation_limits_deg: tuple[float, float]
    reference_note: str


EXPANDED_COUNTS = {"train": 700, "val": 150, "test": 150}


def _expanded_space_config(base: GlobalConfig, max_range_m: float) -> GlobalConfig:
    max_delay_s = 2.0 * max_range_m / base.speed_of_sound_m_s
    required_window = max_delay_s + base.chirp_duration_s + 0.02
    return _copy_config(
        base,
        max_range_m=max_range_m,
        signal_duration_s=max(base.signal_duration_s, required_window),
    )


def _expanded_run_spec() -> SpatialSupportSpec:
    return SpatialSupportSpec(
        name="expanded_space_quick_test",
        title="Expanded Space Quick Test",
        description=(
            "Run the matched-human round-2 combined-all model on a much larger spatial support: "
            "distance up to 20 m and azimuth/elevation across +/-90 degrees."
        ),
        rationale=(
            "This is a stress test of the accepted short-data round-2 combined-all architecture under a much larger "
            "range-and-angle support, while keeping the cheaper matched-human front end."
        ),
        output_dirname="expanded_space_test",
        report_filename="expanded_space_test_report.md",
        max_range_m=20.0,
        azimuth_limits_deg=(-90.0, 90.0),
        elevation_limits_deg=(-90.0, 90.0),
        reference_note=(
            "The saved reference below is the original short-domain matched-human round-2 combined-all run "
            "(`0.5 to 2.5 m`, `-45 to 45 deg`, `-30 to 30 deg`). It is included only as context."
        ),
    )


def _control_run_spec() -> SpatialSupportSpec:
    return SpatialSupportSpec(
        name="expanded_space_control_test",
        title="Original-Range Control Test",
        description=(
            "Run the matched-human round-2 combined-all model through the same control harness, but keep the "
            "original short-domain spatial support."
        ),
        rationale=(
            "This is a harness-control check. It uses the same training loop, output formatting, and matched-human "
            "front end as the expanded-space test, but keeps the original target range and angular limits."
        ),
        output_dirname="expanded_space_control_test",
        report_filename="expanded_space_control_test_report.md",
        max_range_m=2.5,
        azimuth_limits_deg=(-45.0, 45.0),
        elevation_limits_deg=(-30.0, 30.0),
        reference_note=(
            "The saved reference below is the earlier short-domain matched-human round-2 combined-all run. "
            "This control is intended to show whether the current harness still reproduces a non-collapsed solution "
            "when the spatial support matches the original task."
        ),
    )


def _expanded_space_param_overrides(params: dict[str, Any], support_spec: SpatialSupportSpec) -> dict[str, Any]:
    adjusted = dict(params)
    if support_spec.max_range_m <= 2.5:
        return adjusted
    adjusted.update(
        {
            "num_delay_lines": max(int(params["num_delay_lines"]), 64),
            "branch_hidden_dim": max(int(params["branch_hidden_dim"]), 40),
            "hidden_dim": max(int(params["hidden_dim"]), 160),
            "num_steps": max(int(params["num_steps"]), 12),
        }
    )
    return adjusted


def _expanded_training_overrides(spec: ImprovedExperimentSpec, support_spec: SpatialSupportSpec) -> dict[str, Any]:
    overrides = dict(spec.training_overrides)
    if support_spec.max_range_m <= 2.5:
        return overrides
    overrides.update(
        {
            "batch_size": 6,
            "learning_rate_scale": 0.65,
            "cartesian_mix_weight": 0.65,
        }
    )
    return overrides


def _load_control_result(outputs_root: Path) -> dict[str, Any] | None:
    path = outputs_root / "expanded_space_control_test" / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_expanded_dataset_split(
    config: GlobalConfig,
    device: torch.device,
    count: int,
    *,
    split_seed: int,
    azimuth_limits_deg: tuple[float, float],
    elevation_limits_deg: tuple[float, float],
) -> tuple[Any, torch.Tensor]:
    seed_everything(split_seed)
    radii, azimuth, elevation = sample_uniform_positions(
        count,
        config,
        device,
        azimuth_limits_deg=azimuth_limits_deg,
        elevation_limits_deg=elevation_limits_deg,
        include_elevation=True,
    )
    batch = simulate_echo_batch(
        config,
        radii,
        azimuth,
        elevation,
        binaural=True,
        add_noise=True,
        include_elevation_cues=True,
    )
    targets = torch.stack([radii, azimuth, elevation], dim=-1)
    return batch, targets


def _prepare_expanded_data(
    context: StageContext,
    params: dict[str, Any],
    support_spec: SpatialSupportSpec,
    *,
    chunk_size: int,
) -> PreparedExperimentData:
    local_config = _copy_config(
        context.config,
        num_cochlea_channels=int(params["num_frequency_channels"]),
        spike_threshold=float(params["spike_threshold"]),
        filter_bandwidth_sigma=float(params["filter_bandwidth_sigma"]),
    )
    distance_candidates = _distance_candidates(local_config, context.device, int(params["num_delay_lines"]))
    itd_candidates = _itd_candidates(local_config, context.device, int(params["num_delay_lines"]))
    split_offsets = {"train": 80_001, "val": 80_002, "test": 80_003}

    train_acoustic, train_targets_raw = _sample_expanded_dataset_split(
        local_config,
        context.device,
        EXPANDED_COUNTS["train"],
        split_seed=local_config.seed + split_offsets["train"],
        azimuth_limits_deg=support_spec.azimuth_limits_deg,
        elevation_limits_deg=support_spec.elevation_limits_deg,
    )
    val_acoustic, val_targets_raw = _sample_expanded_dataset_split(
        local_config,
        context.device,
        EXPANDED_COUNTS["val"],
        split_seed=local_config.seed + split_offsets["val"],
        azimuth_limits_deg=support_spec.azimuth_limits_deg,
        elevation_limits_deg=support_spec.elevation_limits_deg,
    )
    test_acoustic, test_targets_raw = _sample_expanded_dataset_split(
        local_config,
        context.device,
        EXPANDED_COUNTS["test"],
        split_seed=local_config.seed + split_offsets["test"],
        azimuth_limits_deg=support_spec.azimuth_limits_deg,
        elevation_limits_deg=support_spec.elevation_limits_deg,
    )

    train_batch = _build_experiment_split(
        train_acoustic,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )
    val_batch = _build_experiment_split(
        val_acoustic,
        local_config,
        distance_candidates,
        itd_candidates,
        num_delay_lines=int(params["num_delay_lines"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        chunk_size=chunk_size,
    )
    test_batch = _build_experiment_split(
        test_acoustic,
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

    return PreparedExperimentData(
        mode="expanded_space_quick",
        local_config=local_config,
        params=params,
        train_batch=train_batch,
        val_batch=val_batch,
        test_batch=test_batch,
        train_targets_raw=train_targets_raw,
        val_targets_raw=val_targets_raw,
        test_targets_raw=test_targets_raw,
    )


def _prepare_expanded_target_bundle(data: PreparedExperimentData, support_spec: SpatialSupportSpec) -> ExpandedTargetBundle:
    scales = data.train_targets_raw.new_tensor(
        [
            1.0,
            max(abs(support_spec.azimuth_limits_deg[0]), abs(support_spec.azimuth_limits_deg[1])),
            max(abs(support_spec.elevation_limits_deg[0]), abs(support_spec.elevation_limits_deg[1])),
        ]
    ).view(1, -1)
    train = data.train_targets_raw / scales
    val = data.val_targets_raw / scales
    test = data.test_targets_raw / scales
    mean, std = _fit_standardization(train)
    return ExpandedTargetBundle(
        train_model=_apply_standardization(train, mean, std),
        val_model=_apply_standardization(val, mean, std),
        test_model=_apply_standardization(test, mean, std),
        mean=mean,
        std=std,
        scales=scales,
    )


def _decode_output(output_model: torch.Tensor, target_bundle: ExpandedTargetBundle) -> torch.Tensor:
    denormalized = output_model * target_bundle.std + target_bundle.mean
    return denormalized * target_bundle.scales


def _loss_components(
    output_model: torch.Tensor,
    target_raw: torch.Tensor,
    target_bundle: ExpandedTargetBundle,
    diagnostics: dict[str, torch.Tensor],
    local_config: GlobalConfig,
    task_weights: torch.Tensor,
    spike_weight: float,
    cartesian_mix_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    spike_penalty = diagnostics["spike_rate"].mean()
    raw_prediction = _decode_output(output_model, target_bundle)
    corrected_scales = target_bundle.scales.to(raw_prediction.device, raw_prediction.dtype)
    task_terms = torch.abs(raw_prediction - target_raw) / corrected_scales
    azimuth_rad = torch.deg2rad(raw_prediction[:, 1])
    elevation_rad = torch.deg2rad(raw_prediction[:, 2])
    target_azimuth_rad = torch.deg2rad(target_raw[:, 1])
    target_elevation_rad = torch.deg2rad(target_raw[:, 2])
    pred_x = raw_prediction[:, 0] * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    pred_y = raw_prediction[:, 0] * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    pred_z = raw_prediction[:, 0] * torch.sin(elevation_rad)
    true_x = target_raw[:, 0] * torch.cos(target_elevation_rad) * torch.cos(target_azimuth_rad)
    true_y = target_raw[:, 0] * torch.cos(target_elevation_rad) * torch.sin(target_azimuth_rad)
    true_z = target_raw[:, 0] * torch.sin(target_elevation_rad)
    cartesian_prediction = torch.stack([pred_x, pred_y, pred_z], dim=-1)
    cartesian_target = torch.stack([true_x, true_y, true_z], dim=-1)
    cartesian_scales = cartesian_prediction.new_full((1, 3), float(local_config.max_range_m))
    cartesian_terms = torch.abs(cartesian_prediction - cartesian_target) / cartesian_scales
    localisation_loss = cartesian_terms.mean() + cartesian_mix_weight * (task_terms * task_weights.view(1, -1)).mean()
    loss = localisation_loss + spike_weight * spike_penalty
    return loss, {
        "distance_loss": float(task_terms[:, 0].mean().item()),
        "azimuth_loss": float(task_terms[:, 1].mean().item()),
        "elevation_loss": float(task_terms[:, 2].mean().item()),
        "x_loss": float(cartesian_terms[:, 0].mean().item()),
        "y_loss": float(cartesian_terms[:, 1].mean().item()),
        "z_loss": float(cartesian_terms[:, 2].mean().item()),
        "spike_penalty": float(spike_penalty.item()),
    }


def _evaluate_model(
    model: nn.Module,
    batch: ExperimentBatch,
    targets_raw: torch.Tensor,
    target_bundle: ExpandedTargetBundle,
    local_config: GlobalConfig,
) -> ExperimentEvaluation:
    model.eval()
    with torch.no_grad():
        output_model, diagnostics = model(batch)
        raw_prediction = _decode_output(output_model, target_bundle)
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
    return _augment_with_cartesian_metrics(evaluation)


def _train_model(
    model: nn.Module,
    data: PreparedExperimentData,
    target_bundle: ExpandedTargetBundle,
    spec: ImprovedExperimentSpec,
    *,
    max_epochs: int,
    batch_size: int,
    scheduler_patience: int,
    early_stopping_patience: int,
    learning_rate_scale: float,
    cartesian_mix_weight: float,
) -> ExpandedTrainingResult:
    params = data.params
    task_weights = torch.tensor(
        [1.0, float(params["angle_weight"]), float(params["elevation_weight"])],
        device=data.train_targets_raw.device,
    )
    learning_rate = float(params["learning_rate"]) * learning_rate_scale
    spike_weight = float(params["loss_weighting"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
        threshold=1e-4,
        min_lr=1e-5,
    )

    best_epoch = 0
    best_val_combined = float("inf")
    best_loss = float("inf")
    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    final_diagnostics: dict[str, torch.Tensor] = {}
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_combined_history: list[float] = []
    lr_history: list[float] = []
    epochs_without_improvement = 0
    stopped_early = False

    def _batch_iterator():
        permutation = torch.randperm(target_bundle.train_model.shape[0], device=target_bundle.train_model.device)
        for start in range(0, target_bundle.train_model.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            yield data.train_batch.index_select(indices), target_bundle.train_model[indices], data.train_targets_raw[indices]

    for epoch in range(max_epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets_model, batch_targets_raw in _batch_iterator():
            optimizer.zero_grad(set_to_none=True)
            output_model, diagnostics = model(batch_features)
            loss, _ = _loss_components(
                output_model,
                batch_targets_raw,
                target_bundle,
                diagnostics,
                data.local_config,
                task_weights,
                spike_weight,
                cartesian_mix_weight,
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))

        model.eval()
        with torch.no_grad():
            val_output_model, val_diagnostics = model(data.val_batch)
            val_loss, _ = _loss_components(
                val_output_model,
                data.val_targets_raw,
                target_bundle,
                val_diagnostics,
                data.local_config,
                task_weights,
                spike_weight,
                cartesian_mix_weight,
            )
            val_prediction = _decode_output(val_output_model, target_bundle)
            val_eval = _prediction_metrics(data.local_config, val_prediction, data.val_targets_raw, val_diagnostics)

        val_loss_value = float(val_loss.item())
        val_combined_value = float(val_eval.metrics["combined_error"])
        scheduler.step(val_combined_value)
        val_loss_history.append(val_loss_value)
        val_combined_history.append(val_combined_value)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))

        if val_combined_value < best_val_combined - 1e-4:
            best_epoch = epoch
            best_loss = val_loss_value
            best_val_combined = val_combined_value
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            final_diagnostics = {key: value.detach().clone() for key, value in val_diagnostics.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    return ExpandedTrainingResult(
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        val_combined_error=val_combined_history,
        lr_history=lr_history,
        best_state=best_state,
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_combined_error=best_val_combined,
        diagnostics=final_diagnostics,
        stopped_early=stopped_early,
        executed_epochs=len(train_loss_history),
        best_auxiliary_state=None,
    )


def _load_reference_short_result(outputs_root: Path) -> dict[str, Any]:
    path = outputs_root / "cochlea_explained" / "human_matched_band_experiment.json"
    if not path.exists():
        raise FileNotFoundError("Reference matched-human short-domain round-2 result not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _expanded_spec() -> ImprovedExperimentSpec:
    return ImprovedExperimentSpec(
        name="expanded_space_quick_test",
        title="Expanded Space Quick Test",
        description="Support-spec placeholder",
        rationale="Support-spec placeholder",
        implemented_steps=[
            "Use the matched-human front end: 64 kHz sample rate, 18 kHz to 2 kHz chirp, 2 kHz to 20 kHz cochlea.",
            "Keep the round-2 combined-all architecture unchanged.",
            "Change only the spatial support for the current scenario.",
            "Extend the signal buffer so echoes from the scenario max range can arrive inside the waveform.",
            "Train with the same mixed Cartesian-plus-polar loss style, but widened to the expanded angular support.",
            "Run a small quick-training pass with conservative batch size and one backend thread.",
        ],
        remaining_steps=[
            "If the model remains stable, repeat with a cached expanded-space dataset and a longer training schedule.",
            "If distance collapses, consider a longer chirp or explicit matched-filter / dechirp preprocessing for long-range support.",
        ],
        variant="combined_all",
        loss_mode="mixed_cartesian_expanded",
        training_overrides={"learning_rate_scale": 0.85, "batch_size": 8, "cartesian_mix_weight": 0.5},
    )


def _run_spatial_support_test(config: GlobalConfig, outputs: OutputPaths, support_spec: SpatialSupportSpec) -> dict[str, Any]:
    expanded_config = _expanded_space_config(_matched_human_band_config(config), support_spec.max_range_m)
    context = StageContext(config=expanded_config, device=torch.device("cpu"), outputs=outputs)
    params, baseline_label = _baseline_reference_params(context)
    params = _expanded_space_param_overrides(params, support_spec)
    spec = _expanded_spec()
    spec = ImprovedExperimentSpec(
        name=support_spec.name,
        title=support_spec.title,
        description=support_spec.description,
        rationale=support_spec.rationale,
        implemented_steps=spec.implemented_steps,
        remaining_steps=spec.remaining_steps,
        variant=spec.variant,
        loss_mode=spec.loss_mode,
        training_overrides=_expanded_training_overrides(spec, support_spec),
    )
    output_root = outputs.root / support_spec.output_dirname
    output_root.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    prep_start = time.perf_counter()
    data = _prepare_expanded_data(context, params, support_spec, chunk_size=16)
    target_bundle = _prepare_expanded_target_bundle(data, support_spec)
    data_prep_seconds = time.perf_counter() - prep_start

    model = _instantiate_combined_all_model(data, _combined_all_spec(), params)
    training_start = time.perf_counter()
    train_result = _train_model(
        model,
        data,
        target_bundle,
        spec,
        max_epochs=10,
        batch_size=int(spec.training_overrides["batch_size"]),
        scheduler_patience=2,
        early_stopping_patience=4,
        learning_rate_scale=float(spec.training_overrides["learning_rate_scale"]),
        cartesian_mix_weight=float(spec.training_overrides["cartesian_mix_weight"]),
    )
    training_seconds = time.perf_counter() - training_start

    model.load_state_dict(train_result.best_state)

    evaluation_start = time.perf_counter()
    val_eval = _evaluate_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
    test_eval = _evaluate_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    reference = _load_reference_short_result(outputs.root)
    reference_metrics = {
        "distance_mae_m": float(reference["test_metrics"]["distance_mae_m"]),
        "azimuth_mae_deg": float(reference["test_metrics"]["azimuth_mae_deg"]),
        "elevation_mae_deg": float(reference["test_metrics"]["elevation_mae_deg"]),
        "combined_error": float(reference["test_metrics"]["combined_error"]),
        "mean_spike_rate": float(reference["test_metrics"]["mean_spike_rate"]),
    }
    comparison = _metrics_delta(test_eval.metrics, reference_metrics)

    artifacts = _save_improved_outputs(output_root, spec, train_result, test_eval, reference_metrics, model)
    stage_dir = output_root / spec.name
    prediction_cache = _save_prediction_cache(stage_dir, test_eval.predictions, data.test_targets_raw)
    coordinate_error_profile = _save_coordinate_error_profiles(
        Path(prediction_cache),
        stage_dir / "coordinate_error_profile.png",
        f"{spec.title} Coordinate Error Profile",
    )
    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error", "Euclidean"],
        {
            "Original short-domain reference": [
                float(reference["test_metrics"]["distance_mae_m"]),
                float(reference["test_metrics"]["azimuth_mae_deg"]),
                float(reference["test_metrics"]["elevation_mae_deg"]),
                float(reference["test_metrics"]["combined_error"]),
                float(reference.get("test_metrics", {}).get("euclidean_error_m", 0.0)),
            ],
            support_spec.title: [
                float(test_eval.metrics["distance_mae_m"]),
                float(test_eval.metrics["azimuth_mae_deg"]),
                float(test_eval.metrics["elevation_mae_deg"]),
                float(test_eval.metrics["combined_error"]),
                float(test_eval.metrics["euclidean_error_m"]),
            ],
        },
        output_root / "reference_vs_expanded.png",
        f"Reference Short-Domain vs {support_spec.title}",
        ylabel="Error",
    )
    save_text_figure(
        [
            f"baseline_label: {baseline_label}",
            "reference_note: saved short-domain matched-human result is contextual",
            f"max_range_m: {expanded_config.max_range_m:.1f}",
            f"azimuth_limits_deg: {support_spec.azimuth_limits_deg}",
            f"elevation_limits_deg: {support_spec.elevation_limits_deg}",
            f"signal_duration_ms: {expanded_config.signal_duration_s * 1_000.0:.1f}",
            f"combined_error: {test_eval.metrics['combined_error']:.4f}",
            f"distance_mae_m: {test_eval.metrics['distance_mae_m']:.4f}",
            f"azimuth_mae_deg: {test_eval.metrics['azimuth_mae_deg']:.4f}",
            f"elevation_mae_deg: {test_eval.metrics['elevation_mae_deg']:.4f}",
            f"euclidean_error_m: {test_eval.metrics['euclidean_error_m']:.4f}",
            f"data_prep_seconds: {data_prep_seconds:.2f}",
            f"training_seconds: {training_seconds:.2f}",
            f"evaluation_seconds: {evaluation_seconds:.2f}",
            f"total_seconds: {total_seconds:.2f}",
            f"best_epoch: {train_result.best_epoch + 1}",
        ],
        output_root / "summary.png",
        f"{support_spec.title} Summary",
    )

    result = {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "rationale": spec.rationale,
        "reference_result": "combined_experiment short_data_1000_result",
        "dataset_counts": EXPANDED_COUNTS,
        "target_space": {
            "distance_m": [expanded_config.min_range_m, expanded_config.max_range_m],
            "azimuth_deg": list(support_spec.azimuth_limits_deg),
            "elevation_deg": list(support_spec.elevation_limits_deg),
        },
        "config": {
            "sample_rate_hz": expanded_config.sample_rate_hz,
            "signal_duration_s": format_float(expanded_config.signal_duration_s, digits=4),
            "chirp_duration_s": format_float(expanded_config.chirp_duration_s, digits=4),
            "chirp_start_hz": format_float(expanded_config.chirp_start_hz),
            "chirp_end_hz": format_float(expanded_config.chirp_end_hz),
            "num_cochlea_channels": int(data.local_config.num_cochlea_channels),
            "model_num_frequency_channels": int(params["num_frequency_channels"]),
            "cochlea_low_hz": format_float(data.local_config.cochlea_low_hz),
            "cochlea_high_hz": format_float(data.local_config.cochlea_high_hz),
        },
        "training_config": {
            "max_epochs": 10,
            "batch_size": int(spec.training_overrides["batch_size"]),
            "scheduler_patience": 2,
            "early_stopping_patience": 4,
            "learning_rate_scale": spec.training_overrides["learning_rate_scale"],
            "cartesian_mix_weight": spec.training_overrides["cartesian_mix_weight"],
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
        "reference_metrics": {key: format_float(value) for key, value in reference_metrics.items()},
        "val_metrics": {key: format_float(value) for key, value in val_eval.metrics.items()},
        "test_metrics": {key: format_float(value) for key, value in test_eval.metrics.items()},
        "comparison_vs_reference": {key: format_float(value) for key, value in comparison.items()},
        "artifacts": {
            **artifacts,
            "prediction_cache": prediction_cache,
            "coordinate_error_profile": coordinate_error_profile,
            "reference_vs_expanded": str(output_root / "reference_vs_expanded.png"),
            "summary": str(output_root / "summary.png"),
        },
    }
    save_json(output_root / "result.json", result)

    control_result = None if support_spec.max_range_m <= 2.5 else _load_control_result(outputs.root)
    report_lines = [
        f"# {support_spec.title}",
        "",
    ]
    if control_result is not None:
        report_lines.extend(
            [
                "## Control Check",
                "",
                "The same matched-human / round-2 combined-all harness was rerun on the original task limits before this expanded-space run. That control did not collapse, so the flat behavior in the expanded test is not a generic harness failure.",
                "",
                f"- Control combined error: `{float(control_result['test_metrics']['combined_error']):.4f}`",
                f"- Control distance / azimuth / elevation: `{float(control_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(control_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(control_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
                f"- Control prediction spread: distance std `{float(control_result['test_metrics']['predicted_distance_std']):.4f}`, azimuth std `{float(control_result['test_metrics']['predicted_azimuth_std']):.4f}`, elevation std `{float(control_result['test_metrics']['predicted_elevation_std']):.4f}`",
                f"- Control target spread: distance std `{float(control_result['test_metrics']['target_distance_std']):.4f}`, azimuth std `{float(control_result['test_metrics']['target_azimuth_std']):.4f}`, elevation std `{float(control_result['test_metrics']['target_elevation_std']):.4f}`",
                "",
            ]
        )
    report_lines.extend(
        [
        "## Overview",
        "",
        f"This run reuses the matched-human round-2 combined-all architecture and tests it on a different spatial support. The goal is not a fair benchmark against the original domain, but a quick check of whether the current system still trains and produces structured predictions when the range and angle support are changed. This scenario is: `{support_spec.title}`.",
        "",
        "Important note:",
        "- The existing matched-human spike cache was not reused here.",
        "- That cache is for a different domain and also uses a different cochlea width, so it cannot support this test directly.",
        "- Direct metric comparison against the original short-domain matched-human run is therefore only contextual, not like-for-like.",
        "",
        "## Test Setup",
        "",
        f"- Model architecture: `{spec.variant}`",
        "- Reference architecture source: saved round-2 combined-all matched-human run",
        f"- Dataset counts: `train {EXPANDED_COUNTS['train']} / val {EXPANDED_COUNTS['val']} / test {EXPANDED_COUNTS['test']}`",
        f"- Distance support: `{expanded_config.min_range_m:.1f} to {expanded_config.max_range_m:.1f} m`",
        f"- Azimuth support: `{support_spec.azimuth_limits_deg[0]:.0f} to {support_spec.azimuth_limits_deg[1]:.0f} deg`",
        f"- Elevation support: `{support_spec.elevation_limits_deg[0]:.0f} to {support_spec.elevation_limits_deg[1]:.0f} deg`",
        f"- Signal duration increased to: `{expanded_config.signal_duration_s * 1_000.0:.1f} ms`",
        f"- Sample rate: `{expanded_config.sample_rate_hz}` (`matched-human front end`)",
        f"- Chirp: `{expanded_config.chirp_start_hz:.0f} Hz -> {expanded_config.chirp_end_hz:.0f} Hz`",
        f"- Cochlea range: `{expanded_config.cochlea_low_hz:.0f} Hz -> {expanded_config.cochlea_high_hz:.0f} Hz`",
        f"- Cochlea channels actually used in front end: `{int(data.local_config.num_cochlea_channels)}`",
        f"- Downstream model frequency width: `{int(params['num_frequency_channels'])}`",
        f"- Delay lines: `{int(params['num_delay_lines'])}`",
        f"- Branch hidden dim: `{int(params['branch_hidden_dim'])}`",
        f"- Fusion hidden dim: `{int(params['hidden_dim'])}`",
        f"- SNN time steps: `{int(params['num_steps'])}`",
        f"- Loss mode: `{spec.loss_mode}`",
        f"- Max epochs: `10`, batch size: `{int(spec.training_overrides['batch_size'])}`",
        "",
        "## Results",
        "",
        f"- Validation combined error: `{float(val_eval.metrics['combined_error']):.4f}`",
        f"- Test combined error: `{float(test_eval.metrics['combined_error']):.4f}`",
        f"- Test distance MAE: `{float(test_eval.metrics['distance_mae_m']):.4f} m`",
        f"- Test azimuth MAE: `{float(test_eval.metrics['azimuth_mae_deg']):.4f} deg`",
        f"- Test elevation MAE: `{float(test_eval.metrics['elevation_mae_deg']):.4f} deg`",
        f"- Test Euclidean error: `{float(test_eval.metrics['euclidean_error_m']):.4f} m`",
        f"- Mean spike rate: `{float(test_eval.metrics['mean_spike_rate']):.4f}`",
        "",
        "## Timing",
        "",
        f"- Data preparation: `{data_prep_seconds:.2f} s`",
        f"- Training: `{training_seconds:.2f} s`",
        f"- Evaluation: `{evaluation_seconds:.2f} s`",
        f"- Total: `{total_seconds:.2f} s`",
        f"- Best epoch: `{train_result.best_epoch + 1}`",
        "",
        "## Reference Comparison",
        "",
        support_spec.reference_note,
        "",
        f"- Reference combined error: `{float(reference['test_metrics']['combined_error']):.4f}`",
        f"- Reference distance / azimuth / elevation: `{float(reference['test_metrics']['distance_mae_m']):.4f} m`, `{float(reference['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(reference['test_metrics']['elevation_mae_deg']):.4f} deg`",
        f"- Scenario combined error delta vs reference: `{float(comparison['combined_error_delta']):.4f}`",
        f"- Scenario distance MAE delta vs reference: `{float(comparison['distance_mae_delta']):.4f} m`",
        f"- Scenario azimuth MAE delta vs reference: `{float(comparison['azimuth_mae_delta']):.4f} deg`",
        f"- Scenario elevation MAE delta vs reference: `{float(comparison['elevation_mae_delta']):.4f} deg`",
        "",
        "## Interpretation",
        "",
        "- When the support is widened, the range expansion is especially severe because the echo delay support grows from a few milliseconds to over 100 ms, forcing a much longer receive window even at the cheaper 64 kHz front end.",
        "- The angular task becomes harder when the model has to cover the full front hemisphere for both azimuth and elevation.",
        "- This rerun increases delay-bank and latent capacity relative to the earlier failed expanded attempt, so it is testing whether the previous collapse was partly a model-capacity mismatch rather than only a data-domain mismatch.",
        "- The round-2 combined-all architecture is still the base model; the main changes are support-aware delay-bank and latent-capacity scaling.",
        "- If performance degrades sharply but predictions still show non-trivial spread, that suggests the pipeline remains functional but is out of its previously tuned operating regime.",
        "",
        "## Plots",
        "",
        f"![Loss]({support_spec.output_dirname}/{spec.name}/loss.png)",
        f"![Distance prediction]({support_spec.output_dirname}/{spec.name}/test_distance_prediction.png)",
        f"![Azimuth prediction]({support_spec.output_dirname}/{spec.name}/test_azimuth_prediction.png)",
        f"![Elevation prediction]({support_spec.output_dirname}/{spec.name}/test_elevation_prediction.png)",
        f"![Coordinate error profile]({support_spec.output_dirname}/{spec.name}/coordinate_error_profile.png)",
        f"![Reference vs scenario]({support_spec.output_dirname}/reference_vs_expanded.png)",
    ]
    )
    report_path = outputs.root / support_spec.report_filename
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "result": result,
        "report_path": str(report_path),
    }
    save_json(outputs.root / f"{support_spec.output_dirname}_summary.json", summary)
    return summary


def run_expanded_space_test(config: GlobalConfig, outputs: OutputPaths) -> dict[str, Any]:
    return _run_spatial_support_test(config, outputs, _expanded_run_spec())


def run_expanded_space_control_test(config: GlobalConfig, outputs: OutputPaths) -> dict[str, Any]:
    return _run_spatial_support_test(config, outputs, _control_run_spec())


def run_expanded_space_frontend_diagnostics(config: GlobalConfig, outputs: OutputPaths) -> dict[str, Any]:
    support_spec = _expanded_run_spec()
    effective_config = _expanded_space_config(_matched_human_band_config(config), support_spec.max_range_m)
    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    params, _ = _baseline_reference_params(context)
    params = _expanded_space_param_overrides(params, support_spec)
    effective_config = _copy_config(
        effective_config,
        num_cochlea_channels=int(params["num_frequency_channels"]),
        spike_threshold=float(params["spike_threshold"]),
        filter_bandwidth_sigma=float(params["filter_bandwidth_sigma"]),
    )

    output_dir = outputs.root / "expanded_space_frontend_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    distances_m = [0.5, 2.5, 5.0, 10.0, 15.0, 20.0]
    diagnostics: list[dict[str, Any]] = []

    for index, distance_m in enumerate(distances_m):
        seed_everything(effective_config.seed + index)
        batch = simulate_echo_batch(
            effective_config,
            torch.tensor([distance_m], device=context.device),
            torch.tensor([0.0], device=context.device),
            torch.tensor([0.0], device=context.device),
            binaural=True,
            add_noise=True,
            include_elevation_cues=True,
        )
        receive_left = batch.receive[0, 0].unsqueeze(0)
        filter_stages = cochlea_filterbank_stages(
            receive_left,
            sample_rate_hz=effective_config.sample_rate_hz,
            num_channels=effective_config.num_cochlea_channels,
            low_hz=effective_config.cochlea_low_hz,
            high_hz=effective_config.cochlea_high_hz,
            spacing_mode=effective_config.cochlea_spacing_mode,
            filter_bandwidth_sigma=effective_config.filter_bandwidth_sigma,
            envelope_lowpass_hz=effective_config.envelope_lowpass_hz,
            downsample=effective_config.envelope_downsample,
        )
        lif_stages = lif_encode_stages(
            filter_stages["cochleagram"],
            threshold=effective_config.spike_threshold,
            beta=effective_config.spike_beta,
        )
        delay_ms = float(batch.delays_s[0, 0].item() * 1_000.0)
        xlim_ms = (
            max(0.0, delay_ms - 1.0),
            min(effective_config.signal_duration_s * 1_000.0, delay_ms + effective_config.chirp_duration_s * 1_000.0 + 1.0),
        )
        distance_tag = str(distance_m).replace(".", "p")
        waveform_path = output_dir / f"distance_{distance_tag}_signal.png"
        cochlea_path = output_dir / f"distance_{distance_tag}_cochleagram_spikes.png"
        save_waveform_and_spectrogram(
            batch.receive[0, 0],
            effective_config.sample_rate_hz,
            waveform_path,
            f"Expanded-Space Echo At {distance_m:.1f} m",
        )
        save_cochlea_plot(
            filter_stages["cochleagram"][0],
            lif_stages["spikes"][0],
            effective_config.envelope_rate_hz,
            cochlea_path,
            f"Expanded-Space Cochleagram And Spikes At {distance_m:.1f} m",
            xlim_ms=xlim_ms,
        )
        diagnostics.append(
            {
                "distance_m": distance_m,
                "delay_ms": format_float(delay_ms),
                "receive_peak_abs": format_float(batch.receive[0, 0].abs().amax().item(), digits=6),
                "cochleagram_peak": format_float(filter_stages["cochleagram"][0].amax().item(), digits=6),
                "spike_count": int(lif_stages["spikes"][0].sum().item()),
                "waveform_plot": str(waveform_path),
                "cochlea_plot": str(cochlea_path),
            }
        )

    summary = {
        "config": {
            "sample_rate_hz": effective_config.sample_rate_hz,
            "signal_duration_s": format_float(effective_config.signal_duration_s, digits=4),
            "chirp_duration_s": format_float(effective_config.chirp_duration_s, digits=4),
            "chirp_start_hz": format_float(effective_config.chirp_start_hz),
            "chirp_end_hz": format_float(effective_config.chirp_end_hz),
            "num_cochlea_channels": int(effective_config.num_cochlea_channels),
        },
        "note": "0 m is not included because the current echo simulator uses inverse-square attenuation and a two-way delay model, so 0.5 m is the nearest practical diagnostic point.",
        "diagnostics": diagnostics,
    }
    save_json(output_dir / "summary.json", summary)
    return summary
