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
import torch.nn.functional as F

from models.acoustics import azimuth_spectral_gain_profile, elevation_spectral_gain_profile
from models.experimental_variants import ExperimentBatch
from models.round2_variants import AllRound2CombinedModel, AllRound2Encoder
from models.round3_variants import AzimuthNotchDetectorEncoder, CombFilterElevationEncoder, LIFCoincidenceRound3Encoder, NotchDetectorElevationEncoder, OrthogonalNotchCombinedEncoder
from stages.base import StageContext
from stages.cochlea_explained import _matched_human_band_config
from stages.combined_experiment import _save_coordinate_error_profiles, _save_prediction_cache
from stages.expanded_space_test import SpatialSupportSpec, _expanded_space_config, _prepare_expanded_data
from stages.experiments import ExperimentEvaluation, PreparedExperimentData, _baseline_reference_params, _metrics_delta, _prediction_metrics
from stages.improved_experiments import TargetBundle, _decode_model_output, _is_accepted, _prepare_target_bundle
from stages.round_2_experiments import _augment_with_cartesian_metrics, _cartesian_metrics_from_predictions, _round2_loss_components
from stages.training_improved_experiments import EnhancedTrainingConfig
from utils.common import (
    GlobalConfig,
    format_float,
    save_error_histogram,
    save_grouped_bar_chart,
    save_heatmap,
    save_json,
    save_loss_curve,
    save_prediction_scatter,
    save_text_figure,
    seed_everything,
)


@dataclass
class Round3ExperimentSpec:
    name: str
    title: str
    description: str
    rationale: str
    variant: str
    output_mode: str
    implemented_steps: list[str]
    analysis_focus: list[str]
    training_overrides: dict[str, Any]
    data_variant: str = "base"


@dataclass
class SinCosTargetBundle:
    train_distance_model: torch.Tensor
    val_distance_model: torch.Tensor
    test_distance_model: torch.Tensor
    distance_mean: torch.Tensor
    distance_std: torch.Tensor


@dataclass
class Distance01TargetBundle:
    train_distance01: torch.Tensor
    val_distance01: torch.Tensor
    test_distance01: torch.Tensor
    train_angle_model: torch.Tensor
    val_angle_model: torch.Tensor
    test_angle_model: torch.Tensor
    angle_mean: torch.Tensor
    angle_std: torch.Tensor


def _round3_support_spec() -> SpatialSupportSpec:
    return SpatialSupportSpec(
        name="round_3_small_support",
        title="Round 3 Small-Support Protocol",
        description=(
            "Use the matched-human front end with the short-data protocol, original angular support, "
            "and distance support extended to 5 m."
        ),
        rationale=(
            "This keeps the experiment cheap enough to iterate while testing whether the 140 dB unnormalized front end "
            "and new architecture/target changes remain stable when range support is widened moderately."
        ),
        output_dirname="round_3_experiments",
        report_filename="round_3_experiments_report.md",
        max_range_m=5.0,
        azimuth_limits_deg=(-45.0, 45.0),
        elevation_limits_deg=(-30.0, 30.0),
        reference_note="Round 3 compares every variant only against the fresh round-3 control.",
    )


def _round3_base_config(base: GlobalConfig) -> GlobalConfig:
    support_spec = _round3_support_spec()
    config = _expanded_space_config(_matched_human_band_config(base), support_spec.max_range_m)
    payload = {
        **config.__dict__,
        "normalize_spike_envelope": False,
        "transmit_gain": 1_000.0,
    }
    return GlobalConfig(**payload)


def _round3_variant_config(base: GlobalConfig, data_variant: str) -> GlobalConfig:
    if data_variant == "base":
        return base
    if data_variant == "moving_notch":
        return GlobalConfig(
            **{
                **base.__dict__,
                "elevation_cue_mode": "slope_notch",
                "elevation_notch_strength": 1.8,
                "elevation_notch_width": 0.065,
            }
        )
    if data_variant == "azimuth_moving_notch":
        return GlobalConfig(
            **{
                **base.__dict__,
                "azimuth_cue_mode": "slope_notch",
                "azimuth_spectral_strength": 0.65,
                "azimuth_notch_strength": 1.4,
                "azimuth_notch_width": 0.07,
            }
        )
    if data_variant == "azimuth_notch_only":
        return GlobalConfig(
            **{
                **base.__dict__,
                "azimuth_cue_mode": "notch",
                "azimuth_spectral_strength": 0.0,
                "azimuth_notch_strength": 1.4,
                "azimuth_notch_width": 0.07,
            }
        )
    if data_variant == "orthogonal_combined_notches":
        return GlobalConfig(
            **{
                **base.__dict__,
                "elevation_cue_mode": "slope_notch",
                "elevation_notch_strength": 1.8,
                "elevation_notch_width": 0.06,
                "elevation_notch_center_min": 0.32,
                "elevation_notch_center_max": 0.68,
                "azimuth_cue_mode": "notch",
                "azimuth_spectral_strength": 0.0,
                "azimuth_notch_strength": 1.35,
                "azimuth_notch_width": 0.055,
                "azimuth_notch_center_min": 0.08,
                "azimuth_notch_center_max": 0.28,
                "azimuth_notch_mirror_across_band": True,
            }
        )
    raise ValueError(f"Unsupported round-3 data variant '{data_variant}'.")


def _round3_specs() -> list[Round3ExperimentSpec]:
    return [
        Round3ExperimentSpec(
            name="round3_control_round2_combined_140db_unnormalized",
            title="Round 3 Experiment 0: 140 dB Unnormalized Control",
            description=(
                "Re-run the accepted round-2 combined model under the matched-human 140 dB unnormalized front end, "
                "with the original angular support and range support extended to 5 m."
            ),
            rationale=(
                "This is the fixed control for round 3. It isolates the effect of the new front-end regime and "
                "moderately larger range support before any architectural change is added."
            ),
            variant="control_baseline",
            output_mode="baseline",
            implemented_steps=[
                "Keep the round-2 combined architecture unchanged.",
                "Switch the front end to the matched-human configuration.",
                "Set transmit gain to 1000x under the 1x = 80 dB convention, corresponding to 140 dB.",
                "Disable per-sample spike-envelope normalization.",
                "Increase max range to 5 m and automatically extend the signal window.",
            ],
            analysis_focus=[
                "Whether the 140 dB unnormalized front end remains stable on the 5 m task.",
                "How much prediction spread is retained when support is widened moderately.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 1.0, "cartesian_mix_weight": 0.5},
        ),
        Round3ExperimentSpec(
            name="round3_experiment_1_lif_coincidence_detectors",
            title="Round 3 Experiment 1: Trainable LIF Coincidence Detectors",
            description=(
                "Replace the fixed distance and ITD overlap banks with explicit spike-domain LIF coincidence banks "
                "that receive delayed reference spikes and undelayed target spikes."
            ),
            rationale=(
                "This tests whether making coincidence detection explicitly spiking and mildly trainable improves the "
                "timing pathways without discarding the round-2 inductive bias."
            ),
            variant="lif_coincidence",
            output_mode="baseline",
            implemented_steps=[
                "Keep the full round-2 combined encoder as the base path.",
                "Build delay-aligned reference spike banks for transmit-to-echo and left-to-right timing cues.",
                "Feed the delayed reference spikes and the target spikes into trainable LIF coincidence banks.",
                "Project the resulting coincidence spike rates into residual distance and azimuth latents.",
                "Train detector weights and membrane beta while keeping the detector thresholds fixed.",
            ],
            analysis_focus=[
                "Whether explicit LIF coincidence banks improve distance or azimuth beyond the round-3 control.",
                "Whether the learned beta and input weights stay in a sensible regime rather than saturating.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
        ),
        Round3ExperimentSpec(
            name="round3_experiment_2_comb_filter_elevation",
            title="Round 3 Experiment 2: Comb-Filtered Elevation Features",
            description=(
                "Replace the simple spectral-slope style elevation residual with a comb-response residual that measures "
                "periodic spectral structure across frequency channels."
            ),
            rationale=(
                "This tests whether a richer fixed spectral operator improves elevation cues more than the current "
                "slope-like summary."
            ),
            variant="comb_elevation",
            output_mode="baseline",
            implemented_steps=[
                "Keep the round-2 combined encoder as the base path.",
                "Build spectral norm and notch terms exactly as before.",
                "Replace the residual slope-like term with a multi-lag comb response across channels.",
                "Project the comb feature vector into an elevation residual latent with a small learned gain.",
            ],
            analysis_focus=[
                "Whether the comb response reduces elevation MAE relative to the round-3 control.",
                "Whether the richer spectral filter improves elevation without damaging distance and azimuth.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
        ),
        Round3ExperimentSpec(
            name="round3_experiment_2a_moving_notch_cue",
            title="Round 3 Experiment 2A: Moving-Notch Elevation Cue",
            description=(
                "Keep the round-2 combined model unchanged, but replace the simulator-side slope-only elevation cue "
                "with a slope-plus-moving-notch spectral cue whose notch position shifts with elevation."
            ),
            rationale=(
                "This isolates whether richer elevation information in the acoustic front end improves performance even "
                "before adding any dedicated notch decoder."
            ),
            variant="control_baseline",
            output_mode="baseline",
            implemented_steps=[
                "Keep the round-2 combined encoder and readout unchanged.",
                "Change the simulator elevation cue from `slope` to `slope + moving notch`.",
                "Map elevation smoothly to notch center position across the cochlear frequency span.",
                "Reuse the same matched-human 140 dB unnormalized front end and 5 m support.",
            ],
            analysis_focus=[
                "Whether richer elevation structure in the data alone improves elevation MAE.",
                "Whether the unchanged decoder can exploit the moving-notch cue without an explicit notch bank.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round3_experiment_2b_moving_notch_plus_detectors",
            title="Round 3 Experiment 2B: Moving-Notch Cue Plus Notch Detectors",
            description=(
                "Use the same slope-plus-moving-notch simulator cue as 2A and add an explicit notch-detector bank in "
                "the elevation branch to decode notch position from the spike-domain spectrum."
            ),
            rationale=(
                "This tests the full two-stage hypothesis: a richer cue may need an aligned decoder before it helps."
            ),
            variant="notch_detector_elevation",
            output_mode="baseline",
            implemented_steps=[
                "Use the same simulator-side slope-plus-moving-notch cue as Experiment 2A.",
                "Compute post-cochlea spectral notch features from the binaural spike counts.",
                "Apply a fixed bank of Gaussian notch detectors across channel position.",
                "Project detector responses into a learned residual added only to the elevation latent.",
            ],
            analysis_focus=[
                "Whether explicit notch-location decoding improves elevation beyond both the control and 2A.",
                "Whether the detector responses cover the channel range or collapse to a narrow subset.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round3_experiment_3_sincos_angle_regression",
            title="Round 3 Experiment 3: Sine/Cosine Angle Regression",
            description=(
                "Predict azimuth and elevation as sine/cosine pairs, constrain the outputs toward the unit circle, "
                "and decode them back to angles only for evaluation and Cartesian regularization."
            ),
            rationale=(
                "This tests whether removing angular wraparound from the raw regression target improves optimization."
            ),
            variant="control_baseline",
            output_mode="sincos",
            implemented_steps=[
                "Keep the round-2 combined architecture unchanged up to the final readout.",
                "Increase the readout size from 3 to 5 outputs: distance, azimuth sin/cos, elevation sin/cos.",
                "Normalize the predicted angle pairs before decoding with atan2.",
                "Add a unit-circle penalty on the raw pair norms.",
            ],
            analysis_focus=[
                "Whether angular wraparound handling improves azimuth and elevation errors.",
                "Whether the unit-circle constraint remains active rather than collapsing to near-zero vectors.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35, "unit_penalty_weight": 0.1},
        ),
        Round3ExperimentSpec(
            name="round3_experiment_3a_azimuth_moving_notch_detectors",
            title="Round 3 Experiment 3A: Ear-Specific Azimuth Notch Detectors",
            description=(
                "Add mirrored ear-specific slope-plus-moving-notch azimuth cues upstream of the cochlea and decode them "
                "with a dedicated notch-detector residual as a third azimuth pathway stream alongside ITD and ILD."
            ),
            rationale=(
                "This tests whether azimuth benefits from richer ear-specific spectral structure, not just time and "
                "level differences."
            ),
            variant="azimuth_notch_detector",
            output_mode="baseline",
            implemented_steps=[
                "Keep the base round-2 combined model and 140 dB unnormalized front end.",
                "Inject mirrored ear-specific azimuth slope-plus-moving-notch spectral cues before the cochlea.",
                "Build per-ear spike-domain notch profiles from the left and right receive spectra.",
                "Apply a fixed bank of azimuth notch detectors across channel position for each ear.",
                "Project left, right, and left-right notch-detector responses into an azimuth-only residual latent.",
            ],
            analysis_focus=[
                "Whether explicit ear-specific notch cues improve azimuth beyond the round-3 control.",
                "Whether the notch-detector responses remain asymmetric across ears rather than collapsing to identical patterns.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
            data_variant="azimuth_moving_notch",
        ),
        Round3ExperimentSpec(
            name="round3_experiment_3b_azimuth_notch_only_detectors",
            title="Round 3 Experiment 3B: Ear-Specific Azimuth Notch Detectors Without Slope",
            description=(
                "Add mirrored ear-specific moving-notch azimuth cues upstream of the cochlea, but do not add any extra "
                "azimuth spectral slope cue, then decode them with the same azimuth notch-detector residual."
            ),
            rationale=(
                "This isolates whether the azimuth notch itself is useful, or whether the previous 3A result mainly came "
                "from interaction between the notch and added spectral slope."
            ),
            variant="azimuth_notch_detector",
            output_mode="baseline",
            implemented_steps=[
                "Keep the base round-2 combined model and 140 dB unnormalized front end.",
                "Inject mirrored ear-specific azimuth moving notches before the cochlea, with no added azimuth slope term.",
                "Build per-ear spike-domain notch profiles from the left and right receive spectra.",
                "Apply the same fixed bank of azimuth notch detectors across channel position for each ear.",
                "Project left, right, and left-right notch-detector responses into an azimuth-only residual latent.",
            ],
            analysis_focus=[
                "Whether azimuth improves when only the notch cue is added, without the extra spectral tilt used in 3A.",
                "Whether removing the slope reduces interference with the elevation branch.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
            data_variant="azimuth_notch_only",
        ),
        Round3ExperimentSpec(
            name="round3_experiment_3c_orthogonal_combined_notches",
            title="Round 3 Experiment 3C: Orthogonal Combined Azimuth/Elevation Notches",
            description=(
                "Combine a common-mode mid-band elevation slope-plus-notch cue with mirrored edge-band azimuth notch-only "
                "cues, then decode them with branch-specific detectors that use common-mode features for elevation and "
                "difference features for azimuth."
            ),
            rationale=(
                "This tests whether cue orthogonality in both the simulator and the branch decoders can keep the two "
                "angular tasks from interfering with each other."
            ),
            variant="orthogonal_combined_notches",
            output_mode="baseline",
            implemented_steps=[
                "Inject a common-mode elevation slope-plus-notch cue in the middle of the spectral band for both ears.",
                "Inject mirrored ear-specific azimuth notch-only cues toward the spectral edges.",
                "Decode elevation from binaural-mean notch features with a mid-band detector bank.",
                "Decode azimuth from ear-difference notch features with an edge-focused detector bank.",
                "Keep distance and the rest of the round-2 combined architecture unchanged.",
            ],
            analysis_focus=[
                "Whether separating cue design and detector inputs improves both angles simultaneously.",
                "Whether the new model beats the control and the earlier single-purpose notch experiments on angular error.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.5},
            data_variant="orthogonal_combined_notches",
        ),
        Round3ExperimentSpec(
            name="round3_experiment_4_distance01_labels",
            title="Round 3 Experiment 4: 0-1 Distance Labels",
            description=(
                "Train the distance output against labels normalized to the 0-1 interval, then decode them back to "
                "physical metres for evaluation."
            ),
            rationale=(
                "This tests whether a simpler bounded distance target improves conditioning when the range support is "
                "wider than the original short task."
            ),
            variant="control_baseline",
            output_mode="distance01",
            implemented_steps=[
                "Keep the round-2 combined architecture unchanged up to the final readout.",
                "Apply a sigmoid to the distance output and train it against 0-1 normalized range labels.",
                "Keep angle decoding compatible with the baseline target bundle.",
                "Decode distance back to metres for all reported metrics.",
            ],
            analysis_focus=[
                "Whether the bounded distance code improves distance MAE without harming angle prediction.",
                "Whether the decoded 0-1 distance output saturates near the support limits.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35},
        ),
    ]


def _instantiate_base_encoder(data: PreparedExperimentData, params: dict[str, Any]) -> AllRound2Encoder:
    distance_candidates = torch.linspace(
        int((2.0 * data.local_config.min_range_m / data.local_config.speed_of_sound_m_s) * data.local_config.envelope_rate_hz),
        int((2.0 * data.local_config.max_range_m / data.local_config.speed_of_sound_m_s) * data.local_config.envelope_rate_hz),
        int(params["num_delay_lines"]),
        device=data.train_targets_raw.device,
    ).round().to(torch.long).unique(sorted=True)
    max_itd_s = data.local_config.ear_spacing_m / data.local_config.speed_of_sound_m_s
    max_bins = max(1, int(max_itd_s * data.local_config.envelope_rate_hz) + 2)
    itd_candidates = (
        torch.linspace(-max_bins, max_bins, int(params["num_delay_lines"]), device=data.train_targets_raw.device)
        .round()
        .to(torch.long)
        .unique(sorted=True)
    )
    return AllRound2Encoder(
        distance_dim=data.train_batch.pathway.distance.shape[-1],
        azimuth_dim=data.train_batch.pathway.azimuth.shape[-1],
        elevation_dim=data.train_batch.pathway.elevation.shape[-1],
        branch_hidden_dim=int(params["branch_hidden_dim"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
        num_delay_lines=int(params["num_delay_lines"]),
        distance_candidates=distance_candidates,
        itd_candidates=itd_candidates,
        beta=float(params["membrane_beta"]),
        threshold=float(params["fusion_threshold"]),
        num_steps=int(params["num_steps"]),
    )


def _instantiate_round3_model(
    data: PreparedExperimentData,
    params: dict[str, Any],
    spec: Round3ExperimentSpec,
) -> nn.Module:
    base_encoder = _instantiate_base_encoder(data, params)
    if spec.variant == "control_baseline" and spec.output_mode in {"baseline", "sincos", "distance01"}:
        encoder = base_encoder
    elif spec.variant == "lif_coincidence":
        encoder = LIFCoincidenceRound3Encoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            distance_candidates=base_encoder.distance_candidates,
            itd_candidates=base_encoder.itd_candidates,
        )
    elif spec.variant == "comb_elevation":
        encoder = CombFilterElevationEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    elif spec.variant == "notch_detector_elevation":
        encoder = NotchDetectorElevationEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    elif spec.variant == "azimuth_notch_detector":
        encoder = AzimuthNotchDetectorEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    elif spec.variant == "orthogonal_combined_notches":
        encoder = OrthogonalNotchCombinedEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    else:
        raise ValueError(f"Unsupported round-3 variant '{spec.variant}'.")

    output_dim = 3
    if spec.output_mode == "sincos":
        output_dim = 5

    return AllRound2CombinedModel(
        encoder=encoder,
        hidden_dim=int(params["hidden_dim"]),
        output_dim=output_dim,
        num_steps=int(params["num_steps"]),
        beta=float(params["membrane_beta"]),
        threshold=float(params["fusion_threshold"]),
        reset_mechanism=str(params["reset_mechanism"]),
    ).to(data.train_targets_raw.device)


def _prepare_sincos_target_bundle(data: PreparedExperimentData) -> SinCosTargetBundle:
    train_distance = data.train_targets_raw[:, :1]
    val_distance = data.val_targets_raw[:, :1]
    test_distance = data.test_targets_raw[:, :1]
    distance_mean = train_distance.mean(dim=0, keepdim=True)
    distance_std = train_distance.std(dim=0, keepdim=True).clamp_min(1e-5)
    return SinCosTargetBundle(
        train_distance_model=(train_distance - distance_mean) / distance_std,
        val_distance_model=(val_distance - distance_mean) / distance_std,
        test_distance_model=(test_distance - distance_mean) / distance_std,
        distance_mean=distance_mean,
        distance_std=distance_std,
    )


def _prepare_distance01_target_bundle(data: PreparedExperimentData, local_config: GlobalConfig) -> Distance01TargetBundle:
    denom = max(local_config.max_range_m - local_config.min_range_m, 1e-6)
    train_distance01 = ((data.train_targets_raw[:, :1] - local_config.min_range_m) / denom).clamp(0.0, 1.0)
    val_distance01 = ((data.val_targets_raw[:, :1] - local_config.min_range_m) / denom).clamp(0.0, 1.0)
    test_distance01 = ((data.test_targets_raw[:, :1] - local_config.min_range_m) / denom).clamp(0.0, 1.0)

    train_angles = torch.stack(
        [data.train_targets_raw[:, 1] / 45.0, data.train_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    val_angles = torch.stack(
        [data.val_targets_raw[:, 1] / 45.0, data.val_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    test_angles = torch.stack(
        [data.test_targets_raw[:, 1] / 45.0, data.test_targets_raw[:, 2] / 30.0],
        dim=-1,
    )
    angle_mean = train_angles.mean(dim=0, keepdim=True)
    angle_std = train_angles.std(dim=0, keepdim=True).clamp_min(1e-5)
    return Distance01TargetBundle(
        train_distance01=train_distance01,
        val_distance01=val_distance01,
        test_distance01=test_distance01,
        train_angle_model=(train_angles - angle_mean) / angle_std,
        val_angle_model=(val_angles - angle_mean) / angle_std,
        test_angle_model=(test_angles - angle_mean) / angle_std,
        angle_mean=angle_mean,
        angle_std=angle_std,
    )


def _decode_sincos_output(output_model: torch.Tensor, bundle: SinCosTargetBundle) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distance = output_model[:, :1] * bundle.distance_std + bundle.distance_mean
    az_raw = output_model[:, 1:3]
    el_raw = output_model[:, 3:5]
    az_norm = az_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    el_norm = el_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    az_pair = az_raw / az_norm
    el_pair = el_raw / el_norm
    azimuth = torch.rad2deg(torch.atan2(az_pair[:, 0], az_pair[:, 1])).unsqueeze(-1)
    elevation = torch.rad2deg(torch.atan2(el_pair[:, 0], el_pair[:, 1])).unsqueeze(-1)
    raw = torch.cat([distance, azimuth, elevation], dim=-1)
    aux = {
        "az_pair": az_pair,
        "el_pair": el_pair,
        "az_raw_norm": az_raw.norm(dim=-1),
        "el_raw_norm": el_raw.norm(dim=-1),
    }
    return raw, aux


def _decode_distance01_output(
    output_model: torch.Tensor,
    bundle: Distance01TargetBundle,
    local_config: GlobalConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    denom = max(local_config.max_range_m - local_config.min_range_m, 1e-6)
    distance01 = torch.sigmoid(output_model[:, :1])
    distance = local_config.min_range_m + distance01 * denom
    angle_model = output_model[:, 1:3] * bundle.angle_std + bundle.angle_mean
    azimuth = (angle_model[:, 0:1] * 45.0).clamp(-45.0, 45.0)
    elevation = (angle_model[:, 1:2] * 30.0).clamp(-30.0, 30.0)
    raw = torch.cat([distance, azimuth, elevation], dim=-1)
    aux = {"distance01": distance01, "angle_model": angle_model}
    return raw, aux


def _cartesian_terms(raw_prediction: torch.Tensor, target_raw: torch.Tensor, local_config: GlobalConfig) -> torch.Tensor:
    pred_distance = raw_prediction[:, 0]
    pred_azimuth = torch.deg2rad(raw_prediction[:, 1])
    pred_elevation = torch.deg2rad(raw_prediction[:, 2])
    target_distance = target_raw[:, 0]
    target_azimuth = torch.deg2rad(target_raw[:, 1])
    target_elevation = torch.deg2rad(target_raw[:, 2])

    pred_x = pred_distance * torch.cos(pred_elevation) * torch.cos(pred_azimuth)
    pred_y = pred_distance * torch.cos(pred_elevation) * torch.sin(pred_azimuth)
    pred_z = pred_distance * torch.sin(pred_elevation)
    target_x = target_distance * torch.cos(target_elevation) * torch.cos(target_azimuth)
    target_y = target_distance * torch.cos(target_elevation) * torch.sin(target_azimuth)
    target_z = target_distance * torch.sin(target_elevation)
    cartesian_prediction = torch.stack([pred_x, pred_y, pred_z], dim=-1)
    cartesian_target = torch.stack([target_x, target_y, target_z], dim=-1)
    scale = cartesian_prediction.new_full((1, 3), float(local_config.max_range_m))
    return torch.abs(cartesian_prediction - cartesian_target) / scale


def _evaluate_round3_output(
    raw_prediction: torch.Tensor,
    targets_raw: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    local_config: GlobalConfig,
) -> ExperimentEvaluation:
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


def _round3_loss_and_decode(
    spec: Round3ExperimentSpec,
    output_model: torch.Tensor,
    batch_targets_raw: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    local_config: GlobalConfig,
    params: dict[str, Any],
    baseline_bundle: TargetBundle,
    batch_baseline_model: torch.Tensor,
    sincos_bundle: SinCosTargetBundle,
    batch_sincos_distance_model: torch.Tensor,
    distance01_bundle: Distance01TargetBundle,
    batch_distance01: torch.Tensor,
    batch_distance01_angle_model: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    task_weights = output_model.new_tensor([1.0, float(params["angle_weight"]), float(params["elevation_weight"])])
    spike_penalty = diagnostics["spike_rate"].mean()
    spike_weight = float(params["loss_weighting"])
    cartesian_mix_weight = float(spec.training_overrides.get("cartesian_mix_weight", 0.5))

    if spec.output_mode == "baseline":
        loss, summary = _round2_loss_components(
            output_model,
            batch_baseline_model,
            batch_targets_raw,
            baseline_bundle,
            diagnostics,
            local_config,
            "mixed_cartesian",
            task_weights,
            spike_weight,
            None,
            cartesian_mix_weight=cartesian_mix_weight,
        )
        raw_prediction = _decode_model_output(output_model, baseline_bundle)
        return loss, raw_prediction, summary

    if spec.output_mode == "sincos":
        raw_prediction, aux = _decode_sincos_output(output_model, sincos_bundle)
        distance_loss = F.smooth_l1_loss(output_model[:, :1], batch_sincos_distance_model)
        azimuth_rad = torch.deg2rad(batch_targets_raw[:, 1])
        elevation_rad = torch.deg2rad(batch_targets_raw[:, 2])
        az_target_pair = torch.stack([torch.sin(azimuth_rad), torch.cos(azimuth_rad)], dim=-1)
        el_target_pair = torch.stack([torch.sin(elevation_rad), torch.cos(elevation_rad)], dim=-1)
        az_loss = F.smooth_l1_loss(aux["az_pair"], az_target_pair)
        el_loss = F.smooth_l1_loss(aux["el_pair"], el_target_pair)
        unit_penalty = ((aux["az_raw_norm"] - 1.0).square().mean() + (aux["el_raw_norm"] - 1.0).square().mean()) / 2.0
        cartesian = _cartesian_terms(raw_prediction, batch_targets_raw, local_config)
        localisation_loss = (
            distance_loss
            + task_weights[1] * az_loss
            + task_weights[2] * el_loss
            + cartesian_mix_weight * cartesian.mean()
            + float(spec.training_overrides.get("unit_penalty_weight", 0.1)) * unit_penalty
        )
        loss = localisation_loss + spike_weight * spike_penalty
        return loss, raw_prediction, {
            "distance_loss": float(distance_loss.item()),
            "azimuth_loss": float(az_loss.item()),
            "elevation_loss": float(el_loss.item()),
            "x_loss": float(cartesian[:, 0].mean().item()),
            "y_loss": float(cartesian[:, 1].mean().item()),
            "z_loss": float(cartesian[:, 2].mean().item()),
            "unit_penalty": float(unit_penalty.item()),
            "spike_penalty": float(spike_penalty.item()),
        }

    if spec.output_mode == "distance01":
        raw_prediction, aux = _decode_distance01_output(output_model, distance01_bundle, local_config)
        distance01_loss = F.smooth_l1_loss(aux["distance01"], batch_distance01)
        angle_terms = torch.abs(raw_prediction[:, 1:] - batch_targets_raw[:, 1:]) / raw_prediction.new_tensor([45.0, 30.0])
        cartesian = _cartesian_terms(raw_prediction, batch_targets_raw, local_config)
        localisation_loss = (
            distance01_loss
            + task_weights[1] * angle_terms[:, 0].mean()
            + task_weights[2] * angle_terms[:, 1].mean()
            + cartesian_mix_weight * cartesian.mean()
        )
        loss = localisation_loss + spike_weight * spike_penalty
        return loss, raw_prediction, {
            "distance_loss": float(distance01_loss.item()),
            "azimuth_loss": float(angle_terms[:, 0].mean().item()),
            "elevation_loss": float(angle_terms[:, 1].mean().item()),
            "x_loss": float(cartesian[:, 0].mean().item()),
            "y_loss": float(cartesian[:, 1].mean().item()),
            "z_loss": float(cartesian[:, 2].mean().item()),
            "spike_penalty": float(spike_penalty.item()),
            "distance01_mean": float(aux["distance01"].mean().item()),
        }

    raise ValueError(f"Unsupported output mode '{spec.output_mode}'.")


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


def _save_comb_filter_operator_plot(path: Path, lags: tuple[int, ...] = (2, 4, 6)) -> str:
    max_lag = max(lags)
    offsets = np.arange(-max_lag - 2, max_lag + 3)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    combined = np.zeros_like(offsets, dtype=np.float64)
    for lag in lags:
        kernel = np.zeros_like(offsets, dtype=np.float64)
        kernel[offsets == 0] = 1.0
        kernel[offsets == -lag] = -0.5
        kernel[offsets == lag] = -0.5
        combined += kernel
        ax.plot(offsets, kernel, marker="o", linewidth=1.5, label=f"Lag {lag}")
    combined /= max(1, len(lags))
    ax.plot(offsets, combined, marker="s", linewidth=2.5, linestyle="--", color="black", label="Average operator")
    ax.set_title("Round 3 Elevation Comb Operator")
    ax.set_xlabel("Channel Offset")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_moving_notch_cue_plot(config: GlobalConfig, path: Path) -> str:
    sampled_elevations = torch.linspace(-30.0, 30.0, 121, dtype=torch.float32)
    frequencies_hz = torch.fft.rfftfreq(config.signal_samples, d=1.0 / config.sample_rate_hz)
    gain, _ = elevation_spectral_gain_profile(
        config,
        sampled_elevations,
        frequencies_hz.shape[0],
        device=sampled_elevations.device,
        dtype=torch.float32,
    )
    gain_db = 20.0 * torch.log10(gain.clamp_min(1e-6))
    freq_np = frequencies_hz.detach().cpu().numpy() / 1_000.0
    elev_np = sampled_elevations.detach().cpu().numpy()
    gain_db_np = gain_db.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    visible_max_khz = min(float(config.sample_rate_hz) / 2_000.0, max(config.cochlea_high_hz, config.chirp_start_hz) / 1_000.0 * 1.05)
    visible_mask = freq_np <= visible_max_khz
    contour = ax.contourf(
        freq_np[visible_mask],
        elev_np,
        gain_db_np[:, visible_mask],
        levels=25,
        cmap="coolwarm",
    )
    ax.set_title("Moving-Notch Elevation Cue")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Elevation (deg)")
    fig.colorbar(contour, ax=ax, label="Gain (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_azimuth_moving_notch_cue_plot(config: GlobalConfig, path: Path) -> str:
    sampled_azimuth = torch.linspace(-45.0, 45.0, 121, dtype=torch.float32)
    frequencies_hz = torch.fft.rfftfreq(config.signal_samples, d=1.0 / config.sample_rate_hz)
    gain, _ = azimuth_spectral_gain_profile(
        config,
        sampled_azimuth,
        frequencies_hz.shape[0],
        device=sampled_azimuth.device,
        dtype=torch.float32,
    )
    gain_db = 20.0 * torch.log10(gain.clamp_min(1e-6))
    freq_np = frequencies_hz.detach().cpu().numpy() / 1_000.0
    az_np = sampled_azimuth.detach().cpu().numpy()
    left_db = gain_db[:, 0].detach().cpu().numpy()
    right_db = gain_db[:, 1].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    visible_max_khz = min(float(config.sample_rate_hz) / 2_000.0, max(config.cochlea_high_hz, config.chirp_start_hz) / 1_000.0 * 1.05)
    visible_mask = freq_np <= visible_max_khz
    for axis, db_map, title in zip(axes, [left_db, right_db], ["Left Ear", "Right Ear"], strict=True):
        contour = axis.contourf(
            freq_np[visible_mask],
            az_np,
            db_map[:, visible_mask],
            levels=25,
            cmap="coolwarm",
        )
        axis.set_title(title)
        axis.set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Azimuth (deg)")
    fig.suptitle("Azimuth Moving-Notch Cue")
    fig.colorbar(contour, ax=axes.ravel().tolist(), label="Gain (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_round3_outputs(
    stage_root: Path,
    spec: Round3ExperimentSpec,
    train_loss: list[float],
    val_loss: list[float],
    evaluation: ExperimentEvaluation,
    baseline_metrics: dict[str, Any],
    diagnostics: dict[str, torch.Tensor],
    local_config: GlobalConfig,
) -> dict[str, str]:
    stage_dir = stage_root / spec.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    save_loss_curve(train_loss, val_loss, stage_dir / "loss.png", spec.title)
    save_prediction_scatter(
        evaluation.predictions["target_distance"],
        evaluation.predictions["predicted_distance"],
        stage_dir / "test_distance_prediction.png",
        f"{spec.title} Distance Prediction",
        xlabel="True Distance (m)",
        ylabel="Predicted Distance (m)",
    )
    save_prediction_scatter(
        evaluation.predictions["target_azimuth"],
        evaluation.predictions["predicted_azimuth"],
        stage_dir / "test_azimuth_prediction.png",
        f"{spec.title} Azimuth Prediction",
        xlabel="True Azimuth (deg)",
        ylabel="Predicted Azimuth (deg)",
    )
    save_prediction_scatter(
        evaluation.predictions["target_elevation"],
        evaluation.predictions["predicted_elevation"],
        stage_dir / "test_elevation_prediction.png",
        f"{spec.title} Elevation Prediction",
        xlabel="True Elevation (deg)",
        ylabel="Predicted Elevation (deg)",
    )
    save_error_histogram(
        evaluation.predictions["predicted_elevation"] - evaluation.predictions["target_elevation"],
        stage_dir / "test_elevation_error.png",
        f"{spec.title} Elevation Error",
        xlabel="Elevation Error (deg)",
    )
    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error", "Euclidean"],
        {
            "Control": [
                float(baseline_metrics["distance_mae_m"]),
                float(baseline_metrics["azimuth_mae_deg"]),
                float(baseline_metrics["elevation_mae_deg"]),
                float(baseline_metrics["combined_error"]),
                float(baseline_metrics["euclidean_error_m"]),
            ],
            "Experiment": [
                float(evaluation.metrics["distance_mae_m"]),
                float(evaluation.metrics["azimuth_mae_deg"]),
                float(evaluation.metrics["elevation_mae_deg"]),
                float(evaluation.metrics["combined_error"]),
                float(evaluation.metrics["euclidean_error_m"]),
            ],
        },
        stage_dir / "comparison.png",
        f"{spec.title} vs Control",
        ylabel="Error",
    )
    save_text_figure(
        [
            f"combined_error: {evaluation.metrics['combined_error']:.4f}",
            f"distance_mae_m: {evaluation.metrics['distance_mae_m']:.4f}",
            f"azimuth_mae_deg: {evaluation.metrics['azimuth_mae_deg']:.4f}",
            f"elevation_mae_deg: {evaluation.metrics['elevation_mae_deg']:.4f}",
            f"euclidean_error_m: {evaluation.metrics['euclidean_error_m']:.4f}",
            f"spike_rate: {evaluation.metrics['mean_spike_rate']:.4f}",
            f"predicted_distance_std: {evaluation.metrics['predicted_distance_std']:.4f}",
            f"predicted_azimuth_std: {evaluation.metrics['predicted_azimuth_std']:.4f}",
            f"predicted_elevation_std: {evaluation.metrics['predicted_elevation_std']:.4f}",
        ],
        stage_dir / "summary.png",
        f"{spec.title} Summary",
    )

    artifacts = {
        "loss": str(stage_dir / "loss.png"),
        "test_distance_prediction": str(stage_dir / "test_distance_prediction.png"),
        "test_azimuth_prediction": str(stage_dir / "test_azimuth_prediction.png"),
        "test_elevation_prediction": str(stage_dir / "test_elevation_prediction.png"),
        "comparison": str(stage_dir / "comparison.png"),
        "summary": str(stage_dir / "summary.png"),
    }

    if spec.data_variant == "moving_notch":
        artifacts["moving_notch_cue"] = _save_moving_notch_cue_plot(local_config, stage_dir / "moving_notch_cue.png")
    if spec.data_variant == "orthogonal_combined_notches":
        artifacts["moving_notch_cue"] = _save_moving_notch_cue_plot(local_config, stage_dir / "moving_notch_cue.png")
    if spec.data_variant in {"azimuth_moving_notch", "azimuth_notch_only"}:
        artifacts["azimuth_moving_notch_cue"] = _save_azimuth_moving_notch_cue_plot(
            local_config, stage_dir / "azimuth_moving_notch_cue.png"
        )
    if spec.data_variant == "orthogonal_combined_notches":
        artifacts["azimuth_moving_notch_cue"] = _save_azimuth_moving_notch_cue_plot(
            local_config, stage_dir / "azimuth_moving_notch_cue.png"
        )

    if "lif_distance_left_spikes" in diagnostics:
        save_heatmap(
            diagnostics["lif_distance_left_spikes"][0].detach().cpu(),
            stage_dir / "lif_distance_left_spikes.png",
            f"{spec.title} Left Distance Coincidence Spikes",
            xlabel="Time",
            ylabel="Delay Detector",
        )
        save_heatmap(
            diagnostics["lif_itd_spikes"][0].detach().cpu(),
            stage_dir / "lif_itd_spikes.png",
            f"{spec.title} ITD Coincidence Spikes",
            xlabel="Time",
            ylabel="Delay Detector",
        )
        artifacts["lif_distance_left_spikes"] = str(stage_dir / "lif_distance_left_spikes.png")
        artifacts["lif_itd_spikes"] = str(stage_dir / "lif_itd_spikes.png")
        artifacts["lif_betas"] = _save_line_plot(
            {
                "Distance Left Beta": diagnostics["lif_distance_left_beta"].detach().cpu().numpy(),
                "ITD Beta": diagnostics["lif_itd_beta"].detach().cpu().numpy(),
            },
            stage_dir / "lif_betas.png",
            "Trainable LIF Coincidence Betas",
            "Beta",
        )

    if "comb_response" in diagnostics:
        save_heatmap(
            diagnostics["comb_response"][:1].detach().cpu(),
            stage_dir / "comb_response.png",
            f"{spec.title} Comb Response",
            xlabel="Channel",
            ylabel="Sample",
        )
        artifacts["comb_response"] = str(stage_dir / "comb_response.png")
        artifacts["comb_filter_operator"] = _save_comb_filter_operator_plot(stage_dir / "comb_filter_operator.png")

    if "notch_detector_response" in diagnostics:
        save_heatmap(
            diagnostics["notch_detector_response"][:24].detach().cpu(),
            stage_dir / "notch_detector_response.png",
            f"{spec.title} Notch Detector Responses",
            xlabel="Detector",
            ylabel="Sample",
        )
        artifacts["notch_detector_response"] = str(stage_dir / "notch_detector_response.png")
        artifacts["notch_detector_centers"] = _save_line_plot(
            {"Detector Center": diagnostics["notch_detector_centers"].detach().cpu().numpy()},
            stage_dir / "notch_detector_centers.png",
            "Notch Detector Centers",
            "Channel",
        )

    if "az_notch_left_response" in diagnostics:
        save_heatmap(
            diagnostics["az_notch_left_response"][:24].detach().cpu(),
            stage_dir / "az_notch_left_response.png",
            f"{spec.title} Left Ear Azimuth Notch Responses",
            xlabel="Detector",
            ylabel="Sample",
        )
        save_heatmap(
            diagnostics["az_notch_right_response"][:24].detach().cpu(),
            stage_dir / "az_notch_right_response.png",
            f"{spec.title} Right Ear Azimuth Notch Responses",
            xlabel="Detector",
            ylabel="Sample",
        )
        save_heatmap(
            diagnostics["az_notch_diff_response"][:24].detach().cpu(),
            stage_dir / "az_notch_diff_response.png",
            f"{spec.title} Ear-Difference Notch Responses",
            xlabel="Detector",
            ylabel="Sample",
        )
        artifacts["az_notch_left_response"] = str(stage_dir / "az_notch_left_response.png")
        artifacts["az_notch_right_response"] = str(stage_dir / "az_notch_right_response.png")
        artifacts["az_notch_diff_response"] = str(stage_dir / "az_notch_diff_response.png")
        artifacts["az_notch_detector_centers"] = _save_line_plot(
            {"Detector Center": diagnostics["az_notch_detector_centers"].detach().cpu().numpy()},
            stage_dir / "az_notch_detector_centers.png",
            "Azimuth Notch Detector Centers",
            "Channel",
        )

    if "orthogonal_elevation_response" in diagnostics:
        save_heatmap(
            diagnostics["orthogonal_elevation_response"][:24].detach().cpu(),
            stage_dir / "orthogonal_elevation_response.png",
            f"{spec.title} Orthogonal Elevation Detector Responses",
            xlabel="Detector",
            ylabel="Sample",
        )
        save_heatmap(
            diagnostics["orthogonal_azimuth_response"][:24].detach().cpu(),
            stage_dir / "orthogonal_azimuth_response.png",
            f"{spec.title} Orthogonal Azimuth Detector Responses",
            xlabel="Detector",
            ylabel="Sample",
        )
        artifacts["orthogonal_elevation_response"] = str(stage_dir / "orthogonal_elevation_response.png")
        artifacts["orthogonal_azimuth_response"] = str(stage_dir / "orthogonal_azimuth_response.png")
        artifacts["orthogonal_elevation_centers"] = _save_line_plot(
            {"Elevation Detector Center": diagnostics["orthogonal_elevation_centers"].detach().cpu().numpy()},
            stage_dir / "orthogonal_elevation_centers.png",
            "Orthogonal Elevation Detector Centers",
            "Channel",
        )
        artifacts["orthogonal_azimuth_centers"] = _save_line_plot(
            {"Azimuth Detector Center": diagnostics["orthogonal_azimuth_centers"].detach().cpu().numpy()},
            stage_dir / "orthogonal_azimuth_centers.png",
            "Orthogonal Azimuth Detector Centers",
            "Channel",
        )

    if "output_az_raw_norm" in diagnostics:
        artifacts["angle_norms"] = _save_line_plot(
            {
                "Azimuth Pair Norm": diagnostics["output_az_raw_norm"].detach().cpu().numpy(),
                "Elevation Pair Norm": diagnostics["output_el_raw_norm"].detach().cpu().numpy(),
            },
            stage_dir / "angle_norms.png",
            "Angle Pair Norms",
            "Norm",
            baseline=1.0,
        )

    return artifacts


def _round3_report(
    outputs_root: Path,
    control: dict[str, Any],
    training_config: EnhancedTrainingConfig,
    config: GlobalConfig,
    params: dict[str, Any],
    results: list[dict[str, Any]],
) -> Path:
    results_by_name = {item["name"]: item for item in results}
    lines = [
        "# Round 3 Experiments",
        "",
        "## Overview",
        "",
        "Round 3 keeps the round-2 combined model as the structural baseline, changes the front-end operating regime to a matched-human 140 dB unnormalized setting, extends the range support to 5 m, and then tests targeted pathway and output-code changes against that same fresh control.",
        "",
        "## Protocol",
        "",
        "- Front end: matched-human",
        f"- Sample rate: `{config.sample_rate_hz}` Hz",
        f"- Chirp: `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` over `{config.chirp_duration_s * 1_000.0:.1f} ms`",
        f"- Cochlea band: `{config.cochlea_low_hz:.0f} -> {config.cochlea_high_hz:.0f} Hz`",
        f"- Cochlea channels: `{int(params['num_frequency_channels'])}`",
        "- Envelope normalization before spikes: `off`",
        f"- Transmit gain: `{config.transmit_gain:.0f}x` (`140 dB` under the `1x = 80 dB` convention)",
        f"- Distance support: `{config.min_range_m:.1f} to {config.max_range_m:.1f} m`",
        "- Azimuth support: `-45 to 45 deg`",
        "- Elevation support: `-30 to 30 deg`",
        "- Split: `700 train / 150 validation / 150 test`",
        f"- Max epochs: `{training_config.max_epochs}`",
        f"- Scheduler: `ReduceLROnPlateau`, patience `{training_config.scheduler_patience}`, factor `{training_config.scheduler_factor}`",
        "- Device: `cpu`",
        "- Thread cap: `1`",
        "",
        "## Experiment 0 Control",
        "",
        f"- Combined error: `{control['test_metrics']['combined_error']:.4f}`",
        f"- Distance MAE: `{control['test_metrics']['distance_mae_m']:.4f} m`",
        f"- Azimuth MAE: `{control['test_metrics']['azimuth_mae_deg']:.4f} deg`",
        f"- Elevation MAE: `{control['test_metrics']['elevation_mae_deg']:.4f} deg`",
        f"- Euclidean error: `{control['test_metrics']['euclidean_error_m']:.4f} m`",
        f"- Runtime: `{control['timings']['total_seconds']:.2f} s` total, `{control['timings']['training_seconds']:.2f} s` training",
        "",
        f"![{control['title']} distance](round_3_experiments/{control['name']}/test_distance_prediction.png)",
        f"![{control['title']} coordinate profile](round_3_experiments/{control['name']}/coordinate_error_profile.png)",
        "",
        "## Results Table",
        "",
        "| Experiment | Combined Error | Euclidean (m) | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Accepted |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['title']} | {item['test_metrics']['combined_error']:.4f} | {item['test_metrics']['euclidean_error_m']:.4f} | "
            f"{item['test_metrics']['distance_mae_m']:.4f} | {item['test_metrics']['azimuth_mae_deg']:.4f} | "
            f"{item['test_metrics']['elevation_mae_deg']:.4f} | {'Yes' if item['accepted'] else 'No'} |"
        )

    lines.extend(["", "## Detailed Experiments", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Change: {item['description']}",
                f"- Rationale: {item['rationale']}",
                f"- Decision vs control: `{'ACCEPTED' if item['accepted'] else 'REJECTED'}`",
                f"- Data variant: `{item.get('data_variant', 'base')}`",
                "",
                "Implementation details:",
            ]
        )
        lines.extend([f"- {step}" for step in item["implemented_steps"]])
        lines.extend(["", "Analysis focus:"])
        lines.extend([f"- {step}" for step in item["analysis_focus"]])
        lines.extend(
            [
                "",
                "Metrics:",
                f"- Combined error: `{item['test_metrics']['combined_error']:.4f}`",
                f"- Distance MAE: `{item['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Azimuth MAE: `{item['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Elevation MAE: `{item['test_metrics']['elevation_mae_deg']:.4f} deg`",
                f"- Euclidean error: `{item['test_metrics']['euclidean_error_m']:.4f} m`",
                "",
                "Delta vs control:",
                f"- Combined error delta: `{item['comparison']['combined_error_delta']:.4f}`",
                f"- Distance MAE delta: `{item['comparison']['distance_mae_delta']:.4f}`",
                f"- Azimuth MAE delta: `{item['comparison']['azimuth_mae_delta']:.4f}`",
                f"- Elevation MAE delta: `{item['comparison']['elevation_mae_delta']:.4f}`",
                f"- Euclidean delta: `{item['cartesian_delta']['euclidean_error_delta']:.4f} m`",
                "",
                "Timing:",
                f"- Training: `{item['timings']['training_seconds']:.2f} s`",
                f"- Evaluation: `{item['timings']['evaluation_seconds']:.2f} s`",
                f"- Total: `{item['timings']['total_seconds']:.2f} s`",
                "",
                f"![{item['title']} loss](round_3_experiments/{item['name']}/loss.png)",
                f"![{item['title']} comparison](round_3_experiments/{item['name']}/comparison.png)",
                f"![{item['title']} distance](round_3_experiments/{item['name']}/test_distance_prediction.png)",
                f"![{item['title']} coordinate profile](round_3_experiments/{item['name']}/coordinate_error_profile.png)",
            ]
        )
        if item["artifacts"].get("lif_distance_left_spikes"):
            lines.append(f"![{item['title']} LIF distance spikes](round_3_experiments/{item['name']}/lif_distance_left_spikes.png)")
        if item["artifacts"].get("lif_itd_spikes"):
            lines.append(f"![{item['title']} LIF ITD spikes](round_3_experiments/{item['name']}/lif_itd_spikes.png)")
        if item["artifacts"].get("lif_betas"):
            lines.append(f"![{item['title']} LIF betas](round_3_experiments/{item['name']}/lif_betas.png)")
        if item["artifacts"].get("comb_response"):
            lines.extend(
                [
                    "",
                    "Comb-filter explanation:",
                    "- This is not a temporal comb filter on the waveform.",
                    "- It is a frequency-channel operator applied to the normalized binaural spike-count spectrum.",
                    "- For each lag `k`, it computes `|x[c] - 0.5 * (x[c-k] + x[c+k])|`, so a channel is large when it differs from its symmetric neighbours.",
                    "- Lags `2`, `4`, and `6` are averaged, giving a richer spectral-periodicity cue than the previous simple slope term.",
                ]
            )
            lines.append(f"![{item['title']} comb response](round_3_experiments/{item['name']}/comb_response.png)")
        if item["artifacts"].get("comb_filter_operator"):
            lines.append(f"![{item['title']} comb operator](round_3_experiments/{item['name']}/comb_filter_operator.png)")
        if item["artifacts"].get("moving_notch_cue"):
            lines.extend(
                [
                    "",
                    "Moving-notch cue explanation:",
                    "- The simulator keeps the original elevation-dependent slope cue and multiplies it by a Gaussian spectral notch.",
                    "- The notch center shifts smoothly across frequency with elevation, so lower elevations suppress lower-frequency channels and higher elevations suppress higher-frequency channels.",
                    "- This cue is injected upstream of the cochlea, so it changes the spike distribution before the elevation branch sees the input.",
                ]
            )
            lines.append(f"![{item['title']} moving notch cue](round_3_experiments/{item['name']}/moving_notch_cue.png)")
        if item["artifacts"].get("azimuth_moving_notch_cue"):
            if item.get("data_variant") in {"azimuth_notch_only", "orthogonal_combined_notches"}:
                lines.extend(
                    [
                        "",
                        "Azimuth moving-notch cue explanation:",
                        "- The simulator applies mirrored ear-specific moving notches before the cochlea, without adding any extra azimuth spectral slope.",
                        "- At positive azimuth, the left and right ears receive different notch locations; at negative azimuth the pattern mirrors.",
                        "- This augments the existing ITD and ILD cues with ear-specific spectral asymmetry while trying to reduce interference from added tilt.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "Azimuth moving-notch cue explanation:",
                        "- The simulator applies mirrored ear-specific slope-plus-moving-notch spectral cues before the cochlea.",
                        "- At positive azimuth, the left and right ears receive different notch locations and slope tilts; at negative azimuth the pattern mirrors.",
                        "- This augments the existing ITD and ILD cues with ear-specific spectral asymmetry.",
                    ]
                )
            lines.append(f"![{item['title']} azimuth moving notch cue](round_3_experiments/{item['name']}/azimuth_moving_notch_cue.png)")
        if item["artifacts"].get("orthogonal_elevation_response"):
            lines.extend(
                [
                    "",
                    "Orthogonal decoder explanation:",
                    "- Elevation is decoded only from the binaural-mean notch profile, using a mid-band detector bank.",
                    "- Azimuth is decoded only from ear-difference notch features, using an edge-focused detector bank.",
                    "- This keeps the two angular tasks on different spectral statistics instead of letting both read the same raw spectrum.",
                ]
            )
            lines.append(f"![{item['title']} orthogonal elevation responses](round_3_experiments/{item['name']}/orthogonal_elevation_response.png)")
            lines.append(f"![{item['title']} orthogonal azimuth responses](round_3_experiments/{item['name']}/orthogonal_azimuth_response.png)")
            lines.append(f"![{item['title']} orthogonal elevation centers](round_3_experiments/{item['name']}/orthogonal_elevation_centers.png)")
            lines.append(f"![{item['title']} orthogonal azimuth centers](round_3_experiments/{item['name']}/orthogonal_azimuth_centers.png)")
        if item["name"] == "round3_experiment_3c_orthogonal_combined_notches":
            comparisons = [
                ("Control", control),
                ("Elevation 2A", results_by_name.get("round3_experiment_2a_moving_notch_cue")),
                ("Elevation 2B", results_by_name.get("round3_experiment_2b_moving_notch_plus_detectors")),
                ("Azimuth 3A", results_by_name.get("round3_experiment_3a_azimuth_moving_notch_detectors")),
                ("Azimuth 3B", results_by_name.get("round3_experiment_3b_azimuth_notch_only_detectors")),
            ]
            lines.extend(["", "Angle comparison against prior notch models:"])
            for label, other in comparisons:
                if other is None:
                    continue
                az_delta = item["test_metrics"]["azimuth_mae_deg"] - other["test_metrics"]["azimuth_mae_deg"]
                el_delta = item["test_metrics"]["elevation_mae_deg"] - other["test_metrics"]["elevation_mae_deg"]
                combined_delta = item["test_metrics"]["combined_error"] - other["test_metrics"]["combined_error"]
                lines.append(
                    f"- vs {label}: combined delta `{combined_delta:.4f}`, azimuth delta `{az_delta:.4f} deg`, elevation delta `{el_delta:.4f} deg`"
                )
        if item["artifacts"].get("notch_detector_response"):
            lines.extend(
                [
                    "",
                    "Notch-detector explanation:",
                    "- The elevation branch computes the spike-domain spectral notch profile and applies a fixed bank of Gaussian detectors across channel position.",
                    "- Each detector reports how much notch energy is present near one channel region, turning notch location into an explicit feature vector.",
                    "- That detector-response vector is projected into a learned residual added only to the elevation latent.",
                ]
            )
            lines.append(f"![{item['title']} notch detector responses](round_3_experiments/{item['name']}/notch_detector_response.png)")
        if item["artifacts"].get("notch_detector_centers"):
            lines.append(f"![{item['title']} notch detector centers](round_3_experiments/{item['name']}/notch_detector_centers.png)")
        if item["artifacts"].get("az_notch_left_response"):
            lines.extend(
                [
                    "",
                    "Azimuth notch-detector explanation:",
                    "- The azimuth branch now gets a third residual stream built from ear-specific spectral notch profiles.",
                    "- A fixed detector bank scans notch energy across channel position separately for the left and right ears.",
                    "- Left, right, and ear-difference detector responses are projected into an azimuth-only residual latent and fused with the existing ITD and ILD features.",
                ]
            )
            lines.append(f"![{item['title']} left azimuth notch responses](round_3_experiments/{item['name']}/az_notch_left_response.png)")
        if item["artifacts"].get("az_notch_right_response"):
            lines.append(f"![{item['title']} right azimuth notch responses](round_3_experiments/{item['name']}/az_notch_right_response.png)")
        if item["artifacts"].get("az_notch_diff_response"):
            lines.append(f"![{item['title']} azimuth notch difference responses](round_3_experiments/{item['name']}/az_notch_diff_response.png)")
        if item["artifacts"].get("az_notch_detector_centers"):
            lines.append(f"![{item['title']} azimuth notch detector centers](round_3_experiments/{item['name']}/az_notch_detector_centers.png)")
        if item["artifacts"].get("angle_norms"):
            lines.append(f"![{item['title']} angle norms](round_3_experiments/{item['name']}/angle_norms.png)")
        lines.append("")

    accepted = [item["title"] for item in results if item["accepted"]]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Accepted experiments: {', '.join(accepted) if accepted else 'none'}",
            "- Round 3 uses the fresh 5 m, 140 dB, unnormalized control as the only reference.",
            "- Any accepted variant should be rerun on a longer schedule before it replaces the current short-run baseline.",
        ]
    )
    report_path = outputs_root / "round_3_experiments_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_round_3_experiments(config: GlobalConfig, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=4,
        scheduler_patience=2,
    )
    effective_config = _round3_base_config(config)
    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    base_params, baseline_label = _baseline_reference_params(context)
    support_spec = _round3_support_spec()
    seed_everything(effective_config.seed)

    stage_root = outputs.root / "round_3_experiments"
    stage_root.mkdir(parents=True, exist_ok=True)

    data_cache: dict[str, dict[str, Any]] = {}

    def get_variant_payload(data_variant: str) -> dict[str, Any]:
        if data_variant in data_cache:
            return data_cache[data_variant]
        variant_config = _round3_variant_config(effective_config, data_variant)
        variant_context = StageContext(config=variant_config, device=context.device, outputs=outputs, shared=context.shared)
        prep_start = time.perf_counter()
        variant_data = _prepare_expanded_data(variant_context, base_params, support_spec, chunk_size=16)
        payload = {
            "data": variant_data,
            "baseline_bundle": _prepare_target_bundle(variant_data),
            "sincos_bundle": _prepare_sincos_target_bundle(variant_data),
            "distance01_bundle": _prepare_distance01_target_bundle(variant_data, variant_data.local_config),
            "data_prep_seconds": time.perf_counter() - prep_start,
        }
        data_cache[data_variant] = payload
        return payload

    get_variant_payload("base")

    results: list[dict[str, Any]] = []
    control_result: dict[str, Any] | None = None

    for index, spec in enumerate(_round3_specs(), start=1):
        existing_result_path = stage_root / spec.name / "result.json"
        if existing_result_path.exists():
            existing_result = json.loads(existing_result_path.read_text(encoding="utf-8"))
            if control_result is None:
                control_result = existing_result
            else:
                results.append(existing_result)
            continue

        variant_payload = get_variant_payload(spec.data_variant)
        data: PreparedExperimentData = variant_payload["data"]
        baseline_bundle: TargetBundle = variant_payload["baseline_bundle"]
        sincos_bundle: SinCosTargetBundle = variant_payload["sincos_bundle"]
        distance01_bundle: Distance01TargetBundle = variant_payload["distance01_bundle"]
        data_prep_seconds = float(variant_payload["data_prep_seconds"])

        seed_everything(effective_config.seed + index)
        print(f"[round_3] running {spec.name} on cpu", flush=True)
        model = _instantiate_round3_model(data, base_params, spec)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(base_params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_config.scheduler_factor,
            patience=training_config.scheduler_patience,
            threshold=training_config.scheduler_threshold,
            min_lr=training_config.scheduler_min_lr,
        )

        batch_size = int(spec.training_overrides.get("batch_size", 8))
        train_loss_history: list[float] = []
        val_loss_history: list[float] = []
        best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
        best_val_combined = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        total_start = time.perf_counter()
        training_start = time.perf_counter()
        for epoch in range(training_config.max_epochs):
            model.train()
            permutation = torch.randperm(data.train_targets_raw.shape[0], device=data.train_targets_raw.device)
            batch_losses: list[float] = []
            for start in range(0, data.train_targets_raw.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                batch_features: ExperimentBatch = data.train_batch.index_select(indices)
                batch_targets_raw = data.train_targets_raw[indices]
                batch_baseline_model = baseline_bundle.train_model[indices]
                batch_sincos_distance_model = sincos_bundle.train_distance_model[indices]
                batch_distance01 = distance01_bundle.train_distance01[indices]
                batch_distance01_angle_model = distance01_bundle.train_angle_model[indices]

                optimizer.zero_grad(set_to_none=True)
                output_model, diagnostics = model(batch_features)
                loss, _, _ = _round3_loss_and_decode(
                    spec,
                    output_model,
                    batch_targets_raw,
                    diagnostics,
                    data.local_config,
                    base_params,
                    baseline_bundle,
                    batch_baseline_model,
                    sincos_bundle,
                    batch_sincos_distance_model,
                    distance01_bundle,
                    batch_distance01,
                    batch_distance01_angle_model,
                )
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))

            model.eval()
            with torch.no_grad():
                val_output_model, val_diagnostics = model(data.val_batch)
                val_loss, val_raw_prediction, val_summary = _round3_loss_and_decode(
                    spec,
                    val_output_model,
                    data.val_targets_raw,
                    val_diagnostics,
                    data.local_config,
                    base_params,
                    baseline_bundle,
                    baseline_bundle.val_model,
                    sincos_bundle,
                    sincos_bundle.val_distance_model,
                    distance01_bundle,
                    distance01_bundle.val_distance01,
                    distance01_bundle.val_angle_model,
                )
                val_eval = _evaluate_round3_output(val_raw_prediction, data.val_targets_raw, val_diagnostics, data.local_config)
            val_loss_history.append(float(val_loss.item()))
            scheduler.step(float(val_eval.metrics["combined_error"]))

            if float(val_eval.metrics["combined_error"]) < best_val_combined - training_config.early_stopping_min_delta:
                best_val_combined = float(val_eval.metrics["combined_error"])
                best_epoch = epoch
                best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= training_config.early_stopping_patience:
                break

        training_seconds = time.perf_counter() - training_start
        model.load_state_dict(best_state)

        evaluation_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            test_output_model, test_diagnostics = model(data.test_batch)
            _, test_raw_prediction, test_summary = _round3_loss_and_decode(
                spec,
                test_output_model,
                data.test_targets_raw,
                test_diagnostics,
                data.local_config,
                base_params,
                baseline_bundle,
                baseline_bundle.test_model,
                sincos_bundle,
                sincos_bundle.test_distance_model,
                distance01_bundle,
                distance01_bundle.test_distance01,
                distance01_bundle.test_angle_model,
            )
            if spec.output_mode == "sincos":
                _, decoded_aux = _decode_sincos_output(test_output_model, sincos_bundle)
                test_diagnostics = {
                    **test_diagnostics,
                    "output_az_raw_norm": decoded_aux["az_raw_norm"].detach(),
                    "output_el_raw_norm": decoded_aux["el_raw_norm"].detach(),
                }
            test_eval = _evaluate_round3_output(test_raw_prediction, data.test_targets_raw, test_diagnostics, data.local_config)
        evaluation_seconds = time.perf_counter() - evaluation_start
        total_seconds = time.perf_counter() - total_start

        if control_result is None:
            control_metrics = test_eval.metrics
        else:
            control_metrics = control_result["test_metrics"]

        comparison = _metrics_delta(test_eval.metrics, control_metrics)
        cartesian_delta = {
            "x_mae_delta": float(test_eval.metrics["x_mae_m"] - control_metrics["x_mae_m"]),
            "y_mae_delta": float(test_eval.metrics["y_mae_m"] - control_metrics["y_mae_m"]),
            "z_mae_delta": float(test_eval.metrics["z_mae_m"] - control_metrics["z_mae_m"]),
            "euclidean_error_delta": float(test_eval.metrics["euclidean_error_m"] - control_metrics["euclidean_error_m"]),
        }
        accepted = False if control_result is None else _is_accepted(test_eval.metrics, control_metrics)
        artifacts = _save_round3_outputs(
            stage_root,
            spec,
            train_loss_history,
            val_loss_history,
            test_eval,
            control_metrics,
            test_diagnostics,
            data.local_config,
        )
        stage_dir = stage_root / spec.name
        prediction_cache = _save_prediction_cache(stage_dir, test_eval.predictions, data.test_targets_raw)
        coordinate_profile = _save_coordinate_error_profiles(
            Path(prediction_cache),
            stage_dir / "coordinate_error_profile.png",
            f"{spec.title} Coordinate Error Profile",
        )

        result = {
            "name": spec.name,
            "title": spec.title,
            "description": spec.description,
            "rationale": spec.rationale,
            "implemented_steps": spec.implemented_steps,
            "analysis_focus": spec.analysis_focus,
            "output_mode": spec.output_mode,
            "data_variant": spec.data_variant,
            "accepted": accepted,
            "decision": "CONTROL" if control_result is None else ("ACCEPTED" if accepted else "REJECTED"),
            "baseline_label": baseline_label,
            "test_metrics": {key: float(value) for key, value in test_eval.metrics.items()},
            "comparison": {key: float(value) for key, value in comparison.items()},
            "cartesian_delta": {key: float(value) for key, value in cartesian_delta.items()},
            "training": {
                "best_epoch": best_epoch + 1,
                "executed_epochs": len(train_loss_history),
            },
            "timings": {
                "data_prep_seconds": float(data_prep_seconds),
                "training_seconds": float(training_seconds),
                "evaluation_seconds": float(evaluation_seconds),
                "total_seconds": float(total_seconds),
            },
            "loss_summary": test_summary,
            "artifacts": {
                **artifacts,
                "prediction_cache": prediction_cache,
                "coordinate_error_profile": coordinate_profile,
            },
        }
        save_json(stage_dir / "result.json", result)
        if control_result is None:
            control_result = result
        else:
            results.append(result)

    assert control_result is not None

    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error", "Euclidean"],
        {
            "Control": [
                float(control_result["test_metrics"]["distance_mae_m"]),
                float(control_result["test_metrics"]["azimuth_mae_deg"]),
                float(control_result["test_metrics"]["elevation_mae_deg"]),
                float(control_result["test_metrics"]["combined_error"]),
                float(control_result["test_metrics"]["euclidean_error_m"]),
            ],
            **{
                item["title"]: [
                    float(item["test_metrics"]["distance_mae_m"]),
                    float(item["test_metrics"]["azimuth_mae_deg"]),
                    float(item["test_metrics"]["elevation_mae_deg"]),
                    float(item["test_metrics"]["combined_error"]),
                    float(item["test_metrics"]["euclidean_error_m"]),
                ]
                for item in results
            },
        },
        stage_root / "overall_comparison.png",
        "Round 3 vs Control",
        ylabel="Error",
    )

    report_path = _round3_report(outputs.root, control_result, training_config, effective_config, base_params, results)
    summary = {
        "protocol": {
            "sample_rate_hz": effective_config.sample_rate_hz,
            "transmit_gain": effective_config.transmit_gain,
            "normalize_spike_envelope": effective_config.normalize_spike_envelope,
            "max_range_m": effective_config.max_range_m,
        },
        "control": control_result,
        "experiments": results,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_3_experiments_summary.json", summary)
    save_json(stage_root / "results.json", summary)
    return summary
