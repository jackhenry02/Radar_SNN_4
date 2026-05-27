from __future__ import annotations

"""Trainable SNN readout smoke test for the final integrated model.

This experiment keeps the hand-built distance, azimuth, and elevation pathways
fixed. It caches their raw population codes plus CANN readouts, then trains a
small snnTorch readout to predict final 3D coordinates in the constrained
0.25-5 m, +/-45 degree operating space.
"""

import json
import math
import sys
import time
import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azimuth_pathway.experiments import azimuth_ild_line_attractor as azc
from azimuth_pathway.experiments import azimuth_pathway_first_attempt as az
from distance_pathway.experiments import final_distance_pipeline_with_attractor as dist_cann
from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_line_attractor as elc
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from final_model.experiments import environment_noise_diagnostics as envdiag
from final_model.experiments import final_model_results as final
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "trainable_readout"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "final_model" / "reports" / "trainable_final_readout.md"
COMPARISON_REPORT_PATH = ROOT / "final_model" / "reports" / "trainable_final_readout_comparison.md"
CACHE_PATH = OUTPUT_DIR / "smoke_cache_constrained_0p25_5m_pm45.npz"
RESULTS_PATH = OUTPUT_DIR / "smoke_results.json"
RUN_LABEL = "smoke"

DISTANCE_MIN_M = 0.25
DISTANCE_MAX_M = 5.0
AZIMUTH_LIMIT_DEG = 45.0
ELEVATION_LIMIT_DEG = 45.0
CHANNELS = 48
DISTANCE_BINS = 180
ANGULAR_BINS = 91
SMOKE_SPLITS = {"train": 48, "val": 16, "test": 16}
DATASET_SEED = 8_101
TRAINING_SEED = 8_202
NUM_STEPS = 12
HIDDEN_DIM = 96
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RESIDUAL_SCALE = 0.15
FORCE_CACHE = False
NOISE_STD = 0.0
NOISE_LABEL = "clean"
ACOUSTIC_MODE = "clean"
ENVIRONMENT_NOISE_DB = 0.0
ENVIRONMENT_REVERB = False


def configure_run(args: argparse.Namespace) -> None:
    """Apply command-line options to module-level experiment settings.

    Args:
        args: Parsed command-line arguments.
    """
    global SMOKE_SPLITS
    global DATASET_SEED
    global TRAINING_SEED
    global HIDDEN_DIM
    global BATCH_SIZE
    global EPOCHS
    global LEARNING_RATE
    global FORCE_CACHE
    global NOISE_STD
    global NOISE_LABEL
    global ACOUSTIC_MODE
    global ENVIRONMENT_NOISE_DB
    global ENVIRONMENT_REVERB
    global RUN_LABEL
    global FIGURE_DIR
    global CACHE_PATH
    global RESULTS_PATH

    SMOKE_SPLITS = {"train": args.train, "val": args.val, "test": args.test}
    DATASET_SEED = args.dataset_seed
    TRAINING_SEED = args.training_seed
    HIDDEN_DIM = args.hidden
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    FORCE_CACHE = args.force_cache
    if args.environment_noise_db is not None and (args.noise_std is not None or args.noise_db_spl is not None):
        raise ValueError("Use either receiver-noise flags or --environment-noise-db, not both.")
    if args.environment_reverb and args.environment_noise_db is None:
        raise ValueError("--environment-reverb requires --environment-noise-db.")
    if args.noise_std is not None and args.noise_db_spl is not None:
        raise ValueError("Use either --noise-std or --noise-db-spl, not both.")
    if args.environment_noise_db is not None:
        NOISE_STD = 0.0
        ACOUSTIC_MODE = "environment_noise_reverb" if args.environment_reverb else "environment_noise"
        ENVIRONMENT_NOISE_DB = float(args.environment_noise_db)
        ENVIRONMENT_REVERB = bool(args.environment_reverb)
        base = f"envnoise{int(round(ENVIRONMENT_NOISE_DB))}dB"
        NOISE_LABEL = f"{base}_reverb" if ENVIRONMENT_REVERB else base
    elif args.noise_db_spl is not None:
        NOISE_STD = fdm._noise_std_from_db(float(args.noise_db_spl))
        NOISE_LABEL = f"noise{int(round(float(args.noise_db_spl)))}dB"
        ACOUSTIC_MODE = "receiver_noise"
        ENVIRONMENT_NOISE_DB = 0.0
        ENVIRONMENT_REVERB = False
    elif args.noise_std is not None:
        NOISE_STD = float(args.noise_std)
        NOISE_LABEL = f"noise_std_{str(args.noise_std).replace('.', 'p')}"
        ACOUSTIC_MODE = "receiver_noise"
        ENVIRONMENT_NOISE_DB = 0.0
        ENVIRONMENT_REVERB = False
    else:
        NOISE_STD = 0.0
        NOISE_LABEL = "clean"
        ACOUSTIC_MODE = "clean"
        ENVIRONMENT_NOISE_DB = 0.0
        ENVIRONMENT_REVERB = False
    base_label = f"train{args.train}_val{args.val}_test{args.test}"
    RUN_LABEL = base_label if NOISE_LABEL == "clean" else f"{base_label}_{NOISE_LABEL}"
    FIGURE_DIR = OUTPUT_DIR / "figures" / RUN_LABEL
    CACHE_PATH = OUTPUT_DIR / f"cache_constrained_0p25_5m_pm45_{RUN_LABEL}.npz"
    RESULTS_PATH = OUTPUT_DIR / f"results_{RUN_LABEL}.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for smoke or long training runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=48, help="Number of training samples.")
    parser.add_argument("--val", type=int, default=16, help="Number of validation samples.")
    parser.add_argument("--test", type=int, default=16, help="Number of held-out test samples.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs for each SNN readout.")
    parser.add_argument("--hidden", type=int, default=96, help="Hidden neurons in each SNN readout layer.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--dataset-seed", type=int, default=8_101, help="Random seed for target sampling.")
    parser.add_argument("--training-seed", type=int, default=8_202, help="Random seed for SNN training.")
    parser.add_argument("--force-cache", action="store_true", help="Regenerate features even if the matching cache exists.")
    parser.add_argument("--noise-db-spl", type=float, default=None, help="Add fixed receiver noise converted from an effective dB SPL value.")
    parser.add_argument("--noise-std", type=float, default=None, help="Add fixed receiver noise with this waveform standard deviation.")
    parser.add_argument(
        "--environment-noise-db",
        type=float,
        default=None,
        help="Add call-referenced environmental noise before head-shadow/elevation filtering.",
    )
    parser.add_argument(
        "--environment-reverb",
        action="store_true",
        help="Add delayed echo copies in the environmental-noise simulator.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class FeatureSpec:
    """Feature-vector layout.

    Attributes:
        groups: Mapping from group name to half-open index range.
        input_dim: Total feature dimension.
    """

    groups: dict[str, tuple[int, int]]
    input_dim: int


class UncertaintyWeightedLoss(nn.Module):
    """Learned uncertainty-weighted coordinate loss.

    The output is `[distance_norm, az_sin, az_cos, el_sin, el_cos]`. Distance,
    azimuth, and elevation are treated as three tasks with learned log
    variances.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the multitask loss.

        Args:
            prediction: Predicted normalised coordinate tensor `[batch, 5]`.
            target: Target normalised coordinate tensor `[batch, 5]`.

        Returns:
            Scalar training loss.
        """
        losses = torch.stack(
            [
                torch.mean((prediction[:, 0] - target[:, 0]) ** 2),
                torch.mean((prediction[:, 1:3] - target[:, 1:3]) ** 2),
                torch.mean((prediction[:, 3:5] - target[:, 3:5]) ** 2),
            ]
        )
        precision = torch.exp(-self.log_vars)
        return torch.sum(0.5 * precision * losses + 0.5 * self.log_vars)


class SmallSNNReadout(nn.Module):
    """Small snnTorch regression readout for cached pathway features."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_steps: int) -> None:
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()
        beta_in = torch.full((hidden_dim,), 0.85)
        beta_hidden = torch.full((hidden_dim,), 0.88)
        beta_out = torch.full((output_dim,), 0.92)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=0.9, learn_beta=True, spike_grad=spike_grad)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=0.9, learn_beta=True, spike_grad=spike_grad)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.li_out = snn.Leaky(
            beta=beta_out,
            threshold=1.0,
            learn_beta=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the readout for a fixed number of recurrent timesteps.

        Args:
            features: Normalised feature tensor `[batch, input_dim]`.

        Returns:
            Mean output membrane potential `[batch, output_dim]`.
        """
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_out = self.li_out.init_leaky()
        outputs = []
        for _ in range(self.num_steps):
            current_1 = self.fc_in(features)
            spikes_1, mem_1 = self.lif_in(current_1, mem_1)
            current_2 = self.fc_hidden(spikes_1)
            spikes_2, mem_2 = self.lif_hidden(current_2, mem_2)
            current_out = self.fc_out(spikes_2)
            _, mem_out = self.li_out(current_out, mem_out)
            outputs.append(mem_out)
        return torch.stack(outputs, dim=0).mean(dim=0)


def normalise_population(values: np.ndarray) -> np.ndarray:
    """Normalise a population by its maximum absolute value."""
    values = np.asarray(values, dtype=np.float64)
    return values / max(float(np.max(np.abs(values))), 1e-12)


def margin(population: np.ndarray) -> float:
    """Return winner-minus-runner-up margin for a population code."""
    values = np.sort(np.asarray(population, dtype=np.float64).reshape(-1))
    if values.size < 2:
        return 0.0
    return float(values[-1] - values[-2])


def entropy(population: np.ndarray) -> float:
    """Return normalised entropy of a non-negative population."""
    positive = np.maximum(np.asarray(population, dtype=np.float64), 0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return 0.0
    probs = positive / total
    return float(-(probs * np.log(probs + 1e-12)).sum() / math.log(max(2, probs.size)))


def angle_to_sincos(angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert angles in degrees to sine/cosine coordinates."""
    rad = np.deg2rad(angle_deg)
    return np.sin(rad), np.cos(rad)


def encode_targets(distance_m: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Encode target coordinates for regression."""
    distance_norm = (distance_m - DISTANCE_MIN_M) / (DISTANCE_MAX_M - DISTANCE_MIN_M)
    az_sin, az_cos = angle_to_sincos(azimuth_deg)
    el_sin, el_cos = angle_to_sincos(elevation_deg)
    return np.stack([distance_norm, az_sin, az_cos, el_sin, el_cos], axis=1).astype(np.float32)


def decode_outputs(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode model output coordinates back to metres/degrees."""
    distance = np.clip(encoded[:, 0], 0.0, 1.0) * (DISTANCE_MAX_M - DISTANCE_MIN_M) + DISTANCE_MIN_M
    azimuth = np.rad2deg(np.arctan2(encoded[:, 1], encoded[:, 2]))
    elevation = np.rad2deg(np.arctan2(encoded[:, 3], encoded[:, 4]))
    return distance, azimuth, elevation


def build_feature_spec() -> FeatureSpec:
    """Build the fixed smoke-test feature layout."""
    groups: dict[str, tuple[int, int]] = {}
    start = 0
    for name, size in [
        ("raw_distance_population", DISTANCE_BINS),
        ("raw_azimuth_itd_population", ANGULAR_BINS),
        ("raw_azimuth_ild_population", ANGULAR_BINS),
        ("raw_elevation_population", ANGULAR_BINS),
        ("cann_readouts", 3),
        ("confidence_features", 18),
    ]:
        groups[name] = (start, start + size)
        start += size
    return FeatureSpec(groups=groups, input_dim=start)


def random_targets(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample constrained-space targets."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M, size=samples)
    azimuth = rng.uniform(-AZIMUTH_LIMIT_DEG, AZIMUTH_LIMIT_DEG, size=samples)
    elevation = rng.uniform(-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG, size=samples)
    return distance, azimuth, elevation


def environment_condition() -> envdiag.AcousticCondition:
    """Return the environmental acoustic condition for the current run."""
    return envdiag.AcousticCondition(
        key=ACOUSTIC_MODE,
        name=NOISE_LABEL,
        add_environment_noise=ACOUSTIC_MODE in {"environment_noise", "environment_noise_reverb"},
        add_reverb=ENVIRONMENT_REVERB,
    )


def azimuth_features_from_receive(config, receive: torch.Tensor, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ITD/ILD azimuth populations from one shared binaural waveform."""
    cochlea = fdm._run_cochlea_binaural(config, receive)
    left_spikes, right_spikes = az.run_dynamic_cochlea_spikes(cochlea, config)
    vcn_left = az.vcn_consensus_single_ear(left_spikes, config)
    vcn_right = az.vcn_consensus_single_ear(right_spikes, config)
    itd = az.jeffress_lif_itd_activation(vcn_left, vcn_right, config, bins)
    ild, _, _, _, _ = az.lso_mntb_ild_activation(left_spikes, right_spikes, bins)
    azimuth_cann, _, _, _ = azc.run_cann_readout(itd[None, :], bins)
    return itd, ild, azimuth_cann


def elevation_features_from_receive(
    config,
    receive: torch.Tensor,
    azimuth_deg: float,
    baseline_profile: np.ndarray,
    gain_matrix: np.ndarray,
    centres_hz: np.ndarray,
    elevation_bins: np.ndarray,
    calibration_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Compute the elevation population and CANN readout from one shared waveform."""
    cochlea = fdm._run_cochlea_binaural(config, receive)
    selected_cochleagram = cochlea.right_cochleagram if azimuth_deg >= 0.0 else cochlea.left_cochleagram
    selected_spikes = fdm._dynamic_lif_encode(selected_cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
    profile = elev.dynamic_wideband_inhibited_profile(
        selected_cochleagram,
        selected_spikes,
        elev.DynamicInhibitionParams(),
    )
    equalized = profile / np.maximum(baseline_profile, 1e-4)
    equalized = equalized / np.maximum(equalized.max(), 1e-12)
    elevation_population = elev.dcn_signal_weighted_transfer_response(
        equalized,
        baseline_profile,
        gain_matrix,
        centres_hz,
    )
    elevation_attractor = elc.run_attractor_variants(elevation_population[None, :], elevation_bins)
    elevation_raw = np.asarray(elevation_attractor["reflected"]["prediction"], dtype=np.float64)
    elevation_cann = elc.apply_tuned_inverse_sigmoid(elevation_raw, calibration_params)
    return elevation_population, elevation_cann, selected_spikes


def build_cached_features(force: bool = False) -> dict[str, object]:
    """Generate or load the smoke-test feature cache.

    Args:
        force: If true, regenerate the cache even if it already exists.

    Returns:
        Cache payload with features, targets, baseline outputs, and metadata.
    """
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached features: {CACHE_PATH}")
        loaded = np.load(CACHE_PATH, allow_pickle=True)
        return {key: loaded[key].item() if loaded[key].shape == () else loaded[key] for key in loaded.files}

    ensure_dir(OUTPUT_DIR)
    spec = build_feature_spec()
    total = sum(SMOKE_SPLITS.values())
    print(f"Generating feature cache: {total} samples -> {CACHE_PATH}")
    distance_true, azimuth_true, elevation_true = random_targets(total, DATASET_SEED)
    split_names = np.array(
        ["train"] * SMOKE_SPLITS["train"] + ["val"] * SMOKE_SPLITS["val"] + ["test"] * SMOKE_SPLITS["test"]
    )

    with final.temporary_model_constants(
        channels=CHANNELS,
        distance_bins=DISTANCE_BINS,
        angular_bins=ANGULAR_BINS,
        elevation_limit_deg=ELEVATION_LIMIT_DEG,
    ):
        add_noise = ACOUSTIC_MODE == "receiver_noise" and NOISE_STD > 0.0
        distance_config = replace(
            final.make_distance_config(CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M),
            noise_std=NOISE_STD,
        )
        distance_variant = final.make_distance_variant(distance_config, CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M)
        azimuth_config = replace(final.make_azimuth_config(CHANNELS, DISTANCE_MAX_M), noise_std=NOISE_STD)
        elevation_config = replace(
            final.make_elevation_config(CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M),
            noise_std=NOISE_STD,
        )
        elevation_params, elevation_baseline = final.tune_elevation_calibration(elevation_config)
        distance_bins = fdm._candidate_distances(distance_config)
        azimuth_bins = az.azimuth_grid(AZIMUTH_LIMIT_DEG)
        elevation_bins = elev.elevation_grid()
        centres_hz, elevation_gain_matrix, _ = elev.build_dcn_templates(
            elevation_config,
            elevation_bins,
            elev.DEEP_COMB_DELAYED_COPY_GAIN,
        )
        env_noise_std = (
            envdiag._call_referenced_noise_std(distance_config, ENVIRONMENT_NOISE_DB)
            if ACOUSTIC_MODE in {"environment_noise", "environment_noise_reverb"}
            else 0.0
        )
        env_condition = environment_condition()

        features = np.zeros((total, spec.input_dim), dtype=np.float32)
        baseline_encoded = np.zeros((total, 5), dtype=np.float32)
        baseline_coordinates = np.zeros((total, 3), dtype=np.float32)
        feature_seconds = []

        for index, (distance_m, azimuth_deg, elevation_deg) in enumerate(
            zip(distance_true, azimuth_true, elevation_true)
        ):
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == total:
                print(f"  cached {index + 1}/{total} samples")
            start = time.perf_counter()

            if ACOUSTIC_MODE in {"environment_noise", "environment_noise_reverb"}:
                waveform_rng = torch.Generator().manual_seed(DATASET_SEED + 50_000 + index)
                receive = envdiag._simulate_environment_echo(
                    distance_config,
                    float(distance_m),
                    float(azimuth_deg),
                    float(elevation_deg),
                    condition=env_condition,
                    noise_std=env_noise_std,
                    rng=waveform_rng,
                )
                _, distance_cann, distance_ac = envdiag._distance_prediction_from_receive(
                    distance_config,
                    distance_variant,
                    receive,
                )
                distance_cann = np.asarray([distance_cann], dtype=np.float64)
                distance_population = normalise_population(distance_ac)
                distance_cochlea = fdm._run_cochlea_binaural(distance_config, receive)

                itd_population, ild_population, azimuth_cann = azimuth_features_from_receive(
                    azimuth_config,
                    receive,
                    azimuth_bins,
                )
                azimuth_itd_population = normalise_population(itd_population)
                azimuth_ild_population = normalise_population(ild_population)

                elevation_population, elevation_cann, selected_spikes = elevation_features_from_receive(
                    elevation_config,
                    receive,
                    float(azimuth_deg),
                    elevation_baseline,
                    elevation_gain_matrix,
                    centres_hz,
                    elevation_bins,
                    elevation_params,
                )
            else:
                distance_prediction = fdm._predict_one_3d(
                    distance_config,
                    float(distance_m),
                    float(azimuth_deg),
                    float(elevation_deg),
                    distance_variant,
                    add_noise=add_noise,
                )
                distance_ac = distance_prediction.ac_activation
                distance_population = normalise_population(distance_ac)
                distance_cann, _, _, _ = dist_cann.run_line_attractor(
                    distance_ac[None, :],
                    distance_bins,
                    dist_cann.SC_ATTRACTOR_VARIANTS[1],
                    keep_history=False,
                )
                distance_cochlea = distance_prediction.cochlea

                azimuth_prediction = azc.predict_one_3d(
                    azimuth_config,
                    float(distance_m),
                    float(azimuth_deg),
                    float(elevation_deg),
                    add_noise=add_noise,
                    limit_deg=AZIMUTH_LIMIT_DEG,
                )
                azimuth_itd_population = normalise_population(azimuth_prediction.itd_activation)
                azimuth_ild_population = normalise_population(azimuth_prediction.ild_activation)
                _, itd_population_batch = azc.itd_population_dataset([azimuth_prediction], azimuth_bins)
                azimuth_cann, _, _, _ = azc.run_cann_readout(itd_population_batch, azimuth_bins)
                itd_population = azimuth_prediction.itd_activation
                ild_population = azimuth_prediction.ild_activation

                receive = elev.simulate_full_3d_scene(
                    elevation_config,
                    float(distance_m),
                    float(azimuth_deg),
                    float(elevation_deg),
                    elev.DEEP_COMB_DELAYED_COPY_GAIN,
                )
                if add_noise:
                    receive = receive + elevation_config.noise_std * torch.randn_like(receive)
                elevation_population, elevation_cann, selected_spikes = elevation_features_from_receive(
                    elevation_config,
                    receive,
                    float(azimuth_deg),
                    elevation_baseline,
                    elevation_gain_matrix,
                    centres_hz,
                    elevation_bins,
                    elevation_params,
                )

            confidence = np.array(
                [
                    float(distance_ac.max()),
                    float(distance_ac.sum()),
                    margin(distance_ac),
                    entropy(distance_ac),
                    float(itd_population.max()),
                    float(itd_population.sum()),
                    margin(itd_population),
                    entropy(itd_population),
                    float(ild_population.max()),
                    float(ild_population.sum()),
                    margin(ild_population),
                    entropy(ild_population),
                    float(elevation_population.max()),
                    float(elevation_population.sum()),
                    margin(elevation_population),
                    entropy(elevation_population),
                    float(distance_cochlea.left_spikes.sum() + distance_cochlea.right_spikes.sum()),
                    float(selected_spikes.sum()),
                ],
                dtype=np.float64,
            )

            values = {
                "raw_distance_population": distance_population,
                "raw_azimuth_itd_population": azimuth_itd_population,
                "raw_azimuth_ild_population": azimuth_ild_population,
                "raw_elevation_population": normalise_population(elevation_population),
                "cann_readouts": np.array([distance_cann[0], azimuth_cann[0], elevation_cann[0]], dtype=np.float64),
                "confidence_features": confidence,
            }
            for name, (left, right) in spec.groups.items():
                features[index, left:right] = values[name].astype(np.float32)

            baseline_coordinates[index] = np.array([distance_cann[0], azimuth_cann[0], elevation_cann[0]], dtype=np.float32)
            baseline_encoded[index] = encode_targets(
                np.array([distance_cann[0]]),
                np.array([azimuth_cann[0]]),
                np.array([elevation_cann[0]]),
            )[0]
            feature_seconds.append(time.perf_counter() - start)

    targets_encoded = encode_targets(distance_true, azimuth_true, elevation_true)
    payload = {
        "features": features,
        "targets_encoded": targets_encoded,
        "baseline_encoded": baseline_encoded,
        "baseline_coordinates": baseline_coordinates,
        "true_coordinates": np.stack([distance_true, azimuth_true, elevation_true], axis=1).astype(np.float32),
        "split_names": split_names,
        "feature_groups": spec.groups,
        "feature_seconds_per_sample": float(np.mean(feature_seconds)),
        "setup": {
            "distance_range_m": [DISTANCE_MIN_M, DISTANCE_MAX_M],
            "azimuth_range_deg": [-AZIMUTH_LIMIT_DEG, AZIMUTH_LIMIT_DEG],
            "elevation_range_deg": [-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG],
            "channels": CHANNELS,
            "distance_bins": DISTANCE_BINS,
            "angular_bins": ANGULAR_BINS,
            "splits": SMOKE_SPLITS,
            "dataset_seed": DATASET_SEED,
            "acoustic_mode": ACOUSTIC_MODE,
            "noise_label": NOISE_LABEL,
            "noise_std": NOISE_STD,
            "environment_noise_db": ENVIRONMENT_NOISE_DB,
            "environment_noise_std": env_noise_std,
            "environment_reverb": ENVIRONMENT_REVERB,
        },
    }
    np.savez_compressed(CACHE_PATH, **payload)
    return payload


def split_arrays(cache: dict[str, object], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return features, targets, baseline outputs, and true coordinates for a split."""
    mask = np.asarray(cache["split_names"]) == split
    return (
        np.asarray(cache["features"])[mask],
        np.asarray(cache["targets_encoded"])[mask],
        np.asarray(cache["baseline_encoded"])[mask],
        np.asarray(cache["true_coordinates"])[mask],
    )


def fit_normaliser(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise standardisation parameters."""
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def apply_normaliser(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply feature-wise standardisation."""
    return ((features - mean) / std).astype(np.float32)


def make_loader(features: np.ndarray, targets: np.ndarray, baseline: np.ndarray, shuffle: bool) -> DataLoader:
    """Create a PyTorch dataloader for readout training."""
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(baseline.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def model_output(model: SmallSNNReadout, features: torch.Tensor, baseline: torch.Tensor, mode: str) -> torch.Tensor:
    """Return direct or residual prediction from the SNN."""
    raw = model(features)
    if mode == "residual":
        return baseline + RESIDUAL_SCALE * raw
    if mode == "direct":
        return raw
    raise ValueError(f"Unknown readout mode: {mode}")


def train_one_readout(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_baseline: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    val_baseline: np.ndarray,
    *,
    mode: str,
) -> tuple[SmallSNNReadout, UncertaintyWeightedLoss, dict[str, object]]:
    """Train one SNN readout variant."""
    torch.manual_seed(TRAINING_SEED + (0 if mode == "residual" else 10_000))
    model = SmallSNNReadout(train_features.shape[1], HIDDEN_DIM, 5, NUM_STEPS)
    if mode == "residual":
        # The residual model should begin close to the hand-designed pathway
        # output, not as a large random perturbation of it.
        nn.init.zeros_(model.fc_out.weight)
        nn.init.zeros_(model.fc_out.bias)
    criterion = UncertaintyWeightedLoss()
    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_loader = make_loader(train_features, train_targets, train_baseline, shuffle=True)
    val_loader = make_loader(val_features, val_targets, val_baseline, shuffle=False)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        criterion.train()
        train_losses = []
        for features, targets, baseline in train_loader:
            prediction = model_output(model, features, baseline, mode)
            loss = criterion(prediction, targets)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 2.0)
            optimiser.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        criterion.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets, baseline in val_loader:
                prediction = model_output(model, features, baseline, mode)
                val_losses.append(float(criterion(prediction, targets).item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
                "criterion": {name: value.detach().clone() for name, value in criterion.state_dict().items()},
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        criterion.load_state_dict(best_state["criterion"])

    return model, criterion, {
        **history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "training_seconds": time.perf_counter() - start,
        "learned_log_vars": criterion.log_vars.detach().cpu().numpy().tolist(),
    }


def predict_encoded(model: SmallSNNReadout, features: np.ndarray, baseline: np.ndarray, mode: str) -> np.ndarray:
    """Run a trained readout and return encoded predictions."""
    model.eval()
    outputs = []
    loader = make_loader(features, np.zeros((features.shape[0], 5), dtype=np.float32), baseline, shuffle=False)
    with torch.no_grad():
        for batch_features, _, batch_baseline in loader:
            outputs.append(model_output(model, batch_features, batch_baseline, mode).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def coordinate_metrics(true_coordinates: np.ndarray, encoded_prediction: np.ndarray) -> dict[str, float]:
    """Compute final coordinate metrics from encoded predictions."""
    pred_distance, pred_azimuth, pred_elevation = decode_outputs(encoded_prediction)
    rows = []
    for true, distance, azimuth, elevation in zip(true_coordinates, pred_distance, pred_azimuth, pred_elevation):
        rows.append(
            {
                "true_distance_m": float(true[0]),
                "true_azimuth_deg": float(true[1]),
                "true_elevation_deg": float(true[2]),
                "pred_distance_m": float(distance),
                "pred_azimuth_deg": float(azimuth),
                "pred_elevation_deg": float(elevation),
            }
        )
    return final.localisation_metrics(rows, DISTANCE_MAX_M, AZIMUTH_LIMIT_DEG, ELEVATION_LIMIT_DEG)


def population_center_of_mass(populations: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Decode a batch of non-negative population codes by centre of mass."""
    values = np.maximum(np.asarray(populations, dtype=np.float64), 0.0)
    totals = values.sum(axis=1)
    fallback = float(bins[len(bins) // 2])
    decoded = np.full(values.shape[0], fallback, dtype=np.float64)
    valid = totals > 1e-12
    decoded[valid] = (values[valid] * bins[None, :]).sum(axis=1) / totals[valid]
    return decoded


def raw_com_encoded_from_features(features: np.ndarray, feature_groups: dict[str, tuple[int, int]]) -> np.ndarray:
    """Decode the no-CANN raw populations in a feature matrix by centre of mass."""
    distance_range = np.linspace(DISTANCE_MIN_M, DISTANCE_MAX_M, DISTANCE_BINS, dtype=np.float64)
    angle_range = np.linspace(-AZIMUTH_LIMIT_DEG, AZIMUTH_LIMIT_DEG, ANGULAR_BINS, dtype=np.float64)

    d0, d1 = feature_groups["raw_distance_population"]
    a0, a1 = feature_groups["raw_azimuth_itd_population"]
    e0, e1 = feature_groups["raw_elevation_population"]
    raw_distance = population_center_of_mass(features[:, d0:d1], distance_range)
    raw_azimuth = population_center_of_mass(features[:, a0:a1], angle_range)
    raw_elevation = population_center_of_mass(features[:, e0:e1], angle_range)
    return encode_targets(raw_distance, raw_azimuth, raw_elevation)


def raw_com_metrics_from_cache(cache_path: Path) -> dict[str, float] | None:
    """Compute the no-CANN raw CoM readout from a cached feature file.

    The raw readout uses the distance AC population, azimuth ITD population,
    and elevation DCN population. It is evaluated on the cached test split.
    """
    if not cache_path.exists():
        return None
    loaded = np.load(cache_path, allow_pickle=True)
    features = np.asarray(loaded["features"], dtype=np.float64)
    true_coordinates = np.asarray(loaded["true_coordinates"], dtype=np.float64)
    split_names = np.asarray(loaded["split_names"])
    feature_groups = loaded["feature_groups"].item()
    test_mask = split_names == "test"
    if not np.any(test_mask):
        return None

    test_features = features[test_mask]
    true_test = true_coordinates[test_mask]
    encoded = raw_com_encoded_from_features(test_features, feature_groups)
    return coordinate_metrics(true_test, encoded)


def first_layer_group_weights(model: SmallSNNReadout, feature_groups: dict[str, tuple[int, int]]) -> dict[str, float]:
    """Summarise absolute first-layer weights by feature group."""
    weights = model.fc_in.weight.detach().cpu().numpy()
    summary = {}
    for name, (left, right) in feature_groups.items():
        summary[name] = float(np.mean(np.abs(weights[:, left:right])))
    total = sum(summary.values())
    if total > 1e-12:
        summary = {name: value / total for name, value in summary.items()}
    return summary


def ablation_importance(
    model: SmallSNNReadout,
    features: np.ndarray,
    baseline: np.ndarray,
    true_coordinates: np.ndarray,
    feature_groups: dict[str, tuple[int, int]],
    *,
    mode: str,
) -> dict[str, float]:
    """Measure feature-group importance by zero-ablation on normalised inputs."""
    base_prediction = predict_encoded(model, features, baseline, mode)
    base_metric = coordinate_metrics(true_coordinates, base_prediction)["combined_normalised_error"]
    importance = {}
    for name, (left, right) in feature_groups.items():
        ablated = features.copy()
        ablated[:, left:right] = 0.0
        pred = predict_encoded(model, ablated, baseline, mode)
        metric = coordinate_metrics(true_coordinates, pred)["combined_normalised_error"]
        importance[name] = float(metric - base_metric)
    return importance


def plot_training_curves(histories: dict[str, dict[str, object]], path: Path) -> str:
    """Plot direct and residual training curves."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name, history in histories.items():
        ax.plot(history["train_loss"], label=f"{name} train")
        ax.plot(history["val_loss"], linestyle="--", label=f"{name} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("uncertainty-weighted loss")
    ax.set_title("Smoke-test SNN readout training")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_prediction_scatter(true_coordinates: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> str:
    """Plot true-vs-predicted coordinates for test readouts."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    labels = [("distance", "m", 0), ("azimuth", "deg", 1), ("elevation", "deg", 2)]
    for ax, (label, unit, index) in zip(axes, labels):
        true = true_coordinates[:, index]
        for name, encoded in predictions.items():
            decoded = decode_outputs(encoded)
            pred = decoded[index]
            ax.scatter(true, pred, s=24, alpha=0.62, label=name)
        low, high = float(true.min()), float(true.max())
        ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
        ax.set_xlabel(f"true {label} ({unit})")
        ax.set_ylabel(f"predicted {label} ({unit})")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_raw_baseline_scatter(true_coordinates: np.ndarray, raw_encoded: np.ndarray, baseline_encoded: np.ndarray, path: Path) -> str:
    """Compare no-CANN raw CoM and fixed CANN baseline predictions."""
    return plot_prediction_scatter(
        true_coordinates,
        {"raw no-CANN CoM": raw_encoded, "baseline CANN": baseline_encoded},
        path,
    )


def plot_importance(weight_importance: dict[str, float], ablation: dict[str, float], path: Path) -> str:
    """Plot first-layer weight share and ablation importance."""
    names = list(weight_importance.keys())
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(x, [weight_importance[name] for name in names])
    axes[0].set_title("First-layer absolute weight share")
    axes[0].set_ylabel("normalised share")
    axes[1].bar(x, [ablation[name] for name in names])
    axes[1].set_title("Zero-ablation importance")
    axes[1].set_ylabel("increase in combined error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, path)


def _short_feature_name(name: str) -> str:
    """Return compact labels for feature-group plots."""
    return {
        "raw_distance_population": "distance pop.",
        "raw_azimuth_itd_population": "ITD pop.",
        "raw_azimuth_ild_population": "ILD pop.",
        "raw_elevation_population": "elevation pop.",
        "cann_readouts": "CANN readouts",
        "confidence_features": "confidence",
    }.get(name, name)


def plot_combined_importance(
    named_payloads: list[tuple[str, dict[str, object]]],
    result_key: str,
    title: str,
    ylabel: str,
    path: Path,
) -> str:
    """Plot one feature-importance metric across the full trainable runs."""
    if not named_payloads:
        raise ValueError("No payloads supplied for combined feature-importance plot.")
    first = named_payloads[0][1].get(result_key, {})
    if not isinstance(first, dict) or not first:
        raise ValueError(f"Missing feature-importance key: {result_key}")
    names = list(first.keys())
    x = np.arange(len(names), dtype=np.float64)
    width = 0.78 / max(1, len(named_payloads))
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    for index, (label, payload) in enumerate(named_payloads):
        values_by_name = payload.get(result_key, {})
        if not isinstance(values_by_name, dict):
            values_by_name = {}
        values = [float(values_by_name.get(name, 0.0)) for name in names]
        offset = (index - (len(named_payloads) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=label)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_feature_name(name) for name in names], rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    return save_figure(fig, path)


def markdown_path(path: str | Path, *, report_path: Path = REPORT_PATH) -> str:
    """Return a markdown-friendly path relative to a report directory."""
    return Path(os.path.relpath(Path(path), report_path.parent)).as_posix()


def run_report_path() -> Path:
    """Return the per-run report path."""
    return REPORT_PATH.parent / f"trainable_final_readout_{RUN_LABEL}.md"


def write_report(results: dict[str, object], artifacts: dict[str, str], report_path: Path | None = None) -> Path:
    """Write the per-run trainable readout report.

    Args:
        results: Results payload for the current run.
        artifacts: Generated figure paths.
        report_path: Optional destination. Defaults to the run-labelled report.

    Returns:
        Path to the written report.
    """
    report_path = run_report_path() if report_path is None else report_path
    lines = [
        "# Trainable Final SNN Readout",
        "",
        "This report tests whether the fixed biologically structured pathways can be followed by a small trainable snnTorch readout. The default command is a smoke test; larger runs use the same cache, training, evaluation, and report-writing pipeline.",
        "",
        "## Reproducible Setup",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| distance range | `{DISTANCE_MIN_M}-{DISTANCE_MAX_M} m` |",
        f"| azimuth range | `+/-{AZIMUTH_LIMIT_DEG} deg` |",
        f"| elevation range | `+/-{ELEVATION_LIMIT_DEG} deg` |",
        f"| acoustic mode | `{ACOUSTIC_MODE}` |",
        f"| noise label | `{NOISE_LABEL}` |",
        f"| receiver noise std | `{NOISE_STD:.6g}` |",
        f"| environmental noise | `{ENVIRONMENT_NOISE_DB:.1f} dB`, reverb `{ENVIRONMENT_REVERB}` |",
        f"| train / val / test samples | `{SMOKE_SPLITS['train']} / {SMOKE_SPLITS['val']} / {SMOKE_SPLITS['test']}` |",
        f"| run label | `{RUN_LABEL}` |",
        f"| dataset seed | `{DATASET_SEED}` |",
        f"| training seed | `{TRAINING_SEED}` |",
        f"| cochlear channels | `{CHANNELS}` |",
        f"| distance bins | `{DISTANCE_BINS}` |",
        f"| angular bins | `{ANGULAR_BINS}` |",
        f"| SNN hidden neurons | `{HIDDEN_DIM}` |",
        f"| SNN timesteps | `{NUM_STEPS}` |",
        f"| optimiser | `AdamW`, lr `{LEARNING_RATE}`, weight decay `{WEIGHT_DECAY}` |",
        f"| batch size | `{BATCH_SIZE}` |",
        f"| epochs | `{EPOCHS}` |",
        f"| residual scale | `{RESIDUAL_SCALE}` |",
        "",
        "The cached input vector is:",
        "",
        "```text",
        "raw distance population",
        "raw azimuth ITD population",
        "raw azimuth ILD population",
        "raw elevation population",
        "CANN distance/azimuth/elevation readouts",
        "confidence features and spike counts",
        "```",
        "",
        "The target vector is:",
        "",
        "```text",
        "[distance_norm, az_sin, az_cos, el_sin, el_cos]",
        "```",
        "",
        "Distance is normalised as `(distance - 0.25) / 4.75`. Angles are trained as sine/cosine pairs to avoid discontinuities.",
        "",
        "## Loss Function",
        "",
        "The readout uses learned uncertainty weighting over three tasks:",
        "",
        "$$",
        "L=\\frac{L_d}{2\\sigma_d^2}+\\log\\sigma_d+\\frac{L_a}{2\\sigma_a^2}+\\log\\sigma_a+\\frac{L_e}{2\\sigma_e^2}+\\log\\sigma_e.",
        "$$",
        "",
        "Here `Ld` is distance MSE, `La` is azimuth sine/cosine MSE, and `Le` is elevation sine/cosine MSE.",
        "",
        "## Cached Training Results",
        "",
        "| Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metric in results["test_metrics"].items():
        lines.append(
            f"| {name} | `{metric['distance_mae_m']:.4f} m` | `{metric['azimuth_mae_deg']:.3f} deg` | "
            f"`{metric['elevation_mae_deg']:.3f} deg` | `{metric['euclidean_mae_m']:.4f} m` | "
            f"`{metric['combined_normalised_error']:.4f}` |"
        )
    lines.extend(
        [
            "",
            f"Mean feature-cache generation time was `{results['cache']['feature_seconds_per_sample']:.3f} s/sample` for this run.",
            "",
            f"![Training curves]({markdown_path(artifacts['training_curves'], report_path=report_path)})",
            "",
            f"![Prediction scatter]({markdown_path(artifacts['test_prediction_scatter'], report_path=report_path)})",
            "",
            "The following diagnostic isolates whether the fixed readout collapse is already present in the raw pathway populations or is introduced by the CANN stage.",
            "",
            f"![Raw vs baseline scatter]({markdown_path(artifacts['raw_vs_baseline_scatter'], report_path=report_path)})",
            "",
            "## Feature Importance",
            "",
            "The first diagnostic sums the absolute first-layer weights by feature group. The second zeroes each normalised feature group on the test set and measures the increase in combined error. These are not perfect causal explanations, but they show whether the trained SNN is using the CANN readouts or mostly ignoring them.",
            "",
            "### Residual SNN",
            "",
            f"![Residual feature importance]({markdown_path(artifacts['residual_feature_importance'], report_path=report_path)})",
            "",
            "| Feature group | First-layer share | Ablation delta |",
            "|---|---:|---:|",
        ]
    )
    for name, value in results["residual_weight_importance"].items():
        lines.append(
            f"| {name} | `{value:.4f}` | `{results['residual_ablation_importance'][name]:.4f}` |"
        )
    if "direct_weight_importance" in results and "direct_ablation_importance" in results:
        lines.extend(
            [
                "",
                "### Direct SNN",
                "",
                f"![Direct feature importance]({markdown_path(artifacts['direct_feature_importance'], report_path=report_path)})",
                "",
                "| Feature group | First-layer share | Ablation delta |",
                "|---|---:|---:|",
            ]
        )
        for name, value in results["direct_weight_importance"].items():
            lines.append(
                f"| {name} | `{value:.4f}` | `{results['direct_ablation_importance'][name]:.4f}` |"
            )
    lines.extend(
        [
            "",
            "## Biological Interpretation",
            "",
            "This is best interpreted as a higher-level contextual integration layer. The sensory pathways still produce structured population codes. The small SNN receives raw cue populations, stabilised CANN readouts, and confidence signals, then learns how to combine them when cues are distorted by distance, azimuth, elevation, and pathway confidence.",
            "",
            "The residual variant is especially biologically defensible because it keeps the hand-designed pathway answer as the main estimate and learns a small context-dependent correction.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.append(f"- `cache`: `{CACHE_PATH.relative_to(ROOT)}`")
    lines.append(f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _load_result_file(path: Path) -> dict[str, object] | None:
    """Load a result JSON if it has the expected schema."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("experiment") != "trainable_final_readout":
        return None
    if "test_metrics" not in payload or "setup" not in payload:
        return None
    return payload


def write_comparison_report() -> None:
    """Aggregate available trainable-readout runs into one comparison report."""
    result_files = sorted(OUTPUT_DIR.glob("results_*.json"))
    loaded = [(path, _load_result_file(path)) for path in result_files]
    runs = [(path, payload) for path, payload in loaded if payload is not None]

    def total_samples(payload: dict[str, object]) -> int:
        splits = payload.get("setup", {}).get("splits", {})
        return int(sum(int(value) for value in splits.values())) if isinstance(splits, dict) else 0

    def noise_label(payload: dict[str, object]) -> str:
        setup = payload["setup"]
        return str(setup.get("noise_label", "clean"))

    def acoustic_mode(payload: dict[str, object]) -> str:
        setup = payload["setup"]
        if "acoustic_mode" in setup:
            return str(setup["acoustic_mode"])
        return "clean" if noise_label(payload) in {"clean", "unknown"} else "receiver_noise"

    def append_metric_rows(selected_runs: list[tuple[Path, dict[str, object]]]) -> None:
        for path, payload in selected_runs:
            setup = payload["setup"]
            run_label = str(setup.get("run_label", path.stem.removeprefix("results_")))
            metrics = dict(payload["test_metrics"])
            cache_path = Path(str(payload.get("cache", {}).get("path", "")))
            raw_metrics = raw_com_metrics_from_cache(cache_path)
            if raw_metrics is not None:
                metrics = {"raw": raw_metrics, **metrics}
            result_link = markdown_path(path, report_path=COMPARISON_REPORT_PATH)
            for readout in ["raw", "baseline", "residual", "direct"]:
                if readout not in metrics:
                    continue
                metric = metrics[readout]
                lines.append(
                    f"| `{run_label}` | `{acoustic_mode(payload)}` | `{noise_label(payload)}` | `{readout}` | "
                    f"`{metric['distance_mae_m']:.4f} m` | `{metric['azimuth_mae_deg']:.3f} deg` | "
                    f"`{metric['elevation_mae_deg']:.3f} deg` | `{metric['euclidean_mae_m']:.4f} m` | "
                    f"`{metric['combined_normalised_error']:.4f}` | [json]({result_link}) |"
                )

    primary_runs = [(path, payload) for path, payload in runs if total_samples(payload) >= 100]
    smoke_runs = [(path, payload) for path, payload in runs if total_samples(payload) < 100]
    lines = [
        "# Trainable Final SNN Readout Comparison",
        "",
        "This report is regenerated automatically from available `results_*.json` files. It allows clean, environmental-noise, and environmental-noise-plus-reverb cached runs to coexist without overwriting each other.",
        "",
        "Primary rows use the full cached setup (`2000/400/400` train/validation/test samples). Tiny smoke-test runs are separated because they only verify execution and should not be interpreted as accuracy results.",
        "",
        "`raw` is the no-CANN centre-of-mass readout computed directly from the cached raw distance, ITD azimuth, and elevation populations. `baseline` is the fixed CANN readout. `residual` and `direct` are the trained SNN readouts.",
        "",
    ]
    if not runs:
        lines.extend(["No trainable-readout result files were found.", ""])
        COMPARISON_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Primary Full-Run Summary",
            "",
            "| Run | Acoustic mode | Noise | Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error | Results |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    append_metric_rows(primary_runs)

    full_by_label = {payload["setup"].get("run_label", path.stem.removeprefix("results_")): payload for path, payload in primary_runs}
    clean = full_by_label.get("train2000_val400_test400")
    env = full_by_label.get("train2000_val400_test400_envnoise50dB")
    reverb = full_by_label.get("train2000_val400_test400_envnoise50dB_reverb")
    full_named_payloads = [
        ("clean", clean),
        ("environment noise", env),
        ("noise + reverb", reverb),
    ]
    full_named_payloads = [(label, payload) for label, payload in full_named_payloads if payload is not None]
    if clean and env and reverb:
        clean_res = clean["test_metrics"]["residual"]
        env_res = env["test_metrics"]["residual"]
        reverb_res = reverb["test_metrics"]["residual"]
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- The hand-designed baseline is strongly affected by environmental noise: elevation MAE rises to `{env['test_metrics']['baseline']['elevation_mae_deg']:.3f} deg` because the spectral notch cue is corrupted before cochlear/DCN processing.",
                f"- The residual SNN substantially recovers the environmental-noise case, reducing combined error from `{env['test_metrics']['baseline']['combined_normalised_error']:.4f}` to `{env_res['combined_normalised_error']:.4f}`.",
                f"- Adding the simple late-echo/reverb tail does not destroy the trained residual readout in this setup: residual combined error is `{reverb_res['combined_normalised_error']:.4f}`, close to the environmental-noise-only value `{env_res['combined_normalised_error']:.4f}`.",
                f"- Clean residual performance remains best overall, with combined error `{clean_res['combined_normalised_error']:.4f}` and Euclidean MAE `{clean_res['euclidean_mae_m']:.4f} m`.",
                "",
            ]
        )
        comparison_figure_dir = OUTPUT_DIR / "figures" / "comparison"
        ensure_dir(comparison_figure_dir)
        weight_path = plot_combined_importance(
            full_named_payloads,
            "residual_weight_importance",
            "Residual SNN first-layer feature-group weight share",
            "normalised absolute weight share",
            comparison_figure_dir / "combined_residual_first_layer_weight_importance.png",
        )
        ablation_path = plot_combined_importance(
            full_named_payloads,
            "residual_ablation_importance",
            "Residual SNN zero-ablation feature-group importance",
            "increase in combined error",
            comparison_figure_dir / "combined_residual_zero_ablation_importance.png",
        )
        direct_weight_path = None
        direct_ablation_path = None
        if all("direct_weight_importance" in payload and "direct_ablation_importance" in payload for _, payload in full_named_payloads):
            direct_weight_path = plot_combined_importance(
                full_named_payloads,
                "direct_weight_importance",
                "Direct SNN first-layer feature-group weight share",
                "normalised absolute weight share",
                comparison_figure_dir / "combined_direct_first_layer_weight_importance.png",
            )
            direct_ablation_path = plot_combined_importance(
                full_named_payloads,
                "direct_ablation_importance",
                "Direct SNN zero-ablation feature-group importance",
                "increase in combined error",
                comparison_figure_dir / "combined_direct_zero_ablation_importance.png",
            )
        lines.extend(
            [
                "## Combined Feature Importance",
                "",
                "These plots combine the three full `2000/400/400` runs so that feature use can be compared directly across clean, environmental-noise, and noise-plus-reverb conditions.",
                "",
                f"![Combined first-layer weight importance]({markdown_path(weight_path, report_path=COMPARISON_REPORT_PATH)})",
                "",
                f"![Combined zero-ablation importance]({markdown_path(ablation_path, report_path=COMPARISON_REPORT_PATH)})",
                "",
            ]
        )
        if direct_weight_path is not None and direct_ablation_path is not None:
            lines.extend(
                [
                    "### Direct SNN",
                    "",
                    f"![Combined direct first-layer weight importance]({markdown_path(direct_weight_path, report_path=COMPARISON_REPORT_PATH)})",
                    "",
                    f"![Combined direct zero-ablation importance]({markdown_path(direct_ablation_path, report_path=COMPARISON_REPORT_PATH)})",
                    "",
                ]
            )
        lines.extend(
            [
                "The first-layer plot measures parameter magnitude, while the zero-ablation plot measures the change in test combined error when a normalised feature group is set to zero. The ablation plot is therefore the more useful diagnostic for whether the trained residual SNN depends on a feature group.",
                "",
                "## Scatter Plot Gallery",
                "",
                "These are the per-run scatter plots collected in one place. For each condition, the first plot compares the raw no-CANN readout against the fixed CANN baseline, and the second compares the fixed baseline with the residual and direct trainable SNN readouts.",
                "",
            ]
        )
        for label, payload in full_named_payloads:
            artifacts = payload.get("artifacts", {})
            if not isinstance(artifacts, dict):
                continue
            raw_vs_baseline = artifacts.get("raw_vs_baseline_scatter")
            prediction_scatter = artifacts.get("test_prediction_scatter")
            lines.extend([f"### {label}", ""])
            if raw_vs_baseline:
                lines.extend(
                    [
                        "Raw no-CANN readout versus fixed CANN baseline:",
                        "",
                        f"![{label} raw versus baseline scatter]({markdown_path(raw_vs_baseline, report_path=COMPARISON_REPORT_PATH)})",
                        "",
                    ]
                )
            if prediction_scatter:
                lines.extend(
                    [
                        "Fixed baseline, residual SNN, and direct SNN:",
                        "",
                        f"![{label} trainable readout scatter]({markdown_path(prediction_scatter, report_path=COMPARISON_REPORT_PATH)})",
                        "",
                    ]
                )

    lines.extend(
        [
            "",
            "## Per-Run Reports",
            "",
        ]
    )
    for _, payload in runs:
        run_label = str(payload["setup"].get("run_label", "unknown"))
        report_path = REPORT_PATH.parent / f"trainable_final_readout_{run_label}.md"
        if report_path.exists():
            lines.append(f"- `{run_label}`: [{report_path.name}]({markdown_path(report_path, report_path=COMPARISON_REPORT_PATH)})")
    if smoke_runs:
        lines.extend(
            [
                "",
                "## Smoke-Test Runs",
                "",
                "These rows are retained for reproducibility only. They used fewer than 100 total samples and should not be used for model comparison.",
                "",
                "| Run | Acoustic mode | Noise | Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error | Results |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        append_metric_rows(smoke_runs)
    lines.append("")
    COMPARISON_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the trainable readout smoke test."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    cache = build_cached_features(force=FORCE_CACHE)
    train_x, train_y, train_base, _ = split_arrays(cache, "train")
    val_x, val_y, val_base, _ = split_arrays(cache, "val")
    test_x, test_y, test_base, test_true = split_arrays(cache, "test")
    mean, std = fit_normaliser(train_x)
    train_x = apply_normaliser(train_x, mean, std)
    val_x = apply_normaliser(val_x, mean, std)
    test_x = apply_normaliser(test_x, mean, std)
    feature_groups = dict(cache["feature_groups"].item() if hasattr(cache["feature_groups"], "item") else cache["feature_groups"])
    raw_test_encoded = raw_com_encoded_from_features(np.asarray(cache["features"])[np.asarray(cache["split_names"]) == "test"], feature_groups)

    histories = {}
    models = {}
    criteria = {}
    for mode in ["residual", "direct"]:
        print(f"Training {mode} SNN readout for {EPOCHS} epochs")
        model, criterion, history = train_one_readout(
            train_x,
            train_y,
            train_base,
            val_x,
            val_y,
            val_base,
            mode=mode,
        )
        histories[mode] = history
        models[mode] = model
        criteria[mode] = criterion

    predictions = {
        "baseline": test_base,
        "residual": predict_encoded(models["residual"], test_x, test_base, "residual"),
        "direct": predict_encoded(models["direct"], test_x, test_base, "direct"),
    }
    test_metrics = {"raw": coordinate_metrics(test_true, raw_test_encoded)}
    test_metrics.update({name: coordinate_metrics(test_true, pred) for name, pred in predictions.items()})
    residual_weight_importance = first_layer_group_weights(models["residual"], feature_groups)
    residual_ablation_importance = ablation_importance(
        models["residual"],
        test_x,
        test_base,
        test_true,
        feature_groups,
        mode="residual",
    )
    direct_weight_importance = first_layer_group_weights(models["direct"], feature_groups)
    direct_ablation_importance = ablation_importance(
        models["direct"],
        test_x,
        test_base,
        test_true,
        feature_groups,
        mode="direct",
    )

    artifacts = {
        "training_curves": plot_training_curves(histories, FIGURE_DIR / "training_curves.png"),
        "test_prediction_scatter": plot_prediction_scatter(test_true, predictions, FIGURE_DIR / "test_prediction_scatter.png"),
        "raw_vs_baseline_scatter": plot_raw_baseline_scatter(
            test_true,
            raw_test_encoded,
            test_base,
            FIGURE_DIR / "raw_vs_baseline_scatter.png",
        ),
        "residual_feature_importance": plot_importance(
            residual_weight_importance,
            residual_ablation_importance,
            FIGURE_DIR / "residual_feature_importance.png",
        ),
        "direct_feature_importance": plot_importance(
            direct_weight_importance,
            direct_ablation_importance,
            FIGURE_DIR / "direct_feature_importance.png",
        ),
    }
    results = {
        "experiment": "trainable_final_readout",
        "elapsed_seconds": time.perf_counter() - start,
        "setup": {
            "run_label": RUN_LABEL,
            "splits": SMOKE_SPLITS,
            "dataset_seed": DATASET_SEED,
            "training_seed": TRAINING_SEED,
            "hidden_dim": HIDDEN_DIM,
            "num_steps": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "acoustic_mode": ACOUSTIC_MODE,
            "noise_label": NOISE_LABEL,
            "noise_std": NOISE_STD,
            "environment_noise_db": ENVIRONMENT_NOISE_DB,
            "environment_reverb": ENVIRONMENT_REVERB,
        },
        "cache": {
            "path": str(CACHE_PATH),
            "feature_seconds_per_sample": float(cache["feature_seconds_per_sample"]),
            "feature_dim": int(cache["features"].shape[1]),
            "feature_groups": feature_groups,
        },
        "histories": histories,
        "test_metrics": test_metrics,
        "residual_weight_importance": residual_weight_importance,
        "residual_ablation_importance": residual_ablation_importance,
        "direct_weight_importance": direct_weight_importance,
        "direct_ablation_importance": direct_ablation_importance,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    per_run_report = write_report(results, artifacts)
    if REPORT_PATH != per_run_report:
        write_report(results, artifacts, report_path=REPORT_PATH)
    write_comparison_report()
    return results


if __name__ == "__main__":
    configure_run(parse_args())
    main()
    print(REPORT_PATH)
    print(run_report_path())
    print(COMPARISON_REPORT_PATH)
    print(RESULTS_PATH)
