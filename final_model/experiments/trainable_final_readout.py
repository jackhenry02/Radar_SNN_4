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
from dataclasses import dataclass
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
from final_model.experiments import final_model_results as final
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "trainable_readout"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "final_model" / "reports" / "trainable_final_readout.md"
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
    RUN_LABEL = f"train{args.train}_val{args.val}_test{args.test}"
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
        distance_config = final.make_distance_config(CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M)
        distance_variant = final.make_distance_variant(distance_config, CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M)
        azimuth_config = final.make_azimuth_config(CHANNELS, DISTANCE_MAX_M)
        elevation_config = final.make_elevation_config(CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M)
        elevation_params, elevation_baseline = final.tune_elevation_calibration(elevation_config)
        distance_bins = fdm._candidate_distances(distance_config)
        azimuth_bins = az.azimuth_grid(AZIMUTH_LIMIT_DEG)
        elevation_bins = elev.elevation_grid()
        centres_hz, elevation_gain_matrix, _ = elev.build_dcn_templates(
            elevation_config,
            elevation_bins,
            elev.DEEP_COMB_DELAYED_COPY_GAIN,
        )

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

            distance_prediction = fdm._predict_one_3d(
                distance_config,
                float(distance_m),
                float(azimuth_deg),
                float(elevation_deg),
                distance_variant,
                add_noise=False,
            )
            distance_population = normalise_population(distance_prediction.ac_activation)
            distance_cann, _, _, _ = dist_cann.run_line_attractor(
                distance_prediction.ac_activation[None, :],
                distance_bins,
                dist_cann.SC_ATTRACTOR_VARIANTS[1],
                keep_history=False,
            )

            azimuth_prediction = azc.predict_one_3d(
                azimuth_config,
                float(distance_m),
                float(azimuth_deg),
                float(elevation_deg),
                add_noise=False,
                limit_deg=AZIMUTH_LIMIT_DEG,
            )
            azimuth_itd_population = normalise_population(azimuth_prediction.itd_activation)
            azimuth_ild_population = normalise_population(azimuth_prediction.ild_activation)
            _, itd_population_batch = azc.itd_population_dataset([azimuth_prediction], azimuth_bins)
            azimuth_cann, _, _, _ = azc.run_cann_readout(itd_population_batch, azimuth_bins)

            receive = elev.simulate_full_3d_scene(
                elevation_config,
                float(distance_m),
                float(azimuth_deg),
                float(elevation_deg),
                elev.DEEP_COMB_DELAYED_COPY_GAIN,
            )
            cochlea = fdm._run_cochlea_binaural(elevation_config, receive)
            selected_cochleagram = cochlea.right_cochleagram if azimuth_deg >= 0.0 else cochlea.left_cochleagram
            selected_spikes = fdm._dynamic_lif_encode(selected_cochleagram, elevation_config, fdm.DYNAMIC_COHLEA_SCHEDULE)
            profile = elev.dynamic_wideband_inhibited_profile(
                selected_cochleagram,
                selected_spikes,
                elev.DynamicInhibitionParams(),
            )
            equalized = profile / np.maximum(elevation_baseline, 1e-4)
            equalized = equalized / np.maximum(equalized.max(), 1e-12)
            elevation_population = elev.dcn_signal_weighted_transfer_response(
                equalized,
                elevation_baseline,
                elevation_gain_matrix,
                centres_hz,
            )
            elevation_attractor = elc.run_attractor_variants(elevation_population[None, :], elevation_bins)
            elevation_raw = np.asarray(elevation_attractor["reflected"]["prediction"], dtype=np.float64)
            elevation_cann = elc.apply_tuned_inverse_sigmoid(elevation_raw, elevation_params)

            confidence = np.array(
                [
                    float(distance_prediction.ac_activation.max()),
                    float(distance_prediction.ac_activation.sum()),
                    margin(distance_prediction.ac_activation),
                    entropy(distance_prediction.ac_activation),
                    float(azimuth_prediction.itd_activation.max()),
                    float(azimuth_prediction.itd_activation.sum()),
                    margin(azimuth_prediction.itd_activation),
                    entropy(azimuth_prediction.itd_activation),
                    float(azimuth_prediction.ild_activation.max()),
                    float(azimuth_prediction.ild_activation.sum()),
                    margin(azimuth_prediction.ild_activation),
                    entropy(azimuth_prediction.ild_activation),
                    float(elevation_population.max()),
                    float(elevation_population.sum()),
                    margin(elevation_population),
                    entropy(elevation_population),
                    float(distance_prediction.cochlea.left_spikes.sum() + distance_prediction.cochlea.right_spikes.sum()),
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


def markdown_path(path: str | Path) -> str:
    """Return a markdown-friendly path relative to the report directory."""
    return Path(os.path.relpath(Path(path), REPORT_PATH.parent)).as_posix()


def write_report(results: dict[str, object], artifacts: dict[str, str]) -> None:
    """Write the smoke-test report."""
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
        "The smoke test uses learned uncertainty weighting over three tasks:",
        "",
        "$$",
        "L=\\frac{L_d}{2\\sigma_d^2}+\\log\\sigma_d+\\frac{L_a}{2\\sigma_a^2}+\\log\\sigma_a+\\frac{L_e}{2\\sigma_e^2}+\\log\\sigma_e.",
        "$$",
        "",
        "Here `Ld` is distance MSE, `La` is azimuth sine/cosine MSE, and `Le` is elevation sine/cosine MSE.",
        "",
        "## Smoke-Test Results",
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
            f"Mean feature-cache generation time was `{results['cache']['feature_seconds_per_sample']:.3f} s/sample` for this smoke test.",
            "",
            f"![Training curves]({markdown_path(artifacts['training_curves'])})",
            "",
            f"![Prediction scatter]({markdown_path(artifacts['test_prediction_scatter'])})",
            "",
            "## Feature Importance",
            "",
            "The first diagnostic sums the absolute first-layer weights by feature group. The second zeroes each normalised feature group on the test set and measures the increase in combined error. These are not perfect causal explanations, but they show whether the trained SNN is using the CANN readouts or mostly ignoring them.",
            "",
            f"![Feature importance]({markdown_path(artifacts['residual_feature_importance'])})",
            "",
            "| Feature group | First-layer share | Ablation delta |",
            "|---|---:|---:|",
        ]
    )
    for name, value in results["residual_weight_importance"].items():
        lines.append(
            f"| {name} | `{value:.4f}` | `{results['residual_ablation_importance'][name]:.4f}` |"
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
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


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
    test_metrics = {name: coordinate_metrics(test_true, pred) for name, pred in predictions.items()}
    feature_groups = dict(cache["feature_groups"].item() if hasattr(cache["feature_groups"], "item") else cache["feature_groups"])
    residual_weight_importance = first_layer_group_weights(models["residual"], feature_groups)
    residual_ablation_importance = ablation_importance(
        models["residual"],
        test_x,
        test_base,
        test_true,
        feature_groups,
        mode="residual",
    )

    artifacts = {
        "training_curves": plot_training_curves(histories, FIGURE_DIR / "training_curves.png"),
        "test_prediction_scatter": plot_prediction_scatter(test_true, predictions, FIGURE_DIR / "test_prediction_scatter.png"),
        "residual_feature_importance": plot_importance(
            residual_weight_importance,
            residual_ablation_importance,
            FIGURE_DIR / "residual_feature_importance.png",
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
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, artifacts)
    return results


if __name__ == "__main__":
    configure_run(parse_args())
    main()
    print(REPORT_PATH)
    print(RESULTS_PATH)
