from __future__ import annotations

"""SC line-attractor readout integration experiment.

This script keeps the current full distance pathway unchanged up to the AC
distance map, then compares the existing SC centre-of-mass readout against a
balanced E/I line-attractor SC readout. It is intentionally separate from
`full_distance_pathway_model.py` so the simple readout remains easy to recover.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_models.common.plotting import ensure_dir, save_figure

from distance_pathway.experiments import full_distance_pathway_model as fdm


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "sc_line_attractor_integration"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "sc_line_attractor_integration.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

TAU_S = 0.020
DT_S = 0.001
SIM_TIME_S = 0.060
READOUT_TIME_S = 0.005
RECURRENT_SIGMA_BINS = 8.0
INPUT_TUNING_SIGMA_BINS = 3.0
ALPHA_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]


@dataclass(frozen=True)
class ReadoutResult:
    """Readout comparison for one condition.

    Attributes:
        condition: Human-readable condition name.
        true_distance_m: True target distance.
        baseline_distance_m: Existing simple SC centre-of-mass prediction.
        attractor_distance_m: Balanced line-attractor prediction at 5 ms.
        attractor_trajectory_m: Attractor decoded distance through time.
        seconds_per_sample: Runtime per sample for the attractor readout only.
        ac_activations: Upstream AC population activations.
        azimuth_deg: Optional target azimuths for full-3D diagnostics.
        elevation_deg: Optional target elevations for full-3D diagnostics.
    """

    condition: str
    true_distance_m: np.ndarray
    baseline_distance_m: np.ndarray
    attractor_distance_m: np.ndarray
    attractor_trajectory_m: np.ndarray
    seconds_per_sample: float
    ac_activations: np.ndarray
    azimuth_deg: np.ndarray | None = None
    elevation_deg: np.ndarray | None = None


def gaussian(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate an unnormalised Gaussian."""
    return np.exp(-0.5 * (distance / sigma) ** 2)


def reflected_kernel_matrix(positions_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """Build a reflected finite-line Gaussian kernel.

    Args:
        positions_m: Preferred distance bins.
        sigma_m: Kernel width in metres.

    Returns:
        Kernel matrix `[target, source]`.
    """
    length_m = float(positions_m[-1] - positions_m[0])
    shifted = positions_m - positions_m[0]
    target = shifted[:, None]
    source = shifted[None, :]
    return (
        gaussian(target - source, sigma_m)
        + gaussian(target + source, sigma_m)
        + gaussian(target - (2.0 * length_m - source), sigma_m)
    )


def rescale_to_alpha(matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Scale a recurrent matrix to a requested largest real eigenvalue."""
    if alpha == 0.0:
        return np.zeros_like(matrix)
    current = float(np.max(np.real(np.linalg.eigvals(matrix))))
    if abs(current) < 1e-12:
        raise ValueError("Cannot rescale a matrix with near-zero spectral abscissa.")
    return matrix * (alpha / current)


def reflected_tuning_derivative(positions_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """Return derivative of reflected tuning curves over the same grid.

    Args:
        positions_m: Preferred distance bins and sampled stimulus positions.
        sigma_m: Tuning width in metres.

    Returns:
        Derivative matrix `[stimulus_position, neuron]`.
    """
    length_m = float(positions_m[-1] - positions_m[0])
    shifted = positions_m - positions_m[0]
    stimulus = shifted[:, None]
    neuron = shifted[None, :]
    sigma2 = sigma_m**2
    z1 = neuron - stimulus
    z2 = neuron + stimulus
    z3 = neuron - (2.0 * length_m - stimulus)
    return (
        (z1 / sigma2) * gaussian(z1, sigma_m)
        - (z2 / sigma2) * gaussian(z2, sigma_m)
        - (z3 / sigma2) * gaussian(z3, sigma_m)
    )


def fisher_balanced_input_gains(positions_m: np.ndarray, sigma_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate diagonal input gains that flatten Fisher information.

    The same AC distance grid is used as the attractor grid. A diagonal input
    gain vector is chosen by solving:

    `D^2 u ~= constant`, with `u_i = gain_i^2`.

    This keeps the mapping topographic while compensating for finite-line
    boundary effects. It is a flat-Fisher design, not a learned parameter fit.

    Args:
        positions_m: Distance bins.
        sigma_m: Tuning width in metres.

    Returns:
        Tuple `(gain, raw_fisher, balanced_fisher)`.
    """
    derivative = reflected_tuning_derivative(positions_m, sigma_m)
    design = derivative**2
    target = np.ones(design.shape[0], dtype=np.float64)
    mean_constraint = np.ones((1, design.shape[1]), dtype=np.float64)
    augmented_design = np.vstack([design, 25.0 * mean_constraint])
    augmented_target = np.concatenate([target, [25.0 * design.shape[1]]])
    solution, *_ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    solution = np.clip(solution, 0.10, 10.0)
    solution = solution / np.mean(solution)
    gain = np.sqrt(solution)
    raw_fisher = design.sum(axis=1)
    balanced_fisher = design @ solution
    return gain, raw_fisher, balanced_fisher


def build_balanced_ei_matrix(positions_m: np.ndarray, alpha_prime: float) -> np.ndarray:
    """Build the balanced E/I recurrent matrix for the SC line attractor."""
    bin_width_m = float(np.mean(np.diff(positions_m)))
    sigma_m = RECURRENT_SIGMA_BINS * bin_width_m
    local = reflected_kernel_matrix(positions_m, sigma_m)
    local = local / np.maximum(local.sum(axis=1, keepdims=True), 1e-12)
    local = rescale_to_alpha(local, alpha_prime)
    return np.block([[local, -local], [local, -local]])


def decode_center_of_mass(activity: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    """Decode distance from non-negative population activity."""
    positive = np.maximum(activity, 0.0)
    total = positive.sum(axis=-1)
    decoded = (positive * positions_m[None, :]).sum(axis=-1) / np.maximum(total, 1e-12)
    fallback = 0.5 * (positions_m[0] + positions_m[-1])
    return np.where(total > 1e-12, decoded, fallback)


def run_attractor_readout(
    ac_activations: np.ndarray,
    positions_m: np.ndarray,
    alpha_prime: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the balanced E/I line-attractor readout on AC activations.

    Args:
        ac_activations: AC population activations `[samples, bins]`.
        positions_m: Distance represented by each AC/SC bin.
        alpha_prime: Balanced E/I recurrent gain.

    Returns:
        Tuple `(decoded_at_5ms, decoded_trajectory, seconds_per_sample)`.
    """
    start = time.perf_counter()
    num_samples, num_bins = ac_activations.shape
    bin_width_m = float(np.mean(np.diff(positions_m)))
    input_sigma_m = INPUT_TUNING_SIGMA_BINS * bin_width_m
    input_gains, _, _ = fisher_balanced_input_gains(positions_m, input_sigma_m)
    recurrent = build_balanced_ei_matrix(positions_m, alpha_prime)

    state = np.zeros((num_samples, 2 * num_bins), dtype=np.float64)
    normalised_ac = ac_activations / np.maximum(ac_activations.max(axis=1, keepdims=True), 1e-12)
    state[:, :num_bins] = normalised_ac * input_gains[None, :]

    num_steps = int(round(SIM_TIME_S / DT_S))
    readout_index = int(round(READOUT_TIME_S / DT_S))
    trajectory = np.empty((num_steps + 1, num_samples), dtype=np.float64)
    trajectory[0] = decode_center_of_mass(state[:, :num_bins], positions_m)
    for step in range(1, num_steps + 1):
        state = state + DT_S / TAU_S * (-state + state @ recurrent.T)
        trajectory[step] = decode_center_of_mass(state[:, :num_bins], positions_m)
    seconds_per_sample = (time.perf_counter() - start) / max(1, num_samples)
    return trajectory[readout_index], trajectory.T, seconds_per_sample


def run_attractor_state_history(
    ac_activation: np.ndarray,
    positions_m: np.ndarray,
    alpha_prime: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one AC population through the attractor and keep state history.

    Args:
        ac_activation: One AC activation vector `[bins]`.
        positions_m: Distance grid.
        alpha_prime: Balanced E/I recurrent gain.

    Returns:
        Tuple `(times_s, excitatory_history, decoded_trajectory)`.
    """
    num_bins = ac_activation.shape[0]
    bin_width_m = float(np.mean(np.diff(positions_m)))
    input_sigma_m = INPUT_TUNING_SIGMA_BINS * bin_width_m
    input_gains, _, _ = fisher_balanced_input_gains(positions_m, input_sigma_m)
    recurrent = build_balanced_ei_matrix(positions_m, alpha_prime)

    state = np.zeros(2 * num_bins, dtype=np.float64)
    normalised = ac_activation / max(float(np.max(ac_activation)), 1e-12)
    state[:num_bins] = normalised * input_gains

    num_steps = int(round(SIM_TIME_S / DT_S))
    times = np.arange(num_steps + 1) * DT_S
    excitatory = np.empty((num_steps + 1, num_bins), dtype=np.float64)
    decoded = np.empty(num_steps + 1, dtype=np.float64)
    excitatory[0] = state[:num_bins]
    decoded[0] = decode_center_of_mass(state[None, :num_bins], positions_m)[0]
    for step in range(1, num_steps + 1):
        state = state + DT_S / TAU_S * (-state + recurrent @ state)
        excitatory[step] = state[:num_bins]
        decoded[step] = decode_center_of_mass(state[None, :num_bins], positions_m)[0]
    return times, excitatory, decoded


def metrics(true_m: np.ndarray, pred_m: np.ndarray) -> dict[str, float]:
    """Calculate scalar distance error metrics."""
    error = pred_m - true_m
    return {
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "bias_m": float(np.mean(error)),
    }


def collect_ac_predictions(
    predictions: list[fdm.PathwayPrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract true distance, baseline prediction, and AC activity arrays."""
    true = np.array([prediction.distance_m for prediction in predictions], dtype=np.float64)
    baseline = np.array([prediction.predicted_distance_m for prediction in predictions], dtype=np.float64)
    ac = np.stack([prediction.ac_activation for prediction in predictions], axis=0).astype(np.float64)
    return true, baseline, ac


def run_readout_condition(
    name: str,
    predictions: list[fdm.PathwayPrediction],
    positions_m: np.ndarray,
    alpha_prime: float,
    *,
    azimuth_deg: np.ndarray | None = None,
    elevation_deg: np.ndarray | None = None,
) -> ReadoutResult:
    """Compare baseline and attractor readouts for one condition."""
    true, baseline, ac = collect_ac_predictions(predictions)
    attractor, trajectory, seconds_per_sample = run_attractor_readout(ac, positions_m, alpha_prime)
    return ReadoutResult(
        condition=name,
        true_distance_m=true,
        baseline_distance_m=baseline,
        attractor_distance_m=attractor,
        attractor_trajectory_m=trajectory,
        seconds_per_sample=seconds_per_sample,
        ac_activations=ac,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )


def make_primary_variant(config: fdm.GlobalConfig) -> fdm.PathwayVariant:
    """Create the current primary dynamic full-distance variant."""
    latency_samples = fdm._load_channel_latency(config)
    variants = fdm._make_variants(config, latency_samples)
    return next(variant for variant in variants if variant.key == "dynamic_spike_consensus_facil")


def run_small_space_predictions() -> tuple[fdm.GlobalConfig, fdm.PathwayVariant, np.ndarray, list[fdm.PathwayPrediction], list[fdm.PathwayPrediction]]:
    """Run current pathway predictions for clean and noisy small-space tests."""
    config = fdm._make_config()
    noisy_config = fdm._make_noisy_config(config)
    variant = make_primary_variant(config)
    rng = np.random.default_rng(fdm.RNG_SEED)
    distances = rng.uniform(fdm.MIN_DISTANCE_M, fdm.MAX_DISTANCE_M, size=fdm.NUM_TEST_SAMPLES)
    clean_predictions = fdm._run_variant_predictions(config, distances, variant, add_noise=False)
    torch.manual_seed(fdm.RNG_SEED + 10_000)
    noisy_predictions = fdm._run_variant_predictions(noisy_config, distances, variant, add_noise=True)
    return config, variant, distances, clean_predictions, noisy_predictions


def run_full_space_predictions(
    base_variant: fdm.PathwayVariant,
) -> tuple[fdm.GlobalConfig, list[fdm.PathwayPrediction], list[fdm.PathwayPrediction], dict[str, np.ndarray]]:
    """Run current pathway predictions for the clean and 50 dB full 3D tests."""
    rng = np.random.default_rng(fdm.RNG_SEED + 50_000)
    distances = rng.uniform(fdm.MIN_DISTANCE_M, fdm.FULL_TEST_MAX_DISTANCE_M, size=fdm.FULL_TEST_SAMPLES)
    azimuths = rng.uniform(-fdm.FULL_TEST_AZIMUTH_LIMIT_DEG, fdm.FULL_TEST_AZIMUTH_LIMIT_DEG, size=fdm.FULL_TEST_SAMPLES)
    elevations = rng.uniform(-fdm.FULL_TEST_ELEVATION_LIMIT_DEG, fdm.FULL_TEST_ELEVATION_LIMIT_DEG, size=fdm.FULL_TEST_SAMPLES)

    full_config = fdm._make_full_test_config(noise_std=0.0)
    latency_template = fdm.replace(base_variant, latency_samples=np.zeros(fdm.NUM_CHANNELS, dtype=np.int64))
    full_latency = fdm._calibrate_variant_latency(
        full_config,
        latency_template,
        calibration_distances=np.linspace(fdm.MIN_DISTANCE_M, fdm.FULL_TEST_MAX_DISTANCE_M, 16),
    )
    full_variant = fdm.replace(base_variant, latency_samples=full_latency)

    clean_predictions = [
        fdm._predict_one_3d(
            full_config,
            float(distance),
            float(azimuth),
            float(elevation),
            full_variant,
            add_noise=False,
        )
        for distance, azimuth, elevation in zip(distances, azimuths, elevations)
    ]
    ambient_config = fdm._make_full_test_config(noise_std=fdm._noise_std_from_db(fdm.AMBIENT_NOISE_DB_SPL))
    torch.manual_seed(fdm.RNG_SEED + 60_001)
    ambient_predictions = [
        fdm._predict_one_3d(
            ambient_config,
            float(distance),
            float(azimuth),
            float(elevation),
            full_variant,
            add_noise=True,
        )
        for distance, azimuth, elevation in zip(distances, azimuths, elevations)
    ]
    coordinates = {
        "distance_m": distances,
        "azimuth_deg": azimuths,
        "elevation_deg": elevations,
    }
    return full_config, clean_predictions, ambient_predictions, coordinates


def sweep_alpha(
    clean_predictions: list[fdm.PathwayPrediction],
    noisy_predictions: list[fdm.PathwayPrediction],
    positions_m: np.ndarray,
) -> tuple[list[dict[str, float]], float]:
    """Sweep balanced E/I gain and choose the best alpha."""
    rows = []
    clean_true, _, clean_ac = collect_ac_predictions(clean_predictions)
    noisy_true, _, noisy_ac = collect_ac_predictions(noisy_predictions)
    for alpha_prime in ALPHA_SWEEP:
        clean_pred, _, clean_seconds = run_attractor_readout(clean_ac, positions_m, alpha_prime)
        noisy_pred, _, noisy_seconds = run_attractor_readout(noisy_ac, positions_m, alpha_prime)
        clean_metrics = metrics(clean_true, clean_pred)
        noisy_metrics = metrics(noisy_true, noisy_pred)
        selection_score = 0.5 * (clean_metrics["mae_m"] + noisy_metrics["mae_m"])
        rows.append(
            {
                "alpha_prime": float(alpha_prime),
                "clean_mae_m": clean_metrics["mae_m"],
                "noisy_mae_m": noisy_metrics["mae_m"],
                "selection_score_m": float(selection_score),
                "seconds_per_sample": float(0.5 * (clean_seconds + noisy_seconds)),
            }
        )
    best = min(rows, key=lambda item: item["selection_score_m"])
    return rows, float(best["alpha_prime"])


def plot_fisher_gains(positions_m: np.ndarray, path: Path) -> str:
    """Plot FI-balanced input gains and raw/balanced Fisher curves."""
    bin_width_m = float(np.mean(np.diff(positions_m)))
    sigma_m = INPUT_TUNING_SIGMA_BINS * bin_width_m
    gains, raw_fisher, balanced_fisher = fisher_balanced_input_gains(positions_m, sigma_m)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(positions_m, gains, linewidth=2.0)
    axes[0].set_title("FI-balanced input gains")
    axes[0].set_xlabel("distance bin (m)")
    axes[0].set_ylabel("input gain")
    axes[1].plot(positions_m, raw_fisher / raw_fisher.max(), linewidth=2.0, label="raw")
    axes[1].plot(positions_m, balanced_fisher / balanced_fisher.max(), linewidth=2.0, label="FI-balanced")
    axes[1].set_title("Input Fisher information")
    axes[1].set_xlabel("distance bin (m)")
    axes[1].set_ylabel("relative FI")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def plot_alpha_sweep(alpha_rows: list[dict[str, float]], path: Path) -> str:
    """Plot alpha-prime sweep for clean and noisy small-space tests."""
    alpha = np.array([row["alpha_prime"] for row in alpha_rows])
    clean = np.array([row["clean_mae_m"] for row in alpha_rows]) * 100.0
    noisy = np.array([row["noisy_mae_m"] for row in alpha_rows]) * 100.0
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(alpha, clean, marker="o", linewidth=2.0, label="clean")
    ax.plot(alpha, noisy, marker="o", linewidth=2.0, label="10 dB SNR + jitter")
    ax.set_xlabel("balanced E/I alpha prime")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("SC line-attractor alpha sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_timing(result: ReadoutResult, path: Path) -> str:
    """Plot decoded distance over time for representative examples and MAE."""
    times_ms = np.arange(result.attractor_trajectory_m.shape[1]) * DT_S * 1_000.0
    errors = np.abs(result.attractor_trajectory_m - result.true_distance_m[:, None])
    order = np.argsort(result.true_distance_m)
    example_indices = [
        int(order[len(order) // 5]),
        int(order[len(order) // 2]),
        int(order[4 * len(order) // 5]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for index in example_indices:
        axes[0].plot(times_ms, result.attractor_trajectory_m[index], linewidth=2.0, label=f"true {result.true_distance_m[index]:.2f} m")
        axes[0].axhline(result.true_distance_m[index], color="#111827", linestyle=":", linewidth=1.0)
    axes[0].axvline(READOUT_TIME_S * 1_000.0, color="#dc2626", linestyle="--", label="5 ms readout")
    axes[0].set_title("Attractor decoded distance vs time")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("decoded distance (m)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(times_ms, errors.mean(axis=0) * 100.0, linewidth=2.0)
    axes[1].axvline(READOUT_TIME_S * 1_000.0, color="#dc2626", linestyle="--")
    axes[1].set_title("Population mean absolute error vs time")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("MAE (cm)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def plot_bump_dynamics(
    result: ReadoutResult,
    positions_m: np.ndarray,
    alpha_prime: float,
    path: Path,
) -> str:
    """Plot AC input and line-attractor bump dynamics for example cases.

    Args:
        result: Readout condition to sample examples from.
        positions_m: Distance grid for the attractor.
        alpha_prime: Balanced E/I recurrent gain.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    baseline_error = np.abs(result.baseline_distance_m - result.true_distance_m)
    good_index = int(np.argmin(baseline_error))
    failure_index = int(np.argmax(baseline_error))
    examples = [("good case", good_index), ("failure case", failure_index)]
    snapshot_times_s = [0.0, READOUT_TIME_S, 0.020, 0.060]
    fig, axes = plt.subplots(len(examples), len(snapshot_times_s), figsize=(14.0, 6.6), sharex=True)
    for row, (label, index) in enumerate(examples):
        times, excitatory, decoded = run_attractor_state_history(
            result.ac_activations[index],
            positions_m,
            alpha_prime,
        )
        ac_norm = result.ac_activations[index] / max(float(np.max(result.ac_activations[index])), 1e-12)
        for col, time_s in enumerate(snapshot_times_s):
            time_index = int(np.argmin(np.abs(times - time_s)))
            activity = excitatory[time_index]
            activity = activity / max(float(np.max(activity)), 1e-12)
            ax = axes[row, col]
            ax.plot(positions_m, ac_norm, color="#6b7280", linestyle="--", linewidth=1.5, label="AC input")
            ax.plot(positions_m, activity, color="#2563eb", linewidth=2.0, label="attractor E")
            ax.axvline(result.true_distance_m[index], color="#111827", linestyle=":", linewidth=1.2, label="true")
            ax.axvline(decoded[time_index], color="#dc2626", linestyle="--", linewidth=1.2, label="decoded")
            if row == 0:
                ax.set_title(f"{time_s * 1_000:.0f} ms")
            if col == 0:
                ax.set_ylabel(
                    f"{label}\ntrue {result.true_distance_m[index]:.2f} m\n"
                    f"read {result.attractor_distance_m[index]:.2f} m"
                )
            ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("distance represented by SC neuron (m)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    return save_figure(fig, path)


def plot_prediction_scatter(results: list[ReadoutResult], path: Path) -> str:
    """Plot baseline and attractor prediction scatters."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=False, sharey=False)
    for ax, result in zip(axes.flat, results):
        ax.scatter(result.true_distance_m, result.baseline_distance_m, s=18, alpha=0.55, label="simple COM")
        ax.scatter(result.true_distance_m, result.attractor_distance_m, s=18, alpha=0.55, label="balanced attractor")
        min_d = min(result.true_distance_m.min(), result.baseline_distance_m.min(), result.attractor_distance_m.min())
        max_d = max(result.true_distance_m.max(), result.baseline_distance_m.max(), result.attractor_distance_m.max())
        ax.plot([min_d, max_d], [min_d, max_d], color="#111827", linewidth=1.0)
        ax.set_title(result.condition)
        ax.set_xlabel("true distance (m)")
        ax.set_ylabel("predicted distance (m)")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    return save_figure(fig, path)


def explain_failure(distance_m: float, azimuth_deg: float, elevation_deg: float, predicted_m: float) -> str:
    """Return a heuristic explanation for a full-3D distance failure.

    Args:
        distance_m: True target distance.
        azimuth_deg: True target azimuth.
        elevation_deg: True target elevation.
        predicted_m: Readout distance.

    Returns:
        Short diagnostic explanation.
    """
    reasons = []
    if distance_m > fdm.MAX_DISTANCE_M:
        reasons.append("true range is outside the 5 m range where the pathway is strongest")
    if predicted_m < 0.75 * distance_m:
        reasons.append("readout is biased short, suggesting the AC peak is already pulled to an earlier delay")
    if abs(azimuth_deg) > 70.0:
        reasons.append("extreme azimuth increases binaural path/head-shadow asymmetry")
    if abs(elevation_deg) > 35.0:
        reasons.append("large elevation applies stronger spectral reshaping to the active channels")
    if not reasons:
        reasons.append("likely local ambiguity or broad AC activity rather than an obvious geometric extreme")
    return "; ".join(reasons)


def failure_cases(result: ReadoutResult, *, count: int = 10) -> list[dict[str, object]]:
    """Return worst full-3D failure cases by baseline absolute error."""
    if result.azimuth_deg is None or result.elevation_deg is None:
        return []
    abs_error = np.abs(result.baseline_distance_m - result.true_distance_m)
    indices = np.argsort(abs_error)[::-1][:count]
    cases = []
    for rank, index in enumerate(indices, start=1):
        cases.append(
            {
                "rank": rank,
                "true_distance_m": float(result.true_distance_m[index]),
                "azimuth_deg": float(result.azimuth_deg[index]),
                "elevation_deg": float(result.elevation_deg[index]),
                "baseline_readout_m": float(result.baseline_distance_m[index]),
                "attractor_readout_m": float(result.attractor_distance_m[index]),
                "baseline_error_m": float(result.baseline_distance_m[index] - result.true_distance_m[index]),
                "attractor_error_m": float(result.attractor_distance_m[index] - result.true_distance_m[index]),
                "estimated_reason": explain_failure(
                    float(result.true_distance_m[index]),
                    float(result.azimuth_deg[index]),
                    float(result.elevation_deg[index]),
                    float(result.baseline_distance_m[index]),
                ),
            }
        )
    return cases


def comparison_rows(results: list[ReadoutResult]) -> list[dict[str, object]]:
    """Build metrics rows for report and JSON output."""
    rows: list[dict[str, object]] = []
    for result in results:
        masks = {"all": np.ones_like(result.true_distance_m, dtype=bool)}
        if result.true_distance_m.max() > fdm.MAX_DISTANCE_M:
            masks = {
                "<=5m": result.true_distance_m <= fdm.MAX_DISTANCE_M,
                "<=10m": np.ones_like(result.true_distance_m, dtype=bool),
            }
        for subset, mask in masks.items():
            rows.append(
                {
                    "condition": result.condition,
                    "subset": subset,
                    "num_samples": int(np.count_nonzero(mask)),
                    "baseline_metrics": metrics(result.true_distance_m[mask], result.baseline_distance_m[mask]),
                    "attractor_metrics": metrics(result.true_distance_m[mask], result.attractor_distance_m[mask]),
                    "attractor_seconds_per_sample": result.seconds_per_sample,
                }
            )
    return rows


def write_report(
    alpha_rows: list[dict[str, float]],
    selected_alpha: float,
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the SC line-attractor integration report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    alpha_table = [
        "| "
        f"`{row['alpha_prime']:.2f}` | "
        f"`{row['clean_mae_m'] * 100.0:.3f} cm` | "
        f"`{row['noisy_mae_m'] * 100.0:.3f} cm` | "
        f"`{row['selection_score_m'] * 100.0:.3f} cm` | "
        f"`{row['seconds_per_sample'] * 1_000.0:.2f} ms` |"
        for row in alpha_rows
    ]
    comparison_table = []
    for row in rows:
        baseline = row["baseline_metrics"]
        attractor = row["attractor_metrics"]
        comparison_table.append(
            "| "
            f"{row['condition']} | "
            f"{row['subset']} | "
            f"`{row['num_samples']}` | "
            f"`{baseline['mae_m'] * 100.0:.3f} cm` | "
            f"`{attractor['mae_m'] * 100.0:.3f} cm` | "
            f"`{baseline['rmse_m'] * 100.0:.3f} cm` | "
            f"`{attractor['rmse_m'] * 100.0:.3f} cm` | "
            f"`{baseline['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"`{attractor['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"`{row['attractor_seconds_per_sample'] * 1_000.0:.2f} ms` |"
        )
    failure_table = [
        "| "
        f"{case['rank']} | "
        f"`({case['true_distance_m']:.2f} m, {case['azimuth_deg']:.1f} deg, {case['elevation_deg']:.1f} deg)` | "
        f"`{case['baseline_readout_m']:.2f} m` | "
        f"`{case['attractor_readout_m']:.2f} m` | "
        f"`{case['baseline_error_m']:.2f} m` | "
        f"`{case['attractor_error_m']:.2f} m` | "
        f"{case['estimated_reason']} |"
        for case in failures
    ]
    lines = [
        "# SC Line Attractor Integration",
        "",
        "This report tests a balanced E/I line-attractor readout as an upgraded SC stage. The current full distance pathway is left unchanged up to the AC distance population. The only change is the final readout.",
        "",
        "## Experiment Design",
        "",
        "The baseline is the existing simple SC centre-of-mass readout:",
        "",
        "```text",
        "d_hat = sum_k AC_k d_k / sum_k AC_k",
        "```",
        "",
        "The upgraded readout uses a balanced E/I line attractor with the same distance grid as the AC. Therefore, no spatial resampling is needed:",
        "",
        "```text",
        "N_SC = N_AC",
        "x_SC,k = d_AC,k",
        "```",
        "",
        "The balanced E/I state is:",
        "",
        "```text",
        "r = [r_E, r_I]",
        "W_EI = [[ W0, -W0],",
        "        [ W0, -W0]]",
        "tau dr/dt = -r + W_EI r",
        "```",
        "",
        "The AC population is injected as a brief impulse into the excitatory population only:",
        "",
        "```text",
        "r_E(0) = G_FI * normalise(AC)",
        "r_I(0) = 0",
        "```",
        "",
        "The distance is decoded by centre of mass from the excitatory readout. The requested first readout time is `5 ms`, but the timing plot shows how the decoded distance changes over the full `60 ms` simulation.",
        "",
        "## Fisher-Balanced Input Weights",
        "",
        "The recurrent weights are kept as the reflected line-attractor structure. Input weights are diagonal gains chosen to flatten Fisher information over the distance grid rather than learned from labels.",
        "",
        "For reflected tuning curves `h_i(x)`, independent Gaussian noise gives:",
        "",
        "```text",
        "J(x) = sum_i (g_i dh_i/dx)^2 / sigma_n^2",
        "```",
        "",
        "The gain values are found by solving a constrained least-squares approximation:",
        "",
        "```text",
        "D^2 u ~= constant",
        "u_i = g_i^2",
        "mean(u) = 1",
        "```",
        "",
        "This keeps the input topographic while reducing boundary-related Fisher information dips.",
        "",
        "![Fisher input gains](../outputs/sc_line_attractor_integration/figures/fisher_input_gains.png)",
        "",
        "## Alpha Sweep",
        "",
        "The original ring-model notebook showed that increasing recurrent gain can improve readout accuracy. Here the same idea is tested by sweeping the balanced E/I `alpha_prime` parameter while keeping the recurrent structure fixed.",
        "",
        "![Alpha sweep](../outputs/sc_line_attractor_integration/figures/alpha_sweep.png)",
        "",
        "| alpha prime | Clean MAE | Noisy MAE | Selection score | Runtime/sample |",
        "|---:|---:|---:|---:|---:|",
        *alpha_table,
        "",
        f"The selected alpha for the controlled comparisons is `{selected_alpha:.2f}`, chosen by the mean of clean and noisy small-space MAE.",
        "",
        "## Input Timing Experiment",
        "",
        "The integration uses the readout at `5 ms`. The plot below checks whether that is reasonable by showing representative decoded distances over time and the mean absolute error over time.",
        "",
        "![Readout timing](../outputs/sc_line_attractor_integration/figures/readout_timing.png)",
        "",
        "## Bump Dynamics",
        "",
        "The plot below shows the upstream AC population and the attractor excitatory population at several times. It includes one good full-3D clean case and one failure case. The dashed grey curve is the original AC input, the blue curve is the attractor excitatory bump, the black dotted line is the true distance, and the red dashed line is the decoded distance at that time.",
        "",
        "![Bump dynamics](../outputs/sc_line_attractor_integration/figures/bump_dynamics.png)",
        "",
        "## Controlled Comparisons",
        "",
        "The comparison uses the same upstream AC activations for both readouts, so any difference is caused by the SC readout only.",
        "",
        "![Prediction scatter](../outputs/sc_line_attractor_integration/figures/prediction_scatter.png)",
        "",
        "| Condition | Subset | N | Baseline MAE | Attractor MAE | Baseline RMSE | Attractor RMSE | Baseline max error | Attractor max error | Attractor runtime/sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *comparison_table,
        "",
        "## Full-3D Failure Cases",
        "",
        "The table below lists the worst clean full-3D cases by the original simple-readout absolute distance error. The target coordinate is shown as `(distance, azimuth, elevation)`, while the readouts are one-dimensional distance estimates. The estimated reason is a heuristic diagnostic, not a proven causal attribution.",
        "",
        "| Rank | True coordinate | Simple readout | Attractor readout | Simple error | Attractor error | Estimated reason |",
        "|---:|---|---:|---:|---:|---:|---|",
        *failure_table,
        "",
        "## Interpretation",
        "",
        "Result: this first balanced line-attractor SC integration should be treated as diagnostic, not accepted as the primary readout yet. It is mechanically successful and reversible, but the simple centre-of-mass readout still has lower MAE in the main comparisons.",
        "",
        "- This is an SC readout ablation only; the cochlea, VCN, DNLL, IC, and AC stages are unchanged.",
        "- Matching the attractor neurons to the AC distance grid keeps the interface simple and reversible.",
        "- The alpha sweep shows the best balanced gain is modest, around `alpha_prime = 1`, rather than increasing indefinitely as in the original ring notebook.",
        "- The attractor slightly reduces some max-error values in the `<=5m` full-space subset, but it increases MAE and RMSE overall.",
        "- The `<=10m` rows show that this SC readout does not solve the upstream long-range/angle-induced failure mode; the AC population is already biased before the SC readout.",
        "- The likely next readout experiment is not simply stronger recurrence, but a better-matched input pulse or a time-varying attractor input that preserves the AC confidence profile.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the SC line-attractor integration experiment."""
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    ensure_dir(OUTPUT_DIR)

    torch.manual_seed(fdm.RNG_SEED)
    np.random.seed(fdm.RNG_SEED)

    config, variant, _, clean_small, noisy_small = run_small_space_predictions()
    small_positions = fdm._candidate_distances(config)
    alpha_rows, selected_alpha = sweep_alpha(clean_small, noisy_small, small_positions)

    clean_small_result = run_readout_condition("Small clean 0.25-5m", clean_small, small_positions, selected_alpha)
    noisy_small_result = run_readout_condition("Small noisy 10dB+jitter", noisy_small, small_positions, selected_alpha)

    full_config, clean_full, ambient_full, full_coordinates = run_full_space_predictions(variant)
    full_positions = fdm._candidate_distances(full_config)
    clean_full_result = run_readout_condition(
        "Full 3D clean 0.25-10m",
        clean_full,
        full_positions,
        selected_alpha,
        azimuth_deg=full_coordinates["azimuth_deg"],
        elevation_deg=full_coordinates["elevation_deg"],
    )
    ambient_full_result = run_readout_condition(
        "Full 3D 50dB floor",
        ambient_full,
        full_positions,
        selected_alpha,
        azimuth_deg=full_coordinates["azimuth_deg"],
        elevation_deg=full_coordinates["elevation_deg"],
    )

    results = [clean_small_result, noisy_small_result, clean_full_result, ambient_full_result]
    rows = comparison_rows(results)
    failures = failure_cases(clean_full_result, count=10)
    artifacts = {
        "fisher_input_gains": plot_fisher_gains(small_positions, FIGURE_DIR / "fisher_input_gains.png"),
        "alpha_sweep": plot_alpha_sweep(alpha_rows, FIGURE_DIR / "alpha_sweep.png"),
        "readout_timing": plot_timing(clean_small_result, FIGURE_DIR / "readout_timing.png"),
        "bump_dynamics": plot_bump_dynamics(clean_full_result, full_positions, selected_alpha, FIGURE_DIR / "bump_dynamics.png"),
        "prediction_scatter": plot_prediction_scatter(results, FIGURE_DIR / "prediction_scatter.png"),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "sc_line_attractor_integration",
        "elapsed_seconds": elapsed_s,
        "readout_time_s": READOUT_TIME_S,
        "tau_s": TAU_S,
        "dt_s": DT_S,
        "sim_time_s": SIM_TIME_S,
        "recurrent_sigma_bins": RECURRENT_SIGMA_BINS,
        "input_tuning_sigma_bins": INPUT_TUNING_SIGMA_BINS,
        "alpha_sweep": alpha_rows,
        "selected_alpha_prime": selected_alpha,
        "comparison_rows": rows,
        "full_3d_failure_cases": failures,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(alpha_rows, selected_alpha, rows, failures, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
