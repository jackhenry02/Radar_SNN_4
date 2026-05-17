from __future__ import annotations

"""Final distance pipeline report with SC line-attractor readout.

This experiment keeps the current distance pathway unchanged through the AC
population. It then compares the original SC centre-of-mass readout with a
finite-line balanced E/I attractor readout using the best structured parameters
selected in the finite-line theory report.
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


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "final_distance_pipeline_with_attractor"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "final_distance_pipeline_with_attractor.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

ATTRACTOR_ALPHA_PRIME = 8.0
ATTRACTOR_INPUT_WIDTH_BINS = 3.0
ATTRACTOR_RECURRENT_WIDTH_BINS = 4.0
ATTRACTOR_BETA = 0.8965377489071944
ATTRACTOR_TAU_S = 0.020
ATTRACTOR_DT_S = 0.001
ATTRACTOR_SIM_TIME_S = 0.060
ATTRACTOR_READOUT_TIME_S = 0.060
ATTRACTOR_BASELINE_RATE_HZ = 5.0
ATTRACTOR_STATE_RATE_SCALE_HZ = 20.0
ATTRACTOR_RATE_CAP_HZ = 55.0
SC_SPIKE_BIN_MS = 1.0
SC_INPUT_REFLECTED_OPPONENT = "reflected_opponent"
LOCAL_POPULATION_SIGMA_BINS = 6.0
LOCAL_POPULATION_RADIUS_BINS = 5


@dataclass(frozen=True)
class AttractorPredictionSet:
    """Predictions for one dataset through simple and attractor readouts.

    Attributes:
        condition: Human-readable condition name.
        true_distance_m: True distances.
        simple_distance_m: Original simple centre-of-mass predictions.
        attractor_distance_m: Final line-attractor predictions.
        attractor_trajectory_m: Decoded distance through attractor time.
        local_distance_m: Local population-vector attractor predictions.
        local_trajectory_m: Local population-vector decoded trajectory.
        ac_activations: Upstream AC activity used as input.
        seconds_per_sample: Attractor-only runtime per sample.
        local_seconds_per_sample: Local-vector attractor runtime per sample.
    """

    condition: str
    true_distance_m: np.ndarray
    simple_distance_m: np.ndarray
    attractor_distance_m: np.ndarray
    attractor_trajectory_m: np.ndarray
    local_distance_m: np.ndarray
    local_trajectory_m: np.ndarray
    ac_activations: np.ndarray
    seconds_per_sample: float
    local_seconds_per_sample: float


@dataclass(frozen=True)
class ExampleAttractorHistory:
    """Line-attractor history for one visualised example.

    Attributes:
        times_s: Simulation times.
        state_history: Full E/I state history.
        excitatory_history: Excitatory population history.
        readout_history: Rectified excitatory readout history.
        decoded_trajectory_m: Centre-of-mass distance through time.
        output_spikes: Deterministic output spike raster from excitatory rates.
    """

    times_s: np.ndarray
    state_history: np.ndarray
    excitatory_history: np.ndarray
    readout_history: np.ndarray
    decoded_trajectory_m: np.ndarray
    output_spikes: np.ndarray


def gaussian(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate an unnormalised Gaussian.

    Args:
        distance: Distance from centre.
        sigma: Gaussian standard deviation.

    Returns:
        Gaussian values.
    """
    return np.exp(-0.5 * (distance / sigma) ** 2)


def shifted_positions(positions_m: np.ndarray) -> np.ndarray:
    """Shift a finite distance grid to start at zero.

    Args:
        positions_m: Physical distance grid.

    Returns:
        Shifted coordinate grid.
    """
    return positions_m - positions_m[0]


def reflected_kernel(positions_m: np.ndarray, width_bins: float, *, column_normalise: bool) -> np.ndarray:
    """Build a reflected Gaussian finite-line kernel.

    Args:
        positions_m: Physical distance grid.
        width_bins: Gaussian width in grid bins.
        column_normalise: Whether to normalise source columns. Use this for
            input matrices. Recurrent kernels are row-normalised separately.

    Returns:
        Reflected Gaussian kernel matrix.
    """
    shifted = shifted_positions(positions_m)
    length_m = float(shifted[-1] - shifted[0])
    bin_width_m = float(np.mean(np.diff(positions_m)))
    sigma_m = width_bins * bin_width_m
    target = shifted[:, None]
    source = shifted[None, :]
    matrix = (
        gaussian(target - source, sigma_m)
        + gaussian(target + source, sigma_m)
        + gaussian(target - (2.0 * length_m - source), sigma_m)
    )
    if column_normalise:
        matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
        target_norm = float(np.linalg.norm(np.eye(len(positions_m))))
        matrix = matrix * (target_norm / max(float(np.linalg.norm(matrix)), 1e-12))
    return matrix


def rescale_to_alpha(matrix: np.ndarray, alpha_prime: float) -> np.ndarray:
    """Rescale a matrix so its largest real eigenvalue equals alpha_prime.

    Args:
        matrix: Matrix to rescale.
        alpha_prime: Target largest real eigenvalue.

    Returns:
        Rescaled matrix.
    """
    if alpha_prime == 0.0:
        return np.zeros_like(matrix)
    current = float(np.max(np.real(np.linalg.eigvals(matrix))))
    if abs(current) <= 1e-12:
        raise ValueError("Cannot rescale a matrix with near-zero eigenvalue.")
    return matrix * (alpha_prime / current)


def build_line_attractor_matrices(positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the final AC-to-SC input and balanced recurrent matrices.

    Args:
        positions_m: Shared AC/SC distance grid.

    Returns:
        Tuple `(M, B, W0, W)` where `M` is the reflected input matrix, `B`
        maps AC to E/I state, `W0` is the positive recurrent kernel, and `W`
        is the full balanced E/I recurrence.
    """
    input_matrix = reflected_kernel(positions_m, ATTRACTOR_INPUT_WIDTH_BINS, column_normalise=True)
    recurrent_local = reflected_kernel(positions_m, ATTRACTOR_RECURRENT_WIDTH_BINS, column_normalise=False)
    recurrent_local = recurrent_local / np.maximum(recurrent_local.sum(axis=1, keepdims=True), 1e-12)
    recurrent_local = rescale_to_alpha(recurrent_local, ATTRACTOR_ALPHA_PRIME)
    recurrent = np.block([[recurrent_local, -recurrent_local], [recurrent_local, -recurrent_local]])
    scale = np.sqrt(1.0 + ATTRACTOR_BETA**2)
    input_to_state = np.vstack([input_matrix, -ATTRACTOR_BETA * input_matrix]) / scale
    return input_matrix, input_to_state, recurrent_local, recurrent


def normalise_ac(ac_activations: np.ndarray) -> np.ndarray:
    """Normalise AC maps sample-wise before SC injection.

    Args:
        ac_activations: AC activations `[samples, distance_bins]`.

    Returns:
        Max-normalised AC activations.
    """
    return ac_activations / np.maximum(ac_activations.max(axis=1, keepdims=True), 1e-12)


def decode_center_of_mass(activity: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    """Decode finite-line distance from non-negative population activity.

    Args:
        activity: Population activity with distance on the last axis.
        positions_m: Distance represented by each population bin.

    Returns:
        Decoded distance.
    """
    positive = np.maximum(activity, 0.0)
    total = positive.sum(axis=-1)
    decoded = (positive * positions_m).sum(axis=-1) / np.maximum(total, 1e-12)
    fallback = 0.5 * (positions_m[0] + positions_m[-1])
    return np.where(total > 1e-12, decoded, fallback)


def decode_local_population_vector(activity: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    """Decode attractor activity from a local peak neighbourhood.

    This readout is applied to the **output of the SC attractor**, not to the
    AC input. It finds the peak excitatory neuron, keeps a small neighbourhood
    around that peak, and computes a local centre of mass.

    Args:
        activity: Attractor excitatory/readout activity `[samples, bins]`.
        positions_m: Shared SC distance grid.

    Returns:
        Local population-vector decoded distance for each sample.
    """
    positive = np.maximum(activity, 0.0)
    decoded = np.empty(positive.shape[0], dtype=np.float64)
    radius = int(LOCAL_POPULATION_RADIUS_BINS)
    fallback = 0.5 * (positions_m[0] + positions_m[-1])
    for sample_idx, row in enumerate(positive):
        if float(np.sum(row)) <= 1e-12:
            decoded[sample_idx] = fallback
            continue
        peak_idx = int(np.argmax(row))
        lo = max(0, peak_idx - radius)
        hi = min(row.size, peak_idx + radius + 1)
        local = row[lo:hi]
        total = float(np.sum(local))
        if total <= 1e-12:
            decoded[sample_idx] = float(positions_m[peak_idx])
        else:
            decoded[sample_idx] = float(np.sum(local * positions_m[lo:hi]) / total)
    return decoded


def local_population_window_mask(activity: np.ndarray) -> np.ndarray:
    """Return a mask showing the local-vector readout neighbourhood.

    Args:
        activity: One attractor activity vector `[bins]`.

    Returns:
        Boolean mask over bins used by the local population-vector readout.
    """
    row = np.maximum(activity, 0.0)
    mask = np.zeros(row.shape[0], dtype=bool)
    if float(np.sum(row)) <= 1e-12:
        return mask
    peak_idx = int(np.argmax(row))
    radius = int(LOCAL_POPULATION_RADIUS_BINS)
    lo = max(0, peak_idx - radius)
    hi = min(row.size, peak_idx + radius + 1)
    mask[lo:hi] = True
    return mask


def initialise_sc_state(ac_activations: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    """Initialise SC E/I state from the AC map.

    Args:
        ac_activations: AC activations `[samples, distance_bins]`.
        positions_m: Shared AC/SC distance grid.

    Returns:
        Initial relative-rate SC state `[samples, 2 * distance_bins]`.
    """
    _, input_to_state, _, _ = build_line_attractor_matrices(positions_m)
    ac_input = normalise_ac(ac_activations)
    state = ATTRACTOR_STATE_RATE_SCALE_HZ * (ac_input @ input_to_state.T)
    return clip_state_to_rate_cap(state)


def clip_state_to_rate_cap(state: np.ndarray) -> np.ndarray:
    """Apply the relative-rate cap used by the final SC model.

    Args:
        state: Relative E/I state.

    Returns:
        Clipped relative state.
    """
    lower = -ATTRACTOR_BASELINE_RATE_HZ
    upper = ATTRACTOR_RATE_CAP_HZ - ATTRACTOR_BASELINE_RATE_HZ
    return np.clip(state, lower, upper)


def run_line_attractor(
    ac_activations: np.ndarray,
    positions_m: np.ndarray,
    *,
    keep_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Run the final balanced E/I line attractor on AC activations.

    Args:
        ac_activations: AC activity `[samples, distance_bins]`.
        positions_m: Shared AC/SC distance grid.
        keep_history: Whether to retain full state history.

    Returns:
        Tuple `(final_com_prediction, com_trajectory, final_local_prediction,
        local_trajectory, seconds_per_sample, history)`. `history` is `None`
        unless `keep_history=True`.
    """
    start = time.perf_counter()
    _, _, _, recurrent = build_line_attractor_matrices(positions_m)
    state = initialise_sc_state(ac_activations, positions_m)
    num_samples, state_size = state.shape
    num_bins = ac_activations.shape[1]
    num_steps = int(round(ATTRACTOR_SIM_TIME_S / ATTRACTOR_DT_S))
    readout_index = int(round(ATTRACTOR_READOUT_TIME_S / ATTRACTOR_DT_S))
    trajectory = np.empty((num_samples, num_steps + 1), dtype=np.float64)
    local_trajectory = np.empty((num_samples, num_steps + 1), dtype=np.float64)
    history = np.empty((num_steps + 1, num_samples, state_size), dtype=np.float64) if keep_history else None
    readout = np.maximum(state[:, :num_bins], 0.0)
    trajectory[:, 0] = decode_center_of_mass(readout, positions_m)
    local_trajectory[:, 0] = decode_local_population_vector(readout, positions_m)
    if history is not None:
        history[0] = state
    for step in range(1, num_steps + 1):
        state = state + ATTRACTOR_DT_S / ATTRACTOR_TAU_S * (-state + state @ recurrent.T)
        state = clip_state_to_rate_cap(state)
        readout = np.maximum(state[:, :num_bins], 0.0)
        trajectory[:, step] = decode_center_of_mass(readout, positions_m)
        local_trajectory[:, step] = decode_local_population_vector(readout, positions_m)
        if history is not None:
            history[step] = state
    seconds_per_sample = (time.perf_counter() - start) / max(1, num_samples)
    return (
        trajectory[:, readout_index],
        trajectory,
        local_trajectory[:, readout_index],
        local_trajectory,
        seconds_per_sample,
        history,
    )


def state_history_to_output_spikes(excitatory_history: np.ndarray) -> np.ndarray:
    """Convert attractor excitatory rates into deterministic output spikes.

    This is an illustrative spike readout of the rate-based attractor state.
    A per-neuron accumulator emits a spike whenever integrated rate exceeds
    one event.

    Args:
        excitatory_history: Relative excitatory state `[time, bins]`.

    Returns:
        Binary spike raster `[bins, time]`.
    """
    absolute_rate_hz = np.clip(excitatory_history + ATTRACTOR_BASELINE_RATE_HZ, 0.0, ATTRACTOR_RATE_CAP_HZ)
    accumulator = np.zeros(absolute_rate_hz.shape[1], dtype=np.float64)
    spikes = np.zeros((absolute_rate_hz.shape[1], absolute_rate_hz.shape[0]), dtype=np.float32)
    for time_index, rates in enumerate(absolute_rate_hz):
        accumulator += rates * ATTRACTOR_DT_S
        fired = accumulator >= 1.0
        if np.any(fired):
            spikes[fired, time_index] = 1.0
            accumulator[fired] -= 1.0
    return spikes


def collect_prediction_arrays(
    predictions: list[fdm.PathwayPrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract true distance, simple prediction, and AC activity arrays.

    Args:
        predictions: Full distance-pathway predictions.

    Returns:
        Tuple `(true, simple_prediction, ac_activations)`.
    """
    true = np.array([prediction.distance_m for prediction in predictions], dtype=np.float64)
    simple = np.array([prediction.predicted_distance_m for prediction in predictions], dtype=np.float64)
    ac = np.stack([prediction.ac_activation for prediction in predictions], axis=0).astype(np.float64)
    return true, simple, ac


def metric_dict(true_m: np.ndarray, pred_m: np.ndarray) -> dict[str, float]:
    """Calculate distance error metrics.

    Args:
        true_m: True distances.
        pred_m: Predicted distances.

    Returns:
        MAE, RMSE, max absolute error, and bias in metres.
    """
    error = pred_m - true_m
    return {
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "bias_m": float(np.mean(error)),
    }


def make_primary_variant(config: fdm.GlobalConfig) -> fdm.PathwayVariant:
    """Return the current primary dynamic consensus pathway variant.

    Args:
        config: Acoustic configuration.

    Returns:
        Primary distance-pathway variant.
    """
    latency = fdm._load_channel_latency(config)
    variants = fdm._make_variants(config, latency)
    return next(variant for variant in variants if variant.key == "dynamic_spike_consensus_facil")


def run_small_space_predictions() -> tuple[fdm.GlobalConfig, fdm.PathwayVariant, list[fdm.PathwayPrediction], list[fdm.PathwayPrediction]]:
    """Run small-space clean and noisy predictions for the primary pathway.

    Returns:
        Tuple `(config, variant, clean_predictions, noisy_predictions)`.
    """
    config = fdm._make_config()
    noisy_config = fdm._make_noisy_config(config)
    variant = make_primary_variant(config)
    rng = np.random.default_rng(fdm.RNG_SEED)
    distances = rng.uniform(fdm.MIN_DISTANCE_M, fdm.MAX_DISTANCE_M, size=fdm.NUM_TEST_SAMPLES)
    clean_predictions = fdm._run_variant_predictions(config, distances, variant, add_noise=False)
    torch.manual_seed(fdm.RNG_SEED + 10_000)
    noisy_predictions = fdm._run_variant_predictions(noisy_config, distances, variant, add_noise=True)
    return config, variant, clean_predictions, noisy_predictions


def run_full_space_predictions(
    base_variant: fdm.PathwayVariant,
) -> tuple[fdm.GlobalConfig, list[fdm.PathwayPrediction], list[fdm.PathwayPrediction]]:
    """Run full 3D clean and ambient-noise predictions.

    Args:
        base_variant: Primary small-space variant template.

    Returns:
        Tuple `(full_config, clean_predictions, ambient_predictions)`.
    """
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
    return full_config, clean_predictions, ambient_predictions


def run_condition(
    condition: str,
    predictions: list[fdm.PathwayPrediction],
    positions_m: np.ndarray,
) -> AttractorPredictionSet:
    """Run simple and attractor readouts for one prediction set.

    Args:
        condition: Human-readable condition.
        predictions: Upstream distance-pathway predictions.
        positions_m: Distance grid for the AC/SC map.

    Returns:
        Prediction set with all readouts.
    """
    true, simple, ac = collect_prediction_arrays(predictions)
    attractor, trajectory, local, local_trajectory, seconds_per_sample, _ = run_line_attractor(
        ac,
        positions_m,
        keep_history=False,
    )
    return AttractorPredictionSet(
        condition=condition,
        true_distance_m=true,
        simple_distance_m=simple,
        attractor_distance_m=attractor,
        attractor_trajectory_m=trajectory,
        local_distance_m=local,
        local_trajectory_m=local_trajectory,
        ac_activations=ac,
        seconds_per_sample=seconds_per_sample,
        local_seconds_per_sample=seconds_per_sample,
    )


def choose_example_prediction(predictions: list[fdm.PathwayPrediction]) -> fdm.PathwayPrediction:
    """Choose a representative example close to the median distance.

    Args:
        predictions: Candidate predictions.

    Returns:
        One representative prediction.
    """
    distances = np.array([prediction.distance_m for prediction in predictions])
    median_distance = float(np.median(distances))
    return predictions[int(np.argmin(np.abs(distances - median_distance)))]


def run_example_history(
    prediction: fdm.PathwayPrediction,
    positions_m: np.ndarray,
) -> ExampleAttractorHistory:
    """Run one example through the attractor while retaining full history.

    Args:
        prediction: Upstream pathway prediction.
        positions_m: Shared AC/SC distance grid.

    Returns:
        Attractor history for plotting.
    """
    _, trajectory, _, _, _, state_history = run_line_attractor(
        prediction.ac_activation[None, :].astype(np.float64),
        positions_m,
        keep_history=True,
    )
    if state_history is None:
        raise RuntimeError("Attractor state history was not retained.")
    times = np.arange(state_history.shape[0], dtype=np.float64) * ATTRACTOR_DT_S
    excitatory = state_history[:, 0, : prediction.ac_activation.shape[0]]
    readout = np.maximum(excitatory, 0.0)
    spikes = state_history_to_output_spikes(excitatory)
    return ExampleAttractorHistory(
        times_s=times,
        state_history=state_history[:, 0],
        excitatory_history=excitatory,
        readout_history=readout,
        decoded_trajectory_m=trajectory[0],
        output_spikes=spikes,
    )


def plot_pipeline_diagram(path: Path) -> str:
    """Create a block diagram of the final distance pathway.

    Args:
        path: Figure path.

    Returns:
        Saved figure path.
    """
    fig, ax = plt.subplots(figsize=(13.5, 3.2))
    ax.axis("off")
    labels = [
        "Echo\nwaveform",
        "Cochlea\nIIR + dynamic LIF",
        "VCN/VNLL\n4 kHz mask + consensus",
        "DNLL\nlate suppression",
        "IC\nCD coincidence + facilitation",
        "AC\ntopographic sharpening",
        "SC CANN\nbalanced E/I attractor",
        "Distance\nreadout",
    ]
    x = np.linspace(0.04, 0.96, len(labels))
    for idx, (xpos, label) in enumerate(zip(x, labels)):
        ax.text(
            xpos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eff6ff", edgecolor="#1d4ed8", linewidth=1.2),
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1] - 0.055, 0.5),
                xytext=(xpos + 0.055, 0.5),
                arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.4),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    ax.text(
        0.50,
        0.08,
        "Corollary discharge enters the IC as a latency-adjusted internal sweep copy.",
        ha="center",
        va="center",
        fontsize=9,
        color="#374151",
        transform=ax.transAxes,
    )
    return save_figure(fig, path)


def plot_frequency_time_rasters(
    prediction: fdm.PathwayPrediction,
    config: fdm.GlobalConfig,
    variant: fdm.PathwayVariant,
    path: Path,
) -> str:
    """Plot frequency-time spike rasters for early pathway stages.

    Args:
        prediction: Example pathway prediction.
        config: Acoustic configuration.
        variant: Primary pathway variant.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    centers_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    if variant.dynamic_cochlea_schedule is not None:
        left_dynamic = fdm._dynamic_lif_encode(prediction.cochlea.left_cochleagram, config, variant.dynamic_cochlea_schedule)
        right_dynamic = fdm._dynamic_lif_encode(prediction.cochlea.right_cochleagram, config, variant.dynamic_cochlea_schedule)
        cochlea_raster = torch.maximum(left_dynamic, right_dynamic).detach().cpu().numpy()
    else:
        cochlea_raster = torch.maximum(prediction.cochlea.left_spikes, prediction.cochlea.right_spikes).detach().cpu().numpy()
    stages = [
        ("Cochlear dynamic spike raster", cochlea_raster),
        ("VCN/VNLL consensus onset raster", np.maximum(prediction.vcn_left, prediction.vcn_right)),
        ("DNLL-gated onset raster", prediction.dnll_combined),
        ("Corollary discharge sweep", prediction.cd_raster),
    ]
    fig, axes = plt.subplots(len(stages), 1, figsize=(12.5, 8.8), sharex=True)
    for ax, (title, raster) in zip(axes, stages):
        for channel, frequency_khz in enumerate(centers_khz):
            event_times_ms = np.flatnonzero(raster[channel] > 0.0) / config.sample_rate_hz * 1_000.0
            if event_times_ms.size:
                ax.vlines(event_times_ms, frequency_khz * 0.985, frequency_khz * 1.015, color="#1d4ed8", linewidth=1.0)
        ax.axhline(fdm.VCN_MIN_RESPONSIVE_HZ / 1_000.0, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("frequency\n(kHz)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, prediction.cd_raster.shape[1] / config.sample_rate_hz * 1_000.0)
    return save_figure(fig, path)


def plot_distance_population_stages(
    prediction: fdm.PathwayPrediction,
    positions_m: np.ndarray,
    history: ExampleAttractorHistory,
    path: Path,
) -> str:
    """Plot IC, AC, SC input, and SC final activation over distance.

    Args:
        prediction: Example pathway prediction.
        positions_m: Distance grid.
        history: Attractor history for the same example.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    input_matrix, _, _, _ = build_line_attractor_matrices(positions_m)
    sc_input = normalise_ac(prediction.ac_activation[None, :]) @ input_matrix.T
    sc_input = sc_input[0] / max(float(np.max(sc_input)), 1e-12)
    final_index = int(round(ATTRACTOR_READOUT_TIME_S / ATTRACTOR_DT_S))
    final_sc = history.readout_history[final_index]
    final_sc = final_sc / max(float(np.max(final_sc)), 1e-12)
    curves = [
        ("IC coincidence population", prediction.ic_activation, "#2563eb"),
        ("AC topographic map", prediction.ac_activation, "#16a34a"),
        ("SC reflected/opponent input", sc_input, "#f59e0b"),
        ("SC attractor activation at 60 ms", final_sc, "#dc2626"),
    ]
    fig, axes = plt.subplots(len(curves), 1, figsize=(11.5, 9.2), sharex=True)
    for ax, (title, values, color) in zip(axes, curves):
        norm = values / max(float(np.max(values)), 1e-12)
        ax.plot(positions_m, norm, color=color, linewidth=2.0)
        ax.axvline(prediction.distance_m, color="#111827", linestyle="--", linewidth=1.2, label="true")
        ax.set_title(title)
        ax.set_ylabel("normalised\nactivation")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("represented distance (m)")
    axes[0].legend(frameon=False)
    return save_figure(fig, path)


def plot_line_attractor_dynamics(
    prediction: fdm.PathwayPrediction,
    positions_m: np.ndarray,
    history: ExampleAttractorHistory,
    path: Path,
) -> str:
    """Plot line-attractor activation heatmap and decoded trajectory.

    Args:
        prediction: Example pathway prediction.
        positions_m: Distance grid.
        history: Attractor history.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    time_ms = history.times_s * 1_000.0
    activation = np.maximum(history.excitatory_history, 0.0)
    activation = activation / max(float(np.max(activation)), 1e-12)
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=False)
    image = axes[0].imshow(
        activation.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[time_ms[0], time_ms[-1], positions_m[0], positions_m[-1]],
    )
    axes[0].axhline(prediction.distance_m, color="#e5e7eb", linestyle="--", linewidth=1.2)
    axes[0].axvline(ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#9ca3af", linestyle=":", linewidth=1.2)
    axes[0].set_xlabel("SC time (ms)")
    axes[0].set_ylabel("represented distance (m)")
    axes[0].set_title("Line-attractor activation over time")
    fig.colorbar(image, ax=axes[0], label="normalised E activity")

    axes[1].plot(time_ms, history.decoded_trajectory_m, color="#dc2626", linewidth=2.0, label="attractor readout")
    axes[1].axhline(prediction.distance_m, color="#111827", linestyle="--", label="true distance")
    axes[1].axvline(ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#6b7280", linestyle=":", label="readout time")
    axes[1].set_xlabel("SC time (ms)")
    axes[1].set_ylabel("decoded distance (m)")
    axes[1].set_title("Readout trajectory")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    return save_figure(fig, path)


def plot_line_attractor_output_spikes(
    prediction: fdm.PathwayPrediction,
    positions_m: np.ndarray,
    history: ExampleAttractorHistory,
    path: Path,
) -> str:
    """Plot deterministic output spikes from the line attractor.

    Args:
        prediction: Example pathway prediction.
        positions_m: Distance grid.
        history: Attractor history.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    time_ms = history.times_s * 1_000.0
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    for index, distance_m in enumerate(positions_m):
        event_times = time_ms[np.flatnonzero(history.output_spikes[index] > 0.0)]
        if event_times.size:
            ax.vlines(event_times, distance_m - 0.01, distance_m + 0.01, color="#7c2d12", linewidth=0.8)
    ax.axhline(prediction.distance_m, color="#111827", linestyle="--", linewidth=1.2, label="true distance")
    ax.set_xlabel("SC time (ms)")
    ax.set_ylabel("SC neuron preferred distance (m)")
    ax.set_title("Illustrative output spikes from SC attractor excitatory rates")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_excitatory_rate_comparison(
    prediction: fdm.PathwayPrediction,
    positions_m: np.ndarray,
    history: ExampleAttractorHistory,
    path: Path,
) -> str:
    """Show global and local readouts on the same SC attractor activity.

    Args:
        prediction: Example pathway prediction.
        positions_m: Distance grid.
        history: History for the fixed reflected/opponent attractor.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    final_index = int(round(ATTRACTOR_READOUT_TIME_S / ATTRACTOR_DT_S))
    activity = np.maximum(history.excitatory_history, 0.0)
    activity = activity / max(float(np.max(activity)), 1e-12)
    final_rate = activity[final_index]
    final_rate_2d = final_rate[None, :]
    global_readout = float(decode_center_of_mass(final_rate_2d, positions_m)[0])
    local_readout = float(decode_local_population_vector(final_rate_2d, positions_m)[0])
    local_mask = local_population_window_mask(final_rate)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    image = axes[0].imshow(
        activity.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[history.times_s[0] * 1_000.0, history.times_s[-1] * 1_000.0, positions_m[0], positions_m[-1]],
    )
    axes[0].axhline(prediction.distance_m, color="#e5e7eb", linestyle="--", linewidth=1.0)
    axes[0].axvline(ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("SC time (ms)")
    axes[0].set_ylabel("represented distance (m)")
    axes[0].set_title("Same SC attractor activity")
    fig.colorbar(image, ax=axes[0], label="normalised E")

    axes[1].plot(positions_m, final_rate, color="#dc2626", linewidth=2.0, label="final E profile")
    if np.any(local_mask):
        axes[1].fill_between(
            positions_m,
            0.0,
            final_rate,
            where=local_mask,
            color="#f59e0b",
            alpha=0.32,
            label="local-vector window",
        )
    axes[1].axvline(prediction.distance_m, color="#111827", linestyle="--", linewidth=1.0, label="true")
    axes[1].axvline(global_readout, color="#2563eb", linestyle=":", linewidth=1.4, label="global COM")
    axes[1].axvline(local_readout, color="#f59e0b", linestyle="-.", linewidth=1.4, label="local vector")
    axes[1].set_xlabel("represented distance (m)")
    axes[1].set_ylabel("normalised E")
    axes[1].set_title("Two readouts on one attractor bump")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    return save_figure(fig, path)


def plot_comparison_scatter(results: list[AttractorPredictionSet], path: Path) -> str:
    """Plot simple and attractor prediction scatter comparisons.

    Args:
        results: Prediction sets.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    for ax, result in zip(axes.flat, results):
        ax.scatter(result.true_distance_m, result.simple_distance_m, s=18, alpha=0.58, label="simple COM")
        ax.scatter(result.true_distance_m, result.attractor_distance_m, s=18, alpha=0.58, label="reflected attractor")
        ax.scatter(result.true_distance_m, result.local_distance_m, s=18, alpha=0.58, label="attractor local-vector")
        low = min(
            result.true_distance_m.min(),
            result.simple_distance_m.min(),
            result.attractor_distance_m.min(),
            result.local_distance_m.min(),
        )
        high = max(
            result.true_distance_m.max(),
            result.simple_distance_m.max(),
            result.attractor_distance_m.max(),
            result.local_distance_m.max(),
        )
        ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
        ax.set_title(result.condition)
        ax.set_xlabel("true distance (m)")
        ax.set_ylabel("predicted distance (m)")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(frameon=False)
    return save_figure(fig, path)


def plot_error_bars(results: list[AttractorPredictionSet], path: Path) -> str:
    """Plot MAE for simple and attractor readouts.

    Args:
        results: Prediction sets.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    labels = [result.condition for result in results]
    simple = [metric_dict(result.true_distance_m, result.simple_distance_m)["mae_m"] * 100.0 for result in results]
    attractor = [metric_dict(result.true_distance_m, result.attractor_distance_m)["mae_m"] * 100.0 for result in results]
    local = [metric_dict(result.true_distance_m, result.local_distance_m)["mae_m"] * 100.0 for result in results]
    x = np.arange(len(results))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width, simple, width=width, label="simple COM", color="#6b7280")
    ax.bar(x, attractor, width=width, label="reflected attractor", color="#2563eb")
    ax.bar(x + width, local, width=width, label="attractor local-vector", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("MAE (cm)")
    ax.set_title("Simple readout vs final SC line attractor")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def comparison_rows(results: list[AttractorPredictionSet]) -> list[dict[str, object]]:
    """Build comparison rows, including full-space distance subsets.

    Args:
        results: Prediction sets.

    Returns:
        List of metric rows.
    """
    rows: list[dict[str, object]] = []
    for result in results:
        masks = {"all": np.ones_like(result.true_distance_m, dtype=bool)}
        if result.true_distance_m.max() > fdm.MAX_DISTANCE_M:
            masks = {
                "<=5 m": result.true_distance_m <= fdm.MAX_DISTANCE_M,
                "<=10 m": np.ones_like(result.true_distance_m, dtype=bool),
            }
        for subset, mask in masks.items():
            rows.append(
                {
                    "condition": result.condition,
                    "subset": subset,
                    "num_samples": int(np.count_nonzero(mask)),
                    "simple_metrics": metric_dict(result.true_distance_m[mask], result.simple_distance_m[mask]),
                    "attractor_metrics": metric_dict(result.true_distance_m[mask], result.attractor_distance_m[mask]),
                    "local_metrics": metric_dict(result.true_distance_m[mask], result.local_distance_m[mask]),
                    "attractor_seconds_per_sample": result.seconds_per_sample,
                    "local_seconds_per_sample": result.local_seconds_per_sample,
                }
            )
    return rows


def write_report(
    config: fdm.GlobalConfig,
    full_config: fdm.GlobalConfig,
    variant: fdm.PathwayVariant,
    example: fdm.PathwayPrediction,
    rows: list[dict[str, object]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the final distance pipeline report.

    Args:
        config: Small-space acoustic configuration.
        full_config: Full-space acoustic configuration.
        variant: Primary pathway variant.
        example: Example prediction used for stage plots.
        rows: Comparison metric rows.
        artifacts: Generated artifact paths.
        elapsed_s: Runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _, _, recurrent_local, _ = build_line_attractor_matrices(fdm._candidate_distances(config))
    max_recurrent_eig = float(np.max(np.real(np.linalg.eigvals(recurrent_local))))
    table_rows = []
    for row in rows:
        simple = row["simple_metrics"]
        attractor = row["attractor_metrics"]
        local = row["local_metrics"]
        table_rows.append(
            "| "
            f"{row['condition']} | "
            f"{row['subset']} | "
            f"`{row['num_samples']}` | "
            f"`{simple['mae_m'] * 100.0:.3f} cm` | "
            f"`{attractor['mae_m'] * 100.0:.3f} cm` | "
            f"`{local['mae_m'] * 100.0:.3f} cm` | "
            f"`{simple['rmse_m'] * 100.0:.3f} cm` | "
            f"`{attractor['rmse_m'] * 100.0:.3f} cm` | "
            f"`{local['rmse_m'] * 100.0:.3f} cm` | "
            f"`{simple['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"`{attractor['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"`{local['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"`{row['attractor_seconds_per_sample'] * 1_000.0:.3f} ms` | "
            f"`{row['local_seconds_per_sample'] * 1_000.0:.3f} ms` |"
        )

    lines = [
        "# Final Distance Pipeline With SC Line Attractor",
        "",
        "This report is the final distance-pipeline summary before separate failure-case analysis. It keeps the current primary distance pathway unchanged through the AC distance map, then compares three SC readouts:",
        "",
        "- `simple COM`: the existing centre-of-mass readout directly from the AC map.",
        "- `reflected/opponent SC line attractor`: the finite-line input-theory attractor added after AC.",
        "- `local population-vector readout`: a peak-neighbourhood readout applied to the final SC attractor excitatory activity.",
        "",
        "The comparison is controlled because all three readouts receive the same AC activity. Any difference is caused by the final SC readout only.",
        "",
        "![Pipeline diagram](../outputs/final_distance_pipeline_with_attractor/figures/pipeline_diagram.png)",
        "",
        "## Acoustic And Pathway Setup",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| small-space distance range | `{fdm.MIN_DISTANCE_M:.2f} -> {fdm.MAX_DISTANCE_M:.2f} m` |",
        f"| full-space distance range | `{fdm.MIN_DISTANCE_M:.2f} -> {fdm.FULL_TEST_MAX_DISTANCE_M:.2f} m` |",
        f"| full-space azimuth range | `±{fdm.FULL_TEST_AZIMUTH_LIMIT_DEG:.0f} deg` |",
        f"| full-space elevation range | `±{fdm.FULL_TEST_ELEVATION_LIMIT_DEG:.0f} deg` |",
        f"| sample rate | `{config.sample_rate_hz:.0f} Hz` |",
        f"| call sweep | `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` |",
        f"| call duration | `{config.chirp_duration_s * 1_000.0:.1f} ms` |",
        f"| small-space signal duration | `{config.signal_duration_s * 1_000.0:.1f} ms` |",
        f"| full-space signal duration | `{full_config.signal_duration_s * 1_000.0:.1f} ms` |",
        f"| cochlea channels | `{fdm.NUM_CHANNELS}` |",
        f"| distance bins | `{fdm.NUM_DISTANCE_BINS}` |",
        f"| primary variant | `{variant.name}` |",
        "",
        "## Full Pipeline",
        "",
        "### 1. Echo Waveform",
        "",
        "The simulator generates a received binaural echo from the emitted FM call. In the full-space tests, azimuth changes the binaural timing/level cues and elevation applies spectral filtering before cochlear processing.",
        "",
        "### 2. Cochlea",
        "",
        "The cochlea is the optimised IIR front end from the cochlea mini-model work. It filters the waveform into frequency channels and converts activity to spikes. The current distance model uses the dynamic spike schedule:",
        "",
        "```text",
        "threshold(t): 16x -> 2.5x baseline threshold",
        "beta(t):      0.20 -> 0.60",
        "```",
        "",
        "This makes early high-amplitude noise less likely to spike while still allowing weaker later echoes to pass.",
        "",
        "### 3. VCN/VNLL",
        "",
        "The VCN/VNLL stage is simplified as a causal onset detector. It ignores channels below `4 kHz`, then requires local frequency-time consensus before emitting a first onset spike for each channel:",
        "",
        "```text",
        "count[c,t] = sum spikes in local channel-time window",
        "VCN[c,t] = first spike where count[c,t] >= 3",
        "```",
        "",
        "This replaces the much harder biological VCN/VNLL constant-latency mechanism with a robust engineering approximation.",
        "",
        "### 4. DNLL",
        "",
        "DNLL is modelled as delayed inhibition. It suppresses later onsets after the first echo sweep window, helping the pathway focus on the primary echo:",
        "",
        "```text",
        "suppress_after = first_onset + chirp_duration + padding",
        "```",
        "",
        "### 5. Corollary Discharge",
        "",
        "The corollary discharge is an internal sweep-shaped copy of the expected call response. Per-channel cochlea/VCN latency is added to the CD timing rather than subtracting latency from echo spikes, keeping the whole pathway causal.",
        "",
        "### 6. IC",
        "",
        "The IC compares the VCN/DNLL echo onset against delayed corollary-discharge spikes for every distance bin. For a two-spike LIF coincidence detector:",
        "",
        "```text",
        "delta[c,k] = abs(t_echo[c] - (t_CD[c] + delay[k]))",
        "m[c,k] = 1 + beta_IC^delta[c,k]",
        "score[c,k] = relu(m[c,k] - threshold_IC)",
        "```",
        "",
        "Neighbouring frequency channels add a soft facilitation term when they support the same candidate delay. This helps a sweep-consistent echo dominate isolated noisy events without hard-gating the system.",
        "",
        "### 7. AC",
        "",
        "The AC stage organises the IC coincidence scores into a sharper topographic distance map with a Mexican-hat interaction:",
        "",
        "```text",
        "AC = relu(IC + IC * K_mexican_hat)",
        "K = Gaussian(sigma_exc) - g_inh Gaussian(sigma_inh)",
        "```",
        "",
        "### 8. SC Readouts",
        "",
        "The baseline SC readout is the original centre of mass:",
        "",
        "```text",
        "d_hat_COM = sum_k AC[k] d[k] / sum_k AC[k]",
        "```",
        "",
        "The main upgraded SC readout is the finite-line balanced E/I attractor:",
        "",
        "```text",
        "r = [r_E, r_I]",
        "B = [M; -beta M] / sqrt(1 + beta^2)",
        "W = [[W0, -W0], [W0, -W0]]",
        "tau dr/dt = -r + W r",
        "```",
        "",
        "The final decoded distance is centre of mass over the rectified excitatory population at `60 ms`.",
        "",
        "The additional local population-vector readout is applied after the same SC attractor dynamics. It finds the peak neighbourhood in the final excitatory SC bump and computes a local centre of mass. This tests whether a local readout of the attractor bump is better than a global centre of mass over the whole SC population.",
        "",
        "## Selected SC Attractor Parameters",
        "",
        "These parameters come from the finite-line input theory work. The model uses the reflected finite-line input because it handles boundaries better than a raw Toeplitz matrix. The recurrent matrix is the balanced two-block E/I model. The alpha choice is deliberately lower than the uncapped mathematical optimum because the capped-alpha analysis showed that very large alpha requires unrealistic peak firing rates.",
        "",
        "| SC attractor parameter | Value |",
        "|---|---:|",
        f"| input family | `reflected` |",
        f"| input width | `{ATTRACTOR_INPUT_WIDTH_BINS:.0f}` bins |",
        f"| recurrent width | `{ATTRACTOR_RECURRENT_WIDTH_BINS:.0f}` bins |",
        f"| opponent beta | `{ATTRACTOR_BETA:.3f}` |",
        f"| alpha prime | `{ATTRACTOR_ALPHA_PRIME:.1f}` |",
        f"| recurrent local max eigenvalue | `{max_recurrent_eig:.3f}` |",
        f"| tau | `{ATTRACTOR_TAU_S * 1_000.0:.1f} ms` |",
        f"| simulation step | `{ATTRACTOR_DT_S * 1_000.0:.1f} ms` |",
        f"| readout time | `{ATTRACTOR_READOUT_TIME_S * 1_000.0:.1f} ms` |",
        f"| rate cap | `{ATTRACTOR_RATE_CAP_HZ:.1f} Hz` |",
        f"| local population-vector sigma | `{LOCAL_POPULATION_SIGMA_BINS:.1f}` bins |",
        f"| local population-vector peak radius | `±{LOCAL_POPULATION_RADIUS_BINS}` bins |",
        "",
        "## Example Spike Processing Path",
        "",
        f"The example below uses a target at `{example.distance_m:.2f} m`. Frequency is shown on a log axis; the dotted horizontal line marks the `4 kHz` VCN cut-off.",
        "",
        "![Frequency-time rasters](../outputs/final_distance_pipeline_with_attractor/figures/frequency_time_rasters.png)",
        "",
        "The next figure shows the conversion from IC coincidence scores to AC topographic activity, then into the final SC attractor activity over represented distance.",
        "",
        "![Distance population stages](../outputs/final_distance_pipeline_with_attractor/figures/distance_population_stages.png)",
        "",
        "The attractor dynamics are shown with SC time on the x-axis and represented distance on the y-axis. This is the key visualisation of the line attractor: the activity bump evolves over time but remains organised along the distance line.",
        "",
        "![Line attractor dynamics](../outputs/final_distance_pipeline_with_attractor/figures/line_attractor_dynamics.png)",
        "",
        "The line attractor itself is simulated as a rate model. The spike raster below is an illustrative deterministic spike conversion from the excitatory rate state, included so the final SC output can be visualised as spikes.",
        "",
        "![Line attractor output spikes](../outputs/final_distance_pipeline_with_attractor/figures/line_attractor_output_spikes.png)",
        "",
        "The figure below shows the same SC excitatory activity with both the global COM readout and the local population-vector readout marked on the final bump.",
        "",
        "![SC excitatory rate comparison](../outputs/final_distance_pipeline_with_attractor/figures/sc_excitatory_rate_comparison.png)",
        "",
        "## Readout Comparison",
        "",
        "The simple readout and attractor readout are compared on the same AC activations.",
        "",
        "![Readout MAE comparison](../outputs/final_distance_pipeline_with_attractor/figures/readout_mae_comparison.png)",
        "",
        "![Readout scatter](../outputs/final_distance_pipeline_with_attractor/figures/readout_scatter.png)",
        "",
        "| Condition | Subset | N | Simple MAE | Attractor global-COM MAE | Attractor local-vector MAE | Simple RMSE | Global-COM RMSE | Local-vector RMSE | Simple max error | Global-COM max error | Local-vector max error | Attractor runtime/sample | Attractor + local readout runtime/sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Interpretation",
        "",
        "- The final SC line attractor is now attached as a reversible readout module after AC.",
        "- The comparison is controlled: cochlea, VCN/VNLL, DNLL, IC, and AC are identical for all readouts.",
        "- The line attractor gives a biologically motivated recurrent readout and a clear population-bump visualisation.",
        "- The local population-vector variant is a readout comparator on the same attractor activity; it is useful when distant low-level tails bias the global COM, but it can fail if the attractor peak itself is wrong.",
        "- If the attractor does not improve a condition, that means the AC map already contains the relevant bias or ambiguity; the SC cannot recover information that was lost upstream.",
        "- Failure-case analysis is intentionally deferred to the next report.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the final distance pipeline report experiment.

    Returns:
        JSON-serialisable experiment payload.
    """
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    torch.manual_seed(fdm.RNG_SEED)
    np.random.seed(fdm.RNG_SEED)

    config, variant, clean_small, noisy_small = run_small_space_predictions()
    small_positions = fdm._candidate_distances(config)
    full_config, clean_full, ambient_full = run_full_space_predictions(variant)
    full_positions = fdm._candidate_distances(full_config)

    results = [
        run_condition("Small clean 0.25-5m", clean_small, small_positions),
        run_condition("Small noisy 10dB+jitter", noisy_small, small_positions),
        run_condition("Full 3D clean 0.25-10m", clean_full, full_positions),
        run_condition("Full 3D 50dB floor", ambient_full, full_positions),
    ]
    rows = comparison_rows(results)

    example = choose_example_prediction(clean_small)
    example_history = run_example_history(example, small_positions)

    artifacts = {
        "pipeline_diagram": plot_pipeline_diagram(FIGURE_DIR / "pipeline_diagram.png"),
        "frequency_time_rasters": plot_frequency_time_rasters(
            example,
            config,
            variant,
            FIGURE_DIR / "frequency_time_rasters.png",
        ),
        "distance_population_stages": plot_distance_population_stages(
            example,
            small_positions,
            example_history,
            FIGURE_DIR / "distance_population_stages.png",
        ),
        "line_attractor_dynamics": plot_line_attractor_dynamics(
            example,
            small_positions,
            example_history,
            FIGURE_DIR / "line_attractor_dynamics.png",
        ),
        "line_attractor_output_spikes": plot_line_attractor_output_spikes(
            example,
            small_positions,
            example_history,
            FIGURE_DIR / "line_attractor_output_spikes.png",
        ),
        "sc_excitatory_rate_comparison": plot_excitatory_rate_comparison(
            example,
            small_positions,
            example_history,
            FIGURE_DIR / "sc_excitatory_rate_comparison.png",
        ),
        "readout_mae_comparison": plot_error_bars(results, FIGURE_DIR / "readout_mae_comparison.png"),
        "readout_scatter": plot_comparison_scatter(results, FIGURE_DIR / "readout_scatter.png"),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "final_distance_pipeline_with_attractor",
        "elapsed_seconds": elapsed_s,
        "attractor_parameters": {
            "alpha_prime": ATTRACTOR_ALPHA_PRIME,
            "input_width_bins": ATTRACTOR_INPUT_WIDTH_BINS,
            "recurrent_width_bins": ATTRACTOR_RECURRENT_WIDTH_BINS,
            "beta": ATTRACTOR_BETA,
            "tau_s": ATTRACTOR_TAU_S,
            "dt_s": ATTRACTOR_DT_S,
            "sim_time_s": ATTRACTOR_SIM_TIME_S,
            "readout_time_s": ATTRACTOR_READOUT_TIME_S,
            "rate_cap_hz": ATTRACTOR_RATE_CAP_HZ,
            "local_population_sigma_bins": LOCAL_POPULATION_SIGMA_BINS,
            "local_population_radius_bins": LOCAL_POPULATION_RADIUS_BINS,
        },
        "conditions": [
            {
                "condition": result.condition,
                "true_distance_m": result.true_distance_m.tolist(),
                "simple_distance_m": result.simple_distance_m.tolist(),
                "attractor_distance_m": result.attractor_distance_m.tolist(),
                "local_distance_m": result.local_distance_m.tolist(),
                "attractor_seconds_per_sample": result.seconds_per_sample,
                "local_seconds_per_sample": result.local_seconds_per_sample,
            }
            for result in results
        ],
        "comparison_rows": rows,
        "example_distance_m": example.distance_m,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(config, full_config, variant, example, rows, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
