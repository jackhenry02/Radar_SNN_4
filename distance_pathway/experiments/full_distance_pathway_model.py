from __future__ import annotations

"""Full distance-pathway prototype using the final cochlea front end.

This is the first high-fidelity distance pathway prototype after the smaller
coincidence-detector mini models. It keeps the previous final cochlea model
intact, then adds simplified VCN/VNLL, DNLL, IC, AC, and SC stages.
"""

import json
import math
import sys
import time
from dataclasses import dataclass, replace
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
from mini_models.experiments.final_cochlea_model_analysis import (
    FINAL_Q_FACTOR,
    _log_spaced_centers,
    _run_final_cochlea_once,
)
from mini_models.common.signals import moving_notch_signal_config
from models.acoustics import simulate_echo_batch
from utils.common import GlobalConfig


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "full_distance_pathway"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "full_distance_pathway_model.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"
LATENCY_VECTOR_PATH = ROOT / "distance_pathway" / "outputs" / "cochlea_latency" / "cochlea_latency_samples.npy"

NUM_CHANNELS = 48
NUM_DISTANCE_BINS = 180
MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 5.0
NUM_TEST_SAMPLES = 80
REFERENCE_DISTANCE_M = 2.5
RNG_SEED = 44
LATENCY_CALIBRATION_DISTANCES_M = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, 12)

VCN_REFRACTORY_S = 0.010
VCN_LIF_BETA = 0.92
VCN_LIF_THRESHOLD_FRACTION = 0.03
DNLL_SUPPRESSION_PADDING_S = 0.00075
IC_LIF_BETA = 0.992
IC_LIF_THRESHOLD = 1.45
AC_EXCITE_SIGMA_BINS = 2.0
AC_INHIBIT_SIGMA_BINS = 8.0
AC_INHIBIT_GAIN = 0.58


@dataclass
class CochleaResult:
    """Cochlea output for one binaural received echo.

    Attributes:
        left_cochleagram: Left-ear cochleagram `[channels, time]`.
        right_cochleagram: Right-ear cochleagram `[channels, time]`.
        left_spikes: Left-ear spike raster `[channels, time]`.
        right_spikes: Right-ear spike raster `[channels, time]`.
    """

    left_cochleagram: torch.Tensor
    right_cochleagram: torch.Tensor
    left_spikes: torch.Tensor
    right_spikes: torch.Tensor


@dataclass
class PathwayPrediction:
    """Stage outputs for one full distance-pathway pass.

    Attributes:
        distance_m: True distance in metres.
        predicted_distance_m: SC centre-of-mass prediction.
        cochlea: Cochlear left/right rasters.
        vcn_left: VCN/VNLL onset raster for left ear.
        vcn_right: VCN/VNLL onset raster for right ear.
        dnll_combined: Combined bilateral onset raster after DNLL suppression.
        cd_raster: Corollary-discharge sweep raster.
        ic_activation: IC coincidence activation over distance bins.
        ac_activation: AC topographic activation over distance bins.
    """

    distance_m: float
    predicted_distance_m: float
    cochlea: CochleaResult
    vcn_left: np.ndarray
    vcn_right: np.ndarray
    dnll_combined: np.ndarray
    cd_raster: np.ndarray
    ic_activation: np.ndarray
    ac_activation: np.ndarray


@dataclass(frozen=True)
class PathwayVariant:
    """Configuration for one distance-pathway ablation.

    Attributes:
        key: Stable identifier used in JSON output.
        name: Human-readable variant name.
        vcn_input: Activity source for VCN/VNLL, either `cochleagram` or `spikes`.
        latency_samples: Per-channel latency samples added to the CD expectation.
        note: Short interpretation of what this variant changes.
    """

    key: str
    name: str
    vcn_input: str
    latency_samples: np.ndarray
    note: str


def _make_config() -> GlobalConfig:
    """Create the matched-human final-cochlea distance-pathway config.

    Returns:
        Acoustic configuration for the full distance-pathway prototype.
    """
    base = moving_notch_signal_config(GlobalConfig())
    return replace(
        base,
        num_cochlea_channels=NUM_CHANNELS,
        min_range_m=MIN_DISTANCE_M,
        max_range_m=MAX_DISTANCE_M,
        signal_duration_s=0.036,
        normalize_spike_envelope=False,
        jitter_std_s=0.0,
        noise_std=0.0,
    )


def _candidate_distances() -> np.ndarray:
    """Return distance-bin centres in metres."""
    return np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_DISTANCE_BINS)


def _candidate_delay_samples(config: GlobalConfig) -> np.ndarray:
    """Return candidate round-trip delay samples."""
    distances = _candidate_distances()
    return np.rint(
        (2.0 * distances / config.speed_of_sound_m_s) * config.sample_rate_hz
    ).astype(np.int64)


def _chirp_channel_times(config: GlobalConfig) -> np.ndarray:
    """Return the ideal corollary-discharge sweep time for each cochlea channel.

    Args:
        config: Acoustic configuration.

    Returns:
        Integer sample time at which the emitted chirp crosses each channel.
    """
    centers = _log_spaced_centers(config).detach().cpu().numpy()
    start_hz = float(config.chirp_start_hz)
    end_hz = float(config.chirp_end_hz)
    duration_s = float(config.chirp_duration_s)
    sweep_fraction = (centers - start_hz) / (end_hz - start_hz)
    sweep_fraction = np.clip(sweep_fraction, 0.0, 1.0)
    return np.rint(sweep_fraction * duration_s * config.sample_rate_hz).astype(np.int64)


def _simulate_scene(config: GlobalConfig, distance_m: float) -> torch.Tensor:
    """Simulate one clean binaural echo.

    Args:
        config: Acoustic configuration.
        distance_m: Target radius in metres.

    Returns:
        Received waveform `[ears, time]`.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([distance_m], dtype=torch.float32),
        azimuth_deg=torch.tensor([0.0], dtype=torch.float32),
        elevation_deg=torch.tensor([0.0], dtype=torch.float32),
        binaural=True,
        add_noise=False,
        include_elevation_cues=False,
        transmit_gain=config.transmit_gain,
    )
    return scene.receive[0].detach()


def _run_cochlea_binaural(config: GlobalConfig, receive: torch.Tensor) -> CochleaResult:
    """Run the final cochlea model on left and right received waveforms.

    Args:
        config: Acoustic configuration.
        receive: Binaural received waveform `[ears, time]`.

    Returns:
        Left/right cochleagram and spike rasters.
    """
    left = _run_final_cochlea_once(receive[0], config, FINAL_Q_FACTOR)
    right = _run_final_cochlea_once(receive[1], config, FINAL_Q_FACTOR)
    return CochleaResult(
        left_cochleagram=left.cochleagram,
        right_cochleagram=right.cochleagram,
        left_spikes=left.spikes,
        right_spikes=right.spikes,
    )


def _vcn_vnll_onset_detector(activity_tensor: torch.Tensor, config: GlobalConfig) -> np.ndarray:
    """Simplified causal VCN/VNLL onset detector.

    The biological VCN/VNLL system is simplified into a low-threshold LIF onset
    detector with a long refractory period. This emits at or after the cochlear
    activity arrives; latency compensation is applied to the corollary discharge
    instead of moving echo spikes earlier in time.

    Args:
        activity_tensor: Input activity `[channels, time]`. The main model uses
            the rectified cochleagram; ablations can pass the cochlear spike
            raster to test whether the raster carries enough timing detail.
        config: Acoustic configuration.

    Returns:
        Onset raster `[channels, time]`.
    """
    activity = activity_tensor.detach().cpu().numpy()
    output = np.zeros_like(activity, dtype=np.float32)
    thresholds = VCN_LIF_THRESHOLD_FRACTION * np.maximum(activity.max(axis=1), 1e-12)
    refractory_samples = int(round(VCN_REFRACTORY_S * config.sample_rate_hz))
    membrane = np.zeros(activity.shape[0], dtype=np.float64)
    blocked_until = np.zeros(activity.shape[0], dtype=np.int64)
    fired_once = np.zeros(activity.shape[0], dtype=bool)
    for time_index in range(activity.shape[1]):
        membrane *= VCN_LIF_BETA
        active = time_index >= blocked_until
        membrane[active] += activity[active, time_index]
        fired = (~fired_once) & active & (membrane >= thresholds)
        if np.any(fired):
            output[fired, time_index] = 1.0
            blocked_until[fired] = time_index + refractory_samples
            membrane[fired] = 0.0
            fired_once[fired] = True
    return output


def _vcn_input_tensor(cochlea: CochleaResult, ear: str, source: str) -> torch.Tensor:
    """Select the tensor used as VCN/VNLL input.

    Args:
        cochlea: Binaural cochlea result.
        ear: `left` or `right`.
        source: `cochleagram` or `spikes`.

    Returns:
        The selected `[channels, time]` activity tensor.
    """
    if source == "cochleagram":
        return cochlea.left_cochleagram if ear == "left" else cochlea.right_cochleagram
    if source == "spikes":
        return cochlea.left_spikes if ear == "left" else cochlea.right_spikes
    raise ValueError(f"Unknown VCN input source: {source}")


def _dnll_suppression(vcn_left: np.ndarray, vcn_right: np.ndarray, config: GlobalConfig) -> np.ndarray:
    """Apply simplified DNLL delayed inhibition and bilateral combination.

    Args:
        vcn_left: Left-ear onset raster `[channels, time]`.
        vcn_right: Right-ear onset raster `[channels, time]`.
        config: Acoustic configuration.

    Returns:
        Combined onset raster after suppressing late events.
    """
    combined = np.maximum(vcn_left, vcn_right)
    event_times = np.flatnonzero(combined.sum(axis=0) > 0)
    if event_times.size == 0:
        return combined
    suppress_after = int(event_times[0] + config.chirp_samples + DNLL_SUPPRESSION_PADDING_S * config.sample_rate_hz)
    suppressed = combined.copy()
    if suppress_after < suppressed.shape[1]:
        suppressed[:, suppress_after:] = 0.0
    return suppressed


def _make_cd_raster(
    config: GlobalConfig,
    total_time: int,
    latency_samples: np.ndarray | None = None,
) -> np.ndarray:
    """Create latency-adjusted corollary-discharge sweep raster.

    Args:
        config: Acoustic configuration.
        total_time: Number of time samples.
        latency_samples: Optional per-channel cochlea/onset latency in samples.

    Returns:
        CD raster `[channels, time]`.
    """
    cd_times = _chirp_channel_times(config)
    if latency_samples is not None:
        cd_times = cd_times + latency_samples.astype(np.int64)
    cd = np.zeros((NUM_CHANNELS, total_time), dtype=np.float32)
    valid = (cd_times >= 0) & (cd_times < total_time)
    cd[np.arange(NUM_CHANNELS)[valid], cd_times[valid]] = 1.0
    return cd


def _combined_vcn_input(cochlea: CochleaResult, source: str) -> torch.Tensor:
    """Combine the left/right VCN input source for latency calibration.

    Args:
        cochlea: Binaural cochlea result.
        source: `cochleagram` or `spikes`.

    Returns:
        Combined bilateral activity `[channels, time]`.
    """
    left = _vcn_input_tensor(cochlea, "left", source)
    right = _vcn_input_tensor(cochlea, "right", source)
    return torch.maximum(left, right)


def _calibrate_channel_latency(config: GlobalConfig, vcn_input: str = "cochleagram") -> np.ndarray:
    """Estimate fixed channel latency introduced by cochlea + VCN onset detector.

    Args:
        config: Acoustic configuration.
        vcn_input: VCN/VNLL input source, either `cochleagram` or `spikes`.

    Returns:
        Per-channel latency correction in samples.
    """
    receive = _simulate_scene(config, REFERENCE_DISTANCE_M)
    cochlea = _run_cochlea_binaural(config, receive)
    combined_input = _combined_vcn_input(cochlea, vcn_input)
    vcn_np = _vcn_vnll_onset_detector(combined_input, config) > 0.0
    cd_times = _chirp_channel_times(config)
    reference_delay = int(round((2.0 * REFERENCE_DISTANCE_M / config.speed_of_sound_m_s) * config.sample_rate_hz))
    latency = np.zeros(NUM_CHANNELS, dtype=np.int64)
    valid_latencies = []
    for channel in range(NUM_CHANNELS):
        event_times = np.flatnonzero(vcn_np[channel])
        if event_times.size == 0:
            latency[channel] = 0
            continue
        expected = int(cd_times[channel] + reference_delay)
        latency[channel] = int(event_times[0] - expected)
        valid_latencies.append(latency[channel])
    fallback = int(np.median(valid_latencies)) if valid_latencies else 0
    for channel in range(NUM_CHANNELS):
        if latency[channel] == 0:
            latency[channel] = fallback
    return latency


def _calibrate_channel_latency_over_distances(config: GlobalConfig, vcn_input: str) -> np.ndarray:
    """Estimate detector latency from several distances.

    This is used for ablations where no saved latency vector exists. It mirrors
    the separate cochlea latency experiment by taking the median per-channel
    latency over a fixed calibration grid.

    Args:
        config: Acoustic configuration.
        vcn_input: VCN/VNLL input source, either `cochleagram` or `spikes`.

    Returns:
        Per-channel median latency correction in samples.
    """
    cd_times = _chirp_channel_times(config)
    latency_rows = []
    for distance_m in LATENCY_CALIBRATION_DISTANCES_M:
        receive = _simulate_scene(config, float(distance_m))
        cochlea = _run_cochlea_binaural(config, receive)
        combined_input = _combined_vcn_input(cochlea, vcn_input)
        vcn_np = _vcn_vnll_onset_detector(combined_input, config) > 0.0
        round_trip_delay = int(round((2.0 * float(distance_m) / config.speed_of_sound_m_s) * config.sample_rate_hz))
        row = np.full(NUM_CHANNELS, np.nan, dtype=np.float64)
        for channel in range(NUM_CHANNELS):
            event_times = np.flatnonzero(vcn_np[channel])
            if event_times.size:
                row[channel] = int(event_times[0]) - int(cd_times[channel] + round_trip_delay)
        latency_rows.append(row)
    latency_matrix = np.vstack(latency_rows)
    median_latency = np.rint(np.nanmedian(latency_matrix, axis=0))
    if np.any(~np.isfinite(median_latency)):
        finite = median_latency[np.isfinite(median_latency)]
        fallback = int(np.median(finite)) if finite.size else 0
        median_latency = np.where(np.isfinite(median_latency), median_latency, fallback)
    return median_latency.astype(np.int64)


def _load_channel_latency(config: GlobalConfig) -> np.ndarray:
    """Load the saved refractory-LIF latency vector or fall back to calibration.

    Args:
        config: Acoustic configuration.

    Returns:
        Per-channel latency in samples. This is added to the CD expectation.
    """
    if LATENCY_VECTOR_PATH.exists():
        latency = np.load(LATENCY_VECTOR_PATH).astype(np.int64)
        if latency.shape == (NUM_CHANNELS,):
            return latency
    return _calibrate_channel_latency(config)


def _ic_lif_coincidence(
    dnll_onsets: np.ndarray,
    config: GlobalConfig,
    latency_samples: np.ndarray,
) -> np.ndarray:
    """Compute IC LIF coincidence activation over candidate distances.

    For two input spikes with unit weights, the LIF membrane at the second
    arrival is `1 + beta^delta`. A detector emits/activates if this exceeds
    the threshold. This is equivalent to simulating the two-spike LIF response
    for the clean event case, but avoids a large time loop for every sample.

    Args:
        dnll_onsets: Combined VCN/VNLL onset raster after DNLL suppression.
        config: Acoustic configuration.

    Returns:
        IC activation over candidate distance bins.
    """
    cd_times = _chirp_channel_times(config) + latency_samples.astype(np.int64)
    candidate_delays = _candidate_delay_samples(config)
    activation = np.zeros(NUM_DISTANCE_BINS, dtype=np.float64)
    for channel in range(NUM_CHANNELS):
        echo_times = np.flatnonzero(dnll_onsets[channel] > 0.0)
        if echo_times.size == 0:
            continue
        echo_time = int(echo_times[0])
        expected_times = cd_times[channel] + candidate_delays
        delta = np.abs(echo_time - expected_times)
        membrane_peak = 1.0 + np.power(IC_LIF_BETA, delta)
        activation += np.maximum(0.0, membrane_peak - IC_LIF_THRESHOLD)
    return activation


def _mexican_hat_kernel() -> np.ndarray:
    """Return a static Mexican-hat lateral interaction kernel."""
    radius = int(math.ceil(AC_INHIBIT_SIGMA_BINS * 3.0))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    excite = np.exp(-0.5 * (axis / AC_EXCITE_SIGMA_BINS) ** 2)
    inhibit = np.exp(-0.5 * (axis / AC_INHIBIT_SIGMA_BINS) ** 2)
    excite /= excite.sum()
    inhibit /= inhibit.sum()
    return excite - AC_INHIBIT_GAIN * inhibit


def _ac_topographic_map(ic_activation: np.ndarray) -> np.ndarray:
    """Apply AC topographic sharpening with a Mexican-hat kernel."""
    kernel = _mexican_hat_kernel()
    sharpened = np.convolve(ic_activation, kernel, mode="same")
    ac = np.maximum(0.0, ic_activation + sharpened)
    return ac


def _sc_center_of_mass(ac_activation: np.ndarray) -> float:
    """Read out distance using SC centre of mass.

    Args:
        ac_activation: AC population activity over distance bins.

    Returns:
        Predicted distance in metres.
    """
    distances = _candidate_distances()
    total = ac_activation.sum()
    if total <= 1e-12:
        return float(distances[len(distances) // 2])
    return float(np.sum(ac_activation * distances) / total)


def _predict_one(
    config: GlobalConfig,
    distance_m: float,
    latency_samples: np.ndarray,
    vcn_input: str = "cochleagram",
) -> PathwayPrediction:
    """Run the full distance pathway for one target distance."""
    receive = _simulate_scene(config, distance_m)
    cochlea = _run_cochlea_binaural(config, receive)
    vcn_left = _vcn_vnll_onset_detector(_vcn_input_tensor(cochlea, "left", vcn_input), config)
    vcn_right = _vcn_vnll_onset_detector(_vcn_input_tensor(cochlea, "right", vcn_input), config)
    dnll = _dnll_suppression(vcn_left, vcn_right, config)
    cd = _make_cd_raster(config, receive.shape[-1], latency_samples)
    ic = _ic_lif_coincidence(dnll, config, latency_samples)
    ac = _ac_topographic_map(ic)
    predicted = _sc_center_of_mass(ac)
    return PathwayPrediction(
        distance_m=float(distance_m),
        predicted_distance_m=predicted,
        cochlea=cochlea,
        vcn_left=vcn_left,
        vcn_right=vcn_right,
        dnll_combined=dnll,
        cd_raster=cd,
        ic_activation=ic,
        ac_activation=ac,
    )


def _simulate_ic_membranes_for_plot(
    prediction: PathwayPrediction,
    config: GlobalConfig,
    latency_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Simulate a few explicit LIF membrane traces for visual explanation.

    Args:
        prediction: Pathway prediction to visualise.
        config: Acoustic configuration.

    Returns:
        Tuple `(time_ms, membrane_traces, labels)`.
    """
    candidate_distances = _candidate_distances()
    nearest_index = int(np.argmin(np.abs(candidate_distances - prediction.distance_m)))
    candidate_indices = [
        max(0, nearest_index - 18),
        nearest_index,
        min(NUM_DISTANCE_BINS - 1, nearest_index + 18),
    ]
    labels = [f"{candidate_distances[index]:.2f} m" for index in candidate_indices]
    channel_activity = prediction.dnll_combined.sum(axis=1)
    channel = int(np.argmax(channel_activity))
    echo_times = np.flatnonzero(prediction.dnll_combined[channel] > 0.0)
    if echo_times.size == 0:
        echo_time = 0
    else:
        echo_time = int(echo_times[0])
    cd_time = int(_chirp_channel_times(config)[channel] + latency_samples[channel])
    candidate_delays = _candidate_delay_samples(config)
    total_time = prediction.dnll_combined.shape[1]
    membranes = np.zeros((len(candidate_indices), total_time), dtype=np.float64)
    for trace_index, candidate_index in enumerate(candidate_indices):
        membrane = 0.0
        delayed_cd_time = cd_time + int(candidate_delays[candidate_index])
        for time_index in range(total_time):
            membrane *= IC_LIF_BETA
            if time_index == delayed_cd_time:
                membrane += 1.0
            if time_index == echo_time:
                membrane += 1.0
            if membrane >= IC_LIF_THRESHOLD:
                # Keep the peak visible for explanation, then subtractively reset.
                membranes[trace_index, time_index] = membrane
                membrane -= IC_LIF_THRESHOLD
            else:
                membranes[trace_index, time_index] = membrane
    time_ms = np.arange(total_time) / config.sample_rate_hz * 1_000.0
    return time_ms, membranes, labels


def _plot_stage_rasters(prediction: PathwayPrediction, config: GlobalConfig, path: Path) -> str:
    """Plot cochlea, VCN/VNLL, DNLL, and CD rasters for one example."""
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    time_ms = np.arange(prediction.cochlea.left_spikes.shape[1]) / config.sample_rate_hz * 1_000.0
    cochleagram = torch.maximum(
        prediction.cochlea.left_cochleagram,
        prediction.cochlea.right_cochleagram,
    ).detach().cpu().numpy()
    thresholds = VCN_LIF_THRESHOLD_FRACTION * np.maximum(cochleagram.max(axis=1), 1e-12)
    cochlea_activity = (cochleagram >= thresholds[:, None]).astype(np.float32)
    stages = [
        ("Cochleagram activity crossing VCN threshold", cochlea_activity),
        ("VCN/VNLL first-onset code", np.maximum(prediction.vcn_left, prediction.vcn_right)),
        ("DNLL-gated onset code", prediction.dnll_combined),
        ("Corollary-discharge sweep", prediction.cd_raster),
    ]
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 9), sharex=True)
    for ax, (title, raster) in zip(axes, stages):
        for channel, frequency_khz in enumerate(centers_khz):
            event_times = np.flatnonzero(raster[channel] > 0.0) / config.sample_rate_hz * 1_000.0
            if event_times.size:
                ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color="#1d4ed8")
        ax.set_yscale("log")
        ax.set_ylabel("kHz")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, time_ms[-1])
    return save_figure(fig, path)


def _plot_population_progression(
    prediction: PathwayPrediction,
    config: GlobalConfig,
    latency_samples: np.ndarray,
    path: Path,
) -> str:
    """Plot IC, AC, and SC population activity."""
    distances = _candidate_distances()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    axes[0].plot(distances, prediction.ic_activation, color="#2563eb", linewidth=2.0)
    axes[0].axvline(prediction.distance_m, color="#111827", linestyle="--", label="true")
    axes[0].set_title("IC LIF coincidence population")
    axes[0].set_ylabel("activation")
    axes[0].legend()

    axes[1].plot(distances, prediction.ac_activation, color="#16a34a", linewidth=2.0)
    axes[1].axvline(prediction.distance_m, color="#111827", linestyle="--", label="true")
    axes[1].axvline(prediction.predicted_distance_m, color="#dc2626", linestyle=":", label="SC COM")
    axes[1].set_title("AC topographic map after Mexican-hat sharpening")
    axes[1].set_ylabel("activation")
    axes[1].legend()

    time_ms, membranes, labels = _simulate_ic_membranes_for_plot(prediction, config, latency_samples)
    for trace, label in zip(membranes, labels):
        axes[2].plot(time_ms, trace, linewidth=1.6, label=label)
    axes[2].axhline(IC_LIF_THRESHOLD, color="#dc2626", linestyle="--", label="threshold")
    axes[2].set_title("Example IC LIF membrane traces for short / matched / long bins")
    axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("membrane")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_mexican_hat_matrix(path: Path) -> str:
    """Plot the AC Mexican-hat lateral interaction matrix.

    Args:
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    kernel = _mexican_hat_kernel()
    radius = len(kernel) // 2
    matrix = np.zeros((NUM_DISTANCE_BINS, NUM_DISTANCE_BINS), dtype=np.float64)
    for row in range(NUM_DISTANCE_BINS):
        for offset, weight in enumerate(kernel):
            col = row + offset - radius
            if 0 <= col < NUM_DISTANCE_BINS:
                matrix[row, col] = weight

    distances = _candidate_distances()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    image = axes[0].imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[distances.min(), distances.max(), distances.min(), distances.max()],
    )
    axes[0].set_xlabel("source distance bin (m)")
    axes[0].set_ylabel("target distance bin (m)")
    axes[0].set_title("AC Mexican-hat lateral interaction matrix")
    fig.colorbar(image, ax=axes[0], label="interaction weight")

    offsets = np.arange(-radius, radius + 1)
    bin_width_m = distances[1] - distances[0]
    axes[1].plot(offsets * bin_width_m, kernel, color="#111827", linewidth=2.0)
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0)
    axes[1].set_xlabel("distance offset (m)")
    axes[1].set_ylabel("weight")
    axes[1].set_title("Mexican-hat kernel profile")
    axes[1].grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_accuracy(predictions: list[PathwayPrediction], path: Path) -> str:
    """Plot true-vs-predicted distance and error histogram."""
    true = np.array([prediction.distance_m for prediction in predictions])
    pred = np.array([prediction.predicted_distance_m for prediction in predictions])
    error_cm = (pred - true) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(true, pred, s=18, alpha=0.7)
    axes[0].plot([MIN_DISTANCE_M, MAX_DISTANCE_M], [MIN_DISTANCE_M, MAX_DISTANCE_M], color="#111827")
    axes[0].set_xlabel("true distance (m)")
    axes[0].set_ylabel("predicted distance (m)")
    axes[0].set_title("SC centre-of-mass distance readout")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(error_cm, bins=20, color="#2563eb", alpha=0.8)
    axes[1].axvline(0.0, color="#111827")
    axes[1].set_xlabel("error (cm)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Prediction error")
    axes[1].grid(True, alpha=0.25)
    return save_figure(fig, path)


def _metrics(predictions: list[PathwayPrediction]) -> dict[str, float]:
    """Calculate scalar distance-prediction metrics."""
    true = np.array([prediction.distance_m for prediction in predictions])
    pred = np.array([prediction.predicted_distance_m for prediction in predictions])
    error = pred - true
    return {
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "bias_m": float(np.mean(error)),
    }


def _make_variants(config: GlobalConfig, latency_samples: np.ndarray) -> list[PathwayVariant]:
    """Create the current model and requested ablation variants.

    Args:
        config: Acoustic configuration.
        latency_samples: Saved refractory-LIF cochleagram latency vector.

    Returns:
        Variant definitions to evaluate on the same distance set.
    """
    zero_latency = np.zeros(NUM_CHANNELS, dtype=np.int64)
    spike_latency = _calibrate_channel_latency_over_distances(config, "spikes")
    return [
        PathwayVariant(
            key="cochleagram_lif_latency_adjusted",
            name="Current: cochleagram LIF + latency-adjusted CD",
            vcn_input="cochleagram",
            latency_samples=latency_samples,
            note="Current causal prototype; VCN reads cochleagram and CD expectation is latency-adjusted.",
        ),
        PathwayVariant(
            key="cochleagram_lif_no_latency",
            name="Ablation: cochleagram LIF, no latency vector",
            vcn_input="cochleagram",
            latency_samples=zero_latency,
            note="Tests whether the CD/IC latency vector is responsible for the timing accuracy.",
        ),
        PathwayVariant(
            key="spike_raster_lif_latency_adjusted",
            name="Ablation: spike-raster LIF + matched latency-adjusted CD",
            vcn_input="spikes",
            latency_samples=spike_latency,
            note="Tests whether the VCN can read the cochlear spike raster instead of the cochleagram.",
        ),
    ]


def _run_variant_predictions(
    config: GlobalConfig,
    distances: np.ndarray,
    variant: PathwayVariant,
) -> list[PathwayPrediction]:
    """Run one variant over the shared test distances.

    Args:
        config: Acoustic configuration.
        distances: Test distances in metres.
        variant: Variant configuration.

    Returns:
        Predictions for all distances.
    """
    return [
        _predict_one(
            config,
            float(distance),
            variant.latency_samples,
            vcn_input=variant.vcn_input,
        )
        for distance in distances
    ]


def _write_report(
    config: GlobalConfig,
    latency_samples: np.ndarray,
    predictions: list[PathwayPrediction],
    variant_summaries: list[dict[str, object]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the full distance-pathway report."""
    metric_values = _metrics(predictions)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_delays = _candidate_delay_samples(config)
    lines = [
        "# Full Distance Pathway Model",
        "",
        "This report introduces the first full distance-pathway prototype. It starts with the final cochlea front end and builds a biologically structured distance pathway around it: cochlea, VCN/VNLL, DNLL, corollary discharge, IC, AC, and SC.",
        "",
        "## High-Level Pipeline",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Received binaural echo] --> B[Cochlea<br/>IIR + LIF spikes]",
        "    B --> C[VCN/VNLL<br/>onset sharpening]",
        "    C --> D[DNLL<br/>late inhibition]",
        "    E[Corollary discharge<br/>ideal sweep copy] --> F[IC<br/>LIF coincidence bank]",
        "    D --> F",
        "    F --> G[AC<br/>topographic map + Mexican hat]",
        "    G --> H[SC<br/>centre of mass readout]",
        "    H --> I[distance estimate]",
        "```",
        "",
        "The lower cochlear and onset stages are explicitly bilateral. The IC/AC/SC map is simplified into a single combined distance map covering the whole tested distance range.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz} Hz` |",
        f"| chirp | `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` |",
        f"| chirp duration | `{config.chirp_duration_s * 1_000.0:.1f} ms` |",
        f"| signal duration | `{config.signal_duration_s * 1_000.0:.1f} ms` |",
        f"| cochlea channels | `{config.num_cochlea_channels}` |",
        f"| cochlea Q factor | `{FINAL_Q_FACTOR}` |",
        f"| distance range | `{MIN_DISTANCE_M} -> {MAX_DISTANCE_M} m` |",
        f"| distance bins | `{NUM_DISTANCE_BINS}` |",
        f"| candidate delay range | `{int(candidate_delays.min())} -> {int(candidate_delays.max())} samples` |",
        f"| IC LIF beta | `{IC_LIF_BETA}` |",
        f"| IC LIF threshold | `{IC_LIF_THRESHOLD}` |",
        f"| VCN LIF beta | `{VCN_LIF_BETA}` |",
        f"| VCN threshold fraction | `{VCN_LIF_THRESHOLD_FRACTION}` |",
        f"| AC Mexican-hat inhibit gain | `{AC_INHIBIT_GAIN}` |",
        "",
        "## Stage Details",
        "",
        "### 1. Cochlea",
        "",
        "The cochlea is the final model developed in the cochlea mini-model work: active-window detection, IIR resonator filterbank, half-wave rectification, and TorchScript LIF spike encoding. In this distance-pathway prototype, the VCN/VNLL onset detector reads the rectified cochleagram activity, because the latency experiment showed this gives a more stable onset than the later cochlear spike raster.",
        "",
        "```text",
        "y_c[n] = b0_c*x[n] + 2*r_c*cos(theta_c)*y_c[n-1] - r_c^2*y_c[n-2]",
        "v_c[n] = beta*v_c[n-1] + relu(y_c[n])",
        "spike_c[n] = 1 if v_c[n] >= threshold",
        "```",
        "",
        "### 2. VCN/VNLL",
        "",
        "The VCN/VNLL stage is simplified to a low-threshold LIF onset detector with a long refractory period. This keeps the model causal: the VCN/VNLL onset is emitted at the observed cochleagram onset, not moved earlier in time.",
        "",
        "```text",
        "v_c[t] = beta*v_c[t-1] + cochleagram_c[t]",
        "threshold_c = threshold_fraction*max_t(cochleagram_c[t])",
        "t_vcn,c = first t where v_c[t] >= threshold_c",
        "```",
        "",
        f"The saved cochlea-latency vector is not subtracted from the VCN spikes. It is applied to the corollary-discharge expectation instead. The latency vector ranges from `{int(latency_samples.min())}` to `{int(latency_samples.max())}` samples.",
        "",
        "### 3. DNLL",
        "",
        "The DNLL is simplified as delayed inhibition. After the first echo sweep begins, events after the primary sweep window are suppressed:",
        "",
        "```text",
        "suppress_after = first_onset + chirp_duration + padding",
        "```",
        "",
        "This blocks late secondary echoes in the simplest case. It would need to be relaxed or made object-aware for multi-object tracking.",
        "",
        "### 4. Corollary Discharge",
        "",
        "The corollary discharge is an internal ideal sweep. Each channel receives one spike at the expected time that the emitted chirp crosses that channel frequency, then the saved cochlea/onset latency vector is added to align the CD expectation with causal VCN/VNLL echo onsets.",
        "",
        "```text",
        "f(t) = f_start + (f_end - f_start)*t/T",
        "t_cd,c = T * (f_c - f_start)/(f_end - f_start) + latency_c",
        "```",
        "",
        "### 5. IC LIF Coincidence Bank",
        "",
        "The IC compares the VCN/VNLL echo onset against delayed corollary-discharge spikes for every candidate distance. For clean two-spike coincidence, the LIF membrane peak can be calculated directly:",
        "",
        "```text",
        "delta_c,k = abs(t_echo,c - (t_cd,c + delay_k))",
        "m_c,k = 1 + beta^delta_c,k",
        "IC_k = sum_c relu(m_c,k - threshold)",
        "```",
        "",
        "This is equivalent to a thresholded LIF coincidence detector for two unit input spikes, but evaluated in closed form for speed.",
        "",
        "### 6. AC Topographic Map",
        "",
        "The AC organises the IC population into a sharper distance map using a static Mexican-hat lateral interaction:",
        "",
        "```text",
        "K = Gaussian(sigma_exc) - g_inh*Gaussian(sigma_inh)",
        "AC = relu(IC + conv(IC, K))",
        "```",
        "",
        "![Mexican hat matrix](../outputs/full_distance_pathway/figures/mexican_hat_matrix.png)",
        "",
        "### 7. SC Readout",
        "",
        "The SC readout uses centre of mass over the AC population:",
        "",
        "```text",
        "d_hat = sum_k AC_k*d_k / sum_k AC_k",
        "```",
        "",
        "This uses the whole population, gives sub-bin distance estimates, and resembles reading the mean of a posterior-like activity distribution.",
        "",
        "## Example Stage Progression",
        "",
        "![Stage rasters](../outputs/full_distance_pathway/figures/stage_rasters.png)",
        "",
        "![Population progression](../outputs/full_distance_pathway/figures/population_progression.png)",
        "",
        "## Accuracy Test",
        "",
        f"The first test uses `{NUM_TEST_SAMPLES}` clean distances sampled uniformly from `{MIN_DISTANCE_M}` to `{MAX_DISTANCE_M} m`.",
        "",
        "![Accuracy](../outputs/full_distance_pathway/figures/accuracy.png)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| MAE | `{metric_values['mae_m'] * 100.0:.3f} cm` |",
        f"| RMSE | `{metric_values['rmse_m'] * 100.0:.3f} cm` |",
        f"| max abs error | `{metric_values['max_abs_error_m'] * 100.0:.3f} cm` |",
        f"| bias | `{metric_values['bias_m'] * 100.0:.3f} cm` |",
        "",
        "## Ablation Comparison",
        "",
        "The following variants were run on the same clean `80`-distance test set. The goal is to separate the benefit of the latency-adjusted CD/IC comparison from the benefit of reading the cochleagram rather than the later cochlear spike raster.",
        "",
        "| Variant | VCN input | CD latency vector | MAE | RMSE | Max abs error | Bias | Interpretation |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for summary in variant_summaries:
        metrics = summary["metrics"]
        lines.append(
            "| "
            f"{summary['name']} | "
            f"`{summary['vcn_input']}` | "
            f"{summary['latency_description']} | "
            f"`{metrics['mae_m'] * 100.0:.3f} cm` | "
            f"`{metrics['rmse_m'] * 100.0:.3f} cm` | "
            f"`{metrics['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"`{metrics['bias_m'] * 100.0:.3f} cm` | "
            f"{summary['note']} |"
        )
    lines.extend(
        [
        "",
        "The no-latency ablation isolates the importance of the per-channel timing correction. The spike-raster ablation tests whether the VCN/VNLL onset stage can be driven by the already-spiking cochlea output rather than by the rectified cochleagram activity.",
        "",
        "## Comparison To Previous Full Models",
        "",
        "The table below compares the distance error here against the old trained multi-output models. This is useful context, but it is not a perfectly fair benchmark: the old models estimated distance, azimuth, and elevation together, while this new prototype is a clean distance-only pathway with no angle variation or noise.",
        "",
        "| Model / result | Task | Distance MAE |",
        "|---|---|---:|",
        "| Round 4 combined model | full distance + azimuth + elevation | `7.86 cm` |",
        "| Round 3 `2B + 3` | full distance + azimuth + elevation | `6.46 cm` |",
        "| Round 5 trained-once fixed ridge decoder | full distance + azimuth + elevation with fixed tuned decoder | `4.38 cm` |",
        f"| Full distance pathway prototype | clean distance-only pathway, `{MIN_DISTANCE_M} -> {MAX_DISTANCE_M} m` | `{metric_values['mae_m'] * 100.0:.2f} cm` |",
        "",
        "On nominal distance MAE, this new distance-only pathway is better than the previous full models. The correct interpretation is not that the whole new model is already better overall, because it does not yet solve azimuth/elevation and is tested under cleaner conditions. The useful conclusion is narrower: the new structured distance pathway works as a distance estimator and is competitive enough to justify developing it further.",
        "",
        "## Causality Update",
        "",
        "The previous prototype subtracted the latency vector from echo onsets, which could make VCN/VNLL and DNLL spikes appear before the cochlea output. This version fixes that: VCN/VNLL and DNLL stay causal, and the latency vector is added to the corollary-discharge expectation inside the CD/IC comparison.",
        "",
        "## Interpretation",
        "",
        "- The model now has the intended high-level biological pathway structure rather than just a standalone coincidence detector.",
        "- The VCN/VNLL stage is deliberately simplified; robust biological onset coding is difficult to tune, and here it is represented by a causal low-threshold refractory-LIF onset detector.",
        "- The IC stage is still a simplified LIF coincidence model. It uses the closed-form two-spike LIF peak rather than time-stepping every IC neuron for every sample.",
        "- The AC and SC stages give a smooth population readout, which is useful for sub-bin distance estimates.",
        "- The next optimisation step is to replace dense cochlear rasters with coordinate events so the chosen coordinate accumulator can be used downstream.",
        "- This should be counted as a successful first full-distance-pathway prototype: the full chain from cochlea to SC readout produces a structured distance population and low clean distance error while preserving causal onset timing.",
        "",
        "## Generated Files",
        "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the full distance-pathway prototype and write the report."""
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    ensure_dir(OUTPUT_DIR)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    config = _make_config()
    latency_samples = _load_channel_latency(config)
    rng = np.random.default_rng(RNG_SEED)
    distances = rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M, size=NUM_TEST_SAMPLES)
    variants = _make_variants(config, latency_samples)
    variant_predictions = {
        variant.key: _run_variant_predictions(config, distances, variant)
        for variant in variants
    }
    predictions = variant_predictions["cochleagram_lif_latency_adjusted"]
    example_index = int(np.argmin(np.abs(distances - 3.0)))
    example = predictions[example_index]
    variant_summaries = []
    for variant in variants:
        variant_summaries.append(
            {
                "key": variant.key,
                "name": variant.name,
                "vcn_input": variant.vcn_input,
                "latency_description": "none" if np.all(variant.latency_samples == 0) else "matched",
                "latency_min_samples": int(variant.latency_samples.min()),
                "latency_max_samples": int(variant.latency_samples.max()),
                "note": variant.note,
                "metrics": _metrics(variant_predictions[variant.key]),
            }
        )

    artifacts = {
        "stage_rasters": _plot_stage_rasters(example, config, FIGURE_DIR / "stage_rasters.png"),
        "population_progression": _plot_population_progression(
            example,
            config,
            latency_samples,
            FIGURE_DIR / "population_progression.png",
        ),
        "mexican_hat_matrix": _plot_mexican_hat_matrix(FIGURE_DIR / "mexican_hat_matrix.png"),
        "accuracy": _plot_accuracy(predictions, FIGURE_DIR / "accuracy.png"),
    }
    elapsed_s = time.perf_counter() - start
    metric_values = _metrics(predictions)
    payload = {
        "experiment": "full_distance_pathway_model",
        "elapsed_seconds": elapsed_s,
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "chirp_start_hz": config.chirp_start_hz,
            "chirp_end_hz": config.chirp_end_hz,
            "chirp_duration_s": config.chirp_duration_s,
            "signal_duration_s": config.signal_duration_s,
            "num_channels": config.num_cochlea_channels,
            "cochlea_q_factor": FINAL_Q_FACTOR,
            "distance_bins": NUM_DISTANCE_BINS,
            "ic_lif_beta": IC_LIF_BETA,
            "ic_lif_threshold": IC_LIF_THRESHOLD,
            "vcn_lif_beta": VCN_LIF_BETA,
            "vcn_lif_threshold_fraction": VCN_LIF_THRESHOLD_FRACTION,
            "ac_inhibit_gain": AC_INHIBIT_GAIN,
        },
        "latency_vector_path": str(LATENCY_VECTOR_PATH),
        "latency_samples": latency_samples.tolist(),
        "metrics": metric_values,
        "variant_summaries": variant_summaries,
        "example": {
            "true_distance_m": example.distance_m,
            "predicted_distance_m": example.predicted_distance_m,
        },
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(config, latency_samples, predictions, variant_summaries, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
