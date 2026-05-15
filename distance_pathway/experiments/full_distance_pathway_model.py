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

NUM_CHANNELS = 48
NUM_DISTANCE_BINS = 180
MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 5.0
NUM_TEST_SAMPLES = 80
REFERENCE_DISTANCE_M = 2.5
RNG_SEED = 44

VCN_REFRACTORY_S = 0.010
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


def _vcn_vnll_onset_detector(
    spikes: torch.Tensor,
    latency_samples: np.ndarray,
    config: GlobalConfig,
) -> np.ndarray:
    """Simplified VCN/VNLL onset detector with latency compensation.

    The biological VCN/VNLL system is simplified into a first-spike detector
    with a long refractory period. A fixed per-channel latency calibration is
    subtracted to approximate constant-latency onset coding.

    Args:
        spikes: Cochlear spike raster `[channels, time]`.
        latency_samples: Per-channel latency compensation in samples.
        config: Acoustic configuration.

    Returns:
        Onset raster `[channels, time]`.
    """
    spike_np = spikes.detach().cpu().numpy() > 0.0
    output = np.zeros_like(spike_np, dtype=np.float32)
    refractory_samples = int(round(VCN_REFRACTORY_S * config.sample_rate_hz))
    for channel in range(spike_np.shape[0]):
        event_times = np.flatnonzero(spike_np[channel])
        if event_times.size == 0:
            continue
        first_time = int(event_times[0] - latency_samples[channel])
        if first_time < 0 or first_time >= spike_np.shape[1]:
            continue
        output[channel, first_time] = 1.0
        # The long refractory period makes later spikes irrelevant for this
        # simplified clean single-object model.
        _ = refractory_samples
    return output


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


def _make_cd_raster(config: GlobalConfig, total_time: int) -> np.ndarray:
    """Create ideal corollary-discharge sweep raster.

    Args:
        config: Acoustic configuration.
        total_time: Number of time samples.

    Returns:
        CD raster `[channels, time]`.
    """
    cd_times = _chirp_channel_times(config)
    cd = np.zeros((NUM_CHANNELS, total_time), dtype=np.float32)
    valid = (cd_times >= 0) & (cd_times < total_time)
    cd[np.arange(NUM_CHANNELS)[valid], cd_times[valid]] = 1.0
    return cd


def _calibrate_channel_latency(config: GlobalConfig) -> np.ndarray:
    """Estimate fixed channel latency introduced by cochlea + onset detector.

    Args:
        config: Acoustic configuration.

    Returns:
        Per-channel latency correction in samples.
    """
    receive = _simulate_scene(config, REFERENCE_DISTANCE_M)
    cochlea = _run_cochlea_binaural(config, receive)
    spike_np = torch.maximum(cochlea.left_spikes, cochlea.right_spikes).detach().cpu().numpy() > 0.0
    cd_times = _chirp_channel_times(config)
    reference_delay = int(round((2.0 * REFERENCE_DISTANCE_M / config.speed_of_sound_m_s) * config.sample_rate_hz))
    latency = np.zeros(NUM_CHANNELS, dtype=np.int64)
    valid_latencies = []
    for channel in range(NUM_CHANNELS):
        event_times = np.flatnonzero(spike_np[channel])
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


def _ic_lif_coincidence(
    dnll_onsets: np.ndarray,
    config: GlobalConfig,
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
    cd_times = _chirp_channel_times(config)
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


def _predict_one(config: GlobalConfig, distance_m: float, latency_samples: np.ndarray) -> PathwayPrediction:
    """Run the full distance pathway for one target distance."""
    receive = _simulate_scene(config, distance_m)
    cochlea = _run_cochlea_binaural(config, receive)
    vcn_left = _vcn_vnll_onset_detector(cochlea.left_spikes, latency_samples, config)
    vcn_right = _vcn_vnll_onset_detector(cochlea.right_spikes, latency_samples, config)
    dnll = _dnll_suppression(vcn_left, vcn_right, config)
    cd = _make_cd_raster(config, receive.shape[-1])
    ic = _ic_lif_coincidence(dnll, config)
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
    cd_time = int(_chirp_channel_times(config)[channel])
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
    cochlea_spikes = torch.maximum(
        prediction.cochlea.left_spikes,
        prediction.cochlea.right_spikes,
    ).detach().cpu().numpy()
    stages = [
        ("Cochlea output spikes", cochlea_spikes),
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

    time_ms, membranes, labels = _simulate_ic_membranes_for_plot(prediction, config)
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


def _write_report(
    config: GlobalConfig,
    latency_samples: np.ndarray,
    predictions: list[PathwayPrediction],
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
        f"| AC Mexican-hat inhibit gain | `{AC_INHIBIT_GAIN}` |",
        "",
        "## Stage Details",
        "",
        "### 1. Cochlea",
        "",
        "The cochlea is the final model developed in the cochlea mini-model work: active-window detection, IIR resonator filterbank, half-wave rectification, and TorchScript LIF spike encoding.",
        "",
        "```text",
        "y_c[n] = b0_c*x[n] + 2*r_c*cos(theta_c)*y_c[n-1] - r_c^2*y_c[n-2]",
        "v_c[n] = beta*v_c[n-1] + relu(y_c[n])",
        "spike_c[n] = 1 if v_c[n] >= threshold",
        "```",
        "",
        "### 2. VCN/VNLL",
        "",
        "The VCN/VNLL stage is simplified to a first-onset detector with a long refractory period. A fixed channel latency calibration is subtracted so that cochlear filter/LIF latency is approximately converted into a constant-latency onset code.",
        "",
        "```text",
        "t_vcn,c = first_spike_time_c - latency_calibration_c",
        "```",
        "",
        f"The calibration uses a reference echo at `{REFERENCE_DISTANCE_M} m`. The estimated latency correction ranges from `{int(latency_samples.min())}` to `{int(latency_samples.max())}` samples.",
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
        "The corollary discharge is an internal ideal sweep. Each channel receives one spike at the expected time that the emitted chirp crosses that channel frequency.",
        "",
        "```text",
        "f(t) = f_start + (f_end - f_start)*t/T",
        "t_cd,c = T * (f_c - f_start)/(f_end - f_start)",
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
        "## Causality Caveat In VCN/VNLL And DNLL Plots",
        "",
        "In the stage raster plot, some VCN/VNLL and DNLL onset spikes appear before the raw cochlea output. That is a modelling issue in the current prototype.",
        "",
        "The cause is the latency-calibration step:",
        "",
        "```text",
        "t_vcn,c = first_spike_time_c - latency_calibration_c",
        "```",
        "",
        "This subtracts a fitted per-channel cochlear latency to align the channel onsets into a sharper constant-latency sweep. That is useful as an offline timestamp correction, but it is not a causal online neuron model: a real VCN/VNLL neuron cannot emit a spike before the cochlear spike arrives. The DNLL inherits the same shifted onset timing, so it can also appear early.",
        "",
        "The causal fix is to stop moving neural spikes earlier in time. Instead, one of the following should be used:",
        "",
        "- delay faster channels so all channels align to the slowest latency;",
        "- keep the physical output spike time, but attach a corrected timestamp used only inside the IC comparison;",
        "- shift the corollary-discharge template or delay bank later to compensate for cochlear latency;",
        "- implement an explicit causal VCN/VNLL circuit where constant latency emerges from delayed inhibition rather than timestamp subtraction.",
        "",
        "So this first prototype should be interpreted as a calibrated timing-map proof of concept, not yet as a fully causal VCN/VNLL/DNLL implementation.",
        "",
        "## Interpretation",
        "",
        "- The model now has the intended high-level biological pathway structure rather than just a standalone coincidence detector.",
        "- The VCN/VNLL stage is deliberately simplified; robust biological onset coding is difficult to tune, and here it is represented by first-spike extraction plus latency calibration.",
        "- The IC stage is still a simplified LIF coincidence model. It uses the closed-form two-spike LIF peak rather than time-stepping every IC neuron for every sample.",
        "- The AC and SC stages give a smooth population readout, which is useful for sub-bin distance estimates.",
        "- The next optimisation step is to replace dense cochlear rasters with coordinate events so the chosen coordinate accumulator can be used downstream.",
        "- Despite the causality caveat, this should be counted as a successful first full-distance-pathway prototype: the full chain from cochlea to SC readout produces a structured distance population and low clean distance error.",
        "",
        "## Generated Files",
        "",
    ]
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
    latency_samples = _calibrate_channel_latency(config)
    rng = np.random.default_rng(RNG_SEED)
    distances = rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M, size=NUM_TEST_SAMPLES)
    predictions = [_predict_one(config, float(distance), latency_samples) for distance in distances]
    example_index = int(np.argmin(np.abs(distances - 3.0)))
    example = predictions[example_index]

    artifacts = {
        "stage_rasters": _plot_stage_rasters(example, config, FIGURE_DIR / "stage_rasters.png"),
        "population_progression": _plot_population_progression(example, config, FIGURE_DIR / "population_progression.png"),
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
            "ac_inhibit_gain": AC_INHIBIT_GAIN,
        },
        "latency_samples": latency_samples.tolist(),
        "metrics": metric_values,
        "example": {
            "true_distance_m": example.distance_m,
            "predicted_distance_m": example.predicted_distance_m,
        },
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(config, latency_samples, predictions, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
