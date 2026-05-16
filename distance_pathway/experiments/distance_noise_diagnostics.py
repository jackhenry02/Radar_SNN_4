from __future__ import annotations

"""Noise diagnostics for the full distance pathway.

This script visualises why the clean distance pathway fails under the harsh
10 dB SNR + jitter diagnostic condition. It compares the two VCN input variants:

1. VCN driven by the rectified cochleagram.
2. VCN driven by the cochlear spike raster.
"""

import json
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

from distance_pathway.experiments.full_distance_pathway_model import (
    NOISE_ROBUSTNESS_JITTER_S,
    NOISE_ROBUSTNESS_SNR_DB,
    RNG_SEED,
    VCN_LIF_THRESHOLD_FRACTION,
    _chirp_channel_times,
    _load_channel_latency,
    _make_config,
    _make_noisy_config,
    _run_cochlea_binaural,
    _simulate_scene,
    _vcn_input_tensor,
    _vcn_vnll_onset_detector,
)
from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.experiments.final_cochlea_model_analysis import _log_spaced_centers


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "distance_noise_diagnostics"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "distance_noise_diagnostics.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

EXAMPLE_DISTANCE_M = 3.0
THRESHOLD_MULTIPLIERS = [1.0, 2.0, 4.0, 8.0, 16.0]
BETA_SWEEP_THRESHOLD_MULTIPLIER = 16.0
BETA_SWEEP_VALUES = [0.0, 0.5, 0.75, 0.88, 0.95, 0.98]
DYNAMIC_TEST_DISTANCES_M = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
COMPARISON_DISTANCES_M = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
DYNAMIC_ECHO_MARGIN_SAMPLES = 32
DYNAMIC_ECHO_TAIL_SAMPLES = 128
DYNAMIC_SCHEDULES = [
    {
        "name": "dyn_A_x32_to_x8_beta0_to_0p75",
        "threshold_start_mult": 32.0,
        "threshold_floor_mult": 8.0,
        "threshold_tau_ms": 10.0,
        "beta_start": 0.0,
        "beta_end": 0.75,
        "beta_tau_ms": 18.0,
    },
    {
        "name": "dyn_B_x32_to_x4_beta0_to_0p75",
        "threshold_start_mult": 32.0,
        "threshold_floor_mult": 4.0,
        "threshold_tau_ms": 12.0,
        "beta_start": 0.0,
        "beta_end": 0.75,
        "beta_tau_ms": 18.0,
    },
    {
        "name": "dyn_C_x24_to_x6_beta0_to_0p8",
        "threshold_start_mult": 24.0,
        "threshold_floor_mult": 6.0,
        "threshold_tau_ms": 14.0,
        "beta_start": 0.0,
        "beta_end": 0.80,
        "beta_tau_ms": 22.0,
    },
    {
        "name": "dyn_D_x16_to_x4_beta0p2_to_0p8",
        "threshold_start_mult": 16.0,
        "threshold_floor_mult": 4.0,
        "threshold_tau_ms": 10.0,
        "beta_start": 0.20,
        "beta_end": 0.80,
        "beta_tau_ms": 18.0,
    },
    {
        "name": "dyn_E_x16_to_x2_beta0p2_to_0p88",
        "threshold_start_mult": 16.0,
        "threshold_floor_mult": 2.0,
        "threshold_tau_ms": 16.0,
        "beta_start": 0.20,
        "beta_end": 0.88,
        "beta_tau_ms": 24.0,
    },
]


@dataclass
class DiagnosticModelOutput:
    """Outputs for one VCN input variant.

    Attributes:
        name: Human-readable model name.
        vcn_input: VCN input source, either `cochleagram` or `spikes`.
        vcn_left: Left-ear VCN onset raster.
        vcn_right: Right-ear VCN onset raster.
        first_vcn_times: First combined VCN onset per channel.
        first_vcn_global: First VCN onset over all channels.
    """

    name: str
    vcn_input: str
    vcn_left: np.ndarray
    vcn_right: np.ndarray
    first_vcn_times: np.ndarray
    first_vcn_global: int


def _first_times(raster: np.ndarray) -> np.ndarray:
    """Return the first event time per channel.

    Args:
        raster: Binary raster `[channels, time]`.

    Returns:
        First event sample per channel, with `-1` for missing channels.
    """
    first = np.full(raster.shape[0], -1, dtype=np.int64)
    for channel in range(raster.shape[0]):
        event_times = np.flatnonzero(raster[channel] > 0.0)
        if event_times.size:
            first[channel] = int(event_times[0])
    return first


def _combined_first_global(raster: np.ndarray) -> int:
    """Return the first event time over all channels."""
    event_times = np.flatnonzero(raster.sum(axis=0) > 0.0)
    return int(event_times[0]) if event_times.size else -1


def _run_model_variant(cochlea, config, vcn_input: str, name: str) -> DiagnosticModelOutput:
    """Run one VCN input variant on the same cochlea output.

    Args:
        cochlea: Binaural cochlea result.
        config: Acoustic configuration.
        vcn_input: `cochleagram` or `spikes`.
        name: Human-readable model name.

    Returns:
        Diagnostic model output.
    """
    vcn_left = _vcn_vnll_onset_detector(_vcn_input_tensor(cochlea, "left", vcn_input), config)
    vcn_right = _vcn_vnll_onset_detector(_vcn_input_tensor(cochlea, "right", vcn_input), config)
    combined = np.maximum(vcn_left, vcn_right)
    return DiagnosticModelOutput(
        name=name,
        vcn_input=vcn_input,
        vcn_left=vcn_left,
        vcn_right=vcn_right,
        first_vcn_times=_first_times(combined),
        first_vcn_global=_combined_first_global(combined),
    )


def _plot_cochleagram(cochlea, config, path: Path) -> str:
    """Plot the noisy cochleagram shared by both VCN variants."""
    combined = torch.maximum(cochlea.left_cochleagram, cochlea.right_cochleagram).detach().cpu().numpy()
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    time_ms = np.arange(combined.shape[1]) / config.sample_rate_hz * 1_000.0

    fig, ax = plt.subplots(figsize=(12, 5.2))
    image = ax.imshow(
        combined,
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], centers_khz.min(), centers_khz.max()],
        cmap="magma",
    )
    ax.set_yscale("log")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel centre frequency (kHz)")
    ax.set_title("Noisy cochleagram input to the VCN stage")
    fig.colorbar(image, ax=ax, label="rectified filter output")
    return save_figure(fig, path)


def _plot_spike_raster(cochlea, config, path: Path) -> str:
    """Plot the noisy cochlear spike raster shared by both variants."""
    combined = torch.maximum(cochlea.left_spikes, cochlea.right_spikes).detach().cpu().numpy()
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for channel, frequency_khz in enumerate(centers_khz):
        event_times = np.flatnonzero(combined[channel] > 0.0) / config.sample_rate_hz * 1_000.0
        if event_times.size:
            ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color="#1d4ed8", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel centre frequency (kHz)")
    ax.set_title("Noisy cochlear spike raster")
    ax.set_xlim(0.0, combined.shape[1] / config.sample_rate_hz * 1_000.0)
    ax.grid(True, axis="x", alpha=0.2)
    return save_figure(fig, path)


def _plot_threshold_spike_rasters(threshold_outputs: list[dict[str, object]], config, path: Path) -> str:
    """Plot cochlear spike rasters for each threshold multiplier.

    Args:
        threshold_outputs: Threshold-sweep outputs from `_run_threshold_sweep`.
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    fig, axes = plt.subplots(len(threshold_outputs), 1, figsize=(12, 2.5 * len(threshold_outputs)), sharex=True)
    if len(threshold_outputs) == 1:
        axes = [axes]
    for ax, output in zip(axes, threshold_outputs):
        raster = output["combined_spikes"]
        for channel, frequency_khz in enumerate(centers_khz):
            event_times = np.flatnonzero(raster[channel] > 0.0) / config.sample_rate_hz * 1_000.0
            if event_times.size:
                ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color="#1d4ed8", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_ylabel("kHz")
        ax.set_title(
            f"threshold x{output['threshold_multiplier']:.0f} "
            f"(threshold={output['spike_threshold']:.3g}, spikes={output['spike_count']})"
        )
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, threshold_outputs[0]["combined_spikes"].shape[1] / config.sample_rate_hz * 1_000.0)
    return save_figure(fig, path)


def _plot_threshold_summary(threshold_outputs: list[dict[str, object]], config, path: Path) -> str:
    """Plot spike count and first-spike timing against threshold multiplier.

    Args:
        threshold_outputs: Threshold-sweep outputs from `_run_threshold_sweep`.
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    multipliers = np.array([float(output["threshold_multiplier"]) for output in threshold_outputs])
    spike_counts = np.array([int(output["spike_count"]) for output in threshold_outputs])
    active_channels = np.array([int(output["active_channels"]) for output in threshold_outputs])
    first_global = np.array([
        np.nan if output["first_global_sample"] < 0 else output["first_global_sample"] / config.sample_rate_hz * 1_000.0
        for output in threshold_outputs
    ])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(multipliers, spike_counts, marker="o", linewidth=2.0)
    axes[0].set_ylabel("total spikes")
    axes[0].set_title("Effect of cochlear spike threshold under noisy input")
    axes[1].plot(multipliers, active_channels, marker="o", color="#16a34a", linewidth=2.0)
    axes[1].set_ylabel("active channels")
    axes[2].plot(multipliers, first_global, marker="o", color="#dc2626", linewidth=2.0)
    axes[2].set_ylabel("first spike (ms)")
    axes[2].set_xlabel("cochlear spike threshold multiplier")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_beta_spike_rasters(beta_outputs: list[dict[str, object]], config, path: Path) -> str:
    """Plot cochlear spike rasters for the fixed-threshold beta sweep.

    Args:
        beta_outputs: Beta-sweep outputs from `_run_beta_sweep`.
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    fig, axes = plt.subplots(len(beta_outputs), 1, figsize=(12, 2.5 * len(beta_outputs)), sharex=True)
    if len(beta_outputs) == 1:
        axes = [axes]
    for ax, output in zip(axes, beta_outputs):
        raster = output["combined_spikes"]
        for channel, frequency_khz in enumerate(centers_khz):
            event_times = np.flatnonzero(raster[channel] > 0.0) / config.sample_rate_hz * 1_000.0
            if event_times.size:
                ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color="#7c3aed", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_ylabel("kHz")
        ax.set_title(
            f"threshold x{BETA_SWEEP_THRESHOLD_MULTIPLIER:.0f}, beta={output['spike_beta']:.2f} "
            f"(spikes={output['spike_count']})"
        )
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, beta_outputs[0]["combined_spikes"].shape[1] / config.sample_rate_hz * 1_000.0)
    return save_figure(fig, path)


def _plot_beta_summary(beta_outputs: list[dict[str, object]], config, path: Path) -> str:
    """Plot spike count and first-spike timing against cochlear LIF beta.

    Args:
        beta_outputs: Beta-sweep outputs from `_run_beta_sweep`.
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    betas = np.array([float(output["spike_beta"]) for output in beta_outputs])
    spike_counts = np.array([int(output["spike_count"]) for output in beta_outputs])
    active_channels = np.array([int(output["active_channels"]) for output in beta_outputs])
    first_global = np.array([
        np.nan if output["first_global_sample"] < 0 else output["first_global_sample"] / config.sample_rate_hz * 1_000.0
        for output in beta_outputs
    ])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(betas, spike_counts, marker="o", linewidth=2.0)
    axes[0].set_ylabel("total spikes")
    axes[0].set_title(f"Effect of cochlear LIF decay at threshold x{BETA_SWEEP_THRESHOLD_MULTIPLIER:.0f}")
    axes[1].plot(betas, active_channels, marker="o", color="#16a34a", linewidth=2.0)
    axes[1].set_ylabel("active channels")
    axes[2].plot(betas, first_global, marker="o", color="#dc2626", linewidth=2.0)
    axes[2].set_ylabel("first spike (ms)")
    axes[2].set_xlabel("cochlear LIF beta")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _dynamic_threshold_beta(config, schedule: dict[str, float], num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create time-varying threshold and beta vectors.

    Args:
        config: Acoustic configuration.
        schedule: Dynamic schedule parameters.
        num_samples: Number of time samples.

    Returns:
        Pair `(threshold_t, beta_t)`, each with shape `[time]`.
    """
    time_samples = np.arange(num_samples, dtype=np.float64)
    threshold_tau = max(schedule["threshold_tau_ms"] * 1e-3 * config.sample_rate_hz, 1.0)
    beta_tau = max(schedule["beta_tau_ms"] * 1e-3 * config.sample_rate_hz, 1.0)
    threshold_mult = schedule["threshold_floor_mult"] + (
        schedule["threshold_start_mult"] - schedule["threshold_floor_mult"]
    ) * np.exp(-time_samples / threshold_tau)
    beta = schedule["beta_start"] + (schedule["beta_end"] - schedule["beta_start"]) * (
        1.0 - np.exp(-time_samples / beta_tau)
    )
    return float(config.spike_threshold) * threshold_mult, np.clip(beta, 0.0, 0.999)


def _dynamic_lif_encode(cochleagram: np.ndarray, config, schedule: dict[str, float]) -> np.ndarray:
    """Encode cochleagram activity with dynamic threshold and beta.

    Args:
        cochleagram: Rectified cochleagram `[channels, time]`.
        config: Acoustic configuration.
        schedule: Dynamic threshold/beta schedule.

    Returns:
        Spike raster `[channels, time]`.
    """
    threshold_t, beta_t = _dynamic_threshold_beta(config, schedule, cochleagram.shape[1])
    membrane = np.zeros(cochleagram.shape[0], dtype=np.float64)
    spikes = np.zeros_like(cochleagram, dtype=np.float32)
    for time_index in range(cochleagram.shape[1]):
        membrane = beta_t[time_index] * membrane + cochleagram[:, time_index]
        fired = membrane >= threshold_t[time_index]
        if np.any(fired):
            spikes[fired, time_index] = 1.0
            membrane[fired] = np.maximum(0.0, membrane[fired] - threshold_t[time_index])
    return spikes


def _echo_and_noise_windows(config, distance_m: float, num_samples: int) -> tuple[slice, slice]:
    """Return equal-duration echo and pre-echo noise windows.

    Args:
        config: Acoustic configuration.
        distance_m: Target distance in metres.
        num_samples: Signal length in samples.

    Returns:
        Pair `(echo_window, noise_window)`.
    """
    round_trip = int(round((2.0 * distance_m / config.speed_of_sound_m_s) * config.sample_rate_hz))
    echo_start = max(0, round_trip - DYNAMIC_ECHO_MARGIN_SAMPLES)
    echo_stop = min(num_samples, round_trip + int(config.chirp_samples) + DYNAMIC_ECHO_TAIL_SAMPLES)
    window_len = max(echo_stop - echo_start, 1)
    noise_stop = max(0, echo_start - DYNAMIC_ECHO_MARGIN_SAMPLES)
    noise_start = max(0, noise_stop - window_len)
    if noise_stop <= noise_start:
        noise_start = 0
        noise_stop = min(window_len, num_samples)
    return slice(echo_start, echo_stop), slice(noise_start, noise_stop)


def _spike_snr_db(spikes: np.ndarray, config, distance_m: float) -> dict[str, float]:
    """Calculate spike-count SNR for one distance.

    Args:
        spikes: Spike raster `[channels, time]`.
        config: Acoustic configuration.
        distance_m: Target distance in metres.

    Returns:
        Spike-count SNR summary.
    """
    echo_window, noise_window = _echo_and_noise_windows(config, distance_m, spikes.shape[1])
    echo_count = float(spikes[:, echo_window].sum())
    noise_count = float(spikes[:, noise_window].sum())
    echo_len = max(echo_window.stop - echo_window.start, 1)
    noise_len = max(noise_window.stop - noise_window.start, 1)
    echo_rate = echo_count / echo_len
    noise_rate = noise_count / noise_len
    snr_db = 10.0 * np.log10((echo_rate + 1.0) / (noise_rate + 1.0))
    return {
        "echo_count": echo_count,
        "noise_count": noise_count,
        "echo_rate": float(echo_rate),
        "noise_rate": float(noise_rate),
        "spike_snr_db": float(snr_db),
    }


def _run_dynamic_schedule(noisy_config, schedule: dict[str, float]) -> dict[str, object]:
    """Evaluate one dynamic threshold/beta schedule over several distances.

    Args:
        noisy_config: Acoustic config for noisy simulation.
        schedule: Dynamic threshold/beta schedule.

    Returns:
        Schedule metrics and the example-distance spike raster.
    """
    rows = []
    example_spikes = None
    for distance_index, distance_m in enumerate(DYNAMIC_TEST_DISTANCES_M):
        torch.manual_seed(RNG_SEED + 50_000 + distance_index)
        receive = _simulate_scene(noisy_config, float(distance_m), add_noise=True)
        cochlea = _run_cochlea_binaural(noisy_config, receive)
        cochleagram = torch.maximum(cochlea.left_cochleagram, cochlea.right_cochleagram).detach().cpu().numpy()
        spikes = _dynamic_lif_encode(cochleagram, noisy_config, schedule)
        snr = _spike_snr_db(spikes, noisy_config, float(distance_m))
        rows.append({"distance_m": float(distance_m), **snr, "spike_count": int(spikes.sum())})
        if np.isclose(distance_m, EXAMPLE_DISTANCE_M):
            example_spikes = spikes
    snr_values = np.array([row["spike_snr_db"] for row in rows], dtype=np.float64)
    mean_snr = float(np.mean(snr_values))
    std_snr = float(np.std(snr_values))
    min_snr = float(np.min(snr_values))
    score = mean_snr - std_snr + 0.5 * min_snr
    return {
        "schedule": schedule,
        "distance_rows": rows,
        "mean_spike_snr_db": mean_snr,
        "std_spike_snr_db": std_snr,
        "min_spike_snr_db": min_snr,
        "score": float(score),
        "example_spikes": example_spikes,
    }


def _run_dynamic_sweep(noisy_config) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate all dynamic threshold/beta schedules.

    Args:
        noisy_config: Acoustic config for noisy simulation.

    Returns:
        Pair `(all_results, chosen_result)`.
    """
    results = [_run_dynamic_schedule(noisy_config, schedule) for schedule in DYNAMIC_SCHEDULES]
    chosen = max(results, key=lambda result: result["score"])
    return results, chosen


def _plot_dynamic_chosen_raster(chosen: dict[str, object], config, path: Path) -> str:
    """Plot the chosen dynamic schedule spike raster at the example distance."""
    spikes = chosen["example_spikes"]
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for channel, frequency_khz in enumerate(centers_khz):
        event_times = np.flatnonzero(spikes[channel] > 0.0) / config.sample_rate_hz * 1_000.0
        if event_times.size:
            ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color="#0f766e", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel centre frequency (kHz)")
    ax.set_title(f"Chosen dynamic threshold/beta spike raster at {EXAMPLE_DISTANCE_M:.1f} m")
    ax.set_xlim(0.0, spikes.shape[1] / config.sample_rate_hz * 1_000.0)
    ax.grid(True, axis="x", alpha=0.2)
    return save_figure(fig, path)


def _plot_dynamic_snr(chosen: dict[str, object], path: Path) -> str:
    """Plot spike-count SNR across distance for the chosen schedule."""
    rows = chosen["distance_rows"]
    distances = np.array([row["distance_m"] for row in rows])
    snr = np.array([row["spike_snr_db"] for row in rows])
    echo_rates = np.array([row["echo_rate"] for row in rows])
    noise_rates = np.array([row["noise_rate"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(distances, snr, marker="o", linewidth=2.0, color="#0f766e")
    axes[0].set_ylabel("spike-count SNR (dB)")
    axes[0].set_title("Chosen dynamic schedule SNR across distances")
    axes[1].plot(distances, echo_rates, marker="o", linewidth=2.0, label="echo-window spike rate")
    axes[1].plot(distances, noise_rates, marker="o", linewidth=2.0, label="pre-echo noise spike rate")
    axes[1].set_xlabel("distance (m)")
    axes[1].set_ylabel("spikes / sample")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_dynamic_schedule(chosen: dict[str, object], config, path: Path) -> str:
    """Plot the chosen threshold and beta over time."""
    threshold_t, beta_t = _dynamic_threshold_beta(config, chosen["schedule"], config.signal_samples)
    time_ms = np.arange(config.signal_samples) / config.sample_rate_hz * 1_000.0
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time_ms, threshold_t, color="#dc2626", linewidth=2.0)
    axes[0].set_ylabel("threshold")
    axes[0].set_title("Chosen dynamic cochlear LIF schedule")
    axes[1].plot(time_ms, beta_t, color="#2563eb", linewidth=2.0)
    axes[1].set_ylabel("beta")
    axes[1].set_xlabel("time (ms)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_distance_comparison(
    noisy_config,
    distance_m: float,
    schedule: dict[str, float],
    path: Path,
) -> str:
    """Plot waveform and spike-raster comparisons for one distance.

    Args:
        noisy_config: Acoustic config for noisy simulation.
        distance_m: Target distance in metres.
        schedule: Chosen dynamic threshold/beta schedule.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    torch.manual_seed(RNG_SEED + 60_000 + int(distance_m * 100))
    receive = _simulate_scene(noisy_config, float(distance_m), add_noise=True)
    static_config = replace(
        noisy_config,
        spike_threshold=float(noisy_config.spike_threshold) * BETA_SWEEP_THRESHOLD_MULTIPLIER,
        spike_beta=0.50,
    )
    static_cochlea = _run_cochlea_binaural(static_config, receive)
    dynamic_cochlea = _run_cochlea_binaural(noisy_config, receive)
    cochleagram = torch.maximum(
        dynamic_cochlea.left_cochleagram,
        dynamic_cochlea.right_cochleagram,
    ).detach().cpu().numpy()
    dynamic_spikes = _dynamic_lif_encode(cochleagram, noisy_config, schedule)
    static_spikes = torch.maximum(static_cochlea.left_spikes, static_cochlea.right_spikes).detach().cpu().numpy()
    centers_khz = _log_spaced_centers(noisy_config).detach().cpu().numpy() / 1_000.0
    time_ms = np.arange(receive.shape[-1]) / noisy_config.sample_rate_hz * 1_000.0
    echo_window, noise_window = _echo_and_noise_windows(noisy_config, float(distance_m), receive.shape[-1])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_ms, receive[0].detach().cpu().numpy(), color="#111827", linewidth=0.8, label="left ear")
    axes[0].plot(time_ms, receive[1].detach().cpu().numpy(), color="#64748b", linewidth=0.8, alpha=0.8, label="right ear")
    axes[0].axvspan(
        echo_window.start / noisy_config.sample_rate_hz * 1_000.0,
        echo_window.stop / noisy_config.sample_rate_hz * 1_000.0,
        color="#16a34a",
        alpha=0.15,
        label="expected echo window",
    )
    axes[0].axvspan(
        noise_window.start / noisy_config.sample_rate_hz * 1_000.0,
        noise_window.stop / noisy_config.sample_rate_hz * 1_000.0,
        color="#dc2626",
        alpha=0.12,
        label="pre-echo noise window",
    )
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f"Noisy received call/echo at {distance_m:.1f} m")
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)

    for ax, raster, title, color in [
        (axes[1], static_spikes, "Static robust cochlea: threshold x16, beta 0.50", "#7c3aed"),
        (axes[2], dynamic_spikes, "Dynamic cochlea: chosen threshold/beta schedule", "#0f766e"),
    ]:
        for channel, frequency_khz in enumerate(centers_khz):
            event_times = np.flatnonzero(raster[channel] > 0.0) / noisy_config.sample_rate_hz * 1_000.0
            if event_times.size:
                ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color=color, linewidth=0.75)
        ax.axvspan(
            echo_window.start / noisy_config.sample_rate_hz * 1_000.0,
            echo_window.stop / noisy_config.sample_rate_hz * 1_000.0,
            color="#16a34a",
            alpha=0.10,
        )
        ax.axvspan(
            noise_window.start / noisy_config.sample_rate_hz * 1_000.0,
            noise_window.stop / noisy_config.sample_rate_hz * 1_000.0,
            color="#dc2626",
            alpha=0.08,
        )
        ax.set_yscale("log")
        ax.set_ylabel("kHz")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, time_ms[-1])
    return save_figure(fig, path)


def _plot_distance_comparisons(noisy_config, schedule: dict[str, float]) -> dict[str, str]:
    """Create comparison plots for the requested distances.

    Args:
        noisy_config: Acoustic config for noisy simulation.
        schedule: Chosen dynamic threshold/beta schedule.

    Returns:
        Mapping of artifact names to saved paths.
    """
    artifacts = {}
    for distance_m in COMPARISON_DISTANCES_M:
        key = f"distance_comparison_{int(distance_m)}m"
        artifacts[key] = _plot_distance_comparison(
            noisy_config,
            float(distance_m),
            schedule,
            FIGURE_DIR / f"{key}.png",
        )
    return artifacts


def _plot_vcn_outputs(outputs: list[DiagnosticModelOutput], config, path: Path) -> str:
    """Plot VCN outputs for both noisy variants."""
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    total_time = outputs[0].vcn_left.shape[1]
    time_ms_end = total_time / config.sample_rate_hz * 1_000.0

    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 7.5), sharex=True)
    if len(outputs) == 1:
        axes = [axes]
    for ax, output in zip(axes, outputs):
        combined = np.maximum(output.vcn_left, output.vcn_right)
        for channel, frequency_khz in enumerate(centers_khz):
            event_times = np.flatnonzero(combined[channel] > 0.0) / config.sample_rate_hz * 1_000.0
            if event_times.size:
                ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color="#dc2626", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("kHz")
        ax.set_title(f"{output.name}: VCN output under noisy input")
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, time_ms_end)
    return save_figure(fig, path)


def _plot_expected_vs_vcn(outputs: list[DiagnosticModelOutput], config, latency_samples: np.ndarray, path: Path) -> str:
    """Plot expected latency-adjusted CD sweep against noisy VCN onset times."""
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    expected = _chirp_channel_times(config) + latency_samples
    round_trip = int(round((2.0 * EXAMPLE_DISTANCE_M / config.speed_of_sound_m_s) * config.sample_rate_hz))
    expected_echo = expected + round_trip
    expected_ms = expected_echo / config.sample_rate_hz * 1_000.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(expected_ms, centers_khz, color="#111827", linewidth=2.0, label="expected CD + delay + latency")
    colors = {"cochleagram": "#dc2626", "spikes": "#2563eb"}
    for output in outputs:
        valid = output.first_vcn_times >= 0
        ax.scatter(
            output.first_vcn_times[valid] / config.sample_rate_hz * 1_000.0,
            centers_khz[valid],
            s=24,
            alpha=0.8,
            color=colors.get(output.vcn_input, "#16a34a"),
            label=output.name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel centre frequency (kHz)")
    ax.set_title("Noisy VCN first onsets vs expected latency-adjusted echo sweep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_figure(fig, path)


def _summarize_outputs(outputs: list[DiagnosticModelOutput], config, latency_samples: np.ndarray) -> list[dict[str, object]]:
    """Summarise onset timing errors for the report."""
    expected = _chirp_channel_times(config) + latency_samples
    round_trip = int(round((2.0 * EXAMPLE_DISTANCE_M / config.speed_of_sound_m_s) * config.sample_rate_hz))
    expected_echo = expected + round_trip
    rows = []
    for output in outputs:
        valid = output.first_vcn_times >= 0
        error = output.first_vcn_times[valid] - expected_echo[valid]
        rows.append(
            {
                "name": output.name,
                "vcn_input": output.vcn_input,
                "first_global_sample": output.first_vcn_global,
                "first_global_ms": output.first_vcn_global / config.sample_rate_hz * 1_000.0 if output.first_vcn_global >= 0 else None,
                "valid_channels": int(valid.sum()),
                "mean_abs_timing_error_samples": float(np.mean(np.abs(error))) if error.size else None,
                "median_timing_error_samples": float(np.median(error)) if error.size else None,
                "min_timing_error_samples": int(error.min()) if error.size else None,
                "max_timing_error_samples": int(error.max()) if error.size else None,
            }
        )
    return rows


def _run_threshold_sweep(noisy_config, receive: torch.Tensor) -> list[dict[str, object]]:
    """Rerun the noisy cochlea with raised spike thresholds.

    Args:
        noisy_config: Acoustic config for the noisy condition.
        receive: Shared noisy received waveform `[ears, time]`.

    Returns:
        Per-threshold combined cochlear spike rasters and summary values.
    """
    rows = []
    for multiplier in THRESHOLD_MULTIPLIERS:
        threshold_config = replace(
            noisy_config,
            spike_threshold=float(noisy_config.spike_threshold) * float(multiplier),
        )
        cochlea = _run_cochlea_binaural(threshold_config, receive)
        combined_spikes = torch.maximum(cochlea.left_spikes, cochlea.right_spikes).detach().cpu().numpy()
        first_times = _first_times(combined_spikes)
        rows.append(
            {
                "threshold_multiplier": float(multiplier),
                "spike_threshold": float(threshold_config.spike_threshold),
                "combined_spikes": combined_spikes,
                "spike_count": int(combined_spikes.sum()),
                "active_channels": int(np.sum(first_times >= 0)),
                "first_global_sample": _combined_first_global(combined_spikes),
            }
        )
    return rows


def _run_beta_sweep(noisy_config, receive: torch.Tensor) -> list[dict[str, object]]:
    """Rerun the noisy cochlea at threshold x16 while varying LIF beta.

    Args:
        noisy_config: Acoustic config for the noisy condition.
        receive: Shared noisy received waveform `[ears, time]`.

    Returns:
        Per-beta combined cochlear spike rasters and summary values.
    """
    rows = []
    for beta in BETA_SWEEP_VALUES:
        beta_config = replace(
            noisy_config,
            spike_threshold=float(noisy_config.spike_threshold) * BETA_SWEEP_THRESHOLD_MULTIPLIER,
            spike_beta=float(beta),
        )
        cochlea = _run_cochlea_binaural(beta_config, receive)
        combined_spikes = torch.maximum(cochlea.left_spikes, cochlea.right_spikes).detach().cpu().numpy()
        first_times = _first_times(combined_spikes)
        rows.append(
            {
                "spike_beta": float(beta),
                "spike_threshold": float(beta_config.spike_threshold),
                "combined_spikes": combined_spikes,
                "spike_count": int(combined_spikes.sum()),
                "active_channels": int(np.sum(first_times >= 0)),
                "first_global_sample": _combined_first_global(combined_spikes),
            }
        )
    return rows


def _write_report(results: dict[str, object]) -> None:
    """Write the noise diagnostic report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Distance Noise Diagnostics",
        "",
        "This report diagnoses why the full distance pathway fails under the harsh noise condition used in the signal-analysis mini model.",
        "",
        "Two variants are compared on the same noisy echo:",
        "",
        "- Cochleagram VCN: the VCN onset LIF reads the rectified cochleagram.",
        "- Spike-raster VCN: the VCN onset LIF reads the cochlear spike raster.",
        "",
        "## Noise Condition",
        "",
        f"- Distance: `{EXAMPLE_DISTANCE_M:.2f} m`",
        f"- Additive white receiver noise: `{NOISE_ROBUSTNESS_SNR_DB:.1f} dB` SNR over the active echo window",
        f"- Propagation jitter: `jitter_std = {NOISE_ROBUSTNESS_JITTER_S:.6g} s`",
        f"- Realised `noise_std`: `{results['noise_std']:.6g}`",
        "",
        "The goal is not to prove final robustness. The goal is to see where the failure enters the pathway.",
        "",
        "## Shared Noisy Cochlea Output",
        "",
        "Both VCN variants receive the same noisy simulated echo and the same cochlea front end.",
        "",
        "![Noisy cochleagram](../outputs/distance_noise_diagnostics/figures/noisy_cochleagram.png)",
        "",
        "![Noisy cochlear spike raster](../outputs/distance_noise_diagnostics/figures/noisy_spike_raster.png)",
        "",
        "## VCN Outputs",
        "",
        "![VCN outputs](../outputs/distance_noise_diagnostics/figures/noisy_vcn_outputs.png)",
        "",
        "![Expected vs VCN](../outputs/distance_noise_diagnostics/figures/expected_vs_vcn.png)",
        "",
        "## Timing Summary",
        "",
        "| Model | VCN input | Active channels | First global onset | Mean abs timing error | Median timing error | Error range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["summary_rows"]:
        first_ms = "n/a" if row["first_global_ms"] is None else f"`{row['first_global_ms']:.3f} ms`"
        mean_abs = "n/a" if row["mean_abs_timing_error_samples"] is None else f"`{row['mean_abs_timing_error_samples']:.1f}` samples"
        median = "n/a" if row["median_timing_error_samples"] is None else f"`{row['median_timing_error_samples']:.1f}` samples"
        error_range = "n/a" if row["min_timing_error_samples"] is None else f"`{row['min_timing_error_samples']} -> {row['max_timing_error_samples']}` samples"
        lines.append(
            f"| {row['name']} | `{row['vcn_input']}` | `{row['valid_channels']}` | {first_ms} | {mean_abs} | {median} | {error_range} |"
        )
    lines.extend(
        [
            "",
            "## Cochlear Threshold Sweep",
            "",
            "The first attempted fix is to raise the cochlear spike threshold while keeping the same noisy waveform. This tests whether the noisy cochlear spike raster can be cleaned before changing the downstream VCN logic.",
            "",
            "![Threshold spike rasters](../outputs/distance_noise_diagnostics/figures/threshold_spike_rasters.png)",
            "",
            "![Threshold summary](../outputs/distance_noise_diagnostics/figures/threshold_summary.png)",
            "",
            "| Threshold multiplier | Spike threshold | Total cochlear spikes | Active channels | First global spike |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["threshold_summary_rows"]:
        first_ms = "n/a" if row["first_global_sample"] < 0 else f"`{row['first_global_sample'] / results['sample_rate_hz'] * 1_000.0:.3f} ms`"
        lines.append(
            f"| `x{row['threshold_multiplier']:.0f}` | `{row['spike_threshold']:.3g}` | `{row['spike_count']}` | `{row['active_channels']}` | {first_ms} |"
        )
    lines.extend(
        [
            "",
            "## Cochlear Decay Sweep At 16x Threshold",
            "",
            "Using the `16x` cochlear spike threshold as the best cleanup attempt so far, this sweep varies the cochlear LIF decay/beta. Lower beta leaks faster and should reduce accumulation from isolated noise; higher beta integrates longer and may increase sensitivity.",
            "",
            "![Beta spike rasters](../outputs/distance_noise_diagnostics/figures/beta_spike_rasters.png)",
            "",
            "![Beta summary](../outputs/distance_noise_diagnostics/figures/beta_summary.png)",
            "",
            "| Cochlear beta | Spike threshold | Total cochlear spikes | Active channels | First global spike |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["beta_summary_rows"]:
        first_ms = "n/a" if row["first_global_sample"] < 0 else f"`{row['first_global_sample'] / results['sample_rate_hz'] * 1_000.0:.3f} ms`"
        lines.append(
            f"| `{row['spike_beta']:.2f}` | `{row['spike_threshold']:.3g}` | `{row['spike_count']}` | `{row['active_channels']}` | {first_ms} |"
        )
    chosen = results["dynamic_chosen"]
    chosen_schedule = chosen["schedule"]
    lines.extend(
        [
            "",
            "## Dynamic Threshold And Beta",
            "",
            "The next test addresses the distance-dependent volume problem. Near echoes are loud and arrive early; far echoes are weak and arrive late. A fixed threshold/beta therefore either over-fires on early noise or misses far echoes.",
            "",
            "The tested dynamic LIF uses:",
            "",
            "```text",
            "threshold(t) = threshold_base * (floor + (start - floor)*exp(-t/tau_threshold))",
            "beta(t) = beta_start + (beta_end - beta_start)*(1 - exp(-t/tau_beta))",
            "v_c[t] = beta(t)*v_c[t-1] + cochleagram_c[t]",
            "spike_c[t] = 1 if v_c[t] >= threshold(t)",
            "```",
            "",
            "This is a heuristic rather than an analytic optimum. Analytically deriving the optimum would require a calibrated model of echo amplitude, target reflectivity, channel noise variance, filter group delay, and acceptable false-alarm probability. Here, a small parameter sweep is used as a practical first step.",
            "",
            "Spike-count SNR is measured as the ratio of spike rate in the expected echo window to spike rate in an equal-length pre-echo noise window:",
            "",
            "```text",
            "SNR_spike = 10 log10((echo_rate + 1) / (noise_rate + 1))",
            "```",
            "",
            "![Dynamic schedule](../outputs/distance_noise_diagnostics/figures/dynamic_schedule.png)",
            "",
            "![Dynamic chosen raster](../outputs/distance_noise_diagnostics/figures/dynamic_chosen_raster.png)",
            "",
            "![Dynamic SNR across distance](../outputs/distance_noise_diagnostics/figures/dynamic_snr_across_distance.png)",
            "",
            "| Schedule | Start threshold | Floor threshold | Threshold tau | Beta start | Beta end | Beta tau | Mean SNR | SNR std | Min SNR | Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["dynamic_schedule_rows"]:
        schedule = row["schedule"]
        lines.append(
            f"| `{schedule['name']}` | "
            f"`x{schedule['threshold_start_mult']:.0f}` | "
            f"`x{schedule['threshold_floor_mult']:.0f}` | "
            f"`{schedule['threshold_tau_ms']:.1f} ms` | "
            f"`{schedule['beta_start']:.2f}` | "
            f"`{schedule['beta_end']:.2f}` | "
            f"`{schedule['beta_tau_ms']:.1f} ms` | "
            f"`{row['mean_spike_snr_db']:.2f} dB` | "
            f"`{row['std_spike_snr_db']:.2f} dB` | "
            f"`{row['min_spike_snr_db']:.2f} dB` | "
            f"`{row['score']:.2f}` |"
        )
    lines.extend(
        [
            "",
            f"Chosen schedule: `{chosen_schedule['name']}` with threshold `x{chosen_schedule['threshold_start_mult']:.0f} -> x{chosen_schedule['threshold_floor_mult']:.0f}` and beta `{chosen_schedule['beta_start']:.2f} -> {chosen_schedule['beta_end']:.2f}`.",
            "",
            "| Distance | Echo spikes | Noise spikes | Spike SNR |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in chosen["distance_rows"]:
        lines.append(
            f"| `{row['distance_m']:.2f} m` | `{row['echo_count']:.0f}` | `{row['noise_count']:.0f}` | `{row['spike_snr_db']:.2f} dB` |"
        )
    lines.extend(
        [
            "",
            "## Per-Distance Call Comparisons",
            "",
            "The following plots compare the same noisy call/echo condition at `1, 2, 3, 4, 5 m`. Each figure shows the noisy binaural waveform, the fixed robust cochlea raster (`threshold x16`, `beta=0.5`), and the chosen dynamic threshold/beta raster.",
            "",
        ]
    )
    for distance_m in COMPARISON_DISTANCES_M:
        key = f"distance_comparison_{int(distance_m)}m"
        lines.extend(
            [
                f"### {distance_m:.0f} m",
                "",
                f"![{distance_m:.0f} m comparison](../outputs/distance_noise_diagnostics/figures/{key}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The cochleagram-driven VCN is very sensitive because it uses a low adaptive threshold on continuous cochleagram activity.",
            "- Under strong white noise, early noise energy can cross that low threshold before the real echo onset.",
            "- The spike-raster VCN is slightly more conservative because it waits for the cochlear spike encoder, but it still fails when noise creates false or shifted cochlear spikes.",
            "- The clean 0.32 cm result is therefore a clean-timing result, not yet a robust-noise result.",
            "- Raising the cochlear spike threshold can reduce spike density, but the important question is whether it removes the early false events without deleting the real echo.",
            "- Reducing cochlear beta tests whether faster leak can stop isolated noisy samples from accumulating into false early spikes.",
            "- Dynamic thresholding/beta is a better match to distance-dependent volume: it suppresses early noise strongly, then gradually becomes more sensitive to later weak echoes.",
            "- The next fix should be a more robust VCN onset rule, such as multi-channel agreement, matched sweep gating, higher/refractory adaptive thresholds, or a pre-onset denoising/gain-control stage.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in results["artifacts"].items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.append(f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`")
    lines.append("")
    lines.append(f"Runtime: `{results['elapsed_seconds']:.2f} s`.")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run distance-pathway noise diagnostics."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)

    torch.manual_seed(RNG_SEED + 20_000)
    np.random.seed(RNG_SEED + 20_000)
    clean_config = _make_config()
    noisy_config = _make_noisy_config(clean_config)
    latency_samples = _load_channel_latency(clean_config)
    receive = _simulate_scene(noisy_config, EXAMPLE_DISTANCE_M, add_noise=True)
    cochlea = _run_cochlea_binaural(noisy_config, receive)
    threshold_outputs = _run_threshold_sweep(noisy_config, receive)
    beta_outputs = _run_beta_sweep(noisy_config, receive)
    dynamic_outputs, dynamic_chosen = _run_dynamic_sweep(noisy_config)
    outputs = [
        _run_model_variant(cochlea, noisy_config, "cochleagram", "Cochleagram VCN"),
        _run_model_variant(cochlea, noisy_config, "spikes", "Spike-raster VCN"),
    ]
    artifacts = {
        "noisy_cochleagram": _plot_cochleagram(cochlea, noisy_config, FIGURE_DIR / "noisy_cochleagram.png"),
        "noisy_spike_raster": _plot_spike_raster(cochlea, noisy_config, FIGURE_DIR / "noisy_spike_raster.png"),
        "noisy_vcn_outputs": _plot_vcn_outputs(outputs, noisy_config, FIGURE_DIR / "noisy_vcn_outputs.png"),
        "expected_vs_vcn": _plot_expected_vs_vcn(outputs, noisy_config, latency_samples, FIGURE_DIR / "expected_vs_vcn.png"),
        "threshold_spike_rasters": _plot_threshold_spike_rasters(
            threshold_outputs,
            noisy_config,
            FIGURE_DIR / "threshold_spike_rasters.png",
        ),
        "threshold_summary": _plot_threshold_summary(
            threshold_outputs,
            noisy_config,
            FIGURE_DIR / "threshold_summary.png",
        ),
        "beta_spike_rasters": _plot_beta_spike_rasters(
            beta_outputs,
            noisy_config,
            FIGURE_DIR / "beta_spike_rasters.png",
        ),
        "beta_summary": _plot_beta_summary(
            beta_outputs,
            noisy_config,
            FIGURE_DIR / "beta_summary.png",
        ),
        "dynamic_chosen_raster": _plot_dynamic_chosen_raster(
            dynamic_chosen,
            noisy_config,
            FIGURE_DIR / "dynamic_chosen_raster.png",
        ),
        "dynamic_snr_across_distance": _plot_dynamic_snr(
            dynamic_chosen,
            FIGURE_DIR / "dynamic_snr_across_distance.png",
        ),
        "dynamic_schedule": _plot_dynamic_schedule(
            dynamic_chosen,
            noisy_config,
            FIGURE_DIR / "dynamic_schedule.png",
        ),
    }
    artifacts.update(_plot_distance_comparisons(noisy_config, dynamic_chosen["schedule"]))
    elapsed_s = time.perf_counter() - start
    results = {
        "experiment": "distance_noise_diagnostics",
        "elapsed_seconds": elapsed_s,
        "distance_m": EXAMPLE_DISTANCE_M,
        "target_snr_db": NOISE_ROBUSTNESS_SNR_DB,
        "jitter_std_s": NOISE_ROBUSTNESS_JITTER_S,
        "noise_std": noisy_config.noise_std,
        "sample_rate_hz": noisy_config.sample_rate_hz,
        "summary_rows": _summarize_outputs(outputs, noisy_config, latency_samples),
        "threshold_summary_rows": [
            {
                "threshold_multiplier": row["threshold_multiplier"],
                "spike_threshold": row["spike_threshold"],
                "spike_count": row["spike_count"],
                "active_channels": row["active_channels"],
                "first_global_sample": row["first_global_sample"],
            }
            for row in threshold_outputs
        ],
        "beta_summary_rows": [
            {
                "spike_beta": row["spike_beta"],
                "spike_threshold": row["spike_threshold"],
                "spike_count": row["spike_count"],
                "active_channels": row["active_channels"],
                "first_global_sample": row["first_global_sample"],
            }
            for row in beta_outputs
        ],
        "dynamic_schedule_rows": [
            {
                "schedule": row["schedule"],
                "mean_spike_snr_db": row["mean_spike_snr_db"],
                "std_spike_snr_db": row["std_spike_snr_db"],
                "min_spike_snr_db": row["min_spike_snr_db"],
                "score": row["score"],
            }
            for row in dynamic_outputs
        ],
        "dynamic_chosen": {
            "schedule": dynamic_chosen["schedule"],
            "mean_spike_snr_db": dynamic_chosen["mean_spike_snr_db"],
            "std_spike_snr_db": dynamic_chosen["std_spike_snr_db"],
            "min_spike_snr_db": dynamic_chosen["min_spike_snr_db"],
            "score": dynamic_chosen["score"],
            "distance_rows": dynamic_chosen["distance_rows"],
        },
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(results)
    return results


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
