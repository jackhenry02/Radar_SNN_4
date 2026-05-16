from __future__ import annotations

"""Cochlea latency experiment for the full distance pathway.

This experiment measures how much latency the final cochlea front end adds per
frequency channel, checks whether that latency is stable across distance, and
tests whether a simple first-spike detector is equivalent to a low-threshold
refractory LIF onset detector.
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

from distance_pathway.experiments.full_distance_pathway_model import (
    MAX_DISTANCE_M,
    MIN_DISTANCE_M,
    NUM_CHANNELS,
    REFERENCE_DISTANCE_M,
    _chirp_channel_times,
    _make_config,
    _run_cochlea_binaural,
    _simulate_scene,
)
from distance_pathway.experiments.distance_noise_robustness_experiments import (
    ROBUST_LATENCY_VECTOR_PATH,
    VCN_MIN_RESPONSIVE_HZ,
    _responsive_channel_mask,
    _run_vcn as _run_robust_vcn,
    _variant_config,
    _variant_definitions,
)
from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.experiments.final_cochlea_model_analysis import _log_spaced_centers


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "cochlea_latency"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "cochlea_latency_experiment.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"
LATENCY_VECTOR_PATH = OUTPUT_DIR / "cochlea_latency_samples.npy"

TEST_DISTANCES_M = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, 12)
REFRACTORY_LIF_BETA = 0.92
REFRACTORY_LIF_THRESHOLD_FRACTION = 0.03
REFRACTORY_PERIOD_S = 0.010
CONSISTENCY_STD_THRESHOLD_SAMPLES = 6.0


@dataclass
class LatencyExperimentResult:
    """Cochlea latency experiment outputs.

    Attributes:
        latency_matrix: Refractory-LIF latency per distance and channel.
        first_spike_latency_matrix: Simple first-spike latency per distance and channel.
        latency_vector: Median refractory-LIF channel latency saved for later use.
        first_spike_matrix: First-spike onset times from cochlear spikes.
        refractory_lif_matrix: First-spike onset times from refractory LIF.
    """

    latency_matrix: np.ndarray
    first_spike_latency_matrix: np.ndarray
    latency_vector: np.ndarray
    first_spike_matrix: np.ndarray
    refractory_lif_matrix: np.ndarray


@dataclass
class RobustLatencyResult:
    """Robust-model latency recalibration outputs.

    Attributes:
        latency_matrix: Latency per distance and channel.
        latency_vector: Median latency per channel.
        onset_matrix: First robust VCN onset per distance and channel.
        responsive_mask: Channels allowed to drive the robust VCN.
    """

    latency_matrix: np.ndarray
    latency_vector: np.ndarray
    onset_matrix: np.ndarray
    responsive_mask: np.ndarray


def _round_trip_delay_samples(distance_m: float) -> int:
    """Return round-trip delay samples for a distance."""
    config = _make_config()
    return int(round((2.0 * distance_m / config.speed_of_sound_m_s) * config.sample_rate_hz))


def _first_spike_times(spikes: torch.Tensor) -> np.ndarray:
    """Extract first spike time from each channel.

    Args:
        spikes: Spike raster `[channels, time]`.

    Returns:
        First spike time per channel. Missing channels are `-1`.
    """
    spike_np = spikes.detach().cpu().numpy() > 0.0
    first = np.full(spike_np.shape[0], -1, dtype=np.int64)
    for channel in range(spike_np.shape[0]):
        event_times = np.flatnonzero(spike_np[channel])
        if event_times.size:
            first[channel] = int(event_times[0])
    return first


def _refractory_lif_onsets(cochleagram: torch.Tensor) -> np.ndarray:
    """Extract first onsets using a low-threshold refractory LIF detector.

    Args:
        cochleagram: Rectified cochleagram `[channels, time]`.

    Returns:
        First refractory-LIF onset time per channel. Missing channels are `-1`.
    """
    config = _make_config()
    activity = cochleagram.detach().cpu().numpy()
    thresholds = REFRACTORY_LIF_THRESHOLD_FRACTION * np.maximum(activity.max(axis=1), 1e-12)
    refractory = int(round(REFRACTORY_PERIOD_S * config.sample_rate_hz))
    first = np.full(activity.shape[0], -1, dtype=np.int64)
    membrane = np.zeros(activity.shape[0], dtype=np.float64)
    blocked_until = np.zeros(activity.shape[0], dtype=np.int64)
    for time_index in range(activity.shape[1]):
        membrane *= REFRACTORY_LIF_BETA
        active = time_index >= blocked_until
        membrane[active] += activity[active, time_index]
        fired = (first < 0) & active & (membrane >= thresholds)
        if np.any(fired):
            first[fired] = time_index
            blocked_until[fired] = time_index + refractory
            membrane[fired] = 0.0
    return first


def _measure_latency_for_distance(distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure latency vectors for one distance.

    Args:
        distance_m: Distance in metres.

    Returns:
        Tuple `(refractory_latency, first_spike_latency, first_spike_times, refractory_lif_times)`.
    """
    config = _make_config()
    receive = _simulate_scene(config, float(distance_m))
    cochlea = _run_cochlea_binaural(config, receive)
    combined_spikes = torch.maximum(cochlea.left_spikes, cochlea.right_spikes)
    combined_cochleagram = torch.maximum(cochlea.left_cochleagram, cochlea.right_cochleagram)
    first_spikes = _first_spike_times(combined_spikes)
    refractory_times = _refractory_lif_onsets(combined_cochleagram)
    expected = _chirp_channel_times(config) + _round_trip_delay_samples(float(distance_m))
    first_spike_latency = first_spikes - expected
    first_spike_latency[first_spikes < 0] = np.iinfo(np.int64).min
    refractory_latency = refractory_times - expected
    refractory_latency[refractory_times < 0] = np.iinfo(np.int64).min
    return refractory_latency, first_spike_latency, first_spikes, refractory_times


def _run_experiment() -> LatencyExperimentResult:
    """Run the latency experiment across all distances."""
    latencies = []
    first_spike_latencies = []
    first_spikes = []
    refractory_times = []
    for distance_m in TEST_DISTANCES_M:
        latency, first_latency, first, refractory = _measure_latency_for_distance(float(distance_m))
        latencies.append(latency)
        first_spike_latencies.append(first_latency)
        first_spikes.append(first)
        refractory_times.append(refractory)

    latency_matrix = np.vstack(latencies)
    first_spike_latency_matrix = np.vstack(first_spike_latencies)
    first_spike_matrix = np.vstack(first_spikes)
    refractory_lif_matrix = np.vstack(refractory_times)
    masked = np.where(latency_matrix == np.iinfo(np.int64).min, np.nan, latency_matrix.astype(float))
    latency_vector = np.rint(np.nanmedian(masked, axis=0)).astype(np.int64)
    return LatencyExperimentResult(
        latency_matrix=latency_matrix,
        first_spike_latency_matrix=first_spike_latency_matrix,
        latency_vector=latency_vector,
        first_spike_matrix=first_spike_matrix,
        refractory_lif_matrix=refractory_lif_matrix,
    )


def _robust_variant():
    """Return the selected robust spike-raster distance-pathway variant."""
    for variant in _variant_definitions():
        if variant.key == "spike_tuned_consensus_facil":
            return variant
    raise RuntimeError("Robust spike-raster variant is not defined")


def _run_robust_latency_experiment() -> RobustLatencyResult:
    """Measure latency for the selected robust spike-raster model.

    Returns:
        Robust latency recalibration result.
    """
    base_config = _make_config()
    variant = _robust_variant()
    config = _variant_config(base_config, variant)
    responsive = _responsive_channel_mask(config)
    cd_times = _chirp_channel_times(config)
    latency_rows = []
    onset_rows = []
    for distance_m in TEST_DISTANCES_M:
        receive = _simulate_scene(config, float(distance_m), add_noise=False)
        cochlea = _run_cochlea_binaural(config, receive)
        vcn = _run_robust_vcn(cochlea, config, variant)
        first = _first_spike_times(torch.from_numpy(vcn))
        expected = cd_times + _round_trip_delay_samples(float(distance_m))
        latency = np.full(NUM_CHANNELS, np.nan, dtype=np.float64)
        valid = (first >= 0) & responsive
        latency[valid] = first[valid] - expected[valid]
        latency_rows.append(latency)
        onset_rows.append(first)

    latency_matrix = np.vstack(latency_rows)
    onset_matrix = np.vstack(onset_rows)
    latency_vector = np.zeros(NUM_CHANNELS, dtype=np.int64)
    for channel in range(NUM_CHANNELS):
        values = latency_matrix[:, channel]
        values = values[np.isfinite(values)]
        latency_vector[channel] = int(np.rint(np.median(values))) if values.size else 0
    return RobustLatencyResult(
        latency_matrix=latency_matrix,
        latency_vector=latency_vector,
        onset_matrix=onset_matrix,
        responsive_mask=responsive,
    )


def _plot_latency_heatmap(result: LatencyExperimentResult, path: Path) -> str:
    """Plot refractory-LIF latency by distance and frequency channel."""
    config = _make_config()
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    latency = np.where(
        result.latency_matrix == np.iinfo(np.int64).min,
        np.nan,
        result.latency_matrix.astype(float),
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(
        latency,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        extent=[centers_khz.min(), centers_khz.max(), TEST_DISTANCES_M.min(), TEST_DISTANCES_M.max()],
    )
    ax.set_xscale("log")
    ax.set_xlabel("channel centre frequency (kHz)")
    ax.set_ylabel("distance (m)")
    ax.set_title("Measured refractory-LIF cochlea latency vs expected CD sweep")
    fig.colorbar(image, ax=ax, label="latency (samples)")
    return save_figure(fig, path)


def _plot_latency_vector(result: LatencyExperimentResult, path: Path) -> str:
    """Plot saved refractory-LIF latency vector and distance consistency."""
    config = _make_config()
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    latency = np.where(
        result.latency_matrix == np.iinfo(np.int64).min,
        np.nan,
        result.latency_matrix.astype(float),
    )
    latency_std = np.nanstd(latency, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(centers_khz, result.latency_vector, marker="o", linewidth=1.5)
    axes[0].set_ylabel("median latency (samples)")
    axes[0].set_title("Saved per-channel refractory-LIF latency vector")
    axes[1].plot(centers_khz, latency_std, marker="o", color="#dc2626", linewidth=1.5)
    axes[1].axhline(CONSISTENCY_STD_THRESHOLD_SAMPLES, color="#111827", linestyle="--")
    axes[1].set_ylabel("std across distances (samples)")
    axes[1].set_xlabel("channel centre frequency (kHz)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_detector_comparison(result: LatencyExperimentResult, path: Path) -> str:
    """Plot first-spike vs refractory-LIF onset comparison."""
    valid = (result.first_spike_matrix >= 0) & (result.refractory_lif_matrix >= 0)
    first = result.first_spike_matrix[valid]
    refractory = result.refractory_lif_matrix[valid]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(first, refractory, s=8, alpha=0.35)
    min_time = min(first.min(), refractory.min())
    max_time = max(first.max(), refractory.max())
    axes[0].plot([min_time, max_time], [min_time, max_time], color="#111827")
    axes[0].set_xlabel("first cochlea spike time (samples)")
    axes[0].set_ylabel("refractory LIF onset time (samples)")
    axes[0].set_title("Detector onset comparison")
    diff = refractory - first
    axes[1].hist(diff, bins=40, color="#2563eb", alpha=0.85)
    axes[1].axvline(0.0, color="#111827")
    axes[1].set_xlabel("refractory LIF - first spike (samples)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Onset timing difference")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_robust_latency_heatmap(result: RobustLatencyResult, path: Path) -> str:
    """Plot robust-model latency by distance and frequency channel."""
    config = _variant_config(_make_config(), _robust_variant())
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(
        result.latency_matrix,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        extent=[centers_khz.min(), centers_khz.max(), TEST_DISTANCES_M.min(), TEST_DISTANCES_M.max()],
    )
    ax.axvspan(centers_khz.min(), VCN_MIN_RESPONSIVE_HZ / 1_000.0, color="#111827", alpha=0.12, label="silenced <4 kHz")
    ax.set_xscale("log")
    ax.set_xlabel("channel centre frequency (kHz)")
    ax.set_ylabel("distance (m)")
    ax.set_title("Robust spike-raster model latency vs expected CD sweep")
    ax.legend(loc="upper left")
    fig.colorbar(image, ax=ax, label="latency (samples)")
    return save_figure(fig, path)


def _plot_robust_latency_vector(result: RobustLatencyResult, path: Path) -> str:
    """Plot robust-model latency vector and distance consistency."""
    config = _variant_config(_make_config(), _robust_variant())
    centers_khz = _log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    latency_std = np.full(NUM_CHANNELS, np.nan, dtype=np.float64)
    for channel in range(NUM_CHANNELS):
        values = result.latency_matrix[:, channel]
        values = values[np.isfinite(values)]
        if values.size:
            latency_std[channel] = float(np.std(values))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(
        centers_khz[result.responsive_mask],
        result.latency_vector[result.responsive_mask],
        marker="o",
        linewidth=1.5,
    )
    axes[0].set_ylabel("median latency (samples)")
    axes[0].set_title("Robust model recalibrated latency vector")
    axes[1].plot(
        centers_khz[result.responsive_mask],
        latency_std[result.responsive_mask],
        marker="o",
        color="#dc2626",
        linewidth=1.5,
    )
    axes[1].set_ylabel("std across distances (samples)")
    axes[1].set_xlabel("channel centre frequency (kHz)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.axvspan(centers_khz.min(), VCN_MIN_RESPONSIVE_HZ / 1_000.0, color="#111827", alpha=0.12)
        ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _summary_stats(result: LatencyExperimentResult) -> dict[str, float]:
    """Calculate latency experiment summary statistics."""
    latency = np.where(
        result.latency_matrix == np.iinfo(np.int64).min,
        np.nan,
        result.latency_matrix.astype(float),
    )
    first_latency = np.where(
        result.first_spike_latency_matrix == np.iinfo(np.int64).min,
        np.nan,
        result.first_spike_latency_matrix.astype(float),
    )
    latency_std = np.nanstd(latency, axis=0)
    first_latency_std = np.nanstd(first_latency, axis=0)
    valid = (result.first_spike_matrix >= 0) & (result.refractory_lif_matrix >= 0)
    detector_diff = result.refractory_lif_matrix[valid] - result.first_spike_matrix[valid]
    return {
        "mean_channel_latency_std_samples": float(np.nanmean(latency_std)),
        "max_channel_latency_std_samples": float(np.nanmax(latency_std)),
        "mean_simple_first_spike_latency_std_samples": float(np.nanmean(first_latency_std)),
        "max_simple_first_spike_latency_std_samples": float(np.nanmax(first_latency_std)),
        "median_detector_difference_samples": float(np.median(detector_diff)),
        "p95_abs_detector_difference_samples": float(np.percentile(np.abs(detector_diff), 95.0)),
        "missing_first_spike_fraction": float(np.mean(result.first_spike_matrix < 0)),
        "missing_refractory_lif_fraction": float(np.mean(result.refractory_lif_matrix < 0)),
        "simple_first_spike_swap_safe": bool(np.percentile(np.abs(detector_diff), 95.0) <= CONSISTENCY_STD_THRESHOLD_SAMPLES),
        "successful_latency_vector": bool(np.nanmax(latency_std) <= CONSISTENCY_STD_THRESHOLD_SAMPLES),
    }


def _robust_summary_stats(result: RobustLatencyResult) -> dict[str, float]:
    """Calculate robust-model latency summary statistics."""
    responsive_latency = result.latency_matrix[:, result.responsive_mask]
    latency_std = np.full(responsive_latency.shape[1], np.nan, dtype=np.float64)
    for channel in range(responsive_latency.shape[1]):
        values = responsive_latency[:, channel]
        values = values[np.isfinite(values)]
        if values.size:
            latency_std[channel] = float(np.std(values))
    calibrated = np.isfinite(responsive_latency).any(axis=0)
    return {
        "responsive_channels": int(np.sum(result.responsive_mask)),
        "silenced_channels": int(NUM_CHANNELS - np.sum(result.responsive_mask)),
        "calibrated_responsive_channels": int(np.sum(calibrated)),
        "missing_responsive_channels": int(np.sum(~calibrated)),
        "latency_min_samples": int(np.min(result.latency_vector[result.responsive_mask])),
        "latency_max_samples": int(np.max(result.latency_vector[result.responsive_mask])),
        "latency_mean_std_samples": float(np.nanmean(latency_std)),
        "latency_max_std_samples": float(np.nanmax(latency_std)),
    }


def _write_report(
    result: LatencyExperimentResult,
    robust_result: RobustLatencyResult,
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the latency experiment report."""
    stats = _summary_stats(result)
    robust_stats = _robust_summary_stats(robust_result)
    config = _make_config()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cochlea Latency Experiment",
        "",
        "This experiment measures the per-channel latency introduced by the final cochlea front end plus a low-threshold refractory-LIF onset detector. It tests whether that latency is stable enough to use as a correction vector in the full distance pathway.",
        "",
        "## Method",
        "",
        "For each test distance:",
        "",
        "1. Simulate a clean binaural echo.",
        "2. Run the final cochlea model: IIR resonator bank, rectification, TorchScript LIF spikes.",
        "3. Combine left/right cochleagram activity with a max operation.",
        "4. Apply a low-threshold refractory-LIF onset detector to get one causal onset per channel.",
        "5. Compare that onset time with the expected ideal corollary-discharge sweep time plus the acoustic round-trip delay.",
        "",
        "```text",
        "latency_c,d = refractory_LIF_onset_c,d - (CD_time_c + round_trip_delay_d)",
        "latency_vector_c = median_d(latency_c,d)",
        "```",
        "",
        "The saved vector is intended to be applied to the corollary-discharge template in the full pathway, rather than subtracting latency from echo spikes.",
        "",
        "## Refractory LIF Onset Detector",
        "",
        "The refractory-LIF detector is used because the raw first cochlea spike is distance/amplitude dependent. The detector uses an adaptive per-channel threshold and a long refractory period so it produces one onset event per sweep:",
        "",
        "```text",
        "v_c[t] = beta*v_c[t-1] + cochleagram_c[t]",
        "onset_c = first t where v_c[t] >= threshold_c",
        "threshold_c = fraction * max_t(cochleagram_c[t])",
        "```",
        "",
        f"The refractory period was `{REFRACTORY_PERIOD_S * 1_000.0:.1f} ms`, threshold fraction `{REFRACTORY_LIF_THRESHOLD_FRACTION}`, and beta `{REFRACTORY_LIF_BETA}`.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz} Hz` |",
        f"| channels | `{config.num_cochlea_channels}` |",
        f"| distances tested | `{TEST_DISTANCES_M[0]:.2f} -> {TEST_DISTANCES_M[-1]:.2f} m` |",
        f"| number of distances | `{len(TEST_DISTANCES_M)}` |",
        f"| consistency threshold | `{CONSISTENCY_STD_THRESHOLD_SAMPLES}` samples max std |",
        "",
        "## Results",
        "",
        "![Latency heatmap](../outputs/cochlea_latency/figures/latency_heatmap.png)",
        "",
        "![Latency vector](../outputs/cochlea_latency/figures/latency_vector.png)",
        "",
        "![Detector comparison](../outputs/cochlea_latency/figures/detector_comparison.png)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean channel latency std | `{stats['mean_channel_latency_std_samples']:.3f}` samples |",
        f"| max channel latency std | `{stats['max_channel_latency_std_samples']:.3f}` samples |",
        f"| mean simple first-spike latency std | `{stats['mean_simple_first_spike_latency_std_samples']:.3f}` samples |",
        f"| max simple first-spike latency std | `{stats['max_simple_first_spike_latency_std_samples']:.3f}` samples |",
        f"| missing first-spike fraction | `{stats['missing_first_spike_fraction']:.4f}` |",
        f"| missing refractory-LIF fraction | `{stats['missing_refractory_lif_fraction']:.4f}` |",
        f"| median refractory-LIF minus first-spike time | `{stats['median_detector_difference_samples']:.3f}` samples |",
        f"| p95 abs refractory-LIF difference | `{stats['p95_abs_detector_difference_samples']:.3f}` samples |",
        f"| simple first-spike swap safe | `{stats['simple_first_spike_swap_safe']}` |",
        f"| latency vector accepted | `{stats['successful_latency_vector']}` |",
        "",
        "## Robust Spike-Raster Model Recalibration",
        "",
        "This section shows the recalibrated latency vector for the current robust noisy-distance model:",
        "",
        "```text",
        "Spike VCN + cochlea tuning + VCN consensus + IC facilitation + <4 kHz VCN silence",
        "```",
        "",
        "This is not the same timing regime as the clean cochleagram-LIF model above. It uses the tuned cochlear spike raster, so first-event timing is coarser and less stable, but much more robust under noise.",
        "",
        "![Robust latency heatmap](../outputs/cochlea_latency/figures/robust_latency_heatmap.png)",
        "",
        "![Robust latency vector](../outputs/cochlea_latency/figures/robust_latency_vector.png)",
        "",
        "| Robust calibration property | Value |",
        "|---|---:|",
        f"| responsive channels | `{robust_stats['responsive_channels']}` |",
        f"| silenced channels below 4 kHz | `{robust_stats['silenced_channels']}` |",
        f"| calibrated responsive channels | `{robust_stats['calibrated_responsive_channels']}` |",
        f"| missing responsive channels | `{robust_stats['missing_responsive_channels']}` |",
        f"| latency range over responsive channels | `{robust_stats['latency_min_samples']} -> {robust_stats['latency_max_samples']}` samples |",
        f"| mean latency std across distances | `{robust_stats['latency_mean_std_samples']:.3f}` samples |",
        f"| max latency std across distances | `{robust_stats['latency_max_std_samples']:.3f}` samples |",
        f"| saved robust vector | `{ROBUST_LATENCY_VECTOR_PATH.relative_to(ROOT)}` |",
        "",
        "## Interpretation",
        "",
        "- The latency vector is accepted if refractory-LIF latency is stable across distance, because it can then be treated as a fixed cochlea/front-end delay per channel.",
        "- The simple first-spike method is only safe to use if its onset timing closely matches the refractory-LIF detector.",
        "- The latency correction should be applied to the corollary-discharge expectation or IC comparison, not by moving echo spikes earlier in time.",
        "- The robust spike-raster vector should be interpreted as a lower-precision but noise-tolerant timing calibration.",
        "",
        "## Saved Files",
        "",
        f"- `latency_vector`: `{LATENCY_VECTOR_PATH.relative_to(ROOT)}`",
        f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.append("")
    lines.append(f"Runtime: `{elapsed_s:.2f} s`.")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the cochlea latency experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    result = _run_experiment()
    robust_result = _run_robust_latency_experiment()
    np.save(LATENCY_VECTOR_PATH, result.latency_vector)
    np.save(ROBUST_LATENCY_VECTOR_PATH, robust_result.latency_vector)
    artifacts = {
        "latency_heatmap": _plot_latency_heatmap(result, FIGURE_DIR / "latency_heatmap.png"),
        "latency_vector": _plot_latency_vector(result, FIGURE_DIR / "latency_vector.png"),
        "detector_comparison": _plot_detector_comparison(result, FIGURE_DIR / "detector_comparison.png"),
        "robust_latency_heatmap": _plot_robust_latency_heatmap(
            robust_result,
            FIGURE_DIR / "robust_latency_heatmap.png",
        ),
        "robust_latency_vector": _plot_robust_latency_vector(
            robust_result,
            FIGURE_DIR / "robust_latency_vector.png",
        ),
    }
    elapsed_s = time.perf_counter() - start
    stats = _summary_stats(result)
    payload = {
        "experiment": "cochlea_latency_experiment",
        "elapsed_seconds": elapsed_s,
        "distances_m": TEST_DISTANCES_M.tolist(),
        "latency_vector_samples": result.latency_vector.tolist(),
        "stats": stats,
        "robust_latency_vector_samples": robust_result.latency_vector.tolist(),
        "robust_stats": _robust_summary_stats(robust_result),
        "artifacts": artifacts,
        "latency_vector_path": str(LATENCY_VECTOR_PATH),
        "robust_latency_vector_path": str(ROBUST_LATENCY_VECTOR_PATH),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(result, robust_result, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
