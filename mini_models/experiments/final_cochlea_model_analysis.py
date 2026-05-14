from __future__ import annotations

"""Final cochlea model analysis for the mini-model project.

This script takes the best cochlea candidate from the earlier comparison:

`lfilter IIR + TorchScript LIF + active-window gating`

and moves it into a focused analysis document. It increases the IIR Q factor
to improve frequency selectivity, explains the filter mathematics/stability,
plots theoretical behaviour, and benchmarks channel-count scaling against the
old FFT/IFFT cochlea.
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.common.signals import moving_notch_signal_config
from models.acoustics import cochlea_filterbank_stages, lif_encode_stages, simulate_echo_batch
from utils.common import GlobalConfig


ROOT = PROJECT_ROOT
OUTPUT_DIR = ROOT / "mini_models" / "outputs" / "final_cochlea_model"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = ROOT / "mini_models" / "reports" / "final_cochlea_model_analysis.md"

FINAL_Q_FACTOR = 12.0
IIR_INPUT_GAIN = 1.0
BASE_CHANNELS = 48
CHANNEL_SWEEP = [10, 24, 48, 100, 200, 500, 1000]
ACTIVE_WINDOW_THRESHOLD_FRACTION = 0.02
ACTIVE_WINDOW_PADDING_S = 0.001


@dataclass
class FinalCochleaOutput:
    """Output from the final optimized cochlea.

    Attributes:
        cochleagram: Rectified IIR channel activity, shape `[channels, time]`.
        spikes: Binary LIF spike raster, shape `[channels, time]`.
        centers_hz: Channel centre frequencies.
        elapsed_s: Median runtime in seconds.
        active_window: Metadata for active-window gating.
        flops_estimate: Estimated dense operation count for active samples.
    """

    cochleagram: torch.Tensor
    spikes: torch.Tensor
    centers_hz: torch.Tensor
    elapsed_s: float
    active_window: dict[str, float | int]
    flops_estimate: float


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a CPU numpy array."""
    return values.detach().cpu().numpy()


def _make_config(num_channels: int = BASE_CHANNELS) -> GlobalConfig:
    """Create the matched-human final cochlea configuration.

    Args:
        num_channels: Number of cochlear channels.

    Returns:
        Configuration for the final cochlea analysis.
    """
    base = moving_notch_signal_config(GlobalConfig())
    return GlobalConfig(
        **{
            **base.__dict__,
            "num_cochlea_channels": num_channels,
            "normalize_spike_envelope": False,
        }
    )


def _simulate_input_waveform(config: GlobalConfig) -> torch.Tensor:
    """Simulate one clean left-ear echo waveform.

    Args:
        config: Acoustic configuration.

    Returns:
        One-dimensional waveform.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([3.0]),
        azimuth_deg=torch.tensor([35.0]),
        elevation_deg=torch.tensor([20.0]),
        binaural=True,
        add_noise=False,
        include_elevation_cues=True,
        transmit_gain=config.transmit_gain,
    )
    return scene.receive[0, 0].detach()


def _log_spaced_centers(config: GlobalConfig) -> torch.Tensor:
    """Return log-spaced centre frequencies for the configured channel count.

    Args:
        config: Acoustic configuration.

    Returns:
        Centre frequencies in Hz.
    """
    return torch.logspace(
        math.log10(config.cochlea_low_hz),
        math.log10(config.cochlea_high_hz),
        steps=config.num_cochlea_channels,
    )


def _iir_coefficients(
    config: GlobalConfig,
    *,
    q_factor: float = FINAL_Q_FACTOR,
    input_gain: float = IIR_INPUT_GAIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build second-order IIR resonator coefficients.

    Args:
        config: Acoustic configuration.
        q_factor: Filter quality factor.
        input_gain: Input gain.

    Returns:
        Tuple `(a_coeffs, b_coeffs, centers_hz, pole_radius)`.
    """
    centers = _log_spaced_centers(config)
    theta = 2.0 * math.pi * centers / config.sample_rate_hz
    bandwidth_hz = centers / max(q_factor, 1e-6)
    pole_radius = torch.exp(-math.pi * bandwidth_hz / config.sample_rate_hz)
    feedback_one = 2.0 * pole_radius * torch.cos(theta)
    feedback_two = -(pole_radius.square())
    feedforward = (1.0 - pole_radius).clamp_min(1e-6) * input_gain
    a_coeffs = torch.stack(
        [
            torch.ones_like(feedback_one),
            -feedback_one,
            -feedback_two,
        ],
        dim=-1,
    )
    b_coeffs = torch.stack(
        [
            feedforward,
            torch.zeros_like(feedforward),
            torch.zeros_like(feedforward),
        ],
        dim=-1,
    )
    return a_coeffs, b_coeffs, centers, pole_radius


@torch.jit.script
def _lif_encode_jit(
    envelope: torch.Tensor,
    beta_old: float,
    threshold: float,
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TorchScript LIF encoder with subtractive reset.

    Args:
        envelope: Rectified channel activity `[channels, time]`.
        beta_old: Historical LIF beta at downsampled envelope rate.
        threshold: Spike threshold.
        downsample: Historical envelope downsample factor.

    Returns:
        Pair `(spikes, membrane_trace)`.
    """
    effective_downsample = downsample
    if effective_downsample < 1:
        effective_downsample = 1
    beta_per_sample = beta_old ** (1.0 / float(effective_downsample))
    membrane = torch.zeros(envelope.size(0), dtype=envelope.dtype, device=envelope.device)
    membranes = torch.zeros_like(envelope)
    spikes = torch.zeros_like(envelope)
    for time_index in range(envelope.size(1)):
        membrane.mul_(beta_per_sample).add_(envelope[:, time_index])
        spike = (membrane >= threshold).to(envelope.dtype)
        spikes[:, time_index] = spike
        membrane.sub_(spike * threshold).clamp_(min=0.0)
        membranes[:, time_index] = membrane
    return spikes, membranes


def _active_window_bounds(signal: torch.Tensor, config: GlobalConfig) -> tuple[int, int]:
    """Find a padded active waveform window.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(start, stop)`, with `stop` exclusive.
    """
    threshold = ACTIVE_WINDOW_THRESHOLD_FRACTION * signal.abs().amax().clamp_min(1e-12)
    active_indices = torch.nonzero(signal.abs() >= threshold, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return 0, signal.numel()
    padding = int(round(ACTIVE_WINDOW_PADDING_S * config.sample_rate_hz))
    start = max(0, int(active_indices[0].item()) - padding)
    stop = min(signal.numel(), int(active_indices[-1].item()) + padding + 1)
    return start, stop


def _estimate_final_iir_flops(config: GlobalConfig, processed_samples: int) -> float:
    """Estimate FLOPs for final gated IIR + LIF cochlea.

    Args:
        config: Acoustic configuration.
        processed_samples: Number of samples processed after gating.

    Returns:
        Estimated operation count.
    """
    channels = config.num_cochlea_channels
    iir = channels * processed_samples * 6.0
    rectification = channels * processed_samples
    lif = channels * processed_samples * 5.0
    return iir + rectification + lif


def _estimate_fft_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate FLOPs for the old FFT/IFFT cochlea.

    Args:
        config: Acoustic configuration.
        signal_samples: Number of waveform samples.

    Returns:
        Estimated operation count.
    """
    channels = config.num_cochlea_channels
    frequency_bins = signal_samples // 2 + 1
    envelope_steps = math.ceil(signal_samples / config.envelope_downsample)
    lowpass_kernel = max(3, int(round(config.sample_rate_hz / config.envelope_lowpass_hz)))
    if lowpass_kernel % 2 == 0:
        lowpass_kernel += 1
    fft_cost = 5.0 * signal_samples * math.log2(signal_samples)
    ifft_cost = channels * fft_cost
    complex_filter_mult = channels * frequency_bins * 6.0
    rectification = channels * signal_samples
    smoothing = channels * signal_samples * lowpass_kernel * 2.0
    downsample = channels * envelope_steps * config.envelope_downsample
    lif = channels * envelope_steps * 5.0
    return fft_cost + ifft_cost + complex_filter_mult + rectification + smoothing + downsample + lif


def _run_final_cochlea_once(signal: torch.Tensor, config: GlobalConfig, q_factor: float) -> FinalCochleaOutput:
    """Run the final optimized cochlea once.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.
        q_factor: IIR Q factor.

    Returns:
        Final cochlea output.
    """
    start_sample, stop_sample = _active_window_bounds(signal, config)
    cropped = signal[start_sample:stop_sample]
    a_coeffs, b_coeffs, centers, _ = _iir_coefficients(config, q_factor=q_factor)
    repeated = cropped.to(torch.float32).unsqueeze(0).repeat(config.num_cochlea_channels, 1)
    filtered = AF.lfilter(repeated, a_coeffs, b_coeffs, clamp=False, batching=True).to(signal.dtype)
    cochleagram_crop = F.relu(filtered)
    spikes_crop, _ = _lif_encode_jit(
        cochleagram_crop,
        float(config.spike_beta),
        float(config.spike_threshold),
        int(config.envelope_downsample),
    )
    full_cochleagram = torch.zeros(config.num_cochlea_channels, signal.numel(), dtype=signal.dtype)
    full_spikes = torch.zeros_like(full_cochleagram)
    full_cochleagram[:, start_sample:stop_sample] = cochleagram_crop
    full_spikes[:, start_sample:stop_sample] = spikes_crop
    processed = stop_sample - start_sample
    active_window = {
        "start_sample": start_sample,
        "stop_sample": stop_sample,
        "samples_processed": processed,
        "total_samples": signal.numel(),
        "processed_fraction": processed / max(signal.numel(), 1),
        "padding_s": ACTIVE_WINDOW_PADDING_S,
        "threshold_fraction": ACTIVE_WINDOW_THRESHOLD_FRACTION,
    }
    return FinalCochleaOutput(
        cochleagram=full_cochleagram,
        spikes=full_spikes,
        centers_hz=centers,
        elapsed_s=0.0,
        active_window=active_window,
        flops_estimate=_estimate_final_iir_flops(config, processed),
    )


def _median_runtime_s(function: Callable[[], object], repeats: int) -> float:
    """Benchmark a callable with a median runtime.

    Args:
        function: Zero-argument callable.
        repeats: Number of timed repeats.

    Returns:
        Median elapsed time in seconds.
    """
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _run_final_cochlea_benchmarked(signal: torch.Tensor, config: GlobalConfig, q_factor: float) -> FinalCochleaOutput:
    """Run and benchmark the final cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.
        q_factor: IIR Q factor.

    Returns:
        Final cochlea output with runtime populated.
    """
    elapsed_s = _median_runtime_s(lambda: _run_final_cochlea_once(signal, config, q_factor), repeats=8)
    output = _run_final_cochlea_once(signal, config, q_factor)
    output.elapsed_s = elapsed_s
    return output


def _run_old_fft_once(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the old FFT/IFFT cochlea once.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes)`.
    """
    stages = cochlea_filterbank_stages(
        signal=signal.unsqueeze(0),
        sample_rate_hz=config.sample_rate_hz,
        num_channels=config.num_cochlea_channels,
        low_hz=config.cochlea_low_hz,
        high_hz=config.cochlea_high_hz,
        spacing_mode=config.cochlea_spacing_mode,
        filter_bandwidth_sigma=config.filter_bandwidth_sigma,
        envelope_lowpass_hz=config.envelope_lowpass_hz,
        downsample=config.envelope_downsample,
    )
    lif = lif_encode_stages(
        stages["cochleagram"],
        threshold=config.spike_threshold,
        beta=config.spike_beta,
        normalize_envelope=False,
    )
    return stages["cochleagram"][0], lif["spikes"][0]


def _frequency_response(
    config: GlobalConfig,
    q_factor: float,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute theoretical IIR frequency response.

    Args:
        config: Acoustic configuration.
        q_factor: IIR Q factor.
        frequency_hz: Frequency axis in Hz.

    Returns:
        Tuple `(centers_hz, pole_radius, response)` where response has shape
        `[channels, frequencies]`.
    """
    _, _, centers, radius = _iir_coefficients(config, q_factor=q_factor)
    centers_np = _to_numpy(centers)
    radius_np = _to_numpy(radius)
    omega = 2.0 * np.pi * frequency_hz[None, :] / config.sample_rate_hz
    theta = 2.0 * np.pi * centers_np[:, None] / config.sample_rate_hz
    r = radius_np[:, None]
    b0 = np.maximum(1.0 - r, 1e-12)
    denominator = 1.0 - 2.0 * r * np.cos(theta) * np.exp(-1j * omega) + (r**2) * np.exp(-2j * omega)
    response = np.abs(b0 / denominator)
    response = response / np.maximum(response.max(axis=1, keepdims=True), 1e-12)
    return centers_np, radius_np, response


def _bandwidth_at_half_power(frequency_hz: np.ndarray, response: np.ndarray) -> float:
    """Estimate normalized-response half-power bandwidth.

    Args:
        frequency_hz: Frequency axis in Hz.
        response: One response curve normalized to peak one.

    Returns:
        Approximate -3 dB bandwidth in Hz.
    """
    mask = response >= (1.0 / math.sqrt(2.0))
    if not np.any(mask):
        return 0.0
    active = frequency_hz[mask]
    return float(active[-1] - active[0])


def _plot_q_selectivity(config: GlobalConfig, path: Path) -> str:
    """Plot how Q changes theoretical frequency selectivity.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    frequency_hz = np.linspace(config.cochlea_low_hz, config.cochlea_high_hz, 2200)
    q_values = [5.0, 8.0, 12.0, 16.0]
    target_frequency = 10_000.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bandwidths = []
    for q in q_values:
        centers, _, response = _frequency_response(config, q, frequency_hz)
        channel = int(np.argmin(np.abs(centers - target_frequency)))
        axes[0].plot(frequency_hz / 1_000.0, response[channel], linewidth=1.8, label=f"Q={q:g}")
        bandwidths.append(_bandwidth_at_half_power(frequency_hz, response[channel]))
    axes[0].set_title("Q tuning: response near 10 kHz")
    axes[0].set_xlabel("frequency (kHz)")
    axes[0].set_ylabel("normalized magnitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(q_values, np.array(bandwidths) / 1_000.0, marker="o", color="#0f172a")
    axes[1].set_title("Higher Q narrows bandwidth")
    axes[1].set_xlabel("Q factor")
    axes[1].set_ylabel("-3 dB bandwidth (kHz)")
    axes[1].grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_final_frequency_response(config: GlobalConfig, path: Path) -> str:
    """Plot final Q frequency response heatmap and example channels.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    frequency_hz = np.linspace(config.cochlea_low_hz, config.cochlea_high_hz, 2200)
    centers, _, response = _frequency_response(config, FINAL_Q_FACTOR, frequency_hz)
    example_indices = [0, len(centers) // 2, len(centers) - 1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for index in example_indices:
        axes[0].plot(frequency_hz / 1_000.0, response[index], linewidth=1.8, label=f"{centers[index] / 1_000.0:.1f} kHz")
    axes[0].set_title(f"Final IIR response examples, Q={FINAL_Q_FACTOR:g}")
    axes[0].set_xlabel("frequency (kHz)")
    axes[0].set_ylabel("normalized magnitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)
    image = axes[1].imshow(
        response,
        aspect="auto",
        origin="lower",
        extent=[frequency_hz[0] / 1_000.0, frequency_hz[-1] / 1_000.0, 0, len(centers) - 1],
        cmap="viridis",
    )
    axes[1].set_title("Final IIR filterbank response heatmap")
    axes[1].set_xlabel("frequency (kHz)")
    axes[1].set_ylabel("channel")
    colorbar = fig.colorbar(image, ax=axes[1])
    colorbar.set_label("normalized magnitude")
    return save_figure(fig, path)


def _plot_stability(config: GlobalConfig, path: Path) -> str:
    """Plot pole stability diagnostics for the final IIR bank.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    _, _, centers, radius = _iir_coefficients(config, q_factor=FINAL_Q_FACTOR)
    centers_np = _to_numpy(centers)
    radius_np = _to_numpy(radius)
    theta = 2.0 * np.pi * centers_np / config.sample_rate_hz
    poles_pos = radius_np * np.exp(1j * theta)
    poles_neg = radius_np * np.exp(-1j * theta)
    unit_circle = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 600))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(unit_circle.real, unit_circle.imag, color="#94a3b8", linewidth=1.0, label="unit circle")
    axes[0].scatter(poles_pos.real, poles_pos.imag, s=24, color="#2563eb", label="+frequency pole")
    axes[0].scatter(poles_neg.real, poles_neg.imag, s=24, color="#dc2626", label="-frequency pole")
    axes[0].set_aspect("equal", "box")
    axes[0].set_title("IIR poles stay inside unit circle")
    axes[0].set_xlabel("real")
    axes[0].set_ylabel("imaginary")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(centers_np / 1_000.0, radius_np, marker="o", linewidth=1.5, color="#0f172a")
    axes[1].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.0, label="stability boundary")
    axes[1].set_title("Pole radius by channel")
    axes[1].set_xlabel("centre frequency (kHz)")
    axes[1].set_ylabel("pole radius r")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_final_output(output: FinalCochleaOutput, config: GlobalConfig, path: Path) -> str:
    """Plot final model cochleagram and spike raster.

    Args:
        output: Final cochlea output.
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    cochleagram = _to_numpy(output.cochleagram)
    spikes = _to_numpy(output.spikes > 0.0)
    time_ms = np.arange(cochleagram.shape[-1]) / config.sample_rate_hz * 1_000.0
    centers_khz = _to_numpy(output.centers_hz) / 1_000.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    image = axes[0].imshow(
        np.log1p(cochleagram),
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], centers_khz[0], centers_khz[-1]],
        cmap="magma",
    )
    axes[0].set_title("Final cochlea: gated IIR + TorchScript LIF cochleagram")
    axes[0].set_ylabel("centre frequency (kHz)")
    fig.colorbar(image, ax=axes[0], label="log activity")
    for channel in range(spikes.shape[0]):
        spike_times = time_ms[np.flatnonzero(spikes[channel])]
        if spike_times.size:
            axes[1].vlines(spike_times, channel + 0.15, channel + 0.85, color="#111827", linewidth=0.55)
    axes[1].set_title("Final cochlea: spike raster")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("channel")
    axes[1].set_ylim(0, spikes.shape[0])
    axes[1].set_xlim(0.0, min(time_ms[-1], 24.0))
    return save_figure(fig, path)


def _benchmark_scaling(base_signal: torch.Tensor) -> list[dict[str, float]]:
    """Benchmark final IIR and old FFT cochleae over channel counts.

    Args:
        base_signal: Input waveform.

    Returns:
        List of per-channel-count benchmark dictionaries.
    """
    results = []
    for channels in CHANNEL_SWEEP:
        config = _make_config(channels)
        repeats = 5 if channels <= 200 else 3
        final_runtime = _median_runtime_s(
            lambda config=config: _run_final_cochlea_once(base_signal, config, FINAL_Q_FACTOR),
            repeats=repeats,
        )
        fft_runtime = _median_runtime_s(
            lambda config=config: _run_old_fft_once(base_signal, config),
            repeats=repeats,
        )
        start, stop = _active_window_bounds(base_signal, config)
        processed_samples = stop - start
        results.append(
            {
                "channels": float(channels),
                "final_iir_time_ms": final_runtime * 1_000.0,
                "fft_time_ms": fft_runtime * 1_000.0,
                "final_iir_flops": _estimate_final_iir_flops(config, processed_samples),
                "fft_flops": _estimate_fft_flops(config, base_signal.numel()),
                "processed_samples": float(processed_samples),
                "processed_fraction": processed_samples / max(base_signal.numel(), 1),
            }
        )
    return results


def _plot_scaling(scaling: list[dict[str, float]], runtime_path: Path, flops_path: Path) -> tuple[str, str]:
    """Plot runtime and FLOP scaling.

    Args:
        scaling: Scaling benchmark results.
        runtime_path: Runtime figure path.
        flops_path: FLOP figure path.

    Returns:
        Pair of saved figure paths.
    """
    channels = np.array([item["channels"] for item in scaling])
    final_time = np.array([item["final_iir_time_ms"] for item in scaling])
    fft_time = np.array([item["fft_time_ms"] for item in scaling])
    final_flops = np.array([item["final_iir_flops"] for item in scaling])
    fft_flops = np.array([item["fft_flops"] for item in scaling])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(channels, final_time, marker="o", linewidth=2.0, label="Final gated IIR + TorchScript LIF")
    ax.plot(channels, fft_time, marker="s", linewidth=2.0, label="Old FFT/IFFT cochlea")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Runtime scaling with channel count")
    ax.set_xlabel("number of channels")
    ax.set_ylabel("runtime per waveform (ms)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left")
    runtime_figure = save_figure(fig, runtime_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(channels, final_flops, marker="o", linewidth=2.0, label="Final gated IIR + TorchScript LIF")
    ax.plot(channels, fft_flops, marker="s", linewidth=2.0, label="Old FFT/IFFT cochlea")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Estimated FLOP scaling with channel count")
    ax.set_xlabel("number of channels")
    ax.set_ylabel("estimated FLOPs per waveform")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left")
    flops_figure = save_figure(fig, flops_path)
    return runtime_figure, flops_figure


def _write_report(
    config: GlobalConfig,
    final_output: FinalCochleaOutput,
    artifacts: dict[str, str],
    scaling: list[dict[str, float]],
    elapsed_s: float,
) -> None:
    """Write the final cochlea model report.

    Args:
        config: Base configuration.
        final_output: Final cochlea output.
        artifacts: Figure artifacts.
        scaling: Channel-scaling benchmark results.
        elapsed_s: Total script runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    spike_count = int((final_output.spikes > 0.0).sum().item())
    pole_radius_min = float(_iir_coefficients(config, q_factor=FINAL_Q_FACTOR)[3].min().item())
    pole_radius_max = float(_iir_coefficients(config, q_factor=FINAL_Q_FACTOR)[3].max().item())
    lines = [
        "# Final Cochlea Model And Analysis",
        "",
        "This document consolidates the selected cochlea front end for the next model design. It uses the optimized IIR resonator filterbank, TorchScript LIF spike encoding, and active-window gating.",
        "",
        "## Final Model",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[received waveform] --> B[active window detector]",
        "    B --> C[crop active echo with padding]",
        "    C --> D[torchaudio lfilter IIR resonator bank]",
        "    D --> E[half-wave rectification]",
        "    E --> F[TorchScript LIF with subtractive reset]",
        "    F --> G[full-length spike raster]",
        "```",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz} Hz` |",
        f"| chirp | `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` |",
        f"| cochlea band | `{config.cochlea_low_hz:.0f} -> {config.cochlea_high_hz:.0f} Hz` |",
        f"| channels for final example | `{config.num_cochlea_channels}` |",
        f"| final IIR Q factor | `{FINAL_Q_FACTOR}` |",
        f"| spike threshold | `{config.spike_threshold}` |",
        f"| spike beta | `{config.spike_beta}` |",
        f"| active-window threshold | `{ACTIVE_WINDOW_THRESHOLD_FRACTION} * max(abs(signal))` |",
        f"| active-window padding | `{ACTIVE_WINDOW_PADDING_S * 1_000.0:.1f} ms` |",
        "",
        "The Q factor was increased from the earlier exploratory value of `5` to `12` to make the resonator channels more frequency selective. This improves separation between neighbouring channels, at the cost of longer ringing.",
        "",
        "## Final Output",
        "",
        f"- Runtime for the `48`-channel final example: `{final_output.elapsed_s * 1_000.0:.3f} ms`",
        f"- Estimated FLOPs for active window: `{final_output.flops_estimate:,.0f}`",
        f"- Output spike count: `{spike_count}`",
        f"- Active samples processed: `{final_output.active_window['samples_processed']}` / `{final_output.active_window['total_samples']}`",
        f"- Active fraction: `{final_output.active_window['processed_fraction']:.3f}`",
        "",
        "![Final cochlea output](../outputs/final_cochlea_model/figures/final_cochlea_output.png)",
        "",
        "## IIR Filter Mathematics",
        "",
        "Each channel is implemented as a second-order resonator. For centre frequency `f_c`, sampling rate `f_s`, and quality factor `Q`:",
        "",
        "```text",
        "bandwidth_c = f_c / Q",
        "theta_c = 2*pi*f_c / f_s",
        "r_c = exp(-pi*bandwidth_c / f_s)",
        "```",
        "",
        "The time-domain difference equation is:",
        "",
        "```text",
        "y_c[n] = b0_c*x[n] + 2*r_c*cos(theta_c)*y_c[n-1] - r_c^2*y_c[n-2]",
        "b0_c = 1 - r_c",
        "```",
        "",
        "The transfer function is:",
        "",
        "```text",
        "H_c(z) = b0_c / (1 - 2*r_c*cos(theta_c)*z^-1 + r_c^2*z^-2)",
        "```",
        "",
        "The poles are:",
        "",
        "```text",
        "z = r_c * exp(+/-j*theta_c)",
        "```",
        "",
        f"For the final model, pole radii range from `{pole_radius_min:.4f}` to `{pole_radius_max:.4f}`. Since all pole radii are less than `1`, the IIR filters are stable.",
        "",
        "![Q selectivity](../outputs/final_cochlea_model/figures/q_selectivity.png)",
        "",
        "![Final frequency response](../outputs/final_cochlea_model/figures/final_frequency_response.png)",
        "",
        "![IIR stability](../outputs/final_cochlea_model/figures/iir_stability.png)",
        "",
        "## Channel Scaling",
        "",
        "The plot below compares runtime as the channel count increases from `10` to `1000`. The final IIR model uses active-window gating, so its operation count depends on active samples rather than full waveform length.",
        "",
        "![Runtime scaling](../outputs/final_cochlea_model/figures/channel_scaling_runtime.png)",
        "",
        "![FLOP scaling](../outputs/final_cochlea_model/figures/channel_scaling_flops.png)",
        "",
        "| Channels | Final IIR time (ms) | FFT time (ms) | Final IIR FLOPs | FFT FLOPs |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in scaling:
        lines.append(
            f"| {int(item['channels'])} | {item['final_iir_time_ms']:.3f} | {item['fft_time_ms']:.3f} | "
            f"{item['final_iir_flops']:,.0f} | {item['fft_flops']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Increasing Q from `5` to `12` narrows each IIR channel and improves theoretical frequency selectivity.",
            "- Stability is guaranteed by construction as long as `Q > 0` and `f_c > 0`, because `r = exp(-pi*f_c/(Q*f_s))` lies inside the unit circle.",
            "- The final model is still not truly sparse inside the active window. It is a gated dense computation: silence is skipped, but the echo window is processed by all channels.",
            "- The scaling curves are the key test for whether this front end remains practical as channel count increases.",
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
    """Run the final cochlea analysis.

    Returns:
        JSON-serializable results dictionary.
    """
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPORT_PATH.parent)
    torch.manual_seed(7)

    config = _make_config(BASE_CHANNELS)
    signal = _simulate_input_waveform(config)
    final_output = _run_final_cochlea_benchmarked(signal, config, FINAL_Q_FACTOR)

    artifacts = {
        "final_cochlea_output": _plot_final_output(final_output, config, FIGURE_DIR / "final_cochlea_output.png"),
        "q_selectivity": _plot_q_selectivity(config, FIGURE_DIR / "q_selectivity.png"),
        "final_frequency_response": _plot_final_frequency_response(config, FIGURE_DIR / "final_frequency_response.png"),
        "iir_stability": _plot_stability(config, FIGURE_DIR / "iir_stability.png"),
    }
    scaling = _benchmark_scaling(signal)
    runtime_plot, flops_plot = _plot_scaling(
        scaling,
        FIGURE_DIR / "channel_scaling_runtime.png",
        FIGURE_DIR / "channel_scaling_flops.png",
    )
    artifacts["channel_scaling_runtime"] = runtime_plot
    artifacts["channel_scaling_flops"] = flops_plot

    elapsed_s = time.perf_counter() - start
    _, _, _, radius = _iir_coefficients(config, q_factor=FINAL_Q_FACTOR)
    payload: dict[str, object] = {
        "experiment": "final_cochlea_model_analysis",
        "elapsed_seconds": elapsed_s,
        "final_q_factor": FINAL_Q_FACTOR,
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "num_cochlea_channels": config.num_cochlea_channels,
            "cochlea_low_hz": config.cochlea_low_hz,
            "cochlea_high_hz": config.cochlea_high_hz,
            "spike_threshold": config.spike_threshold,
            "spike_beta": config.spike_beta,
            "normalize_spike_envelope": config.normalize_spike_envelope,
        },
        "final_output": {
            "runtime_ms": final_output.elapsed_s * 1_000.0,
            "flops_estimate": final_output.flops_estimate,
            "spike_count": int((final_output.spikes > 0.0).sum().item()),
            "spike_density": float((final_output.spikes > 0.0).float().mean().item()),
            "active_window": final_output.active_window,
            "pole_radius_min": float(radius.min().item()),
            "pole_radius_max": float(radius.max().item()),
        },
        "scaling": scaling,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(config, final_output, artifacts, scaling, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
