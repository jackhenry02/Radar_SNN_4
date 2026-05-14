from __future__ import annotations

"""Compare candidate cochlea front ends for the mini-model project.

This experiment is intentionally smaller than `outputs/cochlea_explained.md`.
It compares four ways to turn one acoustic waveform into channel-wise spike
rasters:

1. the original FFT/IFFT + envelope + LIF cochlea;
2. a time-domain Conv1D filterbank + LIF cochlea;
3. a time-domain Conv1D filterbank + level-crossing encoder;
4. a direct resonate-and-fire neuron bank.

The code favours clarity over maximum performance because this is a design
experiment. Later mini models can replace individual pieces with optimized
implementations once the mechanisms are accepted.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.common.signals import moving_notch_signal_config
from models.acoustics import cochlea_filterbank_stages, lif_encode_stages, simulate_echo_batch
from utils.common import GlobalConfig


ROOT = PROJECT_ROOT
OUTPUT_DIR = ROOT / "mini_models" / "outputs" / "cochlea_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = ROOT / "mini_models" / "reports" / "cochlea_analysis.md"

NUM_CHANNELS = 48
CONV_KERNEL_SIZE = 129
BENCHMARK_REPEATS = 8
RF_Q_FACTOR = 7.0
RF_INPUT_GAIN = 0.25
RF_THRESHOLD = 1.0


@dataclass
class CochleaResult:
    """Output from one cochlea front-end model.

    Attributes:
        name: Stable model identifier.
        title: Human-readable model name.
        cochleagram: Continuous channel activity, shape `[channels, time]`.
        spikes: Binary spike/event raster, shape `[channels, time]`.
        time_rate_hz: Time base of `cochleagram` and `spikes`.
        center_frequencies_hz: Frequency represented by each channel.
        elapsed_s: Median benchmark runtime for one waveform.
        flops_estimate: Rough dense floating-point operation estimate.
        sops_estimate: Rough spike/event operation estimate.
        notes: Short implementation note.
    """

    name: str
    title: str
    cochleagram: torch.Tensor
    spikes: torch.Tensor
    time_rate_hz: float
    center_frequencies_hz: torch.Tensor
    elapsed_s: float
    flops_estimate: float
    sops_estimate: float
    notes: str


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a CPU numpy array."""
    return values.detach().cpu().numpy()


def _make_config() -> GlobalConfig:
    """Return the shared matched-human cochlea configuration.

    Returns:
        Global configuration used for all four cochlea models.
    """
    base = moving_notch_signal_config(GlobalConfig())
    return GlobalConfig(
        **{
            **base.__dict__,
            "num_cochlea_channels": NUM_CHANNELS,
            "normalize_spike_envelope": False,
        }
    )


def _simulate_input_waveform(config: GlobalConfig) -> torch.Tensor:
    """Simulate one clean left-ear echo waveform.

    Args:
        config: Acoustic configuration.

    Returns:
        One-dimensional left-ear waveform.
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


def _log_spaced_centers(config: GlobalConfig, num_channels: int = NUM_CHANNELS) -> torch.Tensor:
    """Create log-spaced cochlear centre frequencies.

    Args:
        config: Acoustic configuration.
        num_channels: Number of frequency channels.

    Returns:
        Tensor of centre frequencies in Hz.
    """
    return torch.logspace(
        math.log10(config.cochlea_low_hz),
        math.log10(config.cochlea_high_hz),
        steps=num_channels,
    )


def _make_gabor_filterbank(
    config: GlobalConfig,
    *,
    num_channels: int = NUM_CHANNELS,
    kernel_size: int = CONV_KERNEL_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a time-domain Gabor-like cochlear filterbank.

    This is not claimed to be a full biological gammatone model. It is a
    compact time-domain band-pass bank that is useful for comparing a Conv1D
    front end against the old FFT/IFFT implementation.

    Args:
        config: Acoustic configuration.
        num_channels: Number of filters.
        kernel_size: Number of samples in each FIR kernel.

    Returns:
        Pair `(kernels, centers)` where kernels has shape
        `[channels, 1, kernel_size]`.
    """
    centers = _log_spaced_centers(config, num_channels)
    time_axis = (torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0) / config.sample_rate_hz
    # A shorter Gaussian window at high frequency keeps each channel reasonably
    # localized in time while still providing frequency selectivity.
    sigma = (1.8 / centers).unsqueeze(1)
    window = torch.exp(-0.5 * (time_axis.unsqueeze(0) / sigma).square())
    carrier = torch.cos(2.0 * math.pi * centers.unsqueeze(1) * time_axis.unsqueeze(0))
    kernels = window * carrier
    kernels = kernels - kernels.mean(dim=-1, keepdim=True)
    kernels = kernels / kernels.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return kernels.unsqueeze(1), centers


def _lif_encode_full_rate(envelope: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an unnormalized LIF encoder at the raw sample rate.

    Args:
        envelope: Non-negative channel activity, shape `[channels, time]`.
        config: Acoustic configuration.

    Returns:
        Pair `(spikes, membrane)`, both with shape `[channels, time]`.
    """
    # The old LIF beta acted at the downsampled envelope rate. Convert it to an
    # approximately equivalent per-sample leak so removing the downsample block
    # does not make the membrane four times leakier.
    beta_per_sample = float(config.spike_beta) ** (1.0 / max(int(config.envelope_downsample), 1))
    membrane = torch.zeros(envelope.shape[0], dtype=envelope.dtype)
    membranes = []
    spikes = []
    for time_index in range(envelope.shape[-1]):
        membrane = beta_per_sample * membrane + envelope[:, time_index]
        spike = (membrane >= config.spike_threshold).to(envelope.dtype)
        membrane = (membrane - spike * config.spike_threshold).clamp_min(0.0)
        membranes.append(membrane)
        spikes.append(spike)
    return torch.stack(spikes, dim=-1), torch.stack(membranes, dim=-1)


def _run_original_fft_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the old FFT/IFFT + envelope + LIF cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
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
    return stages["cochleagram"][0], lif["spikes"][0], stages["center_frequencies_hz"]


def _run_conv_lif_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the proposed Conv1D filterbank + LIF cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    kernels, centers = _make_gabor_filterbank(config)
    filtered = F.conv1d(signal.view(1, 1, -1), kernels, padding=CONV_KERNEL_SIZE // 2)[0]
    cochleagram = F.relu(filtered)
    spikes, _ = _lif_encode_full_rate(cochleagram, config)
    return cochleagram, spikes, centers


def _run_level_crossing_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a filterbank followed by level-crossing delta modulation.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`. The cochleagram is the signed
        filtered signal; the spike raster combines up and down events.
    """
    kernels, centers = _make_gabor_filterbank(config)
    filtered = F.conv1d(signal.view(1, 1, -1), kernels, padding=CONV_KERNEL_SIZE // 2)[0]
    delta = float(config.spike_threshold)
    reference = filtered[:, :1].clone()
    events = torch.zeros_like(filtered)
    for time_index in range(filtered.shape[-1]):
        value = filtered[:, time_index : time_index + 1]
        up = value - reference >= delta
        down = reference - value >= delta
        event = (up | down).to(filtered.dtype)
        # Move the internal level by all crossed delta steps. This keeps the
        # encoder stable when the waveform jumps by more than one threshold.
        upward_steps = torch.floor((value - reference).clamp_min(0.0) / delta)
        downward_steps = torch.floor((reference - value).clamp_min(0.0) / delta)
        reference = reference + upward_steps * delta - downward_steps * delta
        events[:, time_index] = event.squeeze(1)
    return filtered, events, centers


def _run_rf_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a direct resonate-and-fire neuron bank.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`. The cochleagram is oscillator
        energy and the raster is thresholded RF output spikes.
    """
    centers = _log_spaced_centers(config)
    theta = 2.0 * math.pi * centers / config.sample_rate_hz
    decay = torch.exp(-theta / (2.0 * RF_Q_FACTOR))
    state = torch.zeros_like(centers)
    velocity = torch.zeros_like(centers)
    states = []
    velocities = []
    spikes = []
    for sample in signal:
        velocity = decay * velocity + RF_INPUT_GAIN * sample - theta * state
        state = state + theta * velocity
        spike = (state >= RF_THRESHOLD).to(signal.dtype)
        state = (state - spike * RF_THRESHOLD).clamp_min(0.0)
        states.append(state)
        velocities.append(velocity)
        spikes.append(spike)
    state_trace = torch.stack(states, dim=-1)
    velocity_trace = torch.stack(velocities, dim=-1)
    energy = torch.sqrt(state_trace.square() + velocity_trace.square())
    return energy, torch.stack(spikes, dim=-1), centers


def _median_runtime_s(function: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    """Benchmark a cochlea function using a median runtime.

    Args:
        function: Zero-argument function that runs one cochlea model.

    Returns:
        Median elapsed time in seconds.
    """
    with torch.no_grad():
        function()
        times = []
        for _ in range(BENCHMARK_REPEATS):
            start = time.perf_counter()
            function()
            times.append(time.perf_counter() - start)
    return float(np.median(times))


def _estimate_original_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs for the old FFT/IFFT cochlea.

    Args:
        config: Acoustic configuration.
        signal_samples: Number of waveform samples.

    Returns:
        Rough FLOP count.
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


def _estimate_conv_lif_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs for the Conv1D + LIF cochlea."""
    channels = config.num_cochlea_channels
    convolution = channels * signal_samples * CONV_KERNEL_SIZE * 2.0
    rectification = channels * signal_samples
    lif = channels * signal_samples * 5.0
    return convolution + rectification + lif


def _estimate_level_crossing_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs/comparisons for Conv1D + level crossing."""
    channels = config.num_cochlea_channels
    convolution = channels * signal_samples * CONV_KERNEL_SIZE * 2.0
    comparisons_and_updates = channels * signal_samples * 6.0
    return convolution + comparisons_and_updates


def _estimate_rf_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs/comparisons for the RF neuron bank."""
    channels = config.num_cochlea_channels
    return channels * signal_samples * 10.0


def _plot_cochleagram(result: CochleaResult, path: Path) -> str:
    """Plot continuous channel activity for a cochlea model.

    Args:
        result: Cochlea model output.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    values = _to_numpy(result.cochleagram)
    display = np.log1p(np.maximum(values, 0.0) if result.name != "level_crossing" else np.abs(values))
    time_ms = np.arange(values.shape[-1], dtype=np.float64) / result.time_rate_hz * 1_000.0
    centers_khz = _to_numpy(result.center_frequencies_hz) / 1_000.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], centers_khz[0], centers_khz[-1]],
        cmap="magma",
    )
    ax.set_title(f"{result.title}: cochleagram / channel activity")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel centre frequency (kHz)")
    ax.set_xlim(0.0, min(time_ms[-1], 24.0))
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log activity")
    return save_figure(fig, path)


def _plot_spike_raster(result: CochleaResult, path: Path) -> str:
    """Plot a channel-by-time spike raster for a cochlea model.

    Args:
        result: Cochlea model output.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    spikes = _to_numpy(result.spikes > 0.0)
    time_ms = np.arange(spikes.shape[-1], dtype=np.float64) / result.time_rate_hz * 1_000.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for channel in range(spikes.shape[0]):
        spike_times = time_ms[np.flatnonzero(spikes[channel])]
        if spike_times.size:
            ax.vlines(spike_times, channel + 0.15, channel + 0.85, color="#111827", linewidth=0.55)
    ax.set_title(f"{result.title}: spike raster")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel")
    ax.set_ylim(0, spikes.shape[0])
    ax.set_xlim(0.0, min(time_ms[-1], 24.0))
    return save_figure(fig, path)


def _run_all_models(signal: torch.Tensor, config: GlobalConfig) -> list[CochleaResult]:
    """Run and benchmark all cochlea models.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        List of model results.
    """
    signal_samples = signal.numel()
    specs = [
        (
            "original_fft_lif",
            "Original FFT/IFFT + envelope + LIF",
            lambda: _run_original_fft_cochlea(signal, config),
            _estimate_original_flops(config, signal_samples),
            "Old fixed cochlea: FFT, Gaussian frequency filters, IFFT per channel, rectification, low-pass envelope, downsample, LIF.",
        ),
        (
            "conv1d_lif",
            "Time-domain Conv1D filterbank + LIF",
            lambda: _run_conv_lif_cochlea(signal, config),
            _estimate_conv_lif_flops(config, signal_samples),
            "New dense time-domain cochlea: FIR filterbank directly in time, rectification, full-rate LIF; no explicit envelope low-pass/downsample.",
        ),
        (
            "level_crossing",
            "Time-domain filterbank + level crossing",
            lambda: _run_level_crossing_cochlea(signal, config),
            _estimate_level_crossing_flops(config, signal_samples),
            "New event encoder: FIR filterbank followed by delta-modulation events on each filtered channel.",
        ),
        (
            "rf_bank",
            "Direct resonate-and-fire bank",
            lambda: _run_rf_cochlea(signal, config),
            _estimate_rf_flops(config, signal_samples),
            "New reduced cochlea: raw waveform drives a bank of RF neurons tuned across frequency.",
        ),
    ]
    results = []
    for name, title, function, flops, notes in specs:
        elapsed_s = _median_runtime_s(function)
        cochleagram, spikes, centers = function()
        results.append(
            CochleaResult(
                name=name,
                title=title,
                cochleagram=cochleagram.detach(),
                spikes=spikes.detach(),
                time_rate_hz=config.envelope_rate_hz if name == "original_fft_lif" else float(config.sample_rate_hz),
                center_frequencies_hz=centers.detach(),
                elapsed_s=elapsed_s,
                flops_estimate=float(flops),
                sops_estimate=float((spikes > 0.0).sum().item()),
                notes=notes,
            )
        )
    return results


def _write_report(results: list[CochleaResult], artifacts: dict[str, dict[str, str]], config: GlobalConfig, elapsed_s: float) -> None:
    """Write the cochlea-analysis markdown report.

    Args:
        results: Model outputs and metrics.
        artifacts: Figure paths grouped by model name.
        config: Acoustic configuration.
        elapsed_s: Total experiment runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_name = {result.name: result for result in results}
    lines = [
        "# Mini Model 3: Cochlea Analysis",
        "",
        "This mini model compares four candidate cochlea front ends. The aim is not yet to optimise them, but to check whether the mechanisms produce sensible channel activity and spikes, and to estimate their relative computational cost.",
        "",
        "## Shared Setup",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz} Hz` |",
        f"| chirp | `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` |",
        f"| chirp duration | `{config.chirp_duration_s * 1_000.0:.1f} ms` |",
        f"| signal duration | `{config.signal_duration_s * 1_000.0:.1f} ms` |",
        f"| cochlea band | `{config.cochlea_low_hz:.0f} -> {config.cochlea_high_hz:.0f} Hz` |",
        f"| channels | `{config.num_cochlea_channels}` |",
        f"| spike envelope normalization | `{config.normalize_spike_envelope}` |",
        f"| transmit gain | `{config.transmit_gain:.0f}x` |",
        "",
        "The input is one clean left-ear echo from the matched-human signal setup. Keeping one waveform fixed means the plots compare the front ends, not scene variability.",
        "",
        "## Cost Summary",
        "",
        "| Model | FLOPs estimate | SOPs / output events | Time | Time per channel | Spike density |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        spike_density = result.sops_estimate / float(result.spikes.numel())
        lines.append(
            f"| {result.title} | `{result.flops_estimate:,.0f}` | `{result.sops_estimate:,.0f}` | "
            f"`{result.elapsed_s * 1_000.0:.3f} ms` | `{result.elapsed_s / config.num_cochlea_channels * 1_000.0:.4f} ms` | `{spike_density:.4f}` |"
        )
    lines.extend(
        [
            "",
            "FLOPs are approximate dense-operation counts for one waveform. SOPs are counted here as emitted output spike/events, because downstream event-driven processing cost would scale with those events. This is a first-order proxy, not a hardware-validated energy model.",
            "",
            "## 1. Original FFT/IFFT + Envelope + LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[FFT]",
            "    B --> C[log Gaussian filterbank]",
            "    C --> D[IFFT per channel]",
            "    D --> E[half-wave rectification]",
            "    E --> F[Hann low-pass envelope]",
            "    F --> G[downsample]",
            "    G --> H[LIF spike encoder]",
            "    H --> I[spike raster]",
            "```",
            "",
            "```text",
            "X(f) = FFT{x(t)}",
            "x_c(t) = IFFT{X(f) * G_c(f)}",
            "e_c(t) = downsample(lowpass(max(x_c(t), 0)))",
            "v_c[t] = beta * v_c[t-1] + e_c[t]",
            "spike_c[t] = 1 if v_c[t] >= threshold else 0",
            "v_c[t] = max(v_c[t] - threshold * spike_c[t], 0)",
            "```",
            "",
            by_name["original_fft_lif"].notes,
            "",
            "![Original cochleagram](../outputs/cochlea_analysis/figures/original_fft_lif_cochleagram.png)",
            "",
            "![Original raster](../outputs/cochlea_analysis/figures/original_fft_lif_raster.png)",
            "",
            "## 2. Time-Domain Conv1D Filterbank + LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[Conv1D FIR filterbank]",
            "    B --> C[half-wave rectification]",
            "    C --> D[full-rate LIF spike encoder]",
            "    D --> E[spike raster]",
            "```",
            "",
            "```text",
            "x_c[t] = sum_k h_c[k] * x[t-k]",
            "e_c[t] = max(x_c[t], 0)",
            "v_c[t] = beta_sample * v_c[t-1] + e_c[t]",
            "beta_sample = beta_old^(1 / downsample)",
            "spike_c[t] = 1 if v_c[t] >= threshold else 0",
            "```",
            "",
            by_name["conv1d_lif"].notes,
            "",
            "![Conv1D cochleagram](../outputs/cochlea_analysis/figures/conv1d_lif_cochleagram.png)",
            "",
            "![Conv1D raster](../outputs/cochlea_analysis/figures/conv1d_lif_raster.png)",
            "",
            "## 3. Time-Domain Filterbank + Level Crossing",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[Conv1D FIR filterbank]",
            "    B --> C[level-crossing delta modulator]",
            "    C --> D[up/down event raster]",
            "```",
            "",
            "```text",
            "x_c[t] = sum_k h_c[k] * x[t-k]",
            "if x_c[t] - ref_c[t] >= delta: emit up event, ref_c += n * delta",
            "if ref_c[t] - x_c[t] >= delta: emit down event, ref_c -= n * delta",
            "```",
            "",
            by_name["level_crossing"].notes,
            "",
            "![Level crossing cochleagram](../outputs/cochlea_analysis/figures/level_crossing_cochleagram.png)",
            "",
            "![Level crossing raster](../outputs/cochlea_analysis/figures/level_crossing_raster.png)",
            "",
            "## 4. Direct Resonate-And-Fire Bank",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[RF neuron bank tuned across frequency]",
            "    B --> C[oscillator energy / state]",
            "    B --> D[spike raster]",
            "```",
            "",
            "```text",
            "velocity_c[t] = decay_c * velocity_c[t-1] + gain * x[t] - theta_c * state_c[t-1]",
            "state_c[t] = state_c[t-1] + theta_c * velocity_c[t]",
            "theta_c = 2*pi*f_c/sample_rate",
            "spike_c[t] = 1 if state_c[t] >= threshold else 0",
            "```",
            "",
            by_name["rf_bank"].notes,
            "",
            "![RF cochleagram](../outputs/cochlea_analysis/figures/rf_bank_cochleagram.png)",
            "",
            "![RF raster](../outputs/cochlea_analysis/figures/rf_bank_raster.png)",
            "",
            "## Initial Interpretation",
            "",
            "- The original model is the faithful baseline and has the most envelope-shaped representation, but it pays for FFT/IFFT reconstruction plus smoothing.",
            "- The Conv1D model stays in the time domain and removes explicit low-pass/downsample blocks, but naive FIR convolution is not automatically cheaper unless the kernels are short or optimized.",
            "- The level-crossing model is the cleanest route toward event-based processing after the filterbank, but the filterbank itself is still dense in this first implementation.",
            "- The RF model is the most reduced conceptually because the resonators are both filters and spiking units, but its parameters need careful tuning before using it as a full cochlea replacement.",
            "- Binarisation and event-based processing should be evaluated after we decide which of these mechanisms gives useful spike timing and channel selectivity.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for model_artifacts in artifacts.values():
        for name, path in model_artifacts.items():
            lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the cochlea comparison mini experiment.

    Returns:
        JSON-serializable summary dictionary.
    """
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPORT_PATH.parent)
    torch.manual_seed(7)
    config = _make_config()
    signal = _simulate_input_waveform(config)
    results = _run_all_models(signal, config)

    artifacts: dict[str, dict[str, str]] = {}
    for result in results:
        artifacts[result.name] = {
            "cochleagram": _plot_cochleagram(result, FIGURE_DIR / f"{result.name}_cochleagram.png"),
            "raster": _plot_spike_raster(result, FIGURE_DIR / f"{result.name}_raster.png"),
        }

    elapsed_s = time.perf_counter() - start
    payload: dict[str, object] = {
        "experiment": "cochlea_analysis",
        "elapsed_seconds": elapsed_s,
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "chirp_start_hz": config.chirp_start_hz,
            "chirp_end_hz": config.chirp_end_hz,
            "chirp_duration_s": config.chirp_duration_s,
            "signal_duration_s": config.signal_duration_s,
            "num_cochlea_channels": config.num_cochlea_channels,
            "cochlea_low_hz": config.cochlea_low_hz,
            "cochlea_high_hz": config.cochlea_high_hz,
            "filter_bandwidth_sigma": config.filter_bandwidth_sigma,
            "envelope_lowpass_hz": config.envelope_lowpass_hz,
            "envelope_downsample": config.envelope_downsample,
            "spike_threshold": config.spike_threshold,
            "spike_beta": config.spike_beta,
            "normalize_spike_envelope": config.normalize_spike_envelope,
            "transmit_gain": config.transmit_gain,
            "conv_kernel_size": CONV_KERNEL_SIZE,
            "rf_q_factor": RF_Q_FACTOR,
            "rf_input_gain": RF_INPUT_GAIN,
            "rf_threshold": RF_THRESHOLD,
        },
        "models": [
            {
                "name": result.name,
                "title": result.title,
                "elapsed_seconds": result.elapsed_s,
                "elapsed_ms": result.elapsed_s * 1_000.0,
                "elapsed_ms_per_channel": result.elapsed_s / config.num_cochlea_channels * 1_000.0,
                "flops_estimate": result.flops_estimate,
                "sops_estimate": result.sops_estimate,
                "spike_density": result.sops_estimate / float(result.spikes.numel()),
                "time_rate_hz": result.time_rate_hz,
                "cochleagram_shape": list(result.cochleagram.shape),
                "spike_shape": list(result.spikes.shape),
                "notes": result.notes,
            }
            for result in results
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(results, artifacts, config, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
