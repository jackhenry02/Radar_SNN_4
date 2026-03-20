from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.acoustics import (
    cochlea_filterbank_stages,
    generate_fm_chirp,
    lif_encode_stages,
    simulate_echo_batch,
)
from utils.common import GlobalConfig, OutputPaths, save_cochlea_plot, save_waveform_and_spectrogram, seed_everything


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _finalize(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_transmit_receive_plot(
    transmit: torch.Tensor,
    receive: torch.Tensor,
    sample_rate_hz: int,
    path: Path,
) -> None:
    transmit_np = _to_numpy(transmit)
    receive_np = _to_numpy(receive)
    time_axis_ms = np.arange(transmit_np.shape[-1]) / sample_rate_hz * 1_000.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_axis_ms, transmit_np, linewidth=1.0, color="#355070")
    axes[0].set_title("Transmit Chirp")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(time_axis_ms, receive_np, linewidth=1.0, color="#b56576")
    axes[1].set_title("Left-Ear Echo Input To Cochlea")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude")
    _finalize(path)


def _save_center_frequency_plot(center_frequencies_hz: torch.Tensor, path: Path) -> None:
    center_np = _to_numpy(center_frequencies_hz)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(center_np.shape[0]), center_np / 1_000.0, marker="o", linewidth=1.5)
    ax.set_title("Cochlea Center Frequencies")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Center Frequency (kHz)")
    ax.grid(True, alpha=0.25)
    _finalize(path)


def _save_filter_response_plot(
    frequencies_hz: torch.Tensor,
    filters: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    path: Path,
) -> None:
    freq_np = _to_numpy(frequencies_hz) / 1_000.0
    filters_np = _to_numpy(filters)
    center_np = _to_numpy(center_frequencies_hz)
    channel_indices = np.linspace(0, center_np.shape[0] - 1, 6, dtype=int)
    fig, ax = plt.subplots(figsize=(10, 5))
    for channel_index in channel_indices:
        ax.plot(
            freq_np,
            filters_np[channel_index],
            linewidth=1.5,
            label=f"ch {channel_index} ({center_np[channel_index] / 1000.0:.1f} kHz)",
        )
    ax.set_title("Representative Log-Spaced Filter Responses")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Gain")
    ax.set_xlim(freq_np.min(), freq_np.max())
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    _finalize(path)


def _save_filter_heatmap(
    frequencies_hz: torch.Tensor,
    filters: torch.Tensor,
    path: Path,
) -> None:
    freq_np = _to_numpy(frequencies_hz) / 1_000.0
    filters_np = _to_numpy(filters)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        filters_np,
        aspect="auto",
        origin="lower",
        extent=[freq_np.min(), freq_np.max(), 0, filters_np.shape[0] - 1],
        cmap="viridis",
    )
    ax.set_title("Full Filterbank Response Matrix")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Channel")
    _finalize(path)


def _save_channel_examples(
    filtered: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    sample_rate_hz: int,
    path: Path,
) -> None:
    filtered_np = _to_numpy(filtered)
    center_np = _to_numpy(center_frequencies_hz)
    time_axis_ms = np.arange(filtered_np.shape[-1]) / sample_rate_hz * 1_000.0
    channel_indices = np.linspace(0, filtered_np.shape[0] - 1, 3, dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis, channel_index in zip(axes, channel_indices, strict=True):
        axis.plot(time_axis_ms, filtered_np[channel_index], linewidth=1.0)
        axis.set_ylabel("Amp")
        axis.set_title(f"Filtered Channel {channel_index} ({center_np[channel_index] / 1000.0:.1f} kHz)")
    axes[-1].set_xlabel("Time (ms)")
    _finalize(path)


def _save_rectify_smooth_plot(
    filtered: torch.Tensor,
    rectified: torch.Tensor,
    smoothed: torch.Tensor,
    cochleagram: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    sample_rate_hz: int,
    downsample: int,
    channel_index: int,
    path: Path,
) -> None:
    filtered_np = _to_numpy(filtered[channel_index])
    rectified_np = _to_numpy(rectified[channel_index])
    smoothed_np = _to_numpy(smoothed[channel_index])
    cochlea_np = _to_numpy(cochleagram[channel_index])
    center_khz = float(center_frequencies_hz[channel_index].item()) / 1_000.0
    time_ms = np.arange(filtered_np.shape[-1]) / sample_rate_hz * 1_000.0
    cochlea_time_ms = np.arange(cochlea_np.shape[-1]) / (sample_rate_hz / downsample) * 1_000.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(time_ms, filtered_np, linewidth=1.0, color="#355070")
    axes[0].set_title(f"Representative Channel Pipeline ({center_khz:.1f} kHz)")
    axes[0].set_ylabel("Filtered")
    axes[1].plot(time_ms, rectified_np, linewidth=1.0, color="#6d597a")
    axes[1].set_ylabel("Rectified")
    axes[2].plot(time_ms, smoothed_np, linewidth=1.0, color="#b56576")
    axes[2].set_ylabel("Smoothed")
    axes[3].plot(cochlea_time_ms, cochlea_np, linewidth=1.0, color="#e56b6f")
    axes[3].set_ylabel("Downsampled")
    axes[3].set_xlabel("Time (ms)")
    _finalize(path)


def _save_lowpass_kernel_plot(kernel: torch.Tensor, sample_rate_hz: int, path: Path) -> None:
    kernel_np = _to_numpy(kernel)
    time_ms = np.arange(kernel_np.shape[0]) / sample_rate_hz * 1_000.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ms, kernel_np, linewidth=1.5)
    ax.set_title("Envelope Low-Pass Kernel")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.25)
    _finalize(path)


def _save_membrane_plot(
    scaled_envelope: torch.Tensor,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    envelope_rate_hz: int,
    channel_index: int,
    path: Path,
) -> None:
    scaled_np = _to_numpy(scaled_envelope[channel_index])
    membrane_np = _to_numpy(membrane[channel_index])
    spikes_np = _to_numpy(spikes[channel_index])
    time_axis_ms = np.arange(scaled_np.shape[-1]) / envelope_rate_hz * 1_000.0
    center_khz = float(center_frequencies_hz[channel_index].item()) / 1_000.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_axis_ms, scaled_np, linewidth=1.0, label="scaled envelope")
    axes[0].plot(time_axis_ms, membrane_np, linewidth=1.0, label="membrane")
    axes[0].set_title(f"LIF Dynamics For Representative Channel ({center_khz:.1f} kHz)")
    axes[0].set_ylabel("State")
    axes[0].legend()
    spike_indices = np.nonzero(spikes_np > 0.0)[0]
    if spike_indices.size:
        axes[1].scatter(time_axis_ms[spike_indices], np.ones_like(spike_indices), s=12, c="black")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_yticks([1.0])
    axes[1].set_yticklabels(["spike"])
    axes[1].set_xlabel("Time (ms)")
    _finalize(path)


def run_cochlea_explained(config: GlobalConfig, outputs: OutputPaths) -> dict[str, str]:
    seed_everything(config.seed)
    figure_dir = outputs.root / "cochlea_explained"
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    radii = torch.tensor([1.4], device=device)
    azimuth = torch.tensor([18.0], device=device)
    elevation = torch.tensor([12.0], device=device)
    acoustic_batch = simulate_echo_batch(
        config,
        radii,
        azimuth,
        elevation,
        binaural=True,
        add_noise=False,
        include_elevation_cues=True,
    )
    transmit = acoustic_batch.transmit[0]
    receive_left = acoustic_batch.receive[0, 0]
    chirp, _ = generate_fm_chirp(config, batch_size=1, device=device)

    filter_stages = cochlea_filterbank_stages(
        receive_left.unsqueeze(0),
        sample_rate_hz=config.sample_rate_hz,
        num_channels=config.num_cochlea_channels,
        low_hz=config.cochlea_low_hz,
        high_hz=config.cochlea_high_hz,
        filter_bandwidth_sigma=config.filter_bandwidth_sigma,
        envelope_lowpass_hz=config.envelope_lowpass_hz,
        downsample=config.envelope_downsample,
    )
    lif_stages = lif_encode_stages(
        filter_stages["cochleagram"],
        threshold=config.spike_threshold,
        beta=config.spike_beta,
    )

    representative_channel = int(torch.argmin(torch.abs(filter_stages["center_frequencies_hz"] - 45_000.0)).item())

    save_waveform_and_spectrogram(
        receive_left,
        config.sample_rate_hz,
        figure_dir / "example_signal.png",
        "Example Receive Waveform And Spectrogram",
    )
    _save_transmit_receive_plot(transmit, receive_left, config.sample_rate_hz, figure_dir / "transmit_receive.png")
    _save_center_frequency_plot(filter_stages["center_frequencies_hz"], figure_dir / "center_frequencies.png")
    _save_filter_response_plot(
        filter_stages["frequencies_hz"],
        filter_stages["filters"],
        filter_stages["center_frequencies_hz"],
        figure_dir / "filter_responses.png",
    )
    _save_filter_heatmap(filter_stages["frequencies_hz"], filter_stages["filters"], figure_dir / "filter_heatmap.png")
    _save_channel_examples(
        filter_stages["filtered"][0],
        filter_stages["center_frequencies_hz"],
        config.sample_rate_hz,
        figure_dir / "filtered_channels.png",
    )
    _save_rectify_smooth_plot(
        filter_stages["filtered"][0],
        filter_stages["rectified"][0],
        filter_stages["smoothed"][0],
        filter_stages["cochleagram"][0],
        filter_stages["center_frequencies_hz"],
        config.sample_rate_hz,
        config.envelope_downsample,
        representative_channel,
        figure_dir / "channel_pipeline.png",
    )
    _save_lowpass_kernel_plot(filter_stages["lowpass_kernel"], config.sample_rate_hz, figure_dir / "lowpass_kernel.png")
    save_cochlea_plot(
        filter_stages["cochleagram"][0],
        lif_stages["spikes"][0],
        config.envelope_rate_hz,
        figure_dir / "cochleagram_spikes.png",
        "Final Cochleagram And Spike Raster",
    )
    _save_membrane_plot(
        lif_stages["scaled_envelope"][0],
        lif_stages["membrane"][0],
        lif_stages["spikes"][0],
        filter_stages["center_frequencies_hz"],
        config.envelope_rate_hz,
        representative_channel,
        figure_dir / "membrane_spikes.png",
    )

    report_lines = [
        "# Cochlea Explained",
        "",
        "## Overview",
        "",
        "This document describes the current fixed cochlea front end used by the localisation system. The example figures are generated from one clean left-ear echo scene so the transformations are easy to inspect.",
        "",
        "Example scene:",
        f"- Distance: `{radii.item():.2f} m`",
        f"- Azimuth: `{azimuth.item():.1f} deg`",
        f"- Elevation: `{elevation.item():.1f} deg`",
        "- Binaural simulation: `on`",
        "- Noise: `off` for clarity",
        "",
        "## Pipeline",
        "",
        "```mermaid",
        "graph TD",
        "    A[Receive waveform] --> B[FFT]",
        "    B --> C[Log-spaced Gaussian filterbank]",
        "    C --> D[Inverse FFT per channel]",
        "    D --> E[Half-wave rectification]",
        "    E --> F[Low-pass envelope smoothing]",
        "    F --> G[Temporal downsampling]",
        "    G --> H[LIF spike encoder]",
        "    H --> I[Transmit / receive spike tensors]",
        "```",
        "",
        "## Current Fixed Parameters",
        "",
        "| Parameter | Value | Role |",
        "| --- | --- | --- |",
        f"| `sample_rate_hz` | `{config.sample_rate_hz}` | Raw waveform sampling rate |",
        f"| `num_cochlea_channels` | `{config.num_cochlea_channels}` | Number of frequency channels |",
        f"| `cochlea_low_hz` | `{config.cochlea_low_hz:.0f}` | Lowest cochlear center frequency |",
        f"| `cochlea_high_hz` | `{config.cochlea_high_hz:.0f}` | Highest cochlear center frequency |",
        f"| `filter_bandwidth_sigma` | `{config.filter_bandwidth_sigma:.3f}` | Width of the Gaussian log-frequency filters |",
        f"| `envelope_lowpass_hz` | `{config.envelope_lowpass_hz:.0f}` | Envelope smoothing cutoff proxy |",
        f"| `envelope_downsample` | `{config.envelope_downsample}` | Temporal downsampling factor before spiking |",
        f"| `spike_threshold` | `{config.spike_threshold:.2f}` | LIF firing threshold |",
        f"| `spike_beta` | `{config.spike_beta:.2f}` | LIF leak factor |",
        "",
        "## 1. Input Signal",
        "",
        "The cochlea receives the left-ear echo waveform. The transmitted chirp is shown alongside it for reference.",
        "",
        "![Input spectrogram](cochlea_explained/example_signal.png)",
        "![Transmit vs receive](cochlea_explained/transmit_receive.png)",
        "",
        "## 2. Log-Spaced Filterbank",
        "",
        "The raw waveform is transformed into the frequency domain, multiplied by a bank of Gaussian filters in log-frequency space, and returned to the time domain channel by channel.",
        "",
        "![Center frequencies](cochlea_explained/center_frequencies.png)",
        "![Filter responses](cochlea_explained/filter_responses.png)",
        "![Filter heatmap](cochlea_explained/filter_heatmap.png)",
        "",
        "## 3. Per-Channel Filtered Signals",
        "",
        "After inverse FFT, each channel contains a band-limited version of the original waveform. Low, middle, and high channels respond at different parts of the chirp.",
        "",
        "![Filtered channels](cochlea_explained/filtered_channels.png)",
        "",
        "## 4. Rectification, Smoothing, And Downsampling",
        "",
        "Each channel is half-wave rectified, smoothed with a Hann low-pass kernel, and then downsampled. The downsampled smoothed envelope is the actual input to the spike encoder.",
        "",
        "![Channel pipeline](cochlea_explained/channel_pipeline.png)",
        "![Low-pass kernel](cochlea_explained/lowpass_kernel.png)",
        "",
        "## 5. LIF Spike Encoding",
        "",
        "The smoothed envelope is normalized, integrated through a fixed LIF neuron per channel, thresholded, and reset by subtraction. Spikes are therefore driven by envelope peaks in each frequency band.",
        "",
        "![Membrane and spikes](cochlea_explained/membrane_spikes.png)",
        "",
        "## 6. Final Cochleagram And Spike Raster",
        "",
        "The final cochleagram is the smoothed, downsampled envelope across all channels. The spike raster is the binary output that the rest of the localisation system consumes.",
        "",
        "![Cochleagram and spikes](cochlea_explained/cochleagram_spikes.png)",
        "",
        "## Interface To The Rest Of The Model",
        "",
        "The current barrier is after spike generation:",
        "",
        "- transmit spikes: shape `[batch, channel, time]`",
        "- receive spikes: shape `[batch, ear, channel, time]`",
        "",
        "Everything downstream assumes those spike tensors already exist. That makes the current cochlea replaceable, but the easiest swap is another cochlea that preserves the same spike-tensor contract and envelope-rate time base.",
        "",
        "## Current Interpretation",
        "",
        "This cochlea is fixed and hand-designed. It is not currently trainable. The expensive part is the fixed FFT filterbank plus spike conversion, not the later handcrafted pathway feature extraction.",
    ]
    report_path = outputs.root / "cochlea_explained.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "report": str(report_path),
        "figure_dir": str(figure_dir),
    }
