from __future__ import annotations

"""Generate signal-analysis plots for the mini-model project.

This experiment visualizes the acoustic signal before any neural processing:
the emitted call, the received echoes, binaural head-shadow effects, noise and
jitter, and the elevation spectral notch.
"""

import json
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.common.signals import moving_notch_signal_config
from models.acoustics import (
    generate_fm_chirp,
    pad_signal,
    simulate_echo_batch,
)
from utils.common import GlobalConfig


ROOT = PROJECT_ROOT
OUTPUT_DIR = ROOT / "mini_models" / "outputs" / "signal_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = ROOT / "mini_models" / "reports" / "signal_analysis.md"
ANGLE_LIMIT_DEG = 45.0
EXAMPLE_DISTANCE_M = 3.0
EXAMPLE_AZIMUTH_DEG = 35.0
EXAMPLE_ELEVATION_DEG = 20.0
NOTCH_SWEEP_LOW_HZ = 4_000.0
NOTCH_SWEEP_HIGH_HZ = 16_000.0


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a CPU numpy array."""
    return values.detach().cpu().numpy()


def _time_ms(num_samples: int, sample_rate_hz: int) -> np.ndarray:
    """Return a millisecond time axis for a sampled signal."""
    return np.arange(num_samples, dtype=np.float64) / sample_rate_hz * 1_000.0


def _angle_scale(angle_deg: torch.Tensor, limit_deg: float = ANGLE_LIMIT_DEG) -> torch.Tensor:
    """Map an angle range onto `[-1, 1]` for mini-model cue sweeps.

    Args:
        angle_deg: Angle in degrees.
        limit_deg: Symmetric angular limit in degrees.

    Returns:
        Clipped normalized angle, where `-limit_deg -> -1` and
        `+limit_deg -> +1`.
    """
    return torch.clamp(angle_deg / limit_deg, min=-1.0, max=1.0)


def _band_position(frequency_hz: torch.Tensor, config: GlobalConfig) -> torch.Tensor:
    """Normalize frequency position inside the cochlear band.

    Args:
        frequency_hz: Frequency axis in Hz.
        config: Acoustic configuration.

    Returns:
        Frequency position where `0` is `cochlea_low_hz` and `1` is
        `cochlea_high_hz`.
    """
    width_hz = max(float(config.cochlea_high_hz - config.cochlea_low_hz), 1.0)
    return torch.clamp((frequency_hz - config.cochlea_low_hz) / width_hz, min=0.0, max=1.0)


def _frequency_to_band_position(frequency_hz: float, config: GlobalConfig) -> float:
    """Convert a physical frequency to normalized cochlear-band position.

    Args:
        frequency_hz: Frequency in Hz.
        config: Acoustic configuration.

    Returns:
        Normalized position in the cochlear band.
    """
    width_hz = max(float(config.cochlea_high_hz - config.cochlea_low_hz), 1.0)
    return float(np.clip((frequency_hz - config.cochlea_low_hz) / width_hz, 0.0, 1.0))


def _moving_gaussian_notch_gain(
    config: GlobalConfig,
    elevation_deg: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    include_slope: bool,
) -> torch.Tensor:
    """Create the mini-model moving Gaussian elevation notch.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angles in degrees.
        frequency_hz: Frequency axis in Hz.
        include_slope: Whether to include the broad spectral slope term.

    Returns:
        Linear gain matrix with shape `[num_elevations, num_frequencies]`.
    """
    scale = _angle_scale(elevation_deg.to(frequency_hz.dtype)).unsqueeze(1)
    band_pos = _band_position(frequency_hz, config).unsqueeze(0)
    center_min = _frequency_to_band_position(NOTCH_SWEEP_LOW_HZ, config)
    center_max = _frequency_to_band_position(NOTCH_SWEEP_HIGH_HZ, config)
    center = center_min + (center_max - center_min) * (0.5 * (scale + 1.0))
    notch_width = max(float(config.elevation_notch_width), 1e-3)
    notch_profile = torch.exp(-0.5 * ((band_pos - center) / notch_width).square())
    gain = torch.exp(-float(config.elevation_notch_strength) * notch_profile)
    if include_slope and config.elevation_spectral_strength > 0.0:
        gain = gain * torch.exp(float(config.elevation_spectral_strength) * scale * (band_pos - 0.5))
    return gain


def _butterworth_notch_gain(
    config: GlobalConfig,
    elevation_deg: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    order: int = 4,
    width_fraction: float = 0.055,
    depth: float = 0.92,
) -> torch.Tensor:
    """Create an inverted Butterworth band-pass response as a notch cue.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angles in degrees.
        frequency_hz: Frequency axis in Hz.
        order: Butterworth order. Higher values give sharper notch edges.
        width_fraction: Half-width of the band-pass template as a fraction of
            the cochlear band.
        depth: Maximum notch depth as a linear fraction.

    Returns:
        Linear gain matrix with shape `[num_elevations, num_frequencies]`.
    """
    scale = _angle_scale(elevation_deg.to(frequency_hz.dtype)).unsqueeze(1)
    band_pos = _band_position(frequency_hz, config).unsqueeze(0)
    center_min = _frequency_to_band_position(NOTCH_SWEEP_LOW_HZ, config)
    center_max = _frequency_to_band_position(NOTCH_SWEEP_HIGH_HZ, config)
    center = center_min + (center_max - center_min) * (0.5 * (scale + 1.0))
    normalized_offset = torch.abs((band_pos - center) / max(width_fraction, 1e-3))
    bandpass = 1.0 / torch.sqrt(1.0 + normalized_offset.pow(2 * order))
    return torch.clamp(1.0 - depth * bandpass, min=1e-3, max=1.0)


def _comb_interference_gain(
    config: GlobalConfig,
    elevation_deg: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    lowest_notch_hz: float = 4_000.0,
    highest_notch_hz: float = 16_000.0,
    delayed_copy_gain: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a comb-filter gain from interference with a delayed copy.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angles in degrees.
        frequency_hz: Frequency axis in Hz.
        lowest_notch_hz: First-notch frequency at negative elevation.
        highest_notch_hz: First-notch frequency at positive elevation.
        delayed_copy_gain: Relative amplitude of the delayed copy.

    Returns:
        Tuple of linear gain matrix `[num_elevations, num_frequencies]` and
        lag values in seconds `[num_elevations]`.
    """
    del config
    scale = _angle_scale(elevation_deg.to(frequency_hz.dtype))
    first_notch_hz = lowest_notch_hz + (highest_notch_hz - lowest_notch_hz) * (0.5 * (scale + 1.0))
    lag_s = 1.0 / (2.0 * first_notch_hz.clamp_min(1.0))
    phase = 2.0 * math.pi * lag_s[:, None] * frequency_hz[None, :]
    gain = torch.sqrt(
        1.0 + delayed_copy_gain**2 + 2.0 * delayed_copy_gain * torch.cos(phase)
    ) / (1.0 + delayed_copy_gain)
    return torch.clamp(gain, min=1e-3, max=1.0), lag_s


def _apply_frequency_gain(signal: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Apply a real-valued frequency-domain gain to a waveform.

    Args:
        signal: One-dimensional waveform.
        gain: Real-valued gain for the `rfft` bins of `signal`.

    Returns:
        Waveform after frequency-domain filtering.
    """
    spectrum = torch.fft.rfft(signal)
    return torch.fft.irfft(spectrum * gain.to(signal.device, signal.dtype), n=signal.numel())


def _simulate_reference_scene(config: GlobalConfig, *, add_noise: bool, jitter_std_s: float | None = None) -> object:
    """Simulate one representative binaural echo scene.

    Args:
        config: Acoustic configuration.
        add_noise: Whether to add receiver noise.
        jitter_std_s: Optional override for propagation-delay jitter.

    Returns:
        `AcousticBatch` with one representative scene inside the current
        mini-model angular limits.
    """
    if jitter_std_s is not None:
        config = GlobalConfig(**{**config.__dict__, "jitter_std_s": jitter_std_s})
    device = torch.device("cpu")
    return simulate_echo_batch(
        config,
        radii_m=torch.tensor([EXAMPLE_DISTANCE_M], device=device),
        azimuth_deg=torch.tensor([EXAMPLE_AZIMUTH_DEG], device=device),
        elevation_deg=torch.tensor([EXAMPLE_ELEVATION_DEG], device=device),
        binaural=True,
        add_noise=add_noise,
        include_elevation_cues=True,
        transmit_gain=config.transmit_gain,
    )


def _plot_emitted_spectrogram(config: GlobalConfig, path: Path) -> str:
    """Plot the emitted FM call waveform and spectrogram.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    chirp, _ = generate_fm_chirp(config, batch_size=1, device=torch.device("cpu"), transmit_gain=1.0)
    signal = pad_signal(chirp, config.signal_samples)[0]
    waveform = _to_numpy(signal)
    time_ms = _time_ms(waveform.size, config.sample_rate_hz)

    # Use a shorter analysis window than the earlier plots. The call is only
    # 3 ms long, so a long STFT window smears the chirp and makes the frequency
    # sweep look more spaced out than it really is.
    nfft = 96
    noverlap = 88
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(time_ms, waveform, color="#1f2937", linewidth=1.0)
    axes[0].set_title("Emitted FM call")
    axes[0].set_ylabel("amplitude")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_xlim(0.0, config.chirp_duration_s * 1_000.0 + 1.0)
    # The padded silent part of the signal can create zero-power spectrogram
    # bins; suppress the corresponding log10 warning because it is expected.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        _, _, _, image = axes[1].specgram(
            waveform,
            Fs=config.sample_rate_hz,
            NFFT=nfft,
            noverlap=noverlap,
            pad_to=2048,
            cmap="magma",
        )
    if hasattr(image, "set_interpolation"):
        image.set_interpolation("bilinear")
    axes[1].set_title("Spectrogram of emitted call")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("frequency (kHz)")
    axes[1].set_xlim(0.0, config.chirp_duration_s + 0.001)
    axes[1].set_ylim(config.chirp_end_hz * 0.75, config.chirp_start_hz * 1.15)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1_000.0:.1f}"))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000.0:.0f}"))
    return save_figure(fig, path)


def _plot_emit_receive(config: GlobalConfig, scene: object, path: Path) -> str:
    """Plot emitted and received amplitude against time.

    Args:
        config: Acoustic configuration.
        scene: Simulated scene from `_simulate_reference_scene`.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    transmit = _to_numpy(scene.transmit[0])
    left = _to_numpy(scene.receive[0, 0])
    right = _to_numpy(scene.receive[0, 1])
    mono_echo = 0.5 * (left + right)
    time_ms = _time_ms(transmit.size, config.sample_rate_hz)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(time_ms, transmit, color="#0f172a", linewidth=1.0)
    axes[0].set_title("Emitted call amplitude")
    axes[0].set_ylabel("amplitude")
    axes[1].plot(time_ms, mono_echo, color="#b45309", linewidth=1.0)
    axes[1].set_title("Received echo amplitude, averaged across ears")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("amplitude")
    for delay_s in _to_numpy(scene.delays_s[0]):
        axes[1].axvline(delay_s * 1_000.0, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.75)
    return save_figure(fig, path)


def _plot_binaural_head_shadow(config: GlobalConfig, scene: object, path: Path) -> str:
    """Plot left/right received echoes and the azimuth-dependent head shadow.

    Args:
        config: Acoustic configuration.
        scene: Simulated scene from `_simulate_reference_scene`.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    left = _to_numpy(scene.receive[0, 0])
    right = _to_numpy(scene.receive[0, 1])
    time_ms = _time_ms(left.size, config.sample_rate_hz)
    azimuth_rad = math.radians(float(scene.azimuth_deg[0].item()))
    head_shadow = np.exp(config.head_shadow_strength * np.sin(azimuth_rad) * np.array([-1.0, 1.0]))
    amplitudes = _to_numpy(scene.amplitudes[0])
    effective_gain = amplitudes * head_shadow

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(time_ms, left, color="#2563eb", linewidth=1.0, label="left ear")
    axes[0].plot(time_ms, right, color="#dc2626", linewidth=1.0, label="right ear")
    axes[0].set_title("Binaural received echoes")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(loc="upper right")

    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, amplitudes, width, label="distance attenuation")
    axes[1].bar(x + width / 2, effective_gain, width, label="attenuation * head shadow")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["left", "right"])
    axes[1].set_title("Head shadow changes the effective ear gain")
    axes[1].set_ylabel("linear gain")
    axes[1].legend(loc="upper right")
    return save_figure(fig, path)


def _plot_noise_and_jitter(config: GlobalConfig, path: Path) -> str:
    """Show how noise and jitter alter repeated received echoes.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    torch.manual_seed(7)
    clean = _simulate_reference_scene(config, add_noise=False, jitter_std_s=0.0)
    clean_signal = clean.receive[0, 0]
    active = clean_signal.abs() > 0.02 * clean_signal.abs().amax().clamp_min(1e-12)
    signal_rms = clean_signal[active].square().mean().sqrt() if bool(active.any()) else clean_signal.square().mean().sqrt()
    target_snr_db = 10.0
    diagnostic_noise_std = float(signal_rms.item() / (10.0 ** (target_snr_db / 20.0)))
    diagnostic_jitter_s = 2.5e-4
    noisy_config = GlobalConfig(**{**config.__dict__, "noise_std": diagnostic_noise_std})
    noisy = _simulate_reference_scene(noisy_config, add_noise=True, jitter_std_s=diagnostic_jitter_s)
    jitter_only = _simulate_reference_scene(config, add_noise=False, jitter_std_s=diagnostic_jitter_s)
    time_ms = _time_ms(clean.receive.shape[-1], config.sample_rate_hz)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_ms, _to_numpy(clean.receive[0, 0]), color="#0f172a", linewidth=1.0)
    axes[0].set_title("Clean echo, no noise and no jitter")
    axes[1].plot(time_ms, _to_numpy(noisy.receive[0, 0]), color="#b45309", linewidth=1.0)
    axes[1].set_title(f"Noise at target SNR {target_snr_db:.0f} dB + diagnostic jitter")
    axes[2].plot(time_ms, _to_numpy(jitter_only.receive[0, 0]), color="#7c3aed", linewidth=1.0)
    axes[2].set_title("Diagnostic jitter only")
    axes[2].set_xlabel("time (ms)")
    for ax in axes:
        ax.set_ylabel("left amp.")
        ax.set_xlim(0.0, min(time_ms[-1], 24.0))
    output = save_figure(fig, path)
    _plot_noise_and_jitter.last_summary = {
        "target_snr_db": target_snr_db,
        "diagnostic_noise_std": diagnostic_noise_std,
        "diagnostic_jitter_s": diagnostic_jitter_s,
        "signal_rms_active_window": float(signal_rms.item()),
    }
    return output


def _power_spectrum_db(signal: np.ndarray, sample_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a simple power spectrum in dB.

    Args:
        signal: One-dimensional signal.
        sample_rate_hz: Sampling rate.

    Returns:
        Tuple of frequency axis in Hz and normalized power in dB.
    """
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    power = np.abs(spectrum) ** 2
    power_db = 10.0 * np.log10(power / max(power.max(), 1e-30) + 1e-12)
    frequency_hz = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    return frequency_hz, power_db


def _plot_elevation_notch_psd(config: GlobalConfig, path: Path) -> str:
    """Plot received spectral power and the simulator gain profile for elevations.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    elevations = [-ANGLE_LIMIT_DEG, 0.0, ANGLE_LIMIT_DEG]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = ["#2563eb", "#0f172a", "#dc2626"]

    base_scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([EXAMPLE_DISTANCE_M]),
        azimuth_deg=torch.tensor([0.0]),
        elevation_deg=torch.tensor([0.0]),
        binaural=True,
        add_noise=False,
        include_elevation_cues=False,
        transmit_gain=config.transmit_gain,
    )
    base_signal = base_scene.receive[0, 0]
    frequency_hz_torch = torch.fft.rfftfreq(base_signal.numel(), d=1.0 / config.sample_rate_hz)

    for elevation, color in zip(elevations, colors, strict=True):
        gain = _moving_gaussian_notch_gain(
            config,
            torch.tensor([elevation]),
            frequency_hz_torch,
            include_slope=True,
        )
        signal = _to_numpy(_apply_frequency_gain(base_signal, gain[0]))
        freq_hz, power_db = _power_spectrum_db(signal, config.sample_rate_hz)
        axes[0].plot(freq_hz / 1_000.0, power_db, color=color, linewidth=1.5, label=f"{elevation:+.0f} deg")

        gain_db = 20.0 * np.log10(_to_numpy(gain[0]).clip(1e-8))
        axes[1].plot(freq_hz / 1_000.0, gain_db, color=color, linewidth=1.5, label=f"{elevation:+.0f} deg")

    for ax in axes:
        ax.set_xlim(0.0, config.cochlea_high_hz / 1_000.0 * 1.15)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left")
    axes[0].set_title("Received echo spectral power: elevation notch is visible")
    axes[0].set_ylabel("normalized power (dB)")
    axes[1].set_title("Applied elevation spectral gain profile")
    axes[1].set_ylabel("gain (dB)")
    axes[1].set_xlabel("frequency (kHz)")
    return save_figure(fig, path)


def _plot_elevation_gain_contour(config: GlobalConfig, path: Path) -> str:
    """Plot elevation cue gain as a frequency/elevation contour map.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    frequency_hz = torch.fft.rfftfreq(config.signal_samples, d=1.0 / config.sample_rate_hz)
    elevations = torch.linspace(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG, 241)
    gain = _moving_gaussian_notch_gain(
        config,
        elevations,
        frequency_hz,
        include_slope=True,
    )
    gain_db = 20.0 * np.log10(_to_numpy(gain).clip(1e-8))
    frequency_khz = _to_numpy(frequency_hz) / 1_000.0
    elevations_np = _to_numpy(elevations)
    visible = (frequency_khz >= 0.0) & (frequency_khz <= config.cochlea_high_hz / 1_000.0 * 1.15)

    fig, ax = plt.subplots(figsize=(11, 6))
    contour = ax.contourf(
        frequency_khz[visible],
        elevations_np,
        gain_db[:, visible],
        levels=40,
        cmap="viridis",
    )
    ax.set_title("Elevation spectral gain contour")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("elevation (deg)")
    ax.grid(True, alpha=0.15)
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("gain (dB)")
    return save_figure(fig, path)


def _plot_butterworth_notch_model(config: GlobalConfig, path: Path) -> str:
    """Plot the proposed Butterworth notch elevation cue.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    frequency_hz = torch.fft.rfftfreq(config.signal_samples, d=1.0 / config.sample_rate_hz)
    frequency_khz = _to_numpy(frequency_hz) / 1_000.0
    visible = (frequency_khz >= config.cochlea_low_hz / 1_000.0) & (
        frequency_khz <= config.cochlea_high_hz / 1_000.0
    )
    example_elevations = torch.tensor([-ANGLE_LIMIT_DEG, 0.0, ANGLE_LIMIT_DEG])
    contour_elevations = torch.linspace(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG, 241)
    example_gain = _butterworth_notch_gain(config, example_elevations, frequency_hz)
    contour_gain = _butterworth_notch_gain(config, contour_elevations, frequency_hz)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for elevation, gain, color in zip(
        _to_numpy(example_elevations),
        _to_numpy(example_gain),
        ["#2563eb", "#0f172a", "#dc2626"],
        strict=True,
    ):
        axes[0].plot(frequency_khz[visible], 20.0 * np.log10(gain[visible]), color=color, label=f"{elevation:+.0f} deg")
    axes[0].set_title("Proposed elevation cue: inverted Butterworth band-pass notch")
    axes[0].set_ylabel("gain (dB)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower left")

    contour = axes[1].contourf(
        frequency_khz[visible],
        _to_numpy(contour_elevations),
        20.0 * np.log10(_to_numpy(contour_gain)[:, visible]),
        levels=40,
        cmap="viridis",
    )
    axes[1].set_title("Butterworth notch sweep over +/-45 deg elevation")
    axes[1].set_xlabel("frequency (kHz)")
    axes[1].set_ylabel("elevation (deg)")
    colorbar = fig.colorbar(contour, ax=axes[1])
    colorbar.set_label("gain (dB)")
    return save_figure(fig, path)


def _plot_comb_interference_notch_model(config: GlobalConfig, path: Path) -> str:
    """Plot the proposed elevation-controlled comb-interference cue.

    Args:
        config: Acoustic configuration.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    frequency_hz = torch.fft.rfftfreq(config.signal_samples, d=1.0 / config.sample_rate_hz)
    frequency_khz = _to_numpy(frequency_hz) / 1_000.0
    visible = (frequency_khz >= config.cochlea_low_hz / 1_000.0) & (
        frequency_khz <= config.cochlea_high_hz / 1_000.0
    )
    example_elevations = torch.tensor([-ANGLE_LIMIT_DEG, 0.0, ANGLE_LIMIT_DEG])
    contour_elevations = torch.linspace(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG, 241)
    example_gain, example_lag_s = _comb_interference_gain(config, example_elevations, frequency_hz)
    contour_gain, contour_lag_s = _comb_interference_gain(config, contour_elevations, frequency_hz)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    for elevation, lag_s, gain, color in zip(
        _to_numpy(example_elevations),
        _to_numpy(example_lag_s),
        _to_numpy(example_gain),
        ["#2563eb", "#0f172a", "#dc2626"],
        strict=True,
    ):
        axes[0].plot(
            frequency_khz[visible],
            20.0 * np.log10(gain[visible]),
            color=color,
            label=f"{elevation:+.0f} deg, lag={lag_s * 1e6:.1f} us",
        )
    axes[0].set_title("Proposed elevation cue: comb filtering from delayed self-interference")
    axes[0].set_ylabel("gain (dB)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower left")

    contour = axes[1].contourf(
        frequency_khz[visible],
        _to_numpy(contour_elevations),
        20.0 * np.log10(_to_numpy(contour_gain)[:, visible]),
        levels=40,
        cmap="viridis",
    )
    axes[1].set_title("Comb notch sweep over +/-45 deg elevation")
    axes[1].set_ylabel("elevation (deg)")
    colorbar = fig.colorbar(contour, ax=axes[1])
    colorbar.set_label("gain (dB)")

    axes[2].plot(_to_numpy(contour_elevations), _to_numpy(contour_lag_s) * 1e6, color="#0f172a")
    axes[2].set_title("Elevation-controlled lag")
    axes[2].set_xlabel("elevation (deg)")
    axes[2].set_ylabel("lag (us)")
    axes[2].grid(True, alpha=0.25)
    return save_figure(fig, path)


def _write_report(artifacts: dict[str, str], results: dict[str, object], elapsed_s: float) -> None:
    """Write the signal-analysis markdown report.

    Args:
        artifacts: Mapping from artifact names to saved figure paths.
        results: JSON-serializable result dictionary.
        elapsed_s: Wall-clock runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mini Model 2: Signal Analysis",
        "",
        "This mini model visualizes the acoustic signal before neural processing. It is intended to justify the cues later used by the distance, azimuth, and elevation pathways.",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{results['config']['sample_rate_hz']} Hz` |",
        f"| chirp | `{results['config']['chirp_start_hz']} -> {results['config']['chirp_end_hz']} Hz` |",
        f"| chirp duration | `{results['config']['chirp_duration_s']} s` |",
        f"| signal duration | `{results['config']['signal_duration_s']} s` |",
        f"| azimuth limit for mini sweeps | `+/-{results['limits']['azimuth_deg']} deg` |",
        f"| elevation limit for mini sweeps | `+/-{results['limits']['elevation_deg']} deg` |",
        f"| notch centre sweep | `{results['limits']['notch_sweep_low_hz']} -> {results['limits']['notch_sweep_high_hz']} Hz` |",
        f"| example distance | `{results['scene']['distance_m']} m` |",
        f"| example azimuth | `{results['scene']['azimuth_deg']} deg` |",
        f"| example elevation | `{results['scene']['elevation_deg']} deg` |",
        "",
        "## 1. Emitted Call",
        "",
        "The emitted call is a Hann-windowed FM chirp:",
        "",
        "```text",
        "f(t) = f_start + (f_end - f_start) * t / T",
        "phase(t) = 2*pi * (f_start * t + 0.5 * sweep_rate * t^2)",
        "call(t) = sin(phase(t)) * Hann(t)",
        "```",
        "",
        "![Emitted call spectrogram](../outputs/signal_analysis/figures/emitted_call_spectrogram.png)",
        "",
        "## 2. Emitted And Received Amplitude",
        "",
        "The echo is delayed by propagation time and reduced by inverse-square attenuation:",
        "",
        "```text",
        "delay = path_length / speed_of_sound + jitter",
        "amplitude = 0.7 / max(path_length^2, 0.25)",
        "```",
        "",
        "![Emit receive amplitude](../outputs/signal_analysis/figures/emitted_received_amplitude.png)",
        "",
        "## 3. Binaural Attenuation And Head Shadow",
        "",
        "For binaural reception, the target-to-ear return distance differs slightly between ears. The head-shadow term then applies an azimuth-dependent gain:",
        "",
        "```text",
        "head_shadow_ear = exp(head_shadow_strength * sin(azimuth) * ear_sign)",
        "effective_gain_ear = attenuation_ear * head_shadow_ear",
        "```",
        "",
        "![Binaural head shadow](../outputs/signal_analysis/figures/binaural_head_shadow.png)",
        "",
        "## 4. Noise And Jitter",
        "",
        "Noise is additive Gaussian receiver noise. Jitter perturbs the propagation delay before fractional delay is applied. The main reference plots above use a clean scene so the cue structure is visible; this panel explicitly compares clean, noisy, and jittered versions.",
        "",
        f"For visibility, this diagnostic panel targets `{results['noise_jitter_diagnostic']['target_snr_db']:.1f} dB` SNR over the active echo window, giving `noise_std = {results['noise_jitter_diagnostic']['diagnostic_noise_std']:.6g}` and `jitter_std = {results['noise_jitter_diagnostic']['diagnostic_jitter_s']:.6g} s`. The base simulator default remains `noise_std = {results['config']['noise_std']}` and `jitter_std = {results['config']['jitter_std_s']} s`.",
        "",
        "```text",
        "delay = path_length / speed_of_sound + Normal(0, jitter_std)",
        "receive = echo + Normal(0, noise_std)",
        "```",
        "",
        "![Noise and jitter](../outputs/signal_analysis/figures/noise_and_jitter.png)",
        "",
        "## 5. Elevation Spectral Notch",
        "",
        "For the mini-model analysis, the current elevation cue is shown over `-45` to `+45 deg`. The notch centre is mapped from `4 kHz` to `16 kHz`, giving a `2 kHz` buffer inside the emitted `2 kHz -> 18 kHz` FM sweep. This keeps the old model untouched, but makes the proposed mini-model range explicit.",
        "",
        "```text",
        "elevation_scale = clip(elevation_deg / 45, -1, 1)",
        "notch_center_hz = 4000 + (16000 - 4000) * 0.5 * (elevation_scale + 1)",
        "notch_center = (notch_center_hz - cochlea_low_hz) / (cochlea_high_hz - cochlea_low_hz)",
        "slope_gain = exp(slope_strength * elevation_scale * (freq_norm - 0.5))",
        "gain *= exp(-notch_strength * exp(-0.5 * ((freq_norm - notch_center) / notch_width)^2))",
        "```",
        "",
        "![Elevation notch PSD](../outputs/signal_analysis/figures/elevation_notch_psd.png)",
        "",
        "The contour plot below shows the same elevation cue directly as applied gain, with frequency on the x-axis, elevation on the y-axis, and gain in dB as colour.",
        "",
        "![Elevation gain contour](../outputs/signal_analysis/figures/elevation_gain_contour.png)",
        "",
        "## 6. Proposed Elevation Model A: Butterworth Notch",
        "",
        "This proposed cue removes the broad slope and creates the notch as an inverted Butterworth band-pass template. A true band-pass has maximum response at the centre frequency; subtracting it from one gives a band-stop/notch response.",
        "",
        "```text",
        "x = abs((freq_norm - notch_center) / notch_width)",
        "bandpass = 1 / sqrt(1 + x^(2 * order))",
        "gain = 1 - depth * bandpass",
        "```",
        "",
        "Compared with the Gaussian notch, increasing the Butterworth order makes the notch flatter at the bottom and sharper at the edges. That is useful if we want a cleaner missing-band cue for a disinhibitory elevation detector.",
        "",
        "![Butterworth notch model](../outputs/signal_analysis/figures/butterworth_notch_model.png)",
        "",
        "## 7. Proposed Elevation Model B: Comb-Interference Notches",
        "",
        "This proposed cue also removes the broad slope. The signal is mixed with a slightly delayed copy of itself. Frequency components whose phase is opposite between the direct and delayed copy cancel, producing a comb of notches.",
        "",
        "```text",
        "y(t) = x(t) + alpha * x(t - tau)",
        "H(f) = 1 + alpha * exp(-j * 2*pi*f*tau)",
        "|H(f)| = sqrt(1 + alpha^2 + 2*alpha*cos(2*pi*f*tau))",
        "first_notch = 1 / (2 * tau)",
        "tau(elevation) = 1 / (2 * first_notch_frequency(elevation))",
        "```",
        "",
        "Changing elevation changes the lag, which moves the comb notches across frequency. This is attractive because it produces multiple notches rather than a single handcrafted notch, but it is also more periodic and could create ambiguities if several elevations produce similar notch patterns.",
        "",
        "![Comb interference notch model](../outputs/signal_analysis/figures/comb_interference_notch_model.png)",
        "",
        "## Interpretation",
        "",
        "- Distance is visible as a delay between emitted and received waveforms.",
        "- Azimuth is visible as small timing differences and larger level differences between the two ears.",
        "- Elevation is visible as a frequency-dependent spectral notch over the current `+/-45 deg` mini-model range.",
        "- The Butterworth version makes the single moving notch sharper and less Gaussian.",
        "- The comb-interference version creates several notches from one physical lag parameter.",
        "- Noise and jitter provide the simplest robustness tests for later mini models.",
        "- These plots support using an efference-copy template for timing comparisons rather than treating the emitted call as a literal second cochlear input.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the signal-analysis mini experiment.

    Returns:
        JSON-serializable dictionary containing numeric results and artifact
        paths.
    """
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPORT_PATH.parent)

    torch.manual_seed(7)
    config = moving_notch_signal_config(GlobalConfig())
    scene = _simulate_reference_scene(config, add_noise=False, jitter_std_s=0.0)

    artifacts = {
        "emitted_call_spectrogram": _plot_emitted_spectrogram(config, FIGURE_DIR / "emitted_call_spectrogram.png"),
        "emitted_received_amplitude": _plot_emit_receive(config, scene, FIGURE_DIR / "emitted_received_amplitude.png"),
        "binaural_head_shadow": _plot_binaural_head_shadow(config, scene, FIGURE_DIR / "binaural_head_shadow.png"),
        "noise_and_jitter": _plot_noise_and_jitter(config, FIGURE_DIR / "noise_and_jitter.png"),
        "elevation_notch_psd": _plot_elevation_notch_psd(config, FIGURE_DIR / "elevation_notch_psd.png"),
        "elevation_gain_contour": _plot_elevation_gain_contour(config, FIGURE_DIR / "elevation_gain_contour.png"),
        "butterworth_notch_model": _plot_butterworth_notch_model(
            config,
            FIGURE_DIR / "butterworth_notch_model.png",
        ),
        "comb_interference_notch_model": _plot_comb_interference_notch_model(
            config,
            FIGURE_DIR / "comb_interference_notch_model.png",
        ),
    }

    delays_ms = (_to_numpy(scene.delays_s[0]) * 1_000.0).tolist()
    amplitudes = _to_numpy(scene.amplitudes[0]).tolist()
    noise_jitter_diagnostic = getattr(_plot_noise_and_jitter, "last_summary", {})
    elapsed_s = time.perf_counter() - start
    results: dict[str, object] = {
        "experiment": "signal_analysis",
        "elapsed_seconds": elapsed_s,
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "chirp_start_hz": config.chirp_start_hz,
            "chirp_end_hz": config.chirp_end_hz,
            "chirp_duration_s": config.chirp_duration_s,
            "signal_duration_s": config.signal_duration_s,
            "noise_std": config.noise_std,
            "jitter_std_s": config.jitter_std_s,
            "head_shadow_strength": config.head_shadow_strength,
            "elevation_notch_strength": config.elevation_notch_strength,
            "elevation_notch_width": config.elevation_notch_width,
        },
        "scene": {
            "distance_m": float(scene.radii_m[0].item()),
            "azimuth_deg": float(scene.azimuth_deg[0].item()),
            "elevation_deg": float(scene.elevation_deg[0].item()),
            "delays_ms": delays_ms,
            "amplitudes": amplitudes,
            "itd_ms": None if scene.itd_s is None else float(scene.itd_s[0].item() * 1_000.0),
            "ild_db": None if scene.ild_db is None else float(scene.ild_db[0].item()),
        },
        "limits": {
            "azimuth_deg": ANGLE_LIMIT_DEG,
            "elevation_deg": ANGLE_LIMIT_DEG,
            "notch_sweep_low_hz": NOTCH_SWEEP_LOW_HZ,
            "notch_sweep_high_hz": NOTCH_SWEEP_HIGH_HZ,
        },
        "noise_jitter_diagnostic": noise_jitter_diagnostic,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(artifacts, results, elapsed_s)
    return results


if __name__ == "__main__":
    output = main()
    print(REPORT_PATH)
