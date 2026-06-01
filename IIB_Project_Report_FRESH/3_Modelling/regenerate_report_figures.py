from __future__ import annotations

"""Regenerate report-ready modelling figures from the project code."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import distance_noise_diagnostics as dnd
from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from mini_models.experiments import signal_analysis as sig
from models.acoustics import generate_fm_chirp, pad_signal


REPORT_DIR = ROOT / "IIB_Project_Report" / "3_Modelling"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 480,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_modelled_call(path: Path) -> None:
    config = sig.moving_notch_signal_config(sig.GlobalConfig())
    chirp, _ = generate_fm_chirp(config, batch_size=1, device=torch.device("cpu"), transmit_gain=1.0)
    waveform = pad_signal(chirp, config.signal_samples)[0].detach().cpu().numpy()
    time_ms = np.arange(waveform.size, dtype=np.float64) / config.sample_rate_hz * 1_000.0

    duration_ms = config.chirp_duration_s * 1_000.0
    display_time_ms = 4.0
    display_frequency_khz = 20.0
    chirp_time_ms = np.linspace(0.0, display_time_ms, 340)
    chirp_time_s = chirp_time_ms / 1_000.0
    frequency_hz = np.linspace(0.0, display_frequency_khz * 1_000.0, 440)
    sweep_rate = (config.chirp_end_hz - config.chirp_start_hz) / config.chirp_duration_s
    instantaneous_hz = config.chirp_start_hz + sweep_rate * chirp_time_s
    active = chirp_time_ms <= duration_ms
    active_phase = np.clip(chirp_time_ms / max(duration_ms, 1e-9), 0.0, 1.0)
    envelope = (np.sin(np.pi * active_phase) ** 2) * active.astype(np.float64)
    energy = np.exp(-0.5 * ((frequency_hz[:, None] - instantaneous_hz[None, :]) / 420.0) ** 2)
    energy *= envelope[None, :] ** 0.45
    energy_db = 20.0 * np.log10(np.maximum(energy, 1e-5))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.5), gridspec_kw={"width_ratios": [1.0, 1.08]})
    axes[0].plot(time_ms, waveform, color="#111827", linewidth=1.1)
    axes[0].set_xlim(0.0, 4.0)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Emitted FM call")
    axes[0].grid(True, alpha=0.18)

    axes[1].imshow(
        energy_db,
        extent=(0.0, display_time_ms, frequency_hz[0] / 1_000.0, frequency_hz[-1] / 1_000.0),
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=-45.0,
        vmax=0.0,
        interpolation="bicubic",
    )
    axes[1].plot(chirp_time_ms[active], instantaneous_hz[active] / 1_000.0, color="#e0f2fe", linewidth=1.3)
    axes[1].set_xlim(0.0, 4.0)
    axes[1].set_ylim(0.0, 20.0)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("frequency (kHz)")
    axes[1].set_title("Display-enhanced sweep ridge")
    axes[1].grid(color="white", alpha=0.14, linewidth=0.6)
    fig.tight_layout()
    save(fig, path)


def plot_comb_filter(path: Path) -> None:
    config = sig.moving_notch_signal_config(sig.GlobalConfig())
    frequency_hz = torch.fft.rfftfreq(config.signal_samples, d=1.0 / config.sample_rate_hz)
    frequency_khz = frequency_hz.detach().cpu().numpy() / 1_000.0
    visible = (frequency_khz >= config.cochlea_low_hz / 1_000.0) & (
        frequency_khz <= config.cochlea_high_hz / 1_000.0
    )
    example_elevations = torch.tensor([-45.0, 0.0, 45.0])
    contour_elevations = torch.linspace(-45.0, 45.0, 241)
    example_gain, example_lag_s = sig._comb_interference_gain(config, example_elevations, frequency_hz)
    contour_gain, _ = sig._comb_interference_gain(config, contour_elevations, frequency_hz)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9), gridspec_kw={"width_ratios": [1.0, 1.12]})
    for elevation, lag_s, gain, color in zip(
        example_elevations.detach().cpu().numpy(),
        example_lag_s.detach().cpu().numpy(),
        example_gain.detach().cpu().numpy(),
        ["#2563eb", "#111827", "#dc2626"],
        strict=True,
    ):
        axes[0].plot(
            frequency_khz[visible],
            20.0 * np.log10(gain[visible]),
            color=color,
            linewidth=2.0,
            label=f"{elevation:+.0f} deg, {lag_s * 1e6:.1f} us",
        )
    axes[0].set_xlabel("frequency (kHz)")
    axes[0].set_ylabel("gain (dB)")
    axes[0].set_title("Elevation-dependent comb-filter cue")
    axes[0].legend(loc="lower left", frameon=True)
    axes[0].grid(True, alpha=0.25)

    image = axes[1].contourf(
        frequency_khz[visible],
        contour_elevations.detach().cpu().numpy(),
        20.0 * np.log10(contour_gain.detach().cpu().numpy()[:, visible]),
        levels=36,
        cmap="viridis",
    )
    axes[1].set_xlabel("frequency (kHz)")
    axes[1].set_ylabel("elevation (deg)")
    axes[1].set_title("Moving notch across candidate elevations")
    colorbar = fig.colorbar(image, ax=axes[1], pad=0.02)
    colorbar.set_label("gain (dB)")
    fig.tight_layout()
    save(fig, path)


def _plot_raster(ax: plt.Axes, raster: np.ndarray, centres_khz: np.ndarray, color: str) -> None:
    for channel, frequency_khz in enumerate(centres_khz):
        event_times = np.flatnonzero(raster[channel] > 0.0) / 64_000.0 * 1_000.0
        if event_times.size:
            ax.vlines(event_times, frequency_khz * 0.985, frequency_khz * 1.015, color=color, linewidth=0.8)


def plot_adaptive_cochlea(path: Path) -> None:
    clean_config = dnd._make_config()
    noisy_config = dnd._make_noisy_config(clean_config)
    schedule = dnd._schedule_by_name("dyn_G_x16_to_x2p5_beta0p2_to_0p60")
    distance_m = 5.0

    torch.manual_seed(dnd.RNG_SEED + 60_000 + int(distance_m * 100))
    receive = dnd._simulate_scene(noisy_config, distance_m, add_noise=True)
    static_config = replace(
        noisy_config,
        spike_threshold=float(noisy_config.spike_threshold) * dnd.BETA_SWEEP_THRESHOLD_MULTIPLIER,
        spike_beta=0.50,
    )
    static_cochlea = dnd._run_cochlea_binaural(static_config, receive)
    dynamic_cochlea = dnd._run_cochlea_binaural(noisy_config, receive)
    cochleagram = torch.maximum(dynamic_cochlea.left_cochleagram, dynamic_cochlea.right_cochleagram).detach().cpu().numpy()
    dynamic_spikes = dnd._dynamic_lif_encode(cochleagram, noisy_config, schedule)
    static_spikes = torch.maximum(static_cochlea.left_spikes, static_cochlea.right_spikes).detach().cpu().numpy()
    centres_khz = dnd._log_spaced_centers(noisy_config).detach().cpu().numpy() / 1_000.0
    time_ms = np.arange(receive.shape[-1]) / noisy_config.sample_rate_hz * 1_000.0
    threshold_t, beta_t = fdm._dynamic_threshold_beta(noisy_config, schedule, noisy_config.signal_samples)
    echo_window, noise_window = dnd._echo_and_noise_windows(noisy_config, distance_m, receive.shape[-1])

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.8), sharex=True, gridspec_kw={"height_ratios": [0.8, 1.0, 1.0]})
    threshold_mult = threshold_t / float(noisy_config.spike_threshold)
    axes[0].plot(time_ms, threshold_mult, color="#dc2626", linewidth=2.0, label="threshold multiplier")
    axes[0].axhline(
        schedule["threshold_floor_mult"],
        color="#dc2626",
        linestyle=":",
        linewidth=1.4,
        label=f"threshold asymptote: {schedule['threshold_floor_mult']:.1f}x",
    )
    twin = axes[0].twinx()
    twin.plot(time_ms, beta_t, color="#2563eb", linewidth=2.0, label="membrane leak beta")
    twin.axhline(
        schedule["beta_end"],
        color="#2563eb",
        linestyle=":",
        linewidth=1.4,
        label=f"beta asymptote: {schedule['beta_end']:.2f}",
    )
    axes[0].set_ylabel("threshold multiplier")
    twin.set_ylabel("beta")
    axes[0].set_title("Implemented adaptive cochlear schedule")
    handles, labels = axes[0].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[0].legend(handles + twin_handles, labels + twin_labels, loc="center right", frameon=True)

    for ax, raster, title, color in [
        (axes[1], static_spikes, "Static LIF encoder: high fixed threshold, beta=0.50", "#7c3aed"),
        (axes[2], dynamic_spikes, "Adaptive LIF encoder: threshold x16 -> x2.5, beta 0.20 -> 0.60", "#0f766e"),
    ]:
        _plot_raster(ax, raster, centres_khz, color)
        ax.axvspan(
            echo_window.start / noisy_config.sample_rate_hz * 1_000.0,
            echo_window.stop / noisy_config.sample_rate_hz * 1_000.0,
            color="#16a34a",
            alpha=0.12,
        )
        ax.axvspan(
            noise_window.start / noisy_config.sample_rate_hz * 1_000.0,
            noise_window.stop / noisy_config.sample_rate_hz * 1_000.0,
            color="#dc2626",
            alpha=0.08,
        )
        ax.axhline(dnd.VCN_MIN_RESPONSIVE_KHZ, color="#111827", linestyle=":", linewidth=1.1, alpha=0.65)
        ax.set_yscale("log")
        ax.set_ylabel("frequency (kHz)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.18)
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, time_ms[-1])
    fig.tight_layout()
    save(fig, path)


def plot_ac_mexican_hat(path: Path) -> None:
    kernel = fdm._mexican_hat_kernel()
    radius = len(kernel) // 2
    matrix = np.zeros((fdm.NUM_DISTANCE_BINS, fdm.NUM_DISTANCE_BINS), dtype=np.float64)
    for row in range(fdm.NUM_DISTANCE_BINS):
        for offset, weight in enumerate(kernel):
            col = row + offset - radius
            if 0 <= col < fdm.NUM_DISTANCE_BINS:
                matrix[row, col] = weight

    distances = fdm._candidate_distances()
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    image = axes[0].imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[distances.min(), distances.max(), distances.min(), distances.max()],
    )
    axes[0].set_xlabel("source bin (m)")
    axes[0].set_ylabel("target bin (m)")
    axes[0].set_title("AC lateral matrix")
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.03)
    colorbar.set_label("weight")

    offsets = np.arange(-radius, radius + 1)
    bin_width_m = distances[1] - distances[0]
    axes[1].plot(offsets * bin_width_m, kernel, color="#111827", linewidth=2.2)
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0)
    axes[1].set_xlabel("distance offset (m)")
    axes[1].set_ylabel("weight")
    axes[1].set_title("Mexican-hat kernel")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, path)


def plot_dcn_weights(path: Path) -> None:
    config = elev.make_config()
    baseline_profile = elev.baseline_energy_profile(config)
    centres_hz, bins, weights = elev.signal_notch_weight_matrix(
        config,
        baseline_profile,
        elev.DEEP_COMB_DELAYED_COPY_GAIN,
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    image = ax.imshow(
        weights,
        aspect="auto",
        origin="lower",
        extent=[centres_hz[0] / 1_000.0, centres_hz[-1] / 1_000.0, bins[0], bins[-1]],
        cmap="viridis",
    )
    ax.plot(elev.comb_first_notch_hz(bins) / 1_000.0, bins, color="#f97316", linewidth=1.9, label="first notch")
    ax.set_xlabel("cochlear centre frequency (kHz)")
    ax.set_ylabel("candidate elevation (deg)")
    ax.set_title("Signal-weighted DCN synaptic matrix")
    ax.legend(frameon=True, loc="upper left", fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("normalised weight")
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    apply_style()
    plot_modelled_call(REPORT_DIR / "signals" / "modelled_call.png")
    plot_comb_filter(REPORT_DIR / "signals" / "comb_filter.png")
    plot_adaptive_cochlea(REPORT_DIR / "cochlea" / "schedule.png")
    plot_ac_mexican_hat(REPORT_DIR / "distance" / "AC_MH.png")
    plot_dcn_weights(REPORT_DIR / "elevation" / "weights.png")


if __name__ == "__main__":
    main()
