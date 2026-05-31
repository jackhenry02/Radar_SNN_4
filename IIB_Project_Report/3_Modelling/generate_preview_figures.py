from __future__ import annotations

"""Generate candidate modelling figures without changing report assets."""

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

from azimuth_pathway.experiments import azimuth_pathway_first_attempt as az
from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from final_model.experiments import environment_noise_diagnostics as envdiag
from models.acoustics import generate_fm_chirp, pad_signal


OUTPUT_DIR = ROOT / "IIB_Project_Report" / "3_Modelling" / "preview_figures"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def make_distance_variant(config: fdm.GlobalConfig) -> fdm.PathwayVariant:
    template = fdm.PathwayVariant(
        key="preview_dynamic_distance",
        name="Preview dynamic distance pathway",
        vcn_input="spikes",
        latency_samples=np.zeros(fdm.NUM_CHANNELS, dtype=np.int64),
        dynamic_cochlea_schedule=fdm.DYNAMIC_COHLEA_SCHEDULE,
        vcn_detector="consensus",
        ic_mode="facilitated",
        note="Report-figure preview.",
    )
    latency = fdm._calibrate_variant_latency(
        config,
        template,
        calibration_distances=np.linspace(fdm.MIN_DISTANCE_M, fdm.MAX_DISTANCE_M, 8),
    )
    return replace(template, latency_samples=latency)


def plot_received_waveforms(config: fdm.GlobalConfig) -> None:
    distance_m = 2.5
    azimuth_deg = 15.0
    elevation_deg = 10.0
    target_delay_ms = 2.0 * distance_m / config.speed_of_sound_m_s * 1_000.0
    noise_std = envdiag._call_referenced_noise_std(config, envdiag.ENVIRONMENT_SNR_DB_AT_CALL)
    chirp, _ = generate_fm_chirp(
        config,
        batch_size=1,
        device=torch.device("cpu"),
        transmit_gain=config.transmit_gain,
    )
    transmit = pad_signal(chirp, config.signal_samples)[0].detach().cpu().numpy()
    time_ms = np.arange(config.signal_samples) / config.sample_rate_hz * 1_000.0

    received: list[tuple[str, np.ndarray, str]] = []
    for condition, color in zip(
        envdiag.CONDITIONS,
        ["#2563eb", "#d97706", "#7c3aed"],
        strict=True,
    ):
        generator = torch.Generator(device="cpu").manual_seed(12_503)
        waveform = envdiag._simulate_environment_echo(
            config,
            distance_m,
            azimuth_deg,
            elevation_deg,
            condition=condition,
            noise_std=noise_std,
            rng=generator,
        )[0].detach().cpu().numpy()
        display_name = "Noise-free echo" if condition.key == "clean" else condition.name
        received.append((display_name, waveform, color))

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 7.4), sharex=True)
    axes[0].plot(time_ms, transmit, color="#111827", linewidth=0.9)
    axes[0].set_title("Emitted FM call")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(True, alpha=0.18)

    received_limit = 1.08 * max(float(np.max(np.abs(waveform))) for _, waveform, _ in received)
    late_delays = [target_delay_ms + delay for delay in envdiag.REVERB_DELAYS_MS]
    for index, (ax, (name, waveform, color)) in enumerate(zip(axes[1:], received, strict=True)):
        ax.plot(time_ms, waveform, color=color, linewidth=0.8)
        ax.axvline(target_delay_ms, color="#111827", linestyle="--", linewidth=1.0)
        if index == 2:
            for late_delay in late_delays:
                ax.axvline(late_delay, color="#6b7280", linestyle=":", linewidth=0.8)
        ax.set_ylim(-received_limit, received_limit)
        ax.set_ylabel("amplitude")
        ax.set_title(name)
        ax.grid(True, alpha=0.18)
    axes[1].text(
        target_delay_ms + 0.18,
        0.78 * received_limit,
        "dominant echo",
        fontsize=11,
        color="#111827",
    )
    axes[3].text(
        late_delays[-1] + 0.2,
        0.78 * received_limit,
        "late copies",
        fontsize=11,
        color="#4b5563",
    )
    axes[-1].set_xlim(0.0, 24.0)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(
        "Example simulated return: 2.5 m target, +15 deg azimuth, +10 deg elevation",
        fontsize=15,
    )
    fig.tight_layout()
    save(fig, "received_waveform_conditions.png")


def theoretical_fm_population(
    config: fdm.GlobalConfig,
    target_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centres_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    responsive = centres_khz >= fdm.VCN_MIN_RESPONSIVE_HZ / 1_000.0
    cd_samples = fdm._chirp_channel_times(config)[responsive]
    target_delay = int(round(2.0 * target_distance_m / config.speed_of_sound_m_s * config.sample_rate_hz))
    echo_samples = cd_samples + target_delay
    candidate_distances = fdm._candidate_distances(config)
    candidate_delays = fdm._candidate_delay_samples(config)
    delta = np.abs(echo_samples[:, None] - (cd_samples[:, None] + candidate_delays[None, :]))
    channel_evidence = np.maximum(
        0.0,
        1.0 + np.power(fdm.IC_LIF_BETA, delta) - fdm.IC_LIF_THRESHOLD,
    )
    return centres_khz[responsive], cd_samples, echo_samples, candidate_distances, channel_evidence


def plot_fm_coincidence_population(config: fdm.GlobalConfig) -> None:
    target_distance_m = 2.0
    centres_khz, cd_samples, echo_samples, candidate_distances, evidence = theoretical_fm_population(
        config,
        target_distance_m,
    )
    cd_ms = cd_samples / config.sample_rate_hz * 1_000.0
    echo_ms = echo_samples / config.sample_rate_hz * 1_000.0
    population = evidence.sum(axis=0)
    population /= max(float(population.max()), 1e-12)
    example_candidates = [1.2, target_distance_m, 2.8]
    candidate_colors = ["#d97706", "#059669", "#7c3aed"]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.0), gridspec_kw={"height_ratios": [1.3, 1.0, 1.0]})
    axes[0].scatter(cd_ms, centres_khz, marker="|", s=105, linewidths=1.8, color="#2563eb", label="corollary discharge")
    axes[0].scatter(echo_ms, centres_khz, marker="|", s=105, linewidths=1.8, color="#dc2626", label="echo onset")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_title("FM sweep represented by one onset event per cochlear channel")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(True, alpha=0.18)

    representative_channel = len(centres_khz) // 2
    echo_time_ms = echo_ms[representative_channel]
    axes[1].axvline(echo_time_ms, color="#dc2626", linestyle="--", linewidth=1.3, label="echo onset")
    for row, (candidate, color) in enumerate(zip(example_candidates, candidate_colors, strict=True)):
        candidate_delay_ms = 2.0 * candidate / config.speed_of_sound_m_s * 1_000.0
        shifted_cd_ms = cd_ms[representative_channel] + candidate_delay_ms
        axes[1].scatter(shifted_cd_ms, row, marker="|", s=230, linewidths=2.5, color=color)
        axes[1].hlines(row, min(shifted_cd_ms, echo_time_ms), max(shifted_cd_ms, echo_time_ms), color=color, linewidth=2.0)
    axes[1].set_yticks(range(len(example_candidates)), [f"{candidate:.1f} m candidate" for candidate in example_candidates])
    axes[1].set_xlabel("time (ms)")
    axes[1].set_title("Candidate-specific delayed copies: matched timing gives strongest overlap")
    axes[1].grid(True, axis="x", alpha=0.18)
    axes[1].legend(frameon=False, loc="lower right")

    axes[2].plot(candidate_distances, population, color="#2563eb", linewidth=2.2)
    axes[2].axvline(target_distance_m, color="#111827", linestyle="--", linewidth=1.1, label="true distance")
    for candidate, color in zip(example_candidates, candidate_colors, strict=True):
        axes[2].axvline(candidate, color=color, linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("candidate distance (m)")
    axes[2].set_ylabel("normalised evidence")
    axes[2].set_title("Evidence accumulated across FM channels forms a distance population")
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.22)
    fig.tight_layout()
    save(fig, "fm_coincidence_population_walkthrough.png")


def activity_window_ms(raster: np.ndarray, config: fdm.GlobalConfig, margin_ms: float = 0.45) -> tuple[float, float]:
    events = np.flatnonzero(raster.sum(axis=0) > 0.0)
    if events.size == 0:
        return 0.0, 1.0
    start = events[0] / config.sample_rate_hz * 1_000.0 - margin_ms
    end = events[-1] / config.sample_rate_hz * 1_000.0 + margin_ms
    return max(start, 0.0), min(end, config.signal_duration_s * 1_000.0)


def draw_raster(
    ax: plt.Axes,
    raster: np.ndarray,
    centres_khz: np.ndarray,
    color: str,
    config: fdm.GlobalConfig,
) -> None:
    for channel, frequency_khz in enumerate(centres_khz):
        event_times = np.flatnonzero(raster[channel] > 0.0) / config.sample_rate_hz * 1_000.0
        if event_times.size:
            ax.vlines(event_times, frequency_khz - 0.15, frequency_khz + 0.15, color=color, linewidth=0.9)


def plot_condensed_raster_flow(
    config: fdm.GlobalConfig,
    variant: fdm.PathwayVariant,
    prediction: fdm.PathwayPrediction,
) -> None:
    centres_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    left_dynamic = fdm._dynamic_lif_encode(
        prediction.cochlea.left_cochleagram,
        config,
        variant.dynamic_cochlea_schedule,
    )
    right_dynamic = fdm._dynamic_lif_encode(
        prediction.cochlea.right_cochleagram,
        config,
        variant.dynamic_cochlea_schedule,
    )
    cochlea = torch.maximum(left_dynamic, right_dynamic).detach().cpu().numpy()
    stages = [
        ("Corollary-discharge\nreference", prediction.cd_raster, "#2563eb"),
        ("Cochlear spike\nraster", cochlea, "#7c3aed"),
        ("VCN/VNLL onset\nextraction", np.maximum(prediction.vcn_left, prediction.vcn_right), "#d97706"),
        ("DNLL-gated onset\nraster", prediction.dnll_combined, "#059669"),
    ]
    windows = [activity_window_ms(raster, config) for _, raster, _ in stages]

    fig = plt.figure(figsize=(12.0, 5.3))
    grid = fig.add_gridspec(2, 4, height_ratios=[0.72, 2.2], hspace=0.78, wspace=0.42)
    timeline = fig.add_subplot(grid[0, :])
    timeline.set_xlim(0.0, config.signal_duration_s * 1_000.0)
    timeline.set_ylim(-0.65, 3.65)
    timeline.set_yticks(range(len(stages)), [title.replace("\n", " ") for title, _, _ in stages])
    timeline.set_xlabel("absolute time in trial (ms)")
    timeline.set_title("Overview timeline: cut-outs below retain their absolute time axes")
    for index, ((title, _, color), (start_ms, end_ms)) in enumerate(zip(stages, windows, strict=True)):
        timeline.add_patch(
            plt.Rectangle(
                (start_ms, index - 0.22),
                max(end_ms - start_ms, 0.1),
                0.44,
                facecolor=color,
                edgecolor=color,
                alpha=0.28,
            )
        )
    timeline.grid(True, axis="x", alpha=0.18)

    for index, ((title, raster, color), (start_ms, end_ms)) in enumerate(zip(stages, windows, strict=True)):
        ax = fig.add_subplot(grid[1, index])
        draw_raster(ax, raster, centres_khz, color, config)
        ax.axhline(fdm.VCN_MIN_RESPONSIVE_HZ / 1_000.0, color="#6b7280", linestyle=":", linewidth=0.9)
        ax.set_xlim(start_ms, end_ms)
        ax.set_ylim(3.5, 20.2)
        ax.set_xlabel("absolute time (ms)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.18)
        if index == 0:
            ax.set_ylabel("frequency (kHz)")
        else:
            ax.set_yticklabels([])
        if index < len(stages) - 1:
            ax.text(
                1.10,
                0.52,
                r"$\rightarrow$",
                transform=ax.transAxes,
                fontsize=19,
                ha="center",
                va="center",
                color="#374151",
            )
    fig.suptitle(
        f"Condensed pathway cut-outs for a {prediction.distance_m:.1f} m target",
        fontsize=15,
    )
    fig.subplots_adjust(top=0.87)
    save(fig, "condensed_raster_flow.png")


def plot_condensed_population_flow(
    config: fdm.GlobalConfig,
    prediction: fdm.PathwayPrediction,
) -> None:
    distances = fdm._candidate_distances(config)
    ic = prediction.ic_activation / max(float(np.max(prediction.ic_activation)), 1e-12)
    ac = prediction.ac_activation / max(float(np.max(prediction.ac_activation)), 1e-12)
    decoded_distance = fdm._sc_center_of_mass(prediction.ac_activation, config)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), sharey=True)
    axes[0].plot(distances, ic, color="#2563eb", linewidth=2.2)
    axes[0].fill_between(distances, 0.0, ic, color="#2563eb", alpha=0.12)
    axes[0].axvline(prediction.distance_m, color="#111827", linestyle="--", linewidth=1.1, label="true distance")
    axes[0].set_xlabel("candidate distance (m)")
    axes[0].set_ylabel("normalised population activity")
    axes[0].set_title("IC coincidence population")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(distances, ac, color="#059669", linewidth=2.2)
    axes[1].fill_between(distances, 0.0, ac, color="#059669", alpha=0.12)
    axes[1].axvline(prediction.distance_m, color="#111827", linestyle="--", linewidth=1.1, label="true distance")
    axes[1].axvline(decoded_distance, color="#dc2626", linestyle=":", linewidth=1.3, label="COM readout")
    axes[1].set_xlabel("candidate distance (m)")
    axes[1].set_title("AC population after lateral sharpening")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(True, alpha=0.22)

    axes[0].text(
        1.08,
        0.53,
        r"$\rightarrow$",
        transform=axes[0].transAxes,
        fontsize=22,
        ha="center",
        va="center",
        color="#374151",
    )
    fig.suptitle(
        f"Population formation for a {prediction.distance_m:.1f} m target",
        fontsize=15,
    )
    fig.tight_layout(w_pad=3.0)
    save(fig, "condensed_ic_ac_population_flow.png")


def plot_itd_mechanics() -> None:
    config = az.make_config()
    prediction = az.predict_one(config, 10.0)
    bins = az.azimuth_grid(az.AZIMUTH_LIMIT_DEG)
    centres_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1000.0
    left_first = az.first_event_times(prediction.vcn_left)
    right_first = az.first_event_times(prediction.vcn_right)
    valid = np.flatnonzero((left_first >= 0) & (right_first >= 0))
    representative = valid[np.argmin(np.abs(centres_khz[valid] - 10.0))]
    fs = float(config.sample_rate_hz)
    left_ms = left_first[representative] / fs * 1000.0
    right_ms = right_first[representative] / fs * 1000.0
    observed_us = (right_ms - left_ms) * 1000.0
    right_relative_samples = float(right_first[representative] - left_first[representative])

    active = np.flatnonzero(
        (prediction.vcn_left.sum(axis=0) + prediction.vcn_right.sum(axis=0)) > 0
    )
    crop_start = max(0.0, active.min() / fs * 1000.0 - 0.30)
    crop_end = active.max() / fs * 1000.0 + 0.30

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 3.9),
        gridspec_kw={"width_ratios": [1.55, 0.92, 1.32]},
    )
    fig.suptitle("Azimuth cue: binaural onset timing becomes an ITD population", fontsize=15)

    ax = axes[0]
    for channel, centre in enumerate(centres_khz):
        left_events = np.flatnonzero(prediction.vcn_left[channel] > 0) / fs * 1000.0
        right_events = np.flatnonzero(prediction.vcn_right[channel] > 0) / fs * 1000.0
        if left_events.size:
            ax.scatter(left_events, np.full(left_events.shape, centre), s=12, marker="|", color="#2678b8")
        if right_events.size:
            ax.scatter(right_events, np.full(right_events.shape, centre), s=12, marker="|", color="#d1495b")
    ax.set_xlim(crop_start, crop_end)
    ax.set_ylim(3.5, 19.0)
    ax.axhline(4.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_title("1. VCN onset rasters", loc="left")
    ax.set_xlabel("Time after emission (ms)")
    ax.set_ylabel("Cochlear channel centre (kHz)")
    ax.scatter([], [], marker="|", color="#2678b8", label="Left ear")
    ax.scatter([], [], marker="|", color="#d1495b", label="Right ear")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.scatter([0.0], [1.0], marker="|", s=430, linewidths=3.0, color="#2678b8")
    ax.scatter([right_relative_samples], [0.0], marker="|", s=430, linewidths=3.0, color="#d1495b")
    arrow_y = 0.50
    ax.annotate(
        "",
        xy=(0.0, arrow_y),
        xytext=(right_relative_samples, arrow_y),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.5, "color": "0.22"},
    )
    ax.text(
        right_relative_samples / 2.0,
        arrow_y + 0.09,
        rf"$\Delta n_c={right_first[representative]-left_first[representative]:+.0f}$ samples"
        "\n"
        rf"({observed_us:+.1f} $\mu$s)",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    pad = max(abs(right_relative_samples) * 0.75 + 0.85, 1.35)
    midpoint = right_relative_samples / 2.0
    ax.set_xlim(midpoint - pad, midpoint + pad)
    ax.set_ylim(-0.35, 1.35)
    ax.set_yticks([0.0, 1.0], ["Right-ear onset", "Left-ear onset"])
    ax.set_xlabel("Samples relative to left-ear onset")
    ax.set_title(f"2. One {centres_khz[representative]:.1f} kHz channel", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[2]
    population = az.normalise_population(prediction.itd_activation)
    ax.fill_between(bins, population, color="#79b7d8", alpha=0.42)
    ax.plot(bins, population, color="#2678b8", linewidth=2.2)
    ax.axvline(prediction.true_azimuth_deg, color="0.20", linestyle="--", linewidth=1.6, label="True azimuth")
    ax.axvline(
        prediction.itd_prediction_deg,
        color="#d1495b",
        linestyle=":",
        linewidth=2.0,
        label="COM readout",
    )
    ax.set_xlim(-45.0, 45.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("3. Jeffress-style population", loc="left")
    ax.set_xlabel("Candidate azimuth (degrees)")
    ax.set_ylabel("Normalised activity")
    ax.legend(loc="upper left", frameon=False)

    axes[0].text(1.045, 0.50, r"$\rightarrow$", transform=axes[0].transAxes, fontsize=23, ha="center")
    axes[1].text(1.065, 0.50, r"$\rightarrow$", transform=axes[1].transAxes, fontsize=23, ha="center")
    fig.text(
        0.5,
        0.005,
        "A physical interaural delay is compared against a bank of candidate internal delays; "
        "nearby candidates retain partial coincidence evidence.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    save(fig, "condensed_itd_mechanics.png")


def plot_elevation_mechanics() -> None:
    config = elev.make_config()
    true_elevation = -10.0
    delayed_copy_gain = elev.DEEP_COMB_DELAYED_COPY_GAIN
    baseline = elev.baseline_energy_profile(config)
    prediction = elev.predict_one(
        config,
        true_elevation,
        baseline,
        delayed_copy_gain=delayed_copy_gain,
    )
    before_binaural, after_binaural = elev.simulate_pre_post_comb_scene(
        config,
        true_elevation,
        delayed_copy_gain,
    )
    ear_index = 1 if elev.SELECTED_EAR == "right" else 0
    before = before_binaural[ear_index]
    after = after_binaural[ear_index]
    crop = elev.active_echo_window(before, after)
    freq_hz, before_psd = elev.one_sided_psd(before[crop], config)
    _, after_psd = elev.one_sided_psd(after[crop], config)
    shared_peak = max(float(before_psd.max()), float(after_psd.max()), 1e-12)
    before_db = 10.0 * np.log10(np.maximum(before_psd / shared_peak, 1e-7))
    after_db = 10.0 * np.log10(np.maximum(after_psd / shared_peak, 1e-7))
    centres_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1000.0
    first_notch_khz = float(elev.comb_first_notch_hz(true_elevation)) / 1000.0
    bins = elev.elevation_grid()
    population = prediction.signal_weighted_activation.copy()
    population /= max(float(population.max()), 1e-12)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.4, 3.9),
        gridspec_kw={"width_ratios": [1.45, 1.22, 1.15]},
    )
    fig.suptitle("Elevation cue: pinna filtering becomes a DCN spectral-template population", fontsize=15)

    ax = axes[0]
    ax.plot(freq_hz / 1000.0, before_db, color="0.45", linewidth=1.4, label="Before pinna filtering")
    ax.plot(freq_hz / 1000.0, after_db, color="#5a9f68", linewidth=2.0, label="After pinna filtering")
    ax.axvline(first_notch_khz, color="#d1495b", linestyle="--", linewidth=1.5, label="First notch")
    ax.set_xlim(2.0, 20.0)
    ax.set_ylim(-62.0, 4.0)
    ax.set_title("1. Elevation-dependent spectral notch", loc="left")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Relative spectral power (dB)")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    ax.plot(
        centres_khz,
        prediction.equalized_profile,
        color="#5a9f68",
        linewidth=2.1,
        label="Observed equalised profile",
    )
    ax.plot(
        centres_khz,
        prediction.comb_gain_channels,
        color="0.35",
        linestyle="--",
        linewidth=1.7,
        label="Matched template",
    )
    ax.axvline(first_notch_khz, color="#d1495b", linestyle=":", linewidth=1.4)
    ax.set_xlim(2.0, 20.0)
    ax.set_ylim(-0.04, 1.08)
    ax.set_title("2. Cochlear spectral profile", loc="left")
    ax.set_xlabel("Channel centre frequency (kHz)")
    ax.set_ylabel("Normalised response")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[2]
    ax.fill_between(bins, population, color="#a6cfaa", alpha=0.56)
    ax.plot(bins, population, color="#38864b", linewidth=2.2)
    ax.axvline(true_elevation, color="0.20", linestyle="--", linewidth=1.6, label="True elevation")
    ax.axvline(
        prediction.signal_weighted_com_prediction_deg,
        color="#d1495b",
        linestyle=":",
        linewidth=2.0,
        label="COM readout",
    )
    ax.set_xlim(-45.0, 45.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("3. DCN template population", loc="left")
    ax.set_xlabel("Candidate elevation (degrees)")
    ax.set_ylabel("Normalised activity")
    ax.legend(loc="upper left", frameon=False)

    axes[0].text(1.045, 0.50, r"$\rightarrow$", transform=axes[0].transAxes, fontsize=23, ha="center")
    axes[1].text(1.060, 0.50, r"$\rightarrow$", transform=axes[1].transAxes, fontsize=23, ha="center")
    fig.text(
        0.5,
        0.005,
        "The selected-ear spectrum is equalised and compared with candidate comb-filter templates; "
        "the best spectral match produces the strongest elevation evidence.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    save(fig, "condensed_elevation_mechanics.png")


def plot_single_spike_walkthrough(config: fdm.GlobalConfig) -> None:
    target_distance_m = 2.5
    target_delay_ms = 2.0 * target_distance_m / config.speed_of_sound_m_s * 1_000.0
    time_ms = np.linspace(0.0, 22.0, 2201)
    envelope_sigma_ms = 0.34
    onset_threshold = 0.28
    packet_center_ms = target_delay_ms + envelope_sigma_ms * np.sqrt(-2.0 * np.log(onset_threshold))
    carrier = np.sin(2.0 * np.pi * 2.4 * (time_ms - packet_center_ms))
    envelope = np.exp(-0.5 * ((time_ms - packet_center_ms) / envelope_sigma_ms) ** 2)
    echo_wave = carrier * envelope
    onset_candidates = np.flatnonzero(envelope >= onset_threshold)
    echo_onset_ms = float(time_ms[onset_candidates[0]])

    candidate_distances = fdm._candidate_distances(config)
    candidate_delays_ms = 2.0 * candidate_distances / config.speed_of_sound_m_s * 1_000.0
    delta_samples = np.abs(echo_onset_ms - candidate_delays_ms) * config.sample_rate_hz / 1_000.0
    population = np.maximum(
        0.0,
        1.0 + np.power(fdm.IC_LIF_BETA, delta_samples) - fdm.IC_LIF_THRESHOLD,
    )
    population /= max(float(population.max()), 1e-12)

    displayed_distances = np.array([1.0, 1.75, 2.5, 3.25, 4.0])
    displayed_delays = 2.0 * displayed_distances / config.speed_of_sound_m_s * 1_000.0
    fig, axes = plt.subplots(4, 1, figsize=(8.8, 8.0), gridspec_kw={"height_ratios": [1.0, 0.72, 1.0, 1.0]})

    axes[0].plot(time_ms, echo_wave, color="#2563eb", linewidth=1.1)
    axes[0].plot(time_ms, envelope, color="#9ca3af", linewidth=1.0, linestyle="--", label="wave envelope")
    axes[0].axhline(onset_threshold, color="#d97706", linewidth=1.0, linestyle=":", label="onset threshold")
    axes[0].axvline(echo_onset_ms, color="#dc2626", linewidth=1.1, linestyle="--")
    axes[0].set_xlim(11.5, 17.0)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("1. Received wave packet")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(True, alpha=0.18)

    axes[1].scatter([0.0], [0.68], marker="|", s=300, linewidths=3.0, color="#2563eb")
    axes[1].scatter([echo_onset_ms], [0.30], marker="|", s=300, linewidths=3.0, color="#dc2626")
    axes[1].text(0.25, 0.68, "corollary-discharge spike", va="center", fontsize=9, color="#2563eb")
    axes[1].text(echo_onset_ms + 0.25, 0.30, "extracted echo-onset spike", va="center", fontsize=9, color="#dc2626")
    axes[1].annotate(
        "",
        xy=(echo_onset_ms, 0.49),
        xytext=(0.0, 0.49),
        arrowprops=dict(arrowstyle="<->", color="#374151", linewidth=1.1),
    )
    axes[1].text(0.5 * echo_onset_ms, 0.54, "measured delay", ha="center", va="bottom", fontsize=9)
    axes[1].set_xlim(-0.5, 22.0)
    axes[1].set_ylim(0.05, 0.92)
    axes[1].set_yticks([])
    axes[1].set_title("2. Onset extraction converts the waveform into a binary timing event")
    axes[1].grid(True, axis="x", alpha=0.18)

    axes[2].axvline(echo_onset_ms, color="#dc2626", linestyle="--", linewidth=1.1, label="echo-onset spike")
    for distance_m, delay_ms in zip(displayed_distances, displayed_delays, strict=True):
        matched = bool(np.isclose(distance_m, target_distance_m))
        color = "#059669" if matched else "#6b7280"
        axes[2].scatter(delay_ms, distance_m, marker="|", s=230, linewidths=2.2, color=color)
        axes[2].hlines(distance_m, min(delay_ms, echo_onset_ms), max(delay_ms, echo_onset_ms), color=color, linewidth=1.6)
    axes[2].set_xlim(4.5, 24.0)
    axes[2].set_ylabel("candidate distance (m)")
    axes[2].set_title("3. Candidate-delay comparison units receive differently delayed reference spikes")
    axes[2].legend(frameon=False, loc="lower right")
    axes[2].grid(True, alpha=0.18)

    axes[3].plot(candidate_distances, population, color="#2563eb", linewidth=2.2)
    axes[3].axvline(target_distance_m, color="#111827", linestyle="--", linewidth=1.1, label="true distance")
    axes[3].set_xlabel("candidate distance (m)")
    axes[3].set_ylabel("normalised evidence")
    axes[3].set_title("4. Temporal overlap becomes a population code over candidate distances")
    axes[3].legend(frameon=False)
    axes[3].grid(True, alpha=0.22)
    fig.tight_layout()
    save(fig, "single_spike_coincidence_walkthrough.png")


def main() -> None:
    apply_style()
    torch.manual_seed(12_503)
    np.random.seed(12_503)
    config = fdm._make_config()
    variant = make_distance_variant(config)
    prediction = fdm._predict_one(config, 2.5, variant, add_noise=False)
    plot_received_waveforms(config)
    plot_fm_coincidence_population(config)
    plot_condensed_raster_flow(config, variant, prediction)
    plot_condensed_population_flow(config, prediction)
    plot_itd_mechanics()
    plot_elevation_mechanics()
    plot_single_spike_walkthrough(config)


if __name__ == "__main__":
    main()
