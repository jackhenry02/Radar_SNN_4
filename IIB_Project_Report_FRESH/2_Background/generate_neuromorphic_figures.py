from __future__ import annotations

"""Generate self-contained neuromorphic background figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "IIB_Project_Report" / "2_Background" / "background_figures" / "neuromorphic"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, bbox_inches="tight")
    plt.close(fig)


def simulate_lif(input_current: np.ndarray, beta: float = 0.94, threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    membrane = np.zeros_like(input_current, dtype=np.float64)
    spikes = np.zeros_like(input_current, dtype=np.float64)
    v = 0.0
    for i, current in enumerate(input_current):
        v = beta * v + current
        if v >= threshold:
            spikes[i] = 1.0
            v -= threshold
        membrane[i] = v
    return membrane, spikes


def plot_lif_encoding() -> None:
    t = np.linspace(0.0, 80.0, 500)
    input_current = np.zeros_like(t)
    input_current[(t >= 10.0) & (t < 35.0)] = 0.060
    input_current[(t >= 45.0) & (t < 70.0)] = 0.095
    membrane, spikes = simulate_lif(input_current)

    stimulus_levels = np.array([0.2, 0.45, 0.7, 1.0])
    latency_ms = 46.0 / (stimulus_levels + 0.12)
    rate_hz = 18.0 + 95.0 * stimulus_levels
    spike_trains = []
    for level, rate, latency in zip(stimulus_levels, rate_hz, latency_ms, strict=True):
        start = int(np.clip(latency, 8, 75))
        interval = max(5, int(1_000.0 / rate))
        spike_trains.append(np.arange(start, 120, interval))

    positions = np.linspace(-45.0, 45.0, 91)
    preferred = np.linspace(-45.0, 45.0, 19)
    true_position = 12.0
    population_rate = np.exp(-0.5 * ((preferred - true_position) / 13.0) ** 2)
    decoded = np.sum(preferred * population_rate) / np.sum(population_rate)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    ax = axes[0]
    ax.plot(t, membrane, color="#111827", linewidth=1.8, label="membrane")
    ax.plot(t, input_current / np.max(input_current), color="#2563eb", linewidth=1.4, alpha=0.8, label="input")
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.1, label="threshold")
    spike_times = t[spikes > 0.0]
    ax.vlines(spike_times, 1.06, 1.22, color="#059669", linewidth=1.5, label="spike")
    ax.set_title("LIF integration")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalised value")
    ax.set_ylim(-0.05, 1.28)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=True)

    ax = axes[1]
    for row, train in enumerate(spike_trains):
        ax.vlines(train, row + 0.12, row + 0.88, color="#7c3aed", linewidth=1.0)
    ax.set_yticks(np.arange(len(stimulus_levels)) + 0.5)
    ax.set_yticklabels([f"{level:.1f}" for level in stimulus_levels])
    ax.set_title("Spike encoding")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("input level")
    ax.set_xlim(0, 120)
    ax.grid(True, axis="x", alpha=0.2)

    ax = axes[2]
    ax.bar(preferred, population_rate, width=4.0, color="#0f766e", alpha=0.86)
    ax.axvline(true_position, color="#111827", linewidth=1.4, label="true cue")
    ax.axvline(decoded, color="#f97316", linestyle="--", linewidth=1.5, label="population mean")
    ax.set_xlim(positions.min(), positions.max())
    ax.set_title("Rate-coded population")
    ax.set_xlabel("preferred coordinate (deg)")
    ax.set_ylabel("mean activity")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    save(fig, "lif_spike_population_encoding.png")


def plot_coincidence_population() -> None:
    candidate_delays_ms = np.linspace(0.5, 10.0, 20)
    true_delay_ms = 6.4
    cd_time_ms = 1.0
    echo_time_ms = cd_time_ms + true_delay_ms
    sigma_ms = 0.45
    activation = np.exp(-0.5 * ((candidate_delays_ms - true_delay_ms) / sigma_ms) ** 2)
    activation += 0.05 * np.exp(-0.5 * ((candidate_delays_ms - 2.2) / 0.55) ** 2)
    decoded = np.sum(candidate_delays_ms * activation) / np.sum(activation)
    threshold = 0.62

    timeline = np.linspace(0.0, 11.5, 350)
    candidate_examples = [3.0, true_delay_ms, 8.5]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    ax = axes[0]
    ax.vlines(cd_time_ms, 2.1, 2.9, color="#2563eb", linewidth=2.0)
    ax.text(cd_time_ms, 3.08, "call copy", ha="center", color="#2563eb")
    ax.vlines(echo_time_ms, 0.1, 0.9, color="#dc2626", linewidth=2.0)
    ax.text(echo_time_ms, 1.08, "echo", ha="center", color="#dc2626")
    for i, delay in enumerate(candidate_examples):
        shifted = cd_time_ms + delay
        ax.arrow(cd_time_ms, 2.0 - 0.35 * i, delay, -1.0 + 0.35 * i, head_width=0.08, head_length=0.18, color="#6b7280", length_includes_head=True)
        ax.vlines(shifted, 1.35 - 0.35 * i, 1.85 - 0.35 * i, color="#6b7280", linewidth=1.6)
        ax.text(shifted, 1.22 - 0.35 * i, f"{delay:.1f} ms", ha="center", fontsize=8)
    ax.set_title("Internal delay lines")
    ax.set_xlabel("time (ms)")
    ax.set_yticks([])
    ax.set_xlim(0.0, 11.5)
    ax.set_ylim(-0.1, 3.35)
    ax.grid(True, axis="x", alpha=0.2)

    ax = axes[1]
    for delay, color, label in [
        (3.0, "#6b7280", "too short"),
        (true_delay_ms, "#059669", "matched"),
        (8.5, "#6b7280", "too long"),
    ]:
        mismatch = abs(delay - true_delay_ms)
        response = np.exp(-0.5 * ((timeline - echo_time_ms) / (0.3 + mismatch)) ** 2)
        peak = np.exp(-0.5 * (mismatch / sigma_ms) ** 2)
        ax.plot(timeline, peak * response, color=color, linewidth=1.8, label=label)
    ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="firing threshold")
    ax.set_title("Coincidence response")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("membrane response")
    ax.set_xlim(3.0, 10.5)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, loc="upper left")

    ax = axes[2]
    ax.bar(candidate_delays_ms, activation, width=0.34, color="#0f766e", alpha=0.86)
    ax.axvline(true_delay_ms, color="#111827", linewidth=1.3, label="true delay")
    ax.axvline(decoded, color="#f97316", linestyle="--", linewidth=1.5, label="COM decode")
    ax.set_title("Delay population code")
    ax.set_xlabel("candidate delay (ms)")
    ax.set_ylabel("activity")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=True, loc="upper left")

    fig.tight_layout()
    save(fig, "delay_line_coincidence_population.png")


def main() -> None:
    apply_style()
    plot_lif_encoding()
    plot_coincidence_population()


if __name__ == "__main__":
    main()
