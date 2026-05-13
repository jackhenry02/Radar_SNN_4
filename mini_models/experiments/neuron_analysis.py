from __future__ import annotations

"""Generate the mini-model neuron analysis report.

This script is deliberately standalone: running it creates all neuron-analysis
figures, a JSON result file, and a markdown report under `mini_models/`.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_models.common.neurons import (
    half_peak_to_peak,
    simulate_level_crossing,
    simulate_lif,
    simulate_resonate_and_fire,
    spike_phases,
    vector_strength,
)
from mini_models.common.plotting import ensure_dir, save_figure


ROOT = PROJECT_ROOT
OUTPUT_DIR = ROOT / "mini_models" / "outputs" / "neuron_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = ROOT / "mini_models" / "reports" / "neuron_analysis.md"


def _sine(frequency_hz: float, time_s: np.ndarray, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Create a sinusoidal input signal.

    Args:
        frequency_hz: Frequency of the sine wave.
        time_s: Time axis in seconds.
        amplitude: Peak amplitude.
        phase: Phase offset in radians.

    Returns:
        Sine wave sampled on `time_s`.
    """
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time_s + phase)


def _rectified_drive(frequency_hz: float, time_s: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Create a half-wave rectified sinusoidal drive.

    Args:
        frequency_hz: Frequency of the sinusoid before rectification.
        time_s: Time axis in seconds.
        amplitude: Peak amplitude before rectification.

    Returns:
        Non-negative rectified sine wave.
    """
    return amplitude * np.maximum(0.0, np.sin(2.0 * np.pi * frequency_hz * time_s))


def _plot_lif_micro(path: Path) -> str:
    """Plot LIF response to isolated and clustered input pulses.

    The clustered pulses are included to demonstrate temporal summation:
    isolated pulses decay away, while close pulses accumulate enough membrane
    voltage to cross threshold.

    Args:
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    fs = 10_000
    dt = 1.0 / fs
    duration = 0.12
    time_s = np.arange(int(fs * duration)) * dt
    current = np.zeros_like(time_s)
    for centre_s in [0.018, 0.050, 0.054, 0.058, 0.090]:
        # Square current pulses are easy to read in the plot and make the
        # membrane integration behaviour visually obvious.
        width = int(round(0.002 * fs))
        centre = int(round(centre_s * fs))
        current[max(0, centre - width // 2) : min(current.size, centre + width // 2)] += 2.0
    trace = simulate_lif(current, dt_s=dt, tau_s=0.014, threshold=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(trace.time_s * 1_000, trace.input_current, color="#334155")
    axes[0].set_ylabel("input")
    axes[0].set_title("LIF microdynamics: isolated spikes decay; clustered inputs cross threshold")
    axes[1].plot(trace.time_s * 1_000, trace.membrane, color="#2563eb")
    axes[1].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
    axes[1].set_ylabel("membrane")
    axes[1].legend(loc="upper right")
    axes[2].eventplot(trace.time_s[trace.spikes > 0.0] * 1_000, colors="#111827")
    axes[2].set_ylabel("spikes")
    axes[2].set_xlabel("time (ms)")
    return save_figure(fig, path)


def _plot_rf_micro(path: Path) -> str:
    """Plot RF response to matched and off-frequency sinusoidal input.

    Args:
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    fs = 10_000
    dt = 1.0 / fs
    duration = 0.16
    time_s = np.arange(int(fs * duration)) * dt
    f0 = 120.0
    # The matched input is at the RF tuned frequency; the off-frequency input
    # tests whether the same neuron suppresses non-resonant drive.
    matched = 0.9 * _sine(f0, time_s)
    off = 0.9 * _sine(320.0, time_s)
    matched_trace = simulate_resonate_and_fire(
        matched, dt_s=dt, resonant_frequency_hz=f0, q_factor=10.0, input_gain=0.055, threshold=0.18
    )
    off_trace = simulate_resonate_and_fire(
        off, dt_s=dt, resonant_frequency_hz=f0, q_factor=10.0, input_gain=0.055, threshold=0.18
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s * 1_000, matched, color="#334155", label="120 Hz input")
    axes[0].plot(time_s * 1_000, off, color="#94a3b8", alpha=0.8, label="320 Hz input")
    axes[0].set_ylabel("input")
    axes[0].set_title("RF microdynamics: matched drive builds oscillatory state more efficiently")
    axes[0].legend(loc="upper right")
    axes[1].plot(time_s * 1_000, matched_trace.state, color="#16a34a", label="matched")
    axes[1].plot(time_s * 1_000, off_trace.state, color="#84cc16", alpha=0.8, label="off-frequency")
    axes[1].axhline(0.18, color="#dc2626", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("state")
    axes[1].legend(loc="upper right")
    axes[2].plot(time_s * 1_000, matched_trace.velocity, color="#0f766e", label="matched")
    axes[2].plot(time_s * 1_000, off_trace.velocity, color="#99f6e4", alpha=0.8, label="off-frequency")
    axes[2].set_ylabel("velocity")
    axes[3].eventplot(
        [time_s[matched_trace.spikes > 0.0] * 1_000, time_s[off_trace.spikes > 0.0] * 1_000],
        colors=["#111827", "#64748b"],
        lineoffsets=[1, 0],
    )
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["off", "matched"])
    axes[3].set_xlabel("time (ms)")
    axes[3].set_ylabel("spikes")
    return save_figure(fig, path)


def _plot_level_crossing_micro(path: Path) -> str:
    """Plot level-crossing events for a signal with a sudden activity increase.

    Args:
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    fs = 10_000
    dt = 1.0 / fs
    duration = 0.16
    time_s = np.arange(int(fs * duration)) * dt
    # A low-amplitude slow component followed by a stronger faster component
    # makes the event-rate change easy to see.
    signal = 0.25 * np.sin(2.0 * np.pi * 35.0 * time_s)
    signal += 0.80 * np.sin(2.0 * np.pi * 140.0 * time_s) * (time_s > 0.055)
    signal *= np.exp(-2.0 * time_s)
    trace = simulate_level_crossing(signal, dt_s=dt, delta=0.16, refractory_s=0.0004)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_s * 1_000, signal, color="#334155", label="signal")
    axes[0].step(time_s * 1_000, trace.reference, color="#f97316", linewidth=1.0, where="post", label="internal level")
    axes[0].set_title("Level-crossing microdynamics: events occur only when the signal changes by delta")
    axes[0].set_ylabel("signal")
    axes[0].legend(loc="upper right")
    axes[1].eventplot(time_s[trace.up_spikes > 0.0] * 1_000, colors="#16a34a")
    axes[1].set_ylabel("up events")
    axes[2].eventplot(time_s[trace.down_spikes > 0.0] * 1_000, colors="#dc2626")
    axes[2].set_ylabel("down events")
    axes[2].set_xlabel("time (ms)")
    return save_figure(fig, path)


def _frequency_sweep() -> dict[str, list[float]]:
    """Run macro frequency-response and phase-locking sweeps.

    Returns:
        Dictionary of frequency sweep results. Each list has one value per
        tested input frequency.
    """
    fs = 10_000
    dt = 1.0 / fs
    duration = 0.45
    time_s = np.arange(int(fs * duration)) * dt
    freqs = np.geomspace(8.0, 1_200.0, 56)

    lif_gain = []
    lif_spike_rate = []
    lif_vs = []
    rf_gain = []
    rf_spike_rate = []
    rf_vs = []
    lc_spike_rate = []
    lc_up_vs = []

    for frequency in freqs:
        sine = _sine(float(frequency), time_s)
        # Subthreshold traces use an artificially high threshold so the state
        # can be interpreted as a linear response without reset discontinuities.
        lif_sub = simulate_lif(0.65 * sine, dt_s=dt, tau_s=0.014, threshold=100.0)
        lif_drive = 0.30 + 1.35 * _rectified_drive(float(frequency), time_s)
        lif_spiking = simulate_lif(lif_drive, dt_s=dt, tau_s=0.014, threshold=1.0)

        rf_sub = simulate_resonate_and_fire(
            sine, dt_s=dt, resonant_frequency_hz=120.0, q_factor=10.0, input_gain=0.040, threshold=100.0
        )
        rf_spiking = simulate_resonate_and_fire(
            sine, dt_s=dt, resonant_frequency_hz=120.0, q_factor=10.0, input_gain=0.060, threshold=0.18
        )
        lc = simulate_level_crossing(sine, dt_s=dt, delta=0.20, refractory_s=0.0)

        # The gain curves show macro filtering behaviour; spike rates and
        # vector strength show event-generation and phase-locking behaviour.
        lif_gain.append(half_peak_to_peak(lif_sub.membrane) / 0.65)
        lif_spike_rate.append(float(lif_spiking.spikes.sum() / duration))
        lif_vs.append(vector_strength(lif_spiking.spikes, float(frequency), dt))

        rf_gain.append(half_peak_to_peak(rf_sub.state) / 1.0)
        rf_spike_rate.append(float(rf_spiking.spikes.sum() / duration))
        rf_vs.append(vector_strength(rf_spiking.spikes, float(frequency), dt))

        lc_spike_rate.append(float(lc.total_spikes.sum() / duration))
        lc_up_vs.append(vector_strength(lc.up_spikes, float(frequency), dt))

    return {
        "frequency_hz": freqs.tolist(),
        "lif_gain": lif_gain,
        "lif_spike_rate_hz": lif_spike_rate,
        "lif_vector_strength": lif_vs,
        "rf_gain": rf_gain,
        "rf_spike_rate_hz": rf_spike_rate,
        "rf_vector_strength": rf_vs,
        "level_crossing_spike_rate_hz": lc_spike_rate,
        "level_crossing_up_vector_strength": lc_up_vs,
    }


def _plot_frequency_gain(results: dict[str, list[float]], path: Path) -> str:
    """Plot subthreshold gain versus input frequency.

    Args:
        results: Output from `_frequency_sweep`.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    freqs = np.asarray(results["frequency_hz"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(freqs, results["lif_gain"], linewidth=2.0, label="LIF membrane gain")
    ax.semilogx(freqs, results["rf_gain"], linewidth=2.0, label="RF state gain")
    ax.axvline(120.0, color="#16a34a", linestyle="--", linewidth=1.0, label="RF tuned frequency")
    ax.set_xlabel("input frequency (Hz)")
    ax.set_ylabel("subthreshold amplitude gain")
    ax.set_title("Macro frequency response: LIF is low-pass; RF is band-pass")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_figure(fig, path)


def _plot_spike_rate(results: dict[str, list[float]], path: Path) -> str:
    """Plot output spike/event rate versus input frequency.

    Args:
        results: Output from `_frequency_sweep`.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    freqs = np.asarray(results["frequency_hz"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(freqs, results["lif_spike_rate_hz"], linewidth=2.0, label="LIF")
    ax.semilogx(freqs, results["rf_spike_rate_hz"], linewidth=2.0, label="RF")
    ax.semilogx(freqs, results["level_crossing_spike_rate_hz"], linewidth=2.0, label="Level crossing")
    ax.set_xlabel("input frequency (Hz)")
    ax.set_ylabel("output event rate (spikes/s)")
    ax.set_title("Spike/event rate versus input frequency")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_figure(fig, path)


def _plot_vector_strength(results: dict[str, list[float]], path: Path) -> str:
    """Plot phase-locking vector strength versus input frequency.

    Args:
        results: Output from `_frequency_sweep`.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    freqs = np.asarray(results["frequency_hz"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(freqs, results["lif_vector_strength"], linewidth=2.0, label="LIF")
    ax.semilogx(freqs, results["rf_vector_strength"], linewidth=2.0, label="RF")
    ax.semilogx(freqs, results["level_crossing_up_vector_strength"], linewidth=2.0, label="Level crossing up-events")
    ax.set_xlabel("input frequency (Hz)")
    ax.set_ylabel("vector strength")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Phase locking evidence: vector strength of spike phases")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return save_figure(fig, path)


def _plot_phase_histograms(path: Path) -> str:
    """Plot spike phase histograms for a representative 120 Hz input.

    Args:
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    fs = 10_000
    dt = 1.0 / fs
    duration = 0.65
    time_s = np.arange(int(fs * duration)) * dt
    frequency = 120.0
    sine = _sine(frequency, time_s)
    lif = simulate_lif(0.30 + 1.35 * np.maximum(0.0, sine), dt_s=dt, tau_s=0.014, threshold=1.0)
    rf = simulate_resonate_and_fire(
        sine, dt_s=dt, resonant_frequency_hz=frequency, q_factor=10.0, input_gain=0.060, threshold=0.18
    )
    lc = simulate_level_crossing(sine, dt_s=dt, delta=0.20)

    # Polar histograms show whether spikes happen at consistent phases of the
    # input cycle, which is the intuitive meaning of phase locking.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "polar"})
    traces = [
        ("LIF", spike_phases(lif.spikes, frequency, dt), "#2563eb"),
        ("RF", spike_phases(rf.spikes, frequency, dt), "#16a34a"),
        ("Level crossing up", spike_phases(lc.up_spikes, frequency, dt), "#f97316"),
    ]
    bins = np.linspace(0.0, 2.0 * np.pi, 25)
    for ax, (title, phases, color) in zip(axes, traces, strict=True):
        counts, edges = np.histogram(phases, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        ax.bar(centres, counts, width=width, color=color, alpha=0.75)
        ax.set_title(title)
        ax.set_yticklabels([])
    fig.suptitle("Spike phase distributions for a 120 Hz input")
    return save_figure(fig, path)


def _write_report(artifacts: dict[str, str], results: dict[str, object], elapsed_s: float) -> None:
    """Write the markdown report for the neuron analysis experiment.

    Args:
        artifacts: Mapping from artifact names to saved figure paths.
        results: JSON-serializable results dictionary.
        elapsed_s: Wall-clock runtime of the experiment.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frequency = np.asarray(results["frequency_response"]["frequency_hz"])
    lif_gain = np.asarray(results["frequency_response"]["lif_gain"])
    rf_gain = np.asarray(results["frequency_response"]["rf_gain"])
    rf_peak_frequency = float(frequency[int(np.argmax(rf_gain))])
    lif_half_gain_frequency = float(frequency[np.argmin(np.abs(lif_gain - lif_gain.max() / 2.0))])

    lines = [
        "# Mini Model 1: Neuron Analysis",
        "",
        "This mini model compares three neuron/encoder types that are candidates for the new model design: LIF, resonate-and-fire, and level crossing.",
        "",
        "## Questions",
        "",
        "- What are the mathematical definitions?",
        "- What do their micro temporal dynamics look like?",
        "- What macro frequency response does each produce?",
        "- Do the spikes phase-lock to periodic input?",
        "- Which role should each neuron type play in the redesigned model?",
        "",
        "## Mathematical Definitions",
        "",
        "### Leaky Integrate-and-Fire",
        "",
        "The LIF neuron integrates input current while leaking back toward rest:",
        "",
        "```text",
        "v[t+1] = v[t] + dt/tau * (-v[t] + R * I[t])",
        "spike[t] = 1 if v[t+1] >= threshold else 0",
        "v[t+1] = v[t+1] - threshold  # subtractive reset when spiking",
        "```",
        "",
        "It behaves like a low-pass evidence accumulator. Closely spaced inputs sum; isolated inputs decay before reaching threshold.",
        "",
        "### Resonate-and-Fire",
        "",
        "The RF neuron is a damped second-order oscillator with a spike threshold:",
        "",
        "```text",
        "velocity[t+1] = decay * velocity[t] + input_gain * I[t] - theta * state[t]",
        "state[t+1] = state[t] + theta * velocity[t+1]",
        "spike[t] = 1 if state[t+1] >= threshold else 0",
        "state[t+1] = state[t+1] - threshold  # subtractive reset when spiking",
        "```",
        "",
        "It behaves like a band-pass detector. Inputs near the resonant frequency build state more efficiently than off-frequency inputs.",
        "",
        "### Level-Crossing Encoder",
        "",
        "The level-crossing neuron emits events when the signal changes by a fixed amount:",
        "",
        "```text",
        "if signal[t] - reference >= delta: emit up event; reference += delta",
        "if reference - signal[t] >= delta: emit down event; reference -= delta",
        "```",
        "",
        "This is closer to asynchronous delta modulation than to a standard rate encoder. It encodes change rather than absolute signal level.",
        "",
        "## Micro Temporal Dynamics",
        "",
        "![LIF microdynamics](../outputs/neuron_analysis/figures/lif_microdynamics.png)",
        "",
        "![RF microdynamics](../outputs/neuron_analysis/figures/rf_microdynamics.png)",
        "",
        "![Level-crossing microdynamics](../outputs/neuron_analysis/figures/level_crossing_microdynamics.png)",
        "",
        "## Macro Frequency Response",
        "",
        "![Frequency gain](../outputs/neuron_analysis/figures/frequency_gain.png)",
        "",
        f"The RF neuron peaked at approximately `{rf_peak_frequency:.1f} Hz` in this sweep, close to the configured `120 Hz` resonant frequency. The LIF membrane response behaved as a low-pass filter; its rough half-gain point in this setup was around `{lif_half_gain_frequency:.1f} Hz`.",
        "",
        "![Spike rate](../outputs/neuron_analysis/figures/spike_rate_vs_frequency.png)",
        "",
        "The level-crossing encoder produces more events as the signal oscillates faster, provided the signal crosses enough delta levels. This is useful for change detection, but its event count is not the same thing as a firing-rate estimate from a LIF neuron.",
        "",
        "## Phase Locking",
        "",
        "![Vector strength](../outputs/neuron_analysis/figures/vector_strength_vs_frequency.png)",
        "",
        "Vector strength measures how concentrated spike phases are within the input cycle. A value near `1` indicates strong phase locking; a value near `0` indicates weak or dispersed locking.",
        "",
        "![Phase histograms](../outputs/neuron_analysis/figures/phase_histograms.png)",
        "",
        "The LIF neuron phase-locks when periodic input repeatedly drives it over threshold at a consistent phase. The RF neuron phase-locks most clearly near its resonant band. The level-crossing encoder phase-locks strongly to waveform crossings, especially if up-events and down-events are considered separately.",
        "",
        "## Interpretation For The New Model",
        "",
        "| Neuron / encoder | Best role | Main strength | Main weakness |",
        "|---|---|---|---|",
        "| LIF | Evidence accumulation, coincidence detection, thresholded integration | Simple, robust, interpretable membrane voltage | Low-pass; can blur fast timing if time constants are too long |",
        "| RF | Frequency-selective feature detection | Band-pass selectivity and phase-sensitive dynamics | More parameters; needs careful tuning of frequency/Q/threshold |",
        "| Level crossing | Event-based encoding, change detection, possible cochlea simplification | Sparse, sharp temporal events, cheap comparisons | Sensitive to delta choice; loses absolute level unless multiple thresholds are used |",
        "",
        "For the redesigned model, the cleanest division is likely:",
        "",
        "- use LIF for coincidence/evidence accumulation and ring-map integration;",
        "- use RF for frequency-selective or periodicity-sensitive feature detection;",
        "- use level crossing for fast event-based front-end experiments and possibly spectral-delta notch detection.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend(
        [
            f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`",
            "",
            f"Runtime: `{elapsed_s:.2f} s`.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the neuron analysis experiment.

    Returns:
        JSON-serializable dictionary containing numeric results and artifact
        paths.
    """
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPORT_PATH.parent)

    artifacts = {
        "lif_microdynamics": _plot_lif_micro(FIGURE_DIR / "lif_microdynamics.png"),
        "rf_microdynamics": _plot_rf_micro(FIGURE_DIR / "rf_microdynamics.png"),
        "level_crossing_microdynamics": _plot_level_crossing_micro(FIGURE_DIR / "level_crossing_microdynamics.png"),
    }
    frequency_results = _frequency_sweep()
    artifacts["frequency_gain"] = _plot_frequency_gain(frequency_results, FIGURE_DIR / "frequency_gain.png")
    artifacts["spike_rate_vs_frequency"] = _plot_spike_rate(frequency_results, FIGURE_DIR / "spike_rate_vs_frequency.png")
    artifacts["vector_strength_vs_frequency"] = _plot_vector_strength(
        frequency_results, FIGURE_DIR / "vector_strength_vs_frequency.png"
    )
    artifacts["phase_histograms"] = _plot_phase_histograms(FIGURE_DIR / "phase_histograms.png")

    elapsed_s = time.perf_counter() - start
    results: dict[str, object] = {
        "experiment": "neuron_analysis",
        "elapsed_seconds": elapsed_s,
        "frequency_response": frequency_results,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(artifacts, results, elapsed_s)
    return results


if __name__ == "__main__":
    output = main()
    print(REPORT_PATH)
