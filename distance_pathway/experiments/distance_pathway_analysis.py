from __future__ import annotations

"""Distance pathway reports for simple coincidence and detector optimisation.

The script generates two reports:

1. `simple_coincidence_model.md`
   Explanatory corollary-discharge and delay-line coincidence model.

2. `accuracy_optimisation_testing.md`
   Accuracy/runtime/FLOP/SOP comparison for LIF, RF, and binary detectors under
   clean, noisy, jittered, and noisy+jittered pulse conditions.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_models.common.plotting import ensure_dir, save_figure


SPEED_OF_SOUND_M_S = 343.0
SAMPLE_RATE_HZ = 64_000
TX_INDEX = 64
MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 5.0
NUM_DELAY_LINES = 160
NUM_TEST_SAMPLES = 1_000
JITTER_STD_S = 35e-6
WHITE_NOISE_SNR_DB = 10.0
RNG_SEED = 11

BASE_OUTPUT_DIR = ROOT / "distance_pathway" / "outputs"
SIMPLE_OUTPUT_DIR = BASE_OUTPUT_DIR / "simple_coincidence_model"
SIMPLE_FIGURE_DIR = SIMPLE_OUTPUT_DIR / "figures"
OPT_OUTPUT_DIR = BASE_OUTPUT_DIR / "accuracy_optimisation"
OPT_FIGURE_DIR = OPT_OUTPUT_DIR / "figures"
REPORT_DIR = ROOT / "distance_pathway" / "reports"
SIMPLE_REPORT_PATH = REPORT_DIR / "simple_coincidence_model.md"
OPT_REPORT_PATH = REPORT_DIR / "accuracy_optimisation_testing.md"
RESULTS_PATH = BASE_OUTPUT_DIR / "distance_pathway_results.json"


@dataclass
class Dataset:
    """Synthetic pulse-distance dataset.

    Attributes:
        condition: Name of the signal condition.
        true_distance_m: True target distances in metres.
        ideal_echo_delay_samples: Perfect round-trip echo delay in samples.
        echo_delay_samples: Echo delay in samples after optional jitter.
        waveform: Synthetic pulse waveform after optional white noise,
            shape `[samples, time]`.
        candidate_distance_m: Distances represented by delay lines.
        candidate_delay_samples: Delay-line delays in samples.
        num_time_steps: Length of the synthetic timeline.
        has_noise: Whether additive white noise is present.
        has_jitter: Whether true echo timing jitter is present.
    """

    condition: str
    true_distance_m: np.ndarray
    ideal_echo_delay_samples: np.ndarray
    echo_delay_samples: np.ndarray
    waveform: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    num_time_steps: int
    has_noise: bool
    has_jitter: bool


@dataclass
class DetectorResult:
    """Accuracy and cost result for one detector type and condition."""

    condition: str
    name: str
    predicted_distance_m: np.ndarray
    mae_m: float
    rmse_m: float
    p95_abs_error_m: float
    max_abs_error_m: float
    runtime_ms: float
    flops: float
    sops: float


def _median_runtime_s(function: Callable[[], object], repeats: int = 20) -> float:
    """Measure median runtime for a callable.

    Args:
        function: Zero-argument callable.
        repeats: Number of repeats.

    Returns:
        Median runtime in seconds.
    """
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _base_distances() -> np.ndarray:
    """Create the common true distances used by all conditions."""
    rng = np.random.default_rng(RNG_SEED)
    return rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_TEST_SAMPLES)


def _make_dataset(condition: str, *, has_noise: bool, has_jitter: bool, seed_offset: int) -> Dataset:
    """Create one benchmark condition.

    Args:
        condition: Human-readable condition name.
        has_noise: Whether to add Gaussian white noise to the waveform.
        has_jitter: Whether to jitter the true echo pulse.
        seed_offset: Offset used for condition-specific randomness.

    Returns:
        Dataset for the requested condition.
    """
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    true_distance_m = _base_distances()
    ideal_delay_s = 2.0 * true_distance_m / SPEED_OF_SOUND_M_S
    if has_jitter:
        jitter_s = rng.normal(0.0, JITTER_STD_S, true_distance_m.size)
    else:
        jitter_s = np.zeros_like(true_distance_m)
    echo_delay = np.rint((ideal_delay_s + jitter_s) * SAMPLE_RATE_HZ).astype(np.int64)
    echo_delay = np.clip(echo_delay, 1, None)
    ideal_echo_delay = np.rint(ideal_delay_s * SAMPLE_RATE_HZ).astype(np.int64)

    candidate_distance_m = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_DELAY_LINES)
    candidate_delay_samples = np.rint(
        (2.0 * candidate_distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ
    ).astype(np.int64)

    max_candidate = int(candidate_delay_samples.max())
    max_observed = int(echo_delay.max())
    num_time_steps = TX_INDEX + max(max_candidate, max_observed) + 80
    waveform = np.zeros((true_distance_m.size, num_time_steps), dtype=np.float64)
    echo_indices = TX_INDEX + echo_delay
    waveform[np.arange(true_distance_m.size), echo_indices] = 1.0
    if has_noise:
        noise_std = 10.0 ** (-WHITE_NOISE_SNR_DB / 20.0)
        waveform += rng.normal(0.0, noise_std, size=waveform.shape)
    return Dataset(
        condition=condition,
        true_distance_m=true_distance_m,
        ideal_echo_delay_samples=ideal_echo_delay,
        echo_delay_samples=echo_delay,
        waveform=waveform,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        num_time_steps=num_time_steps,
        has_noise=has_noise,
        has_jitter=has_jitter,
    )


def _all_datasets() -> list[Dataset]:
    """Return the four requested benchmark conditions."""
    return [
        _make_dataset("Clean perfect", has_noise=False, has_jitter=False, seed_offset=0),
        _make_dataset("Added noise", has_noise=True, has_jitter=False, seed_offset=100),
        _make_dataset("Added jitter", has_noise=False, has_jitter=True, seed_offset=200),
        _make_dataset("Noise + jitter", has_noise=True, has_jitter=True, seed_offset=300),
    ]


def _delay_error_tensor(dataset: Dataset) -> np.ndarray:
    """Compute observed-pulse delay error against every candidate delay.

    Args:
        dataset: Benchmark dataset.

    Returns:
        Absolute delay mismatch tensor `[samples, pulses, delay_lines]`.
    """
    return np.abs(
        dataset.observed_pulse_delay_samples[:, :, None] - dataset.candidate_delay_samples[None, None, :]
    )


def _predict_lif(dataset: Dataset) -> np.ndarray:
    """Predict distance with a LIF soft-coincidence detector bank."""
    delay_error = _delay_error_tensor(dataset)
    amplitudes = dataset.observed_pulse_amplitudes[:, :, None]
    beta = 0.982
    input_weight = 0.62
    pulse_scores = amplitudes * input_weight * (1.0 + np.power(beta, delay_error))
    scores = pulse_scores.max(axis=1)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_rf(dataset: Dataset) -> np.ndarray:
    """Predict distance with an RF-style coincidence detector bank."""
    delay_error = _delay_error_tensor(dataset)
    amplitudes = dataset.observed_pulse_amplitudes[:, :, None]
    input_weight = 0.62
    tau_samples = 18.0
    omega = 2.0 * np.pi / 18.0
    afterpotential = np.exp(-delay_error / tau_samples) * np.cos(omega * delay_error)
    pulse_scores = amplitudes * input_weight * (1.0 + afterpotential)
    scores = pulse_scores.max(axis=1)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_binary(dataset: Dataset) -> np.ndarray:
    """Predict distance with a binary delay-line coincidence detector."""
    delay_error = _delay_error_tensor(dataset)
    amplitudes = dataset.observed_pulse_amplitudes[:, :, None]
    tolerance_samples = 2
    pulse_scores = np.where(delay_error <= tolerance_samples, amplitudes, -np.inf)
    scores = pulse_scores.max(axis=1)
    no_match = ~np.isfinite(scores).any(axis=1)
    if np.any(no_match):
        nearest_error = delay_error[no_match].min(axis=1)
        scores[no_match] = -nearest_error
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _score_detector(name: str, dataset: Dataset, predictor: Callable[[Dataset], np.ndarray]) -> DetectorResult:
    """Benchmark and score one detector on one condition."""
    runtime_s = _median_runtime_s(lambda: predictor(dataset))
    predicted = predictor(dataset)
    error = predicted - dataset.true_distance_m
    abs_error = np.abs(error)
    operations = (
        dataset.true_distance_m.size
        * dataset.candidate_distance_m.size
        * dataset.observed_pulse_delay_samples.shape[1]
    )
    if name.startswith("LIF"):
        flops = operations * 8.0
        sops = operations * 2.0
    elif name.startswith("RF"):
        flops = operations * 14.0
        sops = operations * 2.0
    else:
        flops = operations * 1.0
        sops = operations
    return DetectorResult(
        condition=dataset.condition,
        name=name,
        predicted_distance_m=predicted,
        mae_m=float(abs_error.mean()),
        rmse_m=float(np.sqrt(np.mean(error**2))),
        p95_abs_error_m=float(np.percentile(abs_error, 95.0)),
        max_abs_error_m=float(abs_error.max()),
        runtime_ms=runtime_s * 1_000.0,
        flops=flops,
        sops=sops,
    )


def _render_frame(fig: plt.Figure) -> Image.Image:
    """Render a matplotlib figure into a PIL image."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return Image.fromarray(rgba.reshape(height, width, 4), mode="RGBA")


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 100) -> str:
    """Save animation frames as a GIF."""
    if not frames:
        raise ValueError("No animation frames were generated.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    return str(path)


def _plot_explanatory_timeline(dataset: Dataset, path: Path) -> str:
    """Plot corollary discharge, echo, and three example delay lines."""
    target_distance = 2.7
    true_delay = int(round((2.0 * target_distance / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ))
    true_echo = TX_INDEX + true_delay
    candidate_distances = np.array([2.1, target_distance, 3.3])
    candidate_delays = np.rint((2.0 * candidate_distances / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ).astype(int)
    time_ms = np.arange(dataset.num_time_steps) / SAMPLE_RATE_HZ * 1_000.0

    fig, ax = plt.subplots(figsize=(12, 5.5))
    rows = [
        ("corollary discharge", TX_INDEX, "#2563eb"),
        ("echo pulse", true_echo, "#dc2626"),
        ("short delay line", TX_INDEX + candidate_delays[0], "#64748b"),
        ("matched delay line", TX_INDEX + candidate_delays[1], "#16a34a"),
        ("long delay line", TX_INDEX + candidate_delays[2], "#64748b"),
    ]
    for row_index, (label, index, color) in enumerate(rows):
        ax.hlines(row_index, 0, time_ms[-1], color="#e5e7eb", linewidth=1)
        ax.vlines(time_ms[index], row_index - 0.35, row_index + 0.35, color=color, linewidth=3)
        ax.text(time_ms[index] + 0.15, row_index + 0.12, f"{time_ms[index]:.2f} ms", color=color)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_xlabel("time (ms)")
    ax.set_title("Only the matched delay line coincides with the echo")
    ax.set_xlim(0.0, min(time_ms[-1], 22.0))
    ax.grid(True, axis="x", alpha=0.25)
    return save_figure(fig, path)


def _plot_delay_bank_response(dataset: Dataset, path: Path) -> str:
    """Plot detector-bank responses for one example target."""
    target_distance = 2.7
    delay = int(round((2.0 * target_distance / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ))
    delay_error = np.abs(delay - dataset.candidate_delay_samples)
    lif_scores = 0.62 * (1.0 + np.power(0.982, delay_error))
    rf_scores = 0.62 * (1.0 + np.exp(-delay_error / 18.0) * np.cos((2.0 * np.pi / 18.0) * delay_error))
    binary_scores = (delay_error <= 2).astype(float)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(dataset.candidate_distance_m, lif_scores / lif_scores.max(), label="LIF score", linewidth=2.0)
    ax.plot(dataset.candidate_distance_m, rf_scores / rf_scores.max(), label="RF score", linewidth=2.0)
    ax.step(dataset.candidate_distance_m, binary_scores, where="mid", label="binary coincidence", linewidth=2.0)
    ax.axvline(target_distance, color="#111827", linestyle="--", label="true distance")
    ax.set_xlabel("candidate distance (m)")
    ax.set_ylabel("normalized detector response")
    ax.set_title("Delay-line bank response for a 2.7 m target")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_detector_time_examples(path: Path) -> str:
    """Plot amplitude/state against time for one LIF, RF, and binary example."""
    steps = 260
    target_distance = 0.45
    true_delay = int(round((2.0 * target_distance / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ))
    matched_time = TX_INDEX + true_delay
    short_time = matched_time - 16
    long_time = matched_time + 18
    echo_time = matched_time
    time_ms = np.arange(steps) / SAMPLE_RATE_HZ * 1_000.0
    pulse_train = np.zeros(steps)
    for pulse_time, amplitude in [(TX_INDEX, 1.0), (echo_time, 1.0)]:
        if 0 <= pulse_time < steps:
            pulse_train[pulse_time] = amplitude

    candidate_times = [short_time, matched_time, long_time]
    labels = ["short detector", "matched detector", "long detector"]
    colors = ["#64748b", "#16a34a", "#64748b"]
    lif = np.zeros((3, steps))
    rf = np.zeros((3, steps))
    binary = np.zeros((3, steps))
    for det, candidate_time in enumerate(candidate_times):
        voltage = 0.0
        rf_state = 0.0
        rf_velocity = 0.0
        for step in range(steps):
            detector_drive = 0.0
            if step == candidate_time:
                detector_drive += 0.62
            if step == echo_time:
                detector_drive += 0.62
            voltage = 0.86 * voltage + detector_drive
            if voltage >= 1.0:
                voltage -= 1.0
            lif[det, step] = voltage

            rf_velocity = 0.94 * rf_velocity + detector_drive - 0.34 * rf_state
            rf_state = rf_state + 0.34 * rf_velocity
            rf[det, step] = rf_state
            binary[det, step] = 1.0 if abs(candidate_time - echo_time) <= 2 and step == echo_time else 0.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].stem(time_ms, pulse_train, linefmt="#111827", markerfmt=" ", basefmt=" ")
    axes[0].set_title("Input amplitude: corollary discharge pulse and echo pulse")
    axes[0].set_ylabel("amplitude")
    for det, label in enumerate(labels):
        axes[1].plot(time_ms, lif[det], color=colors[det], linewidth=2.0, label=label)
    axes[1].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
    axes[1].set_title("LIF membrane amplitude against time")
    axes[1].set_ylabel("membrane")
    axes[1].legend(loc="upper right")
    for det, label in enumerate(labels):
        axes[2].plot(time_ms, rf[det], color=colors[det], linewidth=2.0, label=label)
    axes[2].set_title("RF resonant state amplitude against time")
    axes[2].set_ylabel("RF state")
    for det, label in enumerate(labels):
        axes[3].step(time_ms, binary[det] + det * 1.2, where="post", color=colors[det], linewidth=2.0, label=label)
    axes[3].set_title("Binary coincidence output against time")
    axes[3].set_ylabel("binary output")
    axes[3].set_xlabel("time (ms)")
    axes[3].set_yticks([0.5, 1.7, 2.9], labels)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(time_ms[max(0, TX_INDEX - 20)], time_ms[min(steps - 1, echo_time + 45)])
    return save_figure(fig, path)


def _save_coincidence_animation(path: Path) -> str:
    """Create a GIF explaining delay-line coincidence detection."""
    frames: list[Image.Image] = []
    steps = 72
    echo_step = 38
    candidate_arrivals = [28, echo_step, 50]
    labels = ["too short", "matched", "too long"]
    colors = ["#64748b", "#16a34a", "#64748b"]
    threshold = 1.0
    beta = 0.82

    membrane = np.zeros((3, steps))
    spikes = np.zeros((3, steps))
    for detector, arrival in enumerate(candidate_arrivals):
        voltage = 0.0
        for step in range(steps):
            voltage *= beta
            if step == arrival:
                voltage += 0.62
            if step == echo_step:
                voltage += 0.62
            if voltage >= threshold:
                spikes[detector, step] = 1.0
                voltage -= threshold
            membrane[detector, step] = voltage

    for step in range(steps):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        fig.suptitle("Distance by Delay-Line Coincidence", fontsize=15, fontweight="bold")
        axes[0].set_title("A corollary discharge is delayed by different amounts")
        axes[0].set_xlim(0, steps - 1)
        axes[0].set_ylim(-0.5, 3.4)
        axes[0].set_yticks([0, 1, 2, 3], ["echo", *labels])
        axes[0].axvline(step, color="#94a3b8", linestyle="--", linewidth=1)
        axes[0].vlines(echo_step, -0.25, 0.25, color="#dc2626", linewidth=4, label="echo")
        for detector, arrival in enumerate(candidate_arrivals):
            axes[0].hlines(detector + 1, 0, steps - 1, color="#e5e7eb")
            if step >= arrival:
                axes[0].vlines(arrival, detector + 0.75, detector + 1.25, color=colors[detector], linewidth=4)
            if step >= echo_step:
                axes[0].vlines(echo_step, -0.25, 0.25, color="#dc2626", linewidth=4)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, axis="x", alpha=0.2)

        axes[1].set_title("Only the matched detector crosses threshold")
        for detector, color in enumerate(colors):
            axes[1].plot(membrane[detector, : step + 1], color=color, linewidth=2.0, label=labels[detector])
            spike_times = np.flatnonzero(spikes[detector, : step + 1])
            if spike_times.size:
                axes[1].scatter(spike_times, np.full_like(spike_times, threshold + 0.05), color=color, zorder=3)
        axes[1].axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.3, label="threshold")
        axes[1].axvline(step, color="#94a3b8", linestyle="--", linewidth=1)
        axes[1].set_xlim(0, steps - 1)
        axes[1].set_ylim(0, 1.25)
        axes[1].set_xlabel("time step")
        axes[1].set_ylabel("membrane")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.2)
        frames.append(_render_frame(fig))
        plt.close(fig)
    return _save_gif(frames, path, duration_ms=95)


def _plot_condition_mae(results_by_condition: dict[str, list[DetectorResult]], path: Path) -> str:
    """Plot MAE by condition and detector."""
    conditions = list(results_by_condition)
    detector_names = [result.name for result in next(iter(results_by_condition.values()))]
    x = np.arange(len(conditions))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, detector in enumerate(detector_names):
        mae_cm = [
            next(result for result in results_by_condition[condition] if result.name == detector).mae_m * 100.0
            for condition in conditions
        ]
        ax.bar(x + (index - 1) * width, mae_cm, width=width, label=detector)
    ax.set_xticks(x, conditions)
    ax.set_ylabel("MAE (cm)")
    ax.set_title("Distance accuracy under clean, noise, jitter, and combined conditions")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, path)


def _plot_accuracy(results: list[DetectorResult], dataset: Dataset, path: Path) -> str:
    """Plot true-vs-predicted distance for one condition."""
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, result in zip(axes, results):
        ax.scatter(dataset.true_distance_m, result.predicted_distance_m, s=8, alpha=0.45)
        ax.plot([MIN_DISTANCE_M, MAX_DISTANCE_M], [MIN_DISTANCE_M, MAX_DISTANCE_M], color="#111827")
        ax.set_title(f"{result.name}\nMAE={result.mae_m * 100:.2f} cm")
        ax.set_xlabel("true distance (m)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("predicted distance (m)")
    return save_figure(fig, path)


def _plot_error_histogram(results: list[DetectorResult], dataset: Dataset, path: Path) -> str:
    """Plot absolute error histograms for one condition."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bins = np.linspace(0.0, 1.6, 45)
    for result in results:
        abs_error = np.abs(result.predicted_distance_m - dataset.true_distance_m)
        ax.hist(abs_error, bins=bins, alpha=0.45, label=result.name)
    ax.set_xlabel("absolute distance error (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Distance error distribution: {dataset.condition}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_costs(results_by_condition: dict[str, list[DetectorResult]], path: Path) -> str:
    """Plot runtime, FLOPs, and SOPs for the noise+jitter condition."""
    results = results_by_condition["Noise + jitter"]
    names = [result.name for result in results]
    runtime = [result.runtime_ms for result in results]
    flops = [result.flops for result in results]
    sops = [result.sops for result in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(names, runtime, color="#2563eb")
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("ms per 1000 samples")
    axes[1].bar(names, flops, color="#f97316")
    axes[1].set_title("Estimated FLOPs")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].bar(names, sops, color="#16a34a")
    axes[2].set_title("SOPs / bit operations")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Cost comparison shown for the hardest condition: noise + jitter")
    return save_figure(fig, path)


def _write_simple_report(dataset: Dataset, artifacts: dict[str, str], elapsed_s: float) -> None:
    """Write the simple explanatory coincidence report."""
    lines = [
        "# Distance Pathway 1: Simple Coincidence Model",
        "",
        "This report explains the simplest distance pathway possible: a corollary discharge pulse, an echo pulse, and a bank of delay-line coincidence detectors.",
        "",
        "## Core Idea",
        "",
        "Distance is encoded by echo latency. If a target is distance `d` away, the sound travels to the target and back, so:",
        "",
        "```text",
        "tau = 2*d / c",
        "delay_samples = round(tau * f_s)",
        "d = c*tau / 2",
        "```",
        "",
        "The system does not need to process the outgoing sound through the cochlea again. It can use an internal corollary discharge, or efference copy, triggered when the call is emitted.",
        "",
        "## Architecture",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[call command] --> B[corollary discharge]",
        "    B --> C1[short delay line]",
        "    B --> C2[medium delay line]",
        "    B --> C3[long delay line]",
        "    D[echo onset pulse] --> E[coincidence detector bank]",
        "    C1 --> E",
        "    C2 --> E",
        "    C3 --> E",
        "    E --> F[winner detector]",
        "    F --> G[distance estimate]",
        "```",
        "",
        "A tuned detector fires when its delayed corollary pulse arrives at the same time as the echo pulse. The winning detector's delay maps directly to a distance.",
        "",
        "![Pulse timeline](../outputs/simple_coincidence_model/figures/pulse_timeline.png)",
        "",
        "![Coincidence animation](../outputs/simple_coincidence_model/figures/coincidence_detection.gif)",
        "",
        "## Amplitude Against Time Examples",
        "",
        "The plot below shows one example response for the LIF, RF, and binary detector forms. It uses the same corollary pulse and echo pulse, but the internal amplitude/state is different for each detector type.",
        "",
        "![Detector time examples](../outputs/simple_coincidence_model/figures/detector_time_examples.png)",
        "",
        "## Example Detector Bank Response",
        "",
        "The plot below shows a single target at `2.7 m`. The matched delay line peaks at the correct distance.",
        "",
        "![Delay bank response](../outputs/simple_coincidence_model/figures/delay_bank_response.png)",
        "",
        "## Parameters Used For The Explanatory Model",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{SAMPLE_RATE_HZ} Hz` |",
        f"| speed of sound | `{SPEED_OF_SOUND_M_S} m/s` |",
        f"| distance range | `{MIN_DISTANCE_M} -> {MAX_DISTANCE_M} m` |",
        f"| delay lines | `{NUM_DELAY_LINES}` |",
        f"| time steps | `{dataset.num_time_steps}` |",
        "",
        "## Biological Interpretation",
        "",
        "This is a simplified Jeffress-style delay-line idea applied to echo delay rather than interaural delay. The corollary discharge acts like the internally generated reference for the emitted call, and the echo onset acts like the sensory return pulse.",
        "",
        "The detector can be implemented as a LIF neuron: one input arrives from the delayed corollary discharge, one from the echo onset, and the neuron only crosses threshold when both arrive close together.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        if name in {"pulse_timeline", "coincidence_animation", "detector_time_examples", "delay_bank_response"}:
            lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend(["", f"Runtime for full script: `{elapsed_s:.2f} s`.", ""])
    SIMPLE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_optimisation_report(
    datasets: list[Dataset],
    results_by_condition: dict[str, list[DetectorResult]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the condition-based LIF/RF/binary optimization report."""
    lines = [
        "# Distance Pathway 2: Accuracy And Optimisation Testing",
        "",
        "This report compares LIF, RF, and binary delay-line coincidence detectors under four pulse conditions: clean, added noise, added jitter, and noise plus jitter.",
        "",
        "## Signal Conditions",
        "",
        "| Condition | True echo jitter | Spurious onset noise |",
        "|---|---:|---:|",
    ]
    for dataset in datasets:
        lines.append(
            f"| {dataset.condition} | `{dataset.has_jitter}` | `{dataset.has_noise}` |"
        )
    lines.extend(
        [
            "",
            "Noise here means extra spurious onset pulses in the echo window. Jitter means Gaussian timing jitter on the true echo pulse. This is still a simplified pulse model, not full waveform noise.",
            "",
            "## Detector Equations",
            "",
            "For all detectors, the mismatch between each observed pulse and candidate delay is:",
            "",
            "```text",
            "delta_p,k = abs(delay_observed[p] - delay_candidate[k])",
            "```",
            "",
            "The LIF and RF detectors score all observed pulses and use the strongest response. The binary detector checks whether any pulse lands inside a small timing window.",
            "",
            "```text",
            "LIF:    score_k = max_p amplitude_p * w * (1 + beta^delta_p,k)",
            "RF:     score_k = max_p amplitude_p * w * (1 + exp(-delta_p,k/tau_rf) * cos(omega_rf * delta_p,k))",
            "Binary: match_k = any_p(delta_p,k <= tolerance)",
            "```",
            "",
            "## Benchmark Setup",
            "",
            "| Parameter | Value |",
            "|---|---:|",
            f"| sample rate | `{SAMPLE_RATE_HZ} Hz` |",
            f"| speed of sound | `{SPEED_OF_SOUND_M_S} m/s` |",
            f"| distance range | `{MIN_DISTANCE_M} -> {MAX_DISTANCE_M} m` |",
            f"| test samples per condition | `{NUM_TEST_SAMPLES}` |",
            f"| delay lines | `{NUM_DELAY_LINES}` |",
            f"| jitter std | `{JITTER_STD_S * 1_000_000.0:.1f} us` |",
            f"| noise pulses | `{NOISE_PULSES}` |",
            f"| noise amplitude range | `{NOISE_MIN_AMPLITUDE} -> {NOISE_MAX_AMPLITUDE}` |",
            "",
            "## Accuracy Across Conditions",
            "",
            "![Condition MAE](../outputs/accuracy_optimisation/figures/condition_mae.png)",
            "",
            "The detailed numeric results are:",
            "",
            "| Condition | Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, results in results_by_condition.items():
        for result in results:
            lines.append(
                f"| {condition} | {result.name} | {result.mae_m * 100.0:.2f} | "
                f"{result.rmse_m * 100.0:.2f} | {result.p95_abs_error_m * 100.0:.2f} | "
                f"{result.max_abs_error_m * 100.0:.2f} | {result.runtime_ms:.3f} | "
                f"{result.flops:,.0f} | {result.sops:,.0f} |"
            )
    lines.extend(
        [
            "",
            "## Hardest-Condition Plots",
            "",
            "The scatter, histogram, and cost plots below use the hardest condition, `Noise + jitter`.",
            "",
            "![Accuracy scatter](../outputs/accuracy_optimisation/figures/accuracy_scatter_noise_jitter.png)",
            "",
            "![Error histogram](../outputs/accuracy_optimisation/figures/error_histogram_noise_jitter.png)",
            "",
            "![Cost comparison](../outputs/accuracy_optimisation/figures/cost_comparison.png)",
            "",
            "## Interpretation",
            "",
            "- Clean perfect signals are essentially a delay quantisation problem, so LIF and binary should be close.",
            "- Jitter tests timing tolerance. LIF remains a useful soft detector because the membrane trace decays smoothly with timing mismatch.",
            "- Noise tests false-onset robustness. Binary is cheap, but can be fooled if a strong false onset lands near another candidate delay.",
            "- RF remains biologically interesting, but its oscillatory side lobes are a weakness for this specific pure-delay task.",
            "- These results still assume onset pulses have already been extracted; the next hard problem is robust onset extraction from real cochlear spike rasters.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        if name in {
            "condition_mae",
            "accuracy_scatter_noise_jitter",
            "error_histogram_noise_jitter",
            "cost_comparison",
        }:
            lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    OPT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the distance pathway analyses."""
    start = time.perf_counter()
    ensure_dir(SIMPLE_FIGURE_DIR)
    ensure_dir(OPT_FIGURE_DIR)
    ensure_dir(REPORT_DIR)
    ensure_dir(BASE_OUTPUT_DIR)

    datasets = _all_datasets()
    results_by_condition: dict[str, list[DetectorResult]] = {}
    for dataset in datasets:
        results_by_condition[dataset.condition] = [
            _score_detector("LIF detector", dataset, _predict_lif),
            _score_detector("RF detector", dataset, _predict_rf),
            _score_detector("Binary detector", dataset, _predict_binary),
        ]

    simple_dataset = datasets[0]
    hard_dataset = datasets[-1]
    hard_results = results_by_condition[hard_dataset.condition]
    artifacts = {
        "pulse_timeline": _plot_explanatory_timeline(simple_dataset, SIMPLE_FIGURE_DIR / "pulse_timeline.png"),
        "coincidence_animation": _save_coincidence_animation(SIMPLE_FIGURE_DIR / "coincidence_detection.gif"),
        "detector_time_examples": _plot_detector_time_examples(SIMPLE_FIGURE_DIR / "detector_time_examples.png"),
        "delay_bank_response": _plot_delay_bank_response(simple_dataset, SIMPLE_FIGURE_DIR / "delay_bank_response.png"),
        "condition_mae": _plot_condition_mae(results_by_condition, OPT_FIGURE_DIR / "condition_mae.png"),
        "accuracy_scatter_noise_jitter": _plot_accuracy(
            hard_results,
            hard_dataset,
            OPT_FIGURE_DIR / "accuracy_scatter_noise_jitter.png",
        ),
        "error_histogram_noise_jitter": _plot_error_histogram(
            hard_results,
            hard_dataset,
            OPT_FIGURE_DIR / "error_histogram_noise_jitter.png",
        ),
        "cost_comparison": _plot_costs(results_by_condition, OPT_FIGURE_DIR / "cost_comparison.png"),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "distance_pathway_analysis",
        "elapsed_seconds": elapsed_s,
        "config": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "speed_of_sound_m_s": SPEED_OF_SOUND_M_S,
            "min_distance_m": MIN_DISTANCE_M,
            "max_distance_m": MAX_DISTANCE_M,
            "num_delay_lines": NUM_DELAY_LINES,
            "num_test_samples": NUM_TEST_SAMPLES,
            "jitter_std_s": JITTER_STD_S,
            "noise_pulses": NOISE_PULSES,
            "noise_min_amplitude": NOISE_MIN_AMPLITUDE,
            "noise_max_amplitude": NOISE_MAX_AMPLITUDE,
            "rng_seed": RNG_SEED,
        },
        "results_by_condition": {
            condition: [
                {
                    "name": result.name,
                    "mae_m": result.mae_m,
                    "rmse_m": result.rmse_m,
                    "p95_abs_error_m": result.p95_abs_error_m,
                    "max_abs_error_m": result.max_abs_error_m,
                    "runtime_ms": result.runtime_ms,
                    "flops": result.flops,
                    "sops": result.sops,
                }
                for result in results
            ]
            for condition, results in results_by_condition.items()
        },
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_simple_report(simple_dataset, artifacts, elapsed_s)
    _write_optimisation_report(datasets, results_by_condition, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(SIMPLE_REPORT_PATH)
    print(OPT_REPORT_PATH)
