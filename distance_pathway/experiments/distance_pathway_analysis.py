from __future__ import annotations

"""Distance pathway analysis: simple model and detector optimization tests.

This script generates two separate reports:

1. `simple_coincidence_model.md`
   A simple explanatory model using a corollary discharge, an echo pulse, and
   distance-tuned delay lines.

2. `accuracy_optimisation_testing.md`
   A synthetic benchmark comparing LIF, RF, and binary coincidence detectors
   for distance estimation.

The experiment deliberately uses ideal pulse timing rather than the full
cochlea. That keeps the distance pathway mechanism isolated and easy to
explain before building the fuller biological and optimized versions.
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
        true_distance_m: True target distances in metres.
        echo_delay_samples: Round-trip echo delay in samples after jitter.
        echo_index: Echo pulse index in the simulated timeline.
        candidate_distance_m: Distances represented by the delay-line bank.
        candidate_delay_samples: Delay-line delays in samples.
        num_time_steps: Length of the synthetic spike timeline.
    """

    true_distance_m: np.ndarray
    echo_delay_samples: np.ndarray
    echo_index: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    num_time_steps: int


@dataclass
class DetectorResult:
    """Accuracy and cost result for one detector type.

    Attributes:
        name: Human-readable detector name.
        predicted_distance_m: Predicted distance for every test sample.
        mae_m: Mean absolute error in metres.
        rmse_m: Root mean squared error in metres.
        p95_abs_error_m: 95th percentile absolute error in metres.
        max_abs_error_m: Worst-case absolute error in metres.
        runtime_ms: Median runtime for the full test set.
        flops: Estimated floating-point operations.
        sops: Estimated synaptic operations or bit operations.
    """

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
        repeats: Number of timed repeats.

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


def _make_dataset() -> Dataset:
    """Create random pulse echo distances and a delay-line bank.

    Returns:
        Synthetic dataset with true echo delays and candidate delay lines.
    """
    rng = np.random.default_rng(RNG_SEED)
    true_distance_m = rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_TEST_SAMPLES)
    round_trip_delay_s = 2.0 * true_distance_m / SPEED_OF_SOUND_M_S
    jitter_s = rng.normal(0.0, JITTER_STD_S, NUM_TEST_SAMPLES)
    echo_delay_samples = np.rint((round_trip_delay_s + jitter_s) * SAMPLE_RATE_HZ).astype(np.int64)
    echo_delay_samples = np.clip(echo_delay_samples, 1, None)
    echo_index = TX_INDEX + echo_delay_samples

    candidate_distance_m = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_DELAY_LINES)
    candidate_delay_samples = np.rint(
        (2.0 * candidate_distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ
    ).astype(np.int64)

    max_candidate = int(candidate_delay_samples.max())
    max_echo = int(echo_delay_samples.max())
    num_time_steps = TX_INDEX + max(max_candidate, max_echo) + 80
    return Dataset(
        true_distance_m=true_distance_m,
        echo_delay_samples=echo_delay_samples,
        echo_index=echo_index,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        num_time_steps=num_time_steps,
    )


def _delay_error_matrix(dataset: Dataset) -> np.ndarray:
    """Compute candidate delay error for every sample and delay line.

    Args:
        dataset: Synthetic dataset.

    Returns:
        Absolute delay mismatch matrix with shape `[samples, delay_lines]`.
    """
    return np.abs(dataset.echo_delay_samples[:, None] - dataset.candidate_delay_samples[None, :])


def _predict_lif(dataset: Dataset) -> np.ndarray:
    """Predict distance with a LIF soft-coincidence detector bank.

    The delayed corollary pulse and echo pulse both add membrane voltage. If
    they arrive close together, residual voltage from the first pulse helps the
    second pulse cross threshold. For this synthetic benchmark, the score can
    be written analytically as:

    `score = w * (1 + beta ** delay_error)`.

    Args:
        dataset: Synthetic dataset.

    Returns:
        Predicted distances in metres.
    """
    delay_error = _delay_error_matrix(dataset)
    beta = 0.982
    input_weight = 0.62
    scores = input_weight * (1.0 + np.power(beta, delay_error))
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_rf(dataset: Dataset) -> np.ndarray:
    """Predict distance with an RF-style coincidence detector bank.

    The RF detector uses a damped oscillatory afterpotential. It still peaks at
    zero timing mismatch, but it can create side lobes if the oscillation is too
    strong or poorly damped.

    Args:
        dataset: Synthetic dataset.

    Returns:
        Predicted distances in metres.
    """
    delay_error = _delay_error_matrix(dataset)
    input_weight = 0.62
    tau_samples = 18.0
    omega = 2.0 * np.pi / 18.0
    afterpotential = np.exp(-delay_error / tau_samples) * np.cos(omega * delay_error)
    scores = input_weight * (1.0 + afterpotential)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_binary(dataset: Dataset) -> np.ndarray:
    """Predict distance with a binary delay-line coincidence detector.

    This detector only checks whether the delayed corollary pulse and echo
    pulse land within a small sample tolerance. If no candidate falls inside
    the tolerance, it falls back to the closest delay.

    Args:
        dataset: Synthetic dataset.

    Returns:
        Predicted distances in metres.
    """
    delay_error = _delay_error_matrix(dataset)
    tolerance_samples = 2
    masked_score = np.where(delay_error <= tolerance_samples, -delay_error, -np.inf)
    no_match = ~np.isfinite(masked_score).any(axis=1)
    if np.any(no_match):
        masked_score[no_match] = -delay_error[no_match]
    return dataset.candidate_distance_m[np.argmax(masked_score, axis=1)]


def _score_detector(name: str, dataset: Dataset, predictor: Callable[[Dataset], np.ndarray]) -> DetectorResult:
    """Benchmark and score one detector.

    Args:
        name: Detector name.
        dataset: Synthetic dataset.
        predictor: Prediction function.

    Returns:
        Detector result summary.
    """
    runtime_s = _median_runtime_s(lambda: predictor(dataset))
    predicted = predictor(dataset)
    error = predicted - dataset.true_distance_m
    abs_error = np.abs(error)
    operations = dataset.true_distance_m.size * dataset.candidate_distance_m.size
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
    """Render a matplotlib figure into a PIL image.

    Args:
        fig: Matplotlib figure.

    Returns:
        Rendered RGBA image.
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return Image.fromarray(rgba.reshape(height, width, 4), mode="RGBA")


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 100) -> str:
    """Save animation frames as a GIF.

    Args:
        frames: Rendered frames.
        path: Output GIF path.
        duration_ms: Duration of each frame.

    Returns:
        Path to the saved GIF.
    """
    if not frames:
        raise ValueError("No animation frames were generated.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    return str(path)


def _plot_explanatory_timeline(dataset: Dataset, path: Path) -> str:
    """Plot corollary discharge, echo, and three example delay lines.

    Args:
        dataset: Synthetic dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
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
    """Plot detector-bank responses for one example target.

    Args:
        dataset: Synthetic dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
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


def _plot_accuracy(results: list[DetectorResult], dataset: Dataset, path: Path) -> str:
    """Plot true-vs-predicted distance for each detector.

    Args:
        results: Detector results.
        dataset: Synthetic dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
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
    """Plot absolute error histograms for each detector.

    Args:
        results: Detector results.
        dataset: Synthetic dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bins = np.linspace(0.0, 0.08, 35)
    for result in results:
        abs_error = np.abs(result.predicted_distance_m - dataset.true_distance_m)
        ax.hist(abs_error, bins=bins, alpha=0.5, label=result.name)
    ax.set_xlabel("absolute distance error (m)")
    ax.set_ylabel("count")
    ax.set_title("Distance error distribution")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def _plot_costs(results: list[DetectorResult], path: Path) -> str:
    """Plot runtime, FLOPs, and SOPs/bit operations.

    Args:
        results: Detector results.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
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
    return save_figure(fig, path)


def _save_coincidence_animation(path: Path) -> str:
    """Create a GIF explaining delay-line coincidence detection.

    Args:
        path: Output GIF path.

    Returns:
        Saved GIF path.
    """
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


def _write_simple_report(dataset: Dataset, artifacts: dict[str, str], elapsed_s: float) -> None:
    """Write the simple explanatory coincidence report.

    Args:
        dataset: Synthetic dataset.
        artifacts: Generated artifact paths.
        elapsed_s: Total script runtime.
    """
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
        if name in {"pulse_timeline", "coincidence_animation", "delay_bank_response"}:
            lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend(["", f"Runtime for full script: `{elapsed_s:.2f} s`.", ""])
    SIMPLE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_optimisation_report(
    dataset: Dataset,
    results: list[DetectorResult],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the LIF/RF/binary optimization report.

    Args:
        dataset: Synthetic dataset.
        results: Detector results.
        artifacts: Generated artifact paths.
        elapsed_s: Total script runtime.
    """
    lines = [
        "# Distance Pathway 2: Accuracy And Optimisation Testing",
        "",
        "This report compares three ways of implementing the simple delay-line distance pathway on ideal pulse timing: LIF, RF, and binary coincidence detection.",
        "",
        "## Detector Equations",
        "",
        "For all detectors, the mismatch between the observed echo delay and candidate delay is:",
        "",
        "```text",
        "delta_k = abs(delay_echo - delay_candidate[k])",
        "```",
        "",
        "### LIF Detector",
        "",
        "```text",
        "score_k = w * (1 + beta^delta_k)",
        "```",
        "",
        "The LIF version is a soft coincidence detector. It is tolerant to small timing offsets because residual membrane voltage decays gradually.",
        "",
        "### RF Detector",
        "",
        "```text",
        "score_k = w * (1 + exp(-delta_k/tau_rf) * cos(omega_rf * delta_k))",
        "```",
        "",
        "The RF version is also soft, but its oscillatory afterpotential can create side lobes. That may be useful for periodicity tasks, but it is not obviously better for pure echo-delay matching.",
        "",
        "### Binary Detector",
        "",
        "```text",
        "match_k = 1 if delta_k <= tolerance else 0",
        "```",
        "",
        "The binary version is the cheapest form. It assumes the upstream system has already produced reliable onset events.",
        "",
        "## Benchmark Setup",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{SAMPLE_RATE_HZ} Hz` |",
        f"| speed of sound | `{SPEED_OF_SOUND_M_S} m/s` |",
        f"| distance range | `{MIN_DISTANCE_M} -> {MAX_DISTANCE_M} m` |",
        f"| test samples | `{NUM_TEST_SAMPLES}` |",
        f"| delay lines | `{NUM_DELAY_LINES}` |",
        f"| jitter std | `{JITTER_STD_S * 1_000_000.0:.1f} us` |",
        "",
        "The jitter prevents the task from being a perfectly quantized lookup problem.",
        "",
        "## Results",
        "",
        "![Accuracy scatter](../outputs/accuracy_optimisation/figures/accuracy_scatter.png)",
        "",
        "![Error histogram](../outputs/accuracy_optimisation/figures/error_histogram.png)",
        "",
        "![Cost comparison](../outputs/accuracy_optimisation/figures/cost_comparison.png)",
        "",
        "| Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.mae_m * 100.0:.2f} | {result.rmse_m * 100.0:.2f} | "
            f"{result.p95_abs_error_m * 100.0:.2f} | {result.max_abs_error_m * 100.0:.2f} | "
            f"{result.runtime_ms:.3f} | {result.flops:,.0f} | {result.sops:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- LIF is the clearest biological soft-coincidence baseline.",
            "- RF is biologically interesting, but for pure delay matching it adds oscillatory side lobes and higher estimated FLOP cost.",
            "- Binary coincidence is the most optimized form for ideal onset events and is the natural candidate for later bit-packing/event-based acceleration.",
            "- The binary result should not be over-interpreted yet, because real cochlear spike rasters will have noise, missed spikes, extra spikes, and frequency-channel structure.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        if name in {"accuracy_scatter", "error_histogram", "cost_comparison"}:
            lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    OPT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the distance pathway analyses.

    Returns:
        JSON-serializable results payload.
    """
    start = time.perf_counter()
    ensure_dir(SIMPLE_FIGURE_DIR)
    ensure_dir(OPT_FIGURE_DIR)
    ensure_dir(REPORT_DIR)
    ensure_dir(BASE_OUTPUT_DIR)

    dataset = _make_dataset()
    results = [
        _score_detector("LIF detector", dataset, _predict_lif),
        _score_detector("RF detector", dataset, _predict_rf),
        _score_detector("Binary detector", dataset, _predict_binary),
    ]
    artifacts = {
        "pulse_timeline": _plot_explanatory_timeline(dataset, SIMPLE_FIGURE_DIR / "pulse_timeline.png"),
        "coincidence_animation": _save_coincidence_animation(SIMPLE_FIGURE_DIR / "coincidence_detection.gif"),
        "delay_bank_response": _plot_delay_bank_response(dataset, SIMPLE_FIGURE_DIR / "delay_bank_response.png"),
        "accuracy_scatter": _plot_accuracy(results, dataset, OPT_FIGURE_DIR / "accuracy_scatter.png"),
        "error_histogram": _plot_error_histogram(results, dataset, OPT_FIGURE_DIR / "error_histogram.png"),
        "cost_comparison": _plot_costs(results, OPT_FIGURE_DIR / "cost_comparison.png"),
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
            "rng_seed": RNG_SEED,
            "num_time_steps": dataset.num_time_steps,
        },
        "results": [
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
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_simple_report(dataset, artifacts, elapsed_s)
    _write_optimisation_report(dataset, results, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(SIMPLE_REPORT_PATH)
    print(OPT_REPORT_PATH)
