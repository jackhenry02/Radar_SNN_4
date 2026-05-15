from __future__ import annotations

"""Clean binary distance-pathway optimisation experiment.

This experiment isolates the clean FM-sweep binary path and compares:

1. A floating-point LIF-style score.
2. The original binary delay loop.
3. An optimized binary method using `torch.unfold`.

The clean test deliberately samples target distances from the delay-line grid.
That means exact binary coincidence is valid without extra dilation, allowing
the benchmark to focus on the simulation strategy rather than interpolation.
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
NUM_TEST_SAMPLES = 256
SWEEP_CHANNELS = 32
SWEEP_DURATION_S = 0.003
RNG_SEED = 31
CHUNK_SIZE = 16

OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "binary_clean_optimisation"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "binary_clean_optimisation.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"


@dataclass
class CleanSweepDataset:
    """Clean grid-aligned FM-sweep spike dataset.

    Attributes:
        true_distance_m: True distances in metres.
        true_candidate_index: Index of the correct delay-line candidate.
        candidate_distance_m: Distance represented by each delay line.
        candidate_delay_samples: Delay represented by each delay line.
        channel_offsets_samples: Corollary-discharge sweep offsets.
        cd_raster: Corollary discharge raster `[samples, channels, time]`.
        echo_raster: Echo raster `[samples, channels, time]`.
    """

    true_distance_m: np.ndarray
    true_candidate_index: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    channel_offsets_samples: np.ndarray
    cd_raster: torch.Tensor
    echo_raster: torch.Tensor


@dataclass
class MethodResult:
    """Result summary for one clean binary-path method."""

    name: str
    predicted_distance_m: np.ndarray
    mae_m: float
    accuracy_percent: float
    runtime_ms: float
    flops: float
    sops: float


def _median_runtime_s(function: Callable[[], object], repeats: int = 3) -> float:
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


def _make_clean_sweep_dataset() -> CleanSweepDataset:
    """Build clean grid-aligned corollary-discharge and echo rasters.

    Returns:
        Clean sweep dataset for the optimisation benchmark.
    """
    rng = np.random.default_rng(RNG_SEED)
    candidate_distance_m = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_DELAY_LINES)
    candidate_delay_samples = np.rint(
        (2.0 * candidate_distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ
    ).astype(np.int64)
    true_candidate_index = rng.integers(0, NUM_DELAY_LINES, size=NUM_TEST_SAMPLES)
    true_distance_m = candidate_distance_m[true_candidate_index]
    true_delay_samples = candidate_delay_samples[true_candidate_index]

    sweep_samples = int(round(SWEEP_DURATION_S * SAMPLE_RATE_HZ))
    channel_offsets = np.rint(np.linspace(0, sweep_samples, SWEEP_CHANNELS)).astype(np.int64)
    total_time = TX_INDEX + int(candidate_delay_samples.max()) + int(channel_offsets.max()) + 96

    cd_raster = torch.zeros((NUM_TEST_SAMPLES, SWEEP_CHANNELS, total_time), dtype=torch.bool)
    echo_raster = torch.zeros_like(cd_raster)
    sample_indices = torch.arange(NUM_TEST_SAMPLES)
    for channel, offset in enumerate(channel_offsets):
        cd_time = TX_INDEX + int(offset)
        cd_raster[:, channel, cd_time] = True
        echo_times = torch.as_tensor(TX_INDEX + offset + true_delay_samples, dtype=torch.long)
        echo_raster[sample_indices, channel, echo_times] = True

    return CleanSweepDataset(
        true_distance_m=true_distance_m,
        true_candidate_index=true_candidate_index,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        channel_offsets_samples=channel_offsets,
        cd_raster=cd_raster,
        echo_raster=echo_raster,
    )


def _predict_lif_score(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict with the original LIF-style analytic score.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted distances in metres.
    """
    true_delay = dataset.candidate_delay_samples[dataset.true_candidate_index]
    delay_error = np.abs(true_delay[:, None] - dataset.candidate_delay_samples[None, :])
    score = np.power(0.982, delay_error)
    return dataset.candidate_distance_m[np.argmax(score, axis=1)]


def _predict_binary_loop(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict with the original iterative binary delay loop.

    This explicitly loops over candidate delays and tests coincidence by
    shifting the echo raster against the corollary-discharge raster.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted distances in metres.
    """
    cd = dataset.cd_raster
    echo = dataset.echo_raster
    time_steps = cd.shape[-1]
    scores = torch.zeros((cd.shape[0], dataset.candidate_delay_samples.size), dtype=torch.int32)
    for index, delay in enumerate(dataset.candidate_delay_samples):
        delay_value = int(delay)
        valid_time = time_steps - delay_value
        coincidence = cd[:, :, :valid_time] & echo[:, :, delay_value : delay_value + valid_time]
        scores[:, index] = coincidence.sum(dim=(1, 2))
    predicted_index = torch.argmax(scores, dim=1).cpu().numpy()
    return dataset.candidate_distance_m[predicted_index]


def _predict_binary_unfold(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict with an unfold-based binary delay tensor.

    The echo raster is unfolded into `[time, delay]` windows. A broadcasted
    boolean `AND` then checks all selected candidate delays for a chunk of
    samples in one operation.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted distances in metres.
    """
    candidate_delays = torch.as_tensor(dataset.candidate_delay_samples, dtype=torch.long)
    max_delay = int(candidate_delays.max().item())
    valid_time = dataset.cd_raster.shape[-1] - max_delay
    predicted_indices = []
    for start in range(0, dataset.cd_raster.shape[0], CHUNK_SIZE):
        stop = min(dataset.cd_raster.shape[0], start + CHUNK_SIZE)
        cd_chunk = dataset.cd_raster[start:stop, :, :valid_time]
        echo_chunk = dataset.echo_raster[start:stop]
        echo_delay_windows = echo_chunk.unfold(dimension=2, size=max_delay + 1, step=1)
        selected_windows = echo_delay_windows.index_select(dim=-1, index=candidate_delays)
        coincidence = cd_chunk.unsqueeze(-1) & selected_windows
        scores = coincidence.sum(dim=(1, 2))
        predicted_indices.append(torch.argmax(scores, dim=1))
    predicted_index = torch.cat(predicted_indices).cpu().numpy()
    return dataset.candidate_distance_m[predicted_index]


def _score_method(
    name: str,
    dataset: CleanSweepDataset,
    predictor: Callable[[CleanSweepDataset], np.ndarray],
    *,
    flops: float,
    sops: float,
) -> MethodResult:
    """Benchmark and score one method.

    Args:
        name: Method name.
        dataset: Clean sweep dataset.
        predictor: Prediction function.
        flops: Estimated FLOP count.
        sops: Estimated binary/synaptic operation count.

    Returns:
        Method result summary.
    """
    runtime = _median_runtime_s(lambda: predictor(dataset), repeats=3)
    predicted = predictor(dataset)
    error = predicted - dataset.true_distance_m
    correct = np.isclose(predicted, dataset.true_distance_m)
    return MethodResult(
        name=name,
        predicted_distance_m=predicted,
        mae_m=float(np.abs(error).mean()),
        accuracy_percent=float(correct.mean() * 100.0),
        runtime_ms=runtime * 1_000.0,
        flops=flops,
        sops=sops,
    )


def _operation_estimates(dataset: CleanSweepDataset) -> dict[str, tuple[float, float]]:
    """Estimate FLOPs/SOPs for each method.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Mapping from method name to `(flops, sops)`.
    """
    samples = dataset.cd_raster.shape[0]
    channels = dataset.cd_raster.shape[1]
    delays = dataset.candidate_delay_samples.size
    max_delay = int(dataset.candidate_delay_samples.max())
    valid_time = dataset.cd_raster.shape[-1] - max_delay
    lif_scores = samples * channels * delays
    binary_and = samples * channels * valid_time * delays
    return {
        "LIF score": (lif_scores * 8.0, lif_scores * 2.0),
        "Binary loop": (0.0, binary_and),
        "Binary unfold": (0.0, binary_and),
    }


def _plot_rasters(dataset: CleanSweepDataset, path: Path) -> str:
    """Plot one clean corollary-discharge and echo raster.

    Args:
        dataset: Clean sweep dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    sample_index = 0
    frequencies_khz = np.linspace(18.0, 2.0, SWEEP_CHANNELS)
    cd = dataset.cd_raster[sample_index].cpu().numpy()
    echo = dataset.echo_raster[sample_index].cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for channel, frequency_khz in enumerate(frequencies_khz):
        cd_times = np.flatnonzero(cd[channel]) / SAMPLE_RATE_HZ * 1_000.0
        echo_times = np.flatnonzero(echo[channel]) / SAMPLE_RATE_HZ * 1_000.0
        axes[0].vlines(cd_times, frequency_khz - 0.18, frequency_khz + 0.18, color="#2563eb")
        axes[1].vlines(echo_times, frequency_khz - 0.18, frequency_khz + 0.18, color="#dc2626")
    axes[0].set_title("Corollary-discharge sweep raster")
    axes[1].set_title("Echo sweep raster")
    axes[1].set_xlabel("time (ms)")
    for ax in axes:
        ax.set_ylabel("frequency (kHz)")
        ax.set_ylim(1.0, 19.0)
        ax.set_xlim(0.0, 35.0)
        ax.grid(True, axis="x", alpha=0.2)
    return save_figure(fig, path)


def _plot_accuracy(results: list[MethodResult], dataset: CleanSweepDataset, path: Path) -> str:
    """Plot true-vs-predicted distance for each method.

    Args:
        results: Method results.
        dataset: Clean sweep dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, result in zip(axes, results):
        ax.scatter(dataset.true_distance_m, result.predicted_distance_m, s=8, alpha=0.45)
        ax.plot([MIN_DISTANCE_M, MAX_DISTANCE_M], [MIN_DISTANCE_M, MAX_DISTANCE_M], color="#111827")
        ax.set_title(f"{result.name}\nMAE={result.mae_m * 100.0:.3f} cm")
        ax.set_xlabel("true distance (m)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("predicted distance (m)")
    return save_figure(fig, path)


def _plot_runtime_cost(results: list[MethodResult], path: Path) -> str:
    """Plot runtime and operation estimates.

    Args:
        results: Method results.
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
    axes[0].set_ylabel("ms")
    axes[1].bar(names, flops, color="#f97316")
    axes[1].set_title("Estimated FLOPs")
    axes[2].bar(names, sops, color="#16a34a")
    axes[2].set_title("Estimated binary ops / SOPs")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, path)


def _write_report(
    dataset: CleanSweepDataset,
    results: list[MethodResult],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the binary clean optimisation report.

    Args:
        dataset: Clean sweep dataset.
        results: Method results.
        artifacts: Generated artifact paths.
        elapsed_s: Total script runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    max_delay = int(dataset.candidate_delay_samples.max())
    valid_time = dataset.cd_raster.shape[-1] - max_delay
    lines = [
        "# Binary Clean Pathway Optimisation",
        "",
        "This report isolates the clean FM-sweep binary distance path and compares an iterative delay-line implementation with an optimized `torch.unfold` implementation.",
        "",
        "## Aim",
        "",
        "The optimisation idea is to stop simulating the distance pathway one time step at a time. Instead, construct binary tensors for the corollary discharge and echo, unfold the echo over the delay dimension, and perform the coincidence test as one large boolean operation.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[CD raster B x F x T] --> C[expand with singleton delay axis]",
        "    B[Echo raster B x F x T] --> D[torch unfold over time]",
        "    D --> E[select candidate delays]",
        "    C --> F[AND over CD and delayed echo]",
        "    E --> F",
        "    F --> G[sum over frequency and time]",
        "    G --> H[argmax distance]",
        "```",
        "",
        "## Important Assumption",
        "",
        "This is a clean, grid-aligned benchmark. Target distances are sampled from the delay-line grid, so exact binary coincidence is valid without dilation. This isolates the compute strategy. In a continuous-distance or noisy setting, the binary path would need either a tolerance window, denser delay lines, or surrounding robustness from the sweep/population code.",
        "",
        "## Input Rasters",
        "",
        "The benchmark uses a simple FM-sweep spike raster: one corollary-discharge spike per frequency channel, and one echo spike per frequency channel shifted by the target delay.",
        "",
        "![Clean sweep rasters](../outputs/binary_clean_optimisation/figures/clean_sweep_rasters.png)",
        "",
        "## Methods",
        "",
        "### Original LIF Score",
        "",
        "```text",
        "score_k = mean_f(beta ^ abs(delay_true - delay_candidate[k]))",
        "```",
        "",
        "This is a floating-point soft coincidence baseline.",
        "",
        "### Original Binary Loop",
        "",
        "```text",
        "for delay in candidate_delays:",
        "    score[delay] = sum(CD[:, :, t] AND echo[:, :, t + delay])",
        "```",
        "",
        "This is binary, but still iterates over delay lines.",
        "",
        "### Optimised Binary Unfold",
        "",
        "```text",
        "echo_windows = echo.unfold(time, max_delay + 1, step=1)",
        "selected = echo_windows[..., candidate_delays]",
        "coincidence = CD[..., valid_time, None] AND selected",
        "score = sum(coincidence over frequency and time)",
        "```",
        "",
        "This creates a time-delay view of the echo and evaluates all candidate delays together for each chunk of samples.",
        "",
        "## Benchmark Setup",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| samples | `{NUM_TEST_SAMPLES}` |",
        f"| frequency channels | `{SWEEP_CHANNELS}` |",
        f"| delay lines | `{NUM_DELAY_LINES}` |",
        f"| sample rate | `{SAMPLE_RATE_HZ} Hz` |",
        f"| sweep duration | `{SWEEP_DURATION_S * 1_000.0:.1f} ms` |",
        f"| max delay | `{max_delay}` samples |",
        f"| valid unfolded time steps | `{valid_time}` |",
        f"| chunk size | `{CHUNK_SIZE}` samples |",
        "",
        "## Results",
        "",
        "![Accuracy scatter](../outputs/binary_clean_optimisation/figures/accuracy_scatter.png)",
        "",
        "![Runtime and cost](../outputs/binary_clean_optimisation/figures/runtime_cost.png)",
        "",
        "| Method | MAE (cm) | Exact accuracy (%) | Runtime (ms) | FLOPs | Binary ops / SOPs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.mae_m * 100.0:.4f} | {result.accuracy_percent:.2f} | "
            f"{result.runtime_ms:.3f} | {result.flops:,.0f} | {result.sops:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The unfold method implements the proposed layer-style binary operation: construct a delay dimension, then apply one boolean coincidence operation per chunk.",
            "- Exact binary coincidence works here without dilation because the benchmark is grid-aligned.",
            "- The operation count is similar to the looped binary method, but the simulation structure changes from explicit delay iteration to tensorized delay evaluation.",
            "- On CPU, tensorization is not guaranteed to be faster because the unfolded delay view can create substantial memory traffic. This is still the right structure for later GPU/MPS or bit-packed experiments.",
            "- The next optimisation would be packed-bit `AND`/popcount, which should reduce the memory and operation cost substantially.",
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
    """Run the binary clean optimisation benchmark.

    Returns:
        JSON-serializable result payload.
    """
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    ensure_dir(OUTPUT_DIR)
    torch.manual_seed(RNG_SEED)

    dataset = _make_clean_sweep_dataset()
    estimates = _operation_estimates(dataset)
    results = [
        _score_method("LIF score", dataset, _predict_lif_score, flops=estimates["LIF score"][0], sops=estimates["LIF score"][1]),
        _score_method("Binary loop", dataset, _predict_binary_loop, flops=estimates["Binary loop"][0], sops=estimates["Binary loop"][1]),
        _score_method("Binary unfold", dataset, _predict_binary_unfold, flops=estimates["Binary unfold"][0], sops=estimates["Binary unfold"][1]),
    ]
    artifacts = {
        "clean_sweep_rasters": _plot_rasters(dataset, FIGURE_DIR / "clean_sweep_rasters.png"),
        "accuracy_scatter": _plot_accuracy(results, dataset, FIGURE_DIR / "accuracy_scatter.png"),
        "runtime_cost": _plot_runtime_cost(results, FIGURE_DIR / "runtime_cost.png"),
    }
    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "binary_clean_optimisation",
        "elapsed_seconds": elapsed_s,
        "config": {
            "samples": NUM_TEST_SAMPLES,
            "frequency_channels": SWEEP_CHANNELS,
            "delay_lines": NUM_DELAY_LINES,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sweep_duration_s": SWEEP_DURATION_S,
            "chunk_size": CHUNK_SIZE,
            "grid_aligned": True,
        },
        "results": [
            {
                "name": result.name,
                "mae_m": result.mae_m,
                "accuracy_percent": result.accuracy_percent,
                "runtime_ms": result.runtime_ms,
                "flops": result.flops,
                "sops": result.sops,
            }
            for result in results
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(dataset, results, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
