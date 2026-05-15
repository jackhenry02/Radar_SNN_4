from __future__ import annotations

"""Clean FM-sweep binary distance-pathway optimisation experiment.

This experiment compares three clean-sweep distance estimators:

1. The original LIF-style score used in the accuracy report.
2. The original binary score used in the accuracy report.
3. A pure bitwise `torch.unfold` implementation that delays candidate
   selection until after the full delay-space coincidence map is collapsed.

The aim is to compare the optimised binary path against the same scoring
logic used in the previous accuracy tests, rather than against a special
grid-aligned benchmark.
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
NUM_TEST_SAMPLES = 32
SWEEP_CHANNELS = 32
SWEEP_DURATION_S = 0.003
RNG_SEED = 31
LIF_BETA = 0.982
ECHO_AMPLITUDE = 0.62

OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "binary_clean_optimisation"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "binary_clean_optimisation.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"


@dataclass
class CleanSweepDataset:
    """Clean continuous-distance FM-sweep spike dataset.

    Attributes:
        true_distance_m: True continuous distances in metres.
        true_delay_samples: Rounded echo delay for each sample.
        nearest_candidate_index: Delay-line index closest to the true distance.
        candidate_distance_m: Distance represented by each delay line.
        candidate_delay_samples: Delay represented by each delay line.
        channel_offsets_samples: Corollary-discharge sweep offsets.
        cd_raster: Corollary discharge raster with shape `[samples, channels, time]`.
        echo_raster: Echo raster with shape `[samples, channels, time]`.
        binary_tolerance_samples: Half-bin tolerance used by the original binary score.
    """

    true_distance_m: np.ndarray
    true_delay_samples: np.ndarray
    nearest_candidate_index: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    channel_offsets_samples: np.ndarray
    cd_raster: torch.Tensor
    echo_raster: torch.Tensor
    binary_tolerance_samples: int


@dataclass
class MethodResult:
    """Result summary for one clean sweep method."""

    name: str
    predicted_distance_m: np.ndarray
    predicted_candidate_index: np.ndarray
    mae_m: float
    nearest_bin_accuracy_percent: float
    runtime_ms: float
    flops: float
    sops: float


def _median_runtime_s(function: Callable[[], object], repeats: int = 5) -> float:
    """Measure median runtime for a callable.

    Args:
        function: Zero-argument callable to benchmark.
        repeats: Number of timed repeats after one warm-up call.

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
    """Build a clean FM-sweep dataset matching the accuracy-test setup.

    Distances are sampled continuously, then represented by the closest
    available delay line. This mirrors the previous clean sweep accuracy test
    and avoids the artificially perfect grid-aligned case.

    Returns:
        Clean sweep dataset for benchmarking.
    """
    rng = np.random.default_rng(RNG_SEED)
    candidate_distance_m = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_DELAY_LINES)
    candidate_delay_samples = np.rint(
        (2.0 * candidate_distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ
    ).astype(np.int64)
    true_distance_m = rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M, size=NUM_TEST_SAMPLES)
    true_delay_samples = np.rint(
        (2.0 * true_distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ
    ).astype(np.int64)
    nearest_candidate_index = np.argmin(
        np.abs(candidate_distance_m[None, :] - true_distance_m[:, None]), axis=1
    )

    sweep_samples = int(round(SWEEP_DURATION_S * SAMPLE_RATE_HZ))
    channel_offsets = np.rint(np.linspace(0, sweep_samples, SWEEP_CHANNELS)).astype(np.int64)
    max_delay = int(candidate_delay_samples.max())
    total_time = TX_INDEX + max_delay + int(channel_offsets.max()) + 96
    binary_tolerance_samples = int(np.ceil(np.median(np.diff(candidate_delay_samples)) / 2.0))

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
        true_delay_samples=true_delay_samples,
        nearest_candidate_index=nearest_candidate_index,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        channel_offsets_samples=channel_offsets,
        cd_raster=cd_raster,
        echo_raster=echo_raster,
        binary_tolerance_samples=binary_tolerance_samples,
    )


def _predict_original_lif_score(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict using the original clean-sweep LIF-style score.

    The original sweep score compares the true echo delay against every
    candidate delay and gives higher score to smaller timing errors.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    delay_error = np.abs(
        dataset.true_delay_samples[:, None] - dataset.candidate_delay_samples[None, :]
    )
    scores = ECHO_AMPLITUDE * (1.0 + np.power(LIF_BETA, delay_error))
    return np.argmax(scores, axis=1)


def _predict_original_binary_score(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict using the original clean-sweep binary score.

    This is the analytic version used in the accuracy report: a delay line
    fires if it lies within half a candidate-bin of the observed delay.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    delay_error = np.abs(
        dataset.true_delay_samples[:, None] - dataset.candidate_delay_samples[None, :]
    )
    scores = delay_error <= dataset.binary_tolerance_samples
    return np.argmax(scores.astype(np.int32), axis=1)


def _dilate_binary_template(template: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a binary template along time using small bitwise shifts.

    Args:
        template: Boolean tensor with shape `[samples, channels, time]`.
        radius: Number of samples to dilate on each side.

    Returns:
        Boolean tensor with each event widened by `radius` samples.
    """
    if radius <= 0:
        return template

    dilated = template.clone()
    for shift in range(1, radius + 1):
        dilated[..., shift:] |= template[..., :-shift]
        dilated[..., :-shift] |= template[..., shift:]
    return dilated


def _predict_optimised_binary_unfold(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict using pure bitwise unfold before candidate selection.

    The critical ordering is:

    1. Unfold the whole echo timeline into all possible delay windows.
    2. Broadcast the corollary-discharge template over the delay axis.
    3. Perform one bitwise `AND` over the full delay space.
    4. Collapse time.
    5. Select the candidate delays at the end.

    The CD template is dilated by the same half-bin tolerance as the original
    binary score, so this method is comparable to the accuracy-test binary
    setup while still avoiding advanced indexing before the `AND`.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    candidate_delays = torch.as_tensor(dataset.candidate_delay_samples, dtype=torch.long)
    max_delay = int(candidate_delays.max().item())
    window_length = dataset.cd_raster.shape[-1] - max_delay

    cd_template = dataset.cd_raster[:, :, :window_length]
    cd_template = _dilate_binary_template(cd_template, dataset.binary_tolerance_samples)
    cd_template = cd_template.unsqueeze(2)

    echo_windows = dataset.echo_raster.unfold(dimension=-1, size=window_length, step=1)
    coincidence = echo_windows & cd_template
    raw_scores = coincidence.any(dim=-1)
    final_scores = raw_scores.index_select(dim=-1, index=candidate_delays).sum(dim=1)
    return torch.argmax(final_scores, dim=1).cpu().numpy()


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
        predictor: Function returning predicted candidate indices.
        flops: Estimated floating-point operation count.
        sops: Estimated binary/synaptic operation count.

    Returns:
        Method result summary.
    """
    runtime = _median_runtime_s(lambda: predictor(dataset), repeats=5)
    predicted_index = predictor(dataset)
    predicted = dataset.candidate_distance_m[predicted_index]
    error = predicted - dataset.true_distance_m
    nearest_correct = predicted_index == dataset.nearest_candidate_index
    return MethodResult(
        name=name,
        predicted_distance_m=predicted,
        predicted_candidate_index=predicted_index,
        mae_m=float(np.abs(error).mean()),
        nearest_bin_accuracy_percent=float(nearest_correct.mean() * 100.0),
        runtime_ms=runtime * 1_000.0,
        flops=flops,
        sops=sops,
    )


def _operation_estimates(dataset: CleanSweepDataset) -> dict[str, tuple[float, float]]:
    """Estimate FLOPs/SOPs for each method.

    Returns:
        Mapping from method name to `(flops, sops)`.
    """
    samples = dataset.cd_raster.shape[0]
    channels = dataset.cd_raster.shape[1]
    candidate_delays = dataset.candidate_delay_samples.size
    max_delay = int(dataset.candidate_delay_samples.max())
    delay_bins = max_delay + 1
    window_length = dataset.cd_raster.shape[-1] - max_delay

    original_score_ops = samples * channels * candidate_delays
    unfold_bit_ops = samples * channels * delay_bins * window_length
    candidate_sum_ops = samples * channels * candidate_delays
    return {
        "Original LIF score": (original_score_ops * 8.0, original_score_ops * 2.0),
        "Original binary score": (0.0, original_score_ops),
        "Optimised binary unfold": (0.0, unfold_bit_ops + candidate_sum_ops),
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
        ax.scatter(dataset.true_distance_m, result.predicted_distance_m, s=18, alpha=0.65)
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
        ax.tick_params(axis="x", rotation=18)
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
    delay_bins = max_delay + 1
    window_length = dataset.cd_raster.shape[-1] - max_delay
    lines = [
        "# Binary Clean Pathway Optimisation",
        "",
        "This report now compares the optimised binary implementation against the same clean FM-sweep LIF and binary scoring setups used in the distance accuracy tests.",
        "",
        "## Aim",
        "",
        "The optimisation idea is to stop evaluating only pre-selected delay candidates inside the main coincidence operation. Instead, the echo raster is unfolded across the full delay space, the bitwise coincidence map is computed, time is collapsed, and candidate delays are selected only at the end.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Echo raster<br/>B x F x T] --> B[unfold over time<br/>B x F x delay_bins x window]",
        "    C[CD raster<br/>B x F x window] --> D[unsqueeze delay axis<br/>B x F x 1 x window]",
        "    B --> E[pure bitwise AND]",
        "    D --> E",
        "    E --> F[any over window]",
        "    F --> G[select candidate delays]",
        "    G --> H[sum over channels]",
        "    H --> I[argmax distance]",
        "```",
        "",
        "## Important Correction",
        "",
        "The previous unfold version selected candidate delays before the bitwise `AND`. That used advanced indexing on the large unfolded tensor and destroyed much of the point of using `unfold`. The corrected ordering is:",
        "",
        "```python",
        "echo_windows = echo.unfold(dimension=-1, size=window_length, step=1)",
        "cd_template = swept_cd.unsqueeze(2)",
        "coincidence = echo_windows & cd_template",
        "raw_scores = coincidence.any(dim=-1)",
        "final_scores = raw_scores[..., candidate_delays].sum(dim=1)",
        "```",
        "",
        "The only candidate-delay selection now happens after time has been collapsed.",
        "",
        "## Input Rasters",
        "",
        "The benchmark uses a clean FM-sweep spike raster: one corollary-discharge spike per frequency channel, and one echo spike per frequency channel shifted by the target delay.",
        "",
        "![Clean sweep rasters](../outputs/binary_clean_optimisation/figures/clean_sweep_rasters.png)",
        "",
        "## Methods",
        "",
        "### Original LIF Score",
        "",
        "```text",
        "score_k = mean_f(A * (1 + beta ^ abs(delay_true - delay_candidate[k])))",
        "```",
        "",
        "This is the soft timing score used in the clean sweep accuracy test.",
        "",
        "### Original Binary Score",
        "",
        "```text",
        "score_k = 1 if abs(delay_true - delay_candidate[k]) <= half_bin_tolerance else 0",
        "```",
        "",
        "This is the binary clean-sweep score from the accuracy test. The tolerance is half the median delay-line spacing.",
        "",
        "### Optimised Binary Unfold",
        "",
        "```text",
        "echo_windows = unfold(echo, window_length)",
        "coincidence = echo_windows AND dilated_cd_template",
        "raw_scores = any(coincidence over time)",
        "final_scores = raw_scores at candidate delays, summed over channels",
        "```",
        "",
        "The CD template is dilated by the same half-bin tolerance as the original binary score. This makes the optimised method comparable to the accuracy-test binary path while preserving the corrected operation ordering.",
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
        f"| full delay bins tested by unfold | `{delay_bins}` |",
        f"| unfolded window length | `{window_length}` samples |",
        f"| binary tolerance | `{dataset.binary_tolerance_samples}` samples |",
        "",
        "The sample count is intentionally modest because the pure full-delay unfold materialises a large boolean coincidence tensor on CPU. This benchmark tests implementation structure; larger sample counts should use batching, packing, or hardware acceleration.",
        "",
        "## Results",
        "",
        "![Accuracy scatter](../outputs/binary_clean_optimisation/figures/accuracy_scatter.png)",
        "",
        "![Runtime and cost](../outputs/binary_clean_optimisation/figures/runtime_cost.png)",
        "",
        "| Method | MAE (cm) | Nearest-bin accuracy (%) | Runtime (ms) | FLOPs | Binary ops / SOPs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.mae_m * 100.0:.4f} | "
            f"{result.nearest_bin_accuracy_percent:.2f} | {result.runtime_ms:.3f} | "
            f"{result.flops:,.0f} | {result.sops:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The original LIF score and original binary score are now the same scoring families as the previous clean sweep accuracy tests.",
            "- The optimised binary implementation now follows the intended pure bitwise logic: no candidate advanced indexing occurs before the `AND`.",
            "- The optimised unfold version performs many more raw binary comparisons because it evaluates every integer delay bin, not only the 160 candidate bins. This is structurally cleaner, but not automatically faster on CPU.",
            "- The result is the right stepping stone for bit-packing: once the boolean tensors are packed into machine words, the same full-delay logic can become `AND` plus popcount rather than dense boolean tensor traffic.",
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
        _score_method(
            "Original LIF score",
            dataset,
            _predict_original_lif_score,
            flops=estimates["Original LIF score"][0],
            sops=estimates["Original LIF score"][1],
        ),
        _score_method(
            "Original binary score",
            dataset,
            _predict_original_binary_score,
            flops=estimates["Original binary score"][0],
            sops=estimates["Original binary score"][1],
        ),
        _score_method(
            "Optimised binary unfold",
            dataset,
            _predict_optimised_binary_unfold,
            flops=estimates["Optimised binary unfold"][0],
            sops=estimates["Optimised binary unfold"][1],
        ),
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
            "binary_tolerance_samples": dataset.binary_tolerance_samples,
            "continuous_distances": True,
            "candidate_selection_after_and": True,
        },
        "results": [
            {
                "name": result.name,
                "mae_m": result.mae_m,
                "nearest_bin_accuracy_percent": result.nearest_bin_accuracy_percent,
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
