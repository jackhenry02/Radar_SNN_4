from __future__ import annotations

"""Fair clean FM-sweep binary distance-pathway optimisation experiment.

This experiment compares distance detectors using only the model inputs:

* corollary-discharge spike raster;
* echo spike raster;
* candidate delay bank.

The true distance and true delay are used only after prediction to calculate
error metrics. This avoids the unfair analytic shortcut used in earlier
diagnostic versions of the experiment.
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
NUM_TEST_SAMPLES = 1_000
SWEEP_CHANNELS = 32
SWEEP_DURATION_S = 0.003
RNG_SEED = 31
LIF_BETA = 0.982

OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "binary_clean_optimisation"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "binary_clean_optimisation.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"


@dataclass
class CleanSweepDataset:
    """Clean continuous-distance FM-sweep spike dataset.

    Attributes:
        true_distance_m: True continuous distances in metres.
        true_delay_samples: Rounded echo delay for each sample. This is used
            only to generate the echo raster and score the final prediction.
        nearest_candidate_index: Delay-line index closest to the true distance.
        candidate_distance_m: Distance represented by each delay line.
        candidate_delay_samples: Delay represented by each delay line.
        channel_offsets_samples: Corollary-discharge sweep offsets.
        cd_raster: Corollary discharge raster `[samples, channels, time]`.
        echo_raster: Echo raster `[samples, channels, time]`.
        binary_tolerance_samples: Half-bin tolerance for binary matching.
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
    """Result summary for one detector method."""

    name: str
    predicted_distance_m: np.ndarray
    predicted_candidate_index: np.ndarray
    mae_m: float
    rmse_m: float
    nearest_bin_accuracy_percent: float
    runtime_ms: float
    flops: float
    sops: float


def _median_runtime_s(function: Callable[[], object], repeats: int = 5) -> float:
    """Measure median runtime for a callable.

    Args:
        function: Zero-argument callable to benchmark.
        repeats: Number of timed repeats after one warm-up.

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
    """Build a clean FM-sweep spike-raster dataset.

    Returns:
        Clean sweep dataset for the fair detector benchmark.
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


def _first_event_times(raster: torch.Tensor) -> np.ndarray:
    """Extract first spike time from each sample/channel raster.

    Args:
        raster: Boolean raster with shape `[samples, channels, time]`.

    Returns:
        First event times with shape `[samples, channels]`. Missing events are
        returned as `-1`.
    """
    has_event = raster.any(dim=-1).cpu().numpy()
    first_time = torch.argmax(raster.to(torch.int16), dim=-1).cpu().numpy().astype(np.int64)
    first_time[~has_event] = -1
    return first_time


def _observed_delay_matrix(dataset: CleanSweepDataset) -> np.ndarray:
    """Estimate echo delays from the input rasters only.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Observed echo-minus-CD delays `[samples, channels]`.
    """
    cd_times = _first_event_times(dataset.cd_raster)
    echo_times = _first_event_times(dataset.echo_raster)
    valid = (cd_times >= 0) & (echo_times >= 0)
    delays = echo_times - cd_times
    delays[~valid] = -1
    return delays


def _paired_event_delays_from_nonzero(dataset: CleanSweepDataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired echo-minus-CD delays from sparse event lists.

    This prototype starts from dense rasters, so `nonzero` still scans the
    raster once. The downstream computation, however, is event-list based and
    does not scan the empty time axis.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Tuple of `(sample_ids, observed_delays)`.
    """
    cd_events = dataset.cd_raster.nonzero(as_tuple=False).cpu().numpy()
    echo_events = dataset.echo_raster.nonzero(as_tuple=False).cpu().numpy()

    cd_order = np.lexsort((cd_events[:, 1], cd_events[:, 0]))
    echo_order = np.lexsort((echo_events[:, 1], echo_events[:, 0]))
    cd_events = cd_events[cd_order]
    echo_events = echo_events[echo_order]

    if cd_events.shape != echo_events.shape or not np.array_equal(cd_events[:, :2], echo_events[:, :2]):
        raise ValueError("Clean event-list benchmark expects one CD and one echo event per sample/channel.")

    sample_ids = echo_events[:, 0].astype(np.int64)
    observed_delays = (echo_events[:, 2] - cd_events[:, 2]).astype(np.int64)
    return sample_ids, observed_delays


def _predict_fair_lif_raster(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict distance with a fair raster-derived LIF soft score.

    The predictor derives observed echo delays from the CD and echo rasters.
    It does not access `true_delay_samples`.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    observed_delays = _observed_delay_matrix(dataset)
    valid = observed_delays >= 0
    delay_error = np.abs(
        observed_delays[:, :, None] - dataset.candidate_delay_samples[None, None, :]
    )
    scores = np.power(LIF_BETA, delay_error) * valid[:, :, None]
    return np.argmax(scores.mean(axis=1), axis=1)


def _predict_fair_binary_raster(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict distance with a fair raster-derived binary score.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    observed_delays = _observed_delay_matrix(dataset)
    valid = observed_delays >= 0
    delay_error = np.abs(
        observed_delays[:, :, None] - dataset.candidate_delay_samples[None, None, :]
    )
    matches = (delay_error <= dataset.binary_tolerance_samples) & valid[:, :, None]
    return np.argmax(matches.mean(axis=1), axis=1)


def _predict_event_list_binary(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict distance using event-list delay voting.

    This method converts rasters into event times, computes one delay per
    channel, then votes for the nearest candidate delay.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    sample_ids, observed_delays = _paired_event_delays_from_nonzero(dataset)
    distance_to_candidates = np.abs(
        observed_delays[:, None] - dataset.candidate_delay_samples[None, :]
    )
    nearest_candidates = np.argmin(distance_to_candidates, axis=1)
    scores = np.zeros((NUM_TEST_SAMPLES, NUM_DELAY_LINES), dtype=np.int32)
    np.add.at(scores, (sample_ids, nearest_candidates), 1)
    return np.argmax(scores, axis=1)


def _dilate_binary_template(template: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a binary template along time using bitwise shifts.

    Args:
        template: Boolean tensor with shape `[samples, channels, time]`.
        radius: Number of samples to dilate on each side.

    Returns:
        Dilated Boolean tensor.
    """
    if radius <= 0:
        return template

    dilated = template.clone()
    for shift in range(1, radius + 1):
        dilated[..., shift:] |= template[..., :-shift]
        dilated[..., :-shift] |= template[..., shift:]
    return dilated


def _raster_to_python_ints(raster: torch.Tensor) -> np.ndarray:
    """Pack each sample/channel Boolean trace into a Python integer bitset.

    Args:
        raster: Boolean raster `[samples, channels, time]`.

    Returns:
        Object array `[samples, channels]` containing Python integer bitsets.
    """
    raster_np = np.ascontiguousarray(raster.cpu().numpy())
    packed = np.packbits(raster_np, axis=-1, bitorder="little")
    flat = packed.reshape(-1, packed.shape[-1])
    bitsets = np.empty(flat.shape[0], dtype=object)
    for index, row in enumerate(flat):
        bitsets[index] = int.from_bytes(row.tobytes(), byteorder="little", signed=False)
    return bitsets.reshape(raster_np.shape[:2])


def _predict_bit_packed_binary(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict distance using packed integer bitsets.

    Each trace is converted into a bitset. Candidate delays are tested using a
    word-level shift and `AND`, instead of dense time-sample comparison.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    dilated_cd = _dilate_binary_template(dataset.cd_raster, dataset.binary_tolerance_samples)
    cd_bitsets = _raster_to_python_ints(dilated_cd)
    echo_bitsets = _raster_to_python_ints(dataset.echo_raster)
    scores = np.zeros((NUM_TEST_SAMPLES, NUM_DELAY_LINES), dtype=np.int32)

    for candidate_index, delay in enumerate(dataset.candidate_delay_samples):
        shifted_echo = echo_bitsets >> int(delay)
        matches = (shifted_echo & cd_bitsets) != 0
        scores[:, candidate_index] = matches.sum(axis=1)
    return np.argmax(scores, axis=1)


def _build_delay_to_candidate_lookup(dataset: CleanSweepDataset) -> np.ndarray:
    """Build a lookup from integer delay to candidate delay index.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Lookup array where invalid delays map to `-1`.
    """
    max_delay = int(dataset.candidate_delay_samples.max())
    lookup = np.full(max_delay + dataset.binary_tolerance_samples + 1, -1, dtype=np.int64)
    for delay in range(lookup.size):
        candidate_index = int(np.argmin(np.abs(dataset.candidate_delay_samples - delay)))
        error = abs(int(dataset.candidate_delay_samples[candidate_index]) - delay)
        if error <= dataset.binary_tolerance_samples:
            lookup[delay] = candidate_index
    return lookup


def _predict_sparse_stack_binary(dataset: CleanSweepDataset) -> np.ndarray:
    """Predict distance using a sparse event-to-candidate accumulator.

    This builds a sparse score stack indexed by `(sample, candidate_delay)`.
    Duplicate entries from multiple frequency channels are coalesced into
    channel votes.

    Args:
        dataset: Clean sweep dataset.

    Returns:
        Predicted candidate indices.
    """
    sample_ids, observed_delays = _paired_event_delays_from_nonzero(dataset)
    lookup = _build_delay_to_candidate_lookup(dataset)
    valid_delay = (observed_delays >= 0) & (observed_delays < lookup.size)
    candidate_ids = np.full_like(observed_delays, -1)
    candidate_ids[valid_delay] = lookup[observed_delays[valid_delay]]
    valid = candidate_ids >= 0

    sample_ids = sample_ids[valid]
    candidate_ids = candidate_ids[valid]
    indices = torch.as_tensor(np.vstack([sample_ids, candidate_ids]), dtype=torch.long)
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    sparse_scores = torch.sparse_coo_tensor(
        indices,
        values,
        size=(NUM_TEST_SAMPLES, NUM_DELAY_LINES),
        check_invariants=False,
    ).coalesce()
    scores = sparse_scores.to_dense().cpu().numpy()
    return np.argmax(scores, axis=1)


def _score_method(
    name: str,
    dataset: CleanSweepDataset,
    predictor: Callable[[CleanSweepDataset], np.ndarray],
    *,
    flops: float,
    sops: float,
) -> MethodResult:
    """Benchmark and score one detector.

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
        rmse_m=float(np.sqrt(np.mean(error**2))),
        nearest_bin_accuracy_percent=float(nearest_correct.mean() * 100.0),
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
    time_steps = dataset.cd_raster.shape[2]
    candidate_delays = dataset.candidate_delay_samples.size
    events = samples * channels
    words = math.ceil(time_steps / 64)
    dense_score_ops = samples * channels * candidate_delays
    return {
        "Fair raster LIF": (dense_score_ops * 8.0, dense_score_ops * 2.0),
        "Fair raster binary": (0.0, dense_score_ops),
        "Event-list binary": (0.0, events * math.ceil(math.log2(candidate_delays))),
        "Bit-packed binary": (0.0, samples * channels * candidate_delays * words),
        "Sparse-stack binary": (0.0, events),
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
    columns = len(results)
    fig, axes = plt.subplots(1, columns, figsize=(4.0 * columns, 4.8), sharex=True, sharey=True)
    if columns == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        ax.scatter(dataset.true_distance_m, result.predicted_distance_m, s=8, alpha=0.35)
        ax.plot([MIN_DISTANCE_M, MAX_DISTANCE_M], [MIN_DISTANCE_M, MAX_DISTANCE_M], color="#111827")
        ax.set_title(f"{result.name}\nMAE={result.mae_m * 100.0:.3f} cm", fontsize=10)
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    axes[0].bar(names, runtime, color="#2563eb")
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("ms")
    axes[1].bar(names, flops, color="#f97316")
    axes[1].set_title("Estimated FLOPs")
    axes[2].bar(names, sops, color="#16a34a")
    axes[2].set_title("Estimated SOPs / integer ops")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, path)


def _write_report(
    dataset: CleanSweepDataset,
    results: list[MethodResult],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the fair binary clean optimisation report.

    Args:
        dataset: Clean sweep dataset.
        results: Method results.
        artifacts: Generated artifact paths.
        elapsed_s: Total script runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    max_delay = int(dataset.candidate_delay_samples.max())
    time_steps = dataset.cd_raster.shape[-1]
    words = math.ceil(time_steps / 64)
    lines = [
        "# Binary Clean Pathway Optimisation",
        "",
        "This report replaces the earlier unfair analytic delay comparison with a fair detector benchmark. Every method receives only the corollary-discharge raster, the echo raster, and the candidate delay bank. The true distance/delay is used only after prediction to calculate the error.",
        "",
        "## Experiment Design",
        "",
        "The input is a clean synthetic FM-sweep spike raster. Each frequency channel has one corollary-discharge spike and one echo spike. The echo spike is shifted by the round-trip delay implied by the target distance, but the predictors are not given this delay directly.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[True distance<br/>used only to generate echo] --> B[Echo raster]",
        "    C[Known CD template] --> D[CD raster]",
        "    D --> E[Detector]",
        "    B --> E",
        "    F[Candidate delay bank] --> E",
        "    E --> G[Predicted delay index]",
        "    G --> H[Predicted distance]",
        "    A --> I[Error calculation only]",
        "    H --> I",
        "```",
        "",
        "This is fair because the detector must infer delay from spike timing in the input rasters. It cannot read `true_delay_samples`, `true_distance_m`, or any precomputed label during prediction.",
        "",
        "## Inputs",
        "",
        "| Input | Shape / meaning | Used by predictors? |",
        "|---|---|---:|",
        f"| CD raster | `{NUM_TEST_SAMPLES} x {SWEEP_CHANNELS} x {time_steps}` Boolean spikes | yes |",
        f"| Echo raster | `{NUM_TEST_SAMPLES} x {SWEEP_CHANNELS} x {time_steps}` Boolean spikes | yes |",
        f"| Candidate delay bank | `{NUM_DELAY_LINES}` candidate delays from `{MIN_DISTANCE_M}` to `{MAX_DISTANCE_M}` m | yes |",
        "| True distance | continuous target distance | error calculation only |",
        "| True delay | generated echo shift | generation/error only, not prediction |",
        "",
        "![Clean sweep rasters](../outputs/binary_clean_optimisation/figures/clean_sweep_rasters.png)",
        "",
        "## Metrics",
        "",
        "- `MAE`: mean absolute distance error in centimetres.",
        "- `RMSE`: root-mean-square distance error in centimetres.",
        "- `Nearest-bin accuracy`: whether the predicted delay line is the candidate delay line closest to the true continuous distance.",
        "- `Runtime`: median wall-clock time over repeated predictor calls on CPU.",
        "- `FLOPs`: rough floating-point operations estimate.",
        "- `SOPs / integer ops`: rough spike/integer operation estimate. These are estimates, not hardware counters.",
        "",
        "## Methods",
        "",
        "### Fair Raster LIF",
        "",
        "Extracts the first CD spike and first echo spike from each channel, computes the observed delay, then uses a soft LIF-like timing score:",
        "",
        "```text",
        "observed_delay_c = echo_time_c - cd_time_c",
        "score_k = mean_c(beta ^ abs(observed_delay_c - candidate_delay_k))",
        "```",
        "",
        "### Fair Raster Binary",
        "",
        "Uses the same observed delays, but each delay line is either matched or not matched:",
        "",
        "```text",
        "score_k = mean_c(abs(observed_delay_c - candidate_delay_k) <= tolerance)",
        "```",
        "",
        "### Event-List Binary",
        "",
        "Converts the rasters into a list of events, computes echo-minus-CD delays, and votes directly for the nearest candidate delay. This removes the dense time axis from the computation.",
        "",
        "```text",
        "events = nonzero(spike_raster)",
        "delay_events = echo_events.time - cd_events.time",
        "score[nearest_candidate(delay_event)] += 1",
        "```",
        "",
        "### Bit-Packed Binary",
        "",
        "Packs every time trace into a Python integer bitset. Candidate delays are tested by shifting the echo bitset and applying a bitwise `AND` with the CD bitset:",
        "",
        "```text",
        "match_k,c = ((echo_bits_c >> candidate_delay_k) AND cd_bits_c) != 0",
        "score_k = sum_c(match_k,c)",
        "```",
        "",
        f"The current trace length is `{time_steps}` samples, so each trace corresponds to about `{words}` 64-bit words conceptually. Python integers are used here for clarity, not maximum speed.",
        "",
        "### Sparse-Stack Binary",
        "",
        "Builds a sparse `(sample, candidate_delay)` score stack from event delays. Multiple channels voting for the same candidate coalesce into a larger score.",
        "",
        "```text",
        "candidate = delay_to_candidate_lookup[observed_delay]",
        "sparse_score[sample, candidate] += 1",
        "prediction = argmax_candidate(sparse_score)",
        "```",
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
        f"| binary tolerance | `{dataset.binary_tolerance_samples}` samples |",
        f"| time samples per trace | `{time_steps}` |",
        "",
        "## Results",
        "",
        "![Accuracy scatter](../outputs/binary_clean_optimisation/figures/accuracy_scatter.png)",
        "",
        "![Runtime and cost](../outputs/binary_clean_optimisation/figures/runtime_cost.png)",
        "",
        "| Method | MAE (cm) | RMSE (cm) | Nearest-bin accuracy (%) | Runtime (ms) | FLOPs | SOPs / integer ops |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.mae_m * 100.0:.4f} | {result.rmse_m * 100.0:.4f} | "
            f"{result.nearest_bin_accuracy_percent:.2f} | {result.runtime_ms:.3f} | "
            f"{result.flops:,.0f} | {result.sops:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The fair raster LIF and fair raster binary methods now consume only input rasters, so they are valid detector baselines.",
            "- The event-list and sparse-stack methods are the best conceptual fit for sparse spikes because they avoid scanning empty time samples.",
            "- The bit-packed method is closer to a real binary hardware implementation, but this Python-int prototype is mainly a correctness and scaling demonstration.",
            "- Dense `unfold` is no longer treated as the main optimised method because it turns sparse spikes into a large dense memory-traffic problem.",
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
    """Run the fair binary clean optimisation benchmark.

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
    methods: list[tuple[str, Callable[[CleanSweepDataset], np.ndarray]]] = [
        ("Fair raster LIF", _predict_fair_lif_raster),
        ("Fair raster binary", _predict_fair_binary_raster),
        ("Event-list binary", _predict_event_list_binary),
        ("Bit-packed binary", _predict_bit_packed_binary),
        ("Sparse-stack binary", _predict_sparse_stack_binary),
    ]
    results = [
        _score_method(
            name,
            dataset,
            predictor,
            flops=estimates[name][0],
            sops=estimates[name][1],
        )
        for name, predictor in methods
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
            "fair_comparison": True,
            "predictors_use_true_delay": False,
        },
        "results": [
            {
                "name": result.name,
                "mae_m": result.mae_m,
                "rmse_m": result.rmse_m,
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
