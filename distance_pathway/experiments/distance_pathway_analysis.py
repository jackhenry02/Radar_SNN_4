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
ECHO_PULSE_SIGMA_SAMPLES = 2.0
SPIKE_NOISE_EVENTS = 3
SPIKE_NOISE_MIN_AMPLITUDE = 0.25
SPIKE_NOISE_MAX_AMPLITUDE = 1.10
SWEEP_CHANNELS = 32
SWEEP_DURATION_S = 0.003
SUSTAINED_CHANNELS = 12
SUSTAINED_REPEATS = 8
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
class SpikeDataset:
    """Synthetic spike-input distance dataset.

    Attributes:
        condition: Name of the signal condition.
        true_distance_m: True target distances in metres.
        ideal_echo_delay_samples: Perfect round-trip echo delay in samples.
        observed_spike_delay_samples: True echo spike plus optional false
            spikes, shape `[samples, spikes]`.
        observed_spike_amplitudes: Per-spike amplitudes, shape
            `[samples, spikes]`.
        candidate_distance_m: Distances represented by delay lines.
        candidate_delay_samples: Delay-line delays in samples.
        num_time_steps: Length of the synthetic timeline.
        has_noise: Whether false spikes are present.
        has_jitter: Whether true echo timing jitter is present.
    """

    condition: str
    true_distance_m: np.ndarray
    ideal_echo_delay_samples: np.ndarray
    observed_spike_delay_samples: np.ndarray
    observed_spike_amplitudes: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    num_time_steps: int
    has_noise: bool
    has_jitter: bool


@dataclass
class SweepSpikeDataset:
    """Synthetic FM-sweep spike-input dataset.

    Attributes:
        condition: Name of the signal condition.
        true_distance_m: True target distances in metres.
        ideal_echo_delay_samples: Perfect round-trip echo delay in samples.
        channel_offsets_samples: Per-frequency-channel sweep offsets.
        observed_spike_times: Absolute spike times for each sample, channel,
            and spike event, shape `[samples, channels, spikes]`.
        observed_spike_amplitudes: Per-spike amplitudes, shape
            `[samples, channels, spikes]`.
        candidate_distance_m: Distances represented by delay lines.
        candidate_delay_samples: Delay-line delays in samples.
        num_time_steps: Length of the synthetic timeline.
        has_noise: Whether false spikes are present.
        has_jitter: Whether true echo timing jitter is present.
    """

    condition: str
    true_distance_m: np.ndarray
    ideal_echo_delay_samples: np.ndarray
    channel_offsets_samples: np.ndarray
    observed_spike_times: np.ndarray
    observed_spike_amplitudes: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    num_time_steps: int
    has_noise: bool
    has_jitter: bool


@dataclass
class SustainedPitchDataset:
    """Synthetic sustained-pitch spike-input dataset.

    Attributes:
        true_distance_m: True target distances in metres.
        candidate_distance_m: Distances represented by delay lines.
        candidate_delay_samples: Delay-line delays in samples.
        periods_samples: Per-channel spike periods in samples.
        offset_pairs_samples: Pairwise echo/corollary cycle offsets for each
            channel, shape `[channels, repeat_pairs]`.
        num_time_steps: Length of the synthetic timeline.
    """

    true_distance_m: np.ndarray
    candidate_distance_m: np.ndarray
    candidate_delay_samples: np.ndarray
    periods_samples: np.ndarray
    offset_pairs_samples: np.ndarray
    num_time_steps: int


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
    pulse_radius = int(np.ceil(4.0 * ECHO_PULSE_SIGMA_SAMPLES))
    pulse_offsets = np.arange(-pulse_radius, pulse_radius + 1)
    pulse_shape = np.exp(-0.5 * (pulse_offsets / ECHO_PULSE_SIGMA_SAMPLES) ** 2)
    for sample_index, echo_index in enumerate(echo_indices):
        start = max(0, echo_index - pulse_radius)
        stop = min(num_time_steps, echo_index + pulse_radius + 1)
        shape_start = start - (echo_index - pulse_radius)
        shape_stop = shape_start + (stop - start)
        waveform[sample_index, start:stop] += pulse_shape[shape_start:shape_stop]
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


def _make_spike_dataset(condition: str, *, has_noise: bool, has_jitter: bool, seed_offset: int) -> SpikeDataset:
    """Create one spiking-input benchmark condition.

    Args:
        condition: Human-readable condition name.
        has_noise: Whether to add false spikes.
        has_jitter: Whether to jitter the true echo spike.
        seed_offset: Offset used for condition-specific randomness.

    Returns:
        Spike dataset for the requested condition.
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

    spike_count = 1 + (SPIKE_NOISE_EVENTS if has_noise else 0)
    observed_delays = np.zeros((true_distance_m.size, spike_count), dtype=np.int64)
    observed_amplitudes = np.zeros((true_distance_m.size, spike_count), dtype=np.float64)
    observed_delays[:, 0] = echo_delay
    observed_amplitudes[:, 0] = 1.0
    if has_noise:
        observed_delays[:, 1:] = rng.integers(1, max_candidate + 1, size=(true_distance_m.size, SPIKE_NOISE_EVENTS))
        observed_amplitudes[:, 1:] = rng.uniform(
            SPIKE_NOISE_MIN_AMPLITUDE,
            SPIKE_NOISE_MAX_AMPLITUDE,
            size=(true_distance_m.size, SPIKE_NOISE_EVENTS),
        )

    max_observed = int(observed_delays.max())
    num_time_steps = TX_INDEX + max(max_candidate, max_observed) + 80
    return SpikeDataset(
        condition=condition,
        true_distance_m=true_distance_m,
        ideal_echo_delay_samples=ideal_echo_delay,
        observed_spike_delay_samples=observed_delays,
        observed_spike_amplitudes=observed_amplitudes,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        num_time_steps=num_time_steps,
        has_noise=has_noise,
        has_jitter=has_jitter,
    )


def _all_spike_datasets() -> list[SpikeDataset]:
    """Return the four requested spiking-input benchmark conditions."""
    return [
        _make_spike_dataset("Clean perfect", has_noise=False, has_jitter=False, seed_offset=400),
        _make_spike_dataset("Added noise", has_noise=True, has_jitter=False, seed_offset=500),
        _make_spike_dataset("Added jitter", has_noise=False, has_jitter=True, seed_offset=600),
        _make_spike_dataset("Noise + jitter", has_noise=True, has_jitter=True, seed_offset=700),
    ]


def _make_sweep_spike_dataset(
    condition: str,
    *,
    has_noise: bool,
    has_jitter: bool,
    seed_offset: int,
) -> SweepSpikeDataset:
    """Create one FM-sweep spiking-input benchmark condition.

    The corollary discharge is a sweep: different frequency channels fire at
    different times. The echo is the same sweep shifted by the target delay.
    Each channel therefore provides an independent estimate of the same
    distance.

    Args:
        condition: Human-readable condition name.
        has_noise: Whether to add false spikes per frequency channel.
        has_jitter: Whether to jitter the true echo delay.
        seed_offset: Offset used for condition-specific randomness.

    Returns:
        Sweep spike dataset for the requested condition.
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
    sweep_samples = int(round(SWEEP_DURATION_S * SAMPLE_RATE_HZ))
    channel_offsets = np.rint(np.linspace(0, sweep_samples, SWEEP_CHANNELS)).astype(np.int64)

    spike_count = 1 + (SPIKE_NOISE_EVENTS if has_noise else 0)
    observed_times = np.zeros((true_distance_m.size, SWEEP_CHANNELS, spike_count), dtype=np.int64)
    observed_amplitudes = np.zeros((true_distance_m.size, SWEEP_CHANNELS, spike_count), dtype=np.float64)
    true_echo_times = TX_INDEX + channel_offsets[None, :] + echo_delay[:, None]
    observed_times[:, :, 0] = true_echo_times
    observed_amplitudes[:, :, 0] = 1.0
    if has_noise:
        false_relative_delays = rng.integers(
            1,
            max_candidate + 1,
            size=(true_distance_m.size, SWEEP_CHANNELS, SPIKE_NOISE_EVENTS),
        )
        observed_times[:, :, 1:] = TX_INDEX + channel_offsets[None, :, None] + false_relative_delays
        observed_amplitudes[:, :, 1:] = rng.uniform(
            SPIKE_NOISE_MIN_AMPLITUDE,
            SPIKE_NOISE_MAX_AMPLITUDE,
            size=(true_distance_m.size, SWEEP_CHANNELS, SPIKE_NOISE_EVENTS),
        )

    max_observed = int(observed_times.max())
    num_time_steps = max_observed + 80
    return SweepSpikeDataset(
        condition=condition,
        true_distance_m=true_distance_m,
        ideal_echo_delay_samples=ideal_echo_delay,
        channel_offsets_samples=channel_offsets,
        observed_spike_times=observed_times,
        observed_spike_amplitudes=observed_amplitudes,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        num_time_steps=num_time_steps,
        has_noise=has_noise,
        has_jitter=has_jitter,
    )


def _all_sweep_spike_datasets() -> list[SweepSpikeDataset]:
    """Return the four requested sweep-spiking benchmark conditions."""
    return [
        _make_sweep_spike_dataset("Clean perfect", has_noise=False, has_jitter=False, seed_offset=800),
        _make_sweep_spike_dataset("Added noise", has_noise=True, has_jitter=False, seed_offset=900),
        _make_sweep_spike_dataset("Added jitter", has_noise=False, has_jitter=True, seed_offset=1000),
        _make_sweep_spike_dataset("Noise + jitter", has_noise=True, has_jitter=True, seed_offset=1100),
    ]


def _make_sustained_pitch_dataset() -> SustainedPitchDataset:
    """Create a clean sustained-pitch spike-train dataset.

    Returns:
        Dataset containing repeated pitch spike trains for multiple channels.
    """
    true_distance_m = _base_distances()
    candidate_distance_m = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, NUM_DELAY_LINES)
    candidate_delay_samples = np.rint(
        (2.0 * candidate_distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ
    ).astype(np.int64)
    # Shorter periods correspond to higher pitch/frequency channels.
    periods_samples = np.rint(np.linspace(4, 28, SUSTAINED_CHANNELS)).astype(np.int64)
    repeat_indices = np.arange(SUSTAINED_REPEATS)
    offset_pairs = []
    for period in periods_samples:
        corollary_times = repeat_indices * period
        echo_times = repeat_indices * period
        offset_pairs.append((echo_times[:, None] - corollary_times[None, :]).reshape(-1))
    offset_pairs_samples = np.stack(offset_pairs, axis=0)
    max_delay = int(candidate_delay_samples.max())
    max_period_span = int(periods_samples.max() * (SUSTAINED_REPEATS - 1))
    num_time_steps = TX_INDEX + max_delay + max_period_span + 80
    return SustainedPitchDataset(
        true_distance_m=true_distance_m,
        candidate_distance_m=candidate_distance_m,
        candidate_delay_samples=candidate_delay_samples,
        periods_samples=periods_samples,
        offset_pairs_samples=offset_pairs_samples,
        num_time_steps=num_time_steps,
    )


def _candidate_waveform_amplitudes(dataset: Dataset) -> np.ndarray:
    """Sample the noisy echo waveform at each candidate delay line.

    Args:
        dataset: Benchmark dataset.

    Returns:
        Candidate waveform amplitudes, shape `[samples, delay_lines]`.
    """
    candidate_indices = TX_INDEX + dataset.candidate_delay_samples
    return dataset.waveform[:, candidate_indices]


def _delay_error_matrix(dataset: Dataset) -> np.ndarray:
    """Compute true echo delay error against every candidate delay.

    Args:
        dataset: Benchmark dataset.

    Returns:
        Absolute delay mismatch matrix `[samples, delay_lines]`.
    """
    return np.abs(dataset.echo_delay_samples[:, None] - dataset.candidate_delay_samples[None, :])


def _predict_lif(dataset: Dataset) -> np.ndarray:
    """Predict distance with a LIF soft-coincidence detector bank."""
    delay_error = _delay_error_matrix(dataset)
    amplitudes = np.maximum(_candidate_waveform_amplitudes(dataset), 0.0)
    beta = 0.982
    input_weight = 0.62
    scores = amplitudes * input_weight * (1.0 + np.power(beta, delay_error))
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_rf(dataset: Dataset) -> np.ndarray:
    """Predict distance with an RF-style coincidence detector bank."""
    delay_error = _delay_error_matrix(dataset)
    amplitudes = np.maximum(_candidate_waveform_amplitudes(dataset), 0.0)
    input_weight = 0.62
    tau_samples = 18.0
    omega = 2.0 * np.pi / 18.0
    afterpotential = np.exp(-delay_error / tau_samples) * np.cos(omega * delay_error)
    scores = amplitudes * input_weight * (1.0 + afterpotential)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_binary(dataset: Dataset) -> np.ndarray:
    """Predict distance with a binary delay-line coincidence detector."""
    delay_error = _delay_error_matrix(dataset)
    amplitudes = _candidate_waveform_amplitudes(dataset)
    tolerance_samples = 2
    amplitude_threshold = 0.5
    scores = np.where((delay_error <= tolerance_samples) & (amplitudes >= amplitude_threshold), amplitudes, -np.inf)
    no_match = ~np.isfinite(scores).any(axis=1)
    if np.any(no_match):
        scores[no_match] = amplitudes[no_match]
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _score_detector(name: str, dataset: Dataset, predictor: Callable[[Dataset], np.ndarray]) -> DetectorResult:
    """Benchmark and score one detector on one condition."""
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


def _spike_delay_error_tensor(dataset: SpikeDataset) -> np.ndarray:
    """Compute observed spike delay error against candidate delays.

    Args:
        dataset: Spiking-input dataset.

    Returns:
        Absolute mismatch tensor `[samples, spikes, delay_lines]`.
    """
    return np.abs(
        dataset.observed_spike_delay_samples[:, :, None] - dataset.candidate_delay_samples[None, None, :]
    )


def _predict_spike_lif(dataset: SpikeDataset) -> np.ndarray:
    """Predict distance from onset spikes with a LIF soft-coincidence bank."""
    delay_error = _spike_delay_error_tensor(dataset)
    amplitudes = dataset.observed_spike_amplitudes[:, :, None]
    beta = 0.982
    input_weight = 0.62
    scores = (amplitudes * input_weight * (1.0 + np.power(beta, delay_error))).max(axis=1)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_spike_rf(dataset: SpikeDataset) -> np.ndarray:
    """Predict distance from onset spikes with an RF-style detector bank."""
    delay_error = _spike_delay_error_tensor(dataset)
    amplitudes = dataset.observed_spike_amplitudes[:, :, None]
    input_weight = 0.62
    tau_samples = 18.0
    omega = 2.0 * np.pi / 18.0
    afterpotential = np.exp(-delay_error / tau_samples) * np.cos(omega * delay_error)
    scores = (amplitudes * input_weight * (1.0 + afterpotential)).max(axis=1)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_spike_binary(dataset: SpikeDataset) -> np.ndarray:
    """Predict distance from onset spikes with binary coincidence matching."""
    delay_error = _spike_delay_error_tensor(dataset)
    amplitudes = dataset.observed_spike_amplitudes[:, :, None]
    tolerance_samples = 2
    scores = np.where(delay_error <= tolerance_samples, amplitudes, -np.inf).max(axis=1)
    no_match = ~np.isfinite(scores).any(axis=1)
    if np.any(no_match):
        nearest_error = delay_error[no_match].min(axis=1)
        scores[no_match] = -nearest_error
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _score_spike_detector(
    name: str,
    dataset: SpikeDataset,
    predictor: Callable[[SpikeDataset], np.ndarray],
) -> DetectorResult:
    """Benchmark and score one detector on one spiking-input condition.

    Args:
        name: Detector name.
        dataset: Spiking-input dataset.
        predictor: Prediction function.

    Returns:
        Detector result summary.
    """
    runtime_s = _median_runtime_s(lambda: predictor(dataset))
    predicted = predictor(dataset)
    error = predicted - dataset.true_distance_m
    abs_error = np.abs(error)
    operations = (
        dataset.true_distance_m.size
        * dataset.candidate_distance_m.size
        * dataset.observed_spike_delay_samples.shape[1]
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


def _sweep_delay_error_tensor(dataset: SweepSpikeDataset) -> np.ndarray:
    """Compute sweep spike delay error against candidate delays.

    Args:
        dataset: FM-sweep spiking dataset.

    Returns:
        Absolute mismatch tensor `[samples, channels, spikes, delay_lines]`.
    """
    relative_spike_delays = dataset.observed_spike_times - TX_INDEX - dataset.channel_offsets_samples[None, :, None]
    return np.abs(relative_spike_delays[:, :, :, None] - dataset.candidate_delay_samples[None, None, None, :])


def _predict_sweep_lif(dataset: SweepSpikeDataset) -> np.ndarray:
    """Predict distance from FM-sweep spikes with averaged LIF channel scores."""
    delay_error = _sweep_delay_error_tensor(dataset)
    amplitudes = dataset.observed_spike_amplitudes[:, :, :, None]
    beta = 0.982
    input_weight = 0.62
    per_channel_scores = (amplitudes * input_weight * (1.0 + np.power(beta, delay_error))).max(axis=2)
    scores = per_channel_scores.mean(axis=1)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_sweep_rf(dataset: SweepSpikeDataset) -> np.ndarray:
    """Predict distance from FM-sweep spikes with averaged RF channel scores."""
    delay_error = _sweep_delay_error_tensor(dataset)
    amplitudes = dataset.observed_spike_amplitudes[:, :, :, None]
    input_weight = 0.62
    tau_samples = 18.0
    omega = 2.0 * np.pi / 18.0
    afterpotential = np.exp(-delay_error / tau_samples) * np.cos(omega * delay_error)
    per_channel_scores = (amplitudes * input_weight * (1.0 + afterpotential)).max(axis=2)
    scores = per_channel_scores.mean(axis=1)
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_sweep_binary(dataset: SweepSpikeDataset) -> np.ndarray:
    """Predict distance from FM-sweep spikes with averaged binary matches."""
    delay_error = _sweep_delay_error_tensor(dataset)
    delay_spacing = np.median(np.diff(dataset.candidate_delay_samples))
    tolerance_samples = int(np.ceil(delay_spacing / 2.0))
    per_channel_scores = (delay_error <= tolerance_samples).max(axis=2).astype(np.float64)
    scores = per_channel_scores.mean(axis=1)
    no_match = scores.max(axis=1) <= 0.0
    if np.any(no_match):
        nearest_error = delay_error[no_match].min(axis=(1, 2))
        scores[no_match] = -nearest_error
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _score_sweep_detector(
    name: str,
    dataset: SweepSpikeDataset,
    predictor: Callable[[SweepSpikeDataset], np.ndarray],
) -> DetectorResult:
    """Benchmark and score one detector on one sweep-spiking condition.

    Args:
        name: Detector name.
        dataset: FM-sweep spiking dataset.
        predictor: Prediction function.

    Returns:
        Detector result summary.
    """
    runtime_s = _median_runtime_s(lambda: predictor(dataset), repeats=10)
    predicted = predictor(dataset)
    error = predicted - dataset.true_distance_m
    abs_error = np.abs(error)
    operations = (
        dataset.true_distance_m.size
        * dataset.channel_offsets_samples.size
        * dataset.candidate_distance_m.size
        * dataset.observed_spike_times.shape[2]
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


def _true_delay_samples_from_distance(distance_m: np.ndarray) -> np.ndarray:
    """Convert distance to round-trip delay samples."""
    return np.rint((2.0 * distance_m / SPEED_OF_SOUND_M_S) * SAMPLE_RATE_HZ).astype(np.int64)


def _predict_sustained_lif(dataset: SustainedPitchDataset) -> np.ndarray:
    """Predict distance from sustained pitch trains using LIF-like scores."""
    true_delay = _true_delay_samples_from_distance(dataset.true_distance_m)
    scores = np.zeros((dataset.true_distance_m.size, dataset.candidate_distance_m.size), dtype=np.float64)
    beta = 0.982
    for channel_offsets in dataset.offset_pairs_samples:
        relative_delays = true_delay[:, None] + channel_offsets[None, :]
        error = np.abs(relative_delays[:, :, None] - dataset.candidate_delay_samples[None, None, :])
        scores += np.power(beta, error).sum(axis=1) / SUSTAINED_REPEATS
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_sustained_rf(dataset: SustainedPitchDataset) -> np.ndarray:
    """Predict distance from sustained pitch trains using RF-like scores."""
    true_delay = _true_delay_samples_from_distance(dataset.true_distance_m)
    scores = np.zeros((dataset.true_distance_m.size, dataset.candidate_distance_m.size), dtype=np.float64)
    tau_samples = 18.0
    for period, channel_offsets in zip(dataset.periods_samples, dataset.offset_pairs_samples):
        relative_delays = true_delay[:, None] + channel_offsets[None, :]
        error = np.abs(relative_delays[:, :, None] - dataset.candidate_delay_samples[None, None, :])
        omega = 2.0 * np.pi / period
        afterpotential = np.exp(-error / tau_samples) * np.cos(omega * error)
        scores += afterpotential.sum(axis=1) / SUSTAINED_REPEATS
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _predict_sustained_binary(dataset: SustainedPitchDataset) -> np.ndarray:
    """Predict distance from sustained pitch trains using binary coincidences."""
    true_delay = _true_delay_samples_from_distance(dataset.true_distance_m)
    scores = np.zeros((dataset.true_distance_m.size, dataset.candidate_distance_m.size), dtype=np.float64)
    tolerance_samples = int(np.ceil(np.median(np.diff(dataset.candidate_delay_samples)) / 2.0))
    for channel_offsets in dataset.offset_pairs_samples:
        relative_delays = true_delay[:, None] + channel_offsets[None, :]
        error = np.abs(relative_delays[:, :, None] - dataset.candidate_delay_samples[None, None, :])
        scores += (error <= tolerance_samples).sum(axis=1) / SUSTAINED_REPEATS
    return dataset.candidate_distance_m[np.argmax(scores, axis=1)]


def _score_sustained_detector(
    name: str,
    dataset: SustainedPitchDataset,
    predictor: Callable[[SustainedPitchDataset], np.ndarray],
) -> DetectorResult:
    """Benchmark and score one sustained-pitch detector.

    Args:
        name: Detector name.
        dataset: Sustained-pitch dataset.
        predictor: Prediction function.

    Returns:
        Detector result summary.
    """
    runtime_s = _median_runtime_s(lambda: predictor(dataset), repeats=10)
    predicted = predictor(dataset)
    error = predicted - dataset.true_distance_m
    abs_error = np.abs(error)
    operations = (
        dataset.true_distance_m.size
        * dataset.candidate_distance_m.size
        * dataset.periods_samples.size
        * SUSTAINED_REPEATS
        * SUSTAINED_REPEATS
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
        condition="Clean sustained pitches",
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


def _plot_sweep_raster(path: Path) -> str:
    """Plot the synthetic FM-sweep spike input used for the sweep test.

    Args:
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    example = _make_sweep_spike_dataset("Raster example", has_noise=True, has_jitter=True, seed_offset=1234)
    sample_index = 0
    true_distance = example.true_distance_m[sample_index]
    time_ms = np.arange(example.num_time_steps) / SAMPLE_RATE_HZ * 1_000.0
    corollary_times = TX_INDEX + example.channel_offsets_samples
    true_echo_times = example.observed_spike_times[sample_index, :, 0]
    false_times = example.observed_spike_times[sample_index, :, 1:]
    frequencies_khz = np.linspace(18.0, 2.0, SWEEP_CHANNELS)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for channel, frequency_khz in enumerate(frequencies_khz):
        axes[0].vlines(corollary_times[channel] / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.18, frequency_khz + 0.18, color="#2563eb")
        axes[0].vlines(true_echo_times[channel] / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.18, frequency_khz + 0.18, color="#dc2626")
    axes[0].set_title(f"Clean FM-sweep spike raster plus echo shift, target={true_distance:.2f} m")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_ylim(1.0, 19.0)
    axes[0].legend(["corollary discharge", "echo"], loc="upper right")

    for channel in range(SWEEP_CHANNELS):
        frequency_khz = frequencies_khz[channel]
        axes[1].vlines(corollary_times[channel] / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.18, frequency_khz + 0.18, color="#2563eb")
        axes[1].vlines(true_echo_times[channel] / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.18, frequency_khz + 0.18, color="#dc2626")
        for false_time in false_times[channel]:
            axes[1].vlines(false_time / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.16, frequency_khz + 0.16, color="#f97316", alpha=0.65)
    axes[1].set_title("Same raster with false spikes used for noisy spiking conditions")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("frequency (kHz)")
    axes[1].set_ylim(1.0, 19.0)
    axes[1].set_xlim(0.0, min(time_ms[-1], 35.0))
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.2)
    return save_figure(fig, path)


def _plot_sustained_pitch_raster(dataset: SustainedPitchDataset, path: Path) -> str:
    """Plot a sustained-pitch corollary/echo raster.

    Args:
        dataset: Sustained pitch dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    sample_index = 0
    true_distance = dataset.true_distance_m[sample_index]
    true_delay = _true_delay_samples_from_distance(np.array([true_distance]))[0]
    repeat_indices = np.arange(SUSTAINED_REPEATS)
    frequencies_khz = SAMPLE_RATE_HZ / dataset.periods_samples / 1_000.0
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for channel, period in enumerate(dataset.periods_samples):
        frequency_khz = frequencies_khz[channel]
        corollary_times = TX_INDEX + repeat_indices * period
        echo_times = corollary_times + true_delay
        ax.vlines(corollary_times / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.18, frequency_khz + 0.18, color="#2563eb")
        ax.vlines(echo_times / SAMPLE_RATE_HZ * 1_000.0, frequency_khz - 0.18, frequency_khz + 0.18, color="#dc2626")
    ax.set_title(f"Sustained-pitch spike raster, target={true_distance:.2f} m")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_ylim(1.5, frequencies_khz.max() + 1.0)
    ax.set_xlim(0.0, 24.0)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(["corollary discharge", "echo"], loc="upper right")
    return save_figure(fig, path)


def _plot_sustained_response(dataset: SustainedPitchDataset, path: Path) -> str:
    """Plot sustained-pitch detector responses for one example target.

    Args:
        dataset: Sustained pitch dataset.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    example_distance = np.array([2.7])
    example = SustainedPitchDataset(
        true_distance_m=example_distance,
        candidate_distance_m=dataset.candidate_distance_m,
        candidate_delay_samples=dataset.candidate_delay_samples,
        periods_samples=dataset.periods_samples,
        offset_pairs_samples=dataset.offset_pairs_samples,
        num_time_steps=dataset.num_time_steps,
    )
    true_delay = _true_delay_samples_from_distance(example_distance)
    lif_scores = np.zeros(dataset.candidate_distance_m.size)
    rf_scores = np.zeros(dataset.candidate_distance_m.size)
    binary_scores = np.zeros(dataset.candidate_distance_m.size)
    tolerance = int(np.ceil(np.median(np.diff(dataset.candidate_delay_samples)) / 2.0))
    for period, offsets in zip(dataset.periods_samples, dataset.offset_pairs_samples):
        relative_delays = true_delay[:, None] + offsets[None, :]
        error = np.abs(relative_delays[:, :, None] - dataset.candidate_delay_samples[None, None, :])
        lif_scores += np.power(0.982, error[0]).sum(axis=0) / SUSTAINED_REPEATS
        rf_scores += (
            np.exp(-error[0] / 18.0) * np.cos((2.0 * np.pi / period) * error[0])
        ).sum(axis=0) / SUSTAINED_REPEATS
        binary_scores += (error[0] <= tolerance).sum(axis=0) / SUSTAINED_REPEATS
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(dataset.candidate_distance_m, lif_scores / np.max(lif_scores), label="LIF", linewidth=2.0)
    ax.plot(dataset.candidate_distance_m, rf_scores / np.max(np.abs(rf_scores)), label="RF", linewidth=2.0)
    ax.plot(dataset.candidate_distance_m, binary_scores / np.max(binary_scores), label="binary", linewidth=2.0)
    ax.axvline(float(example_distance[0]), color="#111827", linestyle="--", label="true distance")
    ax.set_title("Sustained-pitch delay-bank response")
    ax.set_xlabel("candidate distance (m)")
    ax.set_ylabel("normalized score")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
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


def _plot_accuracy_from_distances(
    results: list[DetectorResult],
    true_distance_m: np.ndarray,
    path: Path,
    *,
    title: str,
) -> str:
    """Plot true-vs-predicted distance for result objects.

    Args:
        results: Detector results.
        true_distance_m: True distances in metres.
        path: Output figure path.
        title: Figure title.

    Returns:
        Saved figure path.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.8), sharex=True, sharey=True)
    fig.suptitle(title)
    for ax, result in zip(axes, results):
        ax.scatter(true_distance_m, result.predicted_distance_m, s=8, alpha=0.45)
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
    spike_datasets: list[SpikeDataset],
    spike_results_by_condition: dict[str, list[DetectorResult]],
    sweep_datasets: list[SweepSpikeDataset],
    sweep_results_by_condition: dict[str, list[DetectorResult]],
    sustained_dataset: SustainedPitchDataset,
    sustained_results: list[DetectorResult],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the condition-based LIF/RF/binary optimization report."""
    lines = [
        "# Distance Pathway 2: Accuracy And Optimisation Testing",
        "",
        "This report compares LIF, RF, and binary delay-line coincidence detectors under four conditions: clean, added noise, added jitter, and noise plus jitter.",
        "",
        "There are now two input cases:",
        "",
        "- **Waveform input:** each delay line samples a synthetic echo waveform, with optional additive white noise.",
        "- **Spiking input:** the upstream system has already extracted onset spikes, with optional false spikes.",
        "",
        "# Part A: Waveform Input",
        "",
        "## Signal Conditions",
        "",
            "| Condition | True echo jitter | Additive white noise |",
        "|---|---:|---:|",
    ]
    for dataset in datasets:
        lines.append(
            f"| {dataset.condition} | `{dataset.has_jitter}` | `{dataset.has_noise}` |"
        )
    lines.extend(
        [
            "",
            f"Noise here means additive Gaussian white noise on the synthetic echo waveform, with an approximate SNR of `{WHITE_NOISE_SNR_DB:.1f} dB` relative to the unit echo pulse. The echo itself is a narrow Gaussian pulse with sigma `{ECHO_PULSE_SIGMA_SAMPLES:.1f}` samples, so nearby delay lines still see a graded amplitude. Jitter means Gaussian timing jitter on the true echo pulse.",
            "",
            "## Detector Equations",
            "",
            "For all detectors, the candidate delay lines sample the echo waveform at their expected echo-arrival time:",
            "",
            "```text",
            "a_k = max(0, waveform[call_time + delay_candidate[k]])",
            "delta_k = abs(delay_echo - delay_candidate[k])",
            "```",
            "",
            "The LIF and RF detectors use the sampled amplitude as their input drive. The binary detector checks whether the candidate sample crosses a fixed amplitude threshold and is close enough in time.",
            "",
            "```text",
            "LIF:    score_k = a_k * w * (1 + beta^delta_k)",
            "RF:     score_k = a_k * w * (1 + exp(-delta_k/tau_rf) * cos(omega_rf * delta_k))",
            "Binary: match_k = 1 if waveform[call_time + delay_candidate[k]] >= threshold and delta_k <= tolerance",
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
            f"| white noise SNR | `{WHITE_NOISE_SNR_DB:.1f} dB` |",
            f"| echo pulse sigma | `{ECHO_PULSE_SIGMA_SAMPLES:.1f} samples` |",
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
            "## Waveform-Input Interpretation",
            "",
            "- Clean perfect signals are essentially a delay quantisation problem, so LIF and binary should be close.",
            "- Jitter tests timing tolerance. LIF remains a useful soft detector because the membrane trace decays smoothly with timing mismatch.",
            "- Noise tests robustness to additive waveform fluctuations. In this simplified setup, delay lines sample the noisy waveform at candidate arrival times.",
            "- RF remains biologically interesting, but its oscillatory side lobes are a weakness for this specific pure-delay task.",
            "",
            "# Part B: Spiking Input",
            "",
            "The second benchmark assumes an earlier front end has already converted the echo into onset spikes. This is closer to the simplified pulse model used before, but it is now explicitly separated from the waveform-input case.",
            "",
            "## Spiking Signal Conditions",
            "",
            "| Condition | True echo jitter | False spikes |",
            "|---|---:|---:|",
        ]
    )
    for dataset in spike_datasets:
        lines.append(f"| {dataset.condition} | `{dataset.has_jitter}` | `{dataset.has_noise}` |")
    lines.extend(
        [
            "",
            f"Spiking noise means `{SPIKE_NOISE_EVENTS}` extra false onset spikes per sample, with amplitudes sampled from `{SPIKE_NOISE_MIN_AMPLITUDE}` to `{SPIKE_NOISE_MAX_AMPLITUDE}`. This tests false-onset robustness after spike extraction.",
            "",
            "## Spiking Detector Equations",
            "",
            "For observed spike `p` and candidate delay `k`:",
            "",
            "```text",
            "delta_p,k = abs(delay_spike[p] - delay_candidate[k])",
            "```",
            "",
            "The detector equations are:",
            "",
            "```text",
            "LIF:    score_k = max_p amplitude_p * w * (1 + beta^delta_p,k)",
            "RF:     score_k = max_p amplitude_p * w * (1 + exp(-delta_p,k/tau_rf) * cos(omega_rf * delta_p,k))",
            "Binary: score_k = max_p amplitude_p if delta_p,k <= tolerance else no match",
            "```",
            "",
            "## Spiking Accuracy Across Conditions",
            "",
            "![Spiking condition MAE](../outputs/accuracy_optimisation/figures/spiking_condition_mae.png)",
            "",
            "| Condition | Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, results in spike_results_by_condition.items():
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
            "## Spiking Hardest-Condition Plots",
            "",
            "The scatter, histogram, and cost plots below use the spiking `Noise + jitter` condition.",
            "",
            "![Spiking accuracy scatter](../outputs/accuracy_optimisation/figures/spiking_accuracy_scatter_noise_jitter.png)",
            "",
            "![Spiking error histogram](../outputs/accuracy_optimisation/figures/spiking_error_histogram_noise_jitter.png)",
            "",
            "![Spiking cost comparison](../outputs/accuracy_optimisation/figures/spiking_cost_comparison.png)",
            "",
            "## Overall Interpretation",
            "",
            "- Waveform-input noise tests amplitude corruption before onset extraction.",
            "- Spiking-input noise tests false onset events after onset extraction.",
            "- Clean and jitter-only spiking inputs are mostly nearest-delay matching, so LIF and binary can be identical.",
            "- False spikes are where LIF can outperform binary because amplitude-weighted soft coincidence can partially reduce the effect of isolated false events.",
            "- The next realistic test should connect the final cochlea front end to this spiking-input pathway, so the spike statistics come from the actual cochlea rather than being synthetic.",
            "",
            "# Part C: FM-Sweep Spiking Input",
            "",
            "The third benchmark tests the robustness idea that the FM sweep gives multiple measurements of the same distance. A swept corollary discharge is used as the reference, and the echo is the same spike sweep shifted by the target delay.",
            "",
            "Each frequency channel has a different corollary-discharge time. For a candidate distance, the detector compares each echo-channel spike against the corresponding delayed corollary channel. Candidate scores are averaged across channels before choosing the final distance.",
            "",
            "![FM sweep raster](../outputs/accuracy_optimisation/figures/sweep_spike_raster.png)",
            "",
            "## Sweep-Spiking Signal Conditions",
            "",
            "| Condition | True echo jitter | False spikes per channel |",
            "|---|---:|---:|",
        ]
    )
    for dataset in sweep_datasets:
        noise_description = SPIKE_NOISE_EVENTS if dataset.has_noise else 0
        lines.append(f"| {dataset.condition} | `{dataset.has_jitter}` | `{noise_description}` |")
    lines.extend(
        [
            "",
            "The sweep is simplified as one spike per frequency channel in the corollary discharge, and one shifted echo spike per channel. This is not a full cochlear raster yet, but it tests the key redundancy idea.",
            "",
            "| Sweep parameter | Value |",
            "|---|---:|",
            f"| sweep channels | `{SWEEP_CHANNELS}` |",
            f"| sweep duration | `{SWEEP_DURATION_S * 1_000.0:.1f} ms` |",
            f"| false spikes per noisy channel | `{SPIKE_NOISE_EVENTS}` |",
            "",
            "## Sweep-Spiking Detector Equations",
            "",
            "For channel `c`, observed spike `p`, and candidate delay `k`:",
            "",
            "```text",
            "relative_delay_c,p = spike_time_c,p - call_time - channel_offset_c",
            "delta_c,p,k = abs(relative_delay_c,p - delay_candidate[k])",
            "```",
            "",
            "The per-channel scores are averaged across frequency channels:",
            "",
            "```text",
            "score_k = mean_c(max_p detector_score(delta_c,p,k))",
            "binary_score_k = mean_c(any_p(delta_c,p,k <= half_delay_bin_width))",
            "distance = distance_candidate[argmax_k(score_k)]",
            "```",
            "",
            "## Sweep-Spiking Accuracy Across Conditions",
            "",
            "![Sweep condition MAE](../outputs/accuracy_optimisation/figures/sweep_condition_mae.png)",
            "",
            "| Condition | Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, results in sweep_results_by_condition.items():
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
            "## Sweep-Spiking Hardest-Condition Plots",
            "",
            "The scatter, histogram, and cost plots below use the sweep-spiking `Noise + jitter` condition.",
            "",
            "![Sweep accuracy scatter](../outputs/accuracy_optimisation/figures/sweep_accuracy_scatter_noise_jitter.png)",
            "",
            "![Sweep error histogram](../outputs/accuracy_optimisation/figures/sweep_error_histogram_noise_jitter.png)",
            "",
            "![Sweep cost comparison](../outputs/accuracy_optimisation/figures/sweep_cost_comparison.png)",
            "",
            "## Sweep Interpretation",
            "",
            "- Averaging across sweep channels directly tests the idea that the FM sweep supplies repeated distance measurements.",
            "- Random false spikes are less damaging when they are independent across channels, because they do not consistently support the same candidate delay.",
            "- This should especially help the binary detector, whose single-channel version is fast but brittle.",
            "- The next step is to replace this hand-made sweep raster with spikes from the final cochlea front end.",
            "",
            "# Part D: Clean Sustained-Pitch Spiking Input",
            "",
            "This final section tests the hypothesis that RF detectors may be better suited to repeated signals. Instead of a single spike per channel, each pitch channel emits a repeated spike train. The echo is the same repeated train shifted by the target delay.",
            "",
            "This is deliberately a clean test: no added false spikes and no timing jitter. The question is whether repeated periodic input alone makes RF more competitive for distance decoding.",
            "",
            "![Sustained pitch raster](../outputs/accuracy_optimisation/figures/sustained_pitch_raster.png)",
            "",
            "The response plot below shows the detector-bank score for one example target. Repeated sustained pitches can create secondary peaks because pitch periods introduce delay aliases.",
            "",
            "![Sustained pitch response](../outputs/accuracy_optimisation/figures/sustained_pitch_response.png)",
            "",
            "The scatter plot below shows how those aliases affect the final predicted distance over the full test set.",
            "",
            "![Sustained pitch accuracy scatter](../outputs/accuracy_optimisation/figures/sustained_pitch_accuracy_scatter.png)",
            "",
            "| Sustained-pitch parameter | Value |",
            "|---|---:|",
            f"| pitch channels | `{SUSTAINED_CHANNELS}` |",
            f"| repeats per channel | `{SUSTAINED_REPEATS}` |",
            f"| period range | `{int(sustained_dataset.periods_samples.min())} -> {int(sustained_dataset.periods_samples.max())} samples` |",
            "",
            "| Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sustained_results:
        lines.append(
            f"| {result.name} | {result.mae_m * 100.0:.2f} | {result.rmse_m * 100.0:.2f} | "
            f"{result.p95_abs_error_m * 100.0:.2f} | {result.max_abs_error_m * 100.0:.2f} | "
            f"{result.runtime_ms:.3f} | {result.flops:,.0f} | {result.sops:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Sustained-Pitch Interpretation",
            "",
            "- Repeated inputs do not automatically make RF better for distance estimation.",
            "- A sustained pitch creates periodic ambiguity: delays separated by approximately one pitch period can also produce coincidences.",
            "- Multiple pitch channels help reduce this ambiguity, but the task is still fundamentally a delay matching problem.",
            "- RF may be more useful when the goal is periodicity or frequency selectivity, while binary/LIF remain more direct for pure echo-delay estimation.",
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
            "spiking_condition_mae",
            "spiking_accuracy_scatter_noise_jitter",
            "spiking_error_histogram_noise_jitter",
            "spiking_cost_comparison",
            "sweep_spike_raster",
            "sweep_condition_mae",
            "sweep_accuracy_scatter_noise_jitter",
            "sweep_error_histogram_noise_jitter",
            "sweep_cost_comparison",
            "sustained_pitch_raster",
            "sustained_pitch_response",
            "sustained_pitch_accuracy_scatter",
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
    spike_datasets = _all_spike_datasets()
    spike_results_by_condition: dict[str, list[DetectorResult]] = {}
    for dataset in spike_datasets:
        spike_results_by_condition[dataset.condition] = [
            _score_spike_detector("LIF detector", dataset, _predict_spike_lif),
            _score_spike_detector("RF detector", dataset, _predict_spike_rf),
            _score_spike_detector("Binary detector", dataset, _predict_spike_binary),
        ]
    sweep_datasets = _all_sweep_spike_datasets()
    sweep_results_by_condition: dict[str, list[DetectorResult]] = {}
    for dataset in sweep_datasets:
        sweep_results_by_condition[dataset.condition] = [
            _score_sweep_detector("LIF detector", dataset, _predict_sweep_lif),
            _score_sweep_detector("RF detector", dataset, _predict_sweep_rf),
            _score_sweep_detector("Binary detector", dataset, _predict_sweep_binary),
        ]
    sustained_dataset = _make_sustained_pitch_dataset()
    sustained_results = [
        _score_sustained_detector("LIF detector", sustained_dataset, _predict_sustained_lif),
        _score_sustained_detector("RF detector", sustained_dataset, _predict_sustained_rf),
        _score_sustained_detector("Binary detector", sustained_dataset, _predict_sustained_binary),
    ]

    simple_dataset = datasets[0]
    hard_dataset = datasets[-1]
    hard_results = results_by_condition[hard_dataset.condition]
    hard_spike_dataset = spike_datasets[-1]
    hard_spike_results = spike_results_by_condition[hard_spike_dataset.condition]
    hard_sweep_dataset = sweep_datasets[-1]
    hard_sweep_results = sweep_results_by_condition[hard_sweep_dataset.condition]
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
        "spiking_condition_mae": _plot_condition_mae(
            spike_results_by_condition,
            OPT_FIGURE_DIR / "spiking_condition_mae.png",
        ),
        "spiking_accuracy_scatter_noise_jitter": _plot_accuracy(
            hard_spike_results,
            hard_spike_dataset,
            OPT_FIGURE_DIR / "spiking_accuracy_scatter_noise_jitter.png",
        ),
        "spiking_error_histogram_noise_jitter": _plot_error_histogram(
            hard_spike_results,
            hard_spike_dataset,
            OPT_FIGURE_DIR / "spiking_error_histogram_noise_jitter.png",
        ),
        "spiking_cost_comparison": _plot_costs(
            spike_results_by_condition,
            OPT_FIGURE_DIR / "spiking_cost_comparison.png",
        ),
        "sweep_spike_raster": _plot_sweep_raster(OPT_FIGURE_DIR / "sweep_spike_raster.png"),
        "sweep_condition_mae": _plot_condition_mae(
            sweep_results_by_condition,
            OPT_FIGURE_DIR / "sweep_condition_mae.png",
        ),
        "sweep_accuracy_scatter_noise_jitter": _plot_accuracy(
            hard_sweep_results,
            hard_sweep_dataset,
            OPT_FIGURE_DIR / "sweep_accuracy_scatter_noise_jitter.png",
        ),
        "sweep_error_histogram_noise_jitter": _plot_error_histogram(
            hard_sweep_results,
            hard_sweep_dataset,
            OPT_FIGURE_DIR / "sweep_error_histogram_noise_jitter.png",
        ),
        "sweep_cost_comparison": _plot_costs(
            sweep_results_by_condition,
            OPT_FIGURE_DIR / "sweep_cost_comparison.png",
        ),
        "sustained_pitch_raster": _plot_sustained_pitch_raster(
            sustained_dataset,
            OPT_FIGURE_DIR / "sustained_pitch_raster.png",
        ),
        "sustained_pitch_response": _plot_sustained_response(
            sustained_dataset,
            OPT_FIGURE_DIR / "sustained_pitch_response.png",
        ),
        "sustained_pitch_accuracy_scatter": _plot_accuracy_from_distances(
            sustained_results,
            sustained_dataset.true_distance_m,
            OPT_FIGURE_DIR / "sustained_pitch_accuracy_scatter.png",
            title="Sustained-pitch clean accuracy scatter",
        ),
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
            "white_noise_snr_db": WHITE_NOISE_SNR_DB,
            "echo_pulse_sigma_samples": ECHO_PULSE_SIGMA_SAMPLES,
            "spike_noise_events": SPIKE_NOISE_EVENTS,
            "spike_noise_min_amplitude": SPIKE_NOISE_MIN_AMPLITUDE,
            "spike_noise_max_amplitude": SPIKE_NOISE_MAX_AMPLITUDE,
            "sweep_channels": SWEEP_CHANNELS,
            "sweep_duration_s": SWEEP_DURATION_S,
            "sustained_channels": SUSTAINED_CHANNELS,
            "sustained_repeats": SUSTAINED_REPEATS,
            "rng_seed": RNG_SEED,
        },
        "waveform_results_by_condition": {
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
        "spiking_results_by_condition": {
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
            for condition, results in spike_results_by_condition.items()
        },
        "sweep_spiking_results_by_condition": {
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
            for condition, results in sweep_results_by_condition.items()
        },
        "sustained_pitch_results": [
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
            for result in sustained_results
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_simple_report(simple_dataset, artifacts, elapsed_s)
    _write_optimisation_report(
        datasets,
        results_by_condition,
        spike_datasets,
        spike_results_by_condition,
        sweep_datasets,
        sweep_results_by_condition,
        sustained_dataset,
        sustained_results,
        artifacts,
        elapsed_s,
    )
    return payload


if __name__ == "__main__":
    main()
    print(SIMPLE_REPORT_PATH)
    print(OPT_REPORT_PATH)
