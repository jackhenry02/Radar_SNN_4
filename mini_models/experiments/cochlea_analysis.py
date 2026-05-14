from __future__ import annotations

"""Compare candidate cochlea front ends for the mini-model project.

This experiment is intentionally smaller than `outputs/cochlea_explained.md`.
It compares four ways to turn one acoustic waveform into channel-wise spike
rasters:

1. the original FFT/IFFT + envelope + LIF cochlea;
2. a time-domain Conv1D filterbank + LIF cochlea;
3. a time-domain Conv1D filterbank + level-crossing encoder;
4. a direct resonate-and-fire neuron bank.

The later sections add improvement candidates without deleting the first four
baseline results.

The code favours clarity over maximum performance because this is a design
experiment. Later mini models can replace individual pieces with optimized
implementations once the mechanisms are accepted.
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.common.signals import moving_notch_signal_config
from models.acoustics import cochlea_filterbank_stages, lif_encode_stages, simulate_echo_batch
from utils.common import GlobalConfig


ROOT = PROJECT_ROOT
OUTPUT_DIR = ROOT / "mini_models" / "outputs" / "cochlea_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = ROOT / "mini_models" / "reports" / "cochlea_analysis.md"

NUM_CHANNELS = 48
CONV_KERNEL_SIZE = 129
BENCHMARK_REPEATS = 8
RF_Q_FACTOR = 7.0
RF_INPUT_GAIN = 0.25
RF_THRESHOLD = 1.0
IIR_Q_FACTOR = 5.0
IIR_INPUT_GAIN = 1.0
RF_DAMPED_Q_FACTOR = 3.0
RF_DAMPED_INPUT_GAIN = 0.35
RF_DAMPED_THRESHOLD = 0.5
RF_DAMPING_FACTOR = 0.88
ACTIVE_WINDOW_THRESHOLD_FRACTION = 0.02
ACTIVE_WINDOW_PADDING_S = 0.001


@dataclass
class CochleaResult:
    """Output from one cochlea front-end model.

    Attributes:
        name: Stable model identifier.
        title: Human-readable model name.
        cochleagram: Continuous channel activity, shape `[channels, time]`.
        spikes: Binary spike/event raster, shape `[channels, time]`.
        time_rate_hz: Time base of `cochleagram` and `spikes`.
        center_frequencies_hz: Frequency represented by each channel.
        elapsed_s: Median benchmark runtime for one waveform.
        flops_estimate: Rough dense floating-point operation estimate.
        sops_estimate: Rough spike/event operation estimate.
        notes: Short implementation note.
    """

    name: str
    title: str
    cochleagram: torch.Tensor
    spikes: torch.Tensor
    time_rate_hz: float
    center_frequencies_hz: torch.Tensor
    elapsed_s: float
    flops_estimate: float
    sops_estimate: float
    notes: str


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a CPU numpy array."""
    return values.detach().cpu().numpy()


def _make_config() -> GlobalConfig:
    """Return the shared matched-human cochlea configuration.

    Returns:
        Global configuration used for all four cochlea models.
    """
    base = moving_notch_signal_config(GlobalConfig())
    return GlobalConfig(
        **{
            **base.__dict__,
            "num_cochlea_channels": NUM_CHANNELS,
            "normalize_spike_envelope": False,
        }
    )


def _simulate_input_waveform(config: GlobalConfig) -> torch.Tensor:
    """Simulate one clean left-ear echo waveform.

    Args:
        config: Acoustic configuration.

    Returns:
        One-dimensional left-ear waveform.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([3.0]),
        azimuth_deg=torch.tensor([35.0]),
        elevation_deg=torch.tensor([20.0]),
        binaural=True,
        add_noise=False,
        include_elevation_cues=True,
        transmit_gain=config.transmit_gain,
    )
    return scene.receive[0, 0].detach()


def _log_spaced_centers(config: GlobalConfig, num_channels: int = NUM_CHANNELS) -> torch.Tensor:
    """Create log-spaced cochlear centre frequencies.

    Args:
        config: Acoustic configuration.
        num_channels: Number of frequency channels.

    Returns:
        Tensor of centre frequencies in Hz.
    """
    return torch.logspace(
        math.log10(config.cochlea_low_hz),
        math.log10(config.cochlea_high_hz),
        steps=num_channels,
    )


def _make_gabor_filterbank(
    config: GlobalConfig,
    *,
    num_channels: int = NUM_CHANNELS,
    kernel_size: int = CONV_KERNEL_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a time-domain Gabor-like cochlear filterbank.

    This is not claimed to be a full biological gammatone model. It is a
    compact time-domain band-pass bank that is useful for comparing a Conv1D
    front end against the old FFT/IFFT implementation.

    Args:
        config: Acoustic configuration.
        num_channels: Number of filters.
        kernel_size: Number of samples in each FIR kernel.

    Returns:
        Pair `(kernels, centers)` where kernels has shape
        `[channels, 1, kernel_size]`.
    """
    centers = _log_spaced_centers(config, num_channels)
    time_axis = (torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0) / config.sample_rate_hz
    # A shorter Gaussian window at high frequency keeps each channel reasonably
    # localized in time while still providing frequency selectivity.
    sigma = (1.8 / centers).unsqueeze(1)
    window = torch.exp(-0.5 * (time_axis.unsqueeze(0) / sigma).square())
    carrier = torch.cos(2.0 * math.pi * centers.unsqueeze(1) * time_axis.unsqueeze(0))
    kernels = window * carrier
    kernels = kernels - kernels.mean(dim=-1, keepdim=True)
    kernels = kernels / kernels.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return kernels.unsqueeze(1), centers


def _run_iir_resonator_filterbank(
    signal: torch.Tensor,
    config: GlobalConfig,
    *,
    q_factor: float = IIR_Q_FACTOR,
    input_gain: float = IIR_INPUT_GAIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a gammatone-like IIR resonator bank.

    The update is a lightweight second-order resonator. It is not a full
    four-pole gammatone cascade, but it has the key computational property we
    want to test: each output sample is produced recursively from the current
    input and previous channel states rather than by a long FIR convolution.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.
        q_factor: Filter quality factor. Lower values produce wider filters.
        input_gain: Scaling applied to the waveform before recursion.

    Returns:
        Tuple `(filtered, centers)` where filtered has shape
        `[channels, time]`.
    """
    centers = _log_spaced_centers(config)
    theta = 2.0 * math.pi * centers / config.sample_rate_hz
    bandwidth_hz = centers / max(q_factor, 1e-6)
    pole_radius = torch.exp(-math.pi * bandwidth_hz / config.sample_rate_hz)
    feedback_one = 2.0 * pole_radius * torch.cos(theta)
    feedback_two = -(pole_radius.square())
    feedforward = (1.0 - pole_radius).clamp_min(1e-6) * input_gain
    previous_one = torch.zeros_like(centers)
    previous_two = torch.zeros_like(centers)
    outputs = []
    for sample in signal:
        output = feedforward * sample + feedback_one * previous_one + feedback_two * previous_two
        previous_two = previous_one
        previous_one = output
        outputs.append(output)
    return torch.stack(outputs, dim=-1), centers


def _iir_lfilter_coefficients(
    config: GlobalConfig,
    *,
    q_factor: float = IIR_Q_FACTOR,
    input_gain: float = IIR_INPUT_GAIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build second-order resonator coefficients for `torchaudio.lfilter`.

    The manual IIR recurrence is:

    `y[t] = b0*x[t] + f1*y[t-1] + f2*y[t-2]`.

    `lfilter` uses the standard form:

    `y[t] = b0*x[t] + b1*x[t-1] + b2*x[t-2] - a1*y[t-1] - a2*y[t-2]`.

    Therefore `a1 = -f1` and `a2 = -f2`.

    Args:
        config: Acoustic configuration.
        q_factor: Filter quality factor.
        input_gain: Input gain used for the feedforward term.

    Returns:
        Tuple `(a_coeffs, b_coeffs, centers)` for all channels.
    """
    centers = _log_spaced_centers(config)
    theta = 2.0 * math.pi * centers / config.sample_rate_hz
    bandwidth_hz = centers / max(q_factor, 1e-6)
    pole_radius = torch.exp(-math.pi * bandwidth_hz / config.sample_rate_hz)
    feedback_one = 2.0 * pole_radius * torch.cos(theta)
    feedback_two = -(pole_radius.square())
    feedforward = (1.0 - pole_radius).clamp_min(1e-6) * input_gain
    a_coeffs = torch.stack(
        [
            torch.ones_like(feedback_one),
            -feedback_one,
            -feedback_two,
        ],
        dim=-1,
    )
    b_coeffs = torch.stack(
        [
            feedforward,
            torch.zeros_like(feedforward),
            torch.zeros_like(feedforward),
        ],
        dim=-1,
    )
    return a_coeffs, b_coeffs, centers


def _run_iir_lfilter_filterbank(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the IIR resonator bank using `torchaudio.functional.lfilter`.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(filtered, centers)` where filtered has shape
        `[channels, time]`.
    """
    a_coeffs, b_coeffs, centers = _iir_lfilter_coefficients(config)
    repeated = signal.to(torch.float32).unsqueeze(0).repeat(config.num_cochlea_channels, 1)
    filtered = AF.lfilter(repeated, a_coeffs, b_coeffs, clamp=False, batching=True)
    return filtered.to(signal.dtype), centers


def _level_crossing_encode(filtered: torch.Tensor, *, delta: float) -> torch.Tensor:
    """Encode a filtered signal bank using level-crossing events.

    Args:
        filtered: Signed channel activity, shape `[channels, time]`.
        delta: Required signal change before an event is emitted.

    Returns:
        Binary event raster with shape `[channels, time]`.
    """
    reference = filtered[:, :1].clone()
    events = torch.zeros_like(filtered)
    for time_index in range(filtered.shape[-1]):
        value = filtered[:, time_index : time_index + 1]
        up = value - reference >= delta
        down = reference - value >= delta
        event = (up | down).to(filtered.dtype)
        # Move the internal level by all crossed delta steps. This keeps the
        # encoder stable when the waveform jumps by more than one threshold.
        upward_steps = torch.floor((value - reference).clamp_min(0.0) / delta)
        downward_steps = torch.floor((reference - value).clamp_min(0.0) / delta)
        reference = reference + upward_steps * delta - downward_steps * delta
        events[:, time_index] = event.squeeze(1)
    return events


def _lif_encode_full_rate(envelope: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an unnormalized LIF encoder at the raw sample rate.

    Args:
        envelope: Non-negative channel activity, shape `[channels, time]`.
        config: Acoustic configuration.

    Returns:
        Pair `(spikes, membrane)`, both with shape `[channels, time]`.
    """
    # The old LIF beta acted at the downsampled envelope rate. Convert it to an
    # approximately equivalent per-sample leak so removing the downsample block
    # does not make the membrane four times leakier.
    beta_per_sample = float(config.spike_beta) ** (1.0 / max(int(config.envelope_downsample), 1))
    membrane = torch.zeros(envelope.shape[0], dtype=envelope.dtype)
    membranes = []
    spikes = []
    for time_index in range(envelope.shape[-1]):
        membrane = beta_per_sample * membrane + envelope[:, time_index]
        spike = (membrane >= config.spike_threshold).to(envelope.dtype)
        membrane = (membrane - spike * config.spike_threshold).clamp_min(0.0)
        membranes.append(membrane)
        spikes.append(spike)
    return torch.stack(spikes, dim=-1), torch.stack(membranes, dim=-1)


def _lif_encode_full_rate_optimized(envelope: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full-rate LIF encoder with preallocated tensors.

    This preserves the same sequential LIF dynamics as `_lif_encode_full_rate`,
    but avoids list appends and uses in-place state updates where practical.

    Args:
        envelope: Non-negative channel activity, shape `[channels, time]`.
        config: Acoustic configuration.

    Returns:
        Pair `(spikes, membrane)`, both with shape `[channels, time]`.
    """
    beta_per_sample = float(config.spike_beta) ** (1.0 / max(int(config.envelope_downsample), 1))
    membrane = torch.zeros(envelope.shape[0], dtype=envelope.dtype, device=envelope.device)
    membranes = torch.zeros_like(envelope)
    spikes = torch.zeros_like(envelope)
    for time_index in range(envelope.shape[-1]):
        membrane.mul_(beta_per_sample).add_(envelope[:, time_index])
        spike_bool = membrane >= config.spike_threshold
        spikes[:, time_index] = spike_bool.to(envelope.dtype)
        membrane.sub_(spike_bool.to(envelope.dtype) * config.spike_threshold).clamp_(min=0.0)
        membranes[:, time_index] = membrane
    return spikes, membranes


@torch.jit.script
def _lif_encode_full_rate_jit(
    envelope: torch.Tensor,
    beta_old: float,
    threshold: float,
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TorchScript full-rate LIF encoder.

    Args:
        envelope: Non-negative channel activity, shape `[channels, time]`.
        beta_old: LIF beta used by the old downsampled encoder.
        threshold: Spike threshold.
        downsample: Old envelope downsample factor used to convert beta to a
            per-sample value.

    Returns:
        Pair `(spikes, membrane)`, both with shape `[channels, time]`.
    """
    effective_downsample = downsample
    if effective_downsample < 1:
        effective_downsample = 1
    beta_per_sample = beta_old ** (1.0 / float(effective_downsample))
    membrane = torch.zeros(envelope.size(0), dtype=envelope.dtype, device=envelope.device)
    membranes = torch.zeros_like(envelope)
    spikes = torch.zeros_like(envelope)
    for time_index in range(envelope.size(1)):
        membrane.mul_(beta_per_sample).add_(envelope[:, time_index])
        spike = (membrane >= threshold).to(envelope.dtype)
        spikes[:, time_index] = spike
        membrane.sub_(spike * threshold).clamp_(min=0.0)
        membranes[:, time_index] = membrane
    return spikes, membranes


def _run_original_fft_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the old FFT/IFFT + envelope + LIF cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    stages = cochlea_filterbank_stages(
        signal=signal.unsqueeze(0),
        sample_rate_hz=config.sample_rate_hz,
        num_channels=config.num_cochlea_channels,
        low_hz=config.cochlea_low_hz,
        high_hz=config.cochlea_high_hz,
        spacing_mode=config.cochlea_spacing_mode,
        filter_bandwidth_sigma=config.filter_bandwidth_sigma,
        envelope_lowpass_hz=config.envelope_lowpass_hz,
        downsample=config.envelope_downsample,
    )
    lif = lif_encode_stages(
        stages["cochleagram"],
        threshold=config.spike_threshold,
        beta=config.spike_beta,
        normalize_envelope=False,
    )
    return stages["cochleagram"][0], lif["spikes"][0], stages["center_frequencies_hz"]


def _run_conv_lif_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the proposed Conv1D filterbank + LIF cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    kernels, centers = _make_gabor_filterbank(config)
    filtered = F.conv1d(signal.view(1, 1, -1), kernels, padding=CONV_KERNEL_SIZE // 2)[0]
    cochleagram = F.relu(filtered)
    spikes, _ = _lif_encode_full_rate(cochleagram, config)
    return cochleagram, spikes, centers


def _run_level_crossing_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a filterbank followed by level-crossing delta modulation.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`. The cochleagram is the signed
        filtered signal; the spike raster combines up and down events.
    """
    kernels, centers = _make_gabor_filterbank(config)
    filtered = F.conv1d(signal.view(1, 1, -1), kernels, padding=CONV_KERNEL_SIZE // 2)[0]
    return filtered, _level_crossing_encode(filtered, delta=float(config.spike_threshold)), centers


def _run_iir_lif_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the proposed IIR resonator filterbank + LIF cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    filtered, centers = _run_iir_resonator_filterbank(signal, config)
    cochleagram = F.relu(filtered)
    spikes, _ = _lif_encode_full_rate(cochleagram, config)
    return cochleagram, spikes, centers


def _run_iir_lfilter_lif_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the optimized lfilter IIR + preallocated LIF cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    filtered, centers = _run_iir_lfilter_filterbank(signal, config)
    cochleagram = F.relu(filtered)
    spikes, _ = _lif_encode_full_rate_optimized(cochleagram, config)
    return cochleagram, spikes, centers


def _run_iir_lfilter_jit_lif_cochlea(
    signal: torch.Tensor,
    config: GlobalConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run lfilter IIR with a TorchScript LIF encoder.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    filtered, centers = _run_iir_lfilter_filterbank(signal, config)
    cochleagram = F.relu(filtered)
    spikes, _ = _lif_encode_full_rate_jit(
        cochleagram,
        float(config.spike_beta),
        float(config.spike_threshold),
        int(config.envelope_downsample),
    )
    return cochleagram, spikes, centers


def _active_window_bounds(signal: torch.Tensor, config: GlobalConfig) -> tuple[int, int]:
    """Find a padded active waveform window for gated processing.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(start, stop)` with stop exclusive.
    """
    threshold = ACTIVE_WINDOW_THRESHOLD_FRACTION * signal.abs().amax().clamp_min(1e-12)
    active_indices = torch.nonzero(signal.abs() >= threshold, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return 0, signal.numel()
    padding = int(round(ACTIVE_WINDOW_PADDING_S * config.sample_rate_hz))
    start = max(0, int(active_indices[0].item()) - padding)
    stop = min(signal.numel(), int(active_indices[-1].item()) + padding + 1)
    return start, stop


def _run_iir_lfilter_lif_gated_cochlea(
    signal: torch.Tensor,
    config: GlobalConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run lfilter IIR + optimized LIF only over the active echo window.

    The output is scattered back into full-length tensors so downstream shape
    compatibility is preserved. This is window-gated dense processing, not yet
    true sparse/event-driven filtering.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    start, stop = _active_window_bounds(signal, config)
    cropped = signal[start:stop]
    filtered_crop, centers = _run_iir_lfilter_filterbank(cropped, config)
    cochleagram_crop = F.relu(filtered_crop)
    spikes_crop, _ = _lif_encode_full_rate_optimized(cochleagram_crop, config)
    full_cochleagram = torch.zeros(config.num_cochlea_channels, signal.numel(), dtype=signal.dtype)
    full_spikes = torch.zeros_like(full_cochleagram)
    full_cochleagram[:, start:stop] = cochleagram_crop
    full_spikes[:, start:stop] = spikes_crop
    _run_iir_lfilter_lif_gated_cochlea.last_window = {
        "start_sample": start,
        "stop_sample": stop,
        "samples_processed": stop - start,
        "total_samples": signal.numel(),
        "processed_fraction": (stop - start) / max(signal.numel(), 1),
        "padding_s": ACTIVE_WINDOW_PADDING_S,
        "threshold_fraction": ACTIVE_WINDOW_THRESHOLD_FRACTION,
    }
    return full_cochleagram, full_spikes, centers


def _run_iir_lfilter_jit_lif_gated_cochlea(
    signal: torch.Tensor,
    config: GlobalConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run lfilter IIR + TorchScript LIF only over the active echo window.

    This is the current best optimized faithful LIF candidate: torchaudio
    handles the IIR recursion, TorchScript handles the sequential reset-based
    LIF loop, and active-window gating avoids processing long silent regions.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)` scattered back to full signal
        length.
    """
    start, stop = _active_window_bounds(signal, config)
    cropped = signal[start:stop]
    filtered_crop, centers = _run_iir_lfilter_filterbank(cropped, config)
    cochleagram_crop = F.relu(filtered_crop)
    spikes_crop, _ = _lif_encode_full_rate_jit(
        cochleagram_crop,
        float(config.spike_beta),
        float(config.spike_threshold),
        int(config.envelope_downsample),
    )
    full_cochleagram = torch.zeros(config.num_cochlea_channels, signal.numel(), dtype=signal.dtype)
    full_spikes = torch.zeros_like(full_cochleagram)
    full_cochleagram[:, start:stop] = cochleagram_crop
    full_spikes[:, start:stop] = spikes_crop
    _run_iir_lfilter_jit_lif_gated_cochlea.last_window = {
        "start_sample": start,
        "stop_sample": stop,
        "samples_processed": stop - start,
        "total_samples": signal.numel(),
        "processed_fraction": (stop - start) / max(signal.numel(), 1),
        "padding_s": ACTIVE_WINDOW_PADDING_S,
        "threshold_fraction": ACTIVE_WINDOW_THRESHOLD_FRACTION,
    }
    return full_cochleagram, full_spikes, centers


def _run_iir_level_crossing_cochlea(
    signal: torch.Tensor,
    config: GlobalConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the proposed IIR resonator filterbank + level-crossing cochlea.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    filtered, centers = _run_iir_resonator_filterbank(signal, config)
    return filtered, _level_crossing_encode(filtered, delta=float(config.spike_threshold)), centers


def _run_rf_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a direct resonate-and-fire neuron bank.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`. The cochleagram is oscillator
        energy and the raster is thresholded RF output spikes.
    """
    centers = _log_spaced_centers(config)
    theta = 2.0 * math.pi * centers / config.sample_rate_hz
    decay = torch.exp(-theta / (2.0 * RF_Q_FACTOR))
    state = torch.zeros_like(centers)
    velocity = torch.zeros_like(centers)
    states = []
    velocities = []
    spikes = []
    for sample in signal:
        velocity = decay * velocity + RF_INPUT_GAIN * sample - theta * state
        state = state + theta * velocity
        spike = (state >= RF_THRESHOLD).to(signal.dtype)
        state = (state - spike * RF_THRESHOLD).clamp_min(0.0)
        states.append(state)
        velocities.append(velocity)
        spikes.append(spike)
    state_trace = torch.stack(states, dim=-1)
    velocity_trace = torch.stack(velocities, dim=-1)
    energy = torch.sqrt(state_trace.square() + velocity_trace.square())
    return energy, torch.stack(spikes, dim=-1), centers


def _run_damped_rf_cochlea(signal: torch.Tensor, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a more responsive damped RF neuron bank.

    This variant explicitly separates damping from the Q-derived decay, uses a
    wider resonance band, and lowers the spike threshold. It is intended as a
    better first-pass RF cochlea candidate, not as a final tuned model.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Tuple `(cochleagram, spikes, centers)`.
    """
    centers = _log_spaced_centers(config)
    theta = 2.0 * math.pi * centers / config.sample_rate_hz
    decay = RF_DAMPING_FACTOR * torch.exp(-theta / (2.0 * RF_DAMPED_Q_FACTOR))
    state = torch.zeros_like(centers)
    velocity = torch.zeros_like(centers)
    states = []
    velocities = []
    spikes = []
    for sample in signal:
        velocity = decay * velocity + RF_DAMPED_INPUT_GAIN * sample - theta * state
        state = state + theta * velocity
        spike = (state >= RF_DAMPED_THRESHOLD).to(signal.dtype)
        state = (state - spike * RF_DAMPED_THRESHOLD).clamp_min(0.0)
        states.append(state)
        velocities.append(velocity)
        spikes.append(spike)
    state_trace = torch.stack(states, dim=-1)
    velocity_trace = torch.stack(velocities, dim=-1)
    energy = torch.sqrt(state_trace.square() + velocity_trace.square())
    return energy, torch.stack(spikes, dim=-1), centers


def _median_runtime_s(function: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    """Benchmark a cochlea function using a median runtime.

    Args:
        function: Zero-argument function that runs one cochlea model.

    Returns:
        Median elapsed time in seconds.
    """
    with torch.no_grad():
        function()
        times = []
        for _ in range(BENCHMARK_REPEATS):
            start = time.perf_counter()
            function()
            times.append(time.perf_counter() - start)
    return float(np.median(times))


def _estimate_original_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs for the old FFT/IFFT cochlea.

    Args:
        config: Acoustic configuration.
        signal_samples: Number of waveform samples.

    Returns:
        Rough FLOP count.
    """
    channels = config.num_cochlea_channels
    frequency_bins = signal_samples // 2 + 1
    envelope_steps = math.ceil(signal_samples / config.envelope_downsample)
    lowpass_kernel = max(3, int(round(config.sample_rate_hz / config.envelope_lowpass_hz)))
    if lowpass_kernel % 2 == 0:
        lowpass_kernel += 1
    fft_cost = 5.0 * signal_samples * math.log2(signal_samples)
    ifft_cost = channels * fft_cost
    complex_filter_mult = channels * frequency_bins * 6.0
    rectification = channels * signal_samples
    smoothing = channels * signal_samples * lowpass_kernel * 2.0
    downsample = channels * envelope_steps * config.envelope_downsample
    lif = channels * envelope_steps * 5.0
    return fft_cost + ifft_cost + complex_filter_mult + rectification + smoothing + downsample + lif


def _estimate_conv_lif_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs for the Conv1D + LIF cochlea."""
    channels = config.num_cochlea_channels
    convolution = channels * signal_samples * CONV_KERNEL_SIZE * 2.0
    rectification = channels * signal_samples
    lif = channels * signal_samples * 5.0
    return convolution + rectification + lif


def _estimate_level_crossing_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs/comparisons for Conv1D + level crossing."""
    channels = config.num_cochlea_channels
    convolution = channels * signal_samples * CONV_KERNEL_SIZE * 2.0
    comparisons_and_updates = channels * signal_samples * 6.0
    return convolution + comparisons_and_updates


def _estimate_iir_lif_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs/comparisons for IIR filterbank + LIF."""
    channels = config.num_cochlea_channels
    iir = channels * signal_samples * 6.0
    rectification = channels * signal_samples
    lif = channels * signal_samples * 5.0
    return iir + rectification + lif


def _estimate_iir_level_crossing_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs/comparisons for IIR filterbank + level crossing."""
    channels = config.num_cochlea_channels
    iir = channels * signal_samples * 6.0
    comparisons_and_updates = channels * signal_samples * 6.0
    return iir + comparisons_and_updates


def _estimate_rf_flops(config: GlobalConfig, signal_samples: int) -> float:
    """Estimate dense FLOPs/comparisons for the RF neuron bank."""
    channels = config.num_cochlea_channels
    return channels * signal_samples * 10.0


def _plot_cochleagram(result: CochleaResult, path: Path) -> str:
    """Plot continuous channel activity for a cochlea model.

    Args:
        result: Cochlea model output.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    values = _to_numpy(result.cochleagram)
    display = np.log1p(np.abs(values) if "level_crossing" in result.name else np.maximum(values, 0.0))
    time_ms = np.arange(values.shape[-1], dtype=np.float64) / result.time_rate_hz * 1_000.0
    centers_khz = _to_numpy(result.center_frequencies_hz) / 1_000.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], centers_khz[0], centers_khz[-1]],
        cmap="magma",
    )
    ax.set_title(f"{result.title}: cochleagram / channel activity")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel centre frequency (kHz)")
    ax.set_xlim(0.0, min(time_ms[-1], 24.0))
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log activity")
    return save_figure(fig, path)


def _plot_spike_raster(result: CochleaResult, path: Path) -> str:
    """Plot a channel-by-time spike raster for a cochlea model.

    Args:
        result: Cochlea model output.
        path: Output figure path.

    Returns:
        String path to the saved figure.
    """
    spikes = _to_numpy(result.spikes > 0.0)
    time_ms = np.arange(spikes.shape[-1], dtype=np.float64) / result.time_rate_hz * 1_000.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for channel in range(spikes.shape[0]):
        spike_times = time_ms[np.flatnonzero(spikes[channel])]
        if spike_times.size:
            ax.vlines(spike_times, channel + 0.15, channel + 0.85, color="#111827", linewidth=0.55)
    ax.set_title(f"{result.title}: spike raster")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("channel")
    ax.set_ylim(0, spikes.shape[0])
    ax.set_xlim(0.0, min(time_ms[-1], 24.0))
    return save_figure(fig, path)


def _run_all_models(signal: torch.Tensor, config: GlobalConfig) -> list[CochleaResult]:
    """Run and benchmark all cochlea models.

    Args:
        signal: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        List of model results.
    """
    signal_samples = signal.numel()
    active_start, active_stop = _active_window_bounds(signal, config)
    active_samples = active_stop - active_start
    specs = [
        (
            "original_fft_lif",
            "Original FFT/IFFT + envelope + LIF",
            lambda: _run_original_fft_cochlea(signal, config),
            _estimate_original_flops(config, signal_samples),
            "Old fixed cochlea: FFT, Gaussian frequency filters, IFFT per channel, rectification, low-pass envelope, downsample, LIF.",
        ),
        (
            "conv1d_lif",
            "Time-domain Conv1D filterbank + LIF",
            lambda: _run_conv_lif_cochlea(signal, config),
            _estimate_conv_lif_flops(config, signal_samples),
            "New dense time-domain cochlea: FIR filterbank directly in time, rectification, full-rate LIF; no explicit envelope low-pass/downsample.",
        ),
        (
            "level_crossing",
            "Time-domain filterbank + level crossing",
            lambda: _run_level_crossing_cochlea(signal, config),
            _estimate_level_crossing_flops(config, signal_samples),
            "New event encoder: FIR filterbank followed by delta-modulation events on each filtered channel.",
        ),
        (
            "rf_bank",
            "Direct resonate-and-fire bank",
            lambda: _run_rf_cochlea(signal, config),
            _estimate_rf_flops(config, signal_samples),
            "New reduced cochlea: raw waveform drives a bank of RF neurons tuned across frequency.",
        ),
        (
            "iir_lif",
            "IIR resonator filterbank + LIF",
            lambda: _run_iir_lif_cochlea(signal, config),
            _estimate_iir_lif_flops(config, signal_samples),
            "Improvement candidate: recursive gammatone-like resonator bank, rectification, full-rate LIF.",
        ),
        (
            "iir_lfilter_lif",
            "lfilter IIR + optimized LIF",
            lambda: _run_iir_lfilter_lif_cochlea(signal, config),
            _estimate_iir_lif_flops(config, signal_samples),
            "Optimization candidate: same IIR idea, but the recursive filter is delegated to torchaudio.lfilter and the LIF loop preallocates outputs.",
        ),
        (
            "iir_lfilter_jit_lif",
            "lfilter IIR + TorchScript LIF",
            lambda: _run_iir_lfilter_jit_lif_cochlea(signal, config),
            _estimate_iir_lif_flops(config, signal_samples),
            "Optimization candidate: same lfilter IIR front end, but the sequential LIF loop is compiled with TorchScript.",
        ),
        (
            "iir_lfilter_lif_gated",
            "lfilter IIR + optimized LIF + active-window gating",
            lambda: _run_iir_lfilter_lif_gated_cochlea(signal, config),
            _estimate_iir_lif_flops(config, active_samples),
            "Optimization candidate: same lfilter IIR + optimized LIF, but only over the detected echo window plus padding.",
        ),
        (
            "iir_lfilter_jit_lif_gated",
            "lfilter IIR + TorchScript LIF + active-window gating",
            lambda: _run_iir_lfilter_jit_lif_gated_cochlea(signal, config),
            _estimate_iir_lif_flops(config, active_samples),
            "Current combined optimization candidate: lfilter IIR, TorchScript LIF, and active-window gating.",
        ),
        (
            "iir_level_crossing",
            "IIR resonator filterbank + level crossing",
            lambda: _run_iir_level_crossing_cochlea(signal, config),
            _estimate_iir_level_crossing_flops(config, signal_samples),
            "Improvement candidate: recursive gammatone-like resonator bank followed by delta-modulation events.",
        ),
        (
            "damped_rf_bank",
            "Damped wide-band RF bank",
            lambda: _run_damped_rf_cochlea(signal, config),
            _estimate_rf_flops(config, signal_samples),
            "Improvement candidate: RF bank with explicit damping, lower Q for wider response, and lower threshold.",
        ),
    ]
    results = []
    for name, title, function, flops, notes in specs:
        elapsed_s = _median_runtime_s(function)
        cochleagram, spikes, centers = function()
        results.append(
            CochleaResult(
                name=name,
                title=title,
                cochleagram=cochleagram.detach(),
                spikes=spikes.detach(),
                time_rate_hz=config.envelope_rate_hz if name == "original_fft_lif" else float(config.sample_rate_hz),
                center_frequencies_hz=centers.detach(),
                elapsed_s=elapsed_s,
                flops_estimate=float(flops),
                sops_estimate=float((spikes > 0.0).sum().item()),
                notes=notes,
            )
        )
    return results


def _write_report(results: list[CochleaResult], artifacts: dict[str, dict[str, str]], config: GlobalConfig, elapsed_s: float) -> None:
    """Write the cochlea-analysis markdown report.

    Args:
        results: Model outputs and metrics.
        artifacts: Figure paths grouped by model name.
        config: Acoustic configuration.
        elapsed_s: Total experiment runtime.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_name = {result.name: result for result in results}
    iir_baseline = by_name["iir_lif"]
    lfilter_iir = by_name["iir_lfilter_lif"]
    jit_iir = by_name["iir_lfilter_jit_lif"]
    gated_iir = by_name["iir_lfilter_lif_gated"]
    gated_jit_iir = by_name["iir_lfilter_jit_lif_gated"]
    lfilter_saving = 100.0 * (iir_baseline.elapsed_s - lfilter_iir.elapsed_s) / max(iir_baseline.elapsed_s, 1e-12)
    jit_saving = 100.0 * (iir_baseline.elapsed_s - jit_iir.elapsed_s) / max(iir_baseline.elapsed_s, 1e-12)
    gated_saving = 100.0 * (iir_baseline.elapsed_s - gated_iir.elapsed_s) / max(iir_baseline.elapsed_s, 1e-12)
    gated_jit_saving = 100.0 * (iir_baseline.elapsed_s - gated_jit_iir.elapsed_s) / max(iir_baseline.elapsed_s, 1e-12)
    gated_window = getattr(_run_iir_lfilter_lif_gated_cochlea, "last_window", {})
    gated_jit_window = getattr(_run_iir_lfilter_jit_lif_gated_cochlea, "last_window", {})
    lines = [
        "# Mini Model 3: Cochlea Analysis",
        "",
        "This mini model compares the original four candidate cochlea front ends and then adds three improvement candidates. The aim is not yet to fully optimise them, but to check whether the mechanisms produce sensible channel activity and spikes, and to estimate their relative computational cost.",
        "",
        "## Shared Setup",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz} Hz` |",
        f"| chirp | `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` |",
        f"| chirp duration | `{config.chirp_duration_s * 1_000.0:.1f} ms` |",
        f"| signal duration | `{config.signal_duration_s * 1_000.0:.1f} ms` |",
        f"| cochlea band | `{config.cochlea_low_hz:.0f} -> {config.cochlea_high_hz:.0f} Hz` |",
        f"| channels | `{config.num_cochlea_channels}` |",
        f"| spike envelope normalization | `{config.normalize_spike_envelope}` |",
        f"| transmit gain | `{config.transmit_gain:.0f}x` |",
        "",
        "The input is one clean left-ear echo from the matched-human signal setup. Keeping one waveform fixed means the plots compare the front ends, not scene variability.",
        "",
        "## Cost Summary",
        "",
        "| Model | FLOPs estimate | SOPs / output events | Time | Time per channel | Spike density |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        spike_density = result.sops_estimate / float(result.spikes.numel())
        lines.append(
            f"| {result.title} | `{result.flops_estimate:,.0f}` | `{result.sops_estimate:,.0f}` | "
            f"`{result.elapsed_s * 1_000.0:.3f} ms` | `{result.elapsed_s / config.num_cochlea_channels * 1_000.0:.4f} ms` | `{spike_density:.4f}` |"
        )
    lines.extend(
        [
            "",
            "FLOPs are approximate dense-operation counts for one waveform. SOPs are counted here as emitted output spike/events, because downstream event-driven processing cost would scale with those events. This is a first-order proxy, not a hardware-validated energy model.",
            "",
            "## IIR Optimization Savings",
            "",
            "| Comparison | Time change vs current IIR + LIF | Samples processed |",
            "|---|---:|---:|",
            f"| lfilter IIR + optimized LIF | `{lfilter_saving:+.1f}%` | `{config.signal_samples}` / `{config.signal_samples}` |",
            f"| lfilter IIR + TorchScript LIF | `{jit_saving:+.1f}%` | `{config.signal_samples}` / `{config.signal_samples}` |",
            f"| lfilter IIR + optimized LIF + active-window gating | `{gated_saving:+.1f}%` | `{gated_window.get('samples_processed', 'n/a')}` / `{gated_window.get('total_samples', config.signal_samples)}` |",
            f"| lfilter IIR + TorchScript LIF + active-window gating | `{gated_jit_saving:+.1f}%` | `{gated_jit_window.get('samples_processed', 'n/a')}` / `{gated_jit_window.get('total_samples', config.signal_samples)}` |",
            "",
            "A positive value means faster than the current Python-loop IIR + LIF model. The gated model is window-gated dense processing: it skips the silent parts of the waveform, but still runs dense IIR/LIF updates inside the detected active window.",
            "",
            "## 1. Original FFT/IFFT + Envelope + LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[FFT]",
            "    B --> C[log Gaussian filterbank]",
            "    C --> D[IFFT per channel]",
            "    D --> E[half-wave rectification]",
            "    E --> F[Hann low-pass envelope]",
            "    F --> G[downsample]",
            "    G --> H[LIF spike encoder]",
            "    H --> I[spike raster]",
            "```",
            "",
            "```text",
            "X(f) = FFT{x(t)}",
            "x_c(t) = IFFT{X(f) * G_c(f)}",
            "e_c(t) = downsample(lowpass(max(x_c(t), 0)))",
            "v_c[t] = beta * v_c[t-1] + e_c[t]",
            "spike_c[t] = 1 if v_c[t] >= threshold else 0",
            "v_c[t] = max(v_c[t] - threshold * spike_c[t], 0)",
            "```",
            "",
            by_name["original_fft_lif"].notes,
            "",
            "![Original cochleagram](../outputs/cochlea_analysis/figures/original_fft_lif_cochleagram.png)",
            "",
            "![Original raster](../outputs/cochlea_analysis/figures/original_fft_lif_raster.png)",
            "",
            "## 2. Time-Domain Conv1D Filterbank + LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[Conv1D FIR filterbank]",
            "    B --> C[half-wave rectification]",
            "    C --> D[full-rate LIF spike encoder]",
            "    D --> E[spike raster]",
            "```",
            "",
            "```text",
            "x_c[t] = sum_k h_c[k] * x[t-k]",
            "e_c[t] = max(x_c[t], 0)",
            "v_c[t] = beta_sample * v_c[t-1] + e_c[t]",
            "beta_sample = beta_old^(1 / downsample)",
            "spike_c[t] = 1 if v_c[t] >= threshold else 0",
            "```",
            "",
            by_name["conv1d_lif"].notes,
            "",
            "![Conv1D cochleagram](../outputs/cochlea_analysis/figures/conv1d_lif_cochleagram.png)",
            "",
            "![Conv1D raster](../outputs/cochlea_analysis/figures/conv1d_lif_raster.png)",
            "",
            "## 3. Time-Domain Filterbank + Level Crossing",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[Conv1D FIR filterbank]",
            "    B --> C[level-crossing delta modulator]",
            "    C --> D[up/down event raster]",
            "```",
            "",
            "```text",
            "x_c[t] = sum_k h_c[k] * x[t-k]",
            "if x_c[t] - ref_c[t] >= delta: emit up event, ref_c += n * delta",
            "if ref_c[t] - x_c[t] >= delta: emit down event, ref_c -= n * delta",
            "```",
            "",
            by_name["level_crossing"].notes,
            "",
            "![Level crossing cochleagram](../outputs/cochlea_analysis/figures/level_crossing_cochleagram.png)",
            "",
            "![Level crossing raster](../outputs/cochlea_analysis/figures/level_crossing_raster.png)",
            "",
            "## 4. Direct Resonate-And-Fire Bank",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[RF neuron bank tuned across frequency]",
            "    B --> C[oscillator energy / state]",
            "    B --> D[spike raster]",
            "```",
            "",
            "```text",
            "velocity_c[t] = decay_c * velocity_c[t-1] + gain * x[t] - theta_c * state_c[t-1]",
            "state_c[t] = state_c[t-1] + theta_c * velocity_c[t]",
            "theta_c = 2*pi*f_c/sample_rate",
            "spike_c[t] = 1 if state_c[t] >= threshold else 0",
            "```",
            "",
            by_name["rf_bank"].notes,
            "",
            "![RF cochleagram](../outputs/cochlea_analysis/figures/rf_bank_cochleagram.png)",
            "",
            "![RF raster](../outputs/cochlea_analysis/figures/rf_bank_raster.png)",
            "",
            "## 5. Improvement Candidate: IIR Resonator Filterbank + LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[gammatone-like IIR resonator bank]",
            "    B --> C[half-wave rectification]",
            "    C --> D[full-rate LIF spike encoder]",
            "    D --> E[spike raster]",
            "```",
            "",
            "```text",
            "r_c = exp(-pi * bandwidth_c / sample_rate)",
            "theta_c = 2*pi*f_c/sample_rate",
            "y_c[t] = b_c*x[t] + 2*r_c*cos(theta_c)*y_c[t-1] - r_c^2*y_c[t-2]",
            "e_c[t] = max(y_c[t], 0)",
            "v_c[t] = beta_sample*v_c[t-1] + e_c[t]",
            "```",
            "",
            by_name["iir_lif"].notes,
            "",
            "![IIR LIF cochleagram](../outputs/cochlea_analysis/figures/iir_lif_cochleagram.png)",
            "",
            "![IIR LIF raster](../outputs/cochlea_analysis/figures/iir_lif_raster.png)",
            "",
            "## 6. Optimization Candidate: lfilter IIR + Optimized LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[torchaudio lfilter IIR bank]",
            "    B --> C[half-wave rectification]",
            "    C --> D[preallocated in-place LIF loop]",
            "    D --> E[spike raster]",
            "```",
            "",
            "```text",
            "y_c[t] = b_c*x[t] + 2*r_c*cos(theta_c)*y_c[t-1] - r_c^2*y_c[t-2]",
            "spikes = zeros(channels, samples)",
            "v.mul_(beta_sample).add_(e[:, t])",
            "spikes[:, t] = v >= threshold",
            "v.sub_(threshold * spikes[:, t]).clamp_(min=0)",
            "```",
            "",
            by_name["iir_lfilter_lif"].notes,
            "",
            "![lfilter IIR LIF cochleagram](../outputs/cochlea_analysis/figures/iir_lfilter_lif_cochleagram.png)",
            "",
            "![lfilter IIR LIF raster](../outputs/cochlea_analysis/figures/iir_lfilter_lif_raster.png)",
            "",
            "## 7. Optimization Candidate: lfilter IIR + TorchScript LIF",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[torchaudio lfilter IIR bank]",
            "    B --> C[half-wave rectification]",
            "    C --> D[TorchScript compiled LIF loop]",
            "    D --> E[spike raster]",
            "```",
            "",
            "```text",
            "@torch.jit.script",
            "for t in range(samples):",
            "    v = beta_sample*v + e[:, t]",
            "    spike = v >= threshold",
            "    v = max(v - threshold*spike, 0)",
            "```",
            "",
            by_name["iir_lfilter_jit_lif"].notes,
            "",
            "![TorchScript lfilter IIR LIF cochleagram](../outputs/cochlea_analysis/figures/iir_lfilter_jit_lif_cochleagram.png)",
            "",
            "![TorchScript lfilter IIR LIF raster](../outputs/cochlea_analysis/figures/iir_lfilter_jit_lif_raster.png)",
            "",
            "## 8. Optimization Candidate: lfilter IIR + Optimized LIF + Active-Window Gating",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[detect active echo window]",
            "    B --> C[crop with padding]",
            "    C --> D[torchaudio lfilter IIR bank]",
            "    D --> E[optimized LIF]",
            "    E --> F[scatter spikes back to full time axis]",
            "```",
            "",
            "```text",
            "active = abs(x[t]) >= threshold_fraction * max(abs(x))",
            "window = [first_active - padding, last_active + padding]",
            "run cochlea only over x[window]",
            "```",
            "",
            f"Current active-window settings: threshold fraction `{ACTIVE_WINDOW_THRESHOLD_FRACTION}`, padding `{ACTIVE_WINDOW_PADDING_S * 1_000.0:.1f} ms`, processed fraction `{gated_window.get('processed_fraction', 0.0):.3f}`.",
            "",
            by_name["iir_lfilter_lif_gated"].notes,
            "",
            "![gated lfilter IIR LIF cochleagram](../outputs/cochlea_analysis/figures/iir_lfilter_lif_gated_cochleagram.png)",
            "",
            "![gated lfilter IIR LIF raster](../outputs/cochlea_analysis/figures/iir_lfilter_lif_gated_raster.png)",
            "",
            "## 9. Combined Candidate: lfilter IIR + TorchScript LIF + Active-Window Gating",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[detect active echo window]",
            "    B --> C[crop with padding]",
            "    C --> D[torchaudio lfilter IIR bank]",
            "    D --> E[TorchScript compiled LIF loop]",
            "    E --> F[scatter full-length cochleagram/raster]",
            "```",
            "",
            "```text",
            "active = abs(x[t]) >= threshold_fraction * max(abs(x))",
            "filtered_window = lfilter(x[active_window], a, b)",
            "spikes_window = scripted_LIF(relu(filtered_window))",
            "spikes_full[:, active_window] = spikes_window",
            "```",
            "",
            f"Current active-window settings: threshold fraction `{ACTIVE_WINDOW_THRESHOLD_FRACTION}`, padding `{ACTIVE_WINDOW_PADDING_S * 1_000.0:.1f} ms`, processed fraction `{gated_jit_window.get('processed_fraction', 0.0):.3f}`.",
            "",
            by_name["iir_lfilter_jit_lif_gated"].notes,
            "",
            "![combined gated TorchScript IIR LIF cochleagram](../outputs/cochlea_analysis/figures/iir_lfilter_jit_lif_gated_cochleagram.png)",
            "",
            "![combined gated TorchScript IIR LIF raster](../outputs/cochlea_analysis/figures/iir_lfilter_jit_lif_gated_raster.png)",
            "",
            "## 10. Improvement Candidate: IIR Resonator Filterbank + Level Crossing",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[gammatone-like IIR resonator bank]",
            "    B --> C[level-crossing delta modulator]",
            "    C --> D[event raster]",
            "```",
            "",
            "```text",
            "y_c[t] = b_c*x[t] + 2*r_c*cos(theta_c)*y_c[t-1] - r_c^2*y_c[t-2]",
            "if y_c[t] - ref_c[t] >= delta: emit up event",
            "if ref_c[t] - y_c[t] >= delta: emit down event",
            "```",
            "",
            by_name["iir_level_crossing"].notes,
            "",
            "![IIR level-crossing cochleagram](../outputs/cochlea_analysis/figures/iir_level_crossing_cochleagram.png)",
            "",
            "![IIR level-crossing raster](../outputs/cochlea_analysis/figures/iir_level_crossing_raster.png)",
            "",
            "## 11. Improvement Candidate: Damped Wide-Band RF Bank",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[waveform] --> B[explicitly damped RF neuron bank]",
            "    B --> C[oscillator energy / state]",
            "    B --> D[spike raster]",
            "```",
            "",
            "```text",
            "decay_c = damping * exp(-theta_c / (2*Q))",
            "velocity_c[t] = decay_c*velocity_c[t-1] + gain*x[t] - theta_c*state_c[t-1]",
            "state_c[t] = state_c[t-1] + theta_c*velocity_c[t]",
            "spike_c[t] = 1 if state_c[t] >= lower_threshold else 0",
            "```",
            "",
            f"This variant uses `Q={RF_DAMPED_Q_FACTOR}`, `damping={RF_DAMPING_FACTOR}`, `gain={RF_DAMPED_INPUT_GAIN}`, and `threshold={RF_DAMPED_THRESHOLD}`.",
            "",
            by_name["damped_rf_bank"].notes,
            "",
            "![Damped RF cochleagram](../outputs/cochlea_analysis/figures/damped_rf_bank_cochleagram.png)",
            "",
            "![Damped RF raster](../outputs/cochlea_analysis/figures/damped_rf_bank_raster.png)",
            "",
            "## Initial Interpretation",
            "",
            "- The original model is the faithful baseline and has the most envelope-shaped representation, but it pays for FFT/IFFT reconstruction plus smoothing.",
            "- The Conv1D model stays in the time domain and removes explicit low-pass/downsample blocks, but naive FIR convolution is not automatically cheaper unless the kernels are short or optimized.",
            "- The IIR models test the same time-domain idea with recursive filters rather than long FIR kernels. This should be much cheaper in principle, although this first Python-loop implementation is not fully optimized.",
            "- The lfilter IIR variants test whether the theoretical IIR advantage appears when the recursive filter is moved out of Python.",
            "- The TorchScript LIF variant tests whether the remaining sequential threshold/reset loop can be accelerated without changing LIF dynamics.",
            "- Active-window gating tests a pragmatic event-inspired optimisation: silence is skipped, but the active segment is still processed densely.",
            "- The combined gated TorchScript variant is the current best candidate if we want to preserve reset-based LIF dynamics while reducing unnecessary silent-window compute.",
            "- The level-crossing model is the cleanest route toward event-based processing after the filterbank, but the filterbank itself is still dense in this first implementation.",
            "- The RF models are the most reduced conceptually because the resonators are both filters and spiking units, but their parameters need careful tuning before using them as full cochlea replacements.",
            "- Binarisation and event-based processing should be evaluated after we decide which of these mechanisms gives useful spike timing and channel selectivity.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for model_artifacts in artifacts.values():
        for name, path in model_artifacts.items():
            lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the cochlea comparison mini experiment.

    Returns:
        JSON-serializable summary dictionary.
    """
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPORT_PATH.parent)
    torch.manual_seed(7)
    config = _make_config()
    signal = _simulate_input_waveform(config)
    results = _run_all_models(signal, config)

    artifacts: dict[str, dict[str, str]] = {}
    for result in results:
        artifacts[result.name] = {
            "cochleagram": _plot_cochleagram(result, FIGURE_DIR / f"{result.name}_cochleagram.png"),
            "raster": _plot_spike_raster(result, FIGURE_DIR / f"{result.name}_raster.png"),
        }

    elapsed_s = time.perf_counter() - start
    payload: dict[str, object] = {
        "experiment": "cochlea_analysis",
        "elapsed_seconds": elapsed_s,
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "chirp_start_hz": config.chirp_start_hz,
            "chirp_end_hz": config.chirp_end_hz,
            "chirp_duration_s": config.chirp_duration_s,
            "signal_duration_s": config.signal_duration_s,
            "num_cochlea_channels": config.num_cochlea_channels,
            "cochlea_low_hz": config.cochlea_low_hz,
            "cochlea_high_hz": config.cochlea_high_hz,
            "filter_bandwidth_sigma": config.filter_bandwidth_sigma,
            "envelope_lowpass_hz": config.envelope_lowpass_hz,
            "envelope_downsample": config.envelope_downsample,
            "spike_threshold": config.spike_threshold,
            "spike_beta": config.spike_beta,
            "normalize_spike_envelope": config.normalize_spike_envelope,
            "transmit_gain": config.transmit_gain,
            "conv_kernel_size": CONV_KERNEL_SIZE,
            "iir_q_factor": IIR_Q_FACTOR,
            "iir_input_gain": IIR_INPUT_GAIN,
            "rf_q_factor": RF_Q_FACTOR,
            "rf_input_gain": RF_INPUT_GAIN,
            "rf_threshold": RF_THRESHOLD,
            "rf_damped_q_factor": RF_DAMPED_Q_FACTOR,
            "rf_damped_input_gain": RF_DAMPED_INPUT_GAIN,
            "rf_damped_threshold": RF_DAMPED_THRESHOLD,
            "rf_damping_factor": RF_DAMPING_FACTOR,
        },
        "models": [
            {
                "name": result.name,
                "title": result.title,
                "elapsed_seconds": result.elapsed_s,
                "elapsed_ms": result.elapsed_s * 1_000.0,
                "elapsed_ms_per_channel": result.elapsed_s / config.num_cochlea_channels * 1_000.0,
                "flops_estimate": result.flops_estimate,
                "sops_estimate": result.sops_estimate,
                "spike_density": result.sops_estimate / float(result.spikes.numel()),
                "time_rate_hz": result.time_rate_hz,
                "cochleagram_shape": list(result.cochleagram.shape),
                "spike_shape": list(result.spikes.shape),
                "notes": result.notes,
            }
            for result in results
        ],
        "artifacts": artifacts,
        "active_window_gating": getattr(_run_iir_lfilter_lif_gated_cochlea, "last_window", {}),
        "active_window_gating_jit": getattr(_run_iir_lfilter_jit_lif_gated_cochlea, "last_window", {}),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(results, artifacts, config, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
