from __future__ import annotations

"""Manual neuron and encoder dynamics for mini-model experiments.

The functions in this file intentionally avoid hidden SNN-library behaviour.
They are written as simple time loops so the voltage/state traces in the plots
match the equations in the reports.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LIFTrace:
    """Recorded state from a leaky integrate-and-fire simulation.

    Attributes:
        time_s: Time axis in seconds.
        input_current: Input current supplied at each time step.
        membrane: Membrane voltage after integration and any reset.
        spikes: Binary spike train, with `1` indicating a threshold crossing.
    """

    time_s: np.ndarray
    input_current: np.ndarray
    membrane: np.ndarray
    spikes: np.ndarray


@dataclass
class RFTrace:
    """Recorded state from a resonate-and-fire simulation.

    Attributes:
        time_s: Time axis in seconds.
        input_current: Input drive supplied at each time step.
        state: Oscillator displacement-like state after integration/reset.
        velocity: Oscillator velocity-like state.
        spikes: Binary spike train, with `1` indicating a threshold crossing.
    """

    time_s: np.ndarray
    input_current: np.ndarray
    state: np.ndarray
    velocity: np.ndarray
    spikes: np.ndarray


@dataclass
class LevelCrossingTrace:
    """Recorded state from an asynchronous level-crossing encoder.

    Attributes:
        time_s: Time axis in seconds.
        signal: Input waveform.
        reference: Internal quantized reference level after updates.
        up_spikes: Binary events emitted when the signal rises by `delta`.
        down_spikes: Binary events emitted when the signal falls by `delta`.
    """

    time_s: np.ndarray
    signal: np.ndarray
    reference: np.ndarray
    up_spikes: np.ndarray
    down_spikes: np.ndarray

    @property
    def total_spikes(self) -> np.ndarray:
        """Return the combined up/down event train."""
        return self.up_spikes + self.down_spikes


def simulate_lif(
    input_current: np.ndarray,
    *,
    dt_s: float,
    tau_s: float = 0.012,
    resistance: float = 1.0,
    threshold: float = 1.0,
    reset_mode: str = "subtract",
) -> LIFTrace:
    """Simulate a current-driven leaky integrate-and-fire neuron.

    The update is an explicit Euler approximation of:

    `dv/dt = (-v + R * I) / tau`.

    Args:
        input_current: One-dimensional input current over time.
        dt_s: Simulation timestep in seconds.
        tau_s: Membrane time constant in seconds. Larger values leak more slowly.
        resistance: Input scaling term `R`.
        threshold: Voltage threshold for spike emission.
        reset_mode: Either `"subtract"` for subtractive reset or `"zero"` for
            hard reset to zero.

    Returns:
        Recorded input, membrane voltage, and spike trace.

    Raises:
        ValueError: If `reset_mode` is unsupported.
    """
    current = np.asarray(input_current, dtype=np.float64)
    time_s = np.arange(current.size, dtype=np.float64) * dt_s
    membrane = np.zeros_like(current)
    spikes = np.zeros_like(current)
    voltage = 0.0
    leak = dt_s / max(tau_s, 1e-12)

    for index, drive in enumerate(current):
        # Integrate the input and leak the voltage back toward rest.
        voltage += leak * (-voltage + resistance * drive)
        if voltage >= threshold:
            spikes[index] = 1.0
            if reset_mode == "zero":
                voltage = 0.0
            elif reset_mode == "subtract":
                voltage = max(0.0, voltage - threshold)
            else:
                raise ValueError(f"Unsupported reset mode '{reset_mode}'.")
        membrane[index] = voltage

    return LIFTrace(time_s=time_s, input_current=current, membrane=membrane, spikes=spikes)


def simulate_resonate_and_fire(
    input_current: np.ndarray,
    *,
    dt_s: float,
    resonant_frequency_hz: float = 120.0,
    q_factor: float = 8.0,
    input_gain: float = 0.08,
    threshold: float = 1.0,
    reset_mode: str = "subtract",
) -> RFTrace:
    """Simulate a simple discrete resonate-and-fire neuron.

    This uses a damped second-order oscillator state. It is intentionally
    transparent rather than optimized for gradient training.

    Args:
        input_current: One-dimensional input drive over time.
        dt_s: Simulation timestep in seconds.
        resonant_frequency_hz: Frequency that the oscillator responds to most
            strongly.
        q_factor: Resonance sharpness. Larger values decay more slowly and make
            a narrower band-pass response.
        input_gain: Scaling applied to the external input.
        threshold: State threshold for spike emission.
        reset_mode: Either `"subtract"` for subtractive reset or `"zero"` for
            hard reset to zero.

    Returns:
        Recorded input, oscillator state, velocity, and spike trace.

    Raises:
        ValueError: If `reset_mode` is unsupported.
    """
    current = np.asarray(input_current, dtype=np.float64)
    time_s = np.arange(current.size, dtype=np.float64) * dt_s
    state_trace = np.zeros_like(current)
    velocity_trace = np.zeros_like(current)
    spikes = np.zeros_like(current)
    state = 0.0
    velocity = 0.0
    theta = 2.0 * np.pi * resonant_frequency_hz * dt_s
    decay = np.exp(-theta / max(2.0 * q_factor, 1e-12))

    for index, drive in enumerate(current):
        # This is a compact discrete oscillator: the -theta * state term pulls
        # the state back, while velocity carries the resonance across time.
        velocity = decay * velocity + input_gain * drive - theta * state
        state = state + theta * velocity
        if state >= threshold:
            spikes[index] = 1.0
            if reset_mode == "zero":
                state = 0.0
                velocity = 0.0
            elif reset_mode == "subtract":
                state = max(0.0, state - threshold)
            else:
                raise ValueError(f"Unsupported reset mode '{reset_mode}'.")
        state_trace[index] = state
        velocity_trace[index] = velocity

    return RFTrace(
        time_s=time_s,
        input_current=current,
        state=state_trace,
        velocity=velocity_trace,
        spikes=spikes,
    )


def simulate_level_crossing(
    signal: np.ndarray,
    *,
    dt_s: float,
    delta: float = 0.15,
    refractory_s: float = 0.0,
) -> LevelCrossingTrace:
    """Generate asynchronous up/down events when a signal crosses delta levels.

    This is a simple delta-modulation style encoder. It emits events only when
    the signal has changed enough from its internal reference level.

    Args:
        signal: One-dimensional signal to encode.
        dt_s: Simulation timestep in seconds.
        delta: Signal change required to emit an up/down event.
        refractory_s: Optional dead time after an event. During the refractory
            period the encoder ignores further crossings.

    Returns:
        Recorded input, internal reference level, and up/down event trains.
    """
    waveform = np.asarray(signal, dtype=np.float64)
    time_s = np.arange(waveform.size, dtype=np.float64) * dt_s
    reference = np.zeros_like(waveform)
    up_spikes = np.zeros_like(waveform)
    down_spikes = np.zeros_like(waveform)
    level = float(waveform[0])
    refractory_steps = max(0, int(round(refractory_s / dt_s)))
    next_allowed = 0

    for index, value in enumerate(waveform):
        reference[index] = level
        if index < next_allowed:
            continue
        if value - level >= delta:
            up_spikes[index] = 1.0
            # If the signal jumped across multiple quantization levels in one
            # step, move the reference by the full number of crossed levels.
            steps = max(1, int(np.floor((value - level) / delta)))
            level += steps * delta
            next_allowed = index + refractory_steps
        elif level - value >= delta:
            down_spikes[index] = 1.0
            steps = max(1, int(np.floor((level - value) / delta)))
            level -= steps * delta
            next_allowed = index + refractory_steps
        reference[index] = level

    return LevelCrossingTrace(
        time_s=time_s,
        signal=waveform,
        reference=reference,
        up_spikes=up_spikes,
        down_spikes=down_spikes,
    )


def vector_strength(spikes: np.ndarray, frequency_hz: float, dt_s: float) -> float:
    """Measure spike phase concentration relative to a periodic stimulus.

    Args:
        spikes: Binary or non-zero spike/event train.
        frequency_hz: Frequency of the reference periodic stimulus.
        dt_s: Simulation timestep in seconds.

    Returns:
        Vector strength in `[0, 1]`. Values near `1` indicate strong phase
        locking; values near `0` indicate dispersed spike phases.
    """
    spike_indices = np.flatnonzero(np.asarray(spikes) > 0.0)
    if spike_indices.size == 0:
        return 0.0
    phases = 2.0 * np.pi * frequency_hz * spike_indices.astype(np.float64) * dt_s
    return float(np.abs(np.exp(1j * phases).mean()))


def spike_phases(spikes: np.ndarray, frequency_hz: float, dt_s: float) -> np.ndarray:
    """Return spike phases relative to a reference sinusoid.

    Args:
        spikes: Binary or non-zero spike/event train.
        frequency_hz: Frequency of the reference periodic stimulus.
        dt_s: Simulation timestep in seconds.

    Returns:
        Array of phases in radians in the range `[0, 2*pi)`.
    """
    spike_indices = np.flatnonzero(np.asarray(spikes) > 0.0)
    if spike_indices.size == 0:
        return np.array([], dtype=np.float64)
    phases = 2.0 * np.pi * frequency_hz * spike_indices.astype(np.float64) * dt_s
    return np.mod(phases, 2.0 * np.pi)


def half_peak_to_peak(signal: np.ndarray, *, discard_fraction: float = 0.25) -> float:
    """Estimate steady-state oscillation amplitude.

    Args:
        signal: One-dimensional signal.
        discard_fraction: Initial fraction of the trace to ignore so that
            transients do not dominate the amplitude estimate.

    Returns:
        Half of the peak-to-peak range after discarding the initial transient.
    """
    values = np.asarray(signal, dtype=np.float64)
    start = int(values.size * discard_fraction)
    window = values[start:]
    if window.size == 0:
        return 0.0
    return float(0.5 * (window.max() - window.min()))
