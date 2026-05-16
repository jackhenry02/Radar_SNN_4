from __future__ import annotations

"""Standalone line-attractor analysis for a future SC readout.

This script does not use the current distance pathway outputs. It analyses a
synthetic one-dimensional continuous attractor neural network that could later
replace the simple SC centre-of-mass readout.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "line_attractor_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "line_attractor_model.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"


@dataclass(frozen=True)
class LineAttractorParams:
    """Numerical parameters for the line-attractor analysis.

    Attributes:
        length_m: Physical length of the represented line.
        num_neurons: Number of neurons in one population.
        tau_s: Neural rate time constant.
        dt_s: Integration time step.
        t_end_s: End time for transient simulations.
        tuning_sigma_m: Width of feedforward distance tuning curves.
        recurrent_sigma_m: Width of recurrent local excitation.
        fisher_noise_sigma: Gaussian readout noise used for Fisher information.
        readout_noise_sigma: Gaussian readout noise used for decoding trials.
        symmetric_alpha: Largest recurrent eigenvalue for the symmetric line
            model. Values below one are asymptotically stable; values close to
            one approximate a continuous attractor.
        balanced_alpha_prime: Gain of the balanced E/I nilpotent block. This
            can be greater than one while the continuous-time system remains
            asymptotically stable.
        num_decode_positions: Number of synthetic true positions used in the
            decoding-error experiment.
        num_decode_trials: Noisy readout trials per position.
        rng_seed: Random seed for decoding noise.
    """

    length_m: float = 10.0
    num_neurons: int = 120
    tau_s: float = 0.020
    dt_s: float = 0.001
    t_end_s: float = 0.060
    tuning_sigma_m: float = 0.42
    recurrent_sigma_m: float = 0.70
    fisher_noise_sigma: float = 0.05
    readout_noise_sigma: float = 0.04
    symmetric_alpha: float = 0.985
    balanced_alpha_prime: float = 4.0
    num_decode_positions: int = 45
    num_decode_trials: int = 160
    rng_seed: int = 109


@dataclass(frozen=True)
class ModelMatrices:
    """State-space matrices for one line-attractor model.

    Attributes:
        name: Human-readable model name.
        W: Recurrent matrix.
        B: Input matrix from sensory population code into state.
        C: Readout matrix from state to decoded population activity.
        state_kind: `single` for one population or `balanced_ei` for E/I state.
    """

    name: str
    W: np.ndarray
    B: np.ndarray
    C: np.ndarray
    state_kind: str


def gaussian_kernel(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate an unnormalised Gaussian kernel.

    Args:
        distance: Distance from kernel centre.
        sigma: Kernel width.

    Returns:
        Kernel value.
    """
    return np.exp(-0.5 * (distance / sigma) ** 2)


def positions(params: LineAttractorParams) -> np.ndarray:
    """Return preferred positions for the neural population."""
    return np.linspace(0.0, params.length_m, params.num_neurons)


def raw_population_code(params: LineAttractorParams, stimulus_m: float) -> np.ndarray:
    """Return a plain Gaussian population code on the finite line.

    Args:
        params: Model parameters.
        stimulus_m: True represented position.

    Returns:
        Activity over preferred positions.
    """
    x = positions(params)
    return gaussian_kernel(x - stimulus_m, params.tuning_sigma_m)


def reflected_population_code(params: LineAttractorParams, stimulus_m: float) -> np.ndarray:
    """Return a boundary-corrected population code using reflected images.

    The finite line has boundaries at 0 and L. A raw Gaussian loses neighbours
    near either boundary, reducing total drive and Fisher information. The
    reflected code adds mirrored stimulus images outside the interval, which is
    the same idea as Neumann boundary correction in diffusion problems.

    Args:
        params: Model parameters.
        stimulus_m: True represented position.

    Returns:
        Boundary-corrected activity over preferred positions.
    """
    x = positions(params)
    length = params.length_m
    sigma = params.tuning_sigma_m
    return (
        gaussian_kernel(x - stimulus_m, sigma)
        + gaussian_kernel(x + stimulus_m, sigma)
        + gaussian_kernel(x - (2.0 * length - stimulus_m), sigma)
    )


def population_derivative(
    params: LineAttractorParams,
    stimulus_m: float,
    *,
    reflected: bool,
) -> np.ndarray:
    """Return derivative of population activity with respect to position.

    Args:
        params: Model parameters.
        stimulus_m: True represented position.
        reflected: Whether to use reflected boundary correction.

    Returns:
        Derivative vector `dh/dx`.
    """
    x = positions(params)
    length = params.length_m
    sigma2 = params.tuning_sigma_m**2
    z1 = x - stimulus_m
    deriv = (z1 / sigma2) * gaussian_kernel(z1, params.tuning_sigma_m)
    if reflected:
        z2 = x + stimulus_m
        z3 = x - (2.0 * length - stimulus_m)
        deriv += -(z2 / sigma2) * gaussian_kernel(z2, params.tuning_sigma_m)
        deriv += -(z3 / sigma2) * gaussian_kernel(z3, params.tuning_sigma_m)
    return deriv


def fisher_information_curve(
    params: LineAttractorParams,
    stimulus_grid: np.ndarray,
    *,
    reflected: bool,
) -> np.ndarray:
    """Compute Gaussian-noise Fisher information over position.

    Args:
        params: Model parameters.
        stimulus_grid: Positions where Fisher information is evaluated.
        reflected: Whether to use reflected boundary correction.

    Returns:
        Fisher information at each position.
    """
    values = []
    for stimulus_m in stimulus_grid:
        derivative = population_derivative(params, float(stimulus_m), reflected=reflected)
        values.append(float(np.sum(derivative**2) / params.fisher_noise_sigma**2))
    return np.array(values)


def reflected_recurrent_kernel(params: LineAttractorParams, sigma: float) -> np.ndarray:
    """Return a reflected local recurrent kernel matrix on a finite line.

    Args:
        params: Model parameters.
        sigma: Kernel width.

    Returns:
        Matrix `[target_neuron, source_neuron]`.
    """
    x = positions(params)
    length = params.length_m
    target = x[:, None]
    source = x[None, :]
    return (
        gaussian_kernel(target - source, sigma)
        + gaussian_kernel(target + source, sigma)
        + gaussian_kernel(target - (2.0 * length - source), sigma)
    )


def raw_recurrent_kernel(params: LineAttractorParams, sigma: float) -> np.ndarray:
    """Return an uncorrected local recurrent kernel matrix on a finite line."""
    x = positions(params)
    return gaussian_kernel(x[:, None] - x[None, :], sigma)


def rescale_spectral_abscissa(matrix: np.ndarray, target_alpha: float) -> np.ndarray:
    """Scale a matrix so its largest real eigenvalue equals `target_alpha`.

    Args:
        matrix: Recurrent matrix.
        target_alpha: Desired largest real eigenvalue.

    Returns:
        Rescaled matrix.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    current = float(np.max(np.real(eigenvalues)))
    if abs(current) < 1e-12:
        raise ValueError("Cannot rescale matrix with near-zero spectral abscissa.")
    return matrix * (target_alpha / current)


def build_models(params: LineAttractorParams) -> list[ModelMatrices]:
    """Build no-recurrence, symmetric line, and balanced E/I line models.

    Args:
        params: Model parameters.

    Returns:
        List of model matrix sets.
    """
    m = params.num_neurons
    eye = np.eye(m)
    local = reflected_recurrent_kernel(params, params.recurrent_sigma_m)
    local = local / np.maximum(local.sum(axis=1, keepdims=True), 1e-12)

    no_recurrence = ModelMatrices(
        name="No recurrence",
        W=np.zeros((m, m)),
        B=eye,
        C=eye,
        state_kind="single",
    )
    symmetric_line = ModelMatrices(
        name="Symmetric line attractor",
        W=rescale_spectral_abscissa(local, params.symmetric_alpha),
        B=eye,
        C=eye,
        state_kind="single",
    )

    high_gain_local = rescale_spectral_abscissa(local, params.balanced_alpha_prime)
    balanced_w = np.block(
        [
            [high_gain_local, -high_gain_local],
            [high_gain_local, -high_gain_local],
        ]
    )
    balanced_line = ModelMatrices(
        name="Balanced E/I line attractor",
        W=balanced_w,
        B=np.vstack([eye, np.zeros((m, m))]),
        C=np.hstack([eye, np.zeros((m, m))]),
        state_kind="balanced_ei",
    )
    return [no_recurrence, symmetric_line, balanced_line]


def simulate_state(
    params: LineAttractorParams,
    model: ModelMatrices,
    stimulus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate autonomous rate dynamics after an impulse input.

    The sensory input is treated as an initial pulse into the state. After that,
    the recurrent dynamics evolve without further input:

    `tau dr/dt = -r + W r`.

    Args:
        params: Model parameters.
        model: Model matrices.
        stimulus: Input population code.

    Returns:
        Tuple `(times, state_history, readout_history)`.
    """
    num_steps = int(round(params.t_end_s / params.dt_s))
    times = np.arange(num_steps + 1) * params.dt_s
    state = model.B @ stimulus
    state_history = np.empty((num_steps + 1, state.size), dtype=np.float64)
    readout_history = np.empty((num_steps + 1, params.num_neurons), dtype=np.float64)
    state_history[0] = state
    readout_history[0] = model.C @ state
    for step in range(1, num_steps + 1):
        state = state + params.dt_s / params.tau_s * (-state + model.W @ state)
        state_history[step] = state
        readout_history[step] = model.C @ state
    return times, state_history, readout_history


def system_matrix(params: LineAttractorParams, model: ModelMatrices) -> np.ndarray:
    """Return continuous-time linear system matrix `A = (-I + W)/tau`."""
    return (-np.eye(model.W.shape[0]) + model.W) / params.tau_s


def transition_operator_gains(
    params: LineAttractorParams,
    model: ModelMatrices,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate state and input-output operator gains over time.

    The propagator is advanced with the same Euler step as the state
    simulation. This avoids requiring matrix-exponential dependencies.

    Args:
        params: Model parameters.
        model: Model matrices.

    Returns:
        Tuple `(times, state_operator_norm, input_output_norm)`.
    """
    num_steps = int(round(params.t_end_s / params.dt_s))
    times = np.arange(num_steps + 1) * params.dt_s
    transition = np.eye(model.W.shape[0])
    update = np.eye(model.W.shape[0]) + params.dt_s * system_matrix(params, model)
    state_gain = np.empty(num_steps + 1, dtype=np.float64)
    readout_gain = np.empty(num_steps + 1, dtype=np.float64)
    for step in range(num_steps + 1):
        state_gain[step] = np.linalg.svd(transition, compute_uv=False)[0]
        input_output = model.C @ transition @ model.B
        readout_gain[step] = np.linalg.svd(input_output, compute_uv=False)[0]
        transition = update @ transition
    return times, state_gain, readout_gain


def decode_center_of_mass(params: LineAttractorParams, readout: np.ndarray) -> np.ndarray:
    """Decode line position using a clipped centre-of-mass readout.

    Args:
        params: Model parameters.
        readout: Population activity with final axis indexing neurons.

    Returns:
        Decoded position in metres.
    """
    x = positions(params)
    positive = np.maximum(readout, 0.0)
    total = positive.sum(axis=-1)
    fallback = 0.5 * params.length_m
    decoded = np.sum(positive * x, axis=-1) / np.maximum(total, 1e-12)
    return np.where(total > 1e-12, decoded, fallback)


def run_decoding_trials(
    params: LineAttractorParams,
    models: list[ModelMatrices],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run noisy centre-of-mass decoding for each model.

    Args:
        params: Model parameters.
        models: Models to compare.

    Returns:
        Tuple `(times, mean_absolute_error_by_model)`.
    """
    rng = np.random.default_rng(params.rng_seed)
    true_positions = np.linspace(0.2, params.length_m - 0.2, params.num_decode_positions)
    model_errors: dict[str, list[np.ndarray]] = {model.name: [] for model in models}
    times = None
    for stimulus_m in true_positions:
        stimulus = reflected_population_code(params, float(stimulus_m))
        for model in models:
            sim_times, _, readout = simulate_state(params, model, stimulus)
            times = sim_times
            noise = rng.normal(
                loc=0.0,
                scale=params.readout_noise_sigma,
                size=(params.num_decode_trials, *readout.shape),
            )
            noisy_readout = readout[None, :, :] + noise
            decoded = decode_center_of_mass(params, noisy_readout)
            abs_error = np.abs(decoded - stimulus_m)
            model_errors[model.name].append(abs_error.mean(axis=0))
    assert times is not None
    return times, {
        name: np.mean(np.stack(errors, axis=0), axis=0)
        for name, errors in model_errors.items()
    }


def fisher_through_time(
    params: LineAttractorParams,
    models: list[ModelMatrices],
    stimulus_grid: np.ndarray,
    time_indices: list[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Compute finite-difference Fisher information after recurrent dynamics.

    Args:
        params: Model parameters.
        models: Models to compare.
        stimulus_grid: Positions where Fisher information is evaluated.
        time_indices: Simulation indices where readout Fisher information is
            measured.

    Returns:
        Nested dictionary: model name -> time index -> Fisher curve.
    """
    delta = 1e-3 * params.length_m
    output: dict[str, dict[int, list[float]]] = {
        model.name: {index: [] for index in time_indices}
        for model in models
    }
    for stimulus_m in stimulus_grid:
        low = max(0.0, float(stimulus_m) - delta)
        high = min(params.length_m, float(stimulus_m) + delta)
        if high == low:
            continue
        stim_low = reflected_population_code(params, low)
        stim_high = reflected_population_code(params, high)
        for model in models:
            _, _, read_low = simulate_state(params, model, stim_low)
            _, _, read_high = simulate_state(params, model, stim_high)
            derivative = (read_high - read_low) / (high - low)
            for index in time_indices:
                fisher = float(np.sum(derivative[index] ** 2) / params.fisher_noise_sigma**2)
                output[model.name][index].append(fisher)
    return {
        model_name: {
            index: np.array(values)
            for index, values in per_time.items()
        }
        for model_name, per_time in output.items()
    }


def plot_tuning_and_fisher(params: LineAttractorParams, path: Path) -> str:
    """Plot raw vs boundary-corrected tuning and Fisher information."""
    x = positions(params)
    stimulus_grid = np.linspace(0.0, params.length_m, 300)
    example_positions = [0.25, params.length_m / 2.0, params.length_m - 0.25]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for stimulus_m in example_positions:
        axes[0].plot(x, raw_population_code(params, stimulus_m), linewidth=1.8, label=f"{stimulus_m:.2f} m")
        axes[1].plot(x, reflected_population_code(params, stimulus_m), linewidth=1.8, label=f"{stimulus_m:.2f} m")
    axes[0].set_title("Raw finite-line tuning")
    axes[1].set_title("Reflected boundary-corrected tuning")
    raw_fi = fisher_information_curve(params, stimulus_grid, reflected=False)
    reflected_fi = fisher_information_curve(params, stimulus_grid, reflected=True)
    axes[2].plot(stimulus_grid, raw_fi / np.max(reflected_fi), label="raw", linewidth=2.0)
    axes[2].plot(stimulus_grid, reflected_fi / np.max(reflected_fi), label="reflected", linewidth=2.0)
    axes[2].set_title("Normalised Fisher information")
    axes[2].set_ylabel("relative FI")
    for ax in axes:
        ax.set_xlabel("position (m)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("activity")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False)
    return save_figure(fig, path)


def plot_boundary_weights(params: LineAttractorParams, path: Path) -> str:
    """Plot uncorrected and reflected recurrent kernels plus row sums."""
    raw = raw_recurrent_kernel(params, params.recurrent_sigma_m)
    reflected = reflected_recurrent_kernel(params, params.recurrent_sigma_m)
    raw_norm = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    reflected_norm = reflected / np.maximum(reflected.sum(axis=1, keepdims=True), 1e-12)
    x = positions(params)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    extent = [0.0, params.length_m, 0.0, params.length_m]
    image0 = axes[0, 0].imshow(raw, origin="lower", aspect="auto", extent=extent)
    axes[0, 0].set_title("Raw local recurrent kernel")
    fig.colorbar(image0, ax=axes[0, 0], fraction=0.046)
    image1 = axes[0, 1].imshow(reflected, origin="lower", aspect="auto", extent=extent)
    axes[0, 1].set_title("Reflected boundary-corrected kernel")
    fig.colorbar(image1, ax=axes[0, 1], fraction=0.046)
    axes[1, 0].plot(x, raw.sum(axis=1), label="raw row sum")
    axes[1, 0].plot(x, reflected.sum(axis=1), label="reflected row sum")
    axes[1, 0].set_title("Boundary input loss before balancing")
    axes[1, 0].set_ylabel("row sum")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(x, raw_norm.sum(axis=1), label="raw normalised")
    axes[1, 1].plot(x, reflected_norm.sum(axis=1), label="reflected normalised")
    axes[1, 1].set_title("Row-sum balancing after normalisation")
    axes[1, 1].set_ylabel("row sum")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("position (m)")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("target position (m)")
    axes[0, 1].set_ylabel("target position (m)")
    return save_figure(fig, path)


def plot_model_snapshots(params: LineAttractorParams, models: list[ModelMatrices], path: Path) -> str:
    """Plot readout snapshots for each model after an impulse stimulus."""
    stimulus_m = 0.63 * params.length_m
    stimulus = reflected_population_code(params, stimulus_m)
    snapshot_times = [0.0, 0.010, 0.020, 0.060]
    x = positions(params)
    fig, axes = plt.subplots(len(models), len(snapshot_times), figsize=(13.5, 7.0), sharex=True, sharey="row")
    for row, model in enumerate(models):
        times, _, readout = simulate_state(params, model, stimulus)
        for col, snapshot_s in enumerate(snapshot_times):
            index = int(np.argmin(np.abs(times - snapshot_s)))
            axes[row, col].plot(x, readout[index], linewidth=2.0)
            axes[row, col].axvline(stimulus_m, color="#111827", linestyle=":", linewidth=1.2)
            if row == 0:
                axes[row, col].set_title(f"t = {snapshot_s * 1_000:.0f} ms")
            if col == 0:
                axes[row, col].set_ylabel(model.name)
            axes[row, col].grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("position (m)")
    return save_figure(fig, path)


def plot_stability_and_transients(
    params: LineAttractorParams,
    models: list[ModelMatrices],
    path: Path,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Plot eigenvalue stability and transient operator growth."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    summary: dict[str, dict[str, float]] = {}
    for model in models:
        a_matrix = system_matrix(params, model)
        eigenvalues = np.linalg.eigvals(a_matrix)
        times, state_gain, readout_gain = transition_operator_gains(params, model)
        axes[0].scatter(np.real(eigenvalues), np.imag(eigenvalues), s=14, alpha=0.65, label=model.name)
        axes[1].plot(times / params.tau_s, state_gain, linewidth=2.0, label=model.name)
        axes[2].plot(times / params.tau_s, readout_gain, linewidth=2.0, label=model.name)
        summary[model.name] = {
            "max_real_A": float(np.max(np.real(eigenvalues))),
            "peak_state_gain": float(np.max(state_gain)),
            "peak_readout_gain": float(np.max(readout_gain)),
            "final_state_gain": float(state_gain[-1]),
            "final_readout_gain": float(readout_gain[-1]),
        }
    axes[0].axvline(0.0, color="#111827", linestyle="--", linewidth=1.2)
    axes[0].set_title("Eigenvalues of A = (-I + W)/tau")
    axes[0].set_xlabel("real")
    axes[0].set_ylabel("imaginary")
    axes[1].set_title("State transient growth")
    axes[1].set_xlabel("time / tau")
    axes[1].set_ylabel("||exp(tA)|| approx.")
    axes[1].set_yscale("log")
    axes[2].set_title("Input-output transient growth")
    axes[2].set_xlabel("time / tau")
    axes[2].set_ylabel("||C exp(tA) B|| approx.")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    return save_figure(fig, path), summary


def plot_fisher_through_time(
    params: LineAttractorParams,
    models: list[ModelMatrices],
    path: Path,
) -> str:
    """Plot Fisher information through time for each line model."""
    stimulus_grid = np.linspace(0.05, params.length_m - 0.05, 90)
    time_seconds = [0.0, 0.020, 0.060]
    time_indices = [int(round(value / params.dt_s)) for value in time_seconds]
    fisher = fisher_through_time(params, models, stimulus_grid, time_indices)
    fig, axes = plt.subplots(1, len(models), figsize=(14.0, 3.8), sharey=True)
    for ax, model in zip(axes, models):
        for time_s, time_index in zip(time_seconds, time_indices):
            values = fisher[model.name][time_index]
            ax.plot(stimulus_grid[: values.size], values / np.max(values), linewidth=2.0, label=f"{time_s*1_000:.0f} ms")
        ax.set_title(model.name)
        ax.set_xlabel("true position (m)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("normalised FI")
    axes[-1].legend(frameon=False)
    return save_figure(fig, path)


def plot_decoding_error(
    params: LineAttractorParams,
    models: list[ModelMatrices],
    path: Path,
) -> tuple[str, dict[str, float]]:
    """Plot noisy centre-of-mass decoding error over time."""
    times, errors = run_decoding_trials(params, models)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    final_errors = {}
    for name, mean_abs_error in errors.items():
        ax.plot(times * 1_000.0, mean_abs_error, linewidth=2.0, label=name)
        final_errors[name] = float(mean_abs_error[-1])
    ax.set_title("Noisy centre-of-mass decoding error")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean absolute error (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path), final_errors


def write_report(
    params: LineAttractorParams,
    artifacts: dict[str, str],
    stability_summary: dict[str, dict[str, float]],
    final_decoding_errors: dict[str, float],
    elapsed_s: float,
) -> None:
    """Write the line-attractor markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stability_rows = []
    for name, values in stability_summary.items():
        stability_rows.append(
            "| "
            f"{name} | "
            f"`{values['max_real_A']:.3f}` | "
            f"`{values['peak_state_gain']:.2f}` | "
            f"`{values['peak_readout_gain']:.2f}` | "
            f"`{values['final_state_gain']:.3f}` | "
            f"`{values['final_readout_gain']:.3f}` |"
        )
    error_rows = [
        f"| {name} | `{error:.4f} m` |"
        for name, error in final_decoding_errors.items()
    ]
    lines = [
        "# Line Attractor Model Analysis",
        "",
        "This report analyses a standalone line-attractor model for a future superior-colliculus-style distance readout. It does not use the current distance pathway outputs. The input is a synthetic one-dimensional population code over position.",
        "",
        "## Aim",
        "",
        "The current SC readout in the full distance pathway uses a direct centre of mass over an AC distance map. A continuous attractor neural network is a more neuromorphic alternative: the input briefly creates a bump of activity, recurrent structure maintains and sharpens the bump, and the final distance is decoded from the population.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Synthetic position input] --> B[Gaussian population code]",
        "    B --> C[Line attractor dynamics]",
        "    C --> D[Stable activity bump]",
        "    D --> E[Centre-of-mass readout]",
        "```",
        "",
        "## From Ring To Line",
        "",
        "The previous ring model represented a circular variable, so the distance between neurons wrapped around:",
        "",
        "```text",
        "d_ring(phi_i, phi_j) = angle_wrap(phi_i - phi_j)",
        "W_ij = K(d_ring(phi_i, phi_j))",
        "```",
        "",
        "Distance is not circular. A target at `10 m` is not close to a target at `0 m`, so the ring is opened into a finite line:",
        "",
        "```text",
        "x_i in [0, L]",
        "d_line(i, j) = x_i - x_j",
        "W_ij = K(x_i - x_j)",
        "```",
        "",
        "The main complication is the boundary. A neuron near `0 m` or `L` has fewer neighbours on one side, so an uncorrected line has weaker recurrent input and weaker Fisher information at the edges. This is the finite-line equivalent of losing half of the kernel.",
        "",
        "## Population Code And Fisher Information",
        "",
        "A position `x` is encoded as a Gaussian population bump:",
        "",
        "```text",
        "h_i(x) = exp(-(x_i - x)^2 / (2 sigma_h^2))",
        "```",
        "",
        "For independent Gaussian readout noise with variance `sigma_n^2`, Fisher information is:",
        "",
        "```text",
        "J(x) = (dh/dx)^T Sigma^-1 (dh/dx)",
        "J(x) = (1 / sigma_n^2) * sum_i (dh_i/dx)^2",
        "```",
        "",
        "Fisher information is useful because it measures local discriminability: high `J(x)` means a small change in distance causes a large, noise-robust change in the population pattern. Through the Cramer-Rao bound, any unbiased estimator has variance at least `1 / J(x)`. Therefore, a good readout representation should have high and reasonably flat Fisher information across the whole represented interval.",
        "",
        "## Boundary Correction",
        "",
        "The boundary-corrected line uses reflected image terms:",
        "",
        "```text",
        "h_i_ref(x) = G(x_i - x) + G(x_i + x) + G(x_i - (2L - x))",
        "```",
        "",
        "This is equivalent to reflecting the tuning curve outside the interval. It prevents edge neurons from losing half their input and makes the representation closer to a finite line with no-flux boundaries.",
        "",
        "![Tuning and Fisher information](../outputs/line_attractor_analysis/figures/tuning_and_fisher.png)",
        "",
        "The same correction is applied to recurrent kernels. After that, row-sum balancing ensures every neuron receives comparable total recurrent drive:",
        "",
        "```text",
        "K_ij_ref = G(x_i - x_j) + G(x_i + x_j) + G(x_i - (2L - x_j))",
        "W_ij = alpha * K_ij_ref / sum_j K_ij_ref",
        "```",
        "",
        "![Boundary recurrent weights](../outputs/line_attractor_analysis/figures/boundary_weights.png)",
        "",
        "## Dynamics",
        "",
        "The analysed rate model is:",
        "",
        "```text",
        "tau dr/dt = -r + W r + B u(t)",
        "```",
        "",
        "For this analysis the input is a brief impulse, so after `t=0` the autonomous dynamics are:",
        "",
        "```text",
        "tau dr/dt = -r + W r",
        "A = (-I + W) / tau",
        "r(t) = exp(tA) r(0)",
        "```",
        "",
        "Three models are compared:",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| No recurrence | The bump simply decays. This is the non-attractor baseline. |",
        "| Symmetric line attractor | Local reflected excitation with largest recurrent eigenvalue just below one. This gives a slow, stable bump. |",
        "| Balanced E/I line attractor | A high-gain E/I block with matched excitation and inhibition. This is asymptotically stable but can show transient amplification. |",
        "",
        "## Balancing",
        "",
        "Balancing happens in two senses.",
        "",
        "First, spatial balancing corrects the finite-line boundary problem by keeping row sums approximately equal across position. Without this, boundary neurons receive less recurrent drive than middle neurons.",
        "",
        "Second, E/I balancing uses opposing excitatory and inhibitory pathways:",
        "",
        "```text",
        "W_EI = [[ W0, -W0],",
        "        [ W0, -W0]]",
        "B = [I; 0]",
        "C = [I, 0]",
        "```",
        "",
        "This matrix has strong internal gain but cancels asymptotically because excitation and inhibition are matched. It is non-normal: eigenvalues alone do not describe the short-term response.",
        "",
        "## Transient Growth But Asymptotically Stable",
        "",
        "A continuous-time linear system is asymptotically stable if all eigenvalues of `A` have negative real part. The symmetric line model is stable because the largest eigenvalue of `W` is set below one. The balanced E/I model is also stable, even with high internal gain, because its recurrent block is arranged so the long-term eigenvalues remain below the stability boundary.",
        "",
        "Transient growth is still possible when `A` is non-normal. In that case, eigenvectors are not orthogonal, so activity can temporarily grow before eventually decaying. This is useful for a readout because it can amplify a weak distance bump without requiring unstable persistent activity.",
        "",
        "![Stability and transient growth](../outputs/line_attractor_analysis/figures/stability_transients.png)",
        "",
        "| Model | max Re(A) | peak state gain | peak readout gain | final state gain | final readout gain |",
        "|---|---:|---:|---:|---:|---:|",
        *stability_rows,
        "",
        "## Bump Dynamics",
        "",
        "![Readout snapshots](../outputs/line_attractor_analysis/figures/readout_snapshots.png)",
        "",
        "The symmetric line attractor keeps the bump shape stable for longer than no recurrence. The balanced E/I version can transiently amplify the readout while still decaying eventually.",
        "",
        "## Fisher Information Through Time",
        "",
        "![Fisher through time](../outputs/line_attractor_analysis/figures/fisher_through_time.png)",
        "",
        "The useful target is not simply high Fisher information at one point. For a distance map, the more important property is high and flat Fisher information across the full line, including near the boundaries. This is why boundary correction is central to converting the ring model into a line model.",
        "",
        "## Noisy Decoding Test",
        "",
        "A simple centre-of-mass decoder was applied to noisy readout activity. This is not yet coupled to the distance pathway; it only tests whether the attractor dynamics preserve a decodable line-position bump.",
        "",
        "![Decoding error](../outputs/line_attractor_analysis/figures/decoding_error.png)",
        "",
        "| Model | Mean absolute error at 60 ms |",
        "|---|---:|",
        *error_rows,
        "",
        "## Interpretation",
        "",
        "- A ring attractor becomes a line attractor by replacing circular distance with finite-line distance.",
        "- The line requires explicit boundary correction; otherwise the endpoints have weaker recurrent input and lower Fisher information.",
        "- Fisher information is the correct analysis tool because it measures how well nearby distances can be distinguished under noise.",
        "- The safest SC readout candidate is an asymptotically stable line attractor, not a perfectly neutral one, because stability prevents runaway drift.",
        "- Balanced E/I circuitry can provide transient gain while remaining stable, which is attractive for a neuromorphic readout receiving brief IC/AC input.",
        "- This report analyses the readout model only. Integration with the distance pathway should come later by feeding the AC population into the line-attractor input `u(t)`.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| represented length | `{params.length_m:.1f} m` |",
        f"| neurons per population | `{params.num_neurons}` |",
        f"| tau | `{params.tau_s * 1_000:.1f} ms` |",
        f"| dt | `{params.dt_s * 1_000:.1f} ms` |",
        f"| simulation time | `{params.t_end_s * 1_000:.1f} ms` |",
        f"| tuning sigma | `{params.tuning_sigma_m:.3f} m` |",
        f"| recurrent sigma | `{params.recurrent_sigma_m:.3f} m` |",
        f"| symmetric alpha | `{params.symmetric_alpha:.3f}` |",
        f"| balanced alpha prime | `{params.balanced_alpha_prime:.3f}` |",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the standalone line-attractor analysis."""
    start = time.perf_counter()
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    params = LineAttractorParams()
    models = build_models(params)

    artifacts = {
        "tuning_and_fisher": plot_tuning_and_fisher(params, FIGURE_DIR / "tuning_and_fisher.png"),
        "boundary_weights": plot_boundary_weights(params, FIGURE_DIR / "boundary_weights.png"),
        "readout_snapshots": plot_model_snapshots(params, models, FIGURE_DIR / "readout_snapshots.png"),
    }
    stability_path, stability_summary = plot_stability_and_transients(
        params,
        models,
        FIGURE_DIR / "stability_transients.png",
    )
    artifacts["stability_transients"] = stability_path
    artifacts["fisher_through_time"] = plot_fisher_through_time(
        params,
        models,
        FIGURE_DIR / "fisher_through_time.png",
    )
    decoding_path, final_decoding_errors = plot_decoding_error(
        params,
        models,
        FIGURE_DIR / "decoding_error.png",
    )
    artifacts["decoding_error"] = decoding_path

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "line_attractor_analysis",
        "elapsed_seconds": elapsed_s,
        "params": params.__dict__,
        "stability_summary": stability_summary,
        "final_decoding_errors_m": final_decoding_errors,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(params, artifacts, stability_summary, final_decoding_errors, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
