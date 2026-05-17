from __future__ import annotations

"""Analytical finite-line input theory for the SC line attractor.

This experiment is deliberately separate from the full distance pathway. It
tests how the original ring-model input theory transfers to a finite line:

* circulant ring input matrices become banded finite-line matrices;
* edge effects are handled by no correction, amplitude compensation, or
  reflected-boundary kernels;
* one-population and balanced E/I two-block inputs are compared;
* Fisher information is computed through the linear dynamics rather than by
  fitting labels.
"""

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, svdvals

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "finite_line_input_theory"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "finite_line_input_theory.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"


@dataclass(frozen=True)
class TheoryParams:
    """Configuration for the finite-line theory experiment.

    Attributes:
        length_m: Represented line length.
        num_neurons: Number of neurons in one population.
        tau_s: Rate time constant.
        readout_time_s: Main readout time used for ranking.
        window_time_s: End time for window-integrated Fisher information.
        stimulus_sigma_m: Width of synthetic sensory population code.
        fisher_noise_sigma: Independent Gaussian readout noise standard
            deviation assumed by the Fisher information calculation.
        one_pop_alpha: Spectral scale for one-population recurrent model.
        balanced_alpha_prime: Spectral scale for balanced E/I model.
    """

    length_m: float = 10.0
    num_neurons: int = 96
    tau_s: float = 0.020
    readout_time_s: float = 0.060
    early_time_s: float = 0.005
    window_time_s: float = 0.060
    stimulus_sigma_m: float = 0.42
    fisher_noise_sigma: float = 0.05
    one_pop_alpha: float = 0.985
    balanced_alpha_prime: float = 4.0
    baseline_rate_hz: float = 5.0
    state_rate_scale_hz: float = 20.0


@dataclass(frozen=True)
class InputSpec:
    """Input matrix specification.

    Attributes:
        family: Matrix family name.
        width_bins: Width in distance bins. `0` means identity.
    """

    family: str
    width_bins: float


@dataclass(frozen=True)
class Candidate:
    """One finite-line input/readout candidate.

    Attributes:
        name: Human-readable candidate name.
        input_spec: Input matrix specification.
        block_kind: `one_population`, `balanced_e_only`, or
            `balanced_opponent`.
        recurrent_width_bins: Recurrent kernel width in bins.
        beta: Opponent input gain. Only meaningful for `balanced_opponent`.
        W: Recurrent state matrix.
        B: Input matrix from sensory population to state.
        C: Readout matrix from state to decoded activity.
        M: Single-population finite-line input matrix used to build `B`.
    """

    name: str
    input_spec: InputSpec
    block_kind: str
    recurrent_width_bins: float
    beta: float
    W: np.ndarray
    B: np.ndarray
    C: np.ndarray
    M: np.ndarray


def positions(params: TheoryParams) -> np.ndarray:
    """Return preferred positions of the finite-line population."""
    return np.linspace(0.0, params.length_m, params.num_neurons)


def gaussian(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate an unnormalised Gaussian."""
    return np.exp(-0.5 * (distance / sigma) ** 2)


def reflected_gaussian(target: np.ndarray, source: np.ndarray, length_m: float, sigma_m: float) -> np.ndarray:
    """Evaluate a reflected Gaussian finite-line kernel."""
    return (
        gaussian(target - source, sigma_m)
        + gaussian(target + source, sigma_m)
        + gaussian(target - (2.0 * length_m - source), sigma_m)
    )


def population_code(params: TheoryParams, stimulus_m: float) -> np.ndarray:
    """Return reflected Gaussian sensory code for a position."""
    x = positions(params)
    return reflected_gaussian(x, stimulus_m, params.length_m, params.stimulus_sigma_m)


def population_derivative(params: TheoryParams, stimulus_m: float) -> np.ndarray:
    """Return derivative of reflected sensory code with respect to position.

    Args:
        params: Theory parameters.
        stimulus_m: Position at which the derivative is evaluated.

    Returns:
        Vector `dh/dd` over sensory bins.
    """
    x = positions(params)
    length = params.length_m
    sigma2 = params.stimulus_sigma_m**2
    z1 = x - stimulus_m
    z2 = x + stimulus_m
    z3 = x - (2.0 * length - stimulus_m)
    return (
        (z1 / sigma2) * gaussian(z1, params.stimulus_sigma_m)
        - (z2 / sigma2) * gaussian(z2, params.stimulus_sigma_m)
        - (z3 / sigma2) * gaussian(z3, params.stimulus_sigma_m)
    )


def normalise_frobenius(matrix: np.ndarray, target_norm: float) -> np.ndarray:
    """Normalise a matrix to a fixed Frobenius norm."""
    norm = float(np.linalg.norm(matrix))
    if norm <= 1e-12:
        raise ValueError("Cannot normalise near-zero matrix.")
    return matrix * (target_norm / norm)


def build_input_matrix(params: TheoryParams, spec: InputSpec) -> np.ndarray:
    """Build a structured finite-line input matrix.

    Args:
        params: Theory parameters.
        spec: Matrix family and width.

    Returns:
        Matrix `[target_neuron, source_bin]` with matched Frobenius norm.
    """
    m = params.num_neurons
    identity = np.eye(m)
    target_norm = float(np.linalg.norm(identity))
    if spec.family == "identity":
        return identity.copy()

    x = positions(params)
    bin_width = float(np.mean(np.diff(x)))
    sigma_m = spec.width_bins * bin_width
    target = x[:, None]
    source = x[None, :]
    if spec.family == "toeplitz_raw":
        matrix = gaussian(target - source, sigma_m)
    elif spec.family == "toeplitz_amplitude":
        matrix = gaussian(target - source, sigma_m)
        matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
    elif spec.family == "reflected":
        matrix = reflected_gaussian(target, source, params.length_m, sigma_m)
        matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
    else:
        raise ValueError(f"Unknown input family: {spec.family}")
    return normalise_frobenius(matrix, target_norm)


def recurrent_kernel(params: TheoryParams, width_bins: float) -> np.ndarray:
    """Return reflected row-normalised recurrent kernel."""
    x = positions(params)
    bin_width = float(np.mean(np.diff(x)))
    sigma_m = width_bins * bin_width
    matrix = reflected_gaussian(x[:, None], x[None, :], params.length_m, sigma_m)
    return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)


def rescale_to_alpha(matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Rescale matrix so largest real eigenvalue equals `alpha`."""
    current = float(np.max(np.real(np.linalg.eigvals(matrix))))
    if abs(current) <= 1e-12:
        raise ValueError("Cannot rescale matrix with near-zero spectral abscissa.")
    return matrix * (alpha / current)


def balanced_matrix(local: np.ndarray, alpha_prime: float) -> np.ndarray:
    """Build balanced E/I recurrent matrix from local recurrent kernel."""
    high_gain = rescale_to_alpha(local, alpha_prime)
    return np.block([[high_gain, -high_gain], [high_gain, -high_gain]])


def optimal_beta_from_quadratic(U: float, V: float, Z: float) -> float:
    """Return non-negative beta maximising `(U + 2 beta V + beta^2 Z)/(1 + beta^2)`.

    Args:
        U: Constant quadratic coefficient.
        V: Cross coefficient.
        Z: Quadratic coefficient.

    Returns:
        Non-negative maximising beta. If both roots are invalid, returns `0`.
    """
    if abs(V) <= 1e-12:
        return 0.0 if U >= Z else 10.0
    discriminant = (U - Z) ** 2 + 4.0 * V**2
    roots = [
        (-(U - Z) + np.sqrt(discriminant)) / (2.0 * V),
        (-(U - Z) - np.sqrt(discriminant)) / (2.0 * V),
    ]
    valid = [float(root) for root in roots if np.isfinite(root) and root >= 0.0]
    if not valid:
        return 0.0

    def score(beta: float) -> float:
        return (U + 2.0 * beta * V + beta**2 * Z) / (1.0 + beta**2)

    return max(valid, key=score)


def analytic_beta(
    params: TheoryParams,
    M: np.ndarray,
    K: np.ndarray,
    stimulus_grid: np.ndarray,
    *,
    objective: str,
) -> float:
    """Compute analytic opponent beta for a fixed input matrix.

    The balanced E/I readout derivative for `B = [M; -beta M]` is:

    `exp(-t/tau) / sqrt(1 + beta^2) * (A + a(1 + beta) G)`,

    with `A = M h'(d)`, `G = K M h'(d)`, and `a = t/tau`.

    Args:
        params: Theory parameters.
        M: Finite-line input matrix.
        K: High-gain local recurrent kernel.
        stimulus_grid: Positions at which FI is evaluated.
        objective: `final` or `window`.

    Returns:
        Analytic beta that maximises final-time or window-integrated FI.
    """
    if objective == "final":
        time_grid = np.array([params.readout_time_s])
        weights = np.array([np.exp(-2.0 * params.readout_time_s / params.tau_s)])
    elif objective == "window":
        time_grid = np.linspace(0.0, params.window_time_s, 121)
        weights = np.exp(-2.0 * time_grid / params.tau_s)
    else:
        raise ValueError("objective must be 'final' or 'window'.")

    U = 0.0
    V = 0.0
    Z = 0.0
    for stimulus_m in stimulus_grid:
        h_prime = population_derivative(params, float(stimulus_m))
        A_vec = M @ h_prime
        G_vec = K @ A_vec
        for time_s, weight in zip(time_grid, weights):
            a = time_s / params.tau_s
            P = A_vec + a * G_vec
            Q = a * G_vec
            U += weight * float(np.dot(P, P))
            V += weight * float(np.dot(P, Q))
            Z += weight * float(np.dot(Q, Q))
    return optimal_beta_from_quadratic(U, V, Z)


def build_candidate(
    params: TheoryParams,
    spec: InputSpec,
    recurrent_width_bins: float,
    block_kind: str,
    beta: float,
) -> Candidate:
    """Build one finite-line candidate."""
    m = params.num_neurons
    M = build_input_matrix(params, spec)
    local = recurrent_kernel(params, recurrent_width_bins)
    if block_kind == "one_population":
        W = rescale_to_alpha(local, params.one_pop_alpha)
        B = M
        C = np.eye(m)
        beta_used = 0.0
    elif block_kind == "balanced_e_only":
        W = balanced_matrix(local, params.balanced_alpha_prime)
        B = np.vstack([M, np.zeros_like(M)])
        C = np.hstack([np.eye(m), np.zeros((m, m))])
        beta_used = 0.0
    elif block_kind == "balanced_opponent":
        W = balanced_matrix(local, params.balanced_alpha_prime)
        scale = np.sqrt(1.0 + beta**2)
        B = np.vstack([M, -beta * M]) / scale
        C = np.hstack([np.eye(m), np.zeros((m, m))])
        beta_used = beta
    else:
        raise ValueError(f"Unknown block kind: {block_kind}")
    name = f"{block_kind} | {spec.family} w={spec.width_bins:g} | R={recurrent_width_bins:g} | beta={beta_used:.3g}"
    return Candidate(
        name=name,
        input_spec=spec,
        block_kind=block_kind,
        recurrent_width_bins=recurrent_width_bins,
        beta=beta_used,
        W=W,
        B=B,
        C=C,
        M=M,
    )


def transition_operator(params: TheoryParams, candidate: Candidate, time_s: float) -> np.ndarray:
    """Return `C exp(A t) B` for one candidate."""
    state_size = candidate.W.shape[0]
    A = (-np.eye(state_size) + candidate.W) / params.tau_s
    return candidate.C @ expm(A * time_s) @ candidate.B


def decode_center_of_mass(readout: np.ndarray, x: np.ndarray) -> float:
    """Decode a finite-line position from non-negative population activity."""
    positive = np.maximum(readout, 0.0)
    total = float(np.sum(positive))
    if total <= 1e-12:
        return float(0.5 * (x[0] + x[-1]))
    return float(np.sum(positive * x) / total)


def evaluate_candidate(
    params: TheoryParams,
    candidate: Candidate,
    stimulus_grid: np.ndarray,
) -> dict[str, float | str]:
    """Evaluate FI, Cramer-Rao bound, bias, and stability for a candidate."""
    x = positions(params)
    op_early = transition_operator(params, candidate, params.early_time_s)
    op_final = transition_operator(params, candidate, params.readout_time_s)
    final_fi = []
    early_fi = []
    decoded = []
    for stimulus_m in stimulus_grid:
        h = population_code(params, float(stimulus_m))
        h_prime = population_derivative(params, float(stimulus_m))
        early_derivative = op_early @ h_prime
        final_derivative = op_final @ h_prime
        early_fi.append(float(np.dot(early_derivative, early_derivative) / params.fisher_noise_sigma**2))
        final_fi.append(float(np.dot(final_derivative, final_derivative) / params.fisher_noise_sigma**2))
        decoded.append(decode_center_of_mass(op_final @ h, x))
    early_fi_arr = np.maximum(np.array(early_fi), 1e-12)
    final_fi_arr = np.maximum(np.array(final_fi), 1e-12)
    decoded_arr = np.array(decoded)
    error = decoded_arr - stimulus_grid
    state_matrix = (-np.eye(candidate.W.shape[0]) + candidate.W) / params.tau_s
    max_real_eig = float(np.max(np.real(np.linalg.eigvals(state_matrix))))
    edge_mask = (stimulus_grid < 1.0) | (stimulus_grid > params.length_m - 1.0)
    return {
        "name": candidate.name,
        "family": candidate.input_spec.family,
        "input_width_bins": float(candidate.input_spec.width_bins),
        "block_kind": candidate.block_kind,
        "recurrent_width_bins": float(candidate.recurrent_width_bins),
        "beta": float(candidate.beta),
        "early_fi_mean": float(np.mean(early_fi_arr)),
        "final_fi_mean": float(np.mean(final_fi_arr)),
        "final_fi_min": float(np.min(final_fi_arr)),
        "final_fi_uniformity": float(np.min(final_fi_arr) / np.mean(final_fi_arr)),
        "final_crb_rmse_m": float(np.sqrt(np.mean(1.0 / final_fi_arr))),
        "early_crb_rmse_m": float(np.sqrt(np.mean(1.0 / early_fi_arr))),
        "mean_abs_bias_m": float(np.mean(np.abs(error))),
        "edge_mean_abs_bias_m": float(np.mean(np.abs(error[edge_mask]))),
        "max_abs_bias_m": float(np.max(np.abs(error))),
        "max_real_eig": max_real_eig,
    }


def fisher_curve(params: TheoryParams, candidate: Candidate, stimulus_grid: np.ndarray) -> np.ndarray:
    """Return final-time FI curve for one candidate."""
    op_final = transition_operator(params, candidate, params.readout_time_s)
    values = []
    for stimulus_m in stimulus_grid:
        h_prime = population_derivative(params, float(stimulus_m))
        derivative = op_final @ h_prime
        values.append(float(np.dot(derivative, derivative) / params.fisher_noise_sigma**2))
    return np.array(values)


def response_snapshots(
    params: TheoryParams,
    candidate: Candidate,
    stimulus_m: float,
    times_s: list[float],
) -> dict[float, np.ndarray]:
    """Return readout population snapshots for one stimulus."""
    h = population_code(params, stimulus_m)
    return {
        time_s: transition_operator(params, candidate, time_s) @ h
        for time_s in times_s
    }


def simulate_state_history(
    params: TheoryParams,
    candidate: Candidate,
    stimulus: np.ndarray,
    *,
    alpha_cap_hz: float | None = None,
    t_end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate candidate state dynamics, optionally with firing-rate clipping.

    Args:
        params: Theory parameters.
        candidate: Candidate model.
        stimulus: Sensory population code.
        alpha_cap_hz: Absolute firing-rate cap. If `None`, no cap is applied.
        t_end_s: Optional simulation duration.

    Returns:
        Tuple `(times_s, state_history, readout_history)`.
    """
    t_end = params.window_time_s if t_end_s is None else t_end_s
    dt = 0.001
    num_steps = int(round(t_end / dt))
    times = np.arange(num_steps + 1) * dt
    state = params.state_rate_scale_hz * (candidate.B @ stimulus)
    if alpha_cap_hz is not None:
        lower = -params.baseline_rate_hz
        upper = alpha_cap_hz - params.baseline_rate_hz
        state = np.clip(state, lower, upper)
    state_history = np.empty((num_steps + 1, state.size), dtype=np.float64)
    readout_history = np.empty((num_steps + 1, params.num_neurons), dtype=np.float64)
    state_history[0] = state
    readout_history[0] = candidate.C @ state
    for step in range(1, num_steps + 1):
        state = state + dt / params.tau_s * (-state + candidate.W @ state)
        if alpha_cap_hz is not None:
            state = np.clip(state, lower, upper)
        state_history[step] = state
        readout_history[step] = candidate.C @ state
    return times, state_history, readout_history


def candidate_suite(params: TheoryParams) -> tuple[list[Candidate], dict[str, float]]:
    """Build deterministic structured candidates without label fitting."""
    input_specs = [InputSpec("identity", 0.0)]
    for family in ["toeplitz_raw", "toeplitz_amplitude", "reflected"]:
        for width in [3.0, 6.0, 9.0, 12.0]:
            input_specs.append(InputSpec(family, width))
    recurrent_widths = [4.0, 8.0, 12.0]
    beta_grid = np.linspace(0.0, 2.0, 41)
    beta_stimulus_grid = np.linspace(0.25, params.length_m - 0.25, 61)

    candidates: list[Candidate] = []
    beta_summary: dict[str, float] = {}
    for spec in input_specs:
        for recurrent_width in recurrent_widths:
            local = recurrent_kernel(params, recurrent_width)
            high_gain = rescale_to_alpha(local, params.balanced_alpha_prime)
            M = build_input_matrix(params, spec)
            beta_final = analytic_beta(params, M, high_gain, beta_stimulus_grid, objective="final")
            beta_window = analytic_beta(params, M, high_gain, beta_stimulus_grid, objective="window")
            beta_summary[f"{spec.family}_{spec.width_bins:g}_R{recurrent_width:g}_final"] = beta_final
            beta_summary[f"{spec.family}_{spec.width_bins:g}_R{recurrent_width:g}_window"] = beta_window
            for block_kind in ["one_population", "balanced_e_only"]:
                candidates.append(build_candidate(params, spec, recurrent_width, block_kind, beta=0.0))
            for beta in [1.0, beta_final, beta_window]:
                candidates.append(build_candidate(params, spec, recurrent_width, "balanced_opponent", beta=beta))

    # A beta scan for the most interpretable reflected matrix is included for
    # plotting only, not for choosing an unstructured fitted parameter.
    reference_spec = InputSpec("reflected", 6.0)
    reference_width = 8.0
    for beta in beta_grid:
        candidates.append(build_candidate(params, reference_spec, reference_width, "balanced_opponent", beta=float(beta)))
    return candidates, beta_summary


def compact_rows(rows: list[dict[str, float | str]], *, top_n: int = 12) -> list[dict[str, float | str]]:
    """Return top candidates sorted by final CRB with duplicate beta scans removed."""
    unique = [
        row
        for row in rows
        if not (
            row["family"] == "reflected"
            and row["input_width_bins"] == 6.0
            and row["recurrent_width_bins"] == 8.0
            and row["block_kind"] == "balanced_opponent"
            and row["beta"] not in {0.0, 1.0}
        )
    ]
    return sorted(unique, key=lambda item: (item["final_crb_rmse_m"], item["mean_abs_bias_m"]))[:top_n]


def plot_input_families(params: TheoryParams, path: Path) -> str:
    """Plot representative finite-line input matrices."""
    specs = [
        InputSpec("identity", 0.0),
        InputSpec("toeplitz_raw", 6.0),
        InputSpec("toeplitz_amplitude", 6.0),
        InputSpec("reflected", 6.0),
    ]
    x = positions(params)
    extent = [x[0], x[-1], x[-1], x[0]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for ax, spec in zip(axes.flat, specs):
        matrix = build_input_matrix(params, spec)
        im = ax.imshow(matrix, aspect="auto", extent=extent, cmap="viridis")
        ax.set_title(f"{spec.family}, width {spec.width_bins:g} bins")
        ax.set_xlabel("source distance bin (m)")
        ax.set_ylabel("target SC neuron (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_fisher_curves(
    params: TheoryParams,
    selected: list[Candidate],
    stimulus_grid: np.ndarray,
    path: Path,
) -> str:
    """Plot final-time Fisher information curves for selected candidates."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for candidate in selected:
        fi = fisher_curve(params, candidate, stimulus_grid)
        ax.plot(stimulus_grid, fi / np.mean(fi), linewidth=2.0, label=candidate.name[:72])
    ax.set_xlabel("true distance on finite line (m)")
    ax.set_ylabel("FI / mean FI")
    ax.set_title("Final-time Fisher information uniformity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    return save_figure(fig, path)


def plot_beta_scan(rows: list[dict[str, float | str]], path: Path) -> str:
    """Plot the reflected-width beta scan."""
    scan = [
        row
        for row in rows
        if row["family"] == "reflected"
        and row["input_width_bins"] == 6.0
        and row["recurrent_width_bins"] == 8.0
        and row["block_kind"] == "balanced_opponent"
    ]
    scan = sorted(scan, key=lambda row: row["beta"])
    beta = np.array([float(row["beta"]) for row in scan])
    crb = np.array([float(row["final_crb_rmse_m"]) for row in scan]) * 100.0
    bias = np.array([float(row["mean_abs_bias_m"]) for row in scan]) * 100.0
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(beta, crb, marker="o", linewidth=2.0, label="CRB RMSE")
    ax.plot(beta, bias, marker="s", linewidth=2.0, label="COM bias")
    ax.set_xlabel(r"opponent input $\beta$")
    ax.set_ylabel("distance error proxy (cm)")
    ax.set_title("Analytical beta sensitivity, reflected input width 6")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_width_sweep(rows: list[dict[str, float | str]], path: Path) -> str:
    """Plot best final CRB by input family and width for two-block candidates."""
    families = ["toeplitz_raw", "toeplitz_amplitude", "reflected"]
    widths = [3.0, 6.0, 9.0, 12.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for family in families:
        values = []
        for width in widths:
            matching = [
                row
                for row in rows
                if row["family"] == family
                and row["input_width_bins"] == width
                and row["block_kind"] == "balanced_opponent"
                and row["beta"] <= 2.0
            ]
            best = min(matching, key=lambda row: row["final_crb_rmse_m"])
            values.append(float(best["final_crb_rmse_m"]) * 100.0)
        ax.plot(widths, values, marker="o", linewidth=2.0, label=family)
    ax.set_xlabel("input width (bins)")
    ax.set_ylabel("best final CRB RMSE (cm)")
    ax.set_title("Finite-line input width sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_block_matrices(params: TheoryParams, candidates: list[Candidate], path: Path) -> str:
    """Plot B blocks for selected one-block and two-block candidates."""
    x = positions(params)
    extent = [x[0], x[-1], x[-1], x[0]]
    fig, axes = plt.subplots(len(candidates), 2, figsize=(10.5, 3.0 * len(candidates)))
    if len(candidates) == 1:
        axes = np.array([axes])
    for row_idx, candidate in enumerate(candidates):
        if candidate.B.shape[0] == params.num_neurons:
            exc = candidate.B
            inh = np.zeros_like(exc)
        else:
            exc = candidate.B[: params.num_neurons]
            inh = candidate.B[params.num_neurons :]
        max_abs = max(float(np.max(np.abs(exc))), float(np.max(np.abs(inh))), 1e-12)
        for col_idx, (block, title) in enumerate([(exc, "B_E"), (inh, "B_I")]):
            im = axes[row_idx, col_idx].imshow(
                block,
                aspect="auto",
                extent=extent,
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            axes[row_idx, col_idx].set_title(f"{candidate.block_kind}\n{title}")
            axes[row_idx, col_idx].set_xlabel("source distance bin (m)")
            axes[row_idx, col_idx].set_ylabel("target neuron (m)")
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_response_snapshots(params: TheoryParams, selected: list[Candidate], path: Path) -> str:
    """Plot synthetic bump snapshots for selected candidates."""
    x = positions(params)
    stimulus_m = 4.2
    times = [0.0, params.early_time_s, 0.020, params.readout_time_s]
    fig, axes = plt.subplots(len(selected), len(times), figsize=(14.0, 3.0 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = np.array([axes])
    for row_idx, candidate in enumerate(selected):
        snapshots = response_snapshots(params, candidate, stimulus_m, times)
        for col_idx, time_s in enumerate(times):
            activity = snapshots[time_s]
            activity = activity / max(float(np.max(np.abs(activity))), 1e-12)
            ax = axes[row_idx, col_idx]
            ax.plot(x, activity, linewidth=2.0)
            ax.axvline(stimulus_m, color="#111827", linestyle=":", linewidth=1.2)
            if row_idx == 0:
                ax.set_title(f"{time_s * 1_000:.0f} ms")
            if col_idx == 0:
                ax.set_ylabel(candidate.block_kind)
            ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("distance represented by readout neuron (m)")
    return save_figure(fig, path)


def plot_chosen_matrices(params: TheoryParams, candidate: Candidate, path: Path) -> str:
    """Plot selected candidate input and recurrence matrices."""
    x = positions(params)
    extent = [x[0], x[-1], x[-1], x[0]]
    if candidate.B.shape[0] == params.num_neurons:
        b_exc = candidate.B
        b_inh = np.zeros_like(candidate.B)
    else:
        b_exc = candidate.B[: params.num_neurons]
        b_inh = candidate.B[params.num_neurons :]
    matrices = [
        (candidate.M, "Chosen finite-line input M", "viridis", extent),
        (b_exc, "Excitatory input block B_E", "coolwarm", extent),
        (b_inh, "Inhibitory input block B_I", "coolwarm", extent),
        (candidate.W, "Full recurrent matrix W", "coolwarm", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    for ax, (matrix, title, cmap, matrix_extent) in zip(axes.flat, matrices):
        if cmap == "coolwarm":
            max_abs = max(float(np.max(np.abs(matrix))), 1e-12)
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs, extent=matrix_extent)
        else:
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, extent=matrix_extent)
        ax.set_title(title)
        if matrix_extent is None:
            ax.set_xlabel("source state index")
            ax.set_ylabel("target state index")
        else:
            ax.set_xlabel("source distance bin (m)")
            ax.set_ylabel("target neuron (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_figure(fig, path)


def cosine_basis(params: TheoryParams) -> np.ndarray:
    """Return orthonormal finite-line cosine basis vectors."""
    m = params.num_neurons
    index = np.arange(m)
    basis = []
    for mode in range(m):
        vector = np.cos(np.pi * mode * (index + 0.5) / m)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        basis.append(vector)
    return np.stack(basis, axis=0)


def spatial_frequency_gain(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return gain of a matrix on finite-line cosine spatial modes."""
    values = []
    for vector in basis:
        values.append(float(np.linalg.norm(matrix @ vector) / max(float(np.linalg.norm(vector)), 1e-12)))
    return np.array(values)


def plot_spatial_frequency_response(params: TheoryParams, candidate: Candidate, path: Path) -> str:
    """Plot spatial frequency response curves for input matrices."""
    basis = cosine_basis(params)
    modes = np.arange(params.num_neurons)
    comparison_specs = [
        InputSpec("identity", 0.0),
        InputSpec("toeplitz_raw", candidate.input_spec.width_bins),
        InputSpec("toeplitz_amplitude", candidate.input_spec.width_bins),
        InputSpec("reflected", candidate.input_spec.width_bins),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for spec in comparison_specs:
        matrix = build_input_matrix(params, spec)
        gain = spatial_frequency_gain(matrix, basis)
        gain = gain / max(float(gain[0]), 1e-12)
        ax.plot(modes[:40], gain[:40], linewidth=2.0, label=f"{spec.family} w={spec.width_bins:g}")
    ax.set_xlabel("finite-line cosine spatial mode")
    ax.set_ylabel("normalised gain")
    ax.set_title("Input matrix spatial frequency response")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return save_figure(fig, path)


def system_matrix(params: TheoryParams, candidate: Candidate) -> np.ndarray:
    """Return continuous-time system matrix `A=(-I+W)/tau`."""
    return (-np.eye(candidate.W.shape[0]) + candidate.W) / params.tau_s


def plot_recurrent_spectrum(params: TheoryParams, candidate: Candidate, path: Path) -> str:
    """Plot recurrence eigenvalues and singular values."""
    W = candidate.W
    A = system_matrix(params, candidate)
    eig_w = np.linalg.eigvals(W)
    eig_a = np.linalg.eigvals(A)
    singular_w = svdvals(W)
    singular_a = svdvals(A)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].scatter(np.real(eig_w), np.imag(eig_w), s=14, alpha=0.65)
    axes[0].axvline(0.0, color="#111827", linewidth=1.0)
    axes[0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_title("Eigenvalues of W")
    axes[0].set_xlabel("real")
    axes[0].set_ylabel("imaginary")
    axes[1].scatter(np.real(eig_a), np.imag(eig_a), s=14, alpha=0.65, color="#dc2626")
    axes[1].axvline(0.0, color="#111827", linewidth=1.0)
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_title("Eigenvalues of A=(-I+W)/tau")
    axes[1].set_xlabel("real (s^-1)")
    axes[1].set_ylabel("imaginary (s^-1)")
    axes[2].semilogy(singular_w, linewidth=2.0, label="W")
    axes[2].semilogy(singular_a, linewidth=2.0, label="A")
    axes[2].set_title("Singular value spectra")
    axes[2].set_xlabel("index")
    axes[2].set_ylabel("singular value")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_pseudospectrum(params: TheoryParams, candidate: Candidate, path: Path) -> str:
    """Plot epsilon-pseudospectrum proxy for the continuous-time matrix A."""
    A = system_matrix(params, candidate)
    eig_a = np.linalg.eigvals(A)
    real_grid = np.linspace(-140.0, 40.0, 50)
    imag_grid = np.linspace(-100.0, 100.0, 50)
    sigma_min = np.empty((imag_grid.size, real_grid.size), dtype=np.float64)
    identity = np.eye(A.shape[0])
    for row, imag in enumerate(imag_grid):
        for col, real in enumerate(real_grid):
            z = real + 1j * imag
            sigma_min[row, col] = float(svdvals(z * identity - A)[-1])
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    log_sigma = np.log10(np.maximum(sigma_min, 1e-12))
    contour = ax.contourf(real_grid, imag_grid, log_sigma, levels=32, cmap="magma")
    levels = [-3, -2, -1, 0, 1]
    ax.contour(real_grid, imag_grid, log_sigma, levels=levels, colors="white", linewidths=0.8, alpha=0.75)
    ax.scatter(np.real(eig_a), np.imag(eig_a), s=10, color="#38bdf8", label="eigenvalues")
    ax.axvline(0.0, color="#f8fafc", linestyle="--", linewidth=1.0)
    ax.set_title(r"$\epsilon$-pseudospectrum proxy: $\log_{10}\sigma_{min}(zI-A)$")
    ax.set_xlabel("real z (s^-1)")
    ax.set_ylabel("imaginary z (s^-1)")
    ax.legend(frameon=False)
    fig.colorbar(contour, ax=ax, label=r"$\log_{10}\sigma_{min}$")
    return save_figure(fig, path)


def alpha_sweep(params: TheoryParams, spec: InputSpec, recurrent_width_bins: float) -> list[dict[str, float]]:
    """Sweep balanced alpha for the selected finite-line input family."""
    stimulus_grid = np.linspace(0.1, params.length_m - 0.1, 101)
    beta_grid = np.linspace(0.25, params.length_m - 0.25, 61)
    rows = []
    for alpha in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        sweep_params = replace(params, balanced_alpha_prime=float(alpha))
        local = recurrent_kernel(sweep_params, recurrent_width_bins)
        high_gain = rescale_to_alpha(local, float(alpha)) if alpha > 0.0 else np.zeros_like(local)
        matrix = build_input_matrix(sweep_params, spec)
        beta = analytic_beta(sweep_params, matrix, high_gain, beta_grid, objective="final")
        candidate = build_candidate(sweep_params, spec, recurrent_width_bins, "balanced_opponent", beta=beta)
        metrics = evaluate_candidate(sweep_params, candidate, stimulus_grid)
        rows.append(
            {
                "alpha_prime": float(alpha),
                "beta": float(beta),
                "final_crb_rmse_m": float(metrics["final_crb_rmse_m"]),
                "early_crb_rmse_m": float(metrics["early_crb_rmse_m"]),
                "mean_abs_bias_m": float(metrics["mean_abs_bias_m"]),
                "edge_mean_abs_bias_m": float(metrics["edge_mean_abs_bias_m"]),
                "fi_uniformity": float(metrics["final_fi_uniformity"]),
            }
        )
    return rows


def plot_alpha_sweep(rows: list[dict[str, float]], path: Path) -> str:
    """Plot alpha sweep for the selected reflected/opponent family."""
    alpha = np.array([row["alpha_prime"] for row in rows])
    crb = np.array([row["final_crb_rmse_m"] for row in rows]) * 100.0
    early = np.array([row["early_crb_rmse_m"] for row in rows]) * 100.0
    bias = np.array([row["mean_abs_bias_m"] for row in rows]) * 100.0
    beta = np.array([row["beta"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(alpha, crb, marker="o", linewidth=2.0, label="60 ms CRB")
    axes[0].plot(alpha, early, marker="s", linewidth=2.0, label="5 ms CRB")
    axes[0].plot(alpha, bias, marker="^", linewidth=2.0, label="COM bias")
    axes[0].set_xlabel(r"balanced $\alpha'$")
    axes[0].set_ylabel("distance error proxy (cm)")
    axes[0].set_title("Alpha sweep")
    axes[0].legend(frameon=False)
    axes[1].plot(alpha, beta, marker="o", linewidth=2.0, color="#059669")
    axes[1].set_xlabel(r"balanced $\alpha'$")
    axes[1].set_ylabel(r"analytic opponent $\beta$")
    axes[1].set_title("Analytic beta changes with alpha")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, path)


def capped_alpha_sweep(
    params: TheoryParams,
    spec: InputSpec,
    recurrent_width_bins: float,
) -> list[dict[str, float | str]]:
    """Evaluate alpha sweep with optional biophysical firing-rate caps.

    Capping makes the system nonlinear, so the capped FI/CRB values are
    estimated by finite differences around each synthetic stimulus. This is a
    diagnostic of biological limits, not part of the analytic candidate ranking.

    Args:
        params: Theory parameters.
        spec: Input family selected by the analytic sweep.
        recurrent_width_bins: Selected recurrent width.

    Returns:
        Rows for uncapped, 100 Hz capped, and 55 Hz capped alpha sweeps.
    """
    x = positions(params)
    stimulus_grid = np.linspace(0.2, params.length_m - 0.2, 45)
    delta = 0.01 * params.length_m
    alpha_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    cap_scenarios = [("uncapped", None), ("100 Hz cap", 100.0), ("55 Hz cap", 55.0)]
    beta_grid = np.linspace(0.25, params.length_m - 0.25, 61)
    rows: list[dict[str, float | str]] = []
    final_index = int(round(params.readout_time_s / 0.001))
    for alpha in alpha_values:
        sweep_params = replace(params, balanced_alpha_prime=float(alpha))
        local = recurrent_kernel(sweep_params, recurrent_width_bins)
        high_gain = rescale_to_alpha(local, float(alpha))
        matrix = build_input_matrix(sweep_params, spec)
        beta = analytic_beta(sweep_params, matrix, high_gain, beta_grid, objective="final")
        candidate = build_candidate(sweep_params, spec, recurrent_width_bins, "balanced_opponent", beta=beta)
        for cap_label, cap_hz in cap_scenarios:
            fi_values = []
            decoded_errors = []
            peak_rates = []
            for stimulus_m in stimulus_grid:
                low = max(0.0, float(stimulus_m) - delta)
                high = min(params.length_m, float(stimulus_m) + delta)
                stim = population_code(sweep_params, float(stimulus_m))
                stim_low = population_code(sweep_params, low)
                stim_high = population_code(sweep_params, high)
                _, state, readout = simulate_state_history(sweep_params, candidate, stim, alpha_cap_hz=cap_hz)
                _, _, read_low = simulate_state_history(sweep_params, candidate, stim_low, alpha_cap_hz=cap_hz)
                _, _, read_high = simulate_state_history(sweep_params, candidate, stim_high, alpha_cap_hz=cap_hz)
                derivative = (read_high[final_index] - read_low[final_index]) / max(high - low, 1e-12)
                fi_values.append(float(np.dot(derivative, derivative) / sweep_params.fisher_noise_sigma**2))
                decoded = decode_center_of_mass(readout[final_index], x)
                decoded_errors.append(abs(decoded - stimulus_m))
                peak_rates.append(float(np.max(state) + sweep_params.baseline_rate_hz))
            fi_arr = np.maximum(np.array(fi_values), 1e-12)
            rows.append(
                {
                    "alpha_prime": float(alpha),
                    "cap": cap_label,
                    "cap_hz": float(cap_hz) if cap_hz is not None else -1.0,
                    "beta": float(beta),
                    "final_crb_rmse_m": float(np.sqrt(np.mean(1.0 / fi_arr))),
                    "mean_abs_bias_m": float(np.mean(decoded_errors)),
                    "peak_rate_hz": float(np.max(peak_rates)),
                }
            )
    return rows


def fixed_setup_alpha_sweep(
    params: TheoryParams,
    spec: InputSpec,
    recurrent_width_bins: float,
    beta: float,
) -> list[dict[str, float | str]]:
    """Sweep alpha while keeping every other selected setup parameter fixed.

    Args:
        params: Theory parameters.
        spec: Fixed input matrix family.
        recurrent_width_bins: Fixed recurrent width.
        beta: Fixed opponent input beta from the selected default-alpha setup.

    Returns:
        Rows for uncapped, 100 Hz capped, and 55 Hz capped alpha sweeps.
    """
    x = positions(params)
    stimulus_grid = np.linspace(0.2, params.length_m - 0.2, 45)
    delta = 0.01 * params.length_m
    alpha_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    cap_scenarios = [("uncapped", None), ("100 Hz cap", 100.0), ("55 Hz cap", 55.0)]
    final_index = int(round(params.readout_time_s / 0.001))
    rows: list[dict[str, float | str]] = []
    for alpha in alpha_values:
        sweep_params = replace(params, balanced_alpha_prime=float(alpha))
        candidate = build_candidate(sweep_params, spec, recurrent_width_bins, "balanced_opponent", beta=beta)
        for cap_label, cap_hz in cap_scenarios:
            fi_values = []
            decoded_errors = []
            peak_rates = []
            saturated_fractions = []
            for stimulus_m in stimulus_grid:
                low = max(0.0, float(stimulus_m) - delta)
                high = min(params.length_m, float(stimulus_m) + delta)
                stim = population_code(sweep_params, float(stimulus_m))
                stim_low = population_code(sweep_params, low)
                stim_high = population_code(sweep_params, high)
                _, state, readout = simulate_state_history(sweep_params, candidate, stim, alpha_cap_hz=cap_hz)
                _, _, read_low = simulate_state_history(sweep_params, candidate, stim_low, alpha_cap_hz=cap_hz)
                _, _, read_high = simulate_state_history(sweep_params, candidate, stim_high, alpha_cap_hz=cap_hz)
                derivative = (read_high[final_index] - read_low[final_index]) / max(high - low, 1e-12)
                fi_values.append(float(np.dot(derivative, derivative) / sweep_params.fisher_noise_sigma**2))
                decoded = decode_center_of_mass(readout[final_index], x)
                decoded_errors.append(abs(decoded - stimulus_m))
                peak_rates.append(float(np.max(state) + sweep_params.baseline_rate_hz))
                if cap_hz is None:
                    saturated_fractions.append(0.0)
                else:
                    upper = cap_hz - sweep_params.baseline_rate_hz
                    lower = -sweep_params.baseline_rate_hz
                    saturated = (state >= upper - 1e-9) | (state <= lower + 1e-9)
                    saturated_fractions.append(float(np.mean(saturated)))
            fi_arr = np.maximum(np.array(fi_values), 1e-12)
            rows.append(
                {
                    "alpha_prime": float(alpha),
                    "cap": cap_label,
                    "cap_hz": float(cap_hz) if cap_hz is not None else -1.0,
                    "beta": float(beta),
                    "final_crb_rmse_m": float(np.sqrt(np.mean(1.0 / fi_arr))),
                    "mean_abs_bias_m": float(np.mean(decoded_errors)),
                    "peak_rate_hz": float(np.max(peak_rates)),
                    "saturated_fraction": float(np.mean(saturated_fractions)),
                }
            )
    return rows


def plot_capped_alpha_sweep(rows: list[dict[str, float | str]], path: Path) -> str:
    """Plot alpha sweep under firing-rate caps."""
    labels = ["uncapped", "100 Hz cap", "55 Hz cap"]
    colors = {"uncapped": "#111827", "100 Hz cap": "#f58518", "55 Hz cap": "#4c78a8"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for label in labels:
        subset = [row for row in rows if row["cap"] == label]
        alpha = np.array([float(row["alpha_prime"]) for row in subset])
        crb = np.array([float(row["final_crb_rmse_m"]) for row in subset]) * 100.0
        peak = np.array([float(row["peak_rate_hz"]) for row in subset])
        style = "-" if label == "uncapped" else "--"
        axes[0].plot(alpha, crb, marker="o", linestyle=style, linewidth=2.0, color=colors[label], label=label)
        axes[1].plot(alpha, peak, marker="o", linestyle=style, linewidth=2.0, color=colors[label], label=label)
    for cap_hz, color in [(55.0, colors["55 Hz cap"]), (100.0, colors["100 Hz cap"])]:
        axes[1].axhline(cap_hz, color=color, linestyle=":", linewidth=1.4)
    axes[0].set_xlabel(r"balanced $\alpha'$")
    axes[0].set_ylabel("finite-difference CRB RMSE (cm)")
    axes[0].set_title("Accuracy proxy with firing-rate caps")
    axes[1].set_xlabel(r"balanced $\alpha'$")
    axes[1].set_ylabel("peak absolute neural rate (Hz)")
    axes[1].set_title("Caps bound neural activity")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_fixed_setup_alpha_sweep(rows: list[dict[str, float | str]], path: Path) -> str:
    """Plot fixed-beta fixed-setup alpha sweep."""
    labels = ["uncapped", "100 Hz cap", "55 Hz cap"]
    colors = {"uncapped": "#111827", "100 Hz cap": "#f58518", "55 Hz cap": "#4c78a8"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    for label in labels:
        subset = [row for row in rows if row["cap"] == label]
        alpha = np.array([float(row["alpha_prime"]) for row in subset])
        crb = np.array([float(row["final_crb_rmse_m"]) for row in subset]) * 100.0
        bias = np.array([float(row["mean_abs_bias_m"]) for row in subset]) * 100.0
        saturation = np.array([float(row["saturated_fraction"]) for row in subset]) * 100.0
        style = "-" if label == "uncapped" else "--"
        axes[0].plot(alpha, crb, marker="o", linestyle=style, linewidth=2.0, color=colors[label], label=label)
        axes[1].plot(alpha, bias, marker="o", linestyle=style, linewidth=2.0, color=colors[label], label=label)
        axes[2].plot(alpha, saturation, marker="o", linestyle=style, linewidth=2.0, color=colors[label], label=label)
    axes[0].set_ylabel("finite-difference CRB RMSE (cm)")
    axes[0].set_title("Accuracy proxy")
    axes[1].set_ylabel("mean COM bias (cm)")
    axes[1].set_title("Bias")
    axes[2].set_ylabel("saturated state fraction (%)")
    axes[2].set_title("Saturation")
    for ax in axes:
        ax.set_xlabel(r"balanced $\alpha'$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(r"Fixed setup alpha sweep: input, recurrence width, and $\beta$ held constant")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_rate_traces(params: TheoryParams, spec: InputSpec, recurrent_width_bins: float, path: Path) -> str:
    """Plot maximum firing-rate traces for high-alpha capped/uncapped dynamics."""
    alpha = 12.0
    sweep_params = replace(params, balanced_alpha_prime=alpha)
    beta_grid = np.linspace(0.25, params.length_m - 0.25, 61)
    matrix = build_input_matrix(sweep_params, spec)
    high_gain = rescale_to_alpha(recurrent_kernel(sweep_params, recurrent_width_bins), alpha)
    beta = analytic_beta(sweep_params, matrix, high_gain, beta_grid, objective="final")
    candidate = build_candidate(sweep_params, spec, recurrent_width_bins, "balanced_opponent", beta=beta)
    stimulus = population_code(sweep_params, 4.2)
    scenarios = [("uncapped", None), ("100 Hz cap", 100.0), ("55 Hz cap", 55.0)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = {"uncapped": "#111827", "100 Hz cap": "#f58518", "55 Hz cap": "#4c78a8"}
    for label, cap_hz in scenarios:
        times, state, _ = simulate_state_history(sweep_params, candidate, stimulus, alpha_cap_hz=cap_hz)
        peak_rate = np.max(state, axis=1) + sweep_params.baseline_rate_hz
        style = "-" if label == "uncapped" else "--"
        ax.plot(times * 1000.0, peak_rate, color=colors[label], linestyle=style, linewidth=2.0, label=label)
    ax.axhline(55.0, color=colors["55 Hz cap"], linestyle=":", linewidth=1.4)
    ax.axhline(100.0, color=colors["100 Hz cap"], linestyle=":", linewidth=1.4)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("maximum absolute neural rate (Hz)")
    ax.set_title(r"Rate trace at $\alpha'=12$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def format_table(rows: list[dict[str, float | str]]) -> list[str]:
    """Format report table rows."""
    lines = []
    for row in rows:
        lines.append(
            "| "
            f"{row['block_kind']} | "
            f"{row['family']} | "
            f"`{float(row['input_width_bins']):.0f}` | "
            f"`{float(row['recurrent_width_bins']):.0f}` | "
            f"`{float(row['beta']):.3f}` | "
            f"`{float(row['final_crb_rmse_m']) * 100.0:.3f} cm` | "
            f"`{float(row['early_crb_rmse_m']) * 100.0:.3f} cm` | "
            f"`{float(row['final_fi_uniformity']):.3f}` | "
            f"`{float(row['mean_abs_bias_m']) * 100.0:.3f} cm` | "
            f"`{float(row['edge_mean_abs_bias_m']) * 100.0:.3f} cm` |"
        )
    return lines


def format_alpha_table(rows: list[dict[str, float]]) -> list[str]:
    """Format alpha-sweep table rows."""
    return [
        "| "
        f"`{row['alpha_prime']:.1f}` | "
        f"`{row['beta']:.3f}` | "
        f"`{row['final_crb_rmse_m'] * 100.0:.3f} cm` | "
        f"`{row['early_crb_rmse_m'] * 100.0:.3f} cm` | "
        f"`{row['mean_abs_bias_m'] * 100.0:.3f} cm` | "
        f"`{row['edge_mean_abs_bias_m'] * 100.0:.3f} cm` | "
        f"`{row['fi_uniformity']:.3f}` |"
        for row in rows
    ]


def format_capped_alpha_table(rows: list[dict[str, float | str]]) -> list[str]:
    """Format capped alpha table rows."""
    selected_alphas = {4.0, 8.0, 12.0}
    return [
        "| "
        f"`{row['alpha_prime']:.1f}` | "
        f"{row['cap']} | "
        f"`{row['beta']:.3f}` | "
        f"`{row['final_crb_rmse_m'] * 100.0:.3f} cm` | "
        f"`{row['mean_abs_bias_m'] * 100.0:.3f} cm` | "
        f"`{row['peak_rate_hz']:.1f} Hz` |"
        for row in rows
        if float(row["alpha_prime"]) in selected_alphas
    ]


def format_fixed_alpha_table(rows: list[dict[str, float | str]]) -> list[str]:
    """Format fixed-setup alpha table rows."""
    selected_alphas = {4.0, 8.0, 12.0}
    return [
        "| "
        f"`{row['alpha_prime']:.1f}` | "
        f"{row['cap']} | "
        f"`{row['beta']:.3f}` | "
        f"`{row['final_crb_rmse_m'] * 100.0:.3f} cm` | "
        f"`{row['mean_abs_bias_m'] * 100.0:.3f} cm` | "
        f"`{row['peak_rate_hz']:.1f} Hz` | "
        f"`{row['saturated_fraction'] * 100.0:.1f}%` |"
        for row in rows
        if float(row["alpha_prime"]) in selected_alphas
    ]


def write_report(
    params: TheoryParams,
    rows: list[dict[str, float | str]],
    top_rows: list[dict[str, float | str]],
    alpha_rows: list[dict[str, float]],
    capped_alpha_rows: list[dict[str, float | str]],
    fixed_alpha_rows: list[dict[str, float | str]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the analytical finite-line theory report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = top_rows[0]
    table_rows = format_table(top_rows)
    alpha_table_rows = format_alpha_table(alpha_rows)
    capped_alpha_table_rows = format_capped_alpha_table(capped_alpha_rows)
    fixed_alpha_table_rows = format_fixed_alpha_table(fixed_alpha_rows)
    best_alpha = min(alpha_rows, key=lambda row: row["final_crb_rmse_m"])
    best_fixed_uncapped = min(
        (row for row in fixed_alpha_rows if row["cap"] == "uncapped"),
        key=lambda row: float(row["final_crb_rmse_m"]),
    )
    lines = [
        "# Finite-Line Input Theory For The SC Line Attractor",
        "",
        "This report is a standalone analytical study. It does not use the current distance pathway outputs and does not plug a new SC back into the model. The aim is to transfer the original ring-model input theory to a finite line in a controlled way.",
        "",
        "## Problem",
        "",
        "The ring model used circulant matrices. A circulant matrix is natural on a ring because every neuron has the same neighbourhood, only shifted. A distance line has boundaries, so a direct finite-line replacement must decide what happens near `0 m` and `L`.",
        "",
        "The finite-line state is either one population:",
        "",
        "$$",
        "\\tau \\dot r = -r + W r, \\qquad y = C r",
        "$$",
        "",
        "or a balanced E/I two-block state:",
        "",
        "$$",
        "r = \\begin{bmatrix} r_E \\\\ r_I \\end{bmatrix}, \\qquad",
        "W_{EI} = \\begin{bmatrix} K & -K \\\\ K & -K \\end{bmatrix}, \\qquad",
        "y = \\begin{bmatrix} I & 0 \\end{bmatrix} r.",
        "$$",
        "",
        "The sensory input is a synthetic reflected Gaussian population code $h(d)$, not the real AC map:",
        "",
        "$$",
        "h_i(d) = \\exp\\left[-\\frac{(x_i-d)^2}{2\\sigma_h^2}\\right]",
        "+ \\exp\\left[-\\frac{(x_i+d)^2}{2\\sigma_h^2}\\right]",
        "+ \\exp\\left[-\\frac{(x_i-(2L-d))^2}{2\\sigma_h^2}\\right].",
        "$$",
        "",
        "## Fisher Information Theory",
        "",
        "Fisher information measures local discriminability. If a small distance change produces a large reliable change in neural activity, distance can be estimated accurately. For independent Gaussian readout noise with variance $\\sigma_n^2$:",
        "",
        "$$",
        "J(d,t) = \\frac{1}{\\sigma_n^2}\\left\\|\\frac{\\partial y(d,t)}{\\partial d}\\right\\|_2^2.",
        "$$",
        "",
        "For linear dynamics with an impulse input:",
        "",
        "$$",
        "y(d,t) = C e^{At} B h(d), \\qquad A = \\frac{-I+W}{\\tau}.",
        "$$",
        "",
        "Therefore:",
        "",
        "$$",
        "\\frac{\\partial y(d,t)}{\\partial d} = C e^{At} B h'(d),",
        "$$",
        "",
        "and:",
        "",
        "$$",
        "J(d,t) = \\frac{1}{\\sigma_n^2}\\left\\|C e^{At} B h'(d)\\right\\|_2^2.",
        "$$",
        "",
        "The Cramer-Rao bound gives the local lower bound:",
        "",
        "$$",
        "\\operatorname{Var}(\\hat d) \\geq \\frac{1}{J(d,t)}.",
        "$$",
        "",
        "The ranking metric used here is the mean Cramer-Rao RMSE across the line at the final readout time:",
        "",
        "$$",
        "\\operatorname{CRB}_{\\rm RMSE} = \\sqrt{\\frac{1}{N}\\sum_d \\frac{1}{J(d,T)}}.",
        "$$",
        "",
        "This is not a fitted label error. It is an analytical sensitivity measure through the chosen input matrix, recurrent dynamics, and readout.",
        "",
        "## Finite-Line Input Matrices",
        "",
        "The tested finite-line analogues of the ring circulant input matrix are:",
        "",
        "- `identity`: direct topographic input.",
        "- `toeplitz_raw`: a banded symmetric Toeplitz Gaussian matrix, with no edge correction.",
        "- `toeplitz_amplitude`: the same matrix with column amplitude compensation so edge columns do not lose total drive.",
        "- `reflected`: a reflected-boundary Gaussian matrix, equivalent to no-flux boundary correction.",
        "",
        "All input matrices are normalised to the same Frobenius norm as the identity input, so differences are structural rather than caused by injecting more total power.",
        "",
        "![Input matrix families](../outputs/finite_line_input_theory/figures/input_matrix_families.png)",
        "",
        "## One-Block Versus Two-Block Input",
        "",
        "The original ring notebook did not only use $B=[I;0]$. For the balanced model, it used a two-block input matrix:",
        "",
        "$$",
        "B = \\begin{bmatrix} B_E \\\\ B_I \\end{bmatrix}.",
        "$$",
        "",
        "This report compares:",
        "",
        "$$",
        "B_{1pop}=M, \\qquad B_{E-only}=\\begin{bmatrix}M\\\\0\\end{bmatrix}, \\qquad",
        "B_{opp}=\\frac{1}{\\sqrt{1+\\beta^2}}\\begin{bmatrix}M\\\\-\\beta M\\end{bmatrix}.",
        "$$",
        "",
        "The opponent input is a signed current formulation. It is the direct finite-line analogue of the ring-model E/I input trick, not a literal claim that inhibitory firing rates are negative.",
        "",
        "For the balanced matrix, $W_{EI}^2=0$, so:",
        "",
        "$$",
        "e^{(-I+W_{EI})t/\\tau}=e^{-t/\\tau}\\left(I+\\frac{t}{\\tau}W_{EI}\\right).",
        "$$",
        "",
        "Let $A_d=Mh'(d)$ and $G_d=K M h'(d)$. For $B_{opp}$:",
        "",
        "$$",
        "\\frac{\\partial y}{\\partial d}",
        "=\\frac{e^{-t/\\tau}}{\\sqrt{1+\\beta^2}}\\left[A_d + \\frac{t}{\\tau}(1+\\beta)G_d\\right].",
        "$$",
        "",
        "The FI numerator is therefore a quadratic ratio:",
        "",
        "$$",
        "F(\\beta)=\\frac{U+2\\beta V+\\beta^2 Z}{1+\\beta^2}.",
        "$$",
        "",
        "The stationary condition is:",
        "",
        "$$",
        "V\\beta^2 + (U-Z)\\beta - V = 0.",
        "$$",
        "",
        "This gives an analytic opponent gain for a fixed matrix family and objective, so no label fitting is used.",
        "",
        "![B block matrices](../outputs/finite_line_input_theory/figures/block_input_matrices.png)",
        "",
        "## Chosen Matrix Diagnostics",
        "",
        "The selected candidate uses a reflected finite-line input matrix and a balanced two-block recurrent matrix. The heatmap below shows the chosen input matrix $M$, the two input blocks $B_E$ and $B_I$, and the full recurrent matrix $W$.",
        "",
        "![Chosen matrices](../outputs/finite_line_input_theory/figures/chosen_matrices.png)",
        "",
        "The input matrix can also be analysed as a finite-line spatial filter. Because a line is not periodic, the appropriate clean basis is a cosine basis rather than the ring Fourier basis. The gain curve below shows $\\|M q_k\\|/\\|q_k\\|$ for cosine spatial mode $q_k$.",
        "",
        "![Spatial frequency response](../outputs/finite_line_input_theory/figures/spatial_frequency_response.png)",
        "",
        "## Recurrent Spectrum And Pseudospectrum",
        "",
        "The balanced E/I recurrence is asymptotically stable in continuous time because the system matrix is:",
        "",
        "$$",
        "A = \\frac{-I+W}{\\tau}.",
        "$$",
        "",
        "For the ideal balanced block, eigenvalues alone can look deceptively simple because the block is highly non-normal. Therefore, the report shows both eigenvalue spectra and a pseudospectrum proxy.",
        "",
        "![Recurrent spectrum](../outputs/finite_line_input_theory/figures/recurrent_spectrum.png)",
        "",
        "The pseudospectrum plot shows $\\log_{10}\\sigma_{\\min}(zI-A)$. Regions with small $\\sigma_{\\min}$ indicate where small perturbations could strongly change the apparent spectrum. This is useful for balanced E/I systems because transient amplification can occur even when all eigenvalues are stable.",
        "",
        "![Pseudospectrum](../outputs/finite_line_input_theory/figures/pseudospectrum.png)",
        "",
        "## Candidate Comparison",
        "",
        f"Parameters: `N={params.num_neurons}`, `L={params.length_m:g} m`, `sigma_h={params.stimulus_sigma_m:g} m`, `tau={params.tau_s * 1000:g} ms`, final readout `T={params.readout_time_s * 1000:g} ms`.",
        "",
        "| Block | Input family | Input width | Recurrent width | beta | Final CRB RMSE | 5ms CRB RMSE | FI uniformity | Mean COM bias | Edge COM bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Fisher And Width Sensitivity",
        "",
        "The best candidates have much flatter final-time Fisher information than uncorrected finite-line inputs. Amplitude compensation and reflection both reduce boundary loss, but reflection is the cleaner finite-line analogue of the ring because it preserves no-flux boundary structure.",
        "",
        "![Fisher curves](../outputs/finite_line_input_theory/figures/fisher_curves.png)",
        "",
        "The width sweep below compares the best two-block opponent candidate within each input family. Narrow reflected input generally preserves more position sensitivity; wider input becomes smoother but less discriminating.",
        "",
        "![Width sensitivity](../outputs/finite_line_input_theory/figures/width_sensitivity.png)",
        "",
        "The beta scan uses the reflected input with width `6` and recurrent width `8`. It shows why the two-block input matters: opponent E/I input can increase FI through the balanced transient, but too much opponent drive can increase bias or over-amplify the recurrent term.",
        "",
        "![Beta scan](../outputs/finite_line_input_theory/figures/beta_scan.png)",
        "",
        "## Alpha Sweep",
        "",
        "The selected reflected/opponent family was re-tested across balanced $\\alpha'$. For each $\\alpha'$, the opponent $\\beta$ was recomputed analytically from the final-time Fisher objective. This is still an analytical sweep, not a label fit.",
        "",
        "![Alpha sweep](../outputs/finite_line_input_theory/figures/alpha_sweep.png)",
        "",
        "| alpha prime | analytic beta | Final CRB RMSE | 5ms CRB RMSE | Mean COM bias | Edge COM bias | FI uniformity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *alpha_table_rows,
        "",
        f"The best alpha in this sweep by final CRB RMSE was `{best_alpha['alpha_prime']:.1f}`, with analytic beta `{best_alpha['beta']:.3f}` and final CRB RMSE `{best_alpha['final_crb_rmse_m'] * 100.0:.3f} cm`.",
        "",
        "## Biological Limits On Alpha",
        "",
        "The uncapped alpha sweep should not be interpreted as permission to increase gain indefinitely. This is the same limitation seen in the original ring notebook: increasing $\\alpha'$ improves the idealised accuracy metric, but it also increases neural activity and therefore metabolic cost. Real neurons have maximum firing rates, refractory periods, synaptic limits, and finite energy budgets.",
        "",
        "To illustrate this, the selected reflected/opponent family was re-simulated with a simple firing-rate cap inside the dynamics. The state is interpreted as relative rate around a `5 Hz` baseline, with `20 Hz` per activity unit used for this diagnostic scaling. The caps are applied as:",
        "",
        "$$",
        "r(t) \\leftarrow \\operatorname{clip}\\left(r(t), -r_0, r_{\\max}-r_0\\right),",
        "$$",
        "",
        "where $r_0=5\\,\\mathrm{Hz}$ and $r_{\\max}$ is either `55 Hz` or `100 Hz`. Because clipping makes the system nonlinear, the capped FI curves are estimated with finite differences rather than the analytic linear formula.",
        "",
        "![Capped alpha sweep](../outputs/finite_line_input_theory/figures/capped_alpha_sweep.png)",
        "",
        "| alpha prime | cap | analytic beta | Capped finite-difference CRB RMSE | Mean COM bias | Peak rate |",
        "|---:|---|---:|---:|---:|---:|",
        *capped_alpha_table_rows,
        "",
        "The rate trace below shows the same issue dynamically. Without a cap, high $\\alpha'$ produces increasingly large transient activity. With caps, the activity saturates, so extra gain no longer has the same linear Fisher-information benefit.",
        "",
        "![Capped rate traces](../outputs/finite_line_input_theory/figures/capped_rate_traces.png)",
        "",
        "## Fixed-Setup Alpha Sweep",
        "",
        "The previous alpha sweep recomputed the analytic opponent $\\beta$ for every $\\alpha'$. That is useful for asking what the best analytical gain should be at each recurrence strength, but it mixes two effects: stronger recurrence and retuned input balance.",
        "",
        "The diagnostic below keeps the selected setup fixed and changes only $\\alpha'$:",
        "",
        f"- input family: `{best['family']}`;",
        f"- input width: `{float(best['input_width_bins']):.0f}` bins;",
        f"- recurrent width: `{float(best['recurrent_width_bins']):.0f}` bins;",
        f"- fixed opponent beta: `{float(best['beta']):.3f}`.",
        "",
        "This isolates whether the balanced recurrence gain itself improves the line-attractor sensitivity, and how firing-rate caps change that conclusion.",
        "",
        "![Fixed setup alpha sweep](../outputs/finite_line_input_theory/figures/fixed_setup_alpha_sweep.png)",
        "",
        "| alpha prime | cap | fixed beta | Finite-difference CRB RMSE | Mean COM bias | Peak rate | Saturated state fraction |",
        "|---:|---|---:|---:|---:|---:|---:|",
        *fixed_alpha_table_rows,
        "",
        f"With this fixed setup, the best uncapped tested value was $\\alpha'={float(best_fixed_uncapped['alpha_prime']):.1f}$, giving finite-difference CRB RMSE `{float(best_fixed_uncapped['final_crb_rmse_m']) * 100.0:.3f} cm`. If this curve improves with $\\alpha'$ even when $\\beta$ is fixed, the gain is coming from balanced recurrent amplification rather than from repeatedly retuning the input matrix.",
        "",
        "## Bump Dynamics",
        "",
        "The snapshot plot shows the synthetic readout bump for selected one-population, E-only, and opponent candidates. This is still synthetic theory, not the real AC map.",
        "",
        "![Response snapshots](../outputs/finite_line_input_theory/figures/response_snapshots.png)",
        "",
        "## Interpretation",
        "",
        f"Best analytical candidate in the default-alpha grid by final Cramer-Rao RMSE: `{best['block_kind']}` with `{best['family']}` input, input width `{float(best['input_width_bins']):.0f}`, recurrent width `{float(best['recurrent_width_bins']):.0f}`, beta `{float(best['beta']):.3f}`.",
        "",
        "- The two-block opponent input is the closest finite-line transfer of the original ring-model FI theory.",
        "- The one-block and E-only versions are useful controls, but they do not exploit the balanced E/I transient as directly.",
        "- Edge correction matters. Raw Toeplitz input loses structure near the boundaries; reflected or amplitude-compensated input is more appropriate.",
        "- This report still does not prove the setup will improve the real distance pathway. It only identifies principled finite-line candidates to consider before integration.",
        "- The alpha sweep now behaves more like the original ring theory: stronger balanced recurrence improves the analytical FI metric over this tested range, although this should be capped by biological rate/stability constraints before integration.",
        "- Once firing-rate caps are included, high alpha becomes a tradeoff rather than a free improvement: the idealised FI metric improves, but activity saturates and power/firing-rate demands become unrealistic.",
        "- The next step, if accepted, is to port the best reflected/opponent family into `sc_line_attractor_integration.py` and compare it against the current simple COM readout.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run finite-line input theory experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)

    params = TheoryParams()
    stimulus_grid = np.linspace(0.1, params.length_m - 0.1, 101)
    candidates, beta_summary = candidate_suite(params)
    rows = [evaluate_candidate(params, candidate, stimulus_grid) for candidate in candidates]
    top_rows = compact_rows(rows, top_n=14)

    best_names = [str(row["name"]) for row in top_rows[:4]]
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    selected = [candidate_by_name[name] for name in best_names]
    selected_for_blocks = selected[:3]
    best_candidate = selected[0]
    alpha_rows = alpha_sweep(params, best_candidate.input_spec, best_candidate.recurrent_width_bins)
    capped_alpha_rows = capped_alpha_sweep(params, best_candidate.input_spec, best_candidate.recurrent_width_bins)
    fixed_alpha_rows = fixed_setup_alpha_sweep(
        params,
        best_candidate.input_spec,
        best_candidate.recurrent_width_bins,
        best_candidate.beta,
    )

    artifacts = {
        "input_matrix_families": plot_input_families(params, FIGURE_DIR / "input_matrix_families.png"),
        "chosen_matrices": plot_chosen_matrices(params, best_candidate, FIGURE_DIR / "chosen_matrices.png"),
        "spatial_frequency_response": plot_spatial_frequency_response(
            params,
            best_candidate,
            FIGURE_DIR / "spatial_frequency_response.png",
        ),
        "recurrent_spectrum": plot_recurrent_spectrum(params, best_candidate, FIGURE_DIR / "recurrent_spectrum.png"),
        "pseudospectrum": plot_pseudospectrum(params, best_candidate, FIGURE_DIR / "pseudospectrum.png"),
        "fisher_curves": plot_fisher_curves(params, selected, stimulus_grid, FIGURE_DIR / "fisher_curves.png"),
        "beta_scan": plot_beta_scan(rows, FIGURE_DIR / "beta_scan.png"),
        "width_sensitivity": plot_width_sweep(rows, FIGURE_DIR / "width_sensitivity.png"),
        "alpha_sweep": plot_alpha_sweep(alpha_rows, FIGURE_DIR / "alpha_sweep.png"),
        "capped_alpha_sweep": plot_capped_alpha_sweep(capped_alpha_rows, FIGURE_DIR / "capped_alpha_sweep.png"),
        "fixed_setup_alpha_sweep": plot_fixed_setup_alpha_sweep(
            fixed_alpha_rows,
            FIGURE_DIR / "fixed_setup_alpha_sweep.png",
        ),
        "capped_rate_traces": plot_rate_traces(
            params,
            best_candidate.input_spec,
            best_candidate.recurrent_width_bins,
            FIGURE_DIR / "capped_rate_traces.png",
        ),
        "block_input_matrices": plot_block_matrices(params, selected_for_blocks, FIGURE_DIR / "block_input_matrices.png"),
        "response_snapshots": plot_response_snapshots(params, selected_for_blocks, FIGURE_DIR / "response_snapshots.png"),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "finite_line_input_theory",
        "elapsed_seconds": elapsed_s,
        "params": params.__dict__,
        "beta_summary": beta_summary,
        "alpha_sweep": alpha_rows,
        "capped_alpha_sweep": capped_alpha_rows,
        "fixed_setup_alpha_sweep": fixed_alpha_rows,
        "top_rows": top_rows,
        "all_rows": rows,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(params, rows, top_rows, alpha_rows, capped_alpha_rows, fixed_alpha_rows, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
