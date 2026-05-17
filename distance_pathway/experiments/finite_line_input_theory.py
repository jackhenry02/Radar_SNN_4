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
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

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


def write_report(
    params: TheoryParams,
    rows: list[dict[str, float | str]],
    top_rows: list[dict[str, float | str]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the analytical finite-line theory report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = top_rows[0]
    table_rows = format_table(top_rows)
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
        "## Bump Dynamics",
        "",
        "The snapshot plot shows the synthetic readout bump for selected one-population, E-only, and opponent candidates. This is still synthetic theory, not the real AC map.",
        "",
        "![Response snapshots](../outputs/finite_line_input_theory/figures/response_snapshots.png)",
        "",
        "## Interpretation",
        "",
        f"Best analytical candidate by final Cramer-Rao RMSE: `{best['block_kind']}` with `{best['family']}` input, input width `{float(best['input_width_bins']):.0f}`, recurrent width `{float(best['recurrent_width_bins']):.0f}`, beta `{float(best['beta']):.3f}`.",
        "",
        "- The two-block opponent input is the closest finite-line transfer of the original ring-model FI theory.",
        "- The one-block and E-only versions are useful controls, but they do not exploit the balanced E/I transient as directly.",
        "- Edge correction matters. Raw Toeplitz input loses structure near the boundaries; reflected or amplitude-compensated input is more appropriate.",
        "- This report still does not prove the setup will improve the real distance pathway. It only identifies principled finite-line candidates to consider before integration.",
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

    artifacts = {
        "input_matrix_families": plot_input_families(params, FIGURE_DIR / "input_matrix_families.png"),
        "fisher_curves": plot_fisher_curves(params, selected, stimulus_grid, FIGURE_DIR / "fisher_curves.png"),
        "beta_scan": plot_beta_scan(rows, FIGURE_DIR / "beta_scan.png"),
        "width_sensitivity": plot_width_sweep(rows, FIGURE_DIR / "width_sensitivity.png"),
        "block_input_matrices": plot_block_matrices(params, selected_for_blocks, FIGURE_DIR / "block_input_matrices.png"),
        "response_snapshots": plot_response_snapshots(params, selected_for_blocks, FIGURE_DIR / "response_snapshots.png"),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "finite_line_input_theory",
        "elapsed_seconds": elapsed_s,
        "params": params.__dict__,
        "beta_summary": beta_summary,
        "top_rows": top_rows,
        "all_rows": rows,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(params, rows, top_rows, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
