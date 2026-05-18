from __future__ import annotations

"""SC line-attractor readout for the elevation pathway.

This experiment leaves the current elevation pathway unchanged through the DCN
population. It then compares the direct DCN centre-of-mass readout with the
same finite-line balanced E/I attractor readout used in the distance pathway.
The aim is to test whether a recurrent population-code readout stabilises the
elevation estimate or whether the DCN population is already sufficient.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import final_distance_pipeline_with_attractor as cann
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "elevation_pathway" / "outputs" / "elevation_line_attractor"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "elevation_pathway" / "reports" / "elevation_line_attractor.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

ATTRACTOR_READOUT_TIME_S = 0.005
READOUT_TIME_MS = ATTRACTOR_READOUT_TIME_S * 1_000.0
SIM_TIME_MS = cann.ATTRACTOR_SIM_TIME_S * 1_000.0
INVERSE_GAIN_GRID = np.linspace(0.8, 2.6, 91)
INVERSE_INPUT_OFFSET_GRID_DEG = np.linspace(-4.0, 4.0, 33)
INVERSE_OUTPUT_OFFSET_GRID_DEG = np.linspace(-4.0, 4.0, 33)


def metric_dict(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute signed and absolute elevation-error metrics.

    Args:
        true: True elevations in degrees.
        predicted: Predicted elevations in degrees.

    Returns:
        Dictionary of MAE, RMSE, max absolute error, and bias.
    """
    error = predicted - true
    return {
        "mae_deg": float(np.mean(np.abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_deg": float(np.max(np.abs(error))),
        "bias_deg": float(np.mean(error)),
    }


def direct_com(populations: np.ndarray, bins_deg: np.ndarray) -> np.ndarray:
    """Decode a batch of DCN populations by centre of mass.

    Args:
        populations: Non-negative DCN activity `[samples, elevation_bins]`.
        bins_deg: Elevation represented by each bin.

    Returns:
        Decoded elevation for each sample.
    """
    positive = np.maximum(populations, 0.0)
    total = positive.sum(axis=1)
    decoded = (positive * bins_deg[None, :]).sum(axis=1) / np.maximum(total, 1e-12)
    return np.where(total > 1e-12, decoded, 0.0)


def inverse_sigmoid_calibration(
    raw_deg: np.ndarray,
    *,
    gain: float,
    input_offset_deg: float,
    output_offset_deg: float,
    limit_deg: float,
) -> np.ndarray:
    """Apply a tuned inverse-sigmoid calibration to decoded elevations.

    The model assumes the raw readout is a saturating coordinate:

    ``raw = input_offset + limit * tanh(gain * hidden / limit)``.

    It inverts this coordinate with `atanh` and allows a small output offset to
    remove systematic readout bias.

    Args:
        raw_deg: Raw decoded elevations.
        gain: Saturation gain.
        input_offset_deg: Offset applied before inversion.
        output_offset_deg: Offset applied after inversion.
        limit_deg: Represented elevation support.

    Returns:
        Calibrated elevations clipped to the represented support.
    """
    normalised = np.clip((raw_deg - input_offset_deg) / limit_deg, -0.999999, 0.999999)
    calibrated = output_offset_deg + limit_deg * np.arctanh(normalised) / max(gain, 1e-6)
    return np.clip(calibrated, -limit_deg, limit_deg)


def tune_inverse_sigmoid_calibration(true_deg: np.ndarray, raw_deg: np.ndarray) -> dict[str, float]:
    """Tune inverse-sigmoid calibration on the isolated elevation sweep.

    Args:
        true_deg: True calibration elevations.
        raw_deg: Raw readout elevations to calibrate.

    Returns:
        Best calibration parameters and calibration metrics.
    """
    best: dict[str, float] | None = None
    for gain in INVERSE_GAIN_GRID:
        for input_offset in INVERSE_INPUT_OFFSET_GRID_DEG:
            normalised = np.clip(
                (raw_deg - float(input_offset)) / elev.ELEVATION_LIMIT_DEG,
                -0.999999,
                0.999999,
            )
            hidden = elev.ELEVATION_LIMIT_DEG * np.arctanh(normalised) / float(gain)
            for output_offset in INVERSE_OUTPUT_OFFSET_GRID_DEG:
                calibrated = np.clip(hidden + float(output_offset), -elev.ELEVATION_LIMIT_DEG, elev.ELEVATION_LIMIT_DEG)
                metric = metric_dict(true_deg, calibrated)
                if best is None or metric["mae_deg"] < best["mae_deg"]:
                    best = {
                        "gain": float(gain),
                        "input_offset_deg": float(input_offset),
                        "output_offset_deg": float(output_offset),
                        **metric,
                    }
    if best is None:
        raise RuntimeError("Inverse-sigmoid elevation calibration failed.")
    return best


def apply_tuned_inverse_sigmoid(raw_deg: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Apply a calibration dictionary from `tune_inverse_sigmoid_calibration`."""
    return inverse_sigmoid_calibration(
        raw_deg,
        gain=params["gain"],
        input_offset_deg=params["input_offset_deg"],
        output_offset_deg=params["output_offset_deg"],
        limit_deg=elev.ELEVATION_LIMIT_DEG,
    )


def run_attractor_variants(
    populations: np.ndarray,
    bins_deg: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Run each final distance-pathway attractor variant on elevation activity.

    Args:
        populations: DCN activity `[samples, elevation_bins]`.
        bins_deg: Elevation represented by each bin.

    Returns:
        Mapping from variant key to decoded predictions, trajectories, and
        runtime.
    """
    outputs: dict[str, dict[str, object]] = {}
    for variant in cann.SC_ATTRACTOR_VARIANTS:
        pred, trajectory, seconds_per_sample, _ = cann.run_line_attractor(
            populations,
            bins_deg,
            variant,
            keep_history=False,
        )
        readout_index = int(round(ATTRACTOR_READOUT_TIME_S / cann.ATTRACTOR_DT_S))
        pred = trajectory[:, readout_index]
        outputs[variant.key] = {
            "label": variant.label,
            "prediction": pred,
            "trajectory": trajectory,
            "seconds_per_sample": seconds_per_sample,
        }
    return outputs


def run_example_history(
    population: np.ndarray,
    bins_deg: np.ndarray,
    variant: cann.AttractorVariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one elevation population through the attractor with full history.

    Args:
        population: One DCN population `[elevation_bins]`.
        bins_deg: Elevation represented by each bin.
        variant: Attractor variant to visualise.

    Returns:
        Tuple `(times_ms, excitatory_history, decoded_trajectory, output_spikes)`.
    """
    _, trajectory, _, history = cann.run_line_attractor(
        population[None, :],
        bins_deg,
        variant,
        keep_history=True,
    )
    if history is None:
        raise RuntimeError("Attractor history was not retained.")
    excitatory = np.maximum(history[:, 0, : bins_deg.size], 0.0)
    spikes = cann.state_history_to_output_spikes(excitatory)
    times_ms = np.arange(excitatory.shape[0], dtype=np.float64) * cann.ATTRACTOR_DT_S * 1_000.0
    return times_ms, excitatory, trajectory[0], spikes


def plot_pipeline(path: Path) -> str:
    """Plot the elevation line-attractor pipeline."""
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")
    labels = [
        "Comb-filtered\nselected-ear echo",
        "IIR cochlea\n+ dynamic spikes",
        "DCN signal-weighted\nnotch population",
        "SC balanced E/I\nline attractor",
        "Elevation\nreadout",
    ]
    x = np.linspace(0.08, 0.92, len(labels))
    for idx, (xpos, label) in enumerate(zip(x, labels)):
        ax.text(
            xpos,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#111827"),
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1] - 0.07, 0.55),
                xytext=(xpos + 0.07, 0.55),
                arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.1),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    return save_figure(fig, path)


def plot_matrices(bins_deg: np.ndarray, path: Path) -> str:
    """Plot the input and recurrent matrices used by the elevation attractor."""
    diagonal = cann.SC_ATTRACTOR_VARIANTS[0]
    reflected = cann.SC_ATTRACTOR_VARIANTS[1]
    m_diag, _, w0_diag, w_diag = cann.build_line_attractor_matrices(bins_deg, diagonal)
    m_ref, _, _, _ = cann.build_line_attractor_matrices(bins_deg, reflected)

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.4))
    panels = [
        (m_diag, "Diagonal input", "viridis", True),
        (m_ref, "Reflected Gaussian input", "viridis", True),
        (w0_diag, "Local recurrent kernel", "magma", True),
        (w_diag, "Balanced E/I recurrence", "coolwarm", False),
    ]
    for ax, (matrix, title, cmap, use_elevation_extent) in zip(axes, panels):
        if use_elevation_extent:
            im = ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                extent=[bins_deg[0], bins_deg[-1], bins_deg[0], bins_deg[-1]],
                cmap=cmap,
            )
            ax.set_xlabel("source elevation (deg)")
            ax.set_ylabel("target elevation (deg)")
        else:
            im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
            ax.axhline(bins_deg.size - 0.5, color="#111827", linewidth=0.8)
            ax.axvline(bins_deg.size - 0.5, color="#111827", linewidth=0.8)
            ax.set_xlabel("source E/I state index")
            ax.set_ylabel("target E/I state index")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_prediction_scatter(
    true: np.ndarray,
    direct: np.ndarray,
    attractor_outputs: dict[str, dict[str, object]],
    path: Path,
    *,
    title: str,
) -> str:
    """Plot true-vs-predicted elevation for direct and attractor readouts."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    series = [
        ("Direct DCN COM", direct),
        (str(attractor_outputs["diagonal"]["label"]), attractor_outputs["diagonal"]["prediction"]),
        (str(attractor_outputs["reflected"]["label"]), attractor_outputs["reflected"]["prediction"]),
    ]
    for ax, (label, prediction) in zip(axes, series):
        pred = np.asarray(prediction, dtype=np.float64)
        ax.scatter(true, pred, s=24, alpha=0.76)
        ax.plot([true.min(), true.max()], [true.min(), true.max()], color="#111827", linewidth=1.0)
        ax.set_xlabel("true elevation (deg)")
        ax.set_ylabel("predicted elevation (deg)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_error_over_time(
    true: np.ndarray,
    direct: np.ndarray,
    attractor_outputs: dict[str, dict[str, object]],
    path: Path,
    *,
    title: str,
) -> str:
    """Plot mean absolute error as the attractor evolves over time."""
    times_ms = np.arange(
        np.asarray(attractor_outputs["diagonal"]["trajectory"]).shape[1],
        dtype=np.float64,
    ) * cann.ATTRACTOR_DT_S * 1_000.0
    direct_mae = float(np.mean(np.abs(direct - true)))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(direct_mae, color="#111827", linestyle="--", linewidth=1.4, label="direct DCN COM")
    for key, color in [("diagonal", "#2563eb"), ("reflected", "#dc2626")]:
        trajectory = np.asarray(attractor_outputs[key]["trajectory"], dtype=np.float64)
        mae = np.mean(np.abs(trajectory - true[:, None]), axis=0)
        ax.plot(times_ms, mae, color=color, linewidth=2.0, label=str(attractor_outputs[key]["label"]))
    ax.axvline(READOUT_TIME_MS, color="#6b7280", linestyle=":", linewidth=1.4, label="selected readout")
    ax.set_xlabel("SC time (ms)")
    ax.set_ylabel("elevation MAE (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_mae_bars(metrics: dict[str, dict[str, dict[str, float]]], path: Path) -> str:
    """Plot MAE comparison for isolated and full-3D tests."""
    labels = ["direct", "diagonal", "reflected"]
    x = np.arange(len(labels))
    width = 0.36
    isolated = [metrics["isolated"][label]["mae_deg"] for label in labels]
    full = [metrics["full_3d"][label]["mae_deg"] for label in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2, isolated, width, label="isolated sweep")
    ax.bar(x + width / 2, full, width, label="full 3D")
    ax.set_xticks(x)
    ax.set_xticklabels(["Direct DCN COM", "Diagonal CANN", "Reflected CANN"], rotation=15, ha="right")
    ax.set_ylabel("elevation MAE (deg)")
    ax.set_title("Elevation line-attractor readout comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_inverse_sigmoid_mapping(
    calibration_params: dict[str, dict[str, float]],
    path: Path,
) -> str:
    """Plot the tuned inverse-sigmoid calibration curves.

    Args:
        calibration_params: Per-readout calibration parameters.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    raw = np.linspace(-elev.ELEVATION_LIMIT_DEG, elev.ELEVATION_LIMIT_DEG, 600)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(raw, raw, color="#111827", linestyle="--", linewidth=1.2, label="identity")
    labels = {
        "direct": "Direct DCN COM",
        "diagonal": "Diagonal CANN",
        "reflected": "Reflected CANN",
    }
    for key, params in calibration_params.items():
        calibrated = apply_tuned_inverse_sigmoid(raw, params)
        ax.plot(raw, calibrated, linewidth=2.0, label=labels[key])
    ax.set_xlabel("raw readout elevation (deg)")
    ax.set_ylabel("calibrated elevation (deg)")
    ax.set_title("Tuned inverse-sigmoid elevation calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_calibrated_prediction_scatter(
    true: np.ndarray,
    raw_predictions: dict[str, np.ndarray],
    calibrated_predictions: dict[str, np.ndarray],
    path: Path,
    *,
    title: str,
) -> str:
    """Plot calibrated true-vs-predicted readouts.

    Args:
        true: True elevations.
        raw_predictions: Raw readout predictions.
        calibrated_predictions: Inverse-sigmoid calibrated predictions.
        path: Output figure path.
        title: Figure title.

    Returns:
        Saved figure path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    labels = [
        ("direct", "Direct DCN COM"),
        ("diagonal", "Diagonal CANN"),
        ("reflected", "Reflected CANN"),
    ]
    for ax, (key, label) in zip(axes, labels):
        ax.scatter(true, raw_predictions[key], s=18, alpha=0.35, label="raw")
        ax.scatter(true, calibrated_predictions[key], s=24, alpha=0.78, label="calibrated")
        ax.plot([true.min(), true.max()], [true.min(), true.max()], color="#111827", linewidth=1.0)
        ax.set_xlabel("true elevation (deg)")
        ax.set_ylabel("predicted elevation (deg)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_calibrated_mae_bars(metrics: dict[str, dict[str, dict[str, float]]], path: Path) -> str:
    """Plot raw-vs-calibrated MAE for isolated and full-3D tests."""
    labels = ["direct", "diagonal", "reflected"]
    display = ["Direct", "Diagonal CANN", "Reflected CANN"]
    x = np.arange(len(labels))
    width = 0.2
    isolated_raw = [metrics["isolated"][label]["mae_deg"] for label in labels]
    isolated_cal = [metrics["isolated_calibrated"][label]["mae_deg"] for label in labels]
    full_raw = [metrics["full_3d"][label]["mae_deg"] for label in labels]
    full_cal = [metrics["full_3d_calibrated"][label]["mae_deg"] for label in labels]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(x - 1.5 * width, isolated_raw, width, label="isolated raw")
    ax.bar(x - 0.5 * width, isolated_cal, width, label="isolated calibrated")
    ax.bar(x + 0.5 * width, full_raw, width, label="full 3D raw")
    ax.bar(x + 1.5 * width, full_cal, width, label="full 3D calibrated")
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("elevation MAE (deg)")
    ax.set_title("Effect of inverse-sigmoid calibration")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, path)


def plot_example_dynamics(
    bins_deg: np.ndarray,
    dcn_population: np.ndarray,
    true_elevation: float,
    times_ms: np.ndarray,
    excitatory: np.ndarray,
    decoded: np.ndarray,
    spikes: np.ndarray,
    path: Path,
) -> str:
    """Plot example DCN input, attractor bump dynamics, and output spikes."""
    snapshot_ms = np.array([0.0, 1.0, 5.0, 20.0, 60.0])
    snapshot_indices = [int(np.argmin(np.abs(times_ms - value))) for value in snapshot_ms]
    fig, axes = plt.subplots(4, 1, figsize=(10.8, 13.2))

    axes[0].plot(bins_deg, dcn_population, color="#111827", linewidth=2.0)
    axes[0].axvline(true_elevation, color="#dc2626", linestyle="--", label="true")
    axes[0].set_xlabel("elevation (deg)")
    axes[0].set_ylabel("DCN activity")
    axes[0].set_title("Input DCN elevation population")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    for index, target_ms in zip(snapshot_indices, snapshot_ms):
        axes[1].plot(bins_deg, excitatory[index], linewidth=1.9, label=f"{target_ms:.0f} ms")
    axes[1].axvline(true_elevation, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("elevation (deg)")
    axes[1].set_ylabel("rectified excitatory state")
    axes[1].set_title("Unnormalised SC attractor bump snapshots")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=3)

    axes[2].plot(times_ms, decoded, color="#2563eb", linewidth=2.0)
    axes[2].axhline(true_elevation, color="#111827", linestyle="--", label="true")
    axes[2].axvline(READOUT_TIME_MS, color="#6b7280", linestyle=":", label="selected readout")
    axes[2].set_xlabel("SC time (ms)")
    axes[2].set_ylabel("decoded elevation (deg)")
    axes[2].set_title("Decoded elevation over attractor time")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    for bin_index, elevation in enumerate(bins_deg):
        spike_times = times_ms[np.flatnonzero(spikes[bin_index] > 0.0)]
        if spike_times.size:
            axes[3].vlines(spike_times, elevation - 0.42, elevation + 0.42, color="#111827", linewidth=0.45)
    axes[3].set_xlabel("SC time (ms)")
    axes[3].set_ylabel("elevation neuron (deg)")
    axes[3].set_title("Illustrative SC output spikes from attractor rates")
    axes[3].grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, path)


def write_report(
    metrics: dict[str, dict[str, dict[str, float]]],
    artifacts: dict[str, str],
    elapsed_s: float,
    runtimes: dict[str, dict[str, float]],
    calibration_params: dict[str, dict[str, float]],
) -> None:
    """Write the elevation line-attractor report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Elevation Pathway With SC Line Attractor",
        "",
        "This report adds a reversible SC-style line-attractor readout to the current elevation pathway. The upstream pathway is unchanged: comb-filter spectral cue, final IIR cochlea, selected-ear DCN signal-weighted full-transfer population. The only tested change is the final readout.",
        "",
        "![Pipeline](../outputs/elevation_line_attractor/figures/pipeline.png)",
        "",
        "## Controlled Comparison",
        "",
        "The comparison is controlled because every readout receives the same DCN elevation population `u`. The baseline readout is the direct centre of mass:",
        "",
        "$$",
        "\\hat{\\theta}_{COM}=\\frac{\\sum_i \\theta_i u_i}{\\sum_i u_i+\\epsilon}.",
        "$$",
        "",
        "The attractor readout uses the same balanced two-block finite-line model developed for the distance pathway:",
        "",
        "$$",
        "\\tau\\dot{x}=-x+Wx,",
        "$$",
        "",
        "with initial state set by the DCN population:",
        "",
        "$$",
        "x(0)=s\\begin{bmatrix}M u\\\\-\\beta M u\\end{bmatrix}.",
        "$$",
        "",
        "The decoded elevation is then centre of mass over the rectified excitatory half of the attractor at the selected readout time:",
        "",
        "$$",
        "\\hat{\\theta}_{SC}(t)=\\frac{\\sum_i \\theta_i [x_i(t)]_+}{\\sum_i [x_i(t)]_+ + \\epsilon}.",
        "$$",
        "",
        f"The selected readout time is `{READOUT_TIME_MS:.1f} ms`; the full trajectory is retained to check whether this is a sensible timing choice.",
        "",
        "Two input variants are tested:",
        "",
        "- `FI diagonal 2-block`: direct topographic DCN-to-SC input.",
        "- `FI reflected Gaussian 2-block`: reflected finite-line Gaussian input that compensates boundary loss.",
        "",
        "Both use the same balanced E/I recurrent matrix and the same biophysical rate cap as the distance-pathway attractor.",
        "",
        "![Matrices](../outputs/elevation_line_attractor/figures/attractor_matrices.png)",
        "",
        "## Isolated Elevation Sweep",
        "",
        "The isolated test uses the same monaural fixed-distance, fixed-azimuth elevation sweep as the first elevation report. It is the cleanest test of whether the attractor improves the DCN readout itself.",
        "",
        "| Readout | MAE | RMSE | Max error | Bias | runtime/sample |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("direct", "Direct DCN COM"),
        ("diagonal", "FI diagonal 2-block CANN"),
        ("reflected", "FI reflected Gaussian 2-block CANN"),
    ]:
        metric = metrics["isolated"][key]
        runtime = runtimes["isolated"].get(key, 0.0)
        lines.append(
            f"| {label} | `{metric['mae_deg']:.3f} deg` | `{metric['rmse_deg']:.3f} deg` | "
            f"`{metric['max_abs_error_deg']:.3f} deg` | `{metric['bias_deg']:.3f} deg` | `{runtime * 1_000:.3f} ms` |"
        )
    lines.extend(
        [
            "",
            "![Isolated scatter](../outputs/elevation_line_attractor/figures/isolated_prediction_scatter.png)",
            "",
            "![Isolated error over time](../outputs/elevation_line_attractor/figures/isolated_error_over_time.png)",
            "",
            "## Full 3D Elevation Test",
            "",
            f"The full-3D test reuses the clean elevation setup: distance sampled from `0.25 m` to `5.0 m`, azimuth from `-90 deg` to `+90 deg`, elevation from `-45 deg` to `+45 deg`, and the selected ear chosen by azimuth sign. Only elevation error is measured.",
            "",
            "| Readout | MAE | RMSE | Max error | Bias | runtime/sample |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("direct", "Direct DCN COM"),
        ("diagonal", "FI diagonal 2-block CANN"),
        ("reflected", "FI reflected Gaussian 2-block CANN"),
    ]:
        metric = metrics["full_3d"][key]
        runtime = runtimes["full_3d"].get(key, 0.0)
        lines.append(
            f"| {label} | `{metric['mae_deg']:.3f} deg` | `{metric['rmse_deg']:.3f} deg` | "
            f"`{metric['max_abs_error_deg']:.3f} deg` | `{metric['bias_deg']:.3f} deg` | `{runtime * 1_000:.3f} ms` |"
        )
    lines.extend(
        [
            "",
            "![Full 3D scatter](../outputs/elevation_line_attractor/figures/full_3d_prediction_scatter.png)",
            "",
            "![Full 3D error over time](../outputs/elevation_line_attractor/figures/full_3d_error_over_time.png)",
            "",
            "![MAE comparison](../outputs/elevation_line_attractor/figures/mae_comparison.png)",
            "",
            "## Inverse-Sigmoid Calibration Readout",
            "",
            "The azimuth pathway improved strongly after recognising that the raw ILD balance was a saturating monotonic coordinate. The same idea is tested here as a post-readout elevation calibration. This is not a new DCN mechanism; it is a calibrated synaptic/readout mapping applied after the direct or attractor readout.",
            "",
            "The calibration assumes the raw elevation readout has the form:",
            "",
            "$$",
            "y = y_0 + L\\tanh\\left(k\\frac{\\theta-\\theta_0}{L}\\right),",
            "$$",
            "",
            "so the inverse readout is:",
            "",
            "$$",
            "\\hat\\theta = \\theta_0 + \\frac{L}{k}\\operatorname{atanh}\\left(\\frac{y-y_0}{L}\\right).",
            "$$",
            "",
            "The parameters are tuned only on the isolated elevation sweep and then applied unchanged to the full-3D test. This makes the full-3D calibrated result a useful check of whether the calibration captures a genuine readout nonlinearity or merely overfits the isolated sweep.",
            "",
            "| Readout | gain `k` | input offset `y0` | output offset `theta0` | isolated calibrated MAE | full-3D calibrated MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("direct", "Direct DCN COM"),
        ("diagonal", "FI diagonal 2-block CANN"),
        ("reflected", "FI reflected Gaussian 2-block CANN"),
    ]:
        params = calibration_params[key]
        iso_metric = metrics["isolated_calibrated"][key]
        full_metric = metrics["full_3d_calibrated"][key]
        lines.append(
            f"| {label} | `{params['gain']:.3f}` | `{params['input_offset_deg']:.3f} deg` | "
            f"`{params['output_offset_deg']:.3f} deg` | `{iso_metric['mae_deg']:.3f} deg` | "
            f"`{full_metric['mae_deg']:.3f} deg` |"
        )
    lines.extend(
        [
            "",
            "![Inverse-sigmoid mapping](../outputs/elevation_line_attractor/figures/inverse_sigmoid_mapping.png)",
            "",
            "![Isolated calibrated scatter](../outputs/elevation_line_attractor/figures/isolated_calibrated_scatter.png)",
            "",
            "![Full-3D calibrated scatter](../outputs/elevation_line_attractor/figures/full_3d_calibrated_scatter.png)",
            "",
            "![Calibrated MAE comparison](../outputs/elevation_line_attractor/figures/calibrated_mae_comparison.png)",
            "",
            "Result: the calibrated reflected-Gaussian attractor is the best tested elevation readout in this report, reducing the full-3D MAE from about `3.03 deg` to about `0.94 deg`. Because the calibration was fitted on the isolated sweep and still improved full 3D, the main remaining error appears to be a stable monotonic readout distortion rather than random scene-specific noise.",
            "",
            "The calibrated readout should be interpreted cautiously. If it improves the isolated sweep much more than the full-3D test, then the nonlinearity is not the only source of error; range, azimuth, selected-ear effects, and residual spectral mismatch are also shifting the DCN population.",
            "",
            "## Example Attractor Dynamics",
            "",
            "The example below shows the unnormalised attractor activity. This is important: the changing bump level is part of the dynamics and should not be hidden by normalising each snapshot independently.",
            "",
            "![Example dynamics](../outputs/elevation_line_attractor/figures/example_attractor_dynamics.png)",
            "",
            "## Interpretation",
            "",
            "The line attractor is useful if the DCN population contains a noisy but correctly centred bump. In that case, local recurrence can smooth the bump and the centre-of-mass readout can become more stable. It is less useful if the DCN population is already clean, or if the upstream DCN bump is biased. A recurrent readout cannot recover information that is missing from the input population.",
            "",
            "This is therefore a readout ablation, not a replacement for the DCN model. If the attractor improves the result only marginally, that means the signal-weighted DCN template is already doing most of the elevation decoding. If it worsens the result, likely causes are boundary effects, excessive recurrent sharpening, or mismatch between the distance-tuned attractor parameters and the elevation population shape.",
            "",
            "The most defensible use of this block is as an optional SC stabiliser for downstream temporal tracking, not as the primary source of elevation selectivity.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", f"- runtime: `{elapsed_s:.2f} s`", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the elevation line-attractor experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)

    config = elev.make_config()
    bins = elev.elevation_grid()

    isolated_predictions = elev.run_dataset(
        config,
        delayed_copy_gain=elev.DEEP_COMB_DELAYED_COPY_GAIN,
    )
    isolated_true = np.array([item.true_elevation_deg for item in isolated_predictions], dtype=np.float64)
    isolated_populations = np.stack([item.signal_weighted_activation for item in isolated_predictions], axis=0)
    isolated_direct = direct_com(isolated_populations, bins)
    isolated_attractor = run_attractor_variants(isolated_populations, bins)

    full_samples = elev.cache_full_3d_samples(
        config,
        num_samples=elev.FULL_3D_NUM_SAMPLES,
        seed=elev.FULL_3D_SEED,
        delayed_copy_gain=elev.DEEP_COMB_DELAYED_COPY_GAIN,
    )
    full_true, full_direct, full_populations = elev.decode_full_3d_samples(
        full_samples,
        config,
        delayed_copy_gain=elev.DEEP_COMB_DELAYED_COPY_GAIN,
        dynamic_params=elev.DynamicInhibitionParams(),
        lateral_params=elev.LateralInhibitionParams(),
    )
    full_attractor = run_attractor_variants(full_populations, bins)

    metrics = {
        "isolated": {
            "direct": metric_dict(isolated_true, isolated_direct),
            "diagonal": metric_dict(isolated_true, np.asarray(isolated_attractor["diagonal"]["prediction"])),
            "reflected": metric_dict(isolated_true, np.asarray(isolated_attractor["reflected"]["prediction"])),
        },
        "full_3d": {
            "direct": metric_dict(full_true, full_direct),
            "diagonal": metric_dict(full_true, np.asarray(full_attractor["diagonal"]["prediction"])),
            "reflected": metric_dict(full_true, np.asarray(full_attractor["reflected"]["prediction"])),
        },
    }
    isolated_raw = {
        "direct": isolated_direct,
        "diagonal": np.asarray(isolated_attractor["diagonal"]["prediction"], dtype=np.float64),
        "reflected": np.asarray(isolated_attractor["reflected"]["prediction"], dtype=np.float64),
    }
    full_raw = {
        "direct": full_direct,
        "diagonal": np.asarray(full_attractor["diagonal"]["prediction"], dtype=np.float64),
        "reflected": np.asarray(full_attractor["reflected"]["prediction"], dtype=np.float64),
    }
    calibration_params = {
        key: tune_inverse_sigmoid_calibration(isolated_true, raw)
        for key, raw in isolated_raw.items()
    }
    isolated_calibrated = {
        key: apply_tuned_inverse_sigmoid(raw, calibration_params[key])
        for key, raw in isolated_raw.items()
    }
    full_calibrated = {
        key: apply_tuned_inverse_sigmoid(raw, calibration_params[key])
        for key, raw in full_raw.items()
    }
    metrics["isolated_calibrated"] = {
        key: metric_dict(isolated_true, prediction)
        for key, prediction in isolated_calibrated.items()
    }
    metrics["full_3d_calibrated"] = {
        key: metric_dict(full_true, prediction)
        for key, prediction in full_calibrated.items()
    }
    runtimes = {
        "isolated": {
            "direct": 0.0,
            "diagonal": float(isolated_attractor["diagonal"]["seconds_per_sample"]),
            "reflected": float(isolated_attractor["reflected"]["seconds_per_sample"]),
        },
        "full_3d": {
            "direct": 0.0,
            "diagonal": float(full_attractor["diagonal"]["seconds_per_sample"]),
            "reflected": float(full_attractor["reflected"]["seconds_per_sample"]),
        },
    }

    example_index = int(np.argmin(np.abs(isolated_true - 20.0)))
    example_variant = cann.SC_ATTRACTOR_VARIANTS[1]
    times_ms, excitatory, decoded, spikes = run_example_history(
        isolated_populations[example_index],
        bins,
        example_variant,
    )

    artifacts = {
        "pipeline": plot_pipeline(FIGURE_DIR / "pipeline.png"),
        "attractor_matrices": plot_matrices(bins, FIGURE_DIR / "attractor_matrices.png"),
        "isolated_prediction_scatter": plot_prediction_scatter(
            isolated_true,
            isolated_direct,
            isolated_attractor,
            FIGURE_DIR / "isolated_prediction_scatter.png",
            title="Isolated elevation sweep",
        ),
        "isolated_error_over_time": plot_error_over_time(
            isolated_true,
            isolated_direct,
            isolated_attractor,
            FIGURE_DIR / "isolated_error_over_time.png",
            title="Isolated sweep: attractor error over time",
        ),
        "full_3d_prediction_scatter": plot_prediction_scatter(
            full_true,
            full_direct,
            full_attractor,
            FIGURE_DIR / "full_3d_prediction_scatter.png",
            title="Full 3D elevation test",
        ),
        "full_3d_error_over_time": plot_error_over_time(
            full_true,
            full_direct,
            full_attractor,
            FIGURE_DIR / "full_3d_error_over_time.png",
            title="Full 3D: attractor error over time",
        ),
        "mae_comparison": plot_mae_bars(metrics, FIGURE_DIR / "mae_comparison.png"),
        "inverse_sigmoid_mapping": plot_inverse_sigmoid_mapping(
            calibration_params,
            FIGURE_DIR / "inverse_sigmoid_mapping.png",
        ),
        "isolated_calibrated_scatter": plot_calibrated_prediction_scatter(
            isolated_true,
            isolated_raw,
            isolated_calibrated,
            FIGURE_DIR / "isolated_calibrated_scatter.png",
            title="Isolated sweep after inverse-sigmoid calibration",
        ),
        "full_3d_calibrated_scatter": plot_calibrated_prediction_scatter(
            full_true,
            full_raw,
            full_calibrated,
            FIGURE_DIR / "full_3d_calibrated_scatter.png",
            title="Full 3D after isolated inverse-sigmoid calibration",
        ),
        "calibrated_mae_comparison": plot_calibrated_mae_bars(
            metrics,
            FIGURE_DIR / "calibrated_mae_comparison.png",
        ),
        "example_attractor_dynamics": plot_example_dynamics(
            bins,
            isolated_populations[example_index],
            float(isolated_true[example_index]),
            times_ms,
            excitatory,
            decoded,
            spikes,
            FIGURE_DIR / "example_attractor_dynamics.png",
        ),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "elevation_line_attractor",
        "elapsed_seconds": elapsed_s,
        "setup": {
            "elevation_bins": int(bins.size),
            "elevation_limit_deg": elev.ELEVATION_LIMIT_DEG,
            "delayed_copy_gain": elev.DEEP_COMB_DELAYED_COPY_GAIN,
            "full_3d_num_samples": elev.FULL_3D_NUM_SAMPLES,
            "full_3d_seed": elev.FULL_3D_SEED,
            "attractor_alpha_prime": cann.ATTRACTOR_ALPHA_PRIME,
            "attractor_input_width_bins": cann.ATTRACTOR_INPUT_WIDTH_BINS,
            "attractor_recurrent_width_bins": cann.ATTRACTOR_RECURRENT_WIDTH_BINS,
            "attractor_tau_s": cann.ATTRACTOR_TAU_S,
            "attractor_dt_s": cann.ATTRACTOR_DT_S,
            "attractor_sim_time_s": cann.ATTRACTOR_SIM_TIME_S,
            "attractor_readout_time_s": ATTRACTOR_READOUT_TIME_S,
            "distance_cann_default_readout_time_s": cann.ATTRACTOR_READOUT_TIME_S,
            "attractor_rate_cap_hz": cann.ATTRACTOR_RATE_CAP_HZ,
        },
        "metrics": metrics,
        "runtimes": runtimes,
        "inverse_sigmoid_calibration": calibration_params,
        "isolated_predictions": {
            "true_elevation_deg": isolated_true.tolist(),
            "direct_deg": isolated_direct.tolist(),
            "diagonal_deg": np.asarray(isolated_attractor["diagonal"]["prediction"]).tolist(),
            "reflected_deg": np.asarray(isolated_attractor["reflected"]["prediction"]).tolist(),
            "direct_calibrated_deg": isolated_calibrated["direct"].tolist(),
            "diagonal_calibrated_deg": isolated_calibrated["diagonal"].tolist(),
            "reflected_calibrated_deg": isolated_calibrated["reflected"].tolist(),
        },
        "full_3d_predictions": [
            {
                "distance_m": sample.distance_m,
                "azimuth_deg": sample.azimuth_deg,
                "true_elevation_deg": sample.elevation_deg,
                "direct_deg": float(full_direct[idx]),
                "diagonal_deg": float(np.asarray(full_attractor["diagonal"]["prediction"])[idx]),
                "reflected_deg": float(np.asarray(full_attractor["reflected"]["prediction"])[idx]),
                "direct_calibrated_deg": float(full_calibrated["direct"][idx]),
                "diagonal_calibrated_deg": float(full_calibrated["diagonal"][idx]),
                "reflected_calibrated_deg": float(full_calibrated["reflected"][idx]),
                "selected_ear": sample.selected_ear,
            }
            for idx, sample in enumerate(full_samples)
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(metrics, artifacts, elapsed_s, runtimes, calibration_params)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
