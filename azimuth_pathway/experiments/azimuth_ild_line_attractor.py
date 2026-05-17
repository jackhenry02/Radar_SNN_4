from __future__ import annotations

"""Azimuth inverse-sigmoid ILD pathway with SC line-attractor readout.

This experiment keeps the azimuth acoustic/cochlea/LSO pathway from
`azimuth_pathway_first_attempt.py`, then replaces the final direct centre of
mass readout with the same FI reflected Gaussian two-block balanced E/I line
attractor used in the final distance pathway.
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

from azimuth_pathway.experiments import azimuth_pathway_first_attempt as az
from distance_pathway.experiments import final_distance_pipeline_with_attractor as distance_cann
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "azimuth_pathway" / "outputs" / "ild_line_attractor"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "azimuth_pathway" / "reports" / "azimuth_ild_line_attractor.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

ATTRACTOR_VARIANT = distance_cann.SC_ATTRACTOR_VARIANTS[1]


def metric_dict(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute azimuth error metrics.

    Args:
        true: True azimuths in degrees.
        pred: Predicted azimuths in degrees.

    Returns:
        MAE, RMSE, max absolute error, and bias.
    """
    error = pred - true
    return {
        "mae_deg": float(np.mean(np.abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_deg": float(np.max(np.abs(error))),
        "bias_deg": float(np.mean(error)),
    }


def inverse_ild_population_dataset(
    predictions: list[az.AzimuthPrediction],
    bins_deg: np.ndarray,
    inverse_params: dict[str, float],
    limit_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inverse-sigmoid ILD populations and direct COM predictions.

    Args:
        predictions: Existing azimuth-pathway predictions.
        bins_deg: Represented azimuth bins.
        inverse_params: Tuned inverse-sigmoid mapping parameters.
        limit_deg: Symmetric represented azimuth range.

    Returns:
        Pair `(direct_predictions, populations)`.
    """
    direct, activations = az.decode_inverse_sigmoid_ild(
        predictions,
        bins_deg,
        inverse_params["gain"],
        inverse_params["sigma"],
        limit_deg,
    )
    return direct, np.stack(activations, axis=0)


def run_cann_readout(populations: np.ndarray, bins_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Run the distance-pathway SC line attractor on azimuth populations.

    Args:
        populations: Inverse-sigmoid ILD activations `[samples, azimuth_bins]`.
        bins_deg: Represented azimuth bins in degrees.

    Returns:
        Tuple `(readout, trajectory, seconds_per_sample, history)`.
    """
    return distance_cann.run_line_attractor(
        populations,
        bins_deg,
        ATTRACTOR_VARIANT,
        keep_history=False,
    )


def run_cann_example(population: np.ndarray, bins_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one example through the SC line attractor with full history.

    Args:
        population: One inverse-sigmoid ILD population `[azimuth_bins]`.
        bins_deg: Represented azimuth bins.

    Returns:
        Tuple `(trajectory, excitatory_history, output_spikes)`.
    """
    _, trajectory, _, history = distance_cann.run_line_attractor(
        population[None, :],
        bins_deg,
        ATTRACTOR_VARIANT,
        keep_history=True,
    )
    if history is None:
        raise RuntimeError("Expected line-attractor history.")
    excitatory = history[:, 0, : bins_deg.size]
    spikes = distance_cann.state_history_to_output_spikes(excitatory)
    return trajectory[0], excitatory, spikes


def plot_pipeline(path: Path) -> str:
    """Plot the azimuth CANN readout pipeline."""
    fig, ax = plt.subplots(figsize=(12.8, 3.8))
    ax.axis("off")
    labels = [
        "Binaural echo",
        "Cochlea",
        "LSO/MNTB\nILD balance",
        "Inverse-sigmoid\nmapping",
        "Azimuth\npopulation",
        "SC CANN\nbalanced E/I",
        "COM readout",
    ]
    x = np.linspace(0.06, 0.94, len(labels))
    for idx, (xpos, label) in enumerate(zip(x, labels)):
        ax.text(
            xpos,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0),
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1] - 0.055, 0.55),
                xytext=(xpos + 0.055, 0.55),
                arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.2),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    return save_figure(fig, path)


def plot_matrices(bins_deg: np.ndarray, path: Path) -> str:
    """Plot the input, local recurrence, and full E/I recurrence matrices."""
    input_matrix, _, recurrent_local, recurrent = distance_cann.build_line_attractor_matrices(bins_deg, ATTRACTOR_VARIANT)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    images = [
        (input_matrix, "AC/ILD to SC input M"),
        (recurrent_local, "local recurrent kernel W0"),
        (recurrent, "balanced E/I recurrence W"),
    ]
    for ax, (matrix, title) in zip(axes, images):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("source index")
        ax.set_ylabel("target index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_prediction_scatter(
    true: np.ndarray,
    direct: np.ndarray,
    cann: np.ndarray,
    stress_true: np.ndarray,
    stress_direct: np.ndarray,
    stress_cann: np.ndarray,
    path: Path,
) -> str:
    """Plot no-CANN and CANN prediction scatters for both ranges."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=False)
    datasets = [
        ("+/-45 deg", true, direct, cann),
        ("+/-90 deg", stress_true, stress_direct, stress_cann),
    ]
    for ax, (title, y_true, y_direct, y_cann) in zip(axes, datasets):
        ax.scatter(y_true, y_direct, s=22, alpha=0.62, label="direct COM")
        ax.scatter(y_true, y_cann, s=22, alpha=0.72, label="SC CANN")
        low = float(min(y_true.min(), y_direct.min(), y_cann.min()))
        high = float(max(y_true.max(), y_direct.max(), y_cann.max()))
        ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
        ax.set_xlabel("true azimuth (deg)")
        ax.set_ylabel("predicted azimuth (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_error_over_time(
    true: np.ndarray,
    direct: np.ndarray,
    trajectory: np.ndarray,
    stress_true: np.ndarray,
    stress_direct: np.ndarray,
    stress_trajectory: np.ndarray,
    path: Path,
) -> str:
    """Plot mean azimuth error versus CANN time."""
    time_ms = np.arange(trajectory.shape[1]) * distance_cann.ATTRACTOR_DT_S * 1_000.0
    primary_error = np.mean(np.abs(trajectory - true[:, None]), axis=0)
    stress_error = np.mean(np.abs(stress_trajectory - stress_true[:, None]), axis=0)
    direct_primary = float(np.mean(np.abs(direct - true)))
    direct_stress = float(np.mean(np.abs(stress_direct - stress_true)))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    axes[0].plot(time_ms, primary_error, color="#2563eb", linewidth=2.0, label="SC CANN")
    axes[0].axhline(direct_primary, color="#111827", linestyle="--", linewidth=1.2, label="direct COM")
    axes[0].axvline(distance_cann.ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#dc2626", linestyle=":", linewidth=1.2)
    axes[0].set_title("+/-45 deg")
    axes[1].plot(time_ms, stress_error, color="#2563eb", linewidth=2.0, label="SC CANN")
    axes[1].axhline(direct_stress, color="#111827", linestyle="--", linewidth=1.2, label="direct COM")
    axes[1].axvline(distance_cann.ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#dc2626", linestyle=":", linewidth=1.2)
    axes[1].set_title("+/-90 deg")
    for ax in axes:
        ax.set_xlabel("SC attractor time (ms)")
        ax.set_ylabel("mean absolute azimuth error (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_example_dynamics(
    bins_deg: np.ndarray,
    true_azimuth: float,
    direct_population: np.ndarray,
    trajectory: np.ndarray,
    excitatory: np.ndarray,
    spikes: np.ndarray,
    path: Path,
) -> str:
    """Plot one example CANN input, dynamics, trajectory, and output spikes."""
    time_ms = np.arange(excitatory.shape[0]) * distance_cann.ATTRACTOR_DT_S * 1_000.0
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 12.0))
    axes[0].plot(bins_deg, direct_population, color="#059669", linewidth=2.0)
    axes[0].axvline(true_azimuth, color="#111827", linestyle="--", linewidth=1.2, label="true")
    axes[0].set_xlabel("represented azimuth (deg)")
    axes[0].set_ylabel("ILD activation")
    axes[0].set_title("Inverse-sigmoid ILD population injected into SC")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    im = axes[1].imshow(
        np.maximum(excitatory, 0.0).T,
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], bins_deg[0], bins_deg[-1]],
        cmap="magma",
    )
    axes[1].set_xlabel("SC time (ms)")
    axes[1].set_ylabel("represented azimuth (deg)")
    axes[1].set_title("SC line-attractor excitatory activity")
    fig.colorbar(im, ax=axes[1], label="relative excitatory rate")

    axes[2].plot(time_ms, trajectory, color="#2563eb", linewidth=2.0, label="CANN decoded")
    axes[2].axhline(true_azimuth, color="#111827", linestyle="--", linewidth=1.2, label="true")
    axes[2].axvline(distance_cann.ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#dc2626", linestyle=":", linewidth=1.2)
    axes[2].set_xlabel("SC time (ms)")
    axes[2].set_ylabel("decoded azimuth (deg)")
    axes[2].set_title("CANN readout trajectory")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    for neuron_idx in range(spikes.shape[0]):
        spike_times = time_ms[np.flatnonzero(spikes[neuron_idx] > 0.0)]
        if spike_times.size:
            axes[3].vlines(spike_times, bins_deg[neuron_idx] - 0.35, bins_deg[neuron_idx] + 0.35, color="#111827", linewidth=0.45)
    axes[3].set_xlabel("SC time (ms)")
    axes[3].set_ylabel("represented azimuth (deg)")
    axes[3].set_title("Illustrative output spikes from SC excitatory rates")
    axes[3].grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, path)


def write_report(
    config: object,
    inverse_params: dict[str, float],
    metrics: dict[str, dict[str, float]],
    runtime: dict[str, float],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the azimuth ILD line-attractor report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = [
        ("+/-45 direct inverse-sigmoid ILD", metrics["primary_direct"]),
        ("+/-45 SC CANN", metrics["primary_cann"]),
        ("+/-90 direct inverse-sigmoid ILD", metrics["stress_direct"]),
        ("+/-90 SC CANN", metrics["stress_cann"]),
    ]
    lines = [
        "# Azimuth ILD Pathway With SC Line Attractor",
        "",
        "This report adds the same optimised SC line-attractor readout used in the distance pathway to the calibrated azimuth ILD population. The lower azimuth pathway is unchanged: binaural cochlea, multi-threshold ILD coding, MNTB/LSO opponent comparison, then inverse-sigmoid synaptic mapping.",
        "",
        "![Pipeline diagram](../outputs/ild_line_attractor/figures/pipeline_diagram.png)",
        "",
        "## Model",
        "",
        "The no-attractor baseline decodes the inverse-sigmoid ILD population directly by centre of mass:",
        "",
        "$$",
        "\\hat\\theta_{COM}=\\frac{\\sum_k A_{ILD}^{inv}(\\theta_k)\\theta_k}{\\sum_k A_{ILD}^{inv}(\\theta_k)}.",
        "$$",
        "",
        "The CANN version injects that same population into the FI reflected Gaussian two-block balanced E/I line attractor:",
        "",
        "$$",
        "x(0)=s\\begin{bmatrix}M \\\\ -\\beta M\\end{bmatrix}A_{ILD}^{inv},",
        "$$",
        "",
        "$$",
        "\\tau\\dot{x}=-x+Wx,",
        "\\qquad",
        "W=\\begin{bmatrix}W_0&-W_0\\\\W_0&-W_0\\end{bmatrix}.",
        "$$",
        "",
        "The final readout is centre of mass over the rectified excitatory half of the SC state at the selected readout time.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz:.0f} Hz` |",
        f"| cochlea channels | `{az.NUM_CHANNELS}` |",
        f"| inverse-sigmoid gain `k` | `{inverse_params['gain']:.3f}` |",
        f"| inverse-sigmoid sigma | `{inverse_params['sigma']:.3f}` |",
        f"| SC attractor variant | `{ATTRACTOR_VARIANT.label}` |",
        f"| input width | `{ATTRACTOR_VARIANT.input_width_bins:.0f}` bins |",
        f"| recurrent width | `{distance_cann.ATTRACTOR_RECURRENT_WIDTH_BINS:.0f}` bins |",
        f"| beta | `{ATTRACTOR_VARIANT.beta:.3f}` |",
        f"| alpha prime | `{distance_cann.ATTRACTOR_ALPHA_PRIME:.1f}` |",
        f"| tau | `{distance_cann.ATTRACTOR_TAU_S * 1_000.0:.1f} ms` |",
        f"| readout time | `{distance_cann.ATTRACTOR_READOUT_TIME_S * 1_000.0:.1f} ms` |",
        "",
        "![Attractor matrices](../outputs/ild_line_attractor/figures/attractor_matrices.png)",
        "",
        "## Example Dynamics",
        "",
        "The example below shows the inverse-sigmoid ILD population being injected into the SC attractor, followed by the excitatory rate dynamics, decoded trajectory, and illustrative output spikes.",
        "",
        "![Example dynamics](../outputs/ild_line_attractor/figures/example_dynamics.png)",
        "",
        "## Accuracy",
        "",
        "| Readout | MAE | RMSE | Max error | Bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metric in metric_rows:
        lines.append(
            f"| {label} | `{metric['mae_deg']:.3f} deg` | `{metric['rmse_deg']:.3f} deg` | "
            f"`{metric['max_abs_error_deg']:.3f} deg` | `{metric['bias_deg']:.3f} deg` |"
        )
    lines.extend(
        [
            "",
            "![Prediction scatter](../outputs/ild_line_attractor/figures/prediction_scatter.png)",
            "",
            "![Error over time](../outputs/ild_line_attractor/figures/error_over_time.png)",
            "",
            "## Interpretation",
            "",
            "The attractor is tested as a reversible SC readout module: it receives exactly the same inverse-sigmoid ILD population as the direct COM baseline. If the CANN improves accuracy, it is sharpening or stabilising the population readout. If it does not, then the calibrated ILD population is already close to the useful decoded statistic and recurrence mainly adds smoothing/bias.",
            "",
            "This distinction matters biologically: the CANN is not a replacement for the LSO/MNTB cue computation. It is a candidate superior-colliculus style population stabiliser placed after the cue has already been mapped into azimuth space.",
            "",
            "## Runtime",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| full experiment runtime | `{elapsed_s:.2f} s` |",
            f"| CANN seconds per sample, +/-45 | `{runtime['primary_cann_seconds_per_sample']:.6f}` |",
            f"| CANN seconds per sample, +/-90 | `{runtime['stress_cann_seconds_per_sample']:.6f}` |",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the azimuth ILD line-attractor experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)

    config = az.make_config()
    primary_predictions = az.run_dataset(config, az.AZIMUTH_LIMIT_DEG)
    stress_predictions = az.run_dataset(config, az.STRESS_AZIMUTH_LIMIT_DEG)

    primary_bins = az.azimuth_grid(az.AZIMUTH_LIMIT_DEG)
    stress_bins = az.azimuth_grid(az.STRESS_AZIMUTH_LIMIT_DEG)
    inverse_params = az.tune_inverse_sigmoid_ild_mapping(primary_predictions, primary_bins, az.AZIMUTH_LIMIT_DEG)

    primary_true = np.array([item.true_azimuth_deg for item in primary_predictions], dtype=np.float64)
    stress_true = np.array([item.true_azimuth_deg for item in stress_predictions], dtype=np.float64)
    primary_direct, primary_population = inverse_ild_population_dataset(
        primary_predictions,
        primary_bins,
        inverse_params,
        az.AZIMUTH_LIMIT_DEG,
    )
    stress_direct, stress_population = inverse_ild_population_dataset(
        stress_predictions,
        stress_bins,
        inverse_params,
        az.STRESS_AZIMUTH_LIMIT_DEG,
    )

    primary_cann, primary_trajectory, primary_seconds_per_sample, _ = run_cann_readout(primary_population, primary_bins)
    stress_cann, stress_trajectory, stress_seconds_per_sample, _ = run_cann_readout(stress_population, stress_bins)

    example_index = len(primary_predictions) // 2
    example_trajectory, example_excitatory, example_spikes = run_cann_example(primary_population[example_index], primary_bins)

    metrics = {
        "primary_direct": metric_dict(primary_true, primary_direct),
        "primary_cann": metric_dict(primary_true, primary_cann),
        "stress_direct": metric_dict(stress_true, stress_direct),
        "stress_cann": metric_dict(stress_true, stress_cann),
    }
    runtime = {
        "primary_cann_seconds_per_sample": primary_seconds_per_sample,
        "stress_cann_seconds_per_sample": stress_seconds_per_sample,
    }
    artifacts = {
        "pipeline_diagram": plot_pipeline(FIGURE_DIR / "pipeline_diagram.png"),
        "attractor_matrices": plot_matrices(primary_bins, FIGURE_DIR / "attractor_matrices.png"),
        "prediction_scatter": plot_prediction_scatter(
            primary_true,
            primary_direct,
            primary_cann,
            stress_true,
            stress_direct,
            stress_cann,
            FIGURE_DIR / "prediction_scatter.png",
        ),
        "error_over_time": plot_error_over_time(
            primary_true,
            primary_direct,
            primary_trajectory,
            stress_true,
            stress_direct,
            stress_trajectory,
            FIGURE_DIR / "error_over_time.png",
        ),
        "example_dynamics": plot_example_dynamics(
            primary_bins,
            primary_true[example_index],
            primary_population[example_index],
            example_trajectory,
            example_excitatory,
            example_spikes,
            FIGURE_DIR / "example_dynamics.png",
        ),
    }

    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "azimuth_ild_line_attractor",
        "elapsed_seconds": elapsed_s,
        "inverse_sigmoid_params": inverse_params,
        "attractor_variant": {
            "key": ATTRACTOR_VARIANT.key,
            "label": ATTRACTOR_VARIANT.label,
            "input_family": ATTRACTOR_VARIANT.input_family,
            "input_width_bins": ATTRACTOR_VARIANT.input_width_bins,
            "beta": ATTRACTOR_VARIANT.beta,
            "alpha_prime": distance_cann.ATTRACTOR_ALPHA_PRIME,
            "recurrent_width_bins": distance_cann.ATTRACTOR_RECURRENT_WIDTH_BINS,
            "readout_time_s": distance_cann.ATTRACTOR_READOUT_TIME_S,
        },
        "metrics": metrics,
        "runtime": runtime,
        "primary_predictions": [
            {
                "true_azimuth_deg": float(primary_true[idx]),
                "direct_inverse_ild_deg": float(primary_direct[idx]),
                "cann_deg": float(primary_cann[idx]),
            }
            for idx in range(primary_true.size)
        ],
        "stress_predictions": [
            {
                "true_azimuth_deg": float(stress_true[idx]),
                "direct_inverse_ild_deg": float(stress_direct[idx]),
                "cann_deg": float(stress_cann[idx]),
            }
            for idx in range(stress_true.size)
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(config, inverse_params, metrics, runtime, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
