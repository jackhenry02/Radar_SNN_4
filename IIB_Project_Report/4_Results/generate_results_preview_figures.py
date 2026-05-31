from __future__ import annotations

"""Generate standalone preview figures for the results-chapter redraft.

The report is intentionally not modified. Figures are regenerated from the
saved final-model predictions and training histories.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "IIB_Project_Report" / "4_Results" / "preview_results_redraft"
FINAL_OUTPUT_DIR = ROOT / "final_model" / "outputs"

CONDITIONS = [
    ("clean", "Noise-free", "train2000_val400_test400"),
    ("noise", "Environmental noise", "train2000_val400_test400_envnoise50dB"),
    ("delayed", "Noise + delayed echoes", "train2000_val400_test400_envnoise50dB_reverb"),
]

LOSS_CONDITIONS = [CONDITIONS[0], CONDITIONS[2]]

COLORS = {
    "clean": "#2563eb",
    "noise": "#d97706",
    "delayed": "#7c3aed",
    "raw": "#2563eb",
    "calibrated": "#dc2626",
    "cann": "#059669",
    "direct": "#dc2626",
    "residual": "#059669",
}


def setup_style() -> None:
    """Apply print-readable matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.size": 16.0,
            "axes.titlesize": 17.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 14.0,
            "legend.title_fontsize": 14.0,
            "figure.titlesize": 19.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    """Save one preview figure at high resolution."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return path


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_runs() -> dict[str, dict[str, object]]:
    """Load held-out predictions and cached pathway populations."""
    runs: dict[str, dict[str, object]] = {}
    for key, label, suffix in CONDITIONS:
        pred_path = FINAL_OUTPUT_DIR / "trainable_readout" / f"test_predictions_{suffix}.npz"
        cache_path = FINAL_OUTPUT_DIR / "trainable_readout" / f"cache_constrained_0p25_5m_pm45_{suffix}.npz"
        pred = np.load(pred_path)
        cache = np.load(cache_path, allow_pickle=True)
        mask = np.asarray(cache["split_names"]) == "test"
        groups = cache["feature_groups"].item()
        features = np.asarray(cache["features"], dtype=np.float64)[mask]
        runs[key] = {
            "label": label,
            "true": np.asarray(pred["true_coordinates"], dtype=np.float64),
            "raw": np.asarray(pred["raw_coordinates"], dtype=np.float64),
            "baseline": np.asarray(pred["baseline_coordinates"], dtype=np.float64),
            "direct": np.asarray(pred["direct_coordinates"], dtype=np.float64),
            "residual": np.asarray(pred["residual_coordinates"], dtype=np.float64),
            "features": features,
            "groups": groups,
        }
    return runs


def angular_error(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return wrapped angular error in degrees."""
    return (predicted - true + 180.0) % 360.0 - 180.0


def absolute_errors(true: np.ndarray, predicted: np.ndarray) -> list[np.ndarray]:
    """Return per-coordinate absolute errors."""
    return [
        np.abs(predicted[:, 0] - true[:, 0]),
        np.abs(angular_error(predicted[:, 1], true[:, 1])),
        np.abs(angular_error(predicted[:, 2], true[:, 2])),
    ]


def combined_error(true: np.ndarray, predicted: np.ndarray) -> float:
    """Return the report's combined normalised error."""
    errors = absolute_errors(true, predicted)
    return float(np.mean((errors[0] / 5.0 + errors[1] / 45.0 + errors[2] / 45.0) / 3.0))


def centre_of_mass(populations: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Decode non-negative populations with a centre-of-mass readout."""
    positive = np.maximum(np.asarray(populations, dtype=np.float64), 0.0)
    totals = positive.sum(axis=1)
    decoded = np.full(positive.shape[0], float(bins[len(bins) // 2]), dtype=np.float64)
    valid = totals > 1e-12
    decoded[valid] = (positive[valid] * bins[None, :]).sum(axis=1) / totals[valid]
    return decoded


def inverse_sigmoid_elevation(raw_deg: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Apply the standalone elevation inverse-sigmoid calibration."""
    limit = 45.0
    normalised = np.clip((raw_deg - params["input_offset_deg"]) / limit, -0.999999, 0.999999)
    calibrated = params["output_offset_deg"] + limit * np.arctanh(normalised) / max(params["gain"], 1e-6)
    return np.clip(calibrated, -limit, limit)


def calibrated_direct_coordinates(runs: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    """Apply direct elevation calibration while retaining direct distance/ITD readouts."""
    elevation_results = load_json(ROOT / "elevation_pathway" / "outputs" / "elevation_line_attractor" / "results.json")
    params = elevation_results["inverse_sigmoid_calibration"]["direct"]
    calibrated: dict[str, np.ndarray] = {}
    for key, run in runs.items():
        coords = np.asarray(run["raw"], dtype=np.float64).copy()
        coords[:, 2] = inverse_sigmoid_elevation(coords[:, 2], params)
        calibrated[key] = coords
    return calibrated


def diagonal(ax: plt.Axes, limits: tuple[float, float]) -> None:
    """Draw the ideal prediction line."""
    ax.plot(limits, limits, color="#111827", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)


def scatter_overlay(
    true: np.ndarray,
    readouts: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    name: str,
) -> Path:
    """Overlay readout variants within one panel for each coordinate."""
    specs = [
        ("Distance", "m", 0, (0.25, 5.0)),
        ("Azimuth", "deg", 1, (-45.0, 45.0)),
        ("Elevation", "deg", 2, (-45.0, 45.0)),
    ]
    markers = ["o", "x", "^"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    legend_handles = None
    legend_labels = None
    for ax, (coordinate, unit, axis_index, limits) in zip(axes, specs):
        for readout_index, (readout_label, predicted, color) in enumerate(readouts):
            ax.scatter(
                true[:, axis_index],
                predicted[:, axis_index],
                s=17,
                alpha=0.64,
                color=color,
                marker=markers[readout_index],
                linewidths=0.65,
                label=readout_label,
            )
        diagonal(ax, limits)
        ax.set_title(coordinate)
        ax.set_xlabel(f"true {coordinate.lower()} ({unit})")
        ax.set_ylabel(f"predicted {coordinate.lower()} ({unit})")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(legend_handles, legend_labels, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncols=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return save(fig, name)


def plot_fixed_pathway_gallery(runs: dict[str, dict[str, object]], calibrated: dict[str, np.ndarray]) -> Path:
    """Plot clean and delayed-echo fixed pathways in one compact gallery."""
    specs = [
        ("Distance", "m", 0, (0.25, 5.0)),
        ("Azimuth", "deg", 1, (-45.0, 45.0)),
        ("Elevation", "deg", 2, (-45.0, 45.0)),
    ]
    markers = ["o", "x", "^"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4))
    legend_handles = None
    legend_labels = None
    for row, key in enumerate(["clean", "delayed"]):
        run = runs[key]
        readouts = [
            ("Raw COM", np.asarray(run["raw"]), COLORS["raw"]),
            ("Calibrated COM", calibrated[key], COLORS["calibrated"]),
            ("Calibrated CANN", np.asarray(run["baseline"]), COLORS["cann"]),
        ]
        for col, (coordinate, unit, axis_index, limits) in enumerate(specs):
            ax = axes[row, col]
            for readout_index, (readout_label, predicted, color) in enumerate(readouts):
                ax.scatter(
                    np.asarray(run["true"])[:, axis_index],
                    predicted[:, axis_index],
                    s=15,
                    alpha=0.62,
                    color=color,
                    marker=markers[readout_index],
                    linewidths=0.65,
                    label=readout_label,
                )
            diagonal(ax, limits)
            if row == 0:
                ax.set_title(coordinate)
            ax.set_xlabel(f"true {coordinate.lower()} ({unit})")
            prefix = f"{run['label']}\n" if col == 0 else ""
            ax.set_ylabel(f"{prefix}predicted {coordinate.lower()} ({unit})")
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle("Fixed pathway readouts")
    fig.legend(legend_handles, legend_labels, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncols=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return save(fig, "01_fixed_pathway_scatter.png")


def plot_calibration_context() -> Path:
    """Plot ILD distance dependence and elevation-calibration transfer."""
    azimuth = load_json(ROOT / "azimuth_pathway" / "outputs" / "ild_line_attractor" / "results.json")
    elevation = load_json(ROOT / "elevation_pathway" / "outputs" / "elevation_line_attractor" / "results.json")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    support = azimuth["distance_trend"]["supports"]["azimuth_pm45"]["per_distance"]
    for distance, color in [("1m", "#94a3b8"), ("2m", "#60a5fa"), ("3m", "#2563eb"), ("4m", "#7c3aed"), ("5m", "#dc2626")]:
        row = support[distance]
        axes[0].plot(
            azimuth["distance_trend"]["supports"]["azimuth_pm45"]["azimuth_deg"],
            row["direct_deg"],
            marker="o",
            markersize=2.7,
            linewidth=1.3,
            color=color,
            label=distance,
        )
    diagonal(axes[0], (-45.0, 45.0))
    axes[0].set_title("ILD calibration changes with distance")
    axes[0].set_xlabel("true azimuth (deg)")
    axes[0].set_ylabel("calibrated ILD estimate (deg)")
    axes[0].legend(title="target range", frameon=True, loc="upper left", ncols=2)

    isolated = elevation["isolated_predictions"]
    full = elevation["full_3d_predictions"]
    axes[1].scatter(
        isolated["true_elevation_deg"],
        isolated["reflected_calibrated_deg"],
        s=18,
        alpha=0.68,
        color="#2563eb",
        label="isolated elevation sweep",
    )
    axes[1].scatter(
        [row["true_elevation_deg"] for row in full],
        [row["reflected_calibrated_deg"] for row in full],
        s=17,
        alpha=0.5,
        color="#d97706",
        label="clean full-3D stress sweep",
    )
    diagonal(axes[1], (-45.0, 45.0))
    axes[1].set_title("Elevation calibration transfers in clean 3D sweep")
    axes[1].set_xlabel("true elevation (deg)")
    axes[1].set_ylabel("calibrated elevation estimate (deg)")
    axes[1].legend(frameon=True, loc="upper left")
    fig.suptitle("Calibration behaviour before environmental-noise testing")
    fig.tight_layout()
    return save(fig, "02_calibration_context.png")


def binned_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return median and interquartile range in bins."""
    centres = (bins[:-1] + bins[1:]) / 2.0
    med = np.full_like(centres, np.nan)
    q25 = np.full_like(centres, np.nan)
    q75 = np.full_like(centres, np.nan)
    for index in range(len(centres)):
        upper = x <= bins[index + 1] if index == len(centres) - 1 else x < bins[index + 1]
        mask = (x >= bins[index]) & upper
        if np.any(mask):
            med[index] = np.median(y[mask])
            q25[index] = np.percentile(y[mask], 25)
            q75[index] = np.percentile(y[mask], 75)
    return centres, med, q25, q75


def plot_fixed_failure_curves(runs: dict[str, dict[str, object]]) -> Path:
    """Plot coordinate-dependent raw fixed-pathway errors."""
    specs = [
        (0, np.linspace(0.25, 5.0, 9), "true distance (m)", "absolute distance error (m)"),
        (1, np.linspace(-45.0, 45.0, 10), "true azimuth (deg)", "absolute azimuth error (deg)"),
        (2, np.linspace(-45.0, 45.0, 10), "true elevation (deg)", "absolute elevation error (deg)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for axis_index, (coordinate, bins, xlabel, ylabel) in enumerate(specs):
        ax = axes[axis_index]
        for key, label, _ in CONDITIONS:
            run = runs[key]
            errors = absolute_errors(np.asarray(run["true"]), np.asarray(run["raw"]))[coordinate]
            centres, med, q25, q75 = binned_stats(np.asarray(run["true"])[:, coordinate], errors, bins)
            ax.plot(centres, med, marker="o", linewidth=1.8, markersize=4, color=COLORS[key], label=label)
            ax.fill_between(centres, q25, q75, color=COLORS[key], alpha=0.12, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True, loc="upper left")
    fig.suptitle("Raw fixed-pathway failure structure")
    fig.tight_layout()
    return save(fig, "03_fixed_pathway_failure_curves.png")


def plot_trained_correction_gallery(runs: dict[str, dict[str, object]]) -> Path:
    """Plot fixed, direct-SNN, and residual-SNN correction in the full noise case."""
    run = runs["delayed"]
    return scatter_overlay(
        np.asarray(run["true"]),
        [
            ("Fixed CANN", np.asarray(run["baseline"]), COLORS["raw"]),
            ("Direct SNN", np.asarray(run["direct"]), COLORS["direct"]),
            ("Residual SNN", np.asarray(run["residual"]), COLORS["residual"]),
        ],
        title=f"Trainable fusion readouts: {run['label']}",
        name="04_trained_fusion_delayed_echo_scatter.png",
    )


def plot_feature_ablation() -> Path:
    """Plot residual-SNN zero-ablation importance."""
    feature_labels = {
        "raw_distance_population": "distance\npopulation",
        "raw_azimuth_itd_population": "ITD\npopulation",
        "raw_azimuth_ild_population": "ILD\npopulation",
        "raw_elevation_population": "elevation\npopulation",
        "cann_readouts": "CANN\nscalars",
        "confidence_features": "confidence\nfeatures",
    }
    payloads = []
    for key, label, suffix in CONDITIONS:
        data = load_json(FINAL_OUTPUT_DIR / "trainable_readout" / f"results_{suffix}.json")
        payloads.append((label, data["residual_ablation_importance"], COLORS[key]))
    names = list(payloads[0][1].keys())
    x = np.arange(len(names), dtype=np.float64)
    width = 0.28
    fig, ax = plt.subplots(figsize=(10.3, 5.0))
    for index, (label, values, color) in enumerate(payloads):
        offset = (index - (len(payloads) - 1) / 2.0) * width
        ax.bar(x + offset, [values[name] for name in names], width=width, color=color, label=label)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([feature_labels[name] for name in names])
    ax.set_ylabel("increase in combined error\nafter zero-ablation")
    ax.set_title("Residual SNN dependence on pathway feature groups")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    return save(fig, "05_residual_feature_ablation.png")


def plot_residual_binned_errors(runs: dict[str, dict[str, object]]) -> Path:
    """Plot coordinate-dependent residual-SNN errors."""
    specs = [
        (0, np.linspace(0.25, 5.0, 11), "true distance (m)", "abs. distance error (m)"),
        (1, np.linspace(-45.0, 45.0, 11), "true azimuth (deg)", "abs. azimuth error (deg)"),
        (2, np.linspace(-45.0, 45.0, 11), "true elevation (deg)", "abs. elevation error (deg)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for axis_index, (coordinate, bins, xlabel, ylabel) in enumerate(specs):
        ax = axes[axis_index]
        for key, label, _ in CONDITIONS:
            run = runs[key]
            errors = absolute_errors(np.asarray(run["true"]), np.asarray(run["residual"]))[coordinate]
            centres, med, q25, q75 = binned_stats(np.asarray(run["true"])[:, coordinate], errors, bins)
            ax.plot(centres, med, marker="o", linewidth=1.8, markersize=4, color=COLORS[key], label=label)
            ax.fill_between(centres, q25, q75, color=COLORS[key], alpha=0.13, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True, loc="upper left")
    fig.tight_layout()
    return save(fig, "07_residual_binned_error_curves.png")


def plot_training_loss_curves() -> Path:
    """Plot compact training and validation comparisons for all trained models."""
    suffix = "train2000_val400_test400_envnoise50dB_reverb"
    baseline_data = load_json(FINAL_OUTPUT_DIR / "trainable_input_baselines" / f"results_{suffix}.json")
    baseline_labels = {
        "raw_waveform_direct": "waveform direct",
        "cochlear_raster_direct": "cochlear raster direct",
    }
    baseline_colors = {
        "raw_waveform_direct": "#475569",
        "cochlear_raster_direct": "#2563eb",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.0))
    ax = axes[0]
    for name, label in baseline_labels.items():
        for loss_key, loss_label, linestyle in [
            ("train_loss", "training", "--"),
            ("val_loss", "validation", "-"),
        ]:
            history = baseline_data["histories"][name]
            loss = np.asarray(history[loss_key], dtype=np.float64)
            best_epoch = int(history["best_epoch"])
            series_label = label if loss_key == "val_loss" else "_nolegend_"
            ax.plot(loss, linewidth=1.9, linestyle=linestyle, color=baseline_colors[name], label=series_label)
            if loss_key == "val_loss":
                ax.scatter([best_epoch], [loss[best_epoch]], color=baseline_colors[name], s=32, zorder=3)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("uncertainty-weighted loss")
    ax.set_title("Direct input-only baselines\nfull noise")
    ax.legend(frameon=True, loc="upper left", fontsize=13.0)

    for ax, (mode, title) in zip(axes[1:], [("direct", "Direct fusion SNN"), ("residual", "Residual fusion SNN")]):
        for key, label, suffix in LOSS_CONDITIONS:
            for loss_key, loss_label, linestyle in [
                ("train_loss", "training", "--"),
                ("val_loss", "validation", "-"),
            ]:
                data = load_json(FINAL_OUTPUT_DIR / "trainable_readout" / f"results_{suffix}.json")
                history = data["histories"][mode]
                loss = np.asarray(history[loss_key], dtype=np.float64)
                best_epoch = int(history["best_epoch"])
                series_label = label if loss_key == "val_loss" else "_nolegend_"
                ax.plot(loss, linewidth=1.9, linestyle=linestyle, color=COLORS[key], label=series_label)
                if loss_key == "val_loss":
                    ax.scatter([best_epoch], [loss[best_epoch]], color=COLORS[key], s=32, zorder=3)
        ax.set_xlabel("training epoch")
        ax.set_ylabel("uncertainty-weighted loss")
        ax.set_title(title)
        ax.legend(frameon=True, loc="upper right", fontsize=13.0)
    fig.suptitle("Training and validation loss curves")
    fig.tight_layout()
    return save(fig, "06_training_loss_curves.png")


def write_readme(paths: list[Path], runs: dict[str, dict[str, object]]) -> None:
    """Write a short inspection index."""
    lines = [
        "# Results Redraft Figure Previews",
        "",
        "These previews are generated from the final model outputs. The corresponding report assets are stored in `../redraft_figures/`.",
        "",
        "## Figures",
        "",
    ]
    descriptions = {
        "01_fixed_pathway_scatter.png": "Fixed pathway raw COM, calibrated COM, and calibrated CANN readouts. Rows compare noise-free and full-noise conditions.",
        "02_calibration_context.png": "ILD calibration changes with distance, while the elevation calibration transfers from an isolated sweep to a clean full-3D stress sweep.",
        "03_fixed_pathway_failure_curves.png": "Coordinate-dependent error structure of the raw fixed pathways for all three acoustic conditions.",
        "04_trained_fusion_delayed_echo_scatter.png": "Overlay of fixed, direct-SNN, and residual-SNN predictions with environmental noise and delayed echo copies.",
        "05_residual_feature_ablation.png": "Existing final-model zero-ablation results for all three acoustic conditions.",
        "06_training_loss_curves.png": "Training and validation losses combined into one three-panel figure. The baseline panel includes direct inputs only; dots mark retained validation epochs.",
        "07_residual_binned_error_curves.png": "Coordinate-dependent error structure of the residual SNN for all three acoustic conditions.",
    }
    for path in paths:
        lines.extend([f"- `{path.name}`: {descriptions[path.name]}", ""])

    lines.extend(
        [
            "## Calibration Figure Interpretation",
            "",
            "- Left panel: the calibrated ILD mapping changes with target range. ILD is therefore not a pure azimuth cue in this simulator. This motivates using ITD for the fixed azimuth scalar while retaining the ILD population for trainable fusion.",
            "- Right panel: the inverse-sigmoid elevation calibration fitted on the isolated sweep transfers closely to the clean full-3D stress sweep. This is an empirical result within the tested synthetic cue model; it does not establish environmental-noise robustness or general distance independence.",
            "",
            "## Suggested Distance-Pathway Follow-Up",
            "",
            "A future causal ablation should compare the full distance pathway, a version without DNLL suppression, a version without sweep facilitation, and a version without either mechanism using shared simulated waveforms. Before interpreting DNLL, revise or bypass the current first-onset VCN output: it removes late events upstream of DNLL and prevents the present implementation from isolating the suppression stage cleanly.",
            "",
            "## Training-Loss Availability",
            "",
            "The saved result JSON files contain both training and validation loss histories. Figure `06_training_loss_curves.png` combines the direct input-only baselines, direct fusion SNN, and residual fusion SNN. Dashed lines show training loss, solid lines show validation loss, and dots mark retained validation epochs. Only the noise-free and full-noise conditions are plotted for the fusion models.",
            "",
            "## Resolution Reference",
            "",
            "- Distance population: 180 bins over 0.25--5 m, giving approximately 2.65 cm spacing.",
            "- Azimuth and elevation populations: 91 bins over -45--45 degrees, giving 1 degree spacing.",
            "- Use the phrase `sub-bin precision` for errors below these spacings; this does not establish biological hyperacuity.",
            "",
            "## Combined Error Check",
            "",
        ]
    )
    for key, label, _ in CONDITIONS:
        run = runs[key]
        lines.append(
            f"- {label}: raw `{combined_error(np.asarray(run['true']), np.asarray(run['raw'])):.4f}`, "
            f"fixed CANN `{combined_error(np.asarray(run['true']), np.asarray(run['baseline'])):.4f}`, "
            f"direct SNN `{combined_error(np.asarray(run['true']), np.asarray(run['direct'])):.4f}`, "
            f"residual SNN `{combined_error(np.asarray(run['true']), np.asarray(run['residual'])):.4f}`."
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate every standalone preview."""
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-cache")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    runs = load_final_runs()
    calibrated = calibrated_direct_coordinates(runs)
    paths: list[Path] = []
    paths.append(plot_fixed_pathway_gallery(runs, calibrated))
    paths.append(plot_calibration_context())
    paths.append(plot_fixed_failure_curves(runs))
    paths.append(plot_trained_correction_gallery(runs))
    paths.append(plot_feature_ablation())
    paths.append(plot_training_loss_curves())
    paths.append(plot_residual_binned_errors(runs))
    write_readme(paths, runs)
    print(f"Generated {len(paths)} preview figures in {OUTPUT_DIR}")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
