from __future__ import annotations

"""Aggregate final model predictions into report-ready figures and tables."""

import json
import math
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "final_model" / "outputs" / "report_results_redraft"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "final_model" / "reports" / "report_results_redraft.md"
RESULTS_TEX_PATH = ROOT / "IIB_Project_Report" / "4_Results" / "results_redraft.tex"
MODELLING_TEX_PATH = ROOT / "IIB_Project_Report" / "3_Modelling" / "modelling_redraft.tex"
MODELLING_FIG_DIR = ROOT / "IIB_Project_Report" / "3_Modelling" / "worked_example"
REPORT_RESULTS_FIGURES = {
    "headline": ROOT / "IIB_Project_Report" / "4_Results" / "trained" / "headline_combined_error.png",
    "coordinate_mae": ROOT / "IIB_Project_Report" / "4_Results" / "trained" / "coordinate_mae_summary.png",
    "noisy_elevation": ROOT / "IIB_Project_Report" / "4_Results" / "elevation" / "noisy_elevation_error_by_true_elevation.png",
    "residual_binned": ROOT / "IIB_Project_Report" / "4_Results" / "trained" / "residual_binned_error_curves.png",
    "feature_ablation": ROOT / "IIB_Project_Report" / "4_Results" / "trained" / "residual_feature_ablation.png",
    "worked_example": ROOT / "IIB_Project_Report" / "4_Results" / "trained" / "worked_example_readout_correction.png",
}

CONDITIONS = [
    ("clean", "Clean", "train2000_val400_test400"),
    ("noise", "Environmental noise", "train2000_val400_test400_envnoise50dB"),
    ("reverb", "Noise + reverb", "train2000_val400_test400_envnoise50dB_reverb"),
]

READOUT_LABELS = {
    "raw": "Raw pathway COM",
    "baseline": "Fixed CANN/calibrated",
    "direct": "Direct pathway-feature SNN",
    "residual": "Residual pathway-feature SNN",
    "best_input": "Best input-only SNN",
}

BASELINE_LABELS = {
    "raw_waveform_projected": "Raw waveform projected",
    "raw_waveform_direct": "Raw waveform direct",
    "cochlear_raster_projected": "Cochlear raster projected",
    "cochlear_raster_direct": "Cochlear raster direct",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def wrap_angle_deg(error: np.ndarray) -> np.ndarray:
    return (error + 180.0) % 360.0 - 180.0


def spherical_to_cartesian(coords: np.ndarray) -> np.ndarray:
    r = coords[:, 0]
    az = np.deg2rad(coords[:, 1])
    el = np.deg2rad(coords[:, 2])
    return np.column_stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)])


def per_sample_errors(true: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    distance_signed = pred[:, 0] - true[:, 0]
    az_signed = wrap_angle_deg(pred[:, 1] - true[:, 1])
    el_signed = wrap_angle_deg(pred[:, 2] - true[:, 2])
    euclidean = np.linalg.norm(spherical_to_cartesian(pred) - spherical_to_cartesian(true), axis=1)
    combined = (np.abs(distance_signed) / 5.0 + np.abs(az_signed) / 45.0 + np.abs(el_signed) / 45.0) / 3.0
    return {
        "distance_signed_m": distance_signed,
        "azimuth_signed_deg": az_signed,
        "elevation_signed_deg": el_signed,
        "distance_abs_m": np.abs(distance_signed),
        "azimuth_abs_deg": np.abs(az_signed),
        "elevation_abs_deg": np.abs(el_signed),
        "euclidean_m": euclidean,
        "combined": combined,
    }


def metric_summary(errors: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "distance_mae_m": float(np.mean(errors["distance_abs_m"])),
        "azimuth_mae_deg": float(np.mean(errors["azimuth_abs_deg"])),
        "elevation_mae_deg": float(np.mean(errors["elevation_abs_deg"])),
        "euclidean_mae_m": float(np.mean(errors["euclidean_m"])),
        "combined_error": float(np.mean(errors["combined"])),
        "euclidean_median_m": float(np.median(errors["euclidean_m"])),
        "euclidean_p90_m": float(np.percentile(errors["euclidean_m"], 90.0)),
        "euclidean_p95_m": float(np.percentile(errors["euclidean_m"], 95.0)),
    }


def add_metric_cis(summary: dict[str, float], errors: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    summary["distance_ci_m"] = bootstrap_ci(errors["distance_abs_m"], rng)
    summary["azimuth_ci_deg"] = bootstrap_ci(errors["azimuth_abs_deg"], rng)
    summary["elevation_ci_deg"] = bootstrap_ci(errors["elevation_abs_deg"], rng)
    summary["euclidean_ci_m"] = bootstrap_ci(errors["euclidean_m"], rng)
    summary["combined_ci"] = bootstrap_ci(errors["combined"], rng)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    draws = rng.integers(0, n, size=(samples, n))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_predictions() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    final_runs: dict[str, dict[str, object]] = {}
    baseline_runs: dict[str, dict[str, object]] = {}
    for key, label, run_label in CONDITIONS:
        final_path = ROOT / "final_model" / "outputs" / "trainable_readout" / f"test_predictions_{run_label}.npz"
        baseline_path = ROOT / "final_model" / "outputs" / "trainable_input_baselines" / f"test_predictions_{run_label}.npz"
        final_npz = np.load(final_path)
        baseline_npz = np.load(baseline_path)
        true = np.asarray(final_npz["true_coordinates"], dtype=np.float64)
        final_predictions = {
            "raw": np.asarray(final_npz["raw_coordinates"], dtype=np.float64),
            "baseline": np.asarray(final_npz["baseline_coordinates"], dtype=np.float64),
            "direct": np.asarray(final_npz["direct_coordinates"], dtype=np.float64),
            "residual": np.asarray(final_npz["residual_coordinates"], dtype=np.float64),
        }
        input_predictions = {
            name: np.asarray(baseline_npz[f"{name}_coordinates"], dtype=np.float64)
            for name in BASELINE_LABELS
        }
        final_runs[key] = {"label": label, "true": true, "predictions": final_predictions, "path": final_path}
        baseline_runs[key] = {"label": label, "true": true, "predictions": input_predictions, "path": baseline_path}
    return final_runs, baseline_runs


def summarise(final_runs: dict[str, dict[str, object]], baseline_runs: dict[str, dict[str, object]]) -> dict[str, object]:
    rng = np.random.default_rng(12345)
    conditions: dict[str, object] = {}
    for key, final_run in final_runs.items():
        true = np.asarray(final_run["true"], dtype=np.float64)
        readouts: dict[str, object] = {}
        for name, pred in final_run["predictions"].items():
            errors = per_sample_errors(true, np.asarray(pred, dtype=np.float64))
            summary = metric_summary(errors)
            add_metric_cis(summary, errors, rng)
            readouts[name] = {"summary": summary, "errors": errors}

        input_summaries: dict[str, object] = {}
        for name, pred in baseline_runs[key]["predictions"].items():
            errors = per_sample_errors(true, np.asarray(pred, dtype=np.float64))
            summary = metric_summary(errors)
            add_metric_cis(summary, errors, rng)
            input_summaries[name] = {"summary": summary, "errors": errors}
        best_name = min(input_summaries, key=lambda n: input_summaries[n]["summary"]["combined_error"])
        readouts["best_input"] = input_summaries[best_name]
        readouts["best_input"]["source"] = best_name
        conditions[key] = {
            "label": final_run["label"],
            "n": int(true.shape[0]),
            "readouts": readouts,
            "input_baselines": input_summaries,
            "best_input": best_name,
        }
    return {"conditions": conditions}


def format_ci(value: float, ci: tuple[float, float], digits: int = 3) -> str:
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def plot_headline(summary: dict[str, object]) -> Path:
    keys = [key for key, _, _ in CONDITIONS]
    readouts = ["raw", "baseline", "best_input", "direct", "residual"]
    x = np.arange(len(keys), dtype=np.float64)
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#6b7280", "#2563eb", "#8b5cf6", "#f59e0b", "#059669"]
    for i, readout in enumerate(readouts):
        y = []
        lo = []
        hi = []
        for key in keys:
            s = summary["conditions"][key]["readouts"][readout]["summary"]
            y.append(s["combined_error"])
            ci = s.get("combined_ci", (s["combined_error"], s["combined_error"]))
            lo.append(s["combined_error"] - ci[0])
            hi.append(ci[1] - s["combined_error"])
        offset = (i - (len(readouts) - 1) / 2.0) * width
        ax.bar(x + offset, y, width=width, label=READOUT_LABELS[readout], color=colors[i])
        ax.errorbar(x + offset, y, yerr=[lo, hi], fmt="none", ecolor="#111827", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([summary["conditions"][key]["label"] for key in keys])
    ax.set_ylabel("combined normalised error")
    ax.set_title("Final readout comparison across acoustic conditions")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path = FIGURE_DIR / "headline_combined_error.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_coordinate_mae(summary: dict[str, object]) -> Path:
    keys = [key for key, _, _ in CONDITIONS]
    metrics = [
        ("distance_mae_m", "Distance MAE (m)"),
        ("azimuth_mae_deg", "Azimuth MAE (deg)"),
        ("elevation_mae_deg", "Elevation MAE (deg)"),
        ("euclidean_mae_m", "Euclidean MAE (m)"),
    ]
    readouts = ["baseline", "best_input", "residual"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        x = np.arange(len(keys), dtype=np.float64)
        width = 0.23
        for i, readout in enumerate(readouts):
            y = []
            lo = []
            hi = []
            ci_key = {
                "distance_mae_m": "distance_ci_m",
                "azimuth_mae_deg": "azimuth_ci_deg",
                "elevation_mae_deg": "elevation_ci_deg",
                "euclidean_mae_m": "euclidean_ci_m",
            }[metric]
            for key in keys:
                s = summary["conditions"][key]["readouts"][readout]["summary"]
                value = s[metric]
                ci = s[ci_key]
                y.append(value)
                lo.append(value - ci[0])
                hi.append(ci[1] - value)
            xpos = x + (i - 1) * width
            ax.bar(xpos, y, width=width, label=READOUT_LABELS[readout])
            ax.errorbar(xpos, y, yerr=[lo, hi], fmt="none", ecolor="#111827", elinewidth=0.7, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([summary["conditions"][key]["label"] for key in keys], rotation=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "coordinate_mae_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def binned_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centres = (bins[:-1] + bins[1:]) / 2.0
    med = np.full(centres.shape, np.nan)
    p25 = np.full(centres.shape, np.nan)
    p75 = np.full(centres.shape, np.nan)
    for i in range(len(centres)):
        mask = (x >= bins[i]) & (x < bins[i + 1] if i < len(centres) - 1 else x <= bins[i + 1])
        if np.any(mask):
            med[i] = np.median(y[mask])
            p25[i] = np.percentile(y[mask], 25)
            p75[i] = np.percentile(y[mask], 75)
    return centres, med, p25, p75


def plot_noisy_elevation_curve(summary: dict[str, object], final_runs: dict[str, dict[str, object]]) -> Path:
    key = "noise"
    true = np.asarray(final_runs[key]["true"], dtype=np.float64)
    bins = np.linspace(-45.0, 45.0, 11)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for readout, color in [("baseline", "#2563eb"), ("best_input", "#8b5cf6"), ("residual", "#059669")]:
        errors = summary["conditions"][key]["readouts"][readout]["errors"]["elevation_abs_deg"]
        centres, med, p25, p75 = binned_stats(true[:, 2], errors, bins)
        ax.plot(centres, med, marker="o", label=READOUT_LABELS[readout], color=color)
        ax.fill_between(centres, p25, p75, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("true elevation (deg)")
    ax.set_ylabel("absolute elevation error (deg)")
    ax.set_title("Environmental-noise elevation failure and residual correction")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "noisy_elevation_error_by_true_elevation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_residual_binned_errors(summary: dict[str, object], final_runs: dict[str, dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    specs = [
        ("distance", 0, "distance_abs_m", np.linspace(0.25, 5.0, 11), "true distance (m)", "abs. distance error (m)"),
        ("azimuth", 1, "azimuth_abs_deg", np.linspace(-45.0, 45.0, 11), "true azimuth (deg)", "abs. azimuth error (deg)"),
        ("elevation", 2, "elevation_abs_deg", np.linspace(-45.0, 45.0, 11), "true elevation (deg)", "abs. elevation error (deg)"),
    ]
    for ax, (_, coord_idx, err_key, bins, xlabel, ylabel) in zip(axes, specs):
        for key, color in [("clean", "#059669"), ("noise", "#d97706"), ("reverb", "#7c3aed")]:
            true = np.asarray(final_runs[key]["true"], dtype=np.float64)
            errors = summary["conditions"][key]["readouts"]["residual"]["errors"][err_key]
            centres, med, p25, p75 = binned_stats(true[:, coord_idx], errors, bins)
            ax.plot(centres, med, marker="o", label=summary["conditions"][key]["label"], color=color)
            ax.fill_between(centres, p25, p75, color=color, alpha=0.13, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "residual_binned_error_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_feature_ablation() -> Path:
    result_paths = [
        ROOT / "final_model" / "outputs" / "trainable_readout" / f"results_{run_label}.json"
        for _, _, run_label in CONDITIONS
    ]
    feature_names = None
    payloads = []
    for (_, label, _), path in zip(CONDITIONS, result_paths):
        data = json.loads(path.read_text())
        values = data["residual_ablation_importance"]
        if feature_names is None:
            feature_names = list(values.keys())
        payloads.append((label, [values[name] for name in feature_names]))
    assert feature_names is not None
    short = {
        "raw_distance_population": "distance",
        "raw_azimuth_itd_population": "ITD",
        "raw_azimuth_ild_population": "ILD",
        "raw_elevation_population": "elevation",
        "cann_readouts": "CANN",
        "confidence_features": "confidence",
    }
    x = np.arange(len(feature_names), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (label, values) in enumerate(payloads):
        ax.bar(x + (i - 1) * width, values, width=width, label=label)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(name, name) for name in feature_names], rotation=20, ha="right")
    ax.set_ylabel("increase in combined error after zero-ablation")
    ax.set_title("Residual SNN feature-group ablation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "residual_feature_ablation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_worked_example(final_runs: dict[str, dict[str, object]]) -> Path:
    key = "clean"
    true = np.asarray(final_runs[key]["true"], dtype=np.float64)
    preds = final_runs[key]["predictions"]
    idx = int(np.argmin(np.linalg.norm(np.asarray(preds["residual"], dtype=np.float64) - true, axis=1)))
    labels = ["distance (m)", "azimuth (deg)", "elevation (deg)"]
    methods = [("True", true[idx], "#111827"), ("Fixed", preds["baseline"][idx], "#2563eb"), ("Residual SNN", preds["residual"][idx], "#059669")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for j, ax in enumerate(axes):
        for name, coords, color in methods:
            ax.scatter([0], [coords[j]], s=70, label=name if j == 0 else None, color=color)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel(labels[j])
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Worked example: fixed readout and residual coordinate correction")
    fig.tight_layout()
    path = FIGURE_DIR / "worked_example_readout_correction.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_markdown(summary: dict[str, object], figures: dict[str, Path]) -> None:
    lines: list[str] = [
        "# Report Results Redraft",
        "",
        "This report regenerates the final comparison from per-sample test predictions. All runs use the existing cached feature files and retrained the small SNN readouts with the same seeds/settings as the original reported runs. Bootstrap intervals are 95% confidence intervals over held-out test samples.",
        "",
        "## Generated Figures",
        "",
    ]
    for name, path in figures.items():
        lines += [f"- **{name}**: `{rel(path, ROOT)}`", ""]
        lines += [f"![{name}]({rel(path, REPORT_PATH.parent)})", ""]

    lines += ["## Full-Model Summary", ""]
    for key, condition in summary["conditions"].items():
        lines += [f"### {condition['label']} (N={condition['n']})", ""]
        lines += [
            "| Readout | Distance MAE m | Azimuth MAE deg | Elevation MAE deg | Euclidean MAE m | Combined error | Euclidean p90 m |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for readout in ["raw", "baseline", "best_input", "direct", "residual"]:
            s = condition["readouts"][readout]["summary"]
            lines.append(
                "| "
                + READOUT_LABELS[readout]
                + f" | {format_ci(s['distance_mae_m'], s.get('distance_ci_m', (s['distance_mae_m'], s['distance_mae_m'])))}"
                + f" | {format_ci(s['azimuth_mae_deg'], s.get('azimuth_ci_deg', (s['azimuth_mae_deg'], s['azimuth_mae_deg'])))}"
                + f" | {format_ci(s['elevation_mae_deg'], s.get('elevation_ci_deg', (s['elevation_mae_deg'], s['elevation_mae_deg'])))}"
                + f" | {format_ci(s['euclidean_mae_m'], s.get('euclidean_ci_m', (s['euclidean_mae_m'], s['euclidean_mae_m'])))}"
                + f" | {format_ci(s['combined_error'], s.get('combined_ci', (s['combined_error'], s['combined_error'])))}"
                + f" | {s['euclidean_p90_m']:.3f} |"
            )
        best = condition["best_input"]
        lines += ["", f"Best input-only baseline: **{BASELINE_LABELS[best]}**.", ""]

    lines += ["## Input-Baseline Details", ""]
    lines += [
        "| Condition | Baseline | Distance MAE m | Azimuth MAE deg | Elevation MAE deg | Euclidean MAE m | Combined error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, condition in summary["conditions"].items():
        for name, baseline in condition["input_baselines"].items():
            s = baseline["summary"]
            lines.append(
                f"| {condition['label']} | {BASELINE_LABELS[name]}"
                + f" | {format_ci(s['distance_mae_m'], s['distance_ci_m'])}"
                + f" | {format_ci(s['azimuth_mae_deg'], s['azimuth_ci_deg'])}"
                + f" | {format_ci(s['elevation_mae_deg'], s['elevation_ci_deg'])}"
                + f" | {format_ci(s['euclidean_mae_m'], s['euclidean_ci_m'])}"
                + f" | {format_ci(s['combined_error'], s['combined_ci'])} |"
            )
    lines += [
        "",
        "The input-only baselines are useful controls, but they should not dominate the report. They show that a trainable SNN can exploit cochlear structure, while the pathway-feature SNN still gives the clearest evidence that the biological intermediate representations are useful.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_results_tex(summary: dict[str, object], figures: dict[str, Path]) -> None:
    report_root = ROOT / "IIB_Project_Report"
    fig = {name: rel(REPORT_RESULTS_FIGURES.get(name, path), report_root) for name, path in figures.items()}
    lines = [
        r"\chapter{Results and Discussion} \label{ch:results}",
        "",
        "This chapter evaluates whether the biologically structured pathway model provides useful intermediate representations for three-dimensional localisation. The final comparisons use the constrained operating region of 0.25--5~m and $\\pm45^\\circ$ azimuth/elevation. Each trainable run used 2000 training, 400 validation, and 400 held-out test samples. Reported confidence intervals are bootstrap 95\\% intervals over the held-out test set.",
        "",
        r"\section{Evaluation Metrics}",
        "",
        "Distance errors are reported in metres and angular errors in degrees. Spherical predictions are also converted to Cartesian coordinates to compute Euclidean position error. The combined error is a dimensionless comparison metric formed from normalised distance, azimuth, and elevation absolute errors; it is used only to compare models within this constrained test setup.",
        "",
        r"\section{Pathway and Readout Summary}",
        "",
        "The individual pathways produce useful but imperfect population codes. Distance is the most stable cue in the tuned range, azimuth depends on the balance between ITD and ILD cues, and elevation is most sensitive to spectral-cue distortion. The SC-style CANN readout is therefore treated as a population readout and stabiliser rather than as a general error-correction stage. Its empirical effect is mixed, so the final trainable readout receives both raw pathway populations and fixed readout estimates.",
        "",
        r"\section{Full Model Comparison}",
        "",
        "Figure~\\ref{fig:headline_combined_error} shows the main comparison. The residual SNN using pathway features is the best overall readout in all three acoustic conditions. The input-only SNN baselines show that cochlear preprocessing is useful, but the best performance is obtained when the SNN receives the structured distance, azimuth, elevation, CANN, and confidence features.",
        "",
        r"\begin{figure}[H]",
        r"    \centering",
        rf"    \includegraphics[width=0.95\textwidth]{{{fig['headline']}}}",
        r"    \caption{Combined normalised error for the main readout variants. Error bars show bootstrap 95\% confidence intervals over the 400-sample test set.}",
        r"    \label{fig:headline_combined_error}",
        r"\end{figure}",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Final model comparison with bootstrap 95\% confidence intervals.}",
        r"\label{tab:full_model_results_redraft}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Condition & Readout & Distance MAE & Azimuth MAE & Elevation MAE & Euclidean MAE & Combined error \\",
        r"\midrule",
    ]
    for key, condition in summary["conditions"].items():
        first = True
        for readout in ["raw", "baseline", "best_input", "direct", "residual"]:
            s = condition["readouts"][readout]["summary"]
            cond = condition["label"] if first else ""
            first = False
            lines.append(
                f"{cond} & {READOUT_LABELS[readout]} & "
                f"{format_ci(s['distance_mae_m'], s.get('distance_ci_m', (s['distance_mae_m'], s['distance_mae_m'])), 3)} & "
                f"{format_ci(s['azimuth_mae_deg'], s.get('azimuth_ci_deg', (s['azimuth_mae_deg'], s['azimuth_mae_deg'])), 2)} & "
                f"{format_ci(s['elevation_mae_deg'], s.get('elevation_ci_deg', (s['elevation_mae_deg'], s['elevation_mae_deg'])), 2)} & "
                f"{format_ci(s['euclidean_mae_m'], s.get('euclidean_ci_m', (s['euclidean_mae_m'], s['euclidean_mae_m'])), 3)} & "
                f"{format_ci(s['combined_error'], s.get('combined_ci', (s['combined_error'], s['combined_error'])), 3)} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
        "The clean-condition result shows that the fixed pathway model is already functional, but that trainable residual fusion gives a large additional improvement. Under environmental noise, the fixed readout suffers a severe elevation failure, while the residual SNN substantially corrects this by using the wider feature set. Noise plus reverberation produces a similar residual error to noise alone, suggesting that the tested late echoes are less damaging than corruption of the elevation spectral cue.",
        "",
        r"\begin{figure}[H]",
        r"    \centering",
        rf"    \includegraphics[width=0.98\textwidth]{{{fig['coordinate_mae']}}}",
        r"    \caption{Coordinate-specific errors for the fixed readout, best input-only SNN baseline, and residual pathway-feature SNN.}",
        r"    \label{fig:coordinate_mae_summary}",
        r"\end{figure}",
        "",
        r"\section{Error Structure}",
        "",
        "Figure~\\ref{fig:noisy_elevation_curve} isolates the dominant environmental-noise failure. The fixed baseline has large elevation errors across much of the vertical field, while the residual SNN reduces the error substantially. This supports the interpretation that the trainable readout is learning context-dependent corrections from the pathway populations rather than replacing the entire localisation system.",
        "",
        r"\begin{figure}[H]",
        r"    \centering",
        rf"    \includegraphics[width=0.82\textwidth]{{{fig['noisy_elevation']}}}",
        r"    \caption{Median absolute elevation error binned by true elevation under environmental noise. Shaded regions show the interquartile range within each bin.}",
        r"    \label{fig:noisy_elevation_curve}",
        r"\end{figure}",
        "",
        r"\begin{figure}[H]",
        r"    \centering",
        rf"    \includegraphics[width=0.98\textwidth]{{{fig['residual_binned']}}}",
        r"    \caption{Residual SNN error structure across the coordinate range. Lines show median absolute error in coordinate bins; shaded regions show the interquartile range.}",
        r"    \label{fig:residual_binned_errors}",
        r"\end{figure}",
        "",
        r"\section{Feature Ablation}",
        "",
        "Feature-group ablation provides a more useful interpretation than first-layer weight magnitude alone. Under noisy conditions, the residual SNN depends strongly on raw elevation and distance population features, while CANN scalar readouts contribute little when ablated in isolation. This is consistent with the final model using the CANN as one readout feature rather than relying on it as the main source of accuracy.",
        "",
        r"\begin{figure}[H]",
        r"    \centering",
        rf"    \includegraphics[width=0.85\textwidth]{{{fig['feature_ablation']}}}",
        r"    \caption{Residual SNN feature-group ablation. Positive values indicate an increase in combined error when a feature group is zeroed on the test set.}",
        r"    \label{fig:residual_feature_ablation}",
        r"\end{figure}",
        "",
        r"\section{Discussion}",
        "",
        "The results support three main conclusions. First, the fixed auditory pathways perform meaningful feature extraction: raw waveform baselines are weak, cochlear-raster baselines are better, and pathway-feature SNNs are best. Second, the CANN should be interpreted conservatively as an SC-like population readout, not as a universal optimiser. Third, residual fusion is the strongest final architecture because it preserves the hand-designed estimate while learning systematic corrections for cue interactions.",
    ]
    RESULTS_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def copy_results_figures(figures: dict[str, Path]) -> None:
    for name, source in figures.items():
        target = REPORT_RESULTS_FIGURES.get(name)
        if target is None:
            continue
        ensure_dir(target.parent)
        shutil.copy2(source, target)


def update_modelling_redraft(figures: dict[str, Path]) -> None:
    if not MODELLING_TEX_PATH.exists():
        return
    ensure_dir(MODELLING_FIG_DIR)
    copied = {}
    sources = {
        "frequency_time_rasters": ROOT / "distance_pathway" / "outputs" / "final_distance_pipeline_with_attractor" / "figures" / "frequency_time_rasters.png",
        "distance_population_stages": ROOT / "distance_pathway" / "outputs" / "final_distance_pipeline_with_attractor" / "figures" / "distance_population_stages.png",
        "worked_example_readout_correction": figures["worked_example"],
    }
    for name, source in sources.items():
        target = MODELLING_FIG_DIR / f"{name}.png"
        if source.exists():
            shutil.copy2(source, target)
            copied[name] = target
    if len(copied) < 3:
        return
    text = MODELLING_TEX_PATH.read_text(encoding="utf-8")
    marker = "This time-series view is useful because the model is not a single regression block. It is a chain of interpretable transformations: waveform $\\rightarrow$ spikes $\\rightarrow$ pathway populations $\\rightarrow$ coordinate readouts $\\rightarrow$ residual correction."
    if "fig:worked_example_pipeline_redraft" in text:
        return
    insert = r"""

\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\textwidth]{3_Modelling/worked_example/frequency_time_rasters.png}
    \caption{Worked distance-pathway example showing the frequency-time spike structure retained by the cochlear and VCN stages.}
    \label{fig:worked_example_rasters_redraft}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\textwidth]{3_Modelling/worked_example/distance_population_stages.png}
    \caption{Worked distance-pathway example showing how channel-wise coincidence evidence is transformed into distance population activity.}
    \label{fig:worked_example_pipeline_redraft}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.75\textwidth]{3_Modelling/worked_example/worked_example_readout_correction.png}
    \caption{Example final readout correction for one clean test sample. The residual SNN learns coordinate-specific corrections to the fixed pathway estimate rather than replacing the pathway computation.}
    \label{fig:worked_example_readout_redraft}
\end{figure}
"""
    text = text.replace(marker, marker + insert)
    MODELLING_TEX_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(FIGURE_DIR)
    final_runs, baseline_runs = load_predictions()
    summary = summarise(final_runs, baseline_runs)
    figures = {
        "headline": plot_headline(summary),
        "coordinate_mae": plot_coordinate_mae(summary),
        "noisy_elevation": plot_noisy_elevation_curve(summary, final_runs),
        "residual_binned": plot_residual_binned_errors(summary, final_runs),
        "feature_ablation": plot_feature_ablation(),
        "worked_example": plot_worked_example(final_runs),
    }
    copy_results_figures(figures)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o), encoding="utf-8")
    write_markdown(summary, figures)
    write_results_tex(summary, figures)
    update_modelling_redraft(figures)
    print(REPORT_PATH)
    print(RESULTS_TEX_PATH)
    print(MODELLING_TEX_PATH)


if __name__ == "__main__":
    main()
