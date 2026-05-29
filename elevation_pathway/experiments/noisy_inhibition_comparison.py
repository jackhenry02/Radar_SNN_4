from __future__ import annotations

"""Noisy elevation-pathway test for dynamic and lateral inhibition.

This is a focused diagnostic for the elevation pathway. It reuses the
standalone elevation model from ``elevation_pathway_first_attempt.py`` and
retests the wideband dynamic inhibition and Mexican-hat lateral inhibition
mechanisms under the same environmental-noise convention used by the final
model experiments.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from final_model.experiments import environment_noise_diagnostics as envdiag
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "elevation_pathway" / "outputs" / "noisy_inhibition_comparison"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "elevation_pathway" / "reports" / "noisy_inhibition_comparison.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

NUM_SAMPLES = 160
SEED = 65
ENVIRONMENT_SNR_DB_AT_CALL = 50.0


def random_targets(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw constrained-space targets used by all conditions."""
    rng = np.random.default_rng(seed)
    distances = rng.uniform(0.25, 5.0, size=num_samples)
    azimuths = rng.uniform(-45.0, 45.0, size=num_samples)
    elevations = rng.uniform(-elev.ELEVATION_LIMIT_DEG, elev.ELEVATION_LIMIT_DEG, size=num_samples)
    return distances, azimuths, elevations


def samples_for_condition(
    config,
    distances: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    *,
    condition: envdiag.AcousticCondition,
    noise_std: float,
    seed: int,
) -> list[elev.Full3DSample]:
    """Generate cached cochlea outputs for one acoustic condition."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    samples: list[elev.Full3DSample] = []
    for distance_m, azimuth_deg, elevation_deg in zip(distances, azimuths, elevations):
        if condition.add_environment_noise or condition.add_reverb:
            receive = envdiag._simulate_environment_echo(
                config,
                float(distance_m),
                float(azimuth_deg),
                float(elevation_deg),
                condition=condition,
                noise_std=noise_std,
                rng=rng,
            )
        else:
            receive = elev.simulate_full_3d_scene(
                config,
                float(distance_m),
                float(azimuth_deg),
                float(elevation_deg),
                elev.DEEP_COMB_DELAYED_COPY_GAIN,
            )

        cochlea = fdm._run_cochlea_binaural(config, receive)
        if azimuth_deg >= 0.0:
            selected_ear = "right"
            selected_cochleagram = cochlea.right_cochleagram
        else:
            selected_ear = "left"
            selected_cochleagram = cochlea.left_cochleagram
        selected_spikes = fdm._dynamic_lif_encode(selected_cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
        samples.append(
            elev.Full3DSample(
                distance_m=float(distance_m),
                azimuth_deg=float(azimuth_deg),
                elevation_deg=float(elevation_deg),
                selected_ear=selected_ear,
                selected_cochleagram=selected_cochleagram,
                selected_spikes=selected_spikes,
            )
        )
    return samples


def evaluate_named_variants(samples: list[elev.Full3DSample], config) -> dict[str, dict[str, object]]:
    """Evaluate fixed hand-selected variants before the full parameter sweep."""
    variants = {
        "baseline": (
            elev.DynamicInhibitionParams(gain=0.0, beta=0.0),
            elev.LateralInhibitionParams(gain=0.0),
        ),
        "dynamic_only_moderate": (
            elev.DynamicInhibitionParams(gain=0.5, beta=0.6),
            elev.LateralInhibitionParams(gain=0.0),
        ),
        "dynamic_only_strong": (
            elev.DynamicInhibitionParams(gain=1.0, beta=0.85),
            elev.LateralInhibitionParams(gain=0.0),
        ),
        "lateral_only_mild": (
            elev.DynamicInhibitionParams(gain=0.0, beta=0.0),
            elev.LateralInhibitionParams(gain=0.06),
        ),
        "lateral_only_strong": (
            elev.DynamicInhibitionParams(gain=0.0, beta=0.0),
            elev.LateralInhibitionParams(gain=0.3),
        ),
        "dynamic_plus_lateral": (
            elev.DynamicInhibitionParams(gain=0.5, beta=0.6),
            elev.LateralInhibitionParams(gain=0.06),
        ),
    }
    rows: dict[str, dict[str, object]] = {}
    baseline_cache: dict[tuple[float, float], np.ndarray] = {}
    for name, (dynamic_params, lateral_params) in variants.items():
        cache_key = (dynamic_params.gain, dynamic_params.beta)
        if cache_key not in baseline_cache:
            baseline_cache[cache_key] = elev.baseline_energy_profile(config, dynamic_params)
        true, pred, _ = elev.decode_full_3d_samples(
            samples,
            config,
            delayed_copy_gain=elev.DEEP_COMB_DELAYED_COPY_GAIN,
            dynamic_params=dynamic_params,
            lateral_params=lateral_params,
            baseline_profile=baseline_cache[cache_key],
        )
        rows[name] = {
            "dynamic_params": asdict(dynamic_params),
            "lateral_params": asdict(lateral_params),
            "metrics": elev.metric_dict(true, pred),
            "true": true.tolist(),
            "predicted": pred.tolist(),
        }
    return rows


def best_single_mechanism(rows: list[dict[str, float]], *, mechanism: str) -> dict[str, float]:
    """Return the best row using only one added mechanism."""
    if mechanism == "dynamic":
        candidates = [
            row
            for row in rows
            if row["dynamic_gain"] > 0.0 and row["lateral_gain"] == 0.0
        ]
    elif mechanism == "lateral":
        candidates = [
            row
            for row in rows
            if row["dynamic_gain"] == 0.0 and row["lateral_gain"] > 0.0
        ]
    else:
        candidates = [
            row
            for row in rows
            if row["dynamic_gain"] > 0.0 and row["lateral_gain"] > 0.0
        ]
    return min(candidates, key=lambda row: row["mae_deg"])


def plot_condition_bars(payload: dict[str, object], path: Path) -> str:
    """Plot MAE for baseline and best inhibition variants by condition."""
    conditions = list(payload["conditions"].keys())
    series = [
        ("baseline", "Baseline", "#111827"),
        ("best_dynamic_only", "Best dynamic", "#2563eb"),
        ("best_lateral_only", "Best lateral", "#059669"),
        ("best_combined", "Best combined", "#d97706"),
        ("best_any", "Best swept", "#7c3aed"),
    ]
    values = np.array(
        [
            [payload["conditions"][condition]["summary"][key]["mae_deg"] for key, _, _ in series]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.15
    for idx, (_, label, color) in enumerate(series):
        ax.bar(x + (idx - 2) * width, values[:, idx], width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([payload["conditions"][condition]["name"] for condition in conditions], rotation=12, ha="right")
    ax.set_ylabel("elevation MAE (deg)")
    ax.set_title("Noisy elevation-pathway inhibition sweep")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_error_curves(payload: dict[str, object], path: Path) -> str:
    """Plot binned absolute elevation error for baseline and best noisy variants."""
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bins = np.linspace(-45.0, 45.0, 10)
    centres = 0.5 * (bins[:-1] + bins[1:])
    colors = {
        "baseline": "#111827",
        "best_dynamic_only": "#2563eb",
        "best_lateral_only": "#059669",
        "best_any": "#7c3aed",
    }
    condition = payload["conditions"]["env_noise_50db"]
    for key, color in colors.items():
        if key == "baseline":
            row = condition["named_variants"]["baseline"]
        else:
            row = condition["sweep_predictions"][key]
        true = np.asarray(row["true"], dtype=np.float64)
        pred = np.asarray(row["predicted"], dtype=np.float64)
        error = np.abs(pred - true)
        medians = []
        q1 = []
        q3 = []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (true >= low) & (true < high)
            if not np.any(mask):
                medians.append(np.nan)
                q1.append(np.nan)
                q3.append(np.nan)
                continue
            medians.append(float(np.median(error[mask])))
            q1.append(float(np.percentile(error[mask], 25)))
            q3.append(float(np.percentile(error[mask], 75)))
        medians_arr = np.asarray(medians)
        q1_arr = np.asarray(q1)
        q3_arr = np.asarray(q3)
        label = key.replace("_", " ")
        ax.plot(centres, medians_arr, marker="o", linewidth=1.8, color=color, label=label)
        ax.fill_between(centres, q1_arr, q3_arr, color=color, alpha=0.16, linewidth=0.0)
    ax.set_xlabel("true elevation (deg)")
    ax.set_ylabel("absolute elevation error (deg)")
    ax.set_title("Environmental-noise elevation error by true elevation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_env_noise_scatter(payload: dict[str, object], path: Path) -> str:
    """Plot true versus predicted elevation under environmental noise."""
    condition = payload["conditions"]["env_noise_50db"]
    rows = [
        ("baseline", condition["named_variants"]["baseline"], "#111827"),
        ("best dynamic only", condition["sweep_predictions"]["best_dynamic_only"], "#2563eb"),
        ("best lateral only", condition["sweep_predictions"]["best_lateral_only"], "#059669"),
        ("best swept", condition["sweep_predictions"]["best_any"], "#7c3aed"),
    ]
    fig, axes = plt.subplots(1, len(rows), figsize=(13.2, 3.6), sharex=True, sharey=True)
    for ax, (label, row, color) in zip(axes, rows):
        true = np.asarray(row["true"], dtype=np.float64)
        pred = np.asarray(row["predicted"], dtype=np.float64)
        metrics = row.get("metrics", elev.metric_dict(true, pred))
        ax.scatter(true, pred, s=18, alpha=0.72, color=color, edgecolor="none")
        ax.plot([-45.0, 45.0], [-45.0, 45.0], color="#64748b", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#cbd5e1", linewidth=0.8)
        ax.axvline(0.0, color="#cbd5e1", linewidth=0.8)
        ax.set_title(f"{label}\nMAE={metrics['mae_deg']:.1f} deg", fontsize=10)
        ax.set_xlim(-47.0, 47.0)
        ax.set_ylim(-47.0, 47.0)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("predicted elevation (deg)")
    for ax in axes:
        ax.set_xlabel("true elevation (deg)")
    fig.suptitle("Environmental-noise true versus predicted elevation", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return save_figure(fig, path)


def write_report(payload: dict[str, object]) -> None:
    """Write a compact markdown diagnostic report."""
    lines = [
        "# Noisy inhibition comparison for the elevation pathway",
        "",
        "This diagnostic retests the two extra elevation-pathway mechanisms that were not useful in the original clean sweep: global dynamic wideband inhibition before baseline equalisation, and Mexican-hat lateral inhibition over the DCN elevation population. No model is trained here.",
        "",
        "The acoustic noise convention matches the final-model environmental-noise tests: white noise is added to the returning signal after propagation/attenuation but before head-shadow and comb-filter elevation cue application. The no-comb spectral reference remains noise-free, which is the same stable reference assumption used by the fixed elevation feature extractor.",
        "",
        f"- samples: `{payload['setup']['num_samples']}` matched random targets",
        f"- target range: `0.25--5.0 m`, azimuth `+-45 deg`, elevation `+-45 deg`",
        f"- environmental SNR at call: `{payload['setup']['environment_snr_db_at_call']:.1f} dB`",
        f"- noise std: `{payload['setup']['noise_std']:.6g}`",
        "",
        "![MAE comparison](../outputs/noisy_inhibition_comparison/figures/noisy_inhibition_mae_bars.png)",
        "",
        "![Environmental-noise error curves](../outputs/noisy_inhibition_comparison/figures/env_noise_error_curves.png)",
        "",
        "![Environmental-noise scatter](../outputs/noisy_inhibition_comparison/figures/env_noise_true_vs_predicted_scatter.png)",
        "",
        "## Summary metrics",
        "",
        "| condition | baseline MAE | best dynamic-only | best lateral-only | best combined | best swept |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, condition in payload["conditions"].items():
        summary = condition["summary"]
        lines.append(
            f"| {condition['name']} | "
            f"{summary['baseline']['mae_deg']:.3f} | "
            f"{summary['best_dynamic_only']['mae_deg']:.3f} | "
            f"{summary['best_lateral_only']['mae_deg']:.3f} | "
            f"{summary['best_combined']['mae_deg']:.3f} | "
            f"{summary['best_any']['mae_deg']:.3f} |"
        )
    lines.extend(["", "## Best parameter settings", ""])
    for key, condition in payload["conditions"].items():
        lines.append(f"### {condition['name']}")
        for label in ["best_dynamic_only", "best_lateral_only", "best_combined", "best_any"]:
            row = condition["summary"][label]
            lines.append(
                f"- `{label}`: MAE `{row['mae_deg']:.3f} deg`, RMSE `{row['rmse_deg']:.3f} deg`, "
                f"dynamic gain `{row['dynamic_gain']:.2f}`, beta `{row['dynamic_beta']:.2f}`, "
                f"lateral gain `{row['lateral_gain']:.2f}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "If the best noisy setting still leaves a large MAE, then the dominant failure is not simply a lack of population sharpening. It means the spectral cue reaching the elevation template bank has already been corrupted enough that inhibition mostly sharpens or rescales the wrong evidence. In that case the residual trainable readout is justified as a correction/fusion stage rather than as a replacement for a missing hand-tuned inhibition parameter.",
            "",
            f"- results JSON: `{RESULTS_PATH.relative_to(ROOT)}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    config = elev.make_config()
    noise_std = envdiag._call_referenced_noise_std(config, ENVIRONMENT_SNR_DB_AT_CALL)
    distances, azimuths, elevations = random_targets(NUM_SAMPLES, SEED)

    conditions = [
        envdiag.AcousticCondition("clean", "Noise-free", False, False),
        envdiag.AcousticCondition("env_noise_50db", "Environmental noise", True, False),
        envdiag.AcousticCondition("env_noise_50db_reverb", "Noise + delayed echoes", True, True),
    ]

    condition_payloads: dict[str, object] = {}
    for idx, condition in enumerate(conditions):
        samples = samples_for_condition(
            config,
            distances,
            azimuths,
            elevations,
            condition=condition,
            noise_std=noise_std,
            seed=SEED + 1000 * (idx + 1),
        )
        named = evaluate_named_variants(samples, config)
        sweep = elev.sweep_full_3d_model(samples, config, elev.DEEP_COMB_DELAYED_COPY_GAIN)
        rows = sweep["rows"]
        best_any = sweep["best"]
        best_dynamic = best_single_mechanism(rows, mechanism="dynamic")
        best_lateral = best_single_mechanism(rows, mechanism="lateral")
        best_combined = best_single_mechanism(rows, mechanism="combined")

        summary = {
            "baseline": {
                "dynamic_gain": 0.0,
                "dynamic_beta": 0.0,
                "lateral_gain": 0.0,
                **named["baseline"]["metrics"],
            },
            "best_dynamic_only": best_dynamic,
            "best_lateral_only": best_lateral,
            "best_combined": best_combined,
            "best_any": {
                "dynamic_gain": best_any["dynamic_params"].gain,
                "dynamic_beta": best_any["dynamic_params"].beta,
                "lateral_gain": best_any["lateral_params"].gain,
                **best_any["metrics"],
            },
        }

        sweep_predictions: dict[str, object] = {}
        for label, row in [
            ("best_dynamic_only", best_dynamic),
            ("best_lateral_only", best_lateral),
            ("best_combined", best_combined),
            ("best_any", summary["best_any"]),
        ]:
            dynamic_params = elev.DynamicInhibitionParams(row["dynamic_gain"], row["dynamic_beta"])
            lateral_params = elev.LateralInhibitionParams(gain=row["lateral_gain"])
            true, pred, _ = elev.decode_full_3d_samples(
                samples,
                config,
                delayed_copy_gain=elev.DEEP_COMB_DELAYED_COPY_GAIN,
                dynamic_params=dynamic_params,
                lateral_params=lateral_params,
            )
            sweep_predictions[label] = {
                "true": true.tolist(),
                "predicted": pred.tolist(),
                "metrics": elev.metric_dict(true, pred),
            }

        condition_payloads[condition.key] = {
            "name": condition.name,
            "named_variants": named,
            "summary": summary,
            "sweep_rows": rows,
            "sweep_predictions": sweep_predictions,
        }

    payload: dict[str, object] = {
        "experiment": "noisy_inhibition_comparison",
        "setup": {
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "environment_snr_db_at_call": ENVIRONMENT_SNR_DB_AT_CALL,
            "noise_std": noise_std,
            "distance_range_m": [0.25, 5.0],
            "azimuth_range_deg": [-45.0, 45.0],
            "elevation_range_deg": [-45.0, 45.0],
        },
        "conditions": condition_payloads,
    }
    payload["figures"] = {
        "mae_bars": plot_condition_bars(payload, FIGURE_DIR / "noisy_inhibition_mae_bars.png"),
        "env_noise_error_curves": plot_error_curves(payload, FIGURE_DIR / "env_noise_error_curves.png"),
        "env_noise_scatter": plot_env_noise_scatter(
            payload,
            FIGURE_DIR / "env_noise_true_vs_predicted_scatter.png",
        ),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    write_report(payload)


if __name__ == "__main__":
    main()
