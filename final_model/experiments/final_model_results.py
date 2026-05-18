from __future__ import annotations

"""Integrated final-model experiment.

This combines the three independently developed pathway prototypes:

* distance: dynamic cochlea + VCN consensus + IC/AC + reflected SC CANN;
* azimuth: Jeffress-style ITD population + reflected SC CANN;
* elevation: deep-comb signal-weighted DCN + reflected SC CANN + inverse-sigmoid calibration.

The pathways remain separate modules. This wrapper runs them on the same target
coordinates and reports coordinate errors, Euclidean error, runtime breakdowns,
and diagnostic scaling sweeps.
"""

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azimuth_pathway.experiments import azimuth_ild_line_attractor as azc
from azimuth_pathway.experiments import azimuth_pathway_first_attempt as az
from distance_pathway.experiments import final_distance_pipeline_with_attractor as dist_cann
from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_line_attractor as elc
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "final_model_results"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_RESULTS_PATH = ROOT / "final_model" / "reports" / "final_model_results.md"
REPORT_EXPLAINED_PATH = ROOT / "final_model" / "reports" / "final_model_explained.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

BASE_CHANNELS = 48
BASE_READOUT_BINS = 91
BASE_DISTANCE_BINS = 180
CONSTRAINED_SAMPLES = 24
EXPANDED_SAMPLES = 24
SCALING_SAMPLES = 8
CONSTRAINED_SEED = 321
EXPANDED_SEED = 654
SCALING_SEED = 777
CHANNEL_SWEEP = [16, 32, 48, 96]
READOUT_SWEEP = [45, 91, 181, 361]

OLD_MODEL_RESULTS = {
    "Round 3 2B + 3": {
        "combined_error": 0.0394,
        "distance_mae_m": 0.0646,
        "azimuth_mae_deg": 2.8595,
        "elevation_mae_deg": 2.5258,
        "euclidean_error_m": 0.2043,
    },
    "Round 4 combined": {
        "combined_error": 0.0435,
        "distance_mae_m": 0.0786,
        "azimuth_mae_deg": 2.8320,
        "elevation_mae_deg": 2.7802,
        "euclidean_error_m": 0.2264,
    },
    "Round 5 fixed ridge": {
        "combined_error": 0.0387,
        "distance_mae_m": 0.0438,
        "azimuth_mae_deg": 3.1077,
        "elevation_mae_deg": 2.5876,
        "euclidean_error_m": 0.2069,
    },
}


@contextmanager
def temporary_model_constants(
    *,
    channels: int,
    distance_bins: int,
    angular_bins: int,
    elevation_limit_deg: float,
) -> Iterator[None]:
    """Temporarily patch module-level constants used by pathway prototypes."""
    assignments = [
        (fdm, "NUM_CHANNELS", channels),
        (az, "NUM_CHANNELS", channels),
        (elev, "NUM_CHANNELS", channels),
        (fdm, "NUM_DISTANCE_BINS", distance_bins),
        (az, "NUM_AZIMUTH_BINS", angular_bins),
        (elev, "NUM_ELEVATION_BINS", angular_bins),
        (elev, "ELEVATION_LIMIT_DEG", elevation_limit_deg),
    ]
    old_values = [(module, name, getattr(module, name)) for module, name, _ in assignments]
    try:
        for module, name, value in assignments:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in old_values:
            setattr(module, name, value)


def angular_error_deg(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return wrapped angular error in degrees."""
    return (predicted - true + 180.0) % 360.0 - 180.0


def spherical_to_cartesian(distance_m: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Convert spherical localisation coordinates to Cartesian coordinates."""
    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.deg2rad(elevation_deg)
    x = distance_m * np.cos(el_rad) * np.cos(az_rad)
    y = distance_m * np.cos(el_rad) * np.sin(az_rad)
    z = distance_m * np.sin(el_rad)
    return np.stack([x, y, z], axis=-1)


def localisation_metrics(rows: list[dict[str, float]], max_range_m: float, az_limit_deg: float, el_limit_deg: float) -> dict[str, float]:
    """Compute coordinate and Euclidean metrics for integrated predictions."""
    true_distance = np.array([row["true_distance_m"] for row in rows], dtype=np.float64)
    pred_distance = np.array([row["pred_distance_m"] for row in rows], dtype=np.float64)
    true_az = np.array([row["true_azimuth_deg"] for row in rows], dtype=np.float64)
    pred_az = np.array([row["pred_azimuth_deg"] for row in rows], dtype=np.float64)
    true_el = np.array([row["true_elevation_deg"] for row in rows], dtype=np.float64)
    pred_el = np.array([row["pred_elevation_deg"] for row in rows], dtype=np.float64)
    distance_error = pred_distance - true_distance
    az_error = angular_error_deg(pred_az, true_az)
    el_error = angular_error_deg(pred_el, true_el)
    euclidean = np.linalg.norm(
        spherical_to_cartesian(pred_distance, pred_az, pred_el)
        - spherical_to_cartesian(true_distance, true_az, true_el),
        axis=1,
    )
    combined = (
        np.abs(distance_error) / max(max_range_m, 1e-12)
        + np.abs(az_error) / max(az_limit_deg, 1e-12)
        + np.abs(el_error) / max(el_limit_deg, 1e-12)
    ) / 3.0
    return {
        "distance_mae_m": float(np.mean(np.abs(distance_error))),
        "distance_rmse_m": float(np.sqrt(np.mean(distance_error**2))),
        "distance_bias_m": float(np.mean(distance_error)),
        "azimuth_mae_deg": float(np.mean(np.abs(az_error))),
        "azimuth_rmse_deg": float(np.sqrt(np.mean(az_error**2))),
        "azimuth_bias_deg": float(np.mean(az_error)),
        "elevation_mae_deg": float(np.mean(np.abs(el_error))),
        "elevation_rmse_deg": float(np.sqrt(np.mean(el_error**2))),
        "elevation_bias_deg": float(np.mean(el_error)),
        "euclidean_mae_m": float(np.mean(euclidean)),
        "euclidean_rmse_m": float(np.sqrt(np.mean(euclidean**2))),
        "euclidean_max_m": float(np.max(euclidean)),
        "combined_normalised_error": float(np.mean(combined)),
    }


def random_targets(
    *,
    samples: int,
    min_distance_m: float,
    max_distance_m: float,
    azimuth_limit_deg: float,
    elevation_limit_deg: float,
    seed: int,
) -> list[dict[str, float]]:
    """Create a reproducible random 3D target list."""
    rng = np.random.default_rng(seed)
    distances = rng.uniform(min_distance_m, max_distance_m, size=samples)
    # Avoid exactly zero range in the simulator while still testing the 0 m edge.
    if min_distance_m <= 0.0:
        distances[0] = 0.0
        distances = np.maximum(distances, 0.02)
        distances[0] = 0.02
    return [
        {
            "distance_m": float(distance),
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
        }
        for distance, azimuth, elevation in zip(
            distances,
            rng.uniform(-azimuth_limit_deg, azimuth_limit_deg, size=samples),
            rng.uniform(-elevation_limit_deg, elevation_limit_deg, size=samples),
        )
    ]


def make_distance_config(channels: int, min_distance_m: float, max_distance_m: float) -> fdm.GlobalConfig:
    """Create a distance-pathway config for the requested space."""
    return replace(
        fdm._make_config(),
        num_cochlea_channels=channels,
        min_range_m=max(float(min_distance_m), 0.0),
        max_range_m=float(max_distance_m),
        signal_duration_s=0.036 if max_distance_m <= 5.0 else 0.070,
        jitter_std_s=0.0,
        noise_std=0.0,
    )


def make_distance_variant(config: fdm.GlobalConfig, channels: int, min_distance_m: float, max_distance_m: float) -> fdm.PathwayVariant:
    """Create and calibrate the primary distance pathway variant."""
    template = fdm.PathwayVariant(
        key="final_dynamic_distance",
        name="Final dynamic distance pathway",
        vcn_input="spikes",
        latency_samples=np.zeros(channels, dtype=np.int64),
        dynamic_cochlea_schedule=fdm.DYNAMIC_COHLEA_SCHEDULE,
        vcn_detector="consensus",
        ic_mode="facilitated",
        note="Integrated final-model distance variant.",
    )
    calibration_distances = np.linspace(max(0.25, float(min_distance_m)), float(max_distance_m), 8)
    latency = fdm._calibrate_variant_latency(config, template, calibration_distances=calibration_distances)
    return replace(template, latency_samples=latency)


def make_azimuth_config(channels: int, max_distance_m: float) -> object:
    """Create azimuth-pathway config."""
    return replace(
        azc.make_full_3d_config(noise_std=0.0, max_distance_m=max_distance_m),
        num_cochlea_channels=channels,
        min_range_m=0.0,
        signal_duration_s=0.036 if max_distance_m <= 5.0 else 0.070,
        noise_std=0.0,
        jitter_std_s=0.0,
    )


def azimuth_readout_params() -> dict[str, object]:
    """Return metadata for the selected azimuth readout.

    The final integrated model uses the ITD branch because the standalone
    azimuth diagnostics showed it is more stable than the ILD branch in the
    constrained full-3D setup. No inverse-sigmoid calibration is needed for the
    ITD population.
    """
    return {
        "branch": "ITD",
        "readout": "reflected FI two-block SC CANN",
        "calibration": "none",
    }


def make_elevation_config(channels: int, min_distance_m: float, max_distance_m: float) -> fdm.GlobalConfig:
    """Create elevation-pathway config."""
    return replace(
        elev.make_config(),
        num_cochlea_channels=channels,
        min_range_m=max(float(min_distance_m), 0.0),
        max_range_m=float(max_distance_m),
        signal_duration_s=0.036 if max_distance_m <= 5.0 else 0.070,
        noise_std=0.0,
        jitter_std_s=0.0,
    )


def tune_elevation_calibration(config: fdm.GlobalConfig) -> tuple[dict[str, float], np.ndarray]:
    """Tune reflected-CANN inverse-sigmoid elevation calibration."""
    predictions = elev.run_dataset(config, delayed_copy_gain=elev.DEEP_COMB_DELAYED_COPY_GAIN)
    true = np.array([item.true_elevation_deg for item in predictions], dtype=np.float64)
    populations = np.stack([item.signal_weighted_activation for item in predictions], axis=0)
    bins = elev.elevation_grid()
    attractor = elc.run_attractor_variants(populations, bins)
    raw = np.asarray(attractor["reflected"]["prediction"], dtype=np.float64)
    return elc.tune_inverse_sigmoid_calibration(true, raw), elev.baseline_energy_profile(config)


def predict_distance(config: fdm.GlobalConfig, variant: fdm.PathwayVariant, distance_m: float, azimuth_deg: float, elevation_deg: float) -> tuple[float, float]:
    """Predict distance with pathway runtime."""
    start = time.perf_counter()
    prediction = fdm._predict_one_3d(config, distance_m, azimuth_deg, elevation_deg, variant, add_noise=False)
    bins = fdm._candidate_distances(config)
    cann_pred, _, _, _ = dist_cann.run_line_attractor(
        prediction.ac_activation[None, :],
        bins,
        dist_cann.SC_ATTRACTOR_VARIANTS[1],
        keep_history=False,
    )
    return float(cann_pred[0]), time.perf_counter() - start


def predict_azimuth(config: object, distance_m: float, azimuth_deg: float, elevation_deg: float, azimuth_limit_deg: float) -> tuple[float, float]:
    """Predict azimuth with pathway runtime."""
    start = time.perf_counter()
    bins = az.azimuth_grid(azimuth_limit_deg)
    prediction = azc.predict_one_3d(
        config,
        distance_m,
        azimuth_deg,
        elevation_deg,
        add_noise=False,
        limit_deg=azimuth_limit_deg,
    )
    _, population = azc.itd_population_dataset([prediction], bins)
    cann_pred, _, _, _ = azc.run_cann_readout(population, bins)
    return float(cann_pred[0]), time.perf_counter() - start


def predict_elevation(config: fdm.GlobalConfig, baseline_profile: np.ndarray, calibration_params: dict[str, float], distance_m: float, azimuth_deg: float, elevation_deg: float) -> tuple[float, float]:
    """Predict elevation with pathway runtime."""
    start = time.perf_counter()
    receive = elev.simulate_full_3d_scene(
        config,
        distance_m,
        azimuth_deg,
        elevation_deg,
        elev.DEEP_COMB_DELAYED_COPY_GAIN,
    )
    cochlea = fdm._run_cochlea_binaural(config, receive)
    selected_cochleagram = cochlea.right_cochleagram if azimuth_deg >= 0.0 else cochlea.left_cochleagram
    selected_spikes = fdm._dynamic_lif_encode(selected_cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
    profile = elev.dynamic_wideband_inhibited_profile(
        selected_cochleagram,
        selected_spikes,
        elev.DynamicInhibitionParams(),
    )
    equalized = profile / np.maximum(baseline_profile, 1e-4)
    equalized = equalized / np.maximum(equalized.max(), 1e-12)
    bins = elev.elevation_grid()
    centres_hz, gain_matrix, _ = elev.build_dcn_templates(config, bins, elev.DEEP_COMB_DELAYED_COPY_GAIN)
    population = elev.dcn_signal_weighted_transfer_response(equalized, baseline_profile, gain_matrix, centres_hz)
    attractor = elc.run_attractor_variants(population[None, :], bins)
    raw = np.asarray(attractor["reflected"]["prediction"], dtype=np.float64)
    calibrated = elc.apply_tuned_inverse_sigmoid(raw, calibration_params)
    return float(calibrated[0]), time.perf_counter() - start


def run_integrated_condition(
    *,
    name: str,
    targets: list[dict[str, float]],
    channels: int,
    distance_bins: int,
    angular_bins: int,
    min_distance_m: float,
    max_distance_m: float,
    azimuth_limit_deg: float,
    elevation_limit_deg: float,
) -> dict[str, object]:
    """Run all three final pathways on the same target set."""
    with temporary_model_constants(
        channels=channels,
        distance_bins=distance_bins,
        angular_bins=angular_bins,
        elevation_limit_deg=elevation_limit_deg,
    ):
        prep_start = time.perf_counter()
        distance_config = make_distance_config(channels, min_distance_m, max_distance_m)
        distance_variant = make_distance_variant(distance_config, channels, min_distance_m, max_distance_m)
        azimuth_config = make_azimuth_config(channels, max_distance_m)
        azimuth_params = azimuth_readout_params()
        elevation_config = make_elevation_config(channels, min_distance_m, max_distance_m)
        elevation_params, elevation_baseline = tune_elevation_calibration(elevation_config)
        prep_seconds = time.perf_counter() - prep_start

        rows: list[dict[str, float]] = []
        runtime = {"distance_s": [], "azimuth_s": [], "elevation_s": []}
        for target in targets:
            true_distance = target["distance_m"]
            true_azimuth = target["azimuth_deg"]
            true_elevation = target["elevation_deg"]
            pred_distance, distance_s = predict_distance(distance_config, distance_variant, true_distance, true_azimuth, true_elevation)
            pred_azimuth, azimuth_s = predict_azimuth(azimuth_config, true_distance, true_azimuth, true_elevation, azimuth_limit_deg)
            pred_elevation, elevation_s = predict_elevation(elevation_config, elevation_baseline, elevation_params, true_distance, true_azimuth, true_elevation)
            runtime["distance_s"].append(distance_s)
            runtime["azimuth_s"].append(azimuth_s)
            runtime["elevation_s"].append(elevation_s)
            rows.append(
                {
                    "true_distance_m": true_distance,
                    "true_azimuth_deg": true_azimuth,
                    "true_elevation_deg": true_elevation,
                    "pred_distance_m": pred_distance,
                    "pred_azimuth_deg": pred_azimuth,
                    "pred_elevation_deg": pred_elevation,
                }
            )
    runtime_summary = {key: float(np.mean(values)) for key, values in runtime.items()}
    runtime_summary["total_prediction_s"] = float(sum(runtime_summary.values()))
    return {
        "name": name,
        "channels": channels,
        "distance_bins": distance_bins,
        "angular_bins": angular_bins,
        "num_samples": len(targets),
        "prep_seconds": prep_seconds,
        "runtime_seconds_per_sample": runtime_summary,
        "metrics": localisation_metrics(rows, max_distance_m, azimuth_limit_deg, elevation_limit_deg),
        "predictions": rows,
        "calibrations": {
            "azimuth_itd_cann": azimuth_params,
            "elevation_inverse_sigmoid": elevation_params,
        },
    }


def estimate_costs(channels: int, time_samples: int, distance_bins: int, azimuth_bins: int, elevation_bins: int) -> dict[str, float]:
    """Estimate FLOPs and SOPs for one integrated prediction."""
    cochlea_flops_per_path = channels * time_samples * (9 + 4)
    distance_flops = channels * distance_bins * 8
    azimuth_flops = channels * azimuth_bins * 6
    elevation_flops = channels * elevation_bins * 8
    steps = int(round(dist_cann.ATTRACTOR_SIM_TIME_S / dist_cann.ATTRACTOR_DT_S))
    distance_cann = steps * (2 * distance_bins) ** 2 * 2
    azimuth_cann = steps * (2 * azimuth_bins) ** 2 * 2
    elevation_cann = steps * (2 * elevation_bins) ** 2 * 2
    total_flops = 3 * cochlea_flops_per_path + distance_flops + azimuth_flops + elevation_flops + distance_cann + azimuth_cann + elevation_cann
    expected_spikes = channels * time_samples * 0.06
    sops = expected_spikes + channels * distance_bins + channels * azimuth_bins + channels * elevation_bins
    return {
        "time_samples": int(time_samples),
        "estimated_flops": float(total_flops),
        "estimated_sops": float(sops),
        "cochlea_flops_per_path": float(cochlea_flops_per_path),
        "distance_readout_flops": float(distance_flops + distance_cann),
        "azimuth_readout_flops": float(azimuth_flops + azimuth_cann),
        "elevation_readout_flops": float(elevation_flops + elevation_cann),
    }


def run_channel_sweep(targets: list[dict[str, float]]) -> list[dict[str, object]]:
    """Run constrained-space channel-count sweep."""
    rows = []
    for channels in CHANNEL_SWEEP:
        condition = run_integrated_condition(
            name=f"channel_sweep_{channels}",
            targets=targets,
            channels=channels,
            distance_bins=BASE_DISTANCE_BINS,
            angular_bins=BASE_READOUT_BINS,
            min_distance_m=0.25,
            max_distance_m=5.0,
            azimuth_limit_deg=45.0,
            elevation_limit_deg=45.0,
        )
        condition["cost_estimate"] = estimate_costs(channels, int(0.036 * 64_000), BASE_DISTANCE_BINS, BASE_READOUT_BINS, BASE_READOUT_BINS)
        rows.append(condition)
    return rows


def run_readout_sweep(targets: list[dict[str, float]]) -> list[dict[str, object]]:
    """Run constrained-space readout-bin sweep."""
    rows = []
    for bins in READOUT_SWEEP:
        distance_bins = max(60, int(round(bins * 2.0)))
        condition = run_integrated_condition(
            name=f"readout_sweep_{bins}",
            targets=targets,
            channels=BASE_CHANNELS,
            distance_bins=distance_bins,
            angular_bins=bins,
            min_distance_m=0.25,
            max_distance_m=5.0,
            azimuth_limit_deg=45.0,
            elevation_limit_deg=45.0,
        )
        condition["cost_estimate"] = estimate_costs(BASE_CHANNELS, int(0.036 * 64_000), distance_bins, bins, bins)
        rows.append(condition)
    return rows


def plot_condition_scatter(condition: dict[str, object], path: Path) -> str:
    """Plot true-vs-predicted coordinates for an integrated condition."""
    rows = condition["predictions"]
    true_d = np.array([row["true_distance_m"] for row in rows])
    pred_d = np.array([row["pred_distance_m"] for row in rows])
    true_a = np.array([row["true_azimuth_deg"] for row in rows])
    pred_a = np.array([row["pred_azimuth_deg"] for row in rows])
    true_e = np.array([row["true_elevation_deg"] for row in rows])
    pred_e = np.array([row["pred_elevation_deg"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for ax, (true, pred, label) in zip(
        axes,
        [(true_d, pred_d, "distance (m)"), (true_a, pred_a, "azimuth (deg)"), (true_e, pred_e, "elevation (deg)")],
    ):
        ax.scatter(true, pred, s=26, alpha=0.75)
        low = min(float(true.min()), float(pred.min()))
        high = max(float(true.max()), float(pred.max()))
        ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"predicted {label}")
        ax.grid(True, alpha=0.25)
    fig.suptitle(str(condition["name"]))
    fig.tight_layout()
    return save_figure(fig, path)


def plot_runtime_breakdown(conditions: list[dict[str, object]], path: Path) -> str:
    """Plot pathway runtime breakdown for main conditions."""
    labels = [str(condition["name"]) for condition in conditions]
    distance = np.array([condition["runtime_seconds_per_sample"]["distance_s"] for condition in conditions]) * 1_000
    azimuth = np.array([condition["runtime_seconds_per_sample"]["azimuth_s"] for condition in conditions]) * 1_000
    elevation = np.array([condition["runtime_seconds_per_sample"]["elevation_s"] for condition in conditions]) * 1_000
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(x, distance, label="distance")
    ax.bar(x, azimuth, bottom=distance, label="azimuth")
    ax.bar(x, elevation, bottom=distance + azimuth, label="elevation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("runtime per sample (ms)")
    ax.set_title("Measured runtime breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_scaling(rows: list[dict[str, object]], x_key: str, path: Path, title: str) -> str:
    """Plot error, runtime, FLOPs, and SOPs for a sweep."""
    x = np.array([row[x_key] for row in rows], dtype=np.float64)
    distance = np.array([row["metrics"]["distance_mae_m"] for row in rows]) * 100.0
    azimuth = np.array([row["metrics"]["azimuth_mae_deg"] for row in rows])
    elevation = np.array([row["metrics"]["elevation_mae_deg"] for row in rows])
    runtime = np.array([row["runtime_seconds_per_sample"]["total_prediction_s"] for row in rows]) * 1_000.0
    flops = np.array([row["cost_estimate"]["estimated_flops"] for row in rows]) / 1e6
    sops = np.array([row["cost_estimate"]["estimated_sops"] for row in rows]) / 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    axes[0, 0].plot(x, distance, marker="o", label="distance MAE (cm)")
    axes[0, 0].plot(x, azimuth, marker="s", label="azimuth MAE (deg)")
    axes[0, 0].plot(x, elevation, marker="^", label="elevation MAE (deg)")
    axes[0, 0].set_ylabel("error")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(x, runtime, marker="o", color="#dc2626")
    axes[0, 1].set_ylabel("runtime/sample (ms)")
    axes[1, 0].plot(x, flops, marker="o", color="#2563eb")
    axes[1, 0].set_ylabel("estimated FLOPs (MFLOP)")
    axes[1, 1].plot(x, sops, marker="o", color="#16a34a")
    axes[1, 1].set_ylabel("estimated SOPs (kSOP)")
    for ax in axes.ravel():
        ax.set_xlabel(x_key.replace("_", " "))
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, path)


def format_metric_row(name: str, condition: dict[str, object]) -> str:
    """Format one metric table row."""
    m = condition["metrics"]
    rt = condition["runtime_seconds_per_sample"]["total_prediction_s"] * 1_000.0
    return (
        f"| {name} | `{condition['num_samples']}` | `{m['distance_mae_m']:.4f} m` | "
        f"`{m['azimuth_mae_deg']:.3f} deg` | `{m['elevation_mae_deg']:.3f} deg` | "
        f"`{m['euclidean_mae_m']:.4f} m` | `{m['combined_normalised_error']:.4f}` | `{rt:.2f} ms` |"
    )


def write_results_report(payload: dict[str, object], artifacts: dict[str, str]) -> None:
    """Write the final model results report."""
    constrained = payload["conditions"]["constrained"]
    expanded = payload["conditions"]["expanded"]
    lines = [
        "# Final Integrated Model Results",
        "",
        "This report combines the three independently developed pathways into one final 3D localisation wrapper. Each target is passed through the distance, azimuth, and elevation pathways, and the three coordinate predictions are interpreted as a single spherical coordinate estimate.",
        "",
        "## Final Pathway Choices",
        "",
        "- Distance: dynamic cochlear spikes, VCN consensus, DNLL suppression, IC coincidence with facilitation, AC Mexican-hat map, reflected FI two-block SC line attractor.",
        "- Azimuth: binaural cochlea, VCN onset detection, Jeffress-style ITD population, reflected FI two-block SC line attractor.",
        "- Elevation: comb-filter spectral cue, selected-ear DCN signal-weighted full-transfer population, reflected FI two-block SC line attractor at 5 ms, inverse-sigmoid elevation calibration.",
        "",
        "## Main Full 3D Tests",
        "",
        "| Condition | Samples | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined norm. error | Runtime/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        format_metric_row("Constrained: 0.25-5 m, +/-45 deg", constrained),
        format_metric_row("Expanded: 0-10 m, +/-90 deg", expanded),
        "",
        "The Euclidean error is computed after converting true and predicted spherical coordinates to Cartesian coordinates:",
        "",
        "$$",
        "x=r\\cos e\\cos a,\\quad y=r\\cos e\\sin a,\\quad z=r\\sin e.",
        "$$",
        "",
        "![Constrained predictions](../outputs/final_model_results/figures/constrained_predictions.png)",
        "",
        "![Expanded predictions](../outputs/final_model_results/figures/expanded_predictions.png)",
        "",
        "![Runtime breakdown](../outputs/final_model_results/figures/runtime_breakdown.png)",
        "",
        "## Comparison With Old Models",
        "",
        "These old values are copied from previous experiment summaries. They are not strict like-for-like comparisons because the new system is modular and separately calibrated, while the old systems used trained combined readouts on their original test setup.",
        "",
        "| Model | Combined | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in OLD_MODEL_RESULTS.items():
        lines.append(
            f"| {name} | `{values['combined_error']:.4f}` | `{values['distance_mae_m']:.4f} m` | "
            f"`{values['azimuth_mae_deg']:.3f} deg` | `{values['elevation_mae_deg']:.3f} deg` | "
            f"`{values['euclidean_error_m']:.4f} m` |"
        )
    m = constrained["metrics"]
    lines.append(
        f"| New integrated constrained model | `{m['combined_normalised_error']:.4f}` | `{m['distance_mae_m']:.4f} m` | "
        f"`{m['azimuth_mae_deg']:.3f} deg` | `{m['elevation_mae_deg']:.3f} deg` | `{m['euclidean_mae_m']:.4f} m` |"
    )
    lines.extend(
        [
            "",
            "## Frequency-Channel Scaling",
            "",
            f"This sweep uses `{SCALING_SAMPLES}` constrained-space samples. The model is recalibrated for each channel count. Measured runtime is reported alongside analytical FLOP/SOP estimates.",
            "",
            "![Channel scaling](../outputs/final_model_results/figures/channel_scaling.png)",
            "",
            "| Channels | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Runtime/sample | Est. FLOPs | Est. SOPs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["channel_sweep"]:
        m = row["metrics"]
        c = row["cost_estimate"]
        lines.append(
            f"| `{row['channels']}` | `{m['distance_mae_m']:.4f} m` | `{m['azimuth_mae_deg']:.3f} deg` | "
            f"`{m['elevation_mae_deg']:.3f} deg` | `{m['euclidean_mae_m']:.4f} m` | "
            f"`{row['runtime_seconds_per_sample']['total_prediction_s'] * 1_000.0:.2f} ms` | "
            f"`{c['estimated_flops'] / 1e6:.2f} MFLOP` | `{c['estimated_sops'] / 1e3:.2f} kSOP` |"
        )
    lines.extend(
        [
            "",
            "The leading-order FLOP estimate is:",
            "",
            "$$",
            "F \\approx 3CT(9+4) + 8CD + 6CA + 8CE + 2S[(2D)^2+(2A)^2+(2E)^2],",
            "$$",
            "",
            "where `C` is cochlear channels, `T` is samples, `D/A/E` are distance, azimuth, and elevation readout bins, and `S` is the number of CANN integration steps. The first term is three separate pathway cochleae, each with IIR and LIF work.",
            "",
            "The SOP estimate is:",
            "",
            "$$",
            "Q \\approx \\rho CT + CD + CA + CE,",
            "$$",
            "",
            "where `rho` is an assumed cochlear spike density of `0.06`. This is an event-operation proxy rather than a hardware profiler count.",
            "",
            "## Readout-Neuron Scaling",
            "",
            f"This sweep also uses `{SCALING_SAMPLES}` constrained-space samples. Angular readout bins are swept directly; distance bins are set to roughly twice the angular bin count because the distance pathway covers a metric line with finer useful resolution.",
            "",
            "![Readout scaling](../outputs/final_model_results/figures/readout_scaling.png)",
            "",
            "| Angular readout bins | Distance bins | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Runtime/sample | Est. FLOPs | Est. SOPs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["readout_sweep"]:
        m = row["metrics"]
        c = row["cost_estimate"]
        lines.append(
            f"| `{row['angular_bins']}` | `{row['distance_bins']}` | `{m['distance_mae_m']:.4f} m` | "
            f"`{m['azimuth_mae_deg']:.3f} deg` | `{m['elevation_mae_deg']:.3f} deg` | "
            f"`{m['euclidean_mae_m']:.4f} m` | `{row['runtime_seconds_per_sample']['total_prediction_s'] * 1_000.0:.2f} ms` | "
            f"`{c['estimated_flops'] / 1e6:.2f} MFLOP` | `{c['estimated_sops'] / 1e3:.2f} kSOP` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The integrated model is a first full-system assembly. It deliberately runs the three pathways separately, so the cochlea is recomputed once per pathway. This makes the runtime conservative and keeps pathway timings easy to interpret. A production version should share the binaural cochlea output and pass common spike rasters into all pathways.",
            "",
        "The constrained-space result is the meaningful tuned operating point. The expanded-space result is a stress test. It includes distances near zero, distances out to 10 m, and angular supports beyond the range for which the azimuth and elevation readouts were originally developed.",
            "",
            "The expanded test is labelled `0-10 m` because it targets the zero-range edge case. In the actual simulation, exact zero is replaced by a `0.02 m` safety floor to avoid a singular acoustic path length and inverse-square attenuation term. This should be interpreted as a near-zero stress test, not a physically meaningful target at exactly the ear origin.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.append(f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`")
    REPORT_RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_explained_report(payload: dict[str, object]) -> None:
    """Write a detailed final-model explanation report."""
    constrained = payload["conditions"]["constrained"]
    setup = payload["setup"]
    lines = [
        "# Final Integrated Model Explained",
        "",
        "This document defines the final integrated model in signal order, from target coordinate to predicted coordinate. It is intended as a parameter reference and conceptual walkthrough.",
        "",
        "## Coordinate Convention",
        "",
        "A target is represented by spherical coordinates `(r, a, e)`: distance `r` in metres, azimuth `a` in degrees, and elevation `e` in degrees. Cartesian error is computed using:",
        "",
        "$$",
        "x=r\\cos e\\cos a,\\quad y=r\\cos e\\sin a,\\quad z=r\\sin e.",
        "$$",
        "",
        "## Acoustic Signal",
        "",
        "| Quantity | Value / definition |",
        "|---|---|",
        "| sample rate | `64 kHz` |",
        "| chirp duration | `3 ms` |",
        "| chirp sweep | `18 kHz -> 2 kHz` linear down-sweep |",
        "| transmit gain | `1000x`, treated as 140 dB relative to an 80 dB amplitude-1 reference |",
        "| attenuation | `0.7 / path_length^2` in the simulator |",
        "| binaural cue | path-length ITD plus multiplicative head shadow |",
        "| elevation cue | deep comb filter with delayed-copy gain `0.99` |",
        "",
        "The emitted chirp is:",
        "",
        "$$",
        "s(t)=A w(t)\\sin\\{2\\pi[f_0t+\\tfrac{1}{2}kt^2]\\}.",
        "$$",
        "",
        "The elevation comb cue is:",
        "",
        "$$",
        "y(t)=x(t)+a x(t-\\tau),\\qquad |H(f)|=\\frac{\\sqrt{1+a^2+2a\\cos(2\\pi f\\tau)}}{1+a}.",
        "$$",
        "",
        "## Shared Cochlea",
        "",
        "Each pathway currently runs its own copy of the final IIR cochlea. This is intentionally conservative for timing; the production model should share this stage.",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| default cochlear channels | `{setup['base_channels']}` |",
        "| cochlea centre spacing | logarithmic over the matched-human band |",
        "| final Q factor | `12.0` |",
        "| dynamic threshold | `16x -> 2.5x` spike threshold |",
        "| dynamic beta | `0.20 -> 0.60` |",
        "| VCN low-frequency silence | channels below `4 kHz` ignored for distance/onset processing |",
        "",
        "The dynamic cochlear LIF is:",
        "",
        "$$",
        "v_c[t]=\\beta(t)v_c[t-1]+I_c[t],\\quad S_c[t]=\\mathbb{1}[v_c[t]\\geq \\theta(t)].",
        "$$",
        "",
        "## Distance Pathway",
        "",
        "The distance pathway estimates range from echo delay. It uses dynamic cochlear spikes, VCN consensus, DNLL late suppression, IC coincidence, AC sharpening, and SC CANN readout.",
        "",
        "$$",
        "C_k=\\sum_c \\max(0, 1+\\beta^{|t^{echo}_c-(t^{CD}_c+d_k)|}-\\theta).",
        "$$",
        "",
        "Neighbouring-channel sweep facilitation boosts candidates consistent across the FM sweep. The AC applies a Mexican-hat kernel. The SC uses the reflected FI two-block line attractor and decodes centre of mass over distance bins.",
        "",
        "## Azimuth Pathway",
        "",
        "The azimuth pathway uses the ITD branch as the final selected branch. The lower pathway runs binaural cochleae, VCN onset extraction for each ear, and a Jeffress-style candidate-delay coincidence population.",
        "",
        "$$",
        "\\Delta t_k \\approx \\frac{d_{ear}}{c}\\sin a_k,\\qquad A_k=\\sum_c \\max(0,1+\\beta^{|\\Delta n_c-\\Delta n_k|}-\\theta).",
        "$$",
        "",
        "This ITD population is injected into the reflected FI two-block SC line attractor. The ILD inverse-sigmoid pathway remains useful in the isolated azimuth sweep, but the ITD branch was selected here because it was more stable in the constrained full-3D test.",
        "",
        "## Elevation Pathway",
        "",
        "The elevation pathway is monaural after selected-ear gating by azimuth sign. The DCN template compares the observed spectrum with the expected comb transfer function.",
        "",
        "$$",
        "m_k(f_c)=P_0(f_c)(1-H_k(f_c))^2,",
        "$$",
        "",
        "$$",
        "r_k=\\exp\\left[-\\frac{\\sum_c m_k(f_c)(\\tilde p_c-H_k(f_c))^2}{2\\sigma^2}\\right].",
        "$$",
        "",
        "This DCN population is injected into the reflected FI two-block SC line attractor at a 5 ms readout time. A final inverse-sigmoid calibration corrects the stable monotonic distortion in the raw elevation readout.",
        "",
        "## SC Line Attractor",
        "",
        "All three pathway readouts use the same generic finite-line attractor form:",
        "",
        "$$",
        "\\tau\\dot{x}=-x+Wx,\\qquad x(0)=s\\begin{bmatrix}Mu\\\\-\\beta Mu\\end{bmatrix}.",
        "$$",
        "",
        "This FI-optimised input is a common readout geometry, not a pathway-specific sensory optimisation.",
        "",
        "| Attractor parameter | Value |",
        "|---|---:|",
        "| alpha prime | `4.0` |",
        "| input width | `3 bins` |",
        "| recurrent width | `4 bins` |",
        "| tau | `20 ms` |",
        "| dt | `1 ms` |",
        "| simulation time | `60 ms` |",
        "| rate cap | `55 Hz` |",
        "",
        "## Current Integrated Performance",
        "",
        f"The constrained integrated test used `{constrained['num_samples']}` samples. Its main metrics were: distance MAE `{constrained['metrics']['distance_mae_m']:.4f} m`, azimuth MAE `{constrained['metrics']['azimuth_mae_deg']:.3f} deg`, elevation MAE `{constrained['metrics']['elevation_mae_deg']:.3f} deg`, and Euclidean MAE `{constrained['metrics']['euclidean_mae_m']:.4f} m`.",
        "",
        "## Important Implementation Caveats",
        "",
        "- The three pathways currently simulate or process cochlear activity separately; sharing the cochlea should reduce runtime.",
        "- The azimuth branch uses an untuned ITD CANN readout; the elevation calibration is tuned on a controlled sweep and then reused in full 3D.",
        "- The expanded 0-10 m, +/-90 degree test is intentionally a stress test outside the main tuned operating range.",
        "- Exact zero range is numerically replaced by `0.02 m` to avoid a singular path length; the expanded test should be read as a near-zero-to-10 m stress test.",
        "- FLOPs and SOPs in the results report are analytical estimates, not hardware profiler counts.",
        "",
    ]
    REPORT_EXPLAINED_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run final integrated model experiments and reports."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_RESULTS_PATH.parent)

    constrained_targets = random_targets(
        samples=CONSTRAINED_SAMPLES,
        min_distance_m=0.25,
        max_distance_m=5.0,
        azimuth_limit_deg=45.0,
        elevation_limit_deg=45.0,
        seed=CONSTRAINED_SEED,
    )
    expanded_targets = random_targets(
        samples=EXPANDED_SAMPLES,
        min_distance_m=0.0,
        max_distance_m=10.0,
        azimuth_limit_deg=90.0,
        elevation_limit_deg=90.0,
        seed=EXPANDED_SEED,
    )
    scaling_targets = random_targets(
        samples=SCALING_SAMPLES,
        min_distance_m=0.25,
        max_distance_m=5.0,
        azimuth_limit_deg=45.0,
        elevation_limit_deg=45.0,
        seed=SCALING_SEED,
    )

    constrained = run_integrated_condition(
        name="constrained_5m_pm45",
        targets=constrained_targets,
        channels=BASE_CHANNELS,
        distance_bins=BASE_DISTANCE_BINS,
        angular_bins=BASE_READOUT_BINS,
        min_distance_m=0.25,
        max_distance_m=5.0,
        azimuth_limit_deg=45.0,
        elevation_limit_deg=45.0,
    )
    expanded = run_integrated_condition(
        name="expanded_10m_pm90",
        targets=expanded_targets,
        channels=BASE_CHANNELS,
        distance_bins=BASE_DISTANCE_BINS,
        angular_bins=181,
        min_distance_m=0.0,
        max_distance_m=10.0,
        azimuth_limit_deg=90.0,
        elevation_limit_deg=90.0,
    )
    channel_sweep = run_channel_sweep(scaling_targets)
    readout_sweep = run_readout_sweep(scaling_targets)

    artifacts = {
        "constrained_predictions": plot_condition_scatter(constrained, FIGURE_DIR / "constrained_predictions.png"),
        "expanded_predictions": plot_condition_scatter(expanded, FIGURE_DIR / "expanded_predictions.png"),
        "runtime_breakdown": plot_runtime_breakdown([constrained, expanded], FIGURE_DIR / "runtime_breakdown.png"),
        "channel_scaling": plot_scaling(channel_sweep, "channels", FIGURE_DIR / "channel_scaling.png", "Frequency-channel scaling in constrained space"),
        "readout_scaling": plot_scaling(readout_sweep, "angular_bins", FIGURE_DIR / "readout_scaling.png", "Readout-neuron scaling in constrained space"),
    }
    payload = {
        "experiment": "final_integrated_model",
        "elapsed_seconds": time.perf_counter() - start,
        "setup": {
            "base_channels": BASE_CHANNELS,
            "base_distance_bins": BASE_DISTANCE_BINS,
            "base_angular_bins": BASE_READOUT_BINS,
            "constrained_samples": CONSTRAINED_SAMPLES,
            "expanded_samples": EXPANDED_SAMPLES,
            "scaling_samples": SCALING_SAMPLES,
            "channel_sweep": CHANNEL_SWEEP,
            "readout_sweep": READOUT_SWEEP,
        },
        "conditions": {"constrained": constrained, "expanded": expanded},
        "channel_sweep": channel_sweep,
        "readout_sweep": readout_sweep,
        "old_model_results": OLD_MODEL_RESULTS,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_results_report(payload, artifacts)
    write_explained_report(payload)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_RESULTS_PATH)
    print(REPORT_EXPLAINED_PATH)
