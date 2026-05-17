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
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
FULL_3D_SAMPLES = 80
FULL_3D_MIN_DISTANCE_M = 0.25
FULL_3D_MAX_DISTANCE_M = 10.0
FULL_3D_AZIMUTH_LIMIT_DEG = 90.0
FULL_3D_AZIMUTH_LIMITS_DEG = (45.0, 90.0)
FULL_3D_ELEVATION_LIMIT_DEG = 45.0
FULL_3D_RNG_SEED = 91_000
DISTANCE_TREND_DISTANCES_M = np.arange(1.0, 10.1, 1.0)
DISTANCE_TREND_AZIMUTH_SAMPLES = 31


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


def itd_population_dataset(
    predictions: list[az.AzimuthPrediction],
    bins_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ITD populations and direct COM predictions.

    Args:
        predictions: Existing azimuth-pathway predictions.
        bins_deg: Represented azimuth bins.

    Returns:
        Pair `(direct_predictions, populations)`.
    """
    populations = np.stack([prediction.itd_activation for prediction in predictions], axis=0)
    direct = np.array([az.centre_of_mass(population, bins_deg) for population in populations], dtype=np.float64)
    return direct, populations


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


def make_full_3d_config(noise_std: float = 0.0) -> object:
    """Create a full-3D azimuth test configuration.

    Args:
        noise_std: Fixed receiver noise standard deviation.

    Returns:
        Acoustic configuration covering echoes out to 10 m.
    """
    return replace(
        az.make_config(),
        max_range_m=FULL_3D_MAX_DISTANCE_M,
        signal_duration_s=0.070,
        jitter_std_s=0.0,
        noise_std=float(noise_std),
    )


def predict_one_3d(
    config: object,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    *,
    add_noise: bool,
    limit_deg: float = FULL_3D_AZIMUTH_LIMIT_DEG,
) -> az.AzimuthPrediction:
    """Run the azimuth ILD pathway for one full-3D target.

    Args:
        config: Acoustic configuration.
        distance_m: Target distance in metres.
        azimuth_deg: Target azimuth in degrees.
        elevation_deg: Target elevation in degrees.
        add_noise: Whether to add receiver noise from `config.noise_std`.
        limit_deg: Represented azimuth range.

    Returns:
        Azimuth pathway prediction object.
    """
    bins = az.azimuth_grid(limit_deg)
    scene = az.simulate_echo_batch(
        config,
        radii_m=torch.tensor([distance_m], dtype=torch.float32),
        azimuth_deg=torch.tensor([azimuth_deg], dtype=torch.float32),
        elevation_deg=torch.tensor([elevation_deg], dtype=torch.float32),
        binaural=True,
        add_noise=add_noise,
        include_elevation_cues=True,
        transmit_gain=config.transmit_gain,
    )
    cochlea = distance_cann.fdm._run_cochlea_binaural(config, scene.receive[0].detach())
    left_spikes, right_spikes = az.run_dynamic_cochlea_spikes(cochlea, config)
    vcn_left = az.vcn_consensus_single_ear(left_spikes, config)
    vcn_right = az.vcn_consensus_single_ear(right_spikes, config)
    itd = az.jeffress_lif_itd_activation(vcn_left, vcn_right, config, bins)
    ild, left_code, right_code, left_lso, right_lso = az.lso_mntb_ild_activation(left_spikes, right_spikes, bins)
    combined = az.ITD_WEIGHT * az.normalise_population(itd) + az.ILD_WEIGHT * az.normalise_population(ild)
    return az.AzimuthPrediction(
        true_azimuth_deg=float(azimuth_deg),
        itd_prediction_deg=az.centre_of_mass(itd, bins),
        ild_prediction_deg=az.centre_of_mass(ild, bins),
        combined_prediction_deg=az.centre_of_mass(combined, bins),
        itd_activation=itd,
        ild_activation=ild,
        combined_activation=combined,
        cochlea=cochlea,
        vcn_left=vcn_left,
        vcn_right=vcn_right,
        left_level_code=left_code,
        right_level_code=right_code,
        left_lso=left_lso,
        right_lso=right_lso,
    )


def run_full_3d_condition(
    config: object,
    distances_m: np.ndarray,
    azimuths_deg: np.ndarray,
    elevations_deg: np.ndarray,
    inverse_params: dict[str, float],
    *,
    add_noise: bool,
    limit_deg: float,
) -> dict[str, object]:
    """Run one clean or noisy full-3D azimuth condition.

    Args:
        config: Acoustic configuration.
        distances_m: Target distances.
        azimuths_deg: Target azimuths.
        elevations_deg: Target elevations.
        inverse_params: Tuned inverse-sigmoid mapping parameters.
        add_noise: Whether receiver noise is enabled.
        limit_deg: Symmetric represented azimuth range.

    Returns:
        Condition results with direct and CANN predictions.
    """
    bins = az.azimuth_grid(limit_deg)
    start = time.perf_counter()
    predictions = [
        predict_one_3d(
            config,
            float(distance),
            float(azimuth_value),
            float(elevation),
            add_noise=add_noise,
            limit_deg=limit_deg,
        )
        for distance, azimuth_value, elevation in zip(distances_m, azimuths_deg, elevations_deg)
    ]
    direct, population = inverse_ild_population_dataset(
        predictions,
        bins,
        inverse_params,
        limit_deg,
    )
    cann, trajectory, cann_seconds_per_sample, _ = run_cann_readout(population, bins)
    elapsed = time.perf_counter() - start
    return {
        "predictions": predictions,
        "direct": direct,
        "population": population,
        "cann": cann,
        "trajectory": trajectory,
        "condition_seconds": elapsed,
        "seconds_per_sample": elapsed / max(1, len(predictions)),
        "cann_seconds_per_sample": cann_seconds_per_sample,
    }


def run_full_3d_suite(inverse_params: dict[str, float]) -> dict[str, object]:
    """Run clean and noisy full-3D tests for +/-45 and +/-90 azimuth supports.

    Args:
        inverse_params: Tuned inverse-sigmoid mapping parameters.

    Returns:
        Nested full-3D test results.
    """
    noise_std = distance_cann.fdm._noise_std_from_db(distance_cann.fdm.AMBIENT_NOISE_DB_SPL)
    clean_config = make_full_3d_config(noise_std=0.0)
    noisy_config = make_full_3d_config(noise_std=noise_std)
    supports: dict[str, object] = {}
    for support_index, limit_deg in enumerate(FULL_3D_AZIMUTH_LIMITS_DEG):
        rng = np.random.default_rng(FULL_3D_RNG_SEED + int(limit_deg))
        distances = rng.uniform(FULL_3D_MIN_DISTANCE_M, FULL_3D_MAX_DISTANCE_M, size=FULL_3D_SAMPLES)
        azimuths = rng.uniform(-limit_deg, limit_deg, size=FULL_3D_SAMPLES)
        elevations = rng.uniform(-FULL_3D_ELEVATION_LIMIT_DEG, FULL_3D_ELEVATION_LIMIT_DEG, size=FULL_3D_SAMPLES)

        torch.manual_seed(FULL_3D_RNG_SEED + 1000 + support_index)
        clean = run_full_3d_condition(
            clean_config,
            distances,
            azimuths,
            elevations,
            inverse_params,
            add_noise=False,
            limit_deg=limit_deg,
        )
        torch.manual_seed(FULL_3D_RNG_SEED + 2000 + support_index)
        noisy = run_full_3d_condition(
            noisy_config,
            distances,
            azimuths,
            elevations,
            inverse_params,
            add_noise=True,
            limit_deg=limit_deg,
        )
        key = f"azimuth_pm{int(limit_deg)}"
        supports[key] = {
            "limit_deg": limit_deg,
            "distance_m": distances,
            "azimuth_deg": azimuths,
            "elevation_deg": elevations,
            "clean": clean,
            "noisy": noisy,
        }
    return {
        "num_samples_per_support": FULL_3D_SAMPLES,
        "distance_range_m": [FULL_3D_MIN_DISTANCE_M, FULL_3D_MAX_DISTANCE_M],
        "azimuth_limits_deg": list(FULL_3D_AZIMUTH_LIMITS_DEG),
        "elevation_range_deg": [-FULL_3D_ELEVATION_LIMIT_DEG, FULL_3D_ELEVATION_LIMIT_DEG],
        "noise_db_spl": distance_cann.fdm.AMBIENT_NOISE_DB_SPL,
        "noise_std": noise_std,
        "reference_db_spl": distance_cann.fdm.REFERENCE_DB_SPL,
        "call_db_spl": distance_cann.fdm.CALL_DB_SPL,
        "supports": supports,
    }


def run_distance_trend_suite(inverse_params: dict[str, float]) -> dict[str, object]:
    """Run fixed-distance, zero-elevation azimuth sweeps.

    Args:
        inverse_params: Tuned inverse-sigmoid mapping parameters.

    Returns:
        Per-distance sweeps for +/-45 and +/-90 azimuth supports.
    """
    config = make_full_3d_config(noise_std=0.0)
    supports: dict[str, object] = {}
    for limit_deg in FULL_3D_AZIMUTH_LIMITS_DEG:
        azimuths = np.linspace(-limit_deg, limit_deg, DISTANCE_TREND_AZIMUTH_SAMPLES)
        per_distance: dict[str, object] = {}
        for distance_m in DISTANCE_TREND_DISTANCES_M:
            distances = np.full_like(azimuths, float(distance_m), dtype=np.float64)
            elevations = np.zeros_like(azimuths, dtype=np.float64)
            result = run_full_3d_condition(
                config,
                distances,
                azimuths,
                elevations,
                inverse_params,
                add_noise=False,
                limit_deg=limit_deg,
            )
            bins = az.azimuth_grid(limit_deg)
            itd_direct, itd_population = itd_population_dataset(result["predictions"], bins)
            itd_cann, _, itd_cann_seconds_per_sample, _ = run_cann_readout(itd_population, bins)
            per_distance[f"{distance_m:.0f}m"] = {
                "distance_m": float(distance_m),
                "azimuth_deg": azimuths,
                "direct": result["direct"],
                "cann": result["cann"],
                "itd_direct": itd_direct,
                "itd_cann": itd_cann,
                "direct_metrics": metric_dict(azimuths, result["direct"]),
                "cann_metrics": metric_dict(azimuths, result["cann"]),
                "itd_direct_metrics": metric_dict(azimuths, itd_direct),
                "itd_cann_metrics": metric_dict(azimuths, itd_cann),
                "seconds_per_sample": result["seconds_per_sample"],
                "itd_cann_seconds_per_sample": itd_cann_seconds_per_sample,
            }
        supports[f"azimuth_pm{int(limit_deg)}"] = {
            "limit_deg": float(limit_deg),
            "azimuth_deg": azimuths,
            "per_distance": per_distance,
        }
    return {
        "distances_m": DISTANCE_TREND_DISTANCES_M,
        "azimuth_samples": DISTANCE_TREND_AZIMUTH_SAMPLES,
        "elevation_deg": 0.0,
        "supports": supports,
    }


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


def plot_full_3d_results(
    azimuths_deg: np.ndarray,
    distances_m: np.ndarray,
    clean_direct: np.ndarray,
    clean_cann: np.ndarray,
    noisy_direct: np.ndarray,
    noisy_cann: np.ndarray,
    limit_deg: float,
    path: Path,
) -> str:
    """Plot clean/noisy full-3D true-vs-predicted azimuth.

    Args:
        azimuths_deg: True target azimuths.
        distances_m: True target distances.
        clean_direct: Clean direct predictions.
        clean_cann: Clean CANN predictions.
        noisy_direct: Noisy direct predictions.
        noisy_cann: Noisy CANN predictions.
        limit_deg: Symmetric azimuth support.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex="col", sharey="row")
    scatter_specs = [
        (axes[0, 0], "Clean direct COM", clean_direct),
        (axes[0, 1], "Clean SC CANN", clean_cann),
        (axes[1, 0], "50 dB noise direct COM", noisy_direct),
        (axes[1, 1], "50 dB noise SC CANN", noisy_cann),
    ]
    for ax, title, pred in scatter_specs:
        sc = ax.scatter(azimuths_deg, pred, c=distances_m, s=24, alpha=0.78, cmap="viridis")
        ax.plot([-limit_deg, limit_deg], [-limit_deg, limit_deg], color="#111827", linewidth=1.0)
        ax.axvline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_xlim(-limit_deg, limit_deg)
        ax.set_ylim(-limit_deg, limit_deg)
        ax.set_xlabel("true azimuth (deg)")
        ax.set_ylabel("predicted azimuth (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("distance (m)")
    fig.suptitle(f"Full 3D azimuth test +/-{limit_deg:.0f} deg: distance/elevation varied")
    return save_figure(fig, path)


def plot_distance_trend(
    distance_trend: dict[str, object],
    path: Path,
    *,
    direct_key: str = "direct",
    cann_key: str = "cann",
    label: str = "inverse-sigmoid ILD",
) -> str:
    """Plot fixed-distance azimuth sweeps for all 1 m distance steps.

    Args:
        distance_trend: Output of `run_distance_trend_suite`.
        path: Figure path.
        direct_key: Key for direct-readout predictions in each row.
        cann_key: Key for CANN-readout predictions in each row.
        label: Population label for titles.

    Returns:
        Saved figure path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), sharex=False, sharey=False)
    cmap = plt.get_cmap("viridis")
    distance_values = np.asarray(distance_trend["distances_m"], dtype=np.float64)
    norm = plt.Normalize(float(distance_values.min()), float(distance_values.max()))
    panels = [
        (axes[0, 0], "azimuth_pm45", direct_key, f"+/-45 {label} direct COM"),
        (axes[0, 1], "azimuth_pm45", cann_key, f"+/-45 {label} SC CANN"),
        (axes[1, 0], "azimuth_pm90", direct_key, f"+/-90 {label} direct COM"),
        (axes[1, 1], "azimuth_pm90", cann_key, f"+/-90 {label} SC CANN"),
    ]
    for ax, support_key, readout_key, title in panels:
        support = distance_trend["supports"][support_key]
        limit_deg = float(support["limit_deg"])
        ax.plot([-limit_deg, limit_deg], [-limit_deg, limit_deg], color="#111827", linewidth=1.0)
        for distance_m in distance_values:
            row = support["per_distance"][f"{distance_m:.0f}m"]
            color = cmap(norm(distance_m))
            ax.plot(
                row["azimuth_deg"],
                row[readout_key],
                color=color,
                linewidth=1.35,
                alpha=0.9,
            )
        ax.axvline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_xlim(-limit_deg, limit_deg)
        ax.set_ylim(-limit_deg, limit_deg)
        ax.set_xlabel("true azimuth (deg)")
        ax.set_ylabel("predicted azimuth (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("fixed distance (m)")
    fig.suptitle(f"{label} distance trend at elevation 0 deg: one curve per fixed target distance")
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
    full_3d: dict[str, object],
    distance_trend: dict[str, object],
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
        ("Full 3D +/-45 clean direct COM", metrics["full_3d_pm45_clean_direct"]),
        ("Full 3D +/-45 clean SC CANN", metrics["full_3d_pm45_clean_cann"]),
        ("Full 3D +/-45 50 dB noise direct COM", metrics["full_3d_pm45_noisy_direct"]),
        ("Full 3D +/-45 50 dB noise SC CANN", metrics["full_3d_pm45_noisy_cann"]),
        ("Full 3D +/-90 clean direct COM", metrics["full_3d_pm90_clean_direct"]),
        ("Full 3D +/-90 clean SC CANN", metrics["full_3d_pm90_clean_cann"]),
        ("Full 3D +/-90 50 dB noise direct COM", metrics["full_3d_pm90_noisy_direct"]),
        ("Full 3D +/-90 50 dB noise SC CANN", metrics["full_3d_pm90_noisy_cann"]),
    ]
    itd_metric_rows = [
        ("+/-45 ITD direct COM", metrics["itd_primary_direct"]),
        ("+/-45 ITD SC CANN", metrics["itd_primary_cann"]),
        ("+/-90 ITD direct COM", metrics["itd_stress_direct"]),
        ("+/-90 ITD SC CANN", metrics["itd_stress_cann"]),
        ("Full 3D +/-45 clean ITD direct COM", metrics["itd_full_3d_pm45_clean_direct"]),
        ("Full 3D +/-45 clean ITD SC CANN", metrics["itd_full_3d_pm45_clean_cann"]),
        ("Full 3D +/-45 50 dB noise ITD direct COM", metrics["itd_full_3d_pm45_noisy_direct"]),
        ("Full 3D +/-45 50 dB noise ITD SC CANN", metrics["itd_full_3d_pm45_noisy_cann"]),
        ("Full 3D +/-90 clean ITD direct COM", metrics["itd_full_3d_pm90_clean_direct"]),
        ("Full 3D +/-90 clean ITD SC CANN", metrics["itd_full_3d_pm90_clean_cann"]),
        ("Full 3D +/-90 50 dB noise ITD direct COM", metrics["itd_full_3d_pm90_noisy_direct"]),
        ("Full 3D +/-90 50 dB noise ITD SC CANN", metrics["itd_full_3d_pm90_noisy_cann"]),
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
            "## Full 3D And Noise Tests",
            "",
            f"The full test samples `{full_3d['num_samples_per_support']}` targets per azimuth support. Both tests vary distance `{full_3d['distance_range_m'][0]:.2f} -> {full_3d['distance_range_m'][1]:.2f} m` and elevation `{full_3d['elevation_range_deg'][0]:.0f} -> {full_3d['elevation_range_deg'][1]:.0f} deg`. Two azimuth ranges are tested: `+/-45 deg`, matching the current inverse-sigmoid calibration support, and `+/-90 deg`, the wide-field stress case. The acoustic simulator includes binaural head shadow, path-length ITD, and elevation spectral filtering. Only azimuth error is measured.",
            "",
            f"The noisy condition uses a fixed receiver noise floor of `{full_3d['noise_db_spl']:.0f} dB`, corresponding to `noise_std = {full_3d['noise_std']:.6g}` under the project convention where amplitude `1.0` is `{full_3d['reference_db_spl']:.0f} dB` and the `1000x` call is `{full_3d['call_db_spl']:.0f} dB`. This noise is not re-normalised per target, so farther echoes have lower effective SNR.",
            "",
            "![Full 3D +/-45 results](../outputs/ild_line_attractor/figures/full_3d_results_pm45.png)",
            "",
            "![Full 3D +/-90 results](../outputs/ild_line_attractor/figures/full_3d_results_pm90.png)",
            "",
            "The full-3D error is much larger than the controlled fixed-distance result. The most likely reason is that the inverse-sigmoid mapping was calibrated at one distance and zero elevation, so it assumes one stable relationship between LSO balance and azimuth. In the full scene, distance changes the echo level, elevation filtering changes spectral energy across channels, and the current ILD code collapses the LSO output into one global balance. Those extra variables can shift the balance even when azimuth is unchanged, so the calibrated map no longer represents azimuth alone.",
            "",
            "The 50 dB noise floor barely changes the result, which supports this interpretation: the dominant failure is not random receiver noise, but systematic cue confounding from range/elevation and spectral filtering. A stronger next ILD model should either normalise level/spectrum before the balance calculation, use frequency-dependent LSO populations instead of one global balance, or learn/tune a multidimensional mapping conditioned on distance/elevation-sensitive context.",
            "",
            "## Fixed-Distance Trend Test",
            "",
            f"To isolate the distance effect, the next test fixes elevation at `{distance_trend['elevation_deg']:.0f} deg` and sweeps azimuth at constant distances from `{float(distance_trend['distances_m'][0]):.0f}` to `{float(distance_trend['distances_m'][-1]):.0f} m` in `1 m` steps. The curves are plotted on the same axes so distance-dependent compression, expansion, or saturation can be seen directly.",
            "",
            "![Distance trend](../outputs/ild_line_attractor/figures/distance_trend.png)",
            "",
            "## ITD Swap Test",
            "",
            "The same diagnostics were repeated with the ILD population swapped out for the ITD population from the Jeffress/LIF branch. This tests whether the timing cue is more stable across distance/elevation than the calibrated ILD cue. The readout comparison is kept the same: direct COM over the ITD population versus the same SC CANN.",
            "",
            "| Readout | MAE | RMSE | Max error | Bias |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, metric in itd_metric_rows:
        lines.append(
            f"| {label} | `{metric['mae_deg']:.3f} deg` | `{metric['rmse_deg']:.3f} deg` | "
            f"`{metric['max_abs_error_deg']:.3f} deg` | `{metric['bias_deg']:.3f} deg` |"
        )
    lines.extend(
        [
            "",
            "![ITD full 3D +/-45 results](../outputs/ild_line_attractor/figures/itd_full_3d_results_pm45.png)",
            "",
            "![ITD full 3D +/-90 results](../outputs/ild_line_attractor/figures/itd_full_3d_results_pm90.png)",
            "",
            "![ITD distance trend](../outputs/ild_line_attractor/figures/itd_distance_trend.png)",
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
            f"| ITD CANN seconds per sample, +/-45 | `{runtime['itd_primary_cann_seconds_per_sample']:.6f}` |",
            f"| ITD CANN seconds per sample, +/-90 | `{runtime['itd_stress_cann_seconds_per_sample']:.6f}` |",
            f"| full 3D +/-45 clean seconds per sample | `{runtime['full_3d_pm45_clean_seconds_per_sample']:.6f}` |",
            f"| full 3D +/-45 noisy seconds per sample | `{runtime['full_3d_pm45_noisy_seconds_per_sample']:.6f}` |",
            f"| full 3D +/-90 clean seconds per sample | `{runtime['full_3d_pm90_clean_seconds_per_sample']:.6f}` |",
            f"| full 3D +/-90 noisy seconds per sample | `{runtime['full_3d_pm90_noisy_seconds_per_sample']:.6f}` |",
            f"| fixed-distance trend runtime | `{runtime['distance_trend_seconds']:.2f} s` |",
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
    primary_itd_direct, primary_itd_population = itd_population_dataset(primary_predictions, primary_bins)
    stress_itd_direct, stress_itd_population = itd_population_dataset(stress_predictions, stress_bins)

    primary_cann, primary_trajectory, primary_seconds_per_sample, _ = run_cann_readout(primary_population, primary_bins)
    stress_cann, stress_trajectory, stress_seconds_per_sample, _ = run_cann_readout(stress_population, stress_bins)
    primary_itd_cann, _, primary_itd_seconds_per_sample, _ = run_cann_readout(primary_itd_population, primary_bins)
    stress_itd_cann, _, stress_itd_seconds_per_sample, _ = run_cann_readout(stress_itd_population, stress_bins)
    full_3d = run_full_3d_suite(inverse_params)
    distance_trend_start = time.perf_counter()
    distance_trend = run_distance_trend_suite(inverse_params)
    distance_trend_seconds = time.perf_counter() - distance_trend_start

    example_index = len(primary_predictions) // 2
    example_trajectory, example_excitatory, example_spikes = run_cann_example(primary_population[example_index], primary_bins)

    metrics = {
        "primary_direct": metric_dict(primary_true, primary_direct),
        "primary_cann": metric_dict(primary_true, primary_cann),
        "stress_direct": metric_dict(stress_true, stress_direct),
        "stress_cann": metric_dict(stress_true, stress_cann),
        "itd_primary_direct": metric_dict(primary_true, primary_itd_direct),
        "itd_primary_cann": metric_dict(primary_true, primary_itd_cann),
        "itd_stress_direct": metric_dict(stress_true, stress_itd_direct),
        "itd_stress_cann": metric_dict(stress_true, stress_itd_cann),
    }
    runtime = {
        "primary_cann_seconds_per_sample": primary_seconds_per_sample,
        "stress_cann_seconds_per_sample": stress_seconds_per_sample,
        "itd_primary_cann_seconds_per_sample": primary_itd_seconds_per_sample,
        "itd_stress_cann_seconds_per_sample": stress_itd_seconds_per_sample,
    }
    for limit_deg in FULL_3D_AZIMUTH_LIMITS_DEG:
        support_key = f"azimuth_pm{int(limit_deg)}"
        metric_key = f"full_3d_pm{int(limit_deg)}"
        support = full_3d["supports"][support_key]
        true_azimuth = support["azimuth_deg"]
        clean = support["clean"]
        noisy = support["noisy"]
        bins = az.azimuth_grid(limit_deg)
        clean_itd_direct, clean_itd_population = itd_population_dataset(clean["predictions"], bins)
        noisy_itd_direct, noisy_itd_population = itd_population_dataset(noisy["predictions"], bins)
        clean_itd_cann, _, clean_itd_cann_seconds_per_sample, _ = run_cann_readout(clean_itd_population, bins)
        noisy_itd_cann, _, noisy_itd_cann_seconds_per_sample, _ = run_cann_readout(noisy_itd_population, bins)
        metrics[f"{metric_key}_clean_direct"] = metric_dict(true_azimuth, clean["direct"])
        metrics[f"{metric_key}_clean_cann"] = metric_dict(true_azimuth, clean["cann"])
        metrics[f"{metric_key}_noisy_direct"] = metric_dict(true_azimuth, noisy["direct"])
        metrics[f"{metric_key}_noisy_cann"] = metric_dict(true_azimuth, noisy["cann"])
        metrics[f"itd_{metric_key}_clean_direct"] = metric_dict(true_azimuth, clean_itd_direct)
        metrics[f"itd_{metric_key}_clean_cann"] = metric_dict(true_azimuth, clean_itd_cann)
        metrics[f"itd_{metric_key}_noisy_direct"] = metric_dict(true_azimuth, noisy_itd_direct)
        metrics[f"itd_{metric_key}_noisy_cann"] = metric_dict(true_azimuth, noisy_itd_cann)
        clean["itd_direct"] = clean_itd_direct
        clean["itd_cann"] = clean_itd_cann
        noisy["itd_direct"] = noisy_itd_direct
        noisy["itd_cann"] = noisy_itd_cann
        runtime[f"{metric_key}_clean_seconds_per_sample"] = clean["seconds_per_sample"]
        runtime[f"{metric_key}_noisy_seconds_per_sample"] = noisy["seconds_per_sample"]
        runtime[f"{metric_key}_clean_cann_seconds_per_sample"] = clean["cann_seconds_per_sample"]
        runtime[f"{metric_key}_noisy_cann_seconds_per_sample"] = noisy["cann_seconds_per_sample"]
        runtime[f"itd_{metric_key}_clean_cann_seconds_per_sample"] = clean_itd_cann_seconds_per_sample
        runtime[f"itd_{metric_key}_noisy_cann_seconds_per_sample"] = noisy_itd_cann_seconds_per_sample
    runtime["distance_trend_seconds"] = distance_trend_seconds
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
        "full_3d_results_pm45": plot_full_3d_results(
            full_3d["supports"]["azimuth_pm45"]["azimuth_deg"],
            full_3d["supports"]["azimuth_pm45"]["distance_m"],
            full_3d["supports"]["azimuth_pm45"]["clean"]["direct"],
            full_3d["supports"]["azimuth_pm45"]["clean"]["cann"],
            full_3d["supports"]["azimuth_pm45"]["noisy"]["direct"],
            full_3d["supports"]["azimuth_pm45"]["noisy"]["cann"],
            45.0,
            FIGURE_DIR / "full_3d_results_pm45.png",
        ),
        "full_3d_results_pm90": plot_full_3d_results(
            full_3d["supports"]["azimuth_pm90"]["azimuth_deg"],
            full_3d["supports"]["azimuth_pm90"]["distance_m"],
            full_3d["supports"]["azimuth_pm90"]["clean"]["direct"],
            full_3d["supports"]["azimuth_pm90"]["clean"]["cann"],
            full_3d["supports"]["azimuth_pm90"]["noisy"]["direct"],
            full_3d["supports"]["azimuth_pm90"]["noisy"]["cann"],
            90.0,
            FIGURE_DIR / "full_3d_results_pm90.png",
        ),
        "distance_trend": plot_distance_trend(distance_trend, FIGURE_DIR / "distance_trend.png"),
        "itd_full_3d_results_pm45": plot_full_3d_results(
            full_3d["supports"]["azimuth_pm45"]["azimuth_deg"],
            full_3d["supports"]["azimuth_pm45"]["distance_m"],
            full_3d["supports"]["azimuth_pm45"]["clean"]["itd_direct"],
            full_3d["supports"]["azimuth_pm45"]["clean"]["itd_cann"],
            full_3d["supports"]["azimuth_pm45"]["noisy"]["itd_direct"],
            full_3d["supports"]["azimuth_pm45"]["noisy"]["itd_cann"],
            45.0,
            FIGURE_DIR / "itd_full_3d_results_pm45.png",
        ),
        "itd_full_3d_results_pm90": plot_full_3d_results(
            full_3d["supports"]["azimuth_pm90"]["azimuth_deg"],
            full_3d["supports"]["azimuth_pm90"]["distance_m"],
            full_3d["supports"]["azimuth_pm90"]["clean"]["itd_direct"],
            full_3d["supports"]["azimuth_pm90"]["clean"]["itd_cann"],
            full_3d["supports"]["azimuth_pm90"]["noisy"]["itd_direct"],
            full_3d["supports"]["azimuth_pm90"]["noisy"]["itd_cann"],
            90.0,
            FIGURE_DIR / "itd_full_3d_results_pm90.png",
        ),
        "itd_distance_trend": plot_distance_trend(
            distance_trend,
            FIGURE_DIR / "itd_distance_trend.png",
            direct_key="itd_direct",
            cann_key="itd_cann",
            label="ITD",
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
        "full_3d": {
            "num_samples_per_support": full_3d["num_samples_per_support"],
            "distance_range_m": full_3d["distance_range_m"],
            "azimuth_limits_deg": full_3d["azimuth_limits_deg"],
            "elevation_range_deg": full_3d["elevation_range_deg"],
            "noise_db_spl": full_3d["noise_db_spl"],
            "noise_std": full_3d["noise_std"],
            "reference_db_spl": full_3d["reference_db_spl"],
            "call_db_spl": full_3d["call_db_spl"],
            "supports": {
                key: {
                    "limit_deg": value["limit_deg"],
                    "distance_m": value["distance_m"].tolist(),
                    "azimuth_deg": value["azimuth_deg"].tolist(),
                    "elevation_deg": value["elevation_deg"].tolist(),
                    "clean_direct_deg": value["clean"]["direct"].tolist(),
                    "clean_cann_deg": value["clean"]["cann"].tolist(),
                    "clean_itd_direct_deg": value["clean"]["itd_direct"].tolist(),
                    "clean_itd_cann_deg": value["clean"]["itd_cann"].tolist(),
                    "noisy_direct_deg": value["noisy"]["direct"].tolist(),
                    "noisy_cann_deg": value["noisy"]["cann"].tolist(),
                    "noisy_itd_direct_deg": value["noisy"]["itd_direct"].tolist(),
                    "noisy_itd_cann_deg": value["noisy"]["itd_cann"].tolist(),
                }
                for key, value in full_3d["supports"].items()
            },
        },
        "distance_trend": {
            "distances_m": np.asarray(distance_trend["distances_m"]).tolist(),
            "azimuth_samples": distance_trend["azimuth_samples"],
            "elevation_deg": distance_trend["elevation_deg"],
            "supports": {
                support_key: {
                    "limit_deg": support["limit_deg"],
                    "azimuth_deg": support["azimuth_deg"].tolist(),
                    "per_distance": {
                        distance_key: {
                            "distance_m": row["distance_m"],
                            "direct_deg": row["direct"].tolist(),
                            "cann_deg": row["cann"].tolist(),
                            "itd_direct_deg": row["itd_direct"].tolist(),
                            "itd_cann_deg": row["itd_cann"].tolist(),
                            "direct_metrics": row["direct_metrics"],
                            "cann_metrics": row["cann_metrics"],
                            "itd_direct_metrics": row["itd_direct_metrics"],
                            "itd_cann_metrics": row["itd_cann_metrics"],
                            "seconds_per_sample": row["seconds_per_sample"],
                            "itd_cann_seconds_per_sample": row["itd_cann_seconds_per_sample"],
                        }
                        for distance_key, row in support["per_distance"].items()
                    },
                }
                for support_key, support in distance_trend["supports"].items()
            },
        },
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
    write_report(config, inverse_params, metrics, full_3d, distance_trend, runtime, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
