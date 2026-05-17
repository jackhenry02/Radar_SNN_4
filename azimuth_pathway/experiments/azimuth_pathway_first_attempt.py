from __future__ import annotations

"""First standalone azimuth-pathway prototype.

This experiment keeps the old model untouched and builds a new interpretable
azimuth pathway around the final cochlea developed for the distance pathway.
The prototype separates ITD and ILD computations after the binaural cochlea,
then combines their topographic azimuth populations with a centre-of-mass
readout.
"""

import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import full_distance_pathway_model as fdm
from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.common.signals import moving_notch_signal_config
from models.acoustics import simulate_echo_batch
from utils.common import GlobalConfig


OUTPUT_DIR = ROOT / "azimuth_pathway" / "outputs" / "first_attempt"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "azimuth_pathway" / "reports" / "azimuth_pathway_first_attempt.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

NUM_CHANNELS = 48
NUM_AZIMUTH_BINS = 181
TEST_SAMPLES = 96
FIXED_DISTANCE_M = 3.0
AZIMUTH_LIMIT_DEG = 45.0
STRESS_AZIMUTH_LIMIT_DEG = 90.0
RNG_SEED = 63
VCN_MIN_RESPONSIVE_HZ = 4_000.0
ILD_THRESHOLD_COUNT = 8
ILD_INHIBITION_GAIN = 0.72
ILD_TUNING_SIGMA = 0.20
ITD_LIF_BETA = 0.72
ITD_LIF_THRESHOLD = 1.10
ITD_WEIGHT = 0.90
ILD_WEIGHT = 0.10

OLD_MODEL_AZIMUTH_RESULTS = {
    "Round 3 2B + 3": 2.8595,
    "Round 4 combined": 2.8320,
    "Round 4 LSO/MNTB ILD experiment": 3.1207,
}


@dataclass(frozen=True)
class AzimuthPrediction:
    """Stage outputs for one azimuth-pathway prediction.

    Attributes:
        true_azimuth_deg: True target azimuth.
        itd_prediction_deg: ITD-only centre-of-mass prediction.
        ild_prediction_deg: ILD-only centre-of-mass prediction.
        combined_prediction_deg: Combined IC/SC centre-of-mass prediction.
        itd_activation: ITD topographic activation over azimuth bins.
        ild_activation: ILD topographic activation over azimuth bins.
        combined_activation: Combined IC activation over azimuth bins.
        cochlea: Binaural cochlea result.
        vcn_left: Left-ear VCN onset raster.
        vcn_right: Right-ear VCN onset raster.
        left_level_code: Multi-threshold left-ear ILD code `[channels]`.
        right_level_code: Multi-threshold right-ear ILD code `[channels]`.
        left_lso: Left LSO opponent response `[channels]`.
        right_lso: Right LSO opponent response `[channels]`.
    """

    true_azimuth_deg: float
    itd_prediction_deg: float
    ild_prediction_deg: float
    combined_prediction_deg: float
    itd_activation: np.ndarray
    ild_activation: np.ndarray
    combined_activation: np.ndarray
    cochlea: fdm.CochleaResult
    vcn_left: np.ndarray
    vcn_right: np.ndarray
    left_level_code: np.ndarray
    right_level_code: np.ndarray
    left_lso: np.ndarray
    right_lso: np.ndarray


def make_config() -> GlobalConfig:
    """Create the matched-human binaural azimuth configuration.

    Returns:
        Acoustic configuration matching the recent mini-model setup.
    """
    base = moving_notch_signal_config(GlobalConfig())
    return replace(
        base,
        num_cochlea_channels=NUM_CHANNELS,
        min_range_m=0.25,
        max_range_m=5.0,
        signal_duration_s=0.036,
        normalize_spike_envelope=False,
        jitter_std_s=0.0,
        noise_std=0.0,
        azimuth_cue_mode="none",
    )


def azimuth_grid(limit_deg: float = AZIMUTH_LIMIT_DEG) -> np.ndarray:
    """Return represented azimuth bins in degrees."""
    return np.linspace(-limit_deg, limit_deg, NUM_AZIMUTH_BINS)


def simulate_azimuth_scene(
    config: GlobalConfig,
    azimuth_deg: float,
    *,
    distance_m: float = FIXED_DISTANCE_M,
    add_noise: bool = False,
) -> torch.Tensor:
    """Simulate one binaural echo using the old head-shadow model.

    Args:
        config: Acoustic configuration.
        azimuth_deg: Target azimuth in degrees.
        distance_m: Target range in metres.
        add_noise: Whether to add receiver noise.

    Returns:
        Binaural received waveform `[ears, time]`.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([distance_m], dtype=torch.float32),
        azimuth_deg=torch.tensor([azimuth_deg], dtype=torch.float32),
        elevation_deg=torch.tensor([0.0], dtype=torch.float32),
        binaural=True,
        add_noise=add_noise,
        include_elevation_cues=False,
        transmit_gain=config.transmit_gain,
    )
    return scene.receive[0].detach()


def run_dynamic_cochlea_spikes(cochlea: fdm.CochleaResult, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode left and right cochleagrams with the distance-pathway dynamic LIF.

    Args:
        cochlea: Binaural cochlea output.
        config: Acoustic configuration.

    Returns:
        Pair `(left_spikes, right_spikes)`.
    """
    left = fdm._dynamic_lif_encode(cochlea.left_cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
    right = fdm._dynamic_lif_encode(cochlea.right_cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
    return left, right


def vcn_consensus_single_ear(spikes: torch.Tensor, config: GlobalConfig) -> np.ndarray:
    """Run a separate-ear VCN onset detector.

    Args:
        spikes: Cochlear spike raster `[channels, time]`.
        config: Acoustic configuration.

    Returns:
        First-onset VCN raster `[channels, time]`.
    """
    masked = fdm._apply_vcn_frequency_mask_tensor(spikes, config)
    vcn = fdm._vcn_consensus_detector(masked, config, source="spikes")
    return fdm._apply_vcn_frequency_mask(vcn, config)


def first_event_times(raster: np.ndarray) -> np.ndarray:
    """Return first event sample per channel, using `-1` for missing channels."""
    first = np.full(raster.shape[0], -1, dtype=np.int64)
    for channel in range(raster.shape[0]):
        times = np.flatnonzero(raster[channel] > 0.0)
        if times.size:
            first[channel] = int(times[0])
    return first


def candidate_itd_samples(config: GlobalConfig, azimuth_bins_deg: np.ndarray) -> np.ndarray:
    """Map azimuth candidates to right-minus-left ITD samples.

    Positive azimuth is on the right side of the animal. In the simulator this
    makes the right-ear path shorter, so `right_delay - left_delay` is negative.

    Args:
        config: Acoustic configuration.
        azimuth_bins_deg: Candidate azimuths.

    Returns:
        Candidate ITD values in samples.
    """
    azimuth_rad = np.deg2rad(azimuth_bins_deg)
    itd_s = -float(config.ear_spacing_m) * np.sin(azimuth_rad) / float(config.speed_of_sound_m_s)
    return itd_s * float(config.sample_rate_hz)


def jeffress_lif_itd_activation(vcn_left: np.ndarray, vcn_right: np.ndarray, config: GlobalConfig, bins_deg: np.ndarray) -> np.ndarray:
    """Compute Jeffress-style LIF coincidence activation over azimuth bins.

    Args:
        vcn_left: Left-ear onset raster.
        vcn_right: Right-ear onset raster.
        config: Acoustic configuration.
        bins_deg: Candidate azimuth bins.

    Returns:
        ITD activation over candidate azimuths.
    """
    left_first = first_event_times(vcn_left)
    right_first = first_event_times(vcn_right)
    candidate_itd = candidate_itd_samples(config, bins_deg)
    activation = np.zeros_like(bins_deg, dtype=np.float64)
    valid = (left_first >= 0) & (right_first >= 0)
    for left_time, right_time in zip(left_first[valid], right_first[valid]):
        observed_itd = float(right_time - left_time)
        delta = np.abs(observed_itd - candidate_itd)
        membrane_peak = 1.0 + np.power(ITD_LIF_BETA, delta)
        activation += np.maximum(0.0, membrane_peak - ITD_LIF_THRESHOLD)
    return activation


def multi_threshold_level_code(spikes: torch.Tensor) -> np.ndarray:
    """Create a multi-threshold VCN-like level code from cochlear spikes.

    Each channel gets a bank of threshold neurons. A louder ear crosses more
    thresholds and therefore contributes more level spikes to the ILD pathway.

    Args:
        spikes: Cochlear spike raster `[channels, time]`.

    Returns:
        Level code `[channels]` counting crossed thresholds.
    """
    counts = spikes.detach().cpu().numpy().sum(axis=1).astype(np.float64)
    return counts


def lso_mntb_ild_activation(
    left_spikes: torch.Tensor,
    right_spikes: torch.Tensor,
    bins_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute LSO/MNTB-style ILD activation over azimuth bins.

    Args:
        left_spikes: Left cochlear spike raster.
        right_spikes: Right cochlear spike raster.
        bins_deg: Candidate azimuth bins.

    Returns:
        Tuple `(activation, left_code, right_code, left_lso, right_lso)`.
    """
    left_counts = multi_threshold_level_code(left_spikes)
    right_counts = multi_threshold_level_code(right_spikes)
    pair_scale = np.maximum(np.maximum(left_counts, right_counts), 1.0)
    thresholds = np.linspace(0.08, 0.92, ILD_THRESHOLD_COUNT)
    left_code = (left_counts[:, None] >= thresholds[None, :] * pair_scale[:, None]).sum(axis=1).astype(np.float64)
    right_code = (right_counts[:, None] >= thresholds[None, :] * pair_scale[:, None]).sum(axis=1).astype(np.float64)

    left_lso = np.maximum(0.0, left_code - ILD_INHIBITION_GAIN * right_code)
    right_lso = np.maximum(0.0, right_code - ILD_INHIBITION_GAIN * left_code)
    balance = (right_lso.sum() - left_lso.sum()) / max(float(right_lso.sum() + left_lso.sum()), 1e-12)

    expected = np.sin(np.deg2rad(bins_deg))
    activation = np.exp(-0.5 * ((balance - expected) / ILD_TUNING_SIGMA) ** 2)
    activation *= max(float(right_lso.sum() + left_lso.sum()), 1.0)
    return activation, left_code, right_code, left_lso, right_lso


def normalise_population(activity: np.ndarray) -> np.ndarray:
    """Normalise a non-negative population by its maximum."""
    peak = float(np.max(activity))
    if peak <= 1e-12:
        return np.zeros_like(activity, dtype=np.float64)
    return activity.astype(np.float64) / peak


def centre_of_mass(activity: np.ndarray, bins_deg: np.ndarray) -> float:
    """Decode azimuth by centre of mass over a non-negative population."""
    positive = np.maximum(activity, 0.0)
    total = float(np.sum(positive))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(positive * bins_deg) / total)


def predict_one(config: GlobalConfig, azimuth_deg: float, limit_deg: float = AZIMUTH_LIMIT_DEG) -> AzimuthPrediction:
    """Run the first azimuth pathway for one azimuth.

    Args:
        config: Acoustic configuration.
        azimuth_deg: True azimuth.
        limit_deg: Represented azimuth range.

    Returns:
        Full azimuth pathway prediction.
    """
    bins = azimuth_grid(limit_deg)
    receive = simulate_azimuth_scene(config, azimuth_deg)
    cochlea = fdm._run_cochlea_binaural(config, receive)
    left_spikes, right_spikes = run_dynamic_cochlea_spikes(cochlea, config)
    vcn_left = vcn_consensus_single_ear(left_spikes, config)
    vcn_right = vcn_consensus_single_ear(right_spikes, config)

    itd = jeffress_lif_itd_activation(vcn_left, vcn_right, config, bins)
    ild, left_code, right_code, left_lso, right_lso = lso_mntb_ild_activation(left_spikes, right_spikes, bins)
    combined = ITD_WEIGHT * normalise_population(itd) + ILD_WEIGHT * normalise_population(ild)

    return AzimuthPrediction(
        true_azimuth_deg=float(azimuth_deg),
        itd_prediction_deg=centre_of_mass(itd, bins),
        ild_prediction_deg=centre_of_mass(ild, bins),
        combined_prediction_deg=centre_of_mass(combined, bins),
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


def angular_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return signed angular error for non-wrapping azimuth degrees."""
    return pred - true


def metric_dict(predictions: list[AzimuthPrediction], field: str) -> dict[str, float]:
    """Compute azimuth metrics for a prediction field.

    Args:
        predictions: Prediction records.
        field: Prediction attribute name.

    Returns:
        MAE, RMSE, max absolute error, and bias in degrees.
    """
    true = np.array([item.true_azimuth_deg for item in predictions], dtype=np.float64)
    pred = np.array([getattr(item, field) for item in predictions], dtype=np.float64)
    error = angular_error(pred, true)
    return {
        "mae_deg": float(np.mean(np.abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_deg": float(np.max(np.abs(error))),
        "bias_deg": float(np.mean(error)),
    }


def run_dataset(config: GlobalConfig, limit_deg: float, count: int = TEST_SAMPLES) -> list[AzimuthPrediction]:
    """Run a deterministic azimuth test dataset.

    Args:
        config: Acoustic configuration.
        limit_deg: Symmetric azimuth range in degrees.
        count: Number of samples.

    Returns:
        List of predictions.
    """
    rng = np.random.default_rng(RNG_SEED + int(limit_deg))
    azimuths = rng.uniform(-limit_deg, limit_deg, size=count)
    return [predict_one(config, float(azimuth), limit_deg=limit_deg) for azimuth in azimuths]


def plot_pipeline_diagram(path: Path) -> str:
    """Plot the proposed azimuth-pathway block diagram."""
    fig, ax = plt.subplots(figsize=(13.2, 3.8))
    ax.axis("off")
    labels = [
        "Binaural echo\nhead shadow + ITD",
        "Left/right\ncochlea",
        "ITD VCN\nonset",
        "Jeffress IC\nLIF coincidence",
        "ILD VCN\nlevel bank",
        "MNTB -> LSO\nE/I opponent",
        "IC merge",
        "SC COM\nazimuth",
    ]
    x = np.linspace(0.05, 0.95, len(labels))
    y = [0.62, 0.62, 0.82, 0.82, 0.36, 0.36, 0.62, 0.62]
    for idx, (xpos, ypos, label) in enumerate(zip(x, y, labels)):
        ax.text(
            xpos,
            ypos,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor="#1f2937", linewidth=1.0),
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1] - 0.045, y[idx + 1]),
                xytext=(xpos + 0.045, ypos),
                arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.2),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    return save_figure(fig, path)


def plot_example_stages(prediction: AzimuthPrediction, config: GlobalConfig, path: Path) -> str:
    """Plot waveform, VCN rasters, LSO codes, and azimuth populations."""
    centers_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    time_ms = np.arange(prediction.cochlea.left_spikes.shape[1]) / config.sample_rate_hz * 1_000.0
    bins = azimuth_grid(AZIMUTH_LIMIT_DEG)
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 12.0))

    for channel, freq in enumerate(centers_khz):
        left_times = time_ms[np.flatnonzero(prediction.vcn_left[channel] > 0.0)]
        right_times = time_ms[np.flatnonzero(prediction.vcn_right[channel] > 0.0)]
        if left_times.size:
            axes[0].vlines(left_times, freq * 0.985, freq * 1.015, color="#2563eb", linewidth=0.9)
        if right_times.size:
            axes[0].vlines(right_times, freq * 0.985, freq * 1.015, color="#dc2626", linewidth=0.9)
    axes[0].axhline(VCN_MIN_RESPONSIVE_HZ / 1_000.0, color="#6b7280", linestyle=":", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_title("Separate-ear VCN onset rasters: left blue, right red")
    axes[0].grid(True, axis="x", alpha=0.2)

    axes[1].plot(centers_khz, prediction.left_level_code, color="#2563eb", linewidth=1.8, label="left level code")
    axes[1].plot(centers_khz, prediction.right_level_code, color="#dc2626", linewidth=1.8, label="right level code")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("threshold count")
    axes[1].set_title("Multi-threshold ILD VCN level code")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(centers_khz, prediction.left_lso, color="#2563eb", linewidth=1.8, label="left LSO")
    axes[2].plot(centers_khz, prediction.right_lso, color="#dc2626", linewidth=1.8, label="right LSO")
    axes[2].set_xscale("log")
    axes[2].set_ylabel("opponent response")
    axes[2].set_title("LSO/MNTB opponent output")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    axes[3].plot(bins, normalise_population(prediction.itd_activation), linewidth=2.0, label="ITD")
    axes[3].plot(bins, normalise_population(prediction.ild_activation), linewidth=2.0, label="ILD")
    axes[3].plot(bins, normalise_population(prediction.combined_activation), linewidth=2.0, label="combined")
    axes[3].axvline(prediction.true_azimuth_deg, color="#111827", linestyle="--", linewidth=1.2, label="true")
    axes[3].axvline(prediction.combined_prediction_deg, color="#f59e0b", linestyle=":", linewidth=1.4, label="decoded")
    axes[3].set_xlabel("represented azimuth (deg)")
    axes[3].set_ylabel("normalised activity")
    axes[3].set_title("IC/SC azimuth populations")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(frameon=False)

    fig.tight_layout()
    return save_figure(fig, path)


def plot_prediction_scatter(predictions: list[AzimuthPrediction], path: Path) -> str:
    """Plot predicted versus true azimuth for ITD, ILD, and combined readouts."""
    true = np.array([item.true_azimuth_deg for item in predictions])
    itd = np.array([item.itd_prediction_deg for item in predictions])
    ild = np.array([item.ild_prediction_deg for item in predictions])
    combined = np.array([item.combined_prediction_deg for item in predictions])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(true, itd, s=20, alpha=0.55, label="ITD")
    ax.scatter(true, ild, s=20, alpha=0.55, label="ILD")
    ax.scatter(true, combined, s=22, alpha=0.72, label="combined")
    low = min(float(true.min()), float(itd.min()), float(ild.min()), float(combined.min()))
    high = max(float(true.max()), float(itd.max()), float(ild.max()), float(combined.max()))
    ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
    ax.set_xlabel("true azimuth (deg)")
    ax.set_ylabel("predicted azimuth (deg)")
    ax.set_title("First azimuth pathway predictions")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_error_histogram(predictions: list[AzimuthPrediction], path: Path) -> str:
    """Plot signed combined-readout azimuth error histogram."""
    true = np.array([item.true_azimuth_deg for item in predictions])
    pred = np.array([item.combined_prediction_deg for item in predictions])
    error = pred - true
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(error, bins=24, color="#2563eb", alpha=0.82)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel("combined readout error (deg)")
    ax.set_ylabel("count")
    ax.set_title("Combined azimuth error distribution")
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, path)


def format_metric_row(label: str, metrics: dict[str, float]) -> str:
    """Format one metrics table row."""
    return (
        "| "
        f"{label} | "
        f"`{metrics['mae_deg']:.3f} deg` | "
        f"`{metrics['rmse_deg']:.3f} deg` | "
        f"`{metrics['max_abs_error_deg']:.3f} deg` | "
        f"`{metrics['bias_deg']:.3f} deg` |"
    )


def write_report(
    config: GlobalConfig,
    primary_predictions: list[AzimuthPrediction],
    stress_predictions: list[AzimuthPrediction],
    metrics: dict[str, dict[str, float]],
    artifacts: dict[str, str],
    elapsed_s: float,
) -> None:
    """Write the first azimuth-pathway report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_rows = [
        f"| {name} | `{value:.3f} deg` | old trained full model |"
        for name, value in OLD_MODEL_AZIMUTH_RESULTS.items()
    ]
    metric_rows = [
        format_metric_row("ITD only", metrics["primary_itd"]),
        format_metric_row("ILD only", metrics["primary_ild"]),
        format_metric_row("Combined", metrics["primary_combined"]),
        format_metric_row("Combined stress +/-90 deg", metrics["stress_combined"]),
    ]
    lines = [
        "# Azimuth Pathway First Attempt",
        "",
        "This report starts a new standalone azimuth pathway. It does not modify the old trained model. The aim is to build an interpretable feed-forward azimuth system using the same final cochlea used in the distance pathway.",
        "",
        "![Pipeline diagram](../outputs/first_attempt/figures/pipeline_diagram.png)",
        "",
        "## Acoustic Setup",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| sample rate | `{config.sample_rate_hz:.0f} Hz` |",
        f"| chirp | `{config.chirp_start_hz:.0f} -> {config.chirp_end_hz:.0f} Hz` |",
        f"| chirp duration | `{config.chirp_duration_s * 1_000.0:.1f} ms` |",
        f"| target distance | `{FIXED_DISTANCE_M:.2f} m` |",
        f"| primary azimuth range | `+/-{AZIMUTH_LIMIT_DEG:.0f} deg` |",
        f"| stress azimuth range | `+/-{STRESS_AZIMUTH_LIMIT_DEG:.0f} deg` |",
        f"| ear spacing | `{config.ear_spacing_m:.3f} m` |",
        f"| head shadow strength | `{config.head_shadow_strength:.3f}` |",
        f"| cochlea channels | `{NUM_CHANNELS}` |",
        "",
        "The old head-shadow model is retained. For azimuth $\\theta$, the simulator applies a multiplicative gain to each ear:",
        "",
        "$$",
        "g_{ear}=\\exp(s_h\\sin\\theta\\,[-1,+1]),",
        "$$",
        "",
        "where $s_h$ is the head-shadow strength. Binaural geometry also changes the path length to each ear, creating an ITD.",
        "",
        "## ITD Branch",
        "",
        "The ITD branch uses separate-ear VCN onset rasters. For candidate azimuth $\\theta_k$, the expected right-minus-left ITD is approximated by:",
        "",
        "$$",
        "\\Delta t_k = -\\frac{d_{ear}\\sin\\theta_k}{c}.",
        "$$",
        "",
        "For each frequency channel, a Jeffress-style LIF coincidence detector receives a left onset and a structurally delayed right onset. In the two-spike approximation:",
        "",
        "$$",
        "m_{c,k}=1+\\beta_{ITD}^{|\\Delta n_c-\\Delta n_k|},",
        "\\qquad",
        "a_{c,k}=\\max(0,m_{c,k}-\\vartheta_{ITD}).",
        "$$",
        "",
        "The ITD azimuth population is the sum of $a_{c,k}$ over channels.",
        "",
        "## ILD Branch",
        "",
        "The ILD branch uses a multi-threshold level code. For each ear and frequency channel, a bank of threshold neurons converts spike count into a level count. Louder ears cross more thresholds.",
        "",
        "The MNTB/LSO opponent stage is:",
        "",
        "$$",
        "LSO_L=\\max(0,L-g_I R),",
        "\\qquad",
        "LSO_R=\\max(0,R-g_I L).",
        "$$",
        "",
        "The global opponent balance is:",
        "",
        "$$",
        "b=\\frac{\\sum_c LSO_R(c)-\\sum_c LSO_L(c)}{\\sum_c LSO_R(c)+\\sum_c LSO_L(c)}.",
        "$$",
        "",
        "Candidate azimuths are scored by matching $b$ to $\\sin\\theta_k$ with a Gaussian tuning curve.",
        "",
        "## IC And SC Readout",
        "",
        "The IC combines the normalised ITD and ILD azimuth populations. In this first prototype the ITD branch is deliberately weighted more heavily because the raw ITD detector is already sharp, while the first ILD branch is included as a weaker biological diagnostic cue:",
        "",
        "$$",
        "A_{IC}(\\theta_k)=w_{ITD}\\hat A_{ITD}(\\theta_k)+w_{ILD}\\hat A_{ILD}(\\theta_k).",
        "$$",
        "",
        "The first SC readout is a centre of mass:",
        "",
        "$$",
        "\\hat\\theta=\\frac{\\sum_k A_{IC}(\\theta_k)\\theta_k}{\\sum_k A_{IC}(\\theta_k)}.",
        "$$",
        "",
        "## Example Processing Stages",
        "",
        f"The example target has azimuth `{primary_predictions[len(primary_predictions)//2].true_azimuth_deg:.2f} deg`.",
        "",
        "![Example stages](../outputs/first_attempt/figures/example_stages.png)",
        "",
        "## Accuracy",
        "",
        "The primary test uses the same azimuth support as the old Round 3/4 training setup, `-45 deg` to `+45 deg`, at fixed range. The stress test expands to `-90 deg` to `+90 deg`.",
        "",
        "| Readout | MAE | RMSE | Max error | Bias |",
        "|---|---:|---:|---:|---:|",
        *metric_rows,
        "",
        "![Prediction scatter](../outputs/first_attempt/figures/prediction_scatter.png)",
        "",
        "![Error histogram](../outputs/first_attempt/figures/error_histogram.png)",
        "",
        "## Old Model Comparison",
        "",
        "These values are copied from the old reports and are not rerun here.",
        "",
        "| Model | Azimuth MAE | Notes |",
        "|---|---:|---|",
        *comparison_rows,
        "",
        "This first standalone pathway is not yet a full replacement for the old trained azimuth branch. It is useful because the cue split is explicit: ITD, ILD, and combined populations can be inspected separately before adding an azimuth SC attractor or robustness tests.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.extend([f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`", "", f"Runtime: `{elapsed_s:.2f} s`.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the first azimuth-pathway experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    config = make_config()
    primary_predictions = run_dataset(config, AZIMUTH_LIMIT_DEG)
    stress_predictions = run_dataset(config, STRESS_AZIMUTH_LIMIT_DEG)
    example = primary_predictions[len(primary_predictions) // 2]
    metrics = {
        "primary_itd": metric_dict(primary_predictions, "itd_prediction_deg"),
        "primary_ild": metric_dict(primary_predictions, "ild_prediction_deg"),
        "primary_combined": metric_dict(primary_predictions, "combined_prediction_deg"),
        "stress_combined": metric_dict(stress_predictions, "combined_prediction_deg"),
    }
    artifacts = {
        "pipeline_diagram": plot_pipeline_diagram(FIGURE_DIR / "pipeline_diagram.png"),
        "example_stages": plot_example_stages(example, config, FIGURE_DIR / "example_stages.png"),
        "prediction_scatter": plot_prediction_scatter(primary_predictions, FIGURE_DIR / "prediction_scatter.png"),
        "error_histogram": plot_error_histogram(primary_predictions, FIGURE_DIR / "error_histogram.png"),
    }
    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "azimuth_pathway_first_attempt",
        "elapsed_seconds": elapsed_s,
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "chirp_start_hz": config.chirp_start_hz,
            "chirp_end_hz": config.chirp_end_hz,
            "chirp_duration_s": config.chirp_duration_s,
            "ear_spacing_m": config.ear_spacing_m,
            "head_shadow_strength": config.head_shadow_strength,
            "num_channels": NUM_CHANNELS,
            "fixed_distance_m": FIXED_DISTANCE_M,
        },
        "parameters": {
            "itd_lif_beta": ITD_LIF_BETA,
            "itd_lif_threshold": ITD_LIF_THRESHOLD,
            "ild_threshold_count": ILD_THRESHOLD_COUNT,
            "ild_inhibition_gain": ILD_INHIBITION_GAIN,
            "ild_tuning_sigma": ILD_TUNING_SIGMA,
            "itd_weight": ITD_WEIGHT,
            "ild_weight": ILD_WEIGHT,
        },
        "metrics": metrics,
        "old_model_azimuth_mae_deg": OLD_MODEL_AZIMUTH_RESULTS,
        "primary_predictions": [
            {
                "true_azimuth_deg": item.true_azimuth_deg,
                "itd_prediction_deg": item.itd_prediction_deg,
                "ild_prediction_deg": item.ild_prediction_deg,
                "combined_prediction_deg": item.combined_prediction_deg,
            }
            for item in primary_predictions
        ],
        "stress_predictions": [
            {
                "true_azimuth_deg": item.true_azimuth_deg,
                "combined_prediction_deg": item.combined_prediction_deg,
            }
            for item in stress_predictions
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(config, primary_predictions, stress_predictions, metrics, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
