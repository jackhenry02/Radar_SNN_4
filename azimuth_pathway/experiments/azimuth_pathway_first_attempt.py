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
WARP_POWER_GRID = np.linspace(1.0, 5.0, 81)
WARP_SIGMA_GRID = np.linspace(0.03, 0.30, 55)
INVERSE_SIGMOID_K_GRID = np.linspace(1.0, 10.0, 181)

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


def lso_balance_and_drive(prediction: AzimuthPrediction) -> tuple[float, float]:
    """Return global LSO balance and total opponent drive.

    Args:
        prediction: Azimuth pathway prediction containing LSO populations.

    Returns:
        Pair `(balance, drive)`, where balance is in approximately `[-1, 1]`.
    """
    left_sum = float(np.sum(prediction.left_lso))
    right_sum = float(np.sum(prediction.right_lso))
    drive = max(left_sum + right_sum, 1e-12)
    balance = (right_sum - left_sum) / drive
    return balance, max(drive, 1.0)


def warped_ild_activation(balance: float, drive: float, bins_deg: np.ndarray, power: float, sigma: float) -> np.ndarray:
    """Project LSO balance through a warped synaptic mapping layer.

    The raw LSO balance is too steep near the midline in this prototype. A
    power-law warp compresses intermediate opponent values while keeping the
    sign and edge ordering intact.

    Args:
        balance: Raw opponent balance.
        drive: Total opponent drive.
        bins_deg: Candidate azimuth bins.
        power: Odd monotonic warp exponent.
        sigma: Synaptic tuning width in sine-coordinate units.

    Returns:
        Warped ILD activation over candidate azimuth bins.
    """
    warped_balance = math.copysign(abs(balance) ** power, balance)
    expected = np.sin(np.deg2rad(bins_deg))
    return np.exp(-0.5 * ((warped_balance - expected) / sigma) ** 2) * drive


def decode_warped_ild(
    predictions: list[AzimuthPrediction],
    bins_deg: np.ndarray,
    power: float,
    sigma: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decode warped ILD predictions for a dataset.

    Args:
        predictions: Existing pathway predictions with LSO stage outputs.
        bins_deg: Candidate azimuth bins.
        power: Balance-warp exponent.
        sigma: Synaptic tuning width.

    Returns:
        Pair `(decoded_degrees, activations)`.
    """
    decoded = []
    activations = []
    for prediction in predictions:
        balance, drive = lso_balance_and_drive(prediction)
        activation = warped_ild_activation(balance, drive, bins_deg, power, sigma)
        activations.append(activation)
        decoded.append(centre_of_mass(activation, bins_deg))
    return np.array(decoded, dtype=np.float64), activations


def tune_warped_ild_mapping(predictions: list[AzimuthPrediction], bins_deg: np.ndarray) -> dict[str, float]:
    """Grid-search a monotonic warped synaptic mapping for the ILD branch.

    Args:
        predictions: Calibration predictions.
        bins_deg: Candidate azimuth bins.

    Returns:
        Best warp parameters and calibration metrics.
    """
    true = np.array([item.true_azimuth_deg for item in predictions], dtype=np.float64)
    best: dict[str, float] | None = None
    for power in WARP_POWER_GRID:
        for sigma in WARP_SIGMA_GRID:
            decoded, _ = decode_warped_ild(predictions, bins_deg, float(power), float(sigma))
            error = decoded - true
            mae = float(np.mean(np.abs(error)))
            if best is None or mae < best["mae_deg"]:
                best = {
                    "power": float(power),
                    "sigma": float(sigma),
                    "mae_deg": mae,
                    "rmse_deg": float(np.sqrt(np.mean(error**2))),
                    "max_abs_error_deg": float(np.max(np.abs(error))),
                    "bias_deg": float(np.mean(error)),
                }
    if best is None:
        raise RuntimeError("Warped ILD tuning failed.")
    return best


def inverse_sigmoid_ild_activation(
    balance: float,
    drive: float,
    bins_deg: np.ndarray,
    gain: float,
    sigma: float,
    limit_deg: float,
) -> np.ndarray:
    """Project LSO balance with an inverse-sigmoid synaptic mapping.

    This assumes the measured LSO balance is a saturating function of the
    underlying azimuth coordinate, approximately `b = tanh(k sin(theta))`.
    The mapping therefore estimates the hidden sine-coordinate by applying
    `atanh(b) / k`.

    Args:
        balance: Raw opponent balance.
        drive: Total opponent drive.
        bins_deg: Candidate azimuth bins.
        gain: Saturation gain `k`.
        sigma: Synaptic tuning width in sine-coordinate units.
        limit_deg: Represented azimuth support used for clipping.

    Returns:
        Inverse-sigmoid ILD activation over candidate azimuth bins.
    """
    max_sine = math.sin(math.radians(limit_deg))
    safe_balance = float(np.clip(balance, -0.999999, 0.999999))
    mapped = float(np.arctanh(safe_balance)) / gain
    mapped = float(np.clip(mapped, -max_sine, max_sine))
    expected = np.sin(np.deg2rad(bins_deg))
    return np.exp(-0.5 * ((mapped - expected) / sigma) ** 2) * drive


def decode_inverse_sigmoid_ild(
    predictions: list[AzimuthPrediction],
    bins_deg: np.ndarray,
    gain: float,
    sigma: float,
    limit_deg: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decode inverse-sigmoid ILD predictions for a dataset.

    Args:
        predictions: Existing pathway predictions with LSO stage outputs.
        bins_deg: Candidate azimuth bins.
        gain: Saturation gain `k`.
        sigma: Synaptic tuning width.
        limit_deg: Represented azimuth support.

    Returns:
        Pair `(decoded_degrees, activations)`.
    """
    decoded = []
    activations = []
    for prediction in predictions:
        balance, drive = lso_balance_and_drive(prediction)
        activation = inverse_sigmoid_ild_activation(balance, drive, bins_deg, gain, sigma, limit_deg)
        activations.append(activation)
        decoded.append(centre_of_mass(activation, bins_deg))
    return np.array(decoded, dtype=np.float64), activations


def tune_inverse_sigmoid_ild_mapping(
    predictions: list[AzimuthPrediction],
    bins_deg: np.ndarray,
    limit_deg: float,
) -> dict[str, float]:
    """Grid-search the inverse-sigmoid ILD mapping.

    Args:
        predictions: Calibration predictions.
        bins_deg: Candidate azimuth bins.
        limit_deg: Represented azimuth support.

    Returns:
        Best inverse-sigmoid parameters and calibration metrics.
    """
    true = np.array([item.true_azimuth_deg for item in predictions], dtype=np.float64)
    best: dict[str, float] | None = None
    for gain in INVERSE_SIGMOID_K_GRID:
        for sigma in WARP_SIGMA_GRID:
            decoded, _ = decode_inverse_sigmoid_ild(predictions, bins_deg, float(gain), float(sigma), limit_deg)
            error = decoded - true
            mae = float(np.mean(np.abs(error)))
            if best is None or mae < best["mae_deg"]:
                best = {
                    "gain": float(gain),
                    "sigma": float(sigma),
                    "mae_deg": mae,
                    "rmse_deg": float(np.sqrt(np.mean(error**2))),
                    "max_abs_error_deg": float(np.max(np.abs(error))),
                    "bias_deg": float(np.mean(error)),
                }
    if best is None:
        raise RuntimeError("Inverse-sigmoid ILD tuning failed.")
    return best


def metric_dict_from_arrays(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute azimuth metrics from arrays."""
    error = angular_error(pred, true)
    return {
        "mae_deg": float(np.mean(np.abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_deg": float(np.max(np.abs(error))),
        "bias_deg": float(np.mean(error)),
    }


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


def plot_example_stages(
    prediction: AzimuthPrediction,
    config: GlobalConfig,
    inverse_params: dict[str, float],
    path: Path,
) -> str:
    """Plot VCN rasters, ILD coding, and azimuth populations.

    Args:
        prediction: Example prediction to visualise.
        config: Acoustic configuration.
        inverse_params: Tuned inverse-sigmoid ILD parameters.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    centers_khz = fdm._log_spaced_centers(config).detach().cpu().numpy() / 1_000.0
    time_ms = np.arange(prediction.cochlea.left_spikes.shape[1]) / config.sample_rate_hz * 1_000.0
    bins = azimuth_grid(AZIMUTH_LIMIT_DEG)
    balance, drive = lso_balance_and_drive(prediction)
    inverse_activation = inverse_sigmoid_ild_activation(
        balance,
        drive,
        bins,
        inverse_params["gain"],
        inverse_params["sigma"],
        AZIMUTH_LIMIT_DEG,
    )
    inverse_prediction = centre_of_mass(inverse_activation, bins)
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
    axes[0].set_xlabel("time after emission (ms)")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_title("VCN onset raster after cochlea: left ear blue, right ear red")
    axes[0].grid(True, axis="x", alpha=0.2)

    axes[1].plot(centers_khz, prediction.left_level_code, color="#2563eb", linewidth=1.8, label="left level code")
    axes[1].plot(centers_khz, prediction.right_level_code, color="#dc2626", linewidth=1.8, label="right level code")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("cochlear centre frequency (kHz)")
    axes[1].set_ylabel("threshold count")
    axes[1].set_title("ILD level code: number of crossed loudness thresholds per channel")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(centers_khz, prediction.left_lso, color="#2563eb", linewidth=1.8, label="left LSO")
    axes[2].plot(centers_khz, prediction.right_lso, color="#dc2626", linewidth=1.8, label="right LSO")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("cochlear centre frequency (kHz)")
    axes[2].set_ylabel("rectified E/I response")
    axes[2].set_title("MNTB/LSO opponent output: same-ear excitation minus opposite-ear inhibition")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    axes[3].plot(bins, normalise_population(prediction.itd_activation), linewidth=2.0, label="ITD")
    axes[3].plot(bins, normalise_population(prediction.ild_activation), linewidth=2.0, label="raw ILD")
    axes[3].plot(bins, normalise_population(inverse_activation), linewidth=2.2, label="inverse-sigmoid ILD")
    axes[3].axvline(prediction.true_azimuth_deg, color="#111827", linestyle="--", linewidth=1.2, label="true")
    axes[3].axvline(inverse_prediction, color="#059669", linestyle=":", linewidth=1.6, label="inverse-sigmoid decoded")
    axes[3].set_xlabel("represented azimuth (deg)")
    axes[3].set_ylabel("normalised activity")
    axes[3].set_title("Topographic azimuth populations before SC centre-of-mass readout")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(frameon=False)

    fig.tight_layout()
    return save_figure(fig, path)


def plot_prediction_scatter(
    predictions: list[AzimuthPrediction],
    inverse_predictions_deg: np.ndarray,
    path: Path,
) -> str:
    """Plot predicted versus true azimuth for raw and calibrated readouts.

    Args:
        predictions: Primary azimuth predictions.
        inverse_predictions_deg: Inverse-sigmoid ILD predictions.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    true = np.array([item.true_azimuth_deg for item in predictions])
    itd = np.array([item.itd_prediction_deg for item in predictions])
    ild = np.array([item.ild_prediction_deg for item in predictions])
    combined = np.array([item.combined_prediction_deg for item in predictions])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(true, itd, s=20, alpha=0.55, label="ITD")
    ax.scatter(true, ild, s=20, alpha=0.45, label="raw ILD")
    ax.scatter(true, combined, s=22, alpha=0.55, label="old combined")
    ax.scatter(true, inverse_predictions_deg, s=26, alpha=0.78, label="inverse-sigmoid ILD")
    low = min(float(true.min()), float(itd.min()), float(ild.min()), float(combined.min()), float(inverse_predictions_deg.min()))
    high = max(float(true.max()), float(itd.max()), float(ild.max()), float(combined.max()), float(inverse_predictions_deg.max()))
    ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
    ax.set_xlabel("true azimuth (deg)")
    ax.set_ylabel("predicted azimuth (deg)")
    ax.set_title("Azimuth predictions after ILD inverse-sigmoid calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_error_histogram(true: np.ndarray, pred: np.ndarray, label: str, path: Path) -> str:
    """Plot signed azimuth error histogram.

    Args:
        true: True azimuths in degrees.
        pred: Predicted azimuths in degrees.
        label: Readout label for axes and title.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    error = pred - true
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(error, bins=24, color="#059669", alpha=0.82)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(f"{label} error (deg)")
    ax.set_ylabel("count")
    ax.set_title(f"{label} azimuth error distribution")
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, path)


def plot_range_comparison(
    primary_true: np.ndarray,
    primary_pred: np.ndarray,
    stress_true: np.ndarray,
    stress_pred: np.ndarray,
    path: Path,
) -> str:
    """Compare inverse-sigmoid ILD errors for +/-45 and +/-90 degree tests.

    Args:
        primary_true: True azimuths for the +/-45 degree support.
        primary_pred: Predicted azimuths for the +/-45 degree support.
        stress_true: True azimuths for the +/-90 degree support.
        stress_pred: Predicted azimuths for the +/-90 degree support.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    datasets = [
        ("+/-45 deg", primary_true, primary_pred, AZIMUTH_LIMIT_DEG),
        ("+/-90 deg", stress_true, stress_pred, STRESS_AZIMUTH_LIMIT_DEG),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=False)
    for ax, (label, true, pred, limit) in zip(axes, datasets):
        error = pred - true
        ax.scatter(true, error, s=22, alpha=0.7, color="#059669")
        ax.axhline(0.0, color="#111827", linewidth=1.0)
        ax.axvline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.set_xlim(-limit, limit)
        ax.set_xlabel("true azimuth (deg)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("inverse-sigmoid ILD error (deg)")
    fig.suptitle("Wide-field azimuth error for the improved ILD readout")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_warped_ild_scatter(
    predictions: list[AzimuthPrediction],
    warped_predictions_deg: np.ndarray,
    inverse_predictions_deg: np.ndarray,
    path: Path,
) -> str:
    """Plot raw and calibrated ILD predictions against true azimuth.

    Args:
        predictions: Primary azimuth predictions.
        warped_predictions_deg: Warped ILD decoded azimuths.
        inverse_predictions_deg: Inverse-sigmoid ILD decoded azimuths.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    true = np.array([item.true_azimuth_deg for item in predictions], dtype=np.float64)
    raw = np.array([item.ild_prediction_deg for item in predictions], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(true, raw, s=22, alpha=0.58, label="raw ILD")
    ax.scatter(true, warped_predictions_deg, s=24, alpha=0.64, label="power-warp ILD")
    ax.scatter(true, inverse_predictions_deg, s=26, alpha=0.78, label="inverse-sigmoid ILD")
    low = min(float(true.min()), float(raw.min()), float(warped_predictions_deg.min()), float(inverse_predictions_deg.min()))
    high = max(float(true.max()), float(raw.max()), float(warped_predictions_deg.max()), float(inverse_predictions_deg.max()))
    ax.plot([low, high], [low, high], color="#111827", linewidth=1.0)
    ax.set_xlabel("true azimuth (deg)")
    ax.set_ylabel("decoded azimuth (deg)")
    ax.set_title("Synaptic mapping layers correct the ILD sigmoid")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_warped_ild_mapping(
    predictions: list[AzimuthPrediction],
    warp_params: dict[str, float],
    inverse_params: dict[str, float],
    path: Path,
) -> str:
    """Plot the tuned balance warps and measured LSO balance.

    Args:
        predictions: Primary azimuth predictions.
        warp_params: Tuned warp parameters.
        inverse_params: Tuned inverse-sigmoid parameters.
        path: Figure path.

    Returns:
        Saved figure path.
    """
    true = np.array([item.true_azimuth_deg for item in predictions], dtype=np.float64)
    balances = np.array([lso_balance_and_drive(item)[0] for item in predictions], dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, 501)
    warped = np.sign(grid) * np.abs(grid) ** warp_params["power"]
    max_sine = math.sin(math.radians(AZIMUTH_LIMIT_DEG))
    inverse = np.arctanh(np.clip(grid, -0.999999, 0.999999)) / inverse_params["gain"]
    inverse = np.clip(inverse, -max_sine, max_sine)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].scatter(true, balances, s=24, alpha=0.72, color="#2563eb")
    axes[0].plot(
        np.linspace(-AZIMUTH_LIMIT_DEG, AZIMUTH_LIMIT_DEG, 301),
        np.sin(np.deg2rad(np.linspace(-AZIMUTH_LIMIT_DEG, AZIMUTH_LIMIT_DEG, 301))),
        color="#111827",
        linewidth=1.0,
        label="ideal sin(theta)",
    )
    axes[0].set_xlabel("true azimuth (deg)")
    axes[0].set_ylabel("raw LSO balance")
    axes[0].set_title("Raw ILD balance is too steep near midline")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(grid, grid, color="#9ca3af", linestyle="--", linewidth=1.0, label="identity")
    axes[1].plot(grid, warped, color="#dc2626", linewidth=2.0, label="power warp")
    axes[1].plot(grid, inverse, color="#059669", linewidth=2.0, label="inverse sigmoid")
    axes[1].set_xlabel("raw balance")
    axes[1].set_ylabel("mapped sine-coordinate")
    axes[1].set_title("Synaptic balance mappings")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
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
    warp_params: dict[str, float],
    inverse_params: dict[str, float],
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
        format_metric_row("ILD power warp", metrics["primary_ild_warped"]),
        format_metric_row("ILD inverse sigmoid", metrics["primary_ild_inverse_sigmoid"]),
        format_metric_row("Combined", metrics["primary_combined"]),
        format_metric_row("ILD inverse sigmoid stress +/-90 deg", metrics["stress_ild_inverse_sigmoid"]),
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
        "## Warped Synaptic Mapping Layer",
        "",
        "The raw ILD result has a sigmoid-like calibration error: midline balances are too large, so moderate azimuths are decoded too far from zero. The acoustic head-shadow and LSO/MNTB stages are left unchanged. The correction is a post-LSO synaptic mapping layer.",
        "",
        "For raw LSO balance $b$, the warped balance is:",
        "",
        "$$",
        "\\tilde b = \\operatorname{sign}(b)|b|^\\gamma.",
        "$$",
        "",
        "The warped ILD population is then:",
        "",
        "$$",
        "A^{warp}_{ILD}(\\theta_k)=D\\exp\\left[-\\frac{(\\tilde b-\\sin\\theta_k)^2}{2\\sigma_w^2}\\right],",
        "$$",
        "",
        "where $D=\\sum_c LSO_L(c)+\\sum_c LSO_R(c)$ is the total opponent drive. This is equivalent to changing the synaptic projection from the LSO balance coordinate into the topographic azimuth coordinate.",
        "",
        f"The tuned parameters on this deterministic `+/-45 deg` test were `gamma = {warp_params['power']:.3f}` and `sigma_w = {warp_params['sigma']:.3f}`.",
        "",
        "A second variant treats the raw LSO balance as a saturating sigmoid of the hidden azimuth coordinate. If",
        "",
        "$$",
        "b \\approx \\tanh(k\\sin\\theta),",
        "$$",
        "",
        "then the inverse-sigmoid estimate is:",
        "",
        "$$",
        "\\tilde b = \\frac{\\operatorname{atanh}(b)}{k}.",
        "$$",
        "",
        f"The tuned inverse-sigmoid parameters were `k = {inverse_params['gain']:.3f}` and `sigma_w = {inverse_params['sigma']:.3f}`. This is the better correction in this experiment, which supports the interpretation that the raw ILD pathway is mostly a saturating monotonic map rather than a noisy unordered cue.",
        "",
        "![Warped ILD mapping](../outputs/first_attempt/figures/warped_ild_mapping.png)",
        "",
        "![Warped ILD scatter](../outputs/first_attempt/figures/warped_ild_scatter.png)",
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
        "The primary test uses the same azimuth support as the old Round 3/4 training setup, `-45 deg` to `+45 deg`, at fixed range. The stress test expands to `-90 deg` to `+90 deg`. Since the inverse-sigmoid ILD branch is now the strongest branch, the wide-field error plot and stress row use that readout rather than the older raw combined readout.",
        "",
        "| Readout | MAE | RMSE | Max error | Bias |",
        "|---|---:|---:|---:|---:|",
        *metric_rows,
        "",
        "![Prediction scatter](../outputs/first_attempt/figures/prediction_scatter.png)",
        "",
        "![Error histogram](../outputs/first_attempt/figures/error_histogram.png)",
        "",
        "![Range comparison](../outputs/first_attempt/figures/range_comparison.png)",
        "",
        "## Why The +/-90 Degree Case Is Worse",
        "",
        "The wider-field result is lower accuracy, but it is not a useless failure. It is a useful stress result for three reasons.",
        "",
        "First, both major binaural cues are nonlinear at the edges. ITD is approximately proportional to $\\sin\\theta$, so it flattens near the sides. The improved ILD branch also depends on a saturated LSO balance; the inverse-sigmoid mapping recovers much of this, but edge clipping and finite population width still reduce precision.",
        "",
        "Second, the current head-shadow ILD cue is deliberately simple. It is a smooth multiplicative gain, not a full frequency-dependent head-related transfer function. That means the ILD branch does not yet add enough extra wide-angle information to compensate for ITD saturation.",
        "",
        "Third, the current SC readout is a centre of mass over a single population. At wide angles, broad or asymmetric populations are pulled inward, so extreme azimuths tend to be underestimated. This is exactly the kind of behaviour that an azimuth attractor or better edge-aware readout should be tested against later.",
        "",
        "This makes the `+/-90 deg` result valuable: the `+/-45 deg` case shows the pathway can be very accurate under the old model support, while the `+/-90 deg` case shows the expected biological/geometry limitation of wider-field azimuth estimation.",
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
    primary_bins = azimuth_grid(AZIMUTH_LIMIT_DEG)
    warp_params = tune_warped_ild_mapping(primary_predictions, primary_bins)
    warped_ild_predictions, _ = decode_warped_ild(
        primary_predictions,
        primary_bins,
        warp_params["power"],
        warp_params["sigma"],
    )
    inverse_params = tune_inverse_sigmoid_ild_mapping(primary_predictions, primary_bins, AZIMUTH_LIMIT_DEG)
    inverse_ild_predictions, _ = decode_inverse_sigmoid_ild(
        primary_predictions,
        primary_bins,
        inverse_params["gain"],
        inverse_params["sigma"],
        AZIMUTH_LIMIT_DEG,
    )
    stress_bins = azimuth_grid(STRESS_AZIMUTH_LIMIT_DEG)
    stress_inverse_ild_predictions, _ = decode_inverse_sigmoid_ild(
        stress_predictions,
        stress_bins,
        inverse_params["gain"],
        inverse_params["sigma"],
        STRESS_AZIMUTH_LIMIT_DEG,
    )
    primary_true = np.array([item.true_azimuth_deg for item in primary_predictions], dtype=np.float64)
    stress_true = np.array([item.true_azimuth_deg for item in stress_predictions], dtype=np.float64)
    metrics = {
        "primary_itd": metric_dict(primary_predictions, "itd_prediction_deg"),
        "primary_ild": metric_dict(primary_predictions, "ild_prediction_deg"),
        "primary_ild_warped": metric_dict_from_arrays(primary_true, warped_ild_predictions),
        "primary_ild_inverse_sigmoid": metric_dict_from_arrays(primary_true, inverse_ild_predictions),
        "primary_combined": metric_dict(primary_predictions, "combined_prediction_deg"),
        "stress_combined": metric_dict(stress_predictions, "combined_prediction_deg"),
        "stress_ild_inverse_sigmoid": metric_dict_from_arrays(stress_true, stress_inverse_ild_predictions),
    }
    artifacts = {
        "pipeline_diagram": plot_pipeline_diagram(FIGURE_DIR / "pipeline_diagram.png"),
        "example_stages": plot_example_stages(
            example,
            config,
            inverse_params,
            FIGURE_DIR / "example_stages.png",
        ),
        "prediction_scatter": plot_prediction_scatter(
            primary_predictions,
            inverse_ild_predictions,
            FIGURE_DIR / "prediction_scatter.png",
        ),
        "error_histogram": plot_error_histogram(
            stress_true,
            stress_inverse_ild_predictions,
            "inverse-sigmoid ILD stress +/-90 deg",
            FIGURE_DIR / "error_histogram.png",
        ),
        "range_comparison": plot_range_comparison(
            primary_true,
            inverse_ild_predictions,
            stress_true,
            stress_inverse_ild_predictions,
            FIGURE_DIR / "range_comparison.png",
        ),
        "warped_ild_mapping": plot_warped_ild_mapping(
            primary_predictions,
            warp_params,
            inverse_params,
            FIGURE_DIR / "warped_ild_mapping.png",
        ),
        "warped_ild_scatter": plot_warped_ild_scatter(
            primary_predictions,
            warped_ild_predictions,
            inverse_ild_predictions,
            FIGURE_DIR / "warped_ild_scatter.png",
        ),
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
            "warped_ild_power": warp_params["power"],
            "warped_ild_sigma": warp_params["sigma"],
            "inverse_sigmoid_ild_gain": inverse_params["gain"],
            "inverse_sigmoid_ild_sigma": inverse_params["sigma"],
        },
        "metrics": metrics,
        "warped_ild_tuning": warp_params,
        "inverse_sigmoid_ild_tuning": inverse_params,
        "old_model_azimuth_mae_deg": OLD_MODEL_AZIMUTH_RESULTS,
        "primary_predictions": [
            {
                "true_azimuth_deg": item.true_azimuth_deg,
                "itd_prediction_deg": item.itd_prediction_deg,
                "ild_prediction_deg": item.ild_prediction_deg,
                "ild_warped_prediction_deg": float(warped_ild_predictions[index]),
                "ild_inverse_sigmoid_prediction_deg": float(inverse_ild_predictions[index]),
                "combined_prediction_deg": item.combined_prediction_deg,
            }
            for index, item in enumerate(primary_predictions)
        ],
        "stress_predictions": [
            {
                "true_azimuth_deg": item.true_azimuth_deg,
                "combined_prediction_deg": item.combined_prediction_deg,
                "ild_inverse_sigmoid_prediction_deg": float(stress_inverse_ild_predictions[index]),
            }
            for index, item in enumerate(stress_predictions)
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(config, primary_predictions, stress_predictions, metrics, warp_params, inverse_params, artifacts, elapsed_s)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
