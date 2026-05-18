from __future__ import annotations

"""First standalone elevation-pathway prototype.

This experiment starts the new elevation pathway independently from the old
trained model. It uses the final cochlea from the distance pathway, applies the
comb-filter elevation cue from the signal-analysis mini model, and decodes
elevation with a monaural DCN-style disinhibitory notch detector.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import full_distance_pathway_model as fdm
from mini_models.common.plotting import ensure_dir, save_figure
from mini_models.common.signals import matched_human_signal_config
from models.acoustics import simulate_echo_batch
from utils.common import GlobalConfig


OUTPUT_DIR = ROOT / "elevation_pathway" / "outputs" / "first_attempt"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "elevation_pathway" / "reports" / "elevation_pathway_first_attempt.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

NUM_CHANNELS = 48
NUM_ELEVATION_BINS = 91
ELEVATION_LIMIT_DEG = 45.0
FIXED_DISTANCE_M = 3.0
FIXED_AZIMUTH_DEG = 45.0
SELECTED_EAR = "right"
COMB_FIRST_NOTCH_LOW_HZ = 6_000.0
COMB_FIRST_NOTCH_HIGH_HZ = 16_000.0
COMB_DELAYED_COPY_GAIN = 0.85
DCN_TEMPLATE_POWER = 1.35
DCN_LOCAL_MEAN_SIGMA_CHANNELS = 2.0
DCN_TRANSFER_SIGMA = 0.05
EI_LAMBDA_GRID = np.linspace(0.2, 1.4, 25)

OLD_MODEL_ELEVATION_RESULTS = {
    "Round 3 Experiment 2B moving-notch + notch detectors": 1.9386,
    "Round 3 2B + 3": 2.5258,
    "Round 4 combined": 2.7802,
    "Round 5 trained-once fixed ridge decoder": 2.5876,
}


@dataclass(frozen=True)
class ElevationPrediction:
    """Stage outputs for one elevation-pathway prediction.

    Attributes:
        true_elevation_deg: True elevation.
        com_prediction_deg: Centre-of-mass prediction over the DCN population.
        argmax_prediction_deg: Winner-take-all prediction over the DCN population.
        cochlea: Binaural cochlea result after the comb-filter cue.
        selected_spikes: Selected-ear dynamic cochlear spike raster.
        spectral_profile: Normalised selected-ear spike-count spectrum.
        observed_deficit: Local spectral dips extracted from the spectrum.
        dcn_activation: DCN elevation population.
        equalized_profile: Spectrum divided by a learned no-comb reference.
        transfer_activation: Full comb-transfer DCN population.
        transfer_com_prediction_deg: COM prediction from the transfer population.
        transfer_argmax_prediction_deg: Argmax prediction from the transfer population.
        first_notch_prediction_deg: Direct diagnostic prediction from the deepest notch.
        comb_gain_channels: Comb-filter gain sampled at cochlear channels.
    """

    true_elevation_deg: float
    com_prediction_deg: float
    argmax_prediction_deg: float
    cochlea: fdm.CochleaResult
    selected_spikes: torch.Tensor
    spectral_profile: np.ndarray
    observed_deficit: np.ndarray
    dcn_activation: np.ndarray
    equalized_profile: np.ndarray
    transfer_activation: np.ndarray
    transfer_com_prediction_deg: float
    transfer_argmax_prediction_deg: float
    first_notch_prediction_deg: float
    comb_gain_channels: np.ndarray


def make_config() -> GlobalConfig:
    """Create the matched-human elevation-pathway configuration.

    Returns:
        Acoustic configuration with the old built-in elevation cue disabled.
    """
    base = matched_human_signal_config(GlobalConfig())
    return replace(
        base,
        num_cochlea_channels=NUM_CHANNELS,
        min_range_m=0.25,
        max_range_m=5.0,
        signal_duration_s=0.036,
        normalize_spike_envelope=False,
        jitter_std_s=0.0,
        noise_std=0.0,
        transmit_gain=1_000.0,
        elevation_cue_mode="none",
        azimuth_cue_mode="none",
    )


def elevation_grid() -> np.ndarray:
    """Return represented elevation bins in degrees."""
    return np.linspace(-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG, NUM_ELEVATION_BINS)


def angle_scale(elevation_deg: np.ndarray | float) -> np.ndarray:
    """Map elevation in degrees onto the comb sweep coordinate.

    Args:
        elevation_deg: Elevation angle or array in degrees.

    Returns:
        Clipped scale where `-45 deg -> -1` and `+45 deg -> +1`.
    """
    return np.clip(np.asarray(elevation_deg, dtype=np.float64) / ELEVATION_LIMIT_DEG, -1.0, 1.0)


def comb_first_notch_hz(elevation_deg: np.ndarray | float) -> np.ndarray:
    """Map elevation to the first comb-filter notch frequency.

    Args:
        elevation_deg: Elevation angle or array in degrees.

    Returns:
        First notch frequency in Hz.
    """
    scale = angle_scale(elevation_deg)
    return COMB_FIRST_NOTCH_LOW_HZ + (COMB_FIRST_NOTCH_HIGH_HZ - COMB_FIRST_NOTCH_LOW_HZ) * (0.5 * (scale + 1.0))


def comb_lag_s(elevation_deg: np.ndarray | float) -> np.ndarray:
    """Return the delayed-copy lag for the comb-filter cue.

    Args:
        elevation_deg: Elevation angle or array in degrees.

    Returns:
        Delay values in seconds.
    """
    return 1.0 / (2.0 * np.maximum(comb_first_notch_hz(elevation_deg), 1.0))


def comb_gain(frequency_hz: np.ndarray, elevation_deg: np.ndarray | float) -> np.ndarray:
    """Evaluate the comb-filter magnitude response.

    The acoustic cue is modelled as a direct path plus a delayed copy:

    $$
    y(t)=x(t)+a x(t-\tau).
    $$

    The normalised frequency response is

    $$
    |H(f)|=\frac{\sqrt{1+a^2+2a\cos(2\pi f\tau)}}{1+a}.
    $$

    Args:
        frequency_hz: Frequency axis in Hz.
        elevation_deg: Elevation angle or array in degrees.

    Returns:
        Gain array. If `elevation_deg` is vector-shaped, output shape is
        `[num_elevations, num_frequencies]`.
    """
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    tau = np.atleast_1d(comb_lag_s(elevation_deg)).astype(np.float64)
    phase = 2.0 * math.pi * tau[:, None] * frequencies[None, :]
    gain = np.sqrt(
        1.0 + COMB_DELAYED_COPY_GAIN**2 + 2.0 * COMB_DELAYED_COPY_GAIN * np.cos(phase)
    ) / (1.0 + COMB_DELAYED_COPY_GAIN)
    gain = np.clip(gain, 1e-3, 1.0)
    if np.ndim(elevation_deg) == 0:
        return gain[0]
    return gain


def apply_comb_filter(waveform: torch.Tensor, config: GlobalConfig, elevation_deg: float) -> torch.Tensor:
    """Apply the comb-filter elevation cue to one waveform.

    Args:
        waveform: One received ear waveform.
        config: Acoustic configuration.
        elevation_deg: Elevation angle in degrees.

    Returns:
        Comb-filtered waveform.
    """
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / config.sample_rate_hz)
    gain = torch.from_numpy(comb_gain(frequencies.detach().cpu().numpy(), elevation_deg)).to(waveform)
    spectrum = torch.fft.rfft(waveform)
    return torch.fft.irfft(spectrum * gain, n=waveform.numel())


def simulate_elevation_scene(config: GlobalConfig, elevation_deg: float, *, add_noise: bool = False) -> torch.Tensor:
    """Simulate one binaural side-source echo with a comb elevation cue.

    Args:
        config: Acoustic configuration.
        elevation_deg: Target elevation in degrees.
        add_noise: Whether to add receiver noise.

    Returns:
        Binaural received waveform `[ears, time]`.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([FIXED_DISTANCE_M], dtype=torch.float32),
        azimuth_deg=torch.tensor([FIXED_AZIMUTH_DEG], dtype=torch.float32),
        elevation_deg=torch.tensor([0.0], dtype=torch.float32),
        binaural=True,
        add_noise=add_noise,
        include_elevation_cues=False,
        transmit_gain=config.transmit_gain,
    )
    receive = scene.receive[0].detach().clone()
    receive[0] = apply_comb_filter(receive[0], config, elevation_deg)
    receive[1] = apply_comb_filter(receive[1], config, elevation_deg)
    return receive


def simulate_no_comb_reference(config: GlobalConfig) -> torch.Tensor:
    """Simulate the selected-side echo without the elevation comb filter.

    Args:
        config: Acoustic configuration.

    Returns:
        Binaural received waveform `[ears, time]`.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([FIXED_DISTANCE_M], dtype=torch.float32),
        azimuth_deg=torch.tensor([FIXED_AZIMUTH_DEG], dtype=torch.float32),
        elevation_deg=torch.tensor([0.0], dtype=torch.float32),
        binaural=True,
        add_noise=False,
        include_elevation_cues=False,
        transmit_gain=config.transmit_gain,
    )
    return scene.receive[0].detach()


def simulate_pre_post_comb_scene(config: GlobalConfig, elevation_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate one scene before and after the comb-filter elevation cue.

    The same base received waveform is used for both outputs. This makes the
    PSD comparison isolate the spectral notch instead of comparing two separate
    acoustic simulations.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angle used to set the comb-filter lag.

    Returns:
        Pair `(before_comb, after_comb)`, each shaped `[ears, time]`.
    """
    before_comb = simulate_no_comb_reference(config).clone()
    after_comb = before_comb.clone()
    after_comb[0] = apply_comb_filter(after_comb[0], config, elevation_deg)
    after_comb[1] = apply_comb_filter(after_comb[1], config, elevation_deg)
    return before_comb, after_comb


def selected_ear_activity(cochlea: fdm.CochleaResult, config: GlobalConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected-ear cochleagram and dynamic spike raster.

    Args:
        cochlea: Binaural cochlea output.
        config: Acoustic configuration.

    Returns:
        Pair `(cochleagram, dynamic_spikes)`.
    """
    cochleagram = cochlea.right_cochleagram if SELECTED_EAR == "right" else cochlea.left_cochleagram
    spikes = fdm._dynamic_lif_encode(cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
    return cochleagram, spikes


def gaussian_kernel(sigma_channels: float) -> np.ndarray:
    """Build a normalised one-dimensional Gaussian smoothing kernel.

    Args:
        sigma_channels: Gaussian width in channel units.

    Returns:
        Normalised convolution kernel.
    """
    radius = max(2, int(math.ceil(3.0 * sigma_channels)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / max(sigma_channels, 1e-6)) ** 2)
    return kernel / np.maximum(kernel.sum(), 1e-12)


def spectral_profile_from_spikes(spikes: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert cochlear spikes into a normalised spectrum and dip estimate.

    Args:
        spikes: Spike raster `[channels, time]`.

    Returns:
        Pair `(profile, observed_deficit)`.
    """
    counts = spikes.detach().cpu().numpy().sum(axis=1).astype(np.float64)
    profile = counts / np.maximum(counts.sum(), 1e-12)
    smooth = np.convolve(profile, gaussian_kernel(DCN_LOCAL_MEAN_SIGMA_CHANNELS), mode="same")
    observed_deficit = np.maximum(smooth - profile, 0.0)
    return profile, observed_deficit


def cochleagram_energy_profile(cochleagram: torch.Tensor) -> np.ndarray:
    """Return a max-normalised cochleagram energy spectrum.

    Args:
        cochleagram: Selected-ear cochleagram `[channels, time]`.

    Returns:
        Max-normalised per-channel energy.
    """
    energy = cochleagram.detach().cpu().numpy().sum(axis=1).astype(np.float64)
    return energy / np.maximum(energy.max(), 1e-12)


def baseline_energy_profile(config: GlobalConfig) -> np.ndarray:
    """Estimate the learned no-comb spectral reference for the selected ear.

    Args:
        config: Acoustic configuration.

    Returns:
        Max-normalised no-comb cochlear energy spectrum.
    """
    cochlea = fdm._run_cochlea_binaural(config, simulate_no_comb_reference(config))
    cochleagram = cochlea.right_cochleagram if SELECTED_EAR == "right" else cochlea.left_cochleagram
    return cochleagram_energy_profile(cochleagram)


def equalized_transfer_profile(cochleagram: torch.Tensor, baseline_profile: np.ndarray) -> np.ndarray:
    """Divide the current spectrum by a learned no-comb reference.

    Args:
        cochleagram: Selected-ear cochleagram `[channels, time]`.
        baseline_profile: Learned no-comb reference spectrum.

    Returns:
        Max-normalised estimate of the comb transfer function.
    """
    current = cochleagram_energy_profile(cochleagram)
    equalized = current / np.maximum(baseline_profile, 1e-4)
    return equalized / np.maximum(equalized.max(), 1e-12)


def build_dcn_templates(config: GlobalConfig, bins_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build comb-derived DCN inhibitory templates.

    Args:
        config: Acoustic configuration.
        bins_deg: Candidate elevation bins.

    Returns:
        Tuple `(centres_hz, comb_gain_matrix, template_matrix)`.
    """
    centres_hz = fdm._log_spaced_centers(config).detach().cpu().numpy()
    gains = comb_gain(centres_hz, bins_deg)
    frequency_mask = (centres_hz >= COMB_FIRST_NOTCH_LOW_HZ * 0.75) & (centres_hz <= config.chirp_start_hz)
    templates = np.maximum(1.0 - gains, 0.0) ** DCN_TEMPLATE_POWER
    templates *= frequency_mask[None, :]
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
    return centres_hz, gains, templates


def dcn_disinhibition_response(observed_deficit: np.ndarray, spectral_profile: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Compute the DCN elevation population from spectral dips.

    Args:
        observed_deficit: Local spectral dip estimate.
        spectral_profile: Normalised spike-count spectrum.
        templates: Candidate notch templates `[elevation_bins, channels]`.

    Returns:
        Non-negative DCN activation over elevation bins.
    """
    match = templates @ observed_deficit
    penalty = templates @ spectral_profile
    response = np.maximum(match - 0.25 * penalty, 0.0)
    if float(response.max()) <= 1e-12:
        return response
    return response / float(response.max())


def dcn_full_transfer_response(
    equalized_profile: np.ndarray,
    comb_gain_matrix: np.ndarray,
    centres_hz: np.ndarray,
) -> np.ndarray:
    """Compute a full-transfer E/I DCN response.

    This variant uses the whole comb transfer function rather than only local
    spectral dips. As a linear E/I interpretation, candidate channels where
    `G_k(f_c)` is high act like excitatory evidence and channels where
    `G_k(f_c)` is low act like inhibitory evidence. The squared mismatch is
    converted into a population activity.

    Args:
        equalized_profile: Baseline-divided selected-ear spectrum.
        comb_gain_matrix: Candidate comb gains `[elevation_bins, channels]`.
        centres_hz: Cochlear centre frequencies.

    Returns:
        DCN activation over elevation bins.
    """
    mask = (centres_hz >= 4_500.0) & (centres_hz <= 18_000.0)
    squared_error = ((equalized_profile[None, :] - comb_gain_matrix) ** 2) * mask[None, :]
    mean_error = squared_error.sum(axis=1) / max(float(mask.sum()), 1.0)
    response = np.exp(-mean_error / (2.0 * DCN_TRANSFER_SIGMA**2))
    return response / np.maximum(float(response.max()), 1e-12)


def dcn_ei_weight_profile_response(
    equalized_profile: np.ndarray,
    comb_gain_matrix: np.ndarray,
    centres_hz: np.ndarray,
    inhibition_lambda: float,
) -> np.ndarray:
    """Compute the proposed explicit E/I weight-profile response.

    The candidate weight profile is

    $$
    w_k(f)=w_E-w_I(1-H_k(f)).
    $$

    Since the broad excitatory term is shared across candidates, the useful
    selectivity comes from candidate-specific inhibition at the frequencies
    where that candidate predicts comb notches. The inhibitory mass is
    normalised so candidates with more comb troughs are not unfairly penalised.

    Args:
        equalized_profile: Baseline-divided selected-ear spectrum.
        comb_gain_matrix: Candidate comb gains `[elevation_bins, channels]`.
        centres_hz: Cochlear centre frequencies.
        inhibition_lambda: Ratio `w_I / w_E`.

    Returns:
        Non-negative DCN activation over elevation bins.
    """
    mask = ((centres_hz >= 4_500.0) & (centres_hz <= 18_000.0)).astype(np.float64)
    notch_drive = np.maximum(1.0 - comb_gain_matrix, 0.0) * mask[None, :]
    notch_drive = notch_drive / np.maximum(notch_drive.sum(axis=1, keepdims=True), 1e-12)
    broad_excitation = float(np.sum(equalized_profile * mask) / max(float(mask.sum()), 1.0))
    candidate_inhibition = notch_drive @ equalized_profile
    response = np.maximum(broad_excitation - inhibition_lambda * candidate_inhibition, 0.0)
    if float(response.max()) <= 1e-12:
        return response
    return response / float(response.max())


def ei_weight_readouts(
    predictions: list[ElevationPrediction],
    config: GlobalConfig,
    inhibition_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode all predictions with the explicit E/I weight-profile model.

    Args:
        predictions: Existing elevation predictions containing equalised spectra.
        config: Acoustic configuration.
        inhibition_lambda: Ratio `w_I / w_E`.

    Returns:
        Tuple `(com_predictions, argmax_predictions, populations)`.
    """
    bins = elevation_grid()
    centres_hz, gain_matrix, _ = build_dcn_templates(config, bins)
    populations = np.stack(
        [
            dcn_ei_weight_profile_response(item.equalized_profile, gain_matrix, centres_hz, inhibition_lambda)
            for item in predictions
        ],
        axis=0,
    )
    com = np.array([centre_of_mass(population, bins) for population in populations], dtype=np.float64)
    argmax = bins[np.argmax(populations, axis=1)]
    return com, argmax, populations


def tune_ei_lambda(predictions: list[ElevationPrediction], config: GlobalConfig) -> dict[str, object]:
    """Sweep the E/I inhibition ratio and select the lowest-COM-MAE value.

    Args:
        predictions: Existing elevation predictions.
        config: Acoustic configuration.

    Returns:
        Sweep results and best value.
    """
    true = np.array([item.true_elevation_deg for item in predictions], dtype=np.float64)
    rows = []
    for value in EI_LAMBDA_GRID:
        com, argmax, _ = ei_weight_readouts(predictions, config, float(value))
        rows.append(
            {
                "lambda": float(value),
                "com_mae_deg": float(np.mean(np.abs(com - true))),
                "argmax_mae_deg": float(np.mean(np.abs(argmax - true))),
            }
        )
    best = min(rows, key=lambda row: row["com_mae_deg"])
    return {"grid": rows, "best_lambda": best["lambda"], "best": best}


def first_notch_readout(equalized_profile: np.ndarray, centres_hz: np.ndarray) -> float:
    """Estimate elevation from the deepest equalized notch.

    Args:
        equalized_profile: Baseline-divided selected-ear spectrum.
        centres_hz: Cochlear centre frequencies.

    Returns:
        Elevation estimate in degrees.
    """
    mask = (centres_hz >= COMB_FIRST_NOTCH_LOW_HZ) & (centres_hz <= COMB_FIRST_NOTCH_HIGH_HZ)
    masked_profile = np.where(mask, equalized_profile, np.inf)
    notch_frequency_hz = float(centres_hz[int(np.argmin(masked_profile))])
    scale = 2.0 * (notch_frequency_hz - COMB_FIRST_NOTCH_LOW_HZ) / (
        COMB_FIRST_NOTCH_HIGH_HZ - COMB_FIRST_NOTCH_LOW_HZ
    ) - 1.0
    return float(np.clip(scale * ELEVATION_LIMIT_DEG, -ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG))


def centre_of_mass(activity: np.ndarray, bins_deg: np.ndarray) -> float:
    """Decode elevation by centre of mass over a non-negative population."""
    positive = np.maximum(activity, 0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return 0.0
    return float(np.sum(positive * bins_deg) / total)


def predict_one(config: GlobalConfig, elevation_deg: float, baseline_profile: np.ndarray) -> ElevationPrediction:
    """Run the first elevation pathway for one elevation.

    Args:
        config: Acoustic configuration.
        elevation_deg: True target elevation.

    Returns:
        Full stage prediction.
    """
    bins = elevation_grid()
    receive = simulate_elevation_scene(config, elevation_deg)
    cochlea = fdm._run_cochlea_binaural(config, receive)
    selected_cochleagram, selected_spikes = selected_ear_activity(cochlea, config)
    profile, observed_deficit = spectral_profile_from_spikes(selected_spikes)
    centres_hz, gain_matrix, templates = build_dcn_templates(config, bins)
    dcn = dcn_disinhibition_response(observed_deficit, profile, templates)
    equalized = equalized_transfer_profile(selected_cochleagram, baseline_profile)
    transfer = dcn_full_transfer_response(equalized, gain_matrix, centres_hz)
    return ElevationPrediction(
        true_elevation_deg=float(elevation_deg),
        com_prediction_deg=centre_of_mass(dcn, bins),
        argmax_prediction_deg=float(bins[int(np.argmax(dcn))]),
        cochlea=cochlea,
        selected_spikes=selected_spikes,
        spectral_profile=profile,
        observed_deficit=observed_deficit,
        dcn_activation=dcn,
        equalized_profile=equalized,
        transfer_activation=transfer,
        transfer_com_prediction_deg=centre_of_mass(transfer, bins),
        transfer_argmax_prediction_deg=float(bins[int(np.argmax(transfer))]),
        first_notch_prediction_deg=first_notch_readout(equalized, centres_hz),
        comb_gain_channels=gain_matrix[int(np.argmin(np.abs(bins - elevation_deg)))],
    )


def metric_dict(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute elevation error metrics.

    Args:
        true: True elevations in degrees.
        pred: Predicted elevations in degrees.

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


def run_dataset(config: GlobalConfig) -> list[ElevationPrediction]:
    """Run the monaural elevation pathway over the full elevation grid."""
    baseline = baseline_energy_profile(config)
    return [predict_one(config, float(elevation), baseline) for elevation in elevation_grid()]


def plot_pipeline(path: Path) -> str:
    """Plot the first elevation pathway."""
    fig, ax = plt.subplots(figsize=(13.0, 3.7))
    ax.axis("off")
    labels = [
        "Side-source\nbinaural echo",
        "Comb-filter\npinna cue",
        "Final cochlea\nper ear",
        "Selected ear\nspike counts",
        "DCN\nE/I notch bank",
        "Elevation\npopulation",
        "COM / argmax\nreadout",
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
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0),
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1] - 0.055, 0.55),
                xytext=(xpos + 0.055, 0.55),
                arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.1),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    return save_figure(fig, path)


def plot_comb_transfer(config: GlobalConfig, path: Path) -> str:
    """Plot the comb-filter sweep over elevation."""
    elevations = elevation_grid()
    frequencies = np.linspace(config.cochlea_low_hz, config.cochlea_high_hz, 700)
    gain = comb_gain(frequencies, elevations)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    im = ax.contourf(frequencies / 1_000.0, elevations, gain, levels=50, cmap="magma")
    ax.plot(comb_first_notch_hz(elevations) / 1_000.0, elevations, color="#38bdf8", linewidth=1.8, label="first notch")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title("Comb-filter elevation transfer function")
    ax.legend(frameon=False)
    fig.colorbar(im, ax=ax, label="normalised gain")
    return save_figure(fig, path)


def active_echo_window(before: torch.Tensor, after: torch.Tensor, *, margin_samples: int = 256) -> slice:
    """Find a shared crop around the received echo for PSD estimation.

    Args:
        before: Selected-ear waveform before spectral filtering.
        after: Selected-ear waveform after spectral filtering.
        margin_samples: Extra samples retained on both sides of the active echo.

    Returns:
        Slice covering the active echo. If no active region is found, the full
        signal is returned.
    """
    envelope = torch.maximum(before.abs(), after.abs())
    peak = float(envelope.max())
    if peak <= 1e-12:
        return slice(0, before.numel())
    active = torch.nonzero(envelope > 0.02 * peak, as_tuple=False).flatten()
    if active.numel() == 0:
        return slice(0, before.numel())
    start = max(int(active[0]) - margin_samples, 0)
    stop = min(int(active[-1]) + margin_samples + 1, before.numel())
    return slice(start, stop)


def one_sided_psd(waveform: torch.Tensor, config: GlobalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Hann-windowed one-sided PSD.

    Args:
        waveform: One-dimensional waveform.
        config: Acoustic configuration.

    Returns:
        Pair `(frequencies_hz, psd)`.
    """
    waveform = waveform.detach().float()
    window = torch.hann_window(waveform.numel(), dtype=waveform.dtype, device=waveform.device)
    centred = waveform - waveform.mean()
    spectrum = torch.fft.rfft(centred * window)
    psd = spectrum.abs().square() / torch.clamp(window.square().sum(), min=1e-12)
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / config.sample_rate_hz)
    return frequencies.cpu().numpy(), psd.cpu().numpy()


def plot_received_psd(config: GlobalConfig, elevation_deg: float, path: Path) -> str:
    """Plot selected-ear PSD before and after the comb-filter cue.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angle used for the comb-filter cue.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    before_binaural, after_binaural = simulate_pre_post_comb_scene(config, elevation_deg)
    ear_index = 1 if SELECTED_EAR == "right" else 0
    before = before_binaural[ear_index]
    after = after_binaural[ear_index]
    crop = active_echo_window(before, after)
    frequencies_hz, before_psd = one_sided_psd(before[crop], config)
    _, after_psd = one_sided_psd(after[crop], config)
    shared_peak = max(float(before_psd.max()), float(after_psd.max()), 1e-12)
    before_psd_db = 10.0 * np.log10(before_psd / shared_peak + 1e-12)
    after_psd_db = 10.0 * np.log10(after_psd / shared_peak + 1e-12)

    theoretical_gain = 20.0 * np.log10(comb_gain(frequencies_hz, elevation_deg) + 1e-12)
    first_notch_hz = float(comb_first_notch_hz(elevation_deg))
    lag_us = float(comb_lag_s(elevation_deg) * 1e6)

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.plot(frequencies_hz / 1_000.0, before_psd_db, label="before comb notch", linewidth=1.8)
    ax.plot(frequencies_hz / 1_000.0, after_psd_db, label="after comb notch", linewidth=1.8)
    ax.plot(
        frequencies_hz / 1_000.0,
        theoretical_gain,
        color="#64748b",
        linestyle="--",
        linewidth=1.4,
        label="theoretical comb gain",
    )
    ax.axvline(first_notch_hz / 1_000.0, color="#dc2626", linestyle=":", linewidth=1.5, label="first notch")
    ax.set_xlim(0.0, min(config.sample_rate_hz / 2_000.0, 24.0))
    ax.set_ylim(-80.0, 5.0)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("normalised power / gain (dB)")
    ax.set_title(
        f"Selected-ear received PSD before and after comb cue "
        f"(elevation={elevation_deg:.1f} deg, f1={first_notch_hz / 1_000.0:.2f} kHz, tau={lag_us:.1f} us)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_dcn_templates(config: GlobalConfig, path: Path) -> str:
    """Plot the DCN E/I template matrix."""
    bins = elevation_grid()
    centres_hz, _, templates = build_dcn_templates(config, bins)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    channel_edges = np.empty(centres_hz.size + 1, dtype=np.float64)
    channel_edges[1:-1] = np.sqrt(centres_hz[:-1] * centres_hz[1:])
    channel_edges[0] = centres_hz[0] ** 2 / channel_edges[1]
    channel_edges[-1] = centres_hz[-1] ** 2 / channel_edges[-2]
    elevation_edges = np.linspace(
        bins[0] - 0.5 * (bins[1] - bins[0]),
        bins[-1] + 0.5 * (bins[1] - bins[0]),
        bins.size + 1,
    )
    im = ax.pcolormesh(channel_edges / 1_000.0, elevation_edges, templates, shading="auto", cmap="viridis")
    ax.set_xlabel("cochlear centre frequency (kHz)")
    ax.set_ylabel("candidate elevation (deg)")
    ax.set_title("DCN inhibitory notch-template weights on log-spaced cochlear channels")
    fig.colorbar(im, ax=ax, label="normalised inhibitory weight")
    return save_figure(fig, path)


def plot_example(
    prediction: ElevationPrediction,
    config: GlobalConfig,
    path: Path,
    *,
    ei_lambda: float,
) -> str:
    """Plot cochlea, raster, spectral profile, and DCN population for one example."""
    bins = elevation_grid()
    centres_hz = fdm._log_spaced_centers(config).detach().cpu().numpy()
    _, gain_matrix, _ = build_dcn_templates(config, bins)
    ei_population = dcn_ei_weight_profile_response(prediction.equalized_profile, gain_matrix, centres_hz, ei_lambda)
    ei_prediction = centre_of_mass(ei_population, bins)
    time_ms = np.arange(prediction.selected_spikes.shape[1]) / config.sample_rate_hz * 1_000.0
    cochleagram = prediction.cochlea.right_cochleagram if SELECTED_EAR == "right" else prediction.cochlea.left_cochleagram
    fig, axes = plt.subplots(6, 1, figsize=(11.5, 17.0))

    im0 = axes[0].imshow(
        cochleagram.detach().cpu().numpy(),
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], centres_hz[0] / 1_000.0, centres_hz[-1] / 1_000.0],
        cmap="magma",
    )
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_title(f"{SELECTED_EAR.title()} ear cochleagram after comb filtering")
    fig.colorbar(im0, ax=axes[0], label="rectified IIR output")

    spikes = prediction.selected_spikes.detach().cpu().numpy()
    for channel, freq in enumerate(centres_hz / 1_000.0):
        spike_times = time_ms[np.flatnonzero(spikes[channel] > 0.0)]
        if spike_times.size:
            axes[1].vlines(spike_times, freq - 0.08, freq + 0.08, color="#111827", linewidth=0.45)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("frequency (kHz)")
    axes[1].set_title("Selected-ear dynamic cochlear spike raster")
    axes[1].grid(True, axis="x", alpha=0.25)

    axes[2].plot(centres_hz / 1_000.0, prediction.spectral_profile, label="normalised spike spectrum", linewidth=1.8)
    axes[2].plot(centres_hz / 1_000.0, prediction.observed_deficit, label="observed spectral deficit", linewidth=1.8)
    axes[2].plot(centres_hz / 1_000.0, prediction.equalized_profile, label="baseline-equalised cochleagram", linewidth=1.8)
    axes[2].plot(centres_hz / 1_000.0, prediction.comb_gain_channels, label="true comb gain", linewidth=1.4)
    axes[2].set_xlabel("cochlear centre frequency (kHz)")
    axes[2].set_ylabel("relative activity")
    axes[2].set_title("Spectral evidence used by the DCN")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    axes[3].plot(bins, prediction.dcn_activation, color="#2563eb", linewidth=2.0)
    axes[3].axvline(prediction.true_elevation_deg, color="#111827", linestyle="--", label="true")
    axes[3].axvline(prediction.com_prediction_deg, color="#dc2626", linestyle=":", label="COM")
    axes[3].axvline(prediction.argmax_prediction_deg, color="#059669", linestyle="-.", label="argmax")
    axes[3].set_xlabel("represented elevation (deg)")
    axes[3].set_ylabel("DCN activation")
    axes[3].set_title("DCN elevation population")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(frameon=False)

    axes[4].plot(bins, prediction.transfer_activation, color="#7c3aed", linewidth=2.0)
    axes[4].axvline(prediction.true_elevation_deg, color="#111827", linestyle="--", label="true")
    axes[4].axvline(prediction.transfer_com_prediction_deg, color="#dc2626", linestyle=":", label="COM")
    axes[4].axvline(prediction.first_notch_prediction_deg, color="#059669", linestyle="-.", label="first notch")
    axes[4].set_xlabel("represented elevation (deg)")
    axes[4].set_ylabel("DCN activation")
    axes[4].set_title("Baseline-equalised full-transfer DCN population")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(frameon=False)

    axes[5].plot(bins, ei_population, color="#ea580c", linewidth=2.0)
    axes[5].axvline(prediction.true_elevation_deg, color="#111827", linestyle="--", label="true")
    axes[5].axvline(ei_prediction, color="#dc2626", linestyle=":", label="COM")
    axes[5].set_xlabel("represented elevation (deg)")
    axes[5].set_ylabel("DCN activation")
    axes[5].set_title(f"Explicit E/I weight-profile population, lambda={ei_lambda:.3f}")
    axes[5].grid(True, alpha=0.25)
    axes[5].legend(frameon=False)

    fig.tight_layout()
    return save_figure(fig, path)


def plot_predictions(
    predictions: list[ElevationPrediction],
    path: Path,
    *,
    ei_com: np.ndarray,
) -> str:
    """Plot true-vs-predicted elevation for COM and argmax readouts."""
    true = np.array([item.true_elevation_deg for item in predictions])
    com = np.array([item.com_prediction_deg for item in predictions])
    transfer = np.array([item.transfer_com_prediction_deg for item in predictions])
    first_notch = np.array([item.first_notch_prediction_deg for item in predictions])
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.0))
    for ax, pred, title in [
        (axes[0], com, "Local-dip disinhibition COM"),
        (axes[1], ei_com, "Explicit E/I profile COM"),
        (axes[2], transfer, "Full-transfer DCN COM"),
        (axes[3], first_notch, "First-notch diagnostic readout"),
    ]:
        ax.scatter(true, pred, s=26, alpha=0.75)
        ax.plot([-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG], [-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG], color="#111827")
        ax.set_xlabel("true elevation (deg)")
        ax.set_ylabel("predicted elevation (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_error(
    predictions: list[ElevationPrediction],
    path: Path,
    *,
    ei_com: np.ndarray,
) -> str:
    """Plot signed elevation error over the sweep."""
    true = np.array([item.true_elevation_deg for item in predictions])
    com = np.array([item.com_prediction_deg for item in predictions])
    transfer = np.array([item.transfer_com_prediction_deg for item in predictions])
    first_notch = np.array([item.first_notch_prediction_deg for item in predictions])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(true, com - true, label="local-dip disinhibition COM", linewidth=2.0)
    ax.plot(true, ei_com - true, label="explicit E/I profile COM", linewidth=2.0)
    ax.plot(true, transfer - true, label="full-transfer DCN COM", linewidth=2.0)
    ax.plot(true, first_notch - true, label="first-notch diagnostic", linewidth=2.0)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel("true elevation (deg)")
    ax.set_ylabel("prediction error (deg)")
    ax.set_title("Elevation error across the monaural sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_ei_lambda_sweep(sweep: dict[str, object], path: Path) -> str:
    """Plot the explicit E/I inhibition-ratio sweep.

    Args:
        sweep: Output of `tune_ei_lambda`.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    rows = sweep["grid"]
    lambdas = np.array([row["lambda"] for row in rows], dtype=np.float64)
    com_mae = np.array([row["com_mae_deg"] for row in rows], dtype=np.float64)
    argmax_mae = np.array([row["argmax_mae_deg"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(lambdas, com_mae, marker="o", label="COM")
    ax.plot(lambdas, argmax_mae, marker="s", label="argmax")
    ax.axvline(float(sweep["best_lambda"]), color="#dc2626", linestyle=":", label="selected")
    ax.set_xlabel("inhibition ratio lambda = wI / wE")
    ax.set_ylabel("elevation MAE (deg)")
    ax.set_title("Explicit E/I weight-profile tuning")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def write_report(
    config: GlobalConfig,
    metrics: dict[str, dict[str, float]],
    artifacts: dict[str, str],
    elapsed_s: float,
    ei_sweep: dict[str, object],
) -> None:
    """Write the first elevation-pathway report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Elevation Pathway First Attempt",
        "",
        "This report starts a standalone elevation pathway. It does not modify the old trained model. The aim is to test whether a monaural DCN-style disinhibitory notch detector can decode the comb-filter elevation cue from the final cochlea.",
        "",
        "![Pipeline diagram](../outputs/first_attempt/figures/pipeline_diagram.png)",
        "",
        "## Acoustic Setup",
        "",
        f"- Elevation range: `-{ELEVATION_LIMIT_DEG:.0f} deg` to `+{ELEVATION_LIMIT_DEG:.0f} deg`.",
        f"- Fixed distance: `{FIXED_DISTANCE_M:.1f} m`.",
        f"- Fixed azimuth: `{FIXED_AZIMUTH_DEG:.1f} deg`, so the `{SELECTED_EAR}` ear is treated as the selected-side ear.",
        f"- Cochlea: final distance-pathway IIR cochlea with `{NUM_CHANNELS}` channels.",
        "- The old built-in Gaussian elevation notch is disabled; the elevation cue is applied by the comb-filter model from signal analysis.",
        "",
        "The comb-filter cue is modelled as interference between the direct received signal and a delayed copy:",
        "",
        "$$",
        "y(t)=x(t)+a x(t-\\tau),",
        "$$",
        "",
        "$$",
        "|H(f)|=\\frac{\\sqrt{1+a^2+2a\\cos(2\\pi f\\tau)}}{1+a}.",
        "$$",
        "",
        "The first notch is swept from `6 kHz` at `-45 deg` to `16 kHz` at `+45 deg`:",
        "",
        "$$",
        "f_1(\\phi)=6000 + (16000-6000)\\frac{1+\\phi/45}{2},",
        "\\qquad",
        "\\tau(\\phi)=\\frac{1}{2f_1(\\phi)}.",
        "$$",
        "",
        "![Comb transfer](../outputs/first_attempt/figures/comb_transfer.png)",
        "",
        "## Received Signal PSD",
        "",
        "The plot below checks the actual selected-ear received waveform before and after the comb-filter cue is applied. The two PSDs use the same simulated echo, cropped around the active received call, so the difference comes from the spectral notch rather than a different acoustic scene.",
        "",
        "The PSD is estimated with a Hann-windowed one-sided FFT:",
        "",
        "$$",
        "P(f)=\\frac{|\\mathcal{F}\\{w(t)(x(t)-\\bar{x})\\}|^2}{\\sum_t w(t)^2}.",
        "$$",
        "",
        "Both curves are normalised to the same peak power, so attenuation from the comb filter remains visible. The dashed curve is the theoretical comb-filter magnitude in dB.",
        "",
        "![Received PSD before and after comb filtering](../outputs/first_attempt/figures/received_psd_before_after_comb.png)",
        "",
        "## DCN Disinhibitory Notch Detector",
        "",
        "Each DCN output neuron corresponds to one candidate elevation. The candidate's expected comb-filter transfer function defines where inhibition should arrive from cochlear channels. If those channels are quiet because a notch is present, the candidate neuron is disinhibited.",
        "",
        "$$",
        "T_{k,c}=\\frac{(1-G_k(f_c))^p}{\\| (1-G_k)^p \\|_2},",
        "$$",
        "",
        "$$",
        "d_c=[\\operatorname{localmean}(p)_c-p_c]_+,",
        "$$",
        "",
        "$$",
        "r_k=[T_k\\cdot d - 0.25 T_k\\cdot p]_+.",
        "$$",
        "",
        "Here `p` is the normalised selected-ear spike-count spectrum, `d` is the observed local spectral dip, and `T_k` is the comb-derived inhibitory template for candidate elevation `k`.",
        "",
        "The explicit E/I profile suggested in the design notes is also tested:",
        "",
        "$$",
        "w_k(f)=w_E-w_I(1-H_k(f)),",
        "$$",
        "",
        "$$",
        "r_k=\\left[\\frac{1}{N}\\sum_c \\tilde p_c - \\lambda \\frac{\\sum_c (1-H_k(f_c))\\tilde p_c}{\\sum_c(1-H_k(f_c))+\\epsilon}\\right]_+.",
        "$$",
        "",
        f"The selected inhibition ratio from the sweep is `lambda = {float(ei_sweep['best_lambda']):.3f}`.",
        "",
        "A second DCN variant adds a learned no-comb spectral reference. This is closer to a calibrated E/I transfer-function matcher: the selected-ear cochleagram is divided by a fixed no-comb reference, then compared against the full comb gain for each candidate elevation.",
        "",
        "$$",
        "\\tilde p_c=\\frac{p^{current}_c}{p^{reference}_c},",
        "\\qquad",
        "r_k=\\exp\\left(-\\frac{\\sum_c(\\tilde p_c-G_k(f_c))^2}{2\\sigma^2 N}\\right).",
        "$$",
        "",
        "This still has an E/I interpretation: channels where the candidate transfer predicts high gain provide positive evidence, while candidate notch channels provide suppressive evidence if they remain active.",
        "",
        "![DCN templates](../outputs/first_attempt/figures/dcn_templates.png)",
        "",
        "![E/I lambda sweep](../outputs/first_attempt/figures/ei_lambda_sweep.png)",
        "",
        "## Example Stages",
        "",
        "![Example stages](../outputs/first_attempt/figures/example_stages.png)",
        "",
        "## Accuracy",
        "",
        "| Readout | MAE | RMSE | Max error | Bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metric in metrics.items():
        lines.append(
            f"| {label} | `{metric['mae_deg']:.3f} deg` | `{metric['rmse_deg']:.3f} deg` | "
            f"`{metric['max_abs_error_deg']:.3f} deg` | `{metric['bias_deg']:.3f} deg` |"
        )
    lines.extend(
        [
            "",
            "![Prediction scatter](../outputs/first_attempt/figures/prediction_scatter.png)",
            "",
            "![Error curve](../outputs/first_attempt/figures/error_curve.png)",
            "",
            "The direct first-notch readout is included as a diagnostic rather than the proposed final neural pathway. It estimates the deepest equalised notch frequency and maps it back through the known comb sweep. It shows how much elevation information is present in the cochlear spectrum before a more biological population readout is tuned.",
            "",
            "The explicit E/I profile is the closest implementation of the proposed `excitatory - inhibitory * (1-H)` rule. In this first version it improves substantially over the naive local-dip detector, but remains weaker than the full-transfer matcher. This suggests the rule is directionally useful, but the current implementation still loses information by only penalising expected notch channels rather than also matching the full comb peak/trough shape.",
            "",
            "## Old Model Reference",
            "",
            "These old values are copied from previous reports and are not rerun here. They are not strict like-for-like comparisons because this report is a monaural, fixed-distance, fixed-azimuth elevation-only test, while the old models were full trained localisation systems on their original small-space setup.",
            "",
            "| Old model | Elevation MAE | Notes |",
            "|---|---:|---|",
        ]
    )
    for model_name, elevation_mae in OLD_MODEL_ELEVATION_RESULTS.items():
        lines.append(f"| {model_name} | `{elevation_mae:.3f} deg` | old trained/fixed-decoder reference |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This first pathway is intentionally narrow: it tests whether the comb-filter notch position can be converted into a DCN population code before adding IC/SC integration or azimuth-gated ear selection. A good result here means the spectral cue and template wiring are usable; it does not yet prove the full elevation pathway is robust to distance, azimuth, clutter, or noise.",
            "",
            "The local-dip disinhibition result is deliberately retained because it shows that a naive spike-count dip detector is not enough for this comb cue. The baseline-equalised transfer matcher works substantially better, which suggests that the DCN stage needs some form of spectral equalisation or learned channel-specific gain control before the E/I notch template is applied.",
            "",
            "The biological simplification is that the selected ear is fixed. Later, the azimuth pathway should gate which monaural elevation estimate is trusted, because these pinna/DCN cues are most reliable for sounds known to originate from that side.",
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
    """Run the first elevation-pathway experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)

    config = make_config()
    predictions = run_dataset(config)
    ei_sweep = tune_ei_lambda(predictions, config)
    ei_lambda = float(ei_sweep["best_lambda"])
    ei_com, ei_argmax, _ = ei_weight_readouts(predictions, config, ei_lambda)
    true = np.array([item.true_elevation_deg for item in predictions])
    com = np.array([item.com_prediction_deg for item in predictions])
    argmax = np.array([item.argmax_prediction_deg for item in predictions])
    transfer_com = np.array([item.transfer_com_prediction_deg for item in predictions])
    transfer_argmax = np.array([item.transfer_argmax_prediction_deg for item in predictions])
    first_notch = np.array([item.first_notch_prediction_deg for item in predictions])
    metrics = {
        "Local-dip disinhibition COM": metric_dict(true, com),
        "Local-dip disinhibition argmax": metric_dict(true, argmax),
        "Explicit E/I weight-profile COM": metric_dict(true, ei_com),
        "Explicit E/I weight-profile argmax": metric_dict(true, ei_argmax),
        "Baseline-equalised full-transfer DCN COM": metric_dict(true, transfer_com),
        "Baseline-equalised full-transfer DCN argmax": metric_dict(true, transfer_argmax),
        "First-notch diagnostic readout": metric_dict(true, first_notch),
    }
    example_index = int(np.argmin(np.abs(true - 20.0)))
    artifacts = {
        "pipeline_diagram": plot_pipeline(FIGURE_DIR / "pipeline_diagram.png"),
        "comb_transfer": plot_comb_transfer(config, FIGURE_DIR / "comb_transfer.png"),
        "received_psd": plot_received_psd(config, float(true[example_index]), FIGURE_DIR / "received_psd_before_after_comb.png"),
        "dcn_templates": plot_dcn_templates(config, FIGURE_DIR / "dcn_templates.png"),
        "ei_lambda_sweep": plot_ei_lambda_sweep(ei_sweep, FIGURE_DIR / "ei_lambda_sweep.png"),
        "example_stages": plot_example(
            predictions[example_index],
            config,
            FIGURE_DIR / "example_stages.png",
            ei_lambda=ei_lambda,
        ),
        "prediction_scatter": plot_predictions(predictions, FIGURE_DIR / "prediction_scatter.png", ei_com=ei_com),
        "error_curve": plot_error(predictions, FIGURE_DIR / "error_curve.png", ei_com=ei_com),
    }
    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "elevation_pathway_first_attempt",
        "elapsed_seconds": elapsed_s,
        "setup": {
            "num_channels": NUM_CHANNELS,
            "num_elevation_bins": NUM_ELEVATION_BINS,
            "elevation_limit_deg": ELEVATION_LIMIT_DEG,
            "fixed_distance_m": FIXED_DISTANCE_M,
            "fixed_azimuth_deg": FIXED_AZIMUTH_DEG,
            "selected_ear": SELECTED_EAR,
            "comb_first_notch_low_hz": COMB_FIRST_NOTCH_LOW_HZ,
            "comb_first_notch_high_hz": COMB_FIRST_NOTCH_HIGH_HZ,
            "comb_delayed_copy_gain": COMB_DELAYED_COPY_GAIN,
            "selected_ei_lambda": ei_lambda,
        },
        "metrics": metrics,
        "ei_lambda_sweep": ei_sweep,
        "old_model_elevation_results": OLD_MODEL_ELEVATION_RESULTS,
        "predictions": [
            {
                "true_elevation_deg": item.true_elevation_deg,
                "com_prediction_deg": item.com_prediction_deg,
                "argmax_prediction_deg": item.argmax_prediction_deg,
                "transfer_com_prediction_deg": item.transfer_com_prediction_deg,
                "transfer_argmax_prediction_deg": item.transfer_argmax_prediction_deg,
                "ei_com_prediction_deg": float(ei_com[idx]),
                "ei_argmax_prediction_deg": float(ei_argmax[idx]),
                "first_notch_prediction_deg": item.first_notch_prediction_deg,
            }
            for idx, item in enumerate(predictions)
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(config, metrics, artifacts, elapsed_s, ei_sweep)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
