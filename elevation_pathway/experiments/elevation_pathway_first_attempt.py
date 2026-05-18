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
DEEP_COMB_DELAYED_COPY_GAIN = 0.99
DCN_TEMPLATE_POWER = 1.35
DCN_LOCAL_MEAN_SIGMA_CHANNELS = 2.0
DCN_TRANSFER_SIGMA = 0.05
DCN_SIGNAL_WEIGHTED_SIGMA = 0.05
EI_LAMBDA_GRID = np.linspace(0.2, 1.4, 25)
FULL_3D_NUM_SAMPLES = 160
FULL_3D_SEED = 41
LATERAL_EXC_SIGMA_BINS = 1.2
LATERAL_INH_SIGMA_BINS = 5.0
LATERAL_INH_WEIGHT = 0.65

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
        cochleagram_profile: Max-normalised selected-ear cochleagram spectrum.
        equalized_profile: Spectrum divided by a learned no-comb reference.
        transfer_activation: Full comb-transfer DCN population.
        signal_weighted_activation: Baseline-spectrum-weighted DCN population.
        signal_weighted_com_prediction_deg: COM prediction from weighted population.
        signal_weighted_argmax_prediction_deg: Argmax prediction from weighted population.
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
    cochleagram_profile: np.ndarray
    equalized_profile: np.ndarray
    transfer_activation: np.ndarray
    signal_weighted_activation: np.ndarray
    signal_weighted_com_prediction_deg: float
    signal_weighted_argmax_prediction_deg: float
    transfer_com_prediction_deg: float
    transfer_argmax_prediction_deg: float
    first_notch_prediction_deg: float
    comb_gain_channels: np.ndarray


@dataclass(frozen=True)
class DynamicInhibitionParams:
    """Global wideband inhibition parameters.

    Attributes:
        gain: Strength of global divisive inhibition.
        beta: Leaky retention of the non-spiking inhibitory interneuron.
    """

    gain: float = 0.0
    beta: float = 0.0


@dataclass(frozen=True)
class LateralInhibitionParams:
    """Mexican-hat lateral inhibition parameters over elevation bins.

    Attributes:
        gain: Strength applied to the Mexican-hat recurrent sharpening.
        exc_sigma_bins: Width of the local excitatory centre in bin units.
        inh_sigma_bins: Width of the broader inhibitory surround in bin units.
        inh_weight: Relative surround inhibition strength.
    """

    gain: float = 0.0
    exc_sigma_bins: float = LATERAL_EXC_SIGMA_BINS
    inh_sigma_bins: float = LATERAL_INH_SIGMA_BINS
    inh_weight: float = LATERAL_INH_WEIGHT


@dataclass(frozen=True)
class Full3DSample:
    """Cached cochlea output for one full-3D elevation test sample."""

    distance_m: float
    azimuth_deg: float
    elevation_deg: float
    selected_ear: str
    selected_cochleagram: torch.Tensor
    selected_spikes: torch.Tensor


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


def comb_gain(
    frequency_hz: np.ndarray,
    elevation_deg: np.ndarray | float,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
) -> np.ndarray:
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
        delayed_copy_gain: Relative gain of the delayed copy. Values near
            `1.0` create a much deeper destructive-interference notch.

    Returns:
        Gain array. If `elevation_deg` is vector-shaped, output shape is
        `[num_elevations, num_frequencies]`.
    """
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    tau = np.atleast_1d(comb_lag_s(elevation_deg)).astype(np.float64)
    phase = 2.0 * math.pi * tau[:, None] * frequencies[None, :]
    gain = np.sqrt(
        1.0 + delayed_copy_gain**2 + 2.0 * delayed_copy_gain * np.cos(phase)
    ) / (1.0 + delayed_copy_gain)
    gain = np.clip(gain, 1e-3, 1.0)
    if np.ndim(elevation_deg) == 0:
        return gain[0]
    return gain


def apply_comb_filter(
    waveform: torch.Tensor,
    config: GlobalConfig,
    elevation_deg: float,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
) -> torch.Tensor:
    """Apply the comb-filter elevation cue to one waveform.

    Args:
        waveform: One received ear waveform.
        config: Acoustic configuration.
        elevation_deg: Elevation angle in degrees.
        delayed_copy_gain: Relative gain of the delayed copy.

    Returns:
        Comb-filtered waveform.
    """
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / config.sample_rate_hz)
    gain = torch.from_numpy(
        comb_gain(frequencies.detach().cpu().numpy(), elevation_deg, delayed_copy_gain)
    ).to(waveform)
    spectrum = torch.fft.rfft(waveform)
    return torch.fft.irfft(spectrum * gain, n=waveform.numel())


def simulate_elevation_scene(
    config: GlobalConfig,
    elevation_deg: float,
    *,
    add_noise: bool = False,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
) -> torch.Tensor:
    """Simulate one binaural side-source echo with a comb elevation cue.

    Args:
        config: Acoustic configuration.
        elevation_deg: Target elevation in degrees.
        add_noise: Whether to add receiver noise.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

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
    receive[0] = apply_comb_filter(receive[0], config, elevation_deg, delayed_copy_gain)
    receive[1] = apply_comb_filter(receive[1], config, elevation_deg, delayed_copy_gain)
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


def simulate_pre_post_comb_scene(
    config: GlobalConfig,
    elevation_deg: float,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate one scene before and after the comb-filter elevation cue.

    The same base received waveform is used for both outputs. This makes the
    PSD comparison isolate the spectral notch instead of comparing two separate
    acoustic simulations.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angle used to set the comb-filter lag.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Pair `(before_comb, after_comb)`, each shaped `[ears, time]`.
    """
    before_comb = simulate_no_comb_reference(config).clone()
    after_comb = before_comb.clone()
    after_comb[0] = apply_comb_filter(after_comb[0], config, elevation_deg, delayed_copy_gain)
    after_comb[1] = apply_comb_filter(after_comb[1], config, elevation_deg, delayed_copy_gain)
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


def dynamic_wideband_inhibited_profile(
    cochleagram: torch.Tensor,
    spikes: torch.Tensor,
    params: DynamicInhibitionParams,
) -> np.ndarray:
    """Return a spectrum after global dynamic wideband inhibition.

    A non-spiking leaky inhibitory unit receives the instantaneous spike count
    across all cochlear channels:

    $$
    g_t=\beta g_{t-1}+\frac{1}{C}\sum_c S_{c,t}.
    $$

    The cochleagram drive at that timestep is divisively scaled by
    `1 + gain * g_t`. This suppresses broad high-volume responses while
    preserving relative spectral shape.

    Args:
        cochleagram: Selected-ear cochleagram `[channels, time]`.
        spikes: Selected-ear spike raster `[channels, time]`.
        params: Dynamic inhibition parameters.

    Returns:
        Max-normalised inhibited channel-energy profile.
    """
    if params.gain <= 0.0:
        return cochleagram_energy_profile(cochleagram)
    total_spikes = spikes.detach().float().sum(dim=0) / max(float(spikes.shape[0]), 1.0)
    inhibitory_trace = torch.empty_like(total_spikes)
    state = torch.tensor(0.0, dtype=total_spikes.dtype, device=total_spikes.device)
    for index in range(total_spikes.numel()):
        state = params.beta * state + total_spikes[index]
        inhibitory_trace[index] = state
    scale = 1.0 / (1.0 + params.gain * inhibitory_trace).clamp_min(1e-6)
    inhibited = cochleagram.detach().float() * scale.unsqueeze(0)
    energy = inhibited.cpu().numpy().sum(axis=1).astype(np.float64)
    return energy / np.maximum(energy.max(), 1e-12)


def baseline_energy_profile(
    config: GlobalConfig,
    dynamic_params: DynamicInhibitionParams = DynamicInhibitionParams(),
) -> np.ndarray:
    """Estimate the learned no-comb spectral reference for the selected ear.

    Args:
        config: Acoustic configuration.
        dynamic_params: Optional global inhibition parameters.

    Returns:
        Max-normalised no-comb cochlear energy spectrum.
    """
    cochlea = fdm._run_cochlea_binaural(config, simulate_no_comb_reference(config))
    cochleagram, spikes = selected_ear_activity(cochlea, config)
    return dynamic_wideband_inhibited_profile(cochleagram, spikes, dynamic_params)


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


def build_dcn_templates(
    config: GlobalConfig,
    bins_deg: np.ndarray,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build comb-derived DCN inhibitory templates.

    Args:
        config: Acoustic configuration.
        bins_deg: Candidate elevation bins.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Tuple `(centres_hz, comb_gain_matrix, template_matrix)`.
    """
    centres_hz = fdm._log_spaced_centers(config).detach().cpu().numpy()
    gains = comb_gain(centres_hz, bins_deg, delayed_copy_gain)
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


def dcn_signal_weighted_transfer_response(
    equalized_profile: np.ndarray,
    baseline_profile: np.ndarray,
    comb_gain_matrix: np.ndarray,
    centres_hz: np.ndarray,
) -> np.ndarray:
    """Compute a signal-and-notch-weighted full-transfer DCN response.

    The previous full-transfer matcher implicitly treats each frequency channel
    as equally informative after baseline equalisation. This variant still
    compares transfer shape, but weights each candidate's error by the product
    of available signal energy and expected notch strength:

    $$
    m_k(f_c)=P_0(f_c)(1-H_k(f_c))^2.
    $$

    This accounts for the fact that a notch is only informative if it lies in a
    frequency channel that receives enough signal energy.

    Args:
        equalized_profile: Baseline-divided selected-ear transfer estimate.
        baseline_profile: Max-normalised no-comb selected-ear spectrum.
        comb_gain_matrix: Candidate comb gains `[elevation_bins, channels]`.
        centres_hz: Cochlear centre frequencies.

    Returns:
        DCN activation over elevation bins.
    """
    mask = (centres_hz >= 4_500.0) & (centres_hz <= 18_000.0)
    signal_notch_weight = baseline_profile[None, :] * np.maximum(1.0 - comb_gain_matrix, 0.0) ** 2
    signal_notch_weight *= mask[None, :]
    signal_notch_weight /= np.maximum(signal_notch_weight.sum(axis=1, keepdims=True), 1e-12)
    mean_error = np.sum(signal_notch_weight * (equalized_profile[None, :] - comb_gain_matrix) ** 2, axis=1)
    response = np.exp(-mean_error / (2.0 * DCN_SIGNAL_WEIGHTED_SIGMA**2))
    return response / np.maximum(float(response.max()), 1e-12)


def mexican_hat_matrix(num_bins: int, params: LateralInhibitionParams) -> np.ndarray:
    """Build a finite-line Mexican-hat lateral interaction matrix.

    Args:
        num_bins: Number of represented elevation bins.
        params: Lateral inhibition parameters.

    Returns:
        Matrix `[num_bins, num_bins]` with local excitation and wider
        inhibition. Rows are centred on output neurons.
    """
    indices = np.arange(num_bins, dtype=np.float64)
    distance = np.abs(indices[:, None] - indices[None, :])
    excitation = np.exp(-0.5 * (distance / max(params.exc_sigma_bins, 1e-6)) ** 2)
    inhibition = np.exp(-0.5 * (distance / max(params.inh_sigma_bins, 1e-6)) ** 2)
    kernel = excitation - params.inh_weight * inhibition
    kernel -= kernel.mean(axis=1, keepdims=True)
    norm = np.max(np.abs(kernel), axis=1, keepdims=True)
    return kernel / np.maximum(norm, 1e-12)


def apply_lateral_inhibition(activity: np.ndarray, params: LateralInhibitionParams) -> np.ndarray:
    """Apply Mexican-hat sharpening to one elevation population.

    Args:
        activity: Non-negative elevation population.
        params: Lateral inhibition parameters.

    Returns:
        Sharpened non-negative population, max-normalised when non-zero.
    """
    if params.gain <= 0.0:
        return activity
    lateral = mexican_hat_matrix(activity.size, params) @ activity
    sharpened = np.maximum(activity + params.gain * lateral, 0.0)
    if float(sharpened.max()) <= 1e-12:
        return sharpened
    return sharpened / float(sharpened.max())


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


def predict_one(
    config: GlobalConfig,
    elevation_deg: float,
    baseline_profile: np.ndarray,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
    dynamic_params: DynamicInhibitionParams = DynamicInhibitionParams(),
    lateral_params: LateralInhibitionParams = LateralInhibitionParams(),
) -> ElevationPrediction:
    """Run the first elevation pathway for one elevation.

    Args:
        config: Acoustic configuration.
        elevation_deg: True target elevation.
        baseline_profile: Learned no-comb reference spectrum.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.
        dynamic_params: Global wideband inhibition parameters.
        lateral_params: Population lateral-inhibition parameters.

    Returns:
        Full stage prediction.
    """
    bins = elevation_grid()
    receive = simulate_elevation_scene(config, elevation_deg, delayed_copy_gain=delayed_copy_gain)
    cochlea = fdm._run_cochlea_binaural(config, receive)
    selected_cochleagram, selected_spikes = selected_ear_activity(cochlea, config)
    profile, observed_deficit = spectral_profile_from_spikes(selected_spikes)
    centres_hz, gain_matrix, templates = build_dcn_templates(config, bins, delayed_copy_gain)
    dcn = dcn_disinhibition_response(observed_deficit, profile, templates)
    cochleagram_profile = dynamic_wideband_inhibited_profile(selected_cochleagram, selected_spikes, dynamic_params)
    equalized = cochleagram_profile / np.maximum(baseline_profile, 1e-4)
    equalized = equalized / np.maximum(equalized.max(), 1e-12)
    transfer = dcn_full_transfer_response(equalized, gain_matrix, centres_hz)
    signal_weighted = dcn_signal_weighted_transfer_response(
        equalized,
        baseline_profile,
        gain_matrix,
        centres_hz,
    )
    signal_weighted = apply_lateral_inhibition(signal_weighted, lateral_params)
    return ElevationPrediction(
        true_elevation_deg=float(elevation_deg),
        com_prediction_deg=centre_of_mass(dcn, bins),
        argmax_prediction_deg=float(bins[int(np.argmax(dcn))]),
        cochlea=cochlea,
        selected_spikes=selected_spikes,
        spectral_profile=profile,
        observed_deficit=observed_deficit,
        dcn_activation=dcn,
        cochleagram_profile=cochleagram_profile,
        equalized_profile=equalized,
        transfer_activation=transfer,
        signal_weighted_activation=signal_weighted,
        signal_weighted_com_prediction_deg=centre_of_mass(signal_weighted, bins),
        signal_weighted_argmax_prediction_deg=float(bins[int(np.argmax(signal_weighted))]),
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


def run_dataset(
    config: GlobalConfig,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
    dynamic_params: DynamicInhibitionParams = DynamicInhibitionParams(),
    lateral_params: LateralInhibitionParams = LateralInhibitionParams(),
) -> list[ElevationPrediction]:
    """Run the monaural elevation pathway over the full elevation grid."""
    baseline = baseline_energy_profile(config, dynamic_params)
    return [
        predict_one(config, float(elevation), baseline, delayed_copy_gain, dynamic_params, lateral_params)
        for elevation in elevation_grid()
    ]


def simulate_full_3d_scene(
    config: GlobalConfig,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    delayed_copy_gain: float,
) -> torch.Tensor:
    """Simulate a full-3D binaural echo with the comb elevation cue.

    Args:
        config: Acoustic configuration.
        distance_m: Target distance in metres.
        azimuth_deg: Target azimuth in degrees.
        elevation_deg: Target elevation in degrees.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Binaural received waveform `[ears, time]`.
    """
    scene = simulate_echo_batch(
        config,
        radii_m=torch.tensor([distance_m], dtype=torch.float32),
        azimuth_deg=torch.tensor([azimuth_deg], dtype=torch.float32),
        elevation_deg=torch.tensor([elevation_deg], dtype=torch.float32),
        binaural=True,
        add_noise=False,
        include_elevation_cues=False,
        transmit_gain=config.transmit_gain,
    )
    receive = scene.receive[0].detach().clone()
    receive[0] = apply_comb_filter(receive[0], config, elevation_deg, delayed_copy_gain)
    receive[1] = apply_comb_filter(receive[1], config, elevation_deg, delayed_copy_gain)
    return receive


def cache_full_3d_samples(
    config: GlobalConfig,
    *,
    num_samples: int,
    seed: int,
    delayed_copy_gain: float,
) -> list[Full3DSample]:
    """Generate and cache cochlea outputs for full-3D elevation tests.

    Args:
        config: Acoustic configuration.
        num_samples: Number of random 3D samples.
        seed: Random seed.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Cached full-3D samples. The selected ear is the ipsilateral/nearer ear
        implied by the azimuth sign, which stands in for later azimuth gating.
    """
    rng = np.random.default_rng(seed)
    distances = rng.uniform(0.25, 5.0, size=num_samples)
    azimuths = rng.uniform(-90.0, 90.0, size=num_samples)
    elevations = rng.uniform(-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG, size=num_samples)
    samples: list[Full3DSample] = []
    for distance, azimuth, elevation in zip(distances, azimuths, elevations):
        receive = simulate_full_3d_scene(
            config,
            float(distance),
            float(azimuth),
            float(elevation),
            delayed_copy_gain,
        )
        cochlea = fdm._run_cochlea_binaural(config, receive)
        if azimuth >= 0.0:
            selected_ear = "right"
            cochleagram = cochlea.right_cochleagram
        else:
            selected_ear = "left"
            cochleagram = cochlea.left_cochleagram
        spikes = fdm._dynamic_lif_encode(cochleagram, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
        samples.append(
            Full3DSample(
                distance_m=float(distance),
                azimuth_deg=float(azimuth),
                elevation_deg=float(elevation),
                selected_ear=selected_ear,
                selected_cochleagram=cochleagram,
                selected_spikes=spikes,
            )
        )
    return samples


def decode_full_3d_samples(
    samples: list[Full3DSample],
    config: GlobalConfig,
    *,
    delayed_copy_gain: float,
    dynamic_params: DynamicInhibitionParams,
    lateral_params: LateralInhibitionParams,
    baseline_profile: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode cached full-3D samples with a candidate elevation model.

    Args:
        samples: Cached cochlea outputs.
        config: Acoustic configuration.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.
        dynamic_params: Global wideband inhibition parameters.
        lateral_params: Population lateral-inhibition parameters.
        baseline_profile: Optional cached no-comb reference for these dynamic
            inhibition parameters.

    Returns:
        Tuple `(true_elevations, predicted_elevations, populations)`.
    """
    bins = elevation_grid()
    if baseline_profile is None:
        baseline_profile = baseline_energy_profile(config, dynamic_params)
    centres_hz, gain_matrix, _ = build_dcn_templates(config, bins, delayed_copy_gain)
    true = np.array([sample.elevation_deg for sample in samples], dtype=np.float64)
    predicted = []
    populations = []
    for sample in samples:
        profile = dynamic_wideband_inhibited_profile(
            sample.selected_cochleagram,
            sample.selected_spikes,
            dynamic_params,
        )
        equalized = profile / np.maximum(baseline_profile, 1e-4)
        equalized = equalized / np.maximum(equalized.max(), 1e-12)
        population = dcn_signal_weighted_transfer_response(equalized, baseline_profile, gain_matrix, centres_hz)
        population = apply_lateral_inhibition(population, lateral_params)
        populations.append(population)
        predicted.append(centre_of_mass(population, bins))
    return true, np.array(predicted, dtype=np.float64), np.stack(populations, axis=0)


def sweep_full_3d_model(
    samples: list[Full3DSample],
    config: GlobalConfig,
    delayed_copy_gain: float,
) -> dict[str, object]:
    """Sweep dynamic inhibition and lateral inhibition on full-3D samples.

    Args:
        samples: Cached full-3D cochlea outputs.
        config: Acoustic configuration.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Sweep rows, best parameter set, and best predictions.
    """
    dynamic_gain_values = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    dynamic_beta_values = [0.0, 0.3, 0.6, 0.85]
    lateral_gain_values = [0.0, 0.03, 0.06, 0.1, 0.15, 0.3, 0.6, 1.0]
    rows = []
    best_payload: dict[str, object] | None = None
    baseline_cache: dict[tuple[float, float], np.ndarray] = {}
    for dynamic_gain in dynamic_gain_values:
        beta_values = [0.0] if dynamic_gain == 0.0 else dynamic_beta_values
        for dynamic_beta in beta_values:
            dynamic_params = DynamicInhibitionParams(gain=dynamic_gain, beta=dynamic_beta)
            cache_key = (float(dynamic_gain), float(dynamic_beta))
            baseline_cache[cache_key] = baseline_energy_profile(config, dynamic_params)
            for lateral_gain in lateral_gain_values:
                lateral_params = LateralInhibitionParams(gain=lateral_gain)
                true, pred, populations = decode_full_3d_samples(
                    samples,
                    config,
                    delayed_copy_gain=delayed_copy_gain,
                    dynamic_params=dynamic_params,
                    lateral_params=lateral_params,
                    baseline_profile=baseline_cache[cache_key],
                )
                metric = metric_dict(true, pred)
                row = {
                    "dynamic_gain": float(dynamic_gain),
                    "dynamic_beta": float(dynamic_beta),
                    "lateral_gain": float(lateral_gain),
                    **metric,
                }
                rows.append(row)
                if best_payload is None or metric["mae_deg"] < best_payload["metrics"]["mae_deg"]:
                    best_payload = {
                        "dynamic_params": dynamic_params,
                        "lateral_params": lateral_params,
                        "true": true,
                        "predicted": pred,
                        "populations": populations,
                        "metrics": metric,
                    }
    assert best_payload is not None
    return {"rows": rows, "best": best_payload}


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


def plot_comb_transfer(
    config: GlobalConfig,
    path: Path,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
    title_suffix: str = "",
) -> str:
    """Plot the comb-filter sweep over elevation."""
    elevations = elevation_grid()
    frequencies = np.linspace(config.cochlea_low_hz, config.cochlea_high_hz, 700)
    gain = comb_gain(frequencies, elevations, delayed_copy_gain)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    im = ax.contourf(frequencies / 1_000.0, elevations, gain, levels=50, cmap="magma")
    ax.plot(comb_first_notch_hz(elevations) / 1_000.0, elevations, color="#38bdf8", linewidth=1.8, label="first notch")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(f"Comb-filter elevation transfer function, a={delayed_copy_gain:.2f}{title_suffix}")
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


def plot_received_psd(
    config: GlobalConfig,
    elevation_deg: float,
    path: Path,
    delayed_copy_gain: float = COMB_DELAYED_COPY_GAIN,
) -> str:
    """Plot selected-ear PSD before and after the comb-filter cue.

    Args:
        config: Acoustic configuration.
        elevation_deg: Elevation angle used for the comb-filter cue.
        path: Output figure path.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Saved figure path.
    """
    before_binaural, after_binaural = simulate_pre_post_comb_scene(config, elevation_deg, delayed_copy_gain)
    ear_index = 1 if SELECTED_EAR == "right" else 0
    before = before_binaural[ear_index]
    after = after_binaural[ear_index]
    crop = active_echo_window(before, after)
    frequencies_hz, before_psd = one_sided_psd(before[crop], config)
    _, after_psd = one_sided_psd(after[crop], config)
    shared_peak = max(float(before_psd.max()), float(after_psd.max()), 1e-12)
    before_psd_db = 10.0 * np.log10(before_psd / shared_peak + 1e-12)
    after_psd_db = 10.0 * np.log10(after_psd / shared_peak + 1e-12)

    theoretical_gain = 20.0 * np.log10(comb_gain(frequencies_hz, elevation_deg, delayed_copy_gain) + 1e-12)
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
        f"(elevation={elevation_deg:.1f} deg, a={delayed_copy_gain:.2f}, "
        f"f1={first_notch_hz / 1_000.0:.2f} kHz, tau={lag_us:.1f} us)"
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


def signal_notch_weight_matrix(
    config: GlobalConfig,
    baseline_profile: np.ndarray,
    delayed_copy_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the signal-and-notch synaptic weight matrix.

    Args:
        config: Acoustic configuration.
        baseline_profile: No-comb selected-ear spectrum.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.

    Returns:
        Tuple `(centres_hz, bins_deg, weight_matrix)`.
    """
    bins = elevation_grid()
    centres_hz, gain_matrix, _ = build_dcn_templates(config, bins, delayed_copy_gain)
    weights = baseline_profile[None, :] * np.maximum(1.0 - gain_matrix, 0.0) ** 2
    mask = ((centres_hz >= 4_500.0) & (centres_hz <= 18_000.0)).astype(np.float64)
    weights *= mask[None, :]
    weights /= np.maximum(weights.max(axis=1, keepdims=True), 1e-12)
    return centres_hz, bins, weights


def plot_signal_notch_weights(
    config: GlobalConfig,
    baseline_profile: np.ndarray,
    delayed_copy_gain: float,
    path: Path,
) -> str:
    """Plot the signal-weighted synaptic matrix used by the best DCN model.

    Args:
        config: Acoustic configuration.
        baseline_profile: No-comb selected-ear spectrum.
        delayed_copy_gain: Relative gain of the delayed comb-filter copy.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    centres_hz, bins, weights = signal_notch_weight_matrix(config, baseline_profile, delayed_copy_gain)
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    im = ax.imshow(
        weights,
        aspect="auto",
        origin="lower",
        extent=[centres_hz[0] / 1_000.0, centres_hz[-1] / 1_000.0, bins[0], bins[-1]],
        cmap="mako" if "mako" in plt.colormaps() else "viridis",
    )
    ax.plot(comb_first_notch_hz(bins) / 1_000.0, bins, color="#f97316", linewidth=1.6, label="first notch")
    ax.set_xlabel("cochlear centre frequency (kHz)")
    ax.set_ylabel("candidate elevation (deg)")
    ax.set_title(f"Signal-and-notch synaptic weights, a={delayed_copy_gain:.2f}")
    ax.legend(frameon=False)
    fig.colorbar(im, ax=ax, label="normalised synaptic importance")
    return save_figure(fig, path)


def plot_lateral_matrix(params: LateralInhibitionParams, path: Path) -> str:
    """Plot the Mexican-hat lateral interaction matrix.

    Args:
        params: Lateral inhibition parameters.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    bins = elevation_grid()
    matrix = mexican_hat_matrix(bins.size, params)
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[bins[0], bins[-1], bins[0], bins[-1]],
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xlabel("source elevation neuron (deg)")
    ax.set_ylabel("target elevation neuron (deg)")
    ax.set_title(
        f"Mexican-hat lateral matrix, gain={params.gain:.2f}, "
        f"sigmaE={params.exc_sigma_bins:.1f}, sigmaI={params.inh_sigma_bins:.1f}"
    )
    fig.colorbar(im, ax=ax, label="signed lateral weight")
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


def plot_improvement_predictions(
    current_predictions: list[ElevationPrediction],
    deep_predictions: list[ElevationPrediction],
    path: Path,
) -> str:
    """Plot true-vs-predicted elevation for the proposed improvements.

    Args:
        current_predictions: Predictions using the original comb depth.
        deep_predictions: Predictions using the deeper comb notch.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    true = np.array([item.true_elevation_deg for item in current_predictions])
    current_weighted = np.array([item.signal_weighted_com_prediction_deg for item in current_predictions])
    deep_transfer = np.array([item.transfer_com_prediction_deg for item in deep_predictions])
    deep_weighted = np.array([item.signal_weighted_com_prediction_deg for item in deep_predictions])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, pred, title in [
        (axes[0], current_weighted, f"Current comb a={COMB_DELAYED_COPY_GAIN:.2f}\n+ signal-weighted COM"),
        (axes[1], deep_transfer, f"Deep comb a={DEEP_COMB_DELAYED_COPY_GAIN:.2f}\n+ equalised transfer COM"),
        (axes[2], deep_weighted, f"Deep comb a={DEEP_COMB_DELAYED_COPY_GAIN:.2f}\n+ signal-weighted COM"),
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


def plot_improvement_error(
    current_predictions: list[ElevationPrediction],
    deep_predictions: list[ElevationPrediction],
    path: Path,
) -> str:
    """Plot signed elevation error for the proposed improvements.

    Args:
        current_predictions: Predictions using the original comb depth.
        deep_predictions: Predictions using the deeper comb notch.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    true = np.array([item.true_elevation_deg for item in current_predictions])
    current_weighted = np.array([item.signal_weighted_com_prediction_deg for item in current_predictions])
    deep_transfer = np.array([item.transfer_com_prediction_deg for item in deep_predictions])
    deep_weighted = np.array([item.signal_weighted_com_prediction_deg for item in deep_predictions])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(true, current_weighted - true, label=f"current comb a={COMB_DELAYED_COPY_GAIN:.2f} + signal-weighted", linewidth=2.0)
    ax.plot(true, deep_transfer - true, label=f"deep comb a={DEEP_COMB_DELAYED_COPY_GAIN:.2f} + equalised transfer", linewidth=2.0)
    ax.plot(true, deep_weighted - true, label=f"deep comb a={DEEP_COMB_DELAYED_COPY_GAIN:.2f} + signal-weighted", linewidth=2.0)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel("true elevation (deg)")
    ax.set_ylabel("prediction error (deg)")
    ax.set_title("Improvement comparison error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_full_3d_prediction_scatter(true: np.ndarray, predicted: np.ndarray, path: Path) -> str:
    """Plot full-3D true-vs-predicted elevation.

    Args:
        true: True elevations in degrees.
        predicted: Predicted elevations in degrees.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(true, predicted, s=28, alpha=0.72)
    ax.plot([-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG], [-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG], color="#111827")
    ax.set_xlabel("true elevation (deg)")
    ax.set_ylabel("predicted elevation (deg)")
    ax.set_title("Best model full-3D elevation prediction")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, path)


def plot_full_3d_error_context(samples: list[Full3DSample], true: np.ndarray, predicted: np.ndarray, path: Path) -> str:
    """Plot full-3D elevation error against distance and azimuth.

    Args:
        samples: Cached full-3D samples.
        true: True elevations in degrees.
        predicted: Predicted elevations in degrees.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    error = predicted - true
    distances = np.array([sample.distance_m for sample in samples], dtype=np.float64)
    azimuths = np.array([sample.azimuth_deg for sample in samples], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, x, xlabel in [
        (axes[0], distances, "distance (m)"),
        (axes[1], azimuths, "azimuth (deg)"),
    ]:
        scatter = ax.scatter(x, error, c=true, cmap="coolwarm", s=26, alpha=0.78)
        ax.axhline(0.0, color="#111827", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("elevation error (deg)")
        ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="true elevation (deg)")
    fig.suptitle("Full-3D error context for best elevation model")
    return save_figure(fig, path)


def plot_tuning_sweep(sweep: dict[str, object], path: Path) -> str:
    """Plot the best full-3D MAE over dynamic and lateral gains.

    Args:
        sweep: Output of `sweep_full_3d_model`.
        path: Output figure path.

    Returns:
        Saved figure path.
    """
    rows = sweep["rows"]
    dynamic_gains = sorted({row["dynamic_gain"] for row in rows})
    lateral_gains = sorted({row["lateral_gain"] for row in rows})
    heatmap = np.full((len(dynamic_gains), len(lateral_gains)), np.nan, dtype=np.float64)
    annotations: list[list[str]] = [["" for _ in lateral_gains] for _ in dynamic_gains]
    for row in rows:
        i = dynamic_gains.index(row["dynamic_gain"])
        j = lateral_gains.index(row["lateral_gain"])
        if np.isnan(heatmap[i, j]) or row["mae_deg"] < heatmap[i, j]:
            heatmap[i, j] = row["mae_deg"]
            annotations[i][j] = f"b={row['dynamic_beta']:.2f}"
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(np.arange(len(lateral_gains)))
    ax.set_xticklabels([f"{value:.2f}" for value in lateral_gains])
    ax.set_yticks(np.arange(len(dynamic_gains)))
    ax.set_yticklabels([f"{value:.2f}" for value in dynamic_gains])
    ax.set_xlabel("lateral Mexican-hat gain")
    ax.set_ylabel("wideband inhibition gain")
    ax.set_title("Full-3D tuning sweep: best MAE over beta")
    for i in range(len(dynamic_gains)):
        for j in range(len(lateral_gains)):
            ax.text(j, i, f"{heatmap[i, j]:.2f}\n{annotations[i][j]}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="elevation MAE (deg)")
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
    full_3d_sweep: dict[str, object],
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
        f"Current comb-filter gain: `a = {COMB_DELAYED_COPY_GAIN:.2f}`.",
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
        "## Comb-Depth And Signal-Weighted Improvements",
        "",
        "Two additions are tested without removing the original pathway results.",
        "",
        f"First, the delayed-copy gain is increased from `a = {COMB_DELAYED_COPY_GAIN:.2f}` to `a = {DEEP_COMB_DELAYED_COPY_GAIN:.2f}`. In the ideal comb filter, the first-notch amplitude is:",
        "",
        "$$",
        "|H(f_1)|=\\frac{|1-a|}{1+a}.",
        "$$",
        "",
        "This makes the spectral notch much deeper, which should make the elevation cue easier for a notch detector to identify.",
        "",
        "Second, the matcher weights the transfer-shape error by the expected usefulness of each frequency channel. The no-notch selected-ear spectrum is treated as a baseline signal envelope `P_0(f_c)`, and each candidate elevation defines a signal-and-notch mask:",
        "",
        "$$",
        "m_k(f_c)=P_0(f_c)(1-H_k(f_c))^2.",
        "$$",
        "",
        "The signal-weighted DCN population is then:",
        "",
        "$$",
        "r_k=\\exp\\left(-\\frac{\\sum_c m_k(f_c)(\\tilde p_c-H_k(f_c))^2}{2\\sigma^2}\\right).",
        "$$",
        "",
        "This matters because the emitted sweep and cochlear front end are not flat across frequency. A notch in a weak part of the spectrum should not contribute as much evidence as a notch inside the high-energy part of the received call.",
        "",
        f"Deep-comb gain used for the improvement plots: `a = {DEEP_COMB_DELAYED_COPY_GAIN:.2f}`.",
        "",
        "![Deep comb transfer, a=0.99](../outputs/first_attempt/figures/deep_comb_transfer.png)",
        "",
        "![Deep received PSD, a=0.99](../outputs/first_attempt/figures/deep_received_psd_before_after_comb.png)",
        "",
        "![Improvement prediction scatter](../outputs/first_attempt/figures/improvement_prediction_scatter.png)",
        "",
        "![Improvement error curve](../outputs/first_attempt/figures/improvement_error_curve.png)",
        "",
        "## Synaptic Weights, Lateral Inhibition, And Dynamic Wideband Inhibition",
        "",
        "The best current DCN-style model uses signal-and-notch synaptic weights. Each row corresponds to one candidate elevation neuron, and each column corresponds to one cochlear frequency channel:",
        "",
        "$$",
        "m_k(f_c)=P_0(f_c)(1-H_k(f_c))^2.",
        "$$",
        "",
        "This is interpreted as a fixed frequency-specific synaptic importance profile. It is not claiming the biological DCN computes a Fourier transfer function online; it uses the known transfer function to set biologically plausible frequency-specific weights.",
        "",
        "![Signal-and-notch synaptic weights](../outputs/first_attempt/figures/signal_notch_weight_matrix.png)",
        "",
        "Lateral inhibition is added as a Mexican-hat interaction across the elevation population:",
        "",
        "$$",
        "L_{ij}=\\exp\\left(-\\frac{(i-j)^2}{2\\sigma_E^2}\\right)-\\gamma\\exp\\left(-\\frac{(i-j)^2}{2\\sigma_I^2}\\right).",
        "$$",
        "",
        "$$",
        "r'=[r+\\alpha Lr]_+.",
        "$$",
        "",
        "This gives local cooperation and broader suppression, sharpening the elevation bump without changing the cochlear evidence itself.",
        "",
        "![Mexican-hat lateral matrix](../outputs/first_attempt/figures/mexican_hat_lateral_matrix.png)",
        "",
        "Dynamic wideband inhibition is implemented as a non-spiking leaky interneuron driven by the instantaneous total cochlear spike count:",
        "",
        "$$",
        "g_t=\\beta g_{t-1}+\\frac{1}{C}\\sum_c S_{c,t},",
        "\\qquad",
        "\\hat x_{c,t}=\\frac{x_{c,t}}{1+\\eta g_t}.",
        "$$",
        "",
        "This is a divisive gain-control mechanism. It should reduce sensitivity to distance-dependent volume changes while preserving the relative spectral notch pattern.",
        "",
        "## Full 3D Elevation Test And Tuning",
        "",
        f"The tuned model is tested on `{FULL_3D_NUM_SAMPLES}` clean full-3D samples. Distances are sampled from `0.25 m` to `5.0 m`, azimuth from `-90 deg` to `+90 deg`, and elevation from `-45 deg` to `+45 deg`. Only elevation error is measured. The selected ear is chosen by azimuth sign as a simple stand-in for the later azimuth-gated elevation pathway.",
        "",
        "The full-3D sweep varies:",
        "",
        "- dynamic wideband inhibition gain `eta`;",
        "- dynamic wideband inhibition leak `beta`;",
        "- Mexican-hat lateral gain `alpha`.",
        "",
        "![Full 3D tuning sweep](../outputs/first_attempt/figures/full_3d_tuning_sweep.png)",
        "",
        "Full-3D comparison:",
        "",
        "| Model | MAE | RMSE | Max error | Bias | Parameters |",
        "|---|---:|---:|---:|---:|---|",
    ]
    full_rows = full_3d_sweep["rows"]
    reference_row = min(
        (
            row
            for row in full_rows
            if row["dynamic_gain"] == 0.0 and row["dynamic_beta"] == 0.0 and row["lateral_gain"] == 0.0
        ),
        key=lambda row: row["mae_deg"],
    )
    best = full_3d_sweep["best"]
    best_metric = best["metrics"]
    best_dynamic: DynamicInhibitionParams = best["dynamic_params"]
    best_lateral: LateralInhibitionParams = best["lateral_params"]
    lines.extend(
        [
            f"| Deep-comb signal-weighted DCN, no dynamic/lateral | `{reference_row['mae_deg']:.3f} deg` | `{reference_row['rmse_deg']:.3f} deg` | `{reference_row['max_abs_error_deg']:.3f} deg` | `{reference_row['bias_deg']:.3f} deg` | `eta=0`, `alpha=0` |",
            f"| Best tuned full-3D model | `{best_metric['mae_deg']:.3f} deg` | `{best_metric['rmse_deg']:.3f} deg` | `{best_metric['max_abs_error_deg']:.3f} deg` | `{best_metric['bias_deg']:.3f} deg` | `eta={best_dynamic.gain:.2f}`, `beta={best_dynamic.beta:.2f}`, `alpha={best_lateral.gain:.2f}` |",
            "",
            "In this clean full-3D test the best tuned setting leaves both added mechanisms off. That is still useful: the signal-weighted comb-transfer synaptic matrix already produces a sharp enough population for COM readout, while lateral inhibition over-sharpens the bump and dynamic wideband inhibition slightly distorts the equalised spectral profile. These mechanisms should be revisited under noise, clutter, or stronger distance-dependent amplitude variation.",
            "",
            "![Full 3D prediction scatter](../outputs/first_attempt/figures/full_3d_prediction_scatter.png)",
            "",
            "![Full 3D error context](../outputs/first_attempt/figures/full_3d_error_context.png)",
            "",
        ]
    )
    lines.extend(
        [
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
    )
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
    deep_predictions = run_dataset(config, delayed_copy_gain=DEEP_COMB_DELAYED_COPY_GAIN)
    ei_sweep = tune_ei_lambda(predictions, config)
    ei_lambda = float(ei_sweep["best_lambda"])
    ei_com, ei_argmax, _ = ei_weight_readouts(predictions, config, ei_lambda)
    true = np.array([item.true_elevation_deg for item in predictions])
    com = np.array([item.com_prediction_deg for item in predictions])
    argmax = np.array([item.argmax_prediction_deg for item in predictions])
    transfer_com = np.array([item.transfer_com_prediction_deg for item in predictions])
    transfer_argmax = np.array([item.transfer_argmax_prediction_deg for item in predictions])
    signal_weighted_com = np.array([item.signal_weighted_com_prediction_deg for item in predictions])
    signal_weighted_argmax = np.array([item.signal_weighted_argmax_prediction_deg for item in predictions])
    deep_transfer_com = np.array([item.transfer_com_prediction_deg for item in deep_predictions])
    deep_transfer_argmax = np.array([item.transfer_argmax_prediction_deg for item in deep_predictions])
    deep_signal_weighted_com = np.array([item.signal_weighted_com_prediction_deg for item in deep_predictions])
    deep_signal_weighted_argmax = np.array(
        [item.signal_weighted_argmax_prediction_deg for item in deep_predictions]
    )
    first_notch = np.array([item.first_notch_prediction_deg for item in predictions])
    metrics = {
        "Local-dip disinhibition COM": metric_dict(true, com),
        "Local-dip disinhibition argmax": metric_dict(true, argmax),
        "Explicit E/I weight-profile COM": metric_dict(true, ei_com),
        "Explicit E/I weight-profile argmax": metric_dict(true, ei_argmax),
        "Baseline-equalised full-transfer DCN COM": metric_dict(true, transfer_com),
        "Baseline-equalised full-transfer DCN argmax": metric_dict(true, transfer_argmax),
        "Signal-weighted full-transfer DCN COM": metric_dict(true, signal_weighted_com),
        "Signal-weighted full-transfer DCN argmax": metric_dict(true, signal_weighted_argmax),
        "Deep-comb baseline-equalised full-transfer DCN COM": metric_dict(true, deep_transfer_com),
        "Deep-comb baseline-equalised full-transfer DCN argmax": metric_dict(true, deep_transfer_argmax),
        "Deep-comb signal-weighted full-transfer DCN COM": metric_dict(true, deep_signal_weighted_com),
        "Deep-comb signal-weighted full-transfer DCN argmax": metric_dict(true, deep_signal_weighted_argmax),
        "First-notch diagnostic readout": metric_dict(true, first_notch),
    }
    example_index = int(np.argmin(np.abs(true - 20.0)))
    full_3d_samples = cache_full_3d_samples(
        config,
        num_samples=FULL_3D_NUM_SAMPLES,
        seed=FULL_3D_SEED,
        delayed_copy_gain=DEEP_COMB_DELAYED_COPY_GAIN,
    )
    full_3d_sweep = sweep_full_3d_model(
        full_3d_samples,
        config,
        delayed_copy_gain=DEEP_COMB_DELAYED_COPY_GAIN,
    )
    full_3d_best = full_3d_sweep["best"]
    best_dynamic: DynamicInhibitionParams = full_3d_best["dynamic_params"]
    best_lateral: LateralInhibitionParams = full_3d_best["lateral_params"]
    best_true = full_3d_best["true"]
    best_predicted = full_3d_best["predicted"]
    best_baseline_profile = baseline_energy_profile(config, best_dynamic)
    artifacts = {
        "pipeline_diagram": plot_pipeline(FIGURE_DIR / "pipeline_diagram.png"),
        "comb_transfer": plot_comb_transfer(config, FIGURE_DIR / "comb_transfer.png"),
        "received_psd": plot_received_psd(config, float(true[example_index]), FIGURE_DIR / "received_psd_before_after_comb.png"),
        "deep_comb_transfer": plot_comb_transfer(
            config,
            FIGURE_DIR / "deep_comb_transfer.png",
            delayed_copy_gain=DEEP_COMB_DELAYED_COPY_GAIN,
            title_suffix=" with deeper notch",
        ),
        "deep_received_psd": plot_received_psd(
            config,
            float(true[example_index]),
            FIGURE_DIR / "deep_received_psd_before_after_comb.png",
            delayed_copy_gain=DEEP_COMB_DELAYED_COPY_GAIN,
        ),
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
        "improvement_prediction_scatter": plot_improvement_predictions(
            predictions,
            deep_predictions,
            FIGURE_DIR / "improvement_prediction_scatter.png",
        ),
        "improvement_error_curve": plot_improvement_error(
            predictions,
            deep_predictions,
            FIGURE_DIR / "improvement_error_curve.png",
        ),
        "signal_notch_weight_matrix": plot_signal_notch_weights(
            config,
            best_baseline_profile,
            DEEP_COMB_DELAYED_COPY_GAIN,
            FIGURE_DIR / "signal_notch_weight_matrix.png",
        ),
        "mexican_hat_lateral_matrix": plot_lateral_matrix(
            best_lateral,
            FIGURE_DIR / "mexican_hat_lateral_matrix.png",
        ),
        "full_3d_tuning_sweep": plot_tuning_sweep(
            full_3d_sweep,
            FIGURE_DIR / "full_3d_tuning_sweep.png",
        ),
        "full_3d_prediction_scatter": plot_full_3d_prediction_scatter(
            best_true,
            best_predicted,
            FIGURE_DIR / "full_3d_prediction_scatter.png",
        ),
        "full_3d_error_context": plot_full_3d_error_context(
            full_3d_samples,
            best_true,
            best_predicted,
            FIGURE_DIR / "full_3d_error_context.png",
        ),
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
            "deep_comb_delayed_copy_gain": DEEP_COMB_DELAYED_COPY_GAIN,
            "selected_ei_lambda": ei_lambda,
            "full_3d_num_samples": FULL_3D_NUM_SAMPLES,
            "full_3d_seed": FULL_3D_SEED,
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
                "signal_weighted_com_prediction_deg": item.signal_weighted_com_prediction_deg,
                "signal_weighted_argmax_prediction_deg": item.signal_weighted_argmax_prediction_deg,
                "ei_com_prediction_deg": float(ei_com[idx]),
                "ei_argmax_prediction_deg": float(ei_argmax[idx]),
                "first_notch_prediction_deg": item.first_notch_prediction_deg,
            }
            for idx, item in enumerate(predictions)
        ],
        "deep_comb_predictions": [
            {
                "true_elevation_deg": item.true_elevation_deg,
                "transfer_com_prediction_deg": item.transfer_com_prediction_deg,
                "transfer_argmax_prediction_deg": item.transfer_argmax_prediction_deg,
                "signal_weighted_com_prediction_deg": item.signal_weighted_com_prediction_deg,
                "signal_weighted_argmax_prediction_deg": item.signal_weighted_argmax_prediction_deg,
            }
            for item in deep_predictions
        ],
        "full_3d_sweep": {
            "rows": full_3d_sweep["rows"],
            "best": {
                "dynamic_gain": best_dynamic.gain,
                "dynamic_beta": best_dynamic.beta,
                "lateral_gain": best_lateral.gain,
                "metrics": full_3d_best["metrics"],
            },
        },
        "full_3d_best_predictions": [
            {
                "distance_m": sample.distance_m,
                "azimuth_deg": sample.azimuth_deg,
                "true_elevation_deg": sample.elevation_deg,
                "predicted_elevation_deg": float(prediction),
                "selected_ear": sample.selected_ear,
            }
            for sample, prediction in zip(full_3d_samples, best_predicted)
        ],
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(config, metrics, artifacts, elapsed_s, ei_sweep, full_3d_sweep)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
