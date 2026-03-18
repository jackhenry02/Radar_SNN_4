from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.common import GlobalConfig


@dataclass
class AcousticBatch:
    transmit: torch.Tensor
    receive: torch.Tensor
    delays_s: torch.Tensor
    amplitudes: torch.Tensor
    radii_m: torch.Tensor
    azimuth_deg: torch.Tensor
    elevation_deg: torch.Tensor
    itd_s: torch.Tensor | None = None
    ild_db: torch.Tensor | None = None


def generate_fm_chirp(config: GlobalConfig, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    time_axis = torch.arange(config.chirp_samples, device=device, dtype=torch.float32) / config.sample_rate_hz
    sweep_rate = (config.chirp_end_hz - config.chirp_start_hz) / config.chirp_duration_s
    phase = 2.0 * math.pi * (
        config.chirp_start_hz * time_axis + 0.5 * sweep_rate * time_axis.square()
    )
    window = torch.hann_window(config.chirp_samples, periodic=False, device=device)
    chirp = torch.sin(phase) * window
    chirp = chirp / chirp.abs().amax().clamp_min(1e-6)
    return chirp.unsqueeze(0).repeat(batch_size, 1), time_axis


def pad_signal(signal: torch.Tensor, total_samples: int) -> torch.Tensor:
    padded = torch.zeros(*signal.shape[:-1], total_samples, device=signal.device, dtype=signal.dtype)
    padded[..., : signal.shape[-1]] = signal
    return padded


def fractional_delay(signal: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
    total_samples = signal.shape[-1]
    flat_signal = signal.reshape(-1, total_samples)
    flat_delay = delay_samples.reshape(-1).to(signal.dtype)
    spectrum = torch.fft.rfft(flat_signal, dim=-1)
    frequency_bins = torch.arange(spectrum.shape[-1], device=signal.device, dtype=signal.dtype)
    phase = -2.0 * math.pi * flat_delay[:, None] * frequency_bins[None, :] / total_samples
    phase_shift = torch.polar(torch.ones_like(phase), phase)
    shifted = torch.fft.irfft(spectrum * phase_shift, n=total_samples, dim=-1)
    return shifted.reshape(*delay_samples.shape, total_samples)


def _apply_elevation_spectral_cue(
    signal: torch.Tensor,
    elevation_deg: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0:
        return signal
    total_samples = signal.shape[-1]
    spectrum = torch.fft.rfft(signal, dim=-1)
    frequency_bins = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=signal.device)
    elevation_scale = torch.tanh(torch.deg2rad(elevation_deg) / (math.pi / 6.0))
    spectral_tilt = torch.exp(strength * elevation_scale[:, None, None] * (frequency_bins[None, None, :] - 0.5))
    shaped = torch.fft.irfft(spectrum * spectral_tilt, n=total_samples, dim=-1)
    return shaped


def simulate_echo_batch(
    config: GlobalConfig,
    radii_m: torch.Tensor,
    azimuth_deg: torch.Tensor | None = None,
    elevation_deg: torch.Tensor | None = None,
    binaural: bool = False,
    add_noise: bool = True,
    include_elevation_cues: bool = False,
) -> AcousticBatch:
    batch_size = radii_m.shape[0]
    device = radii_m.device
    azimuth_deg = torch.zeros_like(radii_m) if azimuth_deg is None else azimuth_deg
    elevation_deg = torch.zeros_like(radii_m) if elevation_deg is None else elevation_deg

    chirp, _ = generate_fm_chirp(config, batch_size=batch_size, device=device)
    transmit = pad_signal(chirp, config.signal_samples)

    azimuth_rad = torch.deg2rad(azimuth_deg)
    elevation_rad = torch.deg2rad(elevation_deg)

    x_coord = radii_m * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y_coord = radii_m * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z_coord = radii_m * torch.sin(elevation_rad)

    if binaural:
        ear_offsets = torch.tensor(
            [-config.ear_spacing_m / 2.0, config.ear_spacing_m / 2.0],
            dtype=torch.float32,
            device=device,
        )
        ear_y = y_coord[:, None] - ear_offsets[None, :]
        distance_to_ear = torch.sqrt(x_coord[:, None].square() + ear_y.square() + z_coord[:, None].square())
        path_lengths = radii_m[:, None] + distance_to_ear
        head_shadow = torch.exp(
            config.head_shadow_strength
            * torch.sin(azimuth_rad)[:, None]
            * torch.tensor([-1.0, 1.0], device=device)[None, :]
        )
    else:
        distance_to_ear = radii_m[:, None]
        path_lengths = 2.0 * radii_m[:, None]
        head_shadow = torch.ones(batch_size, 1, device=device)

    jitter_s = torch.randn_like(path_lengths) * config.jitter_std_s
    delays_s = path_lengths / config.speed_of_sound_m_s + jitter_s
    delay_samples = delays_s * config.sample_rate_hz

    base_echo = fractional_delay(transmit.unsqueeze(1).expand(-1, path_lengths.shape[1], -1), delay_samples)
    amplitudes = 0.7 / path_lengths.square().clamp_min(0.25)
    echoes = base_echo * amplitudes[:, :, None] * head_shadow[:, :, None]

    if include_elevation_cues:
        echoes = _apply_elevation_spectral_cue(echoes, elevation_deg, config.elevation_spectral_strength)

    receive = echoes
    if add_noise:
        receive = receive + config.noise_std * torch.randn_like(receive)

    itd_s = None
    ild_db = None
    if binaural:
        itd_s = delays_s[:, 1] - delays_s[:, 0]
        ild_db = 20.0 * torch.log10((amplitudes[:, 1] / amplitudes[:, 0]).clamp_min(1e-6))

    return AcousticBatch(
        transmit=transmit,
        receive=receive,
        delays_s=delays_s,
        amplitudes=amplitudes,
        radii_m=radii_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        itd_s=itd_s,
        ild_db=ild_db,
    )


def matched_filter_distance(
    receive_signal: torch.Tensor,
    chirp_template: torch.Tensor,
    sample_rate_hz: int,
    speed_of_sound_m_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if receive_signal.ndim == 3:
        receive_signal = receive_signal.mean(dim=1)
    template = chirp_template[:, : chirp_template.shape[-1] // 8 * 8]
    if template.shape[-1] == 0:
        template = chirp_template
    kernel = template[:, :].flip(-1).unsqueeze(1)
    correlation = F.conv1d(receive_signal.unsqueeze(1), kernel[:1], padding=0).squeeze(1)
    peak_indices = correlation.argmax(dim=-1)
    delay_s = peak_indices.to(torch.float32) / sample_rate_hz
    distance_m = 0.5 * speed_of_sound_m_s * delay_s
    return distance_m, correlation


def cochlea_filterbank(
    signal: torch.Tensor,
    sample_rate_hz: int,
    num_channels: int,
    low_hz: float,
    high_hz: float,
    envelope_lowpass_hz: float,
    downsample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_samples = signal.shape[-1]
    flat_signal = signal.reshape(-1, total_samples)
    spectrum = torch.fft.rfft(flat_signal, dim=-1)
    frequencies = torch.fft.rfftfreq(total_samples, d=1.0 / sample_rate_hz, device=signal.device)
    safe_frequencies = frequencies.clamp_min(low_hz / 4.0)
    log_frequencies = torch.log(safe_frequencies)
    center_frequencies = torch.logspace(
        math.log10(low_hz),
        math.log10(high_hz),
        num_channels,
        device=signal.device,
    )
    log_centers = torch.log(center_frequencies)
    sigma = 0.16
    filters = torch.exp(-0.5 * ((log_frequencies[None, :] - log_centers[:, None]) / sigma).square())
    filters[:, 0] = 0.0
    filtered = torch.fft.irfft(spectrum[:, None, :] * filters[None, :, :], n=total_samples, dim=-1)
    rectified = F.relu(filtered)

    lowpass_kernel_size = max(3, int(round(sample_rate_hz / envelope_lowpass_hz)))
    if lowpass_kernel_size % 2 == 0:
        lowpass_kernel_size += 1
    kernel = torch.hann_window(lowpass_kernel_size, periodic=False, device=signal.device)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    smoothed = F.conv1d(
        rectified.reshape(-1, 1, total_samples),
        kernel,
        padding=lowpass_kernel_size // 2,
    ).reshape(rectified.shape)
    if downsample > 1:
        smoothed = F.avg_pool1d(smoothed, kernel_size=downsample, stride=downsample)
    return smoothed.reshape(*signal.shape[:-1], num_channels, -1), center_frequencies


def lif_encode(envelope: torch.Tensor, threshold: float, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_steps = envelope.shape[-1]
    scaled_envelope = 1.35 * envelope / envelope.amax().clamp_min(1e-6)
    flat_envelope = scaled_envelope.reshape(-1, total_steps)
    membrane = torch.zeros(flat_envelope.shape[0], device=envelope.device, dtype=envelope.dtype)
    membrane_trace = []
    spikes = []
    for time_index in range(total_steps):
        membrane = beta * membrane + flat_envelope[:, time_index]
        spike = (membrane >= threshold).to(envelope.dtype)
        membrane = (membrane - spike * threshold).clamp_min(0.0)
        membrane_trace.append(membrane)
        spikes.append(spike)
    spike_tensor = torch.stack(spikes, dim=-1).reshape_as(envelope)
    membrane_tensor = torch.stack(membrane_trace, dim=-1).reshape_as(envelope)
    return spike_tensor, membrane_tensor


def cochlea_to_spikes(
    signal: torch.Tensor,
    config: GlobalConfig,
) -> dict[str, torch.Tensor]:
    cochleagram, center_frequencies = cochlea_filterbank(
        signal=signal,
        sample_rate_hz=config.sample_rate_hz,
        num_channels=config.num_cochlea_channels,
        low_hz=config.cochlea_low_hz,
        high_hz=config.cochlea_high_hz,
        envelope_lowpass_hz=config.envelope_lowpass_hz,
        downsample=config.envelope_downsample,
    )
    spikes, membrane = lif_encode(
        envelope=cochleagram,
        threshold=config.spike_threshold,
        beta=config.spike_beta,
    )
    return {
        "cochleagram": cochleagram,
        "spikes": spikes,
        "membrane": membrane,
        "center_frequencies": center_frequencies,
    }


def spike_density(spikes: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = torch.ones(1, 1, kernel_size, device=spikes.device, dtype=spikes.dtype) / kernel_size
    return F.conv1d(spikes.reshape(-1, 1, spikes.shape[-1]), kernel, padding=kernel_size // 2).reshape_as(spikes)


def sample_uniform_positions(
    count: int,
    config: GlobalConfig,
    device: torch.device,
    azimuth_limits_deg: tuple[float, float] = (-60.0, 60.0),
    elevation_limits_deg: tuple[float, float] = (-30.0, 30.0),
    include_elevation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radii = torch.rand(count, device=device) * (config.max_range_m - config.min_range_m) + config.min_range_m
    azimuth = torch.rand(count, device=device) * (azimuth_limits_deg[1] - azimuth_limits_deg[0]) + azimuth_limits_deg[0]
    if include_elevation:
        elevation = (
            torch.rand(count, device=device) * (elevation_limits_deg[1] - elevation_limits_deg[0])
            + elevation_limits_deg[0]
        )
    else:
        elevation = torch.zeros(count, device=device)
    return radii, azimuth, elevation


def balanced_distance_dataset(
    count_per_class: int,
    config: GlobalConfig,
    device: torch.device,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_edges = torch.linspace(config.min_range_m, config.max_range_m, num_classes + 1, device=device)
    radii = []
    labels = []
    for class_index in range(num_classes):
        low_edge = class_edges[class_index]
        high_edge = class_edges[class_index + 1]
        class_radii = torch.rand(count_per_class, device=device) * (high_edge - low_edge) + low_edge
        radii.append(class_radii)
        labels.append(torch.full((count_per_class,), class_index, device=device, dtype=torch.long))
    return torch.cat(radii), torch.cat(labels)
