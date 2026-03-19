from __future__ import annotations

import math
from dataclasses import dataclass

import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pathway_snn import PathwayBatch
from models.snn import surrogate_spike
from utils.common import GlobalConfig


def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    scale = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return waveform / scale


def _temporal_onset(features: torch.Tensor) -> torch.Tensor:
    return F.relu(torch.diff(features, dim=-1, prepend=features[..., :1]))


def _pool_temporal_features(features: torch.Tensor, bins: int = 4) -> torch.Tensor:
    pooled_avg = F.adaptive_avg_pool1d(features, bins).flatten(start_dim=1)
    pooled_max = F.adaptive_max_pool1d(features, bins).flatten(start_dim=1)
    return torch.cat([pooled_avg, pooled_max], dim=-1)


def initial_filterbank_weights(
    num_filters: int,
    kernel_size: int,
    sample_rate_hz: int,
    low_hz: float,
    high_hz: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    weight_device = device if device is not None else torch.device("cpu")
    weight_dtype = dtype if dtype is not None else torch.float32
    time_axis = (
        torch.arange(kernel_size, device=weight_device, dtype=weight_dtype) - (kernel_size - 1) / 2.0
    ) / sample_rate_hz
    center_frequencies = torch.logspace(
        math.log10(low_hz),
        math.log10(high_hz),
        num_filters,
        device=weight_device,
        dtype=weight_dtype,
    )
    sigma = (1.6 / center_frequencies).unsqueeze(-1)
    gaussian_window = torch.exp(-0.5 * (time_axis.unsqueeze(0) / sigma).square())
    carrier = torch.cos(2.0 * math.pi * center_frequencies.unsqueeze(-1) * time_axis.unsqueeze(0))
    weights = gaussian_window * carrier
    weights = weights - weights.mean(dim=-1, keepdim=True)
    weights = weights / weights.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return weights.unsqueeze(1)


@dataclass
class ExperimentBatch:
    transmit_wave: torch.Tensor
    receive_wave: torch.Tensor
    pathway: PathwayBatch | None
    transmit_spikes: torch.Tensor | None
    receive_spikes: torch.Tensor | None
    spike_count: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "ExperimentBatch":
        return ExperimentBatch(
            transmit_wave=self.transmit_wave[indices],
            receive_wave=self.receive_wave[indices],
            pathway=None if self.pathway is None else self.pathway.index_select(indices),
            transmit_spikes=None if self.transmit_spikes is None else self.transmit_spikes[indices],
            receive_spikes=None if self.receive_spikes is None else self.receive_spikes[indices],
            spike_count=self.spike_count[indices],
        )


@dataclass
class BranchEncoding:
    distance_latent: torch.Tensor
    azimuth_latent: torch.Tensor
    elevation_latent: torch.Tensor
    spectral_source: torch.Tensor | None
    spike_proxy: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class HandcraftedBranchEncoder(nn.Module):
    def __init__(self, distance_dim: int, azimuth_dim: int, elevation_dim: int, branch_hidden_dim: int) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.distance_branch = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_branch = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_branch = nn.Linear(elevation_dim, branch_hidden_dim)

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None:
            raise ValueError("HandcraftedBranchEncoder requires pathway features.")
        distance_latent = F.relu(self.distance_branch(batch.pathway.distance))
        azimuth_latent = F.relu(self.azimuth_branch(batch.pathway.azimuth))
        elevation_latent = F.relu(self.elevation_branch(batch.pathway.elevation))
        diagnostics = {
            "distance_latent": distance_latent,
            "azimuth_latent": azimuth_latent,
            "elevation_latent": elevation_latent,
        }
        spectral_source = None
        if batch.receive_spikes is not None:
            spectral_source = batch.receive_spikes.float()
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=spectral_source,
            spike_proxy=batch.spike_count.float(),
            diagnostics=diagnostics,
        )


class LearnedBranchEncoder(nn.Module):
    def __init__(
        self,
        config: GlobalConfig,
        *,
        num_filters: int,
        num_delay_lines: int,
        branch_hidden_dim: int,
        kernel_size: int = 63,
    ) -> None:
        super().__init__()
        self.config = config
        self.branch_hidden_dim = branch_hidden_dim
        self.num_filters = num_filters
        self.num_delay_lines = num_delay_lines
        self.kernel_size = kernel_size

        self.filterbank = nn.Conv1d(1, num_filters, kernel_size=kernel_size, stride=4, padding=kernel_size // 2, bias=False)
        init_weight = initial_filterbank_weights(
            num_filters,
            kernel_size,
            config.sample_rate_hz,
            config.cochlea_low_hz,
            config.cochlea_high_hz,
        )
        with torch.no_grad():
            self.filterbank.weight.copy_(init_weight)
        self.register_buffer("initial_filterbank_weight", init_weight.clone())

        self.distance_delay = nn.Conv1d(num_filters * 2, num_delay_lines, kernel_size=17, padding=8, bias=False)
        self.azimuth_delay = nn.Conv1d(num_filters * 2, num_delay_lines, kernel_size=17, padding=8, bias=False)
        self.register_buffer("initial_distance_kernel", self.distance_delay.weight.detach().clone())
        self.register_buffer("initial_azimuth_kernel", self.azimuth_delay.weight.detach().clone())

        self.distance_branch = nn.Linear(num_delay_lines * 16, branch_hidden_dim)
        self.azimuth_branch = nn.Linear(num_delay_lines * 8, branch_hidden_dim)
        self.spectral_conv1 = nn.Conv2d(2, 8, kernel_size=(3, 5), padding=(1, 2))
        self.spectral_conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1)
        self.elevation_branch = nn.Linear(8 * 4 * 4, branch_hidden_dim)

    def _frontend(self, waveform: torch.Tensor) -> torch.Tensor:
        normalized = _normalize_waveform(waveform)
        cochlea = F.relu(self.filterbank(normalized))
        cochlea = F.avg_pool1d(cochlea, kernel_size=4, stride=4)
        return cochlea

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        transmit = self._frontend(batch.transmit_wave.unsqueeze(1))
        left = self._frontend(batch.receive_wave[:, 0].unsqueeze(1))
        right = self._frontend(batch.receive_wave[:, 1].unsqueeze(1))

        transmit_onset = _temporal_onset(transmit)
        left_onset = _temporal_onset(left)
        right_onset = _temporal_onset(right)

        distance_left = self.distance_delay(torch.cat([transmit_onset, left_onset], dim=1))
        distance_right = self.distance_delay(torch.cat([transmit_onset, right_onset], dim=1))
        distance_features = torch.cat(
            [_pool_temporal_features(distance_left), _pool_temporal_features(distance_right)],
            dim=-1,
        )
        distance_latent = F.relu(self.distance_branch(distance_features))

        azimuth_input = torch.cat([left_onset - right_onset, left_onset + right_onset], dim=1)
        azimuth_map = self.azimuth_delay(azimuth_input)
        azimuth_latent = F.relu(self.azimuth_branch(_pool_temporal_features(azimuth_map)))

        spectral_source = torch.stack([left, right], dim=1)
        elevation_map = F.relu(self.spectral_conv1(spectral_source))
        elevation_map = F.relu(self.spectral_conv2(elevation_map))
        elevation_latent = F.relu(self.elevation_branch(F.adaptive_avg_pool2d(elevation_map, (4, 4)).flatten(start_dim=1)))

        spike_proxy = (left_onset > 0.0).float().sum(dim=(1, 2)) + (right_onset > 0.0).float().sum(dim=(1, 2))
        diagnostics = {
            "transmit_frontend": transmit,
            "left_frontend": left,
            "right_frontend": right,
            "distance_left_map": distance_left,
            "distance_right_map": distance_right,
            "azimuth_map": azimuth_map,
            "elevation_map": elevation_map,
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=spectral_source,
            spike_proxy=spike_proxy,
            diagnostics=diagnostics,
        )


class ResidualElevationEncoder(nn.Module):
    def __init__(self, distance_dim: int, azimuth_dim: int, elevation_dim: int, branch_hidden_dim: int) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.distance_branch = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_branch = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_branch = nn.Linear(elevation_dim, branch_hidden_dim)
        self.elevation_conv1 = nn.Conv2d(2, 8, kernel_size=(5, 7), padding=(2, 3))
        self.elevation_conv2 = nn.Conv2d(8, 8, kernel_size=(3, 5), padding=(1, 2))
        self.elevation_residual = nn.Linear(8 * 4 * 4, branch_hidden_dim)
        self.residual_gain = nn.Parameter(torch.tensor(-0.6))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None or batch.receive_spikes is None:
            raise ValueError("ResidualElevationEncoder requires pathway and spike inputs.")
        distance_latent = F.relu(self.distance_branch(batch.pathway.distance))
        azimuth_latent = F.relu(self.azimuth_branch(batch.pathway.azimuth))
        base_elevation = F.relu(self.elevation_branch(batch.pathway.elevation))

        spectral_source = batch.receive_spikes.float()
        elevation_map = F.relu(self.elevation_conv1(spectral_source))
        elevation_map = F.relu(self.elevation_conv2(elevation_map))
        learned_elevation = self.elevation_residual(F.adaptive_avg_pool2d(elevation_map, (4, 4)).flatten(start_dim=1))
        residual_scale = 0.5 * torch.sigmoid(self.residual_gain)
        elevation_latent = F.relu(base_elevation + residual_scale * learned_elevation)

        diagnostics = {
            "distance_latent": distance_latent,
            "azimuth_latent": azimuth_latent,
            "base_elevation_latent": base_elevation,
            "learned_elevation_latent": learned_elevation,
            "elevation_latent": elevation_latent,
            "elevation_map": elevation_map,
            "elevation_residual_scale": residual_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=spectral_source,
            spike_proxy=batch.spike_count.float(),
            diagnostics=diagnostics,
        )


class ElevationSConvResidualEncoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        sconv_channels: int = 4,
        temporal_pool: int = 12,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.temporal_pool = max(4, temporal_pool)
        self.distance_branch = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_branch = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_branch = nn.Linear(elevation_dim, branch_hidden_dim)
        self.sconv = snn.SConv2dLSTM(
            in_channels=1,
            out_channels=sconv_channels,
            kernel_size=(3, 1),
            threshold=1.0,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="none",
            output=True,
        )
        self.sconv_projection = nn.Linear(sconv_channels, branch_hidden_dim)
        self.residual_gain = nn.Parameter(torch.tensor(-1.0))

    def _sconv_context(self, spectral_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.sconv.reset_mem()
        reduced = F.avg_pool1d(
            spectral_source.reshape(-1, 1, spectral_source.shape[-1]),
            kernel_size=self.temporal_pool,
            stride=self.temporal_pool,
        ).reshape(spectral_source.shape[0], spectral_source.shape[1], spectral_source.shape[2], -1)
        syn = None
        mem = None
        spike_trace = []
        for time_index in range(reduced.shape[-1]):
            frame = reduced[:, :, :, time_index].permute(0, 2, 1).unsqueeze(1)
            spikes, syn, mem = self.sconv(frame, syn, mem)
            spike_trace.append(spikes)
        spike_tensor = torch.stack(spike_trace, dim=1)
        self.sconv.reset_mem()
        pooled = spike_tensor.mean(dim=(1, 3, 4))
        return pooled, spike_tensor

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None or batch.receive_spikes is None:
            raise ValueError("ElevationSConvResidualEncoder requires pathway and spike inputs.")
        distance_latent = F.relu(self.distance_branch(batch.pathway.distance))
        azimuth_latent = F.relu(self.azimuth_branch(batch.pathway.azimuth))
        base_elevation = F.relu(self.elevation_branch(batch.pathway.elevation))

        spectral_source = batch.receive_spikes.float()
        context, spike_trace = self._sconv_context(spectral_source)
        residual = self.sconv_projection(context)
        residual_scale = 0.4 * torch.sigmoid(self.residual_gain)
        elevation_latent = F.relu(base_elevation + residual_scale * residual)
        diagnostics = {
            "distance_latent": distance_latent,
            "azimuth_latent": azimuth_latent,
            "base_elevation_latent": base_elevation,
            "sconv_elevation_residual": residual,
            "elevation_latent": elevation_latent,
            "elevation_sconv_spikes": spike_trace,
            "elevation_residual_scale": residual_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=spectral_source,
            spike_proxy=batch.spike_count.float(),
            diagnostics=diagnostics,
        )


class CombinedElevationEncoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        sconv_channels: int = 4,
        temporal_pool: int = 12,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.temporal_pool = max(4, temporal_pool)
        self.distance_branch = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_branch = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_branch = nn.Linear(elevation_dim, branch_hidden_dim)
        self.elevation_conv1 = nn.Conv2d(2, 8, kernel_size=(5, 7), padding=(2, 3))
        self.elevation_conv2 = nn.Conv2d(8, 8, kernel_size=(3, 5), padding=(1, 2))
        self.elevation_residual = nn.Linear(8 * 4 * 4, branch_hidden_dim)
        self.sconv = snn.SConv2dLSTM(
            in_channels=1,
            out_channels=sconv_channels,
            kernel_size=(3, 1),
            threshold=1.0,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="none",
            output=True,
        )
        self.sconv_projection = nn.Linear(sconv_channels, branch_hidden_dim)
        self.cnn_residual_gain = nn.Parameter(torch.tensor(-0.8))
        self.sconv_residual_gain = nn.Parameter(torch.tensor(-1.0))

    def _sconv_context(self, spectral_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.sconv.reset_mem()
        reduced = F.avg_pool1d(
            spectral_source.reshape(-1, 1, spectral_source.shape[-1]),
            kernel_size=self.temporal_pool,
            stride=self.temporal_pool,
        ).reshape(spectral_source.shape[0], spectral_source.shape[1], spectral_source.shape[2], -1)
        syn = None
        mem = None
        spike_trace = []
        for time_index in range(reduced.shape[-1]):
            frame = reduced[:, :, :, time_index].permute(0, 2, 1).unsqueeze(1)
            spikes, syn, mem = self.sconv(frame, syn, mem)
            spike_trace.append(spikes)
        spike_tensor = torch.stack(spike_trace, dim=1)
        self.sconv.reset_mem()
        pooled = spike_tensor.mean(dim=(1, 3, 4))
        return pooled, spike_tensor

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None or batch.receive_spikes is None:
            raise ValueError("CombinedElevationEncoder requires pathway and spike inputs.")
        distance_latent = F.relu(self.distance_branch(batch.pathway.distance))
        azimuth_latent = F.relu(self.azimuth_branch(batch.pathway.azimuth))
        base_elevation = F.relu(self.elevation_branch(batch.pathway.elevation))

        spectral_source = batch.receive_spikes.float()
        elevation_map = F.relu(self.elevation_conv1(spectral_source))
        elevation_map = F.relu(self.elevation_conv2(elevation_map))
        learned_elevation = self.elevation_residual(F.adaptive_avg_pool2d(elevation_map, (4, 4)).flatten(start_dim=1))

        sconv_context, spike_trace = self._sconv_context(spectral_source)
        sconv_residual = self.sconv_projection(sconv_context)

        cnn_scale = 0.5 * torch.sigmoid(self.cnn_residual_gain)
        sconv_scale = 0.4 * torch.sigmoid(self.sconv_residual_gain)
        elevation_latent = F.relu(base_elevation + cnn_scale * learned_elevation + sconv_scale * sconv_residual)

        diagnostics = {
            "distance_latent": distance_latent,
            "azimuth_latent": azimuth_latent,
            "base_elevation_latent": base_elevation,
            "learned_elevation_latent": learned_elevation,
            "sconv_elevation_residual": sconv_residual,
            "elevation_latent": elevation_latent,
            "elevation_map": elevation_map,
            "elevation_sconv_spikes": spike_trace,
            "cnn_residual_scale": cnn_scale.detach(),
            "sconv_residual_scale": sconv_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=spectral_source,
            spike_proxy=batch.spike_count.float(),
            diagnostics=diagnostics,
        )


class ExperimentalPathwayModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_dim: int,
        output_dim: int,
        num_steps: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
        use_resonant: bool = False,
        use_sconv: bool = False,
        sconv_channels: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.use_resonant = use_resonant
        self.use_sconv = use_sconv

        branch_hidden_dim = getattr(encoder, "branch_hidden_dim")
        extra_dim = sconv_channels if use_sconv else 0
        fused_dim = branch_hidden_dim * 3 + extra_dim

        if use_sconv:
            self.sconv = snn.SConv2dLSTM(
                in_channels=1,
                out_channels=sconv_channels,
                kernel_size=(3, 1),
                threshold=threshold,
                spike_grad=surrogate.fast_sigmoid(),
                reset_mechanism="none",
                output=True,
            )

        if use_resonant:
            self.fusion_in = nn.Linear(fused_dim, hidden_dim)
            self.recurrent = nn.Linear(hidden_dim, hidden_dim)
            self.readout = nn.Linear(hidden_dim, output_dim)
            self.resonant_decay = nn.Parameter(torch.full((hidden_dim,), 0.6))
            self.resonant_frequency = nn.Parameter(torch.full((hidden_dim,), 0.2))
        else:
            self.fusion = nn.Linear(fused_dim, hidden_dim)
            self.fusion_lif = snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=surrogate.fast_sigmoid(),
                reset_mechanism=reset_mechanism,
            )
            self.integration = nn.Linear(hidden_dim, hidden_dim)
            self.integration_lif = snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=surrogate.fast_sigmoid(),
                reset_mechanism=reset_mechanism,
            )
            self.readout = nn.Linear(hidden_dim, output_dim)

    def _sconv_context(self, spectral_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.sconv.reset_mem()
        reduced = F.avg_pool1d(
            spectral_source.reshape(-1, 1, spectral_source.shape[-1]),
            kernel_size=16,
            stride=16,
        ).reshape(spectral_source.shape[0], spectral_source.shape[1], spectral_source.shape[2], -1)
        syn = None
        mem = None
        spike_trace = []
        for time_index in range(reduced.shape[-1]):
            frame = reduced[:, :, :, time_index].permute(0, 2, 1).unsqueeze(1)
            spikes, syn, mem = self.sconv(frame, syn, mem)
            spike_trace.append(spikes)
        spike_tensor = torch.stack(spike_trace, dim=1)
        self.sconv.reset_mem()
        pooled = spike_tensor.mean(dim=(1, 3, 4))
        return pooled, spike_tensor

    def forward(self, batch: ExperimentBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoding = self.encoder(batch)
        fused_parts = [encoding.distance_latent, encoding.azimuth_latent, encoding.elevation_latent]
        diagnostics = {**encoding.diagnostics}
        spike_rates = []

        if self.use_sconv:
            if encoding.spectral_source is None:
                raise ValueError("SConv augmentation requires a spectral source tensor.")
            sconv_latent, sconv_trace = self._sconv_context(encoding.spectral_source)
            fused_parts.append(sconv_latent)
            diagnostics["sconv_spikes"] = sconv_trace
            spike_rates.append(sconv_trace.mean(dim=(1, 2, 3, 4)))

        fused = torch.cat(fused_parts, dim=-1)
        diagnostics["fused_latent"] = fused

        if self.use_resonant:
            base_current = self.fusion_in(fused)
            membrane = torch.zeros_like(base_current)
            resonance = torch.zeros_like(base_current)
            recurrent_current = torch.zeros_like(base_current)
            spike_trace = []
            decay = 0.55 + 0.4 * torch.sigmoid(self.resonant_decay)
            frequency = 0.1 + 0.9 * torch.sigmoid(self.resonant_frequency)
            for _ in range(self.num_steps):
                drive = base_current + recurrent_current
                resonance = decay * resonance + frequency * membrane
                membrane = decay * membrane + drive - resonance
                spikes = surrogate_spike(membrane - self.threshold)
                membrane = membrane - spikes * self.threshold
                recurrent_current = self.recurrent(spikes)
                spike_trace.append(spikes)
            resonant_spikes = torch.stack(spike_trace, dim=1)
            pooled = resonant_spikes.mean(dim=1)
            output = self.readout(pooled)
            diagnostics["resonant_spikes"] = resonant_spikes
            diagnostics["resonant_frequency"] = frequency
            diagnostics["resonant_decay"] = decay
            spike_rates.append(resonant_spikes.mean(dim=(1, 2)))
        else:
            self.fusion_lif.reset_mem()
            self.integration_lif.reset_mem()
            fusion_current = self.fusion(fused)
            fusion_mem = None
            integration_mem = None
            fusion_spike_trace = []
            integration_spike_trace = []
            for _ in range(self.num_steps):
                fusion_spikes, fusion_mem = self.fusion_lif(fusion_current, fusion_mem)
                integration_current = self.integration(fusion_spikes)
                integration_spikes, integration_mem = self.integration_lif(integration_current, integration_mem)
                fusion_spike_trace.append(fusion_spikes)
                integration_spike_trace.append(integration_spikes)
            fusion_spikes = torch.stack(fusion_spike_trace, dim=1)
            integration_spikes = torch.stack(integration_spike_trace, dim=1)
            pooled = integration_spikes.mean(dim=1)
            output = self.readout(pooled)
            diagnostics["fusion_spikes"] = fusion_spikes
            diagnostics["integration_spikes"] = integration_spikes
            spike_rates.append(integration_spikes.mean(dim=(1, 2)))

        if not spike_rates:
            spike_rates.append(encoding.spike_proxy / encoding.spike_proxy.amax().clamp_min(1.0))
        diagnostics["spike_rate"] = torch.stack(spike_rates, dim=0).mean(dim=0)
        diagnostics["spike_proxy"] = encoding.spike_proxy
        return output, diagnostics


class DistanceResonanceModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_dim: int,
        output_dim: int,
        num_steps: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_steps = num_steps
        self.threshold = threshold
        branch_hidden_dim = getattr(encoder, "branch_hidden_dim")

        self.distance_drive = nn.Linear(branch_hidden_dim, branch_hidden_dim)
        self.distance_recurrent = nn.Linear(branch_hidden_dim, branch_hidden_dim)
        self.distance_decay = nn.Parameter(torch.zeros(branch_hidden_dim))
        self.distance_frequency = nn.Parameter(torch.zeros(branch_hidden_dim))

        self.fusion = nn.Linear(branch_hidden_dim * 3, hidden_dim)
        self.fusion_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism=reset_mechanism,
        )
        self.integration = nn.Linear(hidden_dim, hidden_dim)
        self.integration_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism=reset_mechanism,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch: ExperimentBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoding = self.encoder(batch)
        distance_drive = self.distance_drive(encoding.distance_latent)
        membrane = torch.zeros_like(distance_drive)
        resonance = torch.zeros_like(distance_drive)
        recurrent = torch.zeros_like(distance_drive)
        resonance_trace = []

        decay = 0.75 + 0.12 * torch.sigmoid(self.distance_decay)
        frequency = 0.05 + 0.20 * torch.sigmoid(self.distance_frequency)
        for _ in range(self.num_steps):
            drive = distance_drive + recurrent
            resonance = decay * resonance + frequency * membrane
            membrane = decay * membrane + drive - resonance
            spikes = surrogate_spike(membrane - self.threshold)
            membrane = membrane - spikes * self.threshold
            recurrent = self.distance_recurrent(spikes)
            resonance_trace.append(spikes)

        distance_spikes = torch.stack(resonance_trace, dim=1)
        distance_context = distance_spikes.mean(dim=1)
        fused = torch.cat([distance_context, encoding.azimuth_latent, encoding.elevation_latent], dim=-1)

        self.fusion_lif.reset_mem()
        self.integration_lif.reset_mem()
        fusion_current = self.fusion(fused)
        fusion_mem = None
        integration_mem = None
        fusion_spike_trace = []
        integration_spike_trace = []
        for _ in range(self.num_steps):
            fusion_spikes, fusion_mem = self.fusion_lif(fusion_current, fusion_mem)
            integration_current = self.integration(fusion_spikes)
            integration_spikes, integration_mem = self.integration_lif(integration_current, integration_mem)
            fusion_spike_trace.append(fusion_spikes)
            integration_spike_trace.append(integration_spikes)

        fusion_spikes = torch.stack(fusion_spike_trace, dim=1)
        integration_spikes = torch.stack(integration_spike_trace, dim=1)
        pooled = integration_spikes.mean(dim=1)
        output = self.output(pooled)
        diagnostics = {
            **encoding.diagnostics,
            "distance_resonance_spikes": distance_spikes,
            "distance_resonance_decay": decay,
            "distance_resonance_frequency": frequency,
            "fusion_spikes": fusion_spikes,
            "integration_spikes": integration_spikes,
            "spike_rate": 0.5 * (
                distance_spikes.mean(dim=(1, 2)) + integration_spikes.mean(dim=(1, 2))
            ),
        }
        return output, diagnostics
