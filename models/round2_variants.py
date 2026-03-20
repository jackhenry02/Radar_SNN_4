from __future__ import annotations

import math

import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_variants import BranchEncoding, CombinedElevationEncoder, ExperimentBatch
from models.pathway_snn import build_pathway_features
from models.snn import surrogate_spike


def _identity_init_1x1(conv: nn.Conv1d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        diagonal = min(conv.out_channels, conv.in_channels)
        for index in range(diagonal):
            conv.weight[index, index, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _shift_feature_axis(features: torch.Tensor, offsets: torch.Tensor, max_shift: float) -> torch.Tensor:
    feature_dim = features.shape[-1]
    base = torch.arange(feature_dim, device=features.device, dtype=features.dtype)
    shifted = (base + max_shift * torch.tanh(offsets).to(features.dtype)).clamp(0.0, float(feature_dim - 1))
    lower = shifted.floor().to(torch.long)
    upper = shifted.ceil().to(torch.long)
    alpha = (shifted - lower.to(features.dtype)).view(1, -1)
    return features[:, lower] * (1.0 - alpha) + features[:, upper] * alpha


def _shift_channel_axis(features: torch.Tensor, offsets: torch.Tensor, max_shift: float) -> torch.Tensor:
    batch, ears, channels, steps = features.shape
    reshaped = features.permute(0, 1, 3, 2).reshape(-1, channels)
    shifted = _shift_feature_axis(reshaped, offsets, max_shift)
    return shifted.reshape(batch, ears, steps, channels).permute(0, 1, 3, 2)


def _gain_vector(parameter: torch.Tensor, max_delta: float) -> torch.Tensor:
    return 1.0 + max_delta * torch.tanh(parameter)


class SpikeSequencePreprocessor(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        beta: float = 0.9,
        threshold: float = 0.75,
    ) -> None:
        super().__init__()
        self.mix = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        _identity_init_1x1(self.mix)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="subtract",
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        mixed = self.mix(spikes.float())
        self.lif.reset_mem()
        membrane = None
        spike_trace = []
        for time_index in range(mixed.shape[-1]):
            output, membrane = self.lif(mixed[..., time_index], membrane)
            spike_trace.append(output)
        return torch.stack(spike_trace, dim=-1)


class LatentResidualLIF(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        beta: float,
        threshold: float,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.projection = nn.Linear(latent_dim, latent_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="subtract",
        )
        self.gain = nn.Parameter(torch.tensor(-0.9))

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.projection(latent)
        self.lif.reset_mem()
        membrane = None
        spike_trace = []
        for _ in range(self.num_steps):
            spikes, membrane = self.lif(current, membrane)
            spike_trace.append(spikes)
        spike_tensor = torch.stack(spike_trace, dim=1)
        scale = 0.4 * torch.sigmoid(self.gain)
        updated = F.relu(latent + scale * spike_tensor.mean(dim=1))
        return updated, spike_tensor, scale.detach()


class CorollaryResonanceBank(nn.Module):
    def __init__(
        self,
        num_frequency_channels: int,
        *,
        num_resonators: int = 16,
        time_bins: int = 128,
    ) -> None:
        super().__init__()
        self.time_bins = time_bins
        self.trace_projection = nn.Conv1d(3, num_resonators, kernel_size=1, bias=True)
        self.frequency = nn.Parameter(torch.linspace(-1.5, 1.5, num_resonators))
        self.decay = nn.Parameter(torch.zeros(num_resonators))
        self.threshold = nn.Parameter(torch.zeros(num_resonators))
        self.transmit_weight = nn.Parameter(torch.tensor(0.0))
        self.receive_weight = nn.Parameter(torch.tensor(0.0))
        channel_axis = torch.linspace(-1.0, 1.0, num_frequency_channels)
        self.register_buffer("channel_axis", channel_axis)

    def _build_traces(self, batch: ExperimentBatch) -> torch.Tensor:
        if batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("CorollaryResonanceBank requires spike inputs.")
        transmit = batch.transmit_spikes.float().sum(dim=1)
        left = batch.receive_spikes[:, 0].float().sum(dim=1)
        right = batch.receive_spikes[:, 1].float().sum(dim=1)
        receive = 0.5 * (left + right)
        azimuth = left - right

        channel_axis = self.channel_axis.to(batch.receive_spikes.device, batch.receive_spikes.dtype).view(1, 1, -1, 1)
        elevation = (batch.receive_spikes.float() * channel_axis).sum(dim=(1, 2))

        signed_distance = F.softplus(self.receive_weight) * receive - F.softplus(self.transmit_weight) * transmit
        traces = torch.stack([signed_distance, azimuth, elevation], dim=1)
        traces = F.adaptive_avg_pool1d(traces, self.time_bins)
        scale = traces.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        return traces / scale

    def forward(self, batch: ExperimentBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        traces = self._build_traces(batch)
        drive = self.trace_projection(traces)
        state = torch.zeros(drive.shape[0], drive.shape[1], device=drive.device, dtype=drive.dtype)
        velocity = torch.zeros_like(state)
        spike_trace = []

        frequency = 0.02 + 0.18 * torch.sigmoid(self.frequency).view(1, -1)
        decay = 0.80 + 0.18 * torch.sigmoid(self.decay).view(1, -1)
        threshold = 0.35 + 0.75 * torch.sigmoid(self.threshold).view(1, -1)

        for time_index in range(drive.shape[-1]):
            current = drive[..., time_index]
            velocity = decay * velocity + current - frequency * state
            state = state + frequency * velocity
            spikes = surrogate_spike(state - threshold)
            state = state - spikes * threshold
            spike_trace.append(spikes)

        spike_tensor = torch.stack(spike_trace, dim=1)
        pooled = spike_tensor.mean(dim=1)
        diagnostics = {
            "resonant_traces": traces,
            "resonant_drive": drive,
            "resonant_spikes": spike_tensor,
            "resonant_frequency": frequency.detach(),
            "resonant_decay": decay.detach(),
            "resonant_threshold": threshold.detach(),
            "corollary_transmit_weight": F.softplus(self.transmit_weight).detach().view(1),
            "corollary_receive_weight": F.softplus(self.receive_weight).detach().view(1),
        }
        return pooled, diagnostics


class AdaptiveResidualCombinedEncoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        num_frequency_channels: int,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.num_frequency_channels = num_frequency_channels
        self.base_encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )

        distance_half = distance_dim // 2
        azimuth_half = azimuth_dim // 2

        self.distance_left_offsets = nn.Parameter(torch.zeros(distance_half))
        self.distance_right_offsets = nn.Parameter(torch.zeros(distance_half))
        self.distance_left_gain = nn.Parameter(torch.zeros(distance_half))
        self.distance_right_gain = nn.Parameter(torch.zeros(distance_half))

        self.azimuth_itd_offsets = nn.Parameter(torch.zeros(azimuth_half))
        self.azimuth_itd_gain = nn.Parameter(torch.zeros(azimuth_half))
        self.azimuth_ild_gain = nn.Parameter(torch.zeros(azimuth_half))

        self.elevation_norm_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_notch_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_slope_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_norm_gain = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_notch_gain = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_slope_gain = nn.Parameter(torch.zeros(num_frequency_channels))
        self.spectral_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.spectral_gain = nn.Parameter(torch.zeros(num_frequency_channels))

        self.distance_residual = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_residual = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_residual = nn.Linear(elevation_dim, branch_hidden_dim)
        self.spectral_projection = nn.Linear(num_frequency_channels * 2, branch_hidden_dim)

        self.distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.azimuth_gain = nn.Parameter(torch.tensor(-1.2))
        self.elevation_gain = nn.Parameter(torch.tensor(-1.2))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None or batch.receive_spikes is None:
            raise ValueError("AdaptiveResidualCombinedEncoder requires pathway and spike inputs.")
        base = self.base_encoder(batch)

        distance_half = batch.pathway.distance.shape[-1] // 2
        distance_left = _shift_feature_axis(batch.pathway.distance[:, :distance_half], self.distance_left_offsets, 0.75)
        distance_right = _shift_feature_axis(batch.pathway.distance[:, distance_half:], self.distance_right_offsets, 0.75)
        distance_left = distance_left * _gain_vector(self.distance_left_gain, 0.35).view(1, -1)
        distance_right = distance_right * _gain_vector(self.distance_right_gain, 0.35).view(1, -1)
        adapted_distance = torch.cat([distance_left, distance_right], dim=-1)

        azimuth_half = batch.pathway.azimuth.shape[-1] // 2
        itd = _shift_feature_axis(batch.pathway.azimuth[:, :azimuth_half], self.azimuth_itd_offsets, 0.75)
        itd = itd * _gain_vector(self.azimuth_itd_gain, 0.35).view(1, -1)
        ild = batch.pathway.azimuth[:, azimuth_half:] * _gain_vector(self.azimuth_ild_gain, 0.35).view(1, -1)
        adapted_azimuth = torch.cat([itd, ild], dim=-1)

        spectral_norm = _shift_feature_axis(
            batch.pathway.elevation[:, : self.num_frequency_channels],
            self.elevation_norm_offsets,
            1.5,
        ) * _gain_vector(self.elevation_norm_gain, 0.35).view(1, -1)
        spectral_notches = _shift_feature_axis(
            batch.pathway.elevation[:, self.num_frequency_channels : 2 * self.num_frequency_channels],
            self.elevation_notch_offsets,
            1.5,
        ) * _gain_vector(self.elevation_notch_gain, 0.35).view(1, -1)
        spectral_slope = _shift_feature_axis(
            batch.pathway.elevation[:, 2 * self.num_frequency_channels : 3 * self.num_frequency_channels],
            self.elevation_slope_offsets,
            1.5,
        ) * _gain_vector(self.elevation_slope_gain, 0.35).view(1, -1)
        adapted_elevation = torch.cat([spectral_norm, spectral_notches, spectral_slope], dim=-1)

        adapted_spectral = _shift_channel_axis(batch.receive_spikes.float(), self.spectral_offsets, 1.5)
        adapted_spectral = adapted_spectral * _gain_vector(self.spectral_gain, 0.35).view(1, 1, -1, 1)
        spectral_summary = adapted_spectral.sum(dim=-1).flatten(start_dim=1)

        distance_residual = self.distance_residual(F.layer_norm(adapted_distance, (adapted_distance.shape[-1],)))
        azimuth_residual = self.azimuth_residual(F.layer_norm(adapted_azimuth, (adapted_azimuth.shape[-1],)))
        elevation_residual = self.elevation_residual(F.layer_norm(adapted_elevation, (adapted_elevation.shape[-1],)))
        elevation_residual = elevation_residual + 0.25 * self.spectral_projection(
            F.layer_norm(spectral_summary, (spectral_summary.shape[-1],))
        )

        distance_scale = 0.35 * torch.sigmoid(self.distance_gain)
        azimuth_scale = 0.35 * torch.sigmoid(self.azimuth_gain)
        elevation_scale = 0.35 * torch.sigmoid(self.elevation_gain)

        diagnostics = {
            **base.diagnostics,
            "adaptive_distance_offsets_left": torch.tanh(self.distance_left_offsets).detach(),
            "adaptive_distance_offsets_right": torch.tanh(self.distance_right_offsets).detach(),
            "adaptive_itd_offsets": torch.tanh(self.azimuth_itd_offsets).detach(),
            "adaptive_spectral_offsets": torch.tanh(self.spectral_offsets).detach(),
            "adaptive_distance_gains_left": _gain_vector(self.distance_left_gain, 0.35).detach(),
            "adaptive_distance_gains_right": _gain_vector(self.distance_right_gain, 0.35).detach(),
            "adaptive_itd_gains": _gain_vector(self.azimuth_itd_gain, 0.35).detach(),
            "adaptive_ild_gains": _gain_vector(self.azimuth_ild_gain, 0.35).detach(),
            "adaptive_spectral_gains": _gain_vector(self.spectral_gain, 0.35).detach(),
            "adaptive_distance_scale": distance_scale.detach(),
            "adaptive_azimuth_scale": azimuth_scale.detach(),
            "adaptive_elevation_scale": elevation_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * distance_residual),
            azimuth_latent=F.relu(base.azimuth_latent + azimuth_scale * azimuth_residual),
            elevation_latent=F.relu(base.elevation_latent + elevation_scale * elevation_residual),
            spectral_source=adapted_spectral,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class PrePathwayResidualEncoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        num_frequency_channels: int,
        num_delay_lines: int,
        distance_candidates: torch.Tensor,
        itd_candidates: torch.Tensor,
        beta: float,
        threshold: float,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.num_frequency_channels = num_frequency_channels
        self.num_delay_lines = num_delay_lines
        self.register_buffer("distance_candidates", distance_candidates)
        self.register_buffer("itd_candidates", itd_candidates)
        self.base_encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )
        self.transmit_pre = SpikeSequencePreprocessor(num_frequency_channels, beta=beta, threshold=threshold)
        self.receive_pre = SpikeSequencePreprocessor(num_frequency_channels, beta=beta, threshold=threshold)
        self.distance_residual = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_residual = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_residual = nn.Linear(elevation_dim, branch_hidden_dim)
        self.distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.azimuth_gain = nn.Parameter(torch.tensor(-1.2))
        self.elevation_gain = nn.Parameter(torch.tensor(-1.2))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None or batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("PrePathwayResidualEncoder requires pathway and spike inputs.")
        base = self.base_encoder(batch)

        transmit_processed = self.transmit_pre(batch.transmit_spikes.float())
        left_processed = self.receive_pre(batch.receive_spikes[:, 0].float())
        right_processed = self.receive_pre(batch.receive_spikes[:, 1].float())
        processed_receive = torch.stack([left_processed, right_processed], dim=1)
        processed_pathway, _ = build_pathway_features(
            transmit_processed,
            processed_receive,
            self.distance_candidates,
            self.itd_candidates,
            num_delay_lines=self.num_delay_lines,
            num_frequency_channels=self.num_frequency_channels,
        )

        distance_residual = self.distance_residual(
            F.layer_norm(processed_pathway.distance, (processed_pathway.distance.shape[-1],))
        )
        azimuth_residual = self.azimuth_residual(
            F.layer_norm(processed_pathway.azimuth, (processed_pathway.azimuth.shape[-1],))
        )
        elevation_residual = self.elevation_residual(
            F.layer_norm(processed_pathway.elevation, (processed_pathway.elevation.shape[-1],))
        )

        distance_scale = 0.30 * torch.sigmoid(self.distance_gain)
        azimuth_scale = 0.30 * torch.sigmoid(self.azimuth_gain)
        elevation_scale = 0.30 * torch.sigmoid(self.elevation_gain)
        diagnostics = {
            **base.diagnostics,
            "pre_pathway_distance_scale": distance_scale.detach(),
            "pre_pathway_azimuth_scale": azimuth_scale.detach(),
            "pre_pathway_elevation_scale": elevation_scale.detach(),
            "pre_pathway_left_spikes": left_processed[:1].detach(),
            "pre_pathway_right_spikes": right_processed[:1].detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * distance_residual),
            azimuth_latent=F.relu(base.azimuth_latent + azimuth_scale * azimuth_residual),
            elevation_latent=F.relu(base.elevation_latent + elevation_scale * elevation_residual),
            spectral_source=processed_receive,
            spike_proxy=processed_pathway.spike_count.float(),
            diagnostics=diagnostics,
        )


class PostPathwayResidualEncoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        beta: float,
        threshold: float,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )
        self.distance_post = LatentResidualLIF(branch_hidden_dim, beta=beta, threshold=threshold, num_steps=num_steps)
        self.azimuth_post = LatentResidualLIF(branch_hidden_dim, beta=beta, threshold=threshold, num_steps=num_steps)
        self.elevation_post = LatentResidualLIF(branch_hidden_dim, beta=beta, threshold=threshold, num_steps=num_steps)

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        base = self.base_encoder(batch)
        distance_latent, distance_trace, distance_scale = self.distance_post(base.distance_latent)
        azimuth_latent, azimuth_trace, azimuth_scale = self.azimuth_post(base.azimuth_latent)
        elevation_latent, elevation_trace, elevation_scale = self.elevation_post(base.elevation_latent)
        diagnostics = {
            **base.diagnostics,
            "post_pathway_distance_spikes": distance_trace,
            "post_pathway_azimuth_spikes": azimuth_trace,
            "post_pathway_elevation_spikes": elevation_trace,
            "post_pathway_distance_scale": distance_scale,
            "post_pathway_azimuth_scale": azimuth_scale,
            "post_pathway_elevation_scale": elevation_scale,
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class PerPathwayResonanceEncoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        num_frequency_channels: int,
        num_resonators: int = 16,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )
        self.resonance = CorollaryResonanceBank(num_frequency_channels, num_resonators=num_resonators)
        self.distance_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.azimuth_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.elevation_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.distance_gain = nn.Parameter(torch.tensor(-1.1))
        self.azimuth_gain = nn.Parameter(torch.tensor(-1.1))
        self.elevation_gain = nn.Parameter(torch.tensor(-1.1))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        base = self.base_encoder(batch)
        resonant_latent, resonant_diagnostics = self.resonance(batch)
        distance_scale = 0.30 * torch.sigmoid(self.distance_gain)
        azimuth_scale = 0.30 * torch.sigmoid(self.azimuth_gain)
        elevation_scale = 0.30 * torch.sigmoid(self.elevation_gain)
        diagnostics = {
            **base.diagnostics,
            **resonant_diagnostics,
            "per_pathway_resonance_distance_scale": distance_scale.detach(),
            "per_pathway_resonance_azimuth_scale": azimuth_scale.detach(),
            "per_pathway_resonance_elevation_scale": elevation_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * self.distance_projection(resonant_latent)),
            azimuth_latent=F.relu(base.azimuth_latent + azimuth_scale * self.azimuth_projection(resonant_latent)),
            elevation_latent=F.relu(base.elevation_latent + elevation_scale * self.elevation_projection(resonant_latent)),
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class FusionResonanceAugmentedModel(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_steps: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
        *,
        num_frequency_channels: int,
        num_resonators: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )
        self.resonance = CorollaryResonanceBank(num_frequency_channels, num_resonators=num_resonators)
        self.resonant_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.num_steps = num_steps
        self.fusion = nn.Linear(branch_hidden_dim * 4, hidden_dim)
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

    def forward(self, batch: ExperimentBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoding = self.encoder(batch)
        resonant_latent, resonant_diagnostics = self.resonance(batch)
        fused = torch.cat(
            [
                encoding.distance_latent,
                encoding.azimuth_latent,
                encoding.elevation_latent,
                F.relu(self.resonant_projection(resonant_latent)),
            ],
            dim=-1,
        )
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
        output = self.readout(integration_spikes.mean(dim=1))
        diagnostics = {
            **encoding.diagnostics,
            **resonant_diagnostics,
            "fusion_spikes": fusion_spikes,
            "integration_spikes": integration_spikes,
            "fused_latent": fused,
            "spike_rate": 0.5 * integration_spikes.mean(dim=(1, 2)) + 0.5 * resonant_diagnostics["resonant_spikes"].mean(dim=(1, 2)),
            "spike_proxy": encoding.spike_proxy,
        }
        return output, diagnostics


class AllRound2Encoder(nn.Module):
    def __init__(
        self,
        distance_dim: int,
        azimuth_dim: int,
        elevation_dim: int,
        branch_hidden_dim: int,
        *,
        num_frequency_channels: int,
        num_delay_lines: int,
        distance_candidates: torch.Tensor,
        itd_candidates: torch.Tensor,
        beta: float,
        threshold: float,
        num_steps: int,
        num_resonators: int = 16,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.num_frequency_channels = num_frequency_channels
        self.num_delay_lines = num_delay_lines
        self.register_buffer("distance_candidates", distance_candidates)
        self.register_buffer("itd_candidates", itd_candidates)

        self.base_encoder = CombinedElevationEncoder(
            distance_dim=distance_dim,
            azimuth_dim=azimuth_dim,
            elevation_dim=elevation_dim,
            branch_hidden_dim=branch_hidden_dim,
        )

        distance_half = distance_dim // 2
        azimuth_half = azimuth_dim // 2
        self.distance_left_offsets = nn.Parameter(torch.zeros(distance_half))
        self.distance_right_offsets = nn.Parameter(torch.zeros(distance_half))
        self.distance_left_gain = nn.Parameter(torch.zeros(distance_half))
        self.distance_right_gain = nn.Parameter(torch.zeros(distance_half))
        self.azimuth_itd_offsets = nn.Parameter(torch.zeros(azimuth_half))
        self.azimuth_itd_gain = nn.Parameter(torch.zeros(azimuth_half))
        self.azimuth_ild_gain = nn.Parameter(torch.zeros(azimuth_half))
        self.elevation_norm_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_notch_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_slope_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_norm_gain = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_notch_gain = nn.Parameter(torch.zeros(num_frequency_channels))
        self.elevation_slope_gain = nn.Parameter(torch.zeros(num_frequency_channels))
        self.spectral_offsets = nn.Parameter(torch.zeros(num_frequency_channels))
        self.spectral_gain = nn.Parameter(torch.zeros(num_frequency_channels))

        self.adaptive_distance_residual = nn.Linear(distance_dim, branch_hidden_dim)
        self.adaptive_azimuth_residual = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.adaptive_elevation_residual = nn.Linear(elevation_dim, branch_hidden_dim)
        self.adaptive_spectral_projection = nn.Linear(num_frequency_channels * 2, branch_hidden_dim)
        self.adaptive_distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.adaptive_azimuth_gain = nn.Parameter(torch.tensor(-1.2))
        self.adaptive_elevation_gain = nn.Parameter(torch.tensor(-1.2))

        self.transmit_pre = SpikeSequencePreprocessor(num_frequency_channels, beta=beta, threshold=0.75)
        self.receive_pre = SpikeSequencePreprocessor(num_frequency_channels, beta=beta, threshold=0.75)
        self.pre_distance_residual = nn.Linear(distance_dim, branch_hidden_dim)
        self.pre_azimuth_residual = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.pre_elevation_residual = nn.Linear(elevation_dim, branch_hidden_dim)
        self.pre_distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.pre_azimuth_gain = nn.Parameter(torch.tensor(-1.2))
        self.pre_elevation_gain = nn.Parameter(torch.tensor(-1.2))

        self.resonance = CorollaryResonanceBank(num_frequency_channels, num_resonators=num_resonators)
        self.resonance_distance_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.resonance_azimuth_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.resonance_elevation_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.fusion_resonance_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.resonance_distance_gain = nn.Parameter(torch.tensor(-1.1))
        self.resonance_azimuth_gain = nn.Parameter(torch.tensor(-1.1))
        self.resonance_elevation_gain = nn.Parameter(torch.tensor(-1.1))

        self.distance_post = LatentResidualLIF(branch_hidden_dim, beta=beta, threshold=0.85 * threshold, num_steps=num_steps)
        self.azimuth_post = LatentResidualLIF(branch_hidden_dim, beta=beta, threshold=0.85 * threshold, num_steps=num_steps)
        self.elevation_post = LatentResidualLIF(branch_hidden_dim, beta=beta, threshold=0.85 * threshold, num_steps=num_steps)

    def _adaptive_residuals(self, batch: ExperimentBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        distance_half = batch.pathway.distance.shape[-1] // 2
        distance_left = _shift_feature_axis(batch.pathway.distance[:, :distance_half], self.distance_left_offsets, 0.75)
        distance_right = _shift_feature_axis(batch.pathway.distance[:, distance_half:], self.distance_right_offsets, 0.75)
        distance_left = distance_left * _gain_vector(self.distance_left_gain, 0.35).view(1, -1)
        distance_right = distance_right * _gain_vector(self.distance_right_gain, 0.35).view(1, -1)
        adapted_distance = torch.cat([distance_left, distance_right], dim=-1)

        azimuth_half = batch.pathway.azimuth.shape[-1] // 2
        itd = _shift_feature_axis(batch.pathway.azimuth[:, :azimuth_half], self.azimuth_itd_offsets, 0.75)
        itd = itd * _gain_vector(self.azimuth_itd_gain, 0.35).view(1, -1)
        ild = batch.pathway.azimuth[:, azimuth_half:] * _gain_vector(self.azimuth_ild_gain, 0.35).view(1, -1)
        adapted_azimuth = torch.cat([itd, ild], dim=-1)

        spectral_norm = _shift_feature_axis(
            batch.pathway.elevation[:, : self.num_frequency_channels],
            self.elevation_norm_offsets,
            1.5,
        ) * _gain_vector(self.elevation_norm_gain, 0.35).view(1, -1)
        spectral_notches = _shift_feature_axis(
            batch.pathway.elevation[:, self.num_frequency_channels : 2 * self.num_frequency_channels],
            self.elevation_notch_offsets,
            1.5,
        ) * _gain_vector(self.elevation_notch_gain, 0.35).view(1, -1)
        spectral_slope = _shift_feature_axis(
            batch.pathway.elevation[:, 2 * self.num_frequency_channels : 3 * self.num_frequency_channels],
            self.elevation_slope_offsets,
            1.5,
        ) * _gain_vector(self.elevation_slope_gain, 0.35).view(1, -1)
        adapted_elevation = torch.cat([spectral_norm, spectral_notches, spectral_slope], dim=-1)

        adapted_spectral = _shift_channel_axis(batch.receive_spikes.float(), self.spectral_offsets, 1.5)
        adapted_spectral = adapted_spectral * _gain_vector(self.spectral_gain, 0.35).view(1, 1, -1, 1)
        spectral_summary = adapted_spectral.sum(dim=-1).flatten(start_dim=1)

        distance_residual = self.adaptive_distance_residual(F.layer_norm(adapted_distance, (adapted_distance.shape[-1],)))
        azimuth_residual = self.adaptive_azimuth_residual(F.layer_norm(adapted_azimuth, (adapted_azimuth.shape[-1],)))
        elevation_residual = self.adaptive_elevation_residual(F.layer_norm(adapted_elevation, (adapted_elevation.shape[-1],)))
        elevation_residual = elevation_residual + 0.25 * self.adaptive_spectral_projection(
            F.layer_norm(spectral_summary, (spectral_summary.shape[-1],))
        )

        diagnostics = {
            "adaptive_distance_offsets_left": torch.tanh(self.distance_left_offsets).detach(),
            "adaptive_distance_offsets_right": torch.tanh(self.distance_right_offsets).detach(),
            "adaptive_itd_offsets": torch.tanh(self.azimuth_itd_offsets).detach(),
            "adaptive_spectral_offsets": torch.tanh(self.spectral_offsets).detach(),
            "adaptive_distance_gains_left": _gain_vector(self.distance_left_gain, 0.35).detach(),
            "adaptive_distance_gains_right": _gain_vector(self.distance_right_gain, 0.35).detach(),
            "adaptive_itd_gains": _gain_vector(self.azimuth_itd_gain, 0.35).detach(),
            "adaptive_ild_gains": _gain_vector(self.azimuth_ild_gain, 0.35).detach(),
            "adaptive_spectral_gains": _gain_vector(self.spectral_gain, 0.35).detach(),
        }
        return distance_residual, azimuth_residual, elevation_residual, adapted_spectral, diagnostics

    def _pre_pathway_residuals(self, batch: ExperimentBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        transmit_processed = self.transmit_pre(batch.transmit_spikes.float())
        left_processed = self.receive_pre(batch.receive_spikes[:, 0].float())
        right_processed = self.receive_pre(batch.receive_spikes[:, 1].float())
        processed_receive = torch.stack([left_processed, right_processed], dim=1)
        processed_pathway, _ = build_pathway_features(
            transmit_processed,
            processed_receive,
            self.distance_candidates,
            self.itd_candidates,
            num_delay_lines=self.num_delay_lines,
            num_frequency_channels=self.num_frequency_channels,
        )

        distance_residual = self.pre_distance_residual(
            F.layer_norm(processed_pathway.distance, (processed_pathway.distance.shape[-1],))
        )
        azimuth_residual = self.pre_azimuth_residual(
            F.layer_norm(processed_pathway.azimuth, (processed_pathway.azimuth.shape[-1],))
        )
        elevation_residual = self.pre_elevation_residual(
            F.layer_norm(processed_pathway.elevation, (processed_pathway.elevation.shape[-1],))
        )
        diagnostics = {
            "pre_pathway_left_spikes": left_processed[:1].detach(),
            "pre_pathway_right_spikes": right_processed[:1].detach(),
        }
        return distance_residual, azimuth_residual, elevation_residual, processed_pathway.spike_count.float(), diagnostics

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.pathway is None or batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("AllRound2Encoder requires pathway and spike inputs.")
        base = self.base_encoder(batch)

        adaptive_distance, adaptive_azimuth, adaptive_elevation, adapted_spectral, adaptive_diagnostics = self._adaptive_residuals(batch)
        pre_distance, pre_azimuth, pre_elevation, pre_spike_proxy, pre_diagnostics = self._pre_pathway_residuals(batch)
        resonance_pooled, resonance_diagnostics = self.resonance(batch)

        adaptive_distance_scale = 0.28 * torch.sigmoid(self.adaptive_distance_gain)
        adaptive_azimuth_scale = 0.28 * torch.sigmoid(self.adaptive_azimuth_gain)
        adaptive_elevation_scale = 0.28 * torch.sigmoid(self.adaptive_elevation_gain)
        pre_distance_scale = 0.22 * torch.sigmoid(self.pre_distance_gain)
        pre_azimuth_scale = 0.22 * torch.sigmoid(self.pre_azimuth_gain)
        pre_elevation_scale = 0.22 * torch.sigmoid(self.pre_elevation_gain)
        resonance_distance_scale = 0.22 * torch.sigmoid(self.resonance_distance_gain)
        resonance_azimuth_scale = 0.22 * torch.sigmoid(self.resonance_azimuth_gain)
        resonance_elevation_scale = 0.22 * torch.sigmoid(self.resonance_elevation_gain)

        distance_latent = F.relu(
            base.distance_latent
            + adaptive_distance_scale * adaptive_distance
            + pre_distance_scale * pre_distance
            + resonance_distance_scale * self.resonance_distance_projection(resonance_pooled)
        )
        azimuth_latent = F.relu(
            base.azimuth_latent
            + adaptive_azimuth_scale * adaptive_azimuth
            + pre_azimuth_scale * pre_azimuth
            + resonance_azimuth_scale * self.resonance_azimuth_projection(resonance_pooled)
        )
        elevation_latent = F.relu(
            base.elevation_latent
            + adaptive_elevation_scale * adaptive_elevation
            + pre_elevation_scale * pre_elevation
            + resonance_elevation_scale * self.resonance_elevation_projection(resonance_pooled)
        )

        distance_latent, distance_post_spikes, distance_post_scale = self.distance_post(distance_latent)
        azimuth_latent, azimuth_post_spikes, azimuth_post_scale = self.azimuth_post(azimuth_latent)
        elevation_latent, elevation_post_spikes, elevation_post_scale = self.elevation_post(elevation_latent)

        diagnostics = {
            **base.diagnostics,
            **adaptive_diagnostics,
            **pre_diagnostics,
            **resonance_diagnostics,
            "adaptive_distance_scale": adaptive_distance_scale.detach(),
            "adaptive_azimuth_scale": adaptive_azimuth_scale.detach(),
            "adaptive_elevation_scale": adaptive_elevation_scale.detach(),
            "pre_pathway_distance_scale": pre_distance_scale.detach(),
            "pre_pathway_azimuth_scale": pre_azimuth_scale.detach(),
            "pre_pathway_elevation_scale": pre_elevation_scale.detach(),
            "per_pathway_resonance_distance_scale": resonance_distance_scale.detach(),
            "per_pathway_resonance_azimuth_scale": resonance_azimuth_scale.detach(),
            "per_pathway_resonance_elevation_scale": resonance_elevation_scale.detach(),
            "post_pathway_distance_spikes": distance_post_spikes,
            "post_pathway_azimuth_spikes": azimuth_post_spikes,
            "post_pathway_elevation_spikes": elevation_post_spikes,
            "post_pathway_distance_scale": distance_post_scale,
            "post_pathway_azimuth_scale": azimuth_post_scale,
            "post_pathway_elevation_scale": elevation_post_scale,
            "fusion_resonance_latent": resonance_pooled,
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=adapted_spectral,
            spike_proxy=0.5 * (base.spike_proxy + pre_spike_proxy),
            diagnostics=diagnostics,
        )


class AllRound2CombinedModel(nn.Module):
    def __init__(
        self,
        encoder: AllRound2Encoder,
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
        branch_hidden_dim = getattr(encoder, "branch_hidden_dim")
        resonance_dim = encoder.fusion_resonance_projection.in_features
        self.fusion_resonance_projection = nn.Linear(resonance_dim, branch_hidden_dim)
        self.fusion = nn.Linear(branch_hidden_dim * 4, hidden_dim)
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

    def forward(self, batch: ExperimentBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoding = self.encoder(batch)
        fusion_resonance = F.relu(self.fusion_resonance_projection(encoding.diagnostics["fusion_resonance_latent"]))
        fused = torch.cat(
            [encoding.distance_latent, encoding.azimuth_latent, encoding.elevation_latent, fusion_resonance],
            dim=-1,
        )

        self.fusion_lif.reset_mem()
        self.integration_lif.reset_mem()
        fusion_current = self.fusion(fused)
        fusion_mem = None
        integration_mem = None
        fusion_trace = []
        integration_trace = []
        for _ in range(self.num_steps):
            fusion_spikes, fusion_mem = self.fusion_lif(fusion_current, fusion_mem)
            integration_current = self.integration(fusion_spikes)
            integration_spikes, integration_mem = self.integration_lif(integration_current, integration_mem)
            fusion_trace.append(fusion_spikes)
            integration_trace.append(integration_spikes)

        fusion_spikes = torch.stack(fusion_trace, dim=1)
        integration_spikes = torch.stack(integration_trace, dim=1)
        output = self.readout(integration_spikes.mean(dim=1))
        resonance_spikes = encoding.diagnostics["resonant_spikes"]
        diagnostics = {
            **encoding.diagnostics,
            "fused_latent": fused,
            "fusion_resonance_projected": fusion_resonance,
            "fusion_spikes": fusion_spikes,
            "integration_spikes": integration_spikes,
            "spike_rate": (
                integration_spikes.mean(dim=(1, 2))
                + resonance_spikes.mean(dim=(1, 2))
                + encoding.diagnostics["post_pathway_distance_spikes"].mean(dim=(1, 2))
                + encoding.diagnostics["post_pathway_azimuth_spikes"].mean(dim=(1, 2))
                + encoding.diagnostics["post_pathway_elevation_spikes"].mean(dim=(1, 2))
            )
            / 5.0,
            "spike_proxy": encoding.spike_proxy,
        }
        return output, diagnostics
