from __future__ import annotations

import math

import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_variants import BranchEncoding, ExperimentBatch
from models.pathway_snn import build_pathway_features, resize_feature
from models.round3_variants import CoincidenceLIFBank
from models.snn import collapse_spikes, onset_pathway, surrogate_spike


def _identity_init_conv1d(conv: nn.Conv1d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        diagonal = min(conv.out_channels, conv.in_channels)
        center = conv.kernel_size[0] // 2
        for index in range(diagonal):
            conv.weight[index, index, center] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _identity_init_conv2d(conv: nn.Conv2d) -> None:
    with torch.no_grad():
        conv.weight.zero_()
        diagonal = min(conv.out_channels, conv.in_channels)
        center_y = conv.kernel_size[0] // 2
        center_x = conv.kernel_size[1] // 2
        for index in range(diagonal):
            conv.weight[index, index, center_y, center_x] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _normalize_traces(traces: torch.Tensor) -> torch.Tensor:
    scale = traces.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return traces / scale


def _split_elevation_bands(receive_spikes: torch.Tensor) -> torch.Tensor:
    channels = receive_spikes.shape[2]
    lower_stop = max(1, channels // 3)
    middle_stop = max(lower_stop + 1, (2 * channels) // 3)
    common = receive_spikes.float().sum(dim=1)
    lower = common[:, :lower_stop].sum(dim=1)
    middle = common[:, lower_stop:middle_stop].sum(dim=1)
    upper = common[:, middle_stop:].sum(dim=1)
    return torch.stack([lower, middle, upper], dim=1)


class FullReplacementLIFTimingEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: nn.Module,
        branch_hidden_dim: int,
        distance_candidates: torch.Tensor,
        itd_candidates: torch.Tensor,
        num_delay_lines: int,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_delay_lines = num_delay_lines
        self.register_buffer("distance_candidates", distance_candidates)
        self.register_buffer("itd_candidates", itd_candidates)

        self.distance_left_bank = CoincidenceLIFBank(distance_candidates.numel())
        self.distance_right_bank = CoincidenceLIFBank(distance_candidates.numel())
        self.itd_bank = CoincidenceLIFBank(itd_candidates.numel())

        self.distance_projection = nn.Linear(distance_candidates.numel() * 2, branch_hidden_dim)
        self.azimuth_projection = nn.Linear(itd_candidates.numel() + num_delay_lines, branch_hidden_dim)

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.transmit_spikes is None or batch.receive_spikes is None or batch.pathway is None:
            raise ValueError("FullReplacementLIFTimingEncoder requires pathway and spike inputs.")
        base = self.base_encoder(batch)

        transmit_onsets = collapse_spikes(onset_pathway(batch.transmit_spikes.float()))
        left_onsets = collapse_spikes(onset_pathway(batch.receive_spikes[:, 0].float()))
        right_onsets = collapse_spikes(onset_pathway(batch.receive_spikes[:, 1].float()))

        distance_left, distance_left_spikes, left_diag = self.distance_left_bank(
            transmit_onsets, left_onsets, self.distance_candidates
        )
        distance_right, distance_right_spikes, right_diag = self.distance_right_bank(
            transmit_onsets, right_onsets, self.distance_candidates
        )
        itd_scores, itd_spikes, itd_diag = self.itd_bank(left_onsets, right_onsets, self.itd_candidates)

        ild_features = batch.pathway.azimuth[:, self.num_delay_lines :]
        distance_latent = F.relu(self.distance_projection(torch.cat([distance_left, distance_right], dim=-1)))
        azimuth_latent = F.relu(self.azimuth_projection(torch.cat([itd_scores, ild_features], dim=-1)))

        diagnostics = {
            **base.diagnostics,
            "lif_distance_left_spikes": distance_left_spikes,
            "lif_distance_right_spikes": distance_right_spikes,
            "lif_itd_spikes": itd_spikes,
            "lif_distance_left_beta": left_diag["beta"],
            "lif_distance_right_beta": right_diag["beta"],
            "lif_itd_beta": itd_diag["beta"],
            "lif_distance_left_reference_weight": left_diag["reference_weight"],
            "lif_distance_left_target_weight": left_diag["target_weight"],
            "lif_distance_right_reference_weight": right_diag["reference_weight"],
            "lif_distance_right_target_weight": right_diag["target_weight"],
            "lif_itd_reference_weight": itd_diag["reference_weight"],
            "lif_itd_target_weight": itd_diag["target_weight"],
            "round4_timing_replacement": torch.tensor(1.0, device=distance_latent.device),
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=base.elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class SharedBackbonePrePathwayEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: nn.Module,
        num_frequency_channels: int,
        num_delay_lines: int,
        distance_candidates: torch.Tensor,
        itd_candidates: torch.Tensor,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = getattr(base_encoder, "branch_hidden_dim")
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_frequency_channels = num_frequency_channels
        self.num_delay_lines = num_delay_lines
        self.register_buffer("distance_candidates", distance_candidates)
        self.register_buffer("itd_candidates", itd_candidates)

        self.transmit_backbone = nn.Conv1d(num_frequency_channels, num_frequency_channels, kernel_size=5, padding=2)
        self.receive_backbone = nn.Conv2d(2, 2, kernel_size=(5, 7), padding=(2, 3))
        _identity_init_conv1d(self.transmit_backbone)
        _identity_init_conv2d(self.receive_backbone)

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("SharedBackbonePrePathwayEncoder requires spike inputs.")

        transmit_processed = F.relu(self.transmit_backbone(batch.transmit_spikes.float()))
        receive_processed = F.relu(self.receive_backbone(batch.receive_spikes.float()))
        pathway, _ = build_pathway_features(
            transmit_processed,
            receive_processed,
            self.distance_candidates,
            self.itd_candidates,
            num_delay_lines=self.num_delay_lines,
            num_frequency_channels=self.num_frequency_channels,
        )
        spike_proxy = receive_processed.sum(dim=(-1, -2, -3)).float() + transmit_processed.sum(dim=(-1, -2)).float()
        processed_batch = ExperimentBatch(
            transmit_wave=batch.transmit_wave,
            receive_wave=batch.receive_wave,
            pathway=pathway,
            transmit_spikes=transmit_processed,
            receive_spikes=receive_processed,
            spike_count=spike_proxy,
        )
        base = self.base_encoder(processed_batch)
        diagnostics = {
            **base.diagnostics,
            "shared_backbone_transmit_kernel": self.transmit_backbone.weight.detach(),
            "shared_backbone_receive_kernel": self.receive_backbone.weight.detach(),
            "shared_backbone_receive_preview": receive_processed[:1].detach(),
        }
        return BranchEncoding(
            distance_latent=base.distance_latent,
            azimuth_latent=base.azimuth_latent,
            elevation_latent=base.elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=spike_proxy,
            diagnostics=diagnostics,
        )


class PostPathwayICConvEncoder(nn.Module):
    def __init__(self, *, base_encoder: nn.Module, branch_hidden_dim: int, ic_channels: int = 6) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.ic_conv1 = nn.Conv2d(3, ic_channels, kernel_size=(3, 5), padding=(1, 2))
        self.ic_conv2 = nn.Conv2d(ic_channels, ic_channels, kernel_size=(3, 3), padding=1)
        self.ic_projection = nn.Linear(ic_channels * 2 * 4, branch_hidden_dim * 3)
        self.distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.azimuth_gain = nn.Parameter(torch.tensor(-1.2))
        self.elevation_gain = nn.Parameter(torch.tensor(-1.2))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        base = self.base_encoder(batch)
        distance_trace = base.diagnostics["post_pathway_distance_spikes"]
        azimuth_trace = base.diagnostics["post_pathway_azimuth_spikes"]
        elevation_trace = base.diagnostics["post_pathway_elevation_spikes"]
        ic_input = torch.stack([distance_trace, azimuth_trace, elevation_trace], dim=1)
        ic_map = F.relu(self.ic_conv1(ic_input))
        ic_map = F.relu(self.ic_conv2(ic_map))
        ic_pooled = F.adaptive_avg_pool2d(ic_map, (2, 4)).flatten(start_dim=1)
        ic_residual = self.ic_projection(ic_pooled).view(ic_input.shape[0], 3, self.branch_hidden_dim)

        distance_scale = 0.25 * torch.sigmoid(self.distance_gain)
        azimuth_scale = 0.25 * torch.sigmoid(self.azimuth_gain)
        elevation_scale = 0.25 * torch.sigmoid(self.elevation_gain)

        diagnostics = {
            **base.diagnostics,
            "ic_conv_map": ic_map.detach(),
            "ic_distance_scale": distance_scale.detach(),
            "ic_azimuth_scale": azimuth_scale.detach(),
            "ic_elevation_scale": elevation_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * ic_residual[:, 0]),
            azimuth_latent=F.relu(base.azimuth_latent + azimuth_scale * ic_residual[:, 1]),
            elevation_latent=F.relu(base.elevation_latent + elevation_scale * ic_residual[:, 2]),
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class BioILDLSOEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: nn.Module,
        branch_hidden_dim: int,
        num_frequency_channels: int,
        num_delay_lines: int,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_frequency_channels = num_frequency_channels
        self.num_delay_lines = num_delay_lines

        self.left_exc_raw = nn.Parameter(torch.zeros(num_frequency_channels))
        self.left_inh_raw = nn.Parameter(torch.zeros(num_frequency_channels))
        self.right_exc_raw = nn.Parameter(torch.zeros(num_frequency_channels))
        self.right_inh_raw = nn.Parameter(torch.zeros(num_frequency_channels))

        self.azimuth_projection = nn.Linear(num_delay_lines * 2, branch_hidden_dim)

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.receive_spikes is None or batch.pathway is None:
            raise ValueError("BioILDLSOEncoder requires pathway and spike inputs.")
        base = self.base_encoder(batch)

        left_counts = batch.receive_spikes[:, 0].float().sum(dim=-1)
        right_counts = batch.receive_spikes[:, 1].float().sum(dim=-1)

        left_exc = F.softplus(self.left_exc_raw).view(1, -1) * left_counts
        right_exc = F.softplus(self.right_exc_raw).view(1, -1) * right_counts
        mntb_left = left_exc
        mntb_right = right_exc
        left_lso = F.relu(left_exc - F.softplus(self.left_inh_raw).view(1, -1) * mntb_right)
        right_lso = F.relu(right_exc - F.softplus(self.right_inh_raw).view(1, -1) * mntb_left)

        lso_compare = right_lso - left_lso
        lso_total = left_lso + right_lso
        normalized_compare = lso_compare / lso_total.clamp_min(1.0)
        bio_ild = torch.cat([left_lso, right_lso, lso_compare, normalized_compare], dim=-1)
        bio_ild_resized = resize_feature(bio_ild, self.num_delay_lines)
        fixed_itd = batch.pathway.azimuth[:, : self.num_delay_lines]
        azimuth_features = torch.cat([fixed_itd, bio_ild_resized], dim=-1)
        azimuth_latent = F.relu(self.azimuth_projection(F.layer_norm(azimuth_features, (azimuth_features.shape[-1],))))

        diagnostics = {
            **base.diagnostics,
            "bio_lso_left": left_lso.detach(),
            "bio_lso_right": right_lso.detach(),
            "bio_lso_compare": lso_compare.detach(),
            "bio_lso_normalized_compare": normalized_compare.detach(),
        }
        return BranchEncoding(
            distance_latent=base.distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=base.elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class DistanceSpikeSumEncoder(nn.Module):
    def __init__(self, *, base_encoder: nn.Module, branch_hidden_dim: int) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.distance_projection = nn.Linear(5, branch_hidden_dim)
        self.distance_gain = nn.Parameter(torch.tensor(-1.1))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.receive_spikes is None:
            raise ValueError("DistanceSpikeSumEncoder requires spike inputs.")
        base = self.base_encoder(batch)

        left_total = batch.receive_spikes[:, 0].float().sum(dim=(1, 2))
        right_total = batch.receive_spikes[:, 1].float().sum(dim=(1, 2))
        total = left_total + right_total
        features = torch.stack(
            [
                left_total,
                right_total,
                total,
                left_total - right_total,
                torch.log1p(total),
            ],
            dim=-1,
        )
        residual = self.distance_projection(F.layer_norm(features, (features.shape[-1],)))
        distance_scale = 0.25 * torch.sigmoid(self.distance_gain)

        diagnostics = {
            **base.diagnostics,
            "distance_spike_sum_features": features.detach(),
            "distance_spike_sum_scale": distance_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * residual),
            azimuth_latent=base.azimuth_latent,
            elevation_latent=base.elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class QResonanceBank(nn.Module):
    def __init__(
        self,
        input_channels: int,
        *,
        num_resonators: int,
        q_init: float,
        time_bins: int = 128,
    ) -> None:
        super().__init__()
        self.time_bins = time_bins
        self.drive_projection = nn.Conv1d(input_channels, num_resonators, kernel_size=1, bias=True)
        self.frequency_raw = nn.Parameter(torch.linspace(-1.2, 1.2, num_resonators))
        q_center = math.log(math.exp(max(q_init - 1.5, 1e-3)) - 1.0)
        self.q_raw = nn.Parameter(torch.full((num_resonators,), float(q_center)))
        self.threshold_raw = nn.Parameter(torch.zeros(num_resonators))

    def forward(self, traces: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled_traces = F.adaptive_avg_pool1d(traces, self.time_bins)
        pooled_traces = _normalize_traces(pooled_traces)
        drive = self.drive_projection(pooled_traces)

        state = torch.zeros(drive.shape[0], drive.shape[1], device=drive.device, dtype=drive.dtype)
        velocity = torch.zeros_like(state)
        spike_trace = []

        frequency = 0.03 + 0.17 * torch.sigmoid(self.frequency_raw).view(1, -1)
        q_factor = 1.5 + F.softplus(self.q_raw).view(1, -1)
        decay = torch.exp(-1.0 / q_factor)
        threshold = 0.35 + 0.75 * torch.sigmoid(self.threshold_raw).view(1, -1)

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
            "frequency": frequency.detach().squeeze(0),
            "q_factor": q_factor.detach().squeeze(0),
            "decay": decay.detach().squeeze(0),
            "threshold": threshold.detach().squeeze(0),
            "spikes": spike_tensor.detach(),
        }
        return pooled, diagnostics


class PathwayQResonanceEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: nn.Module,
        branch_hidden_dim: int,
        num_frequency_channels: int,
        num_resonators: int = 12,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_frequency_channels = num_frequency_channels

        self.distance_bank = QResonanceBank(2, num_resonators=num_resonators, q_init=6.0)
        self.azimuth_bank = QResonanceBank(3, num_resonators=num_resonators, q_init=2.5)
        self.elevation_bank = QResonanceBank(3, num_resonators=num_resonators, q_init=12.0)

        self.distance_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.azimuth_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.elevation_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.distance_gain = nn.Parameter(torch.tensor(-1.1))
        self.azimuth_gain = nn.Parameter(torch.tensor(-1.1))
        self.elevation_gain = nn.Parameter(torch.tensor(-1.1))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("PathwayQResonanceEncoder requires spike inputs.")
        base = self.base_encoder(batch)

        transmit = batch.transmit_spikes.float().sum(dim=1)
        left = batch.receive_spikes[:, 0].float().sum(dim=1)
        right = batch.receive_spikes[:, 1].float().sum(dim=1)
        common = 0.5 * (left + right)

        distance_traces = torch.stack([transmit, common], dim=1)
        azimuth_traces = torch.stack([left, right, right - left], dim=1)
        elevation_traces = _split_elevation_bands(batch.receive_spikes)

        distance_pooled, distance_diag = self.distance_bank(distance_traces)
        azimuth_pooled, azimuth_diag = self.azimuth_bank(azimuth_traces)
        elevation_pooled, elevation_diag = self.elevation_bank(elevation_traces)

        distance_scale = 0.22 * torch.sigmoid(self.distance_gain)
        azimuth_scale = 0.22 * torch.sigmoid(self.azimuth_gain)
        elevation_scale = 0.22 * torch.sigmoid(self.elevation_gain)

        diagnostics = {
            **base.diagnostics,
            "pathway_q_distance_frequency": distance_diag["frequency"],
            "pathway_q_distance_q": distance_diag["q_factor"],
            "pathway_q_distance_spikes": distance_diag["spikes"],
            "pathway_q_azimuth_frequency": azimuth_diag["frequency"],
            "pathway_q_azimuth_q": azimuth_diag["q_factor"],
            "pathway_q_azimuth_spikes": azimuth_diag["spikes"],
            "pathway_q_elevation_frequency": elevation_diag["frequency"],
            "pathway_q_elevation_q": elevation_diag["q_factor"],
            "pathway_q_elevation_spikes": elevation_diag["spikes"],
            "pathway_q_distance_scale": distance_scale.detach(),
            "pathway_q_azimuth_scale": azimuth_scale.detach(),
            "pathway_q_elevation_scale": elevation_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * self.distance_projection(distance_pooled)),
            azimuth_latent=F.relu(base.azimuth_latent + azimuth_scale * self.azimuth_projection(azimuth_pooled)),
            elevation_latent=F.relu(base.elevation_latent + elevation_scale * self.elevation_projection(elevation_pooled)),
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class CombinedAcceptedRound4Encoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: nn.Module,
        branch_hidden_dim: int,
        num_frequency_channels: int,
        num_delay_lines: int,
        distance_candidates: torch.Tensor,
        itd_candidates: torch.Tensor,
        num_resonators: int = 12,
        ic_channels: int = 6,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_frequency_channels = num_frequency_channels
        self.num_delay_lines = num_delay_lines
        self.register_buffer("distance_candidates", distance_candidates)
        self.register_buffer("itd_candidates", itd_candidates)

        self.distance_left_bank = CoincidenceLIFBank(distance_candidates.numel())
        self.distance_right_bank = CoincidenceLIFBank(distance_candidates.numel())
        self.itd_bank = CoincidenceLIFBank(itd_candidates.numel())
        self.distance_projection = nn.Linear(distance_candidates.numel() * 2, branch_hidden_dim)
        self.azimuth_projection = nn.Linear(itd_candidates.numel() + num_delay_lines, branch_hidden_dim)

        self.left_exc_raw = nn.Parameter(torch.zeros(num_frequency_channels))
        self.left_inh_raw = nn.Parameter(torch.zeros(num_frequency_channels))
        self.right_exc_raw = nn.Parameter(torch.zeros(num_frequency_channels))
        self.right_inh_raw = nn.Parameter(torch.zeros(num_frequency_channels))

        self.distance_spike_projection = nn.Linear(5, branch_hidden_dim)
        self.distance_spike_gain = nn.Parameter(torch.tensor(-1.1))

        self.distance_bank = QResonanceBank(2, num_resonators=num_resonators, q_init=6.0)
        self.azimuth_bank = QResonanceBank(3, num_resonators=num_resonators, q_init=2.5)
        self.elevation_bank = QResonanceBank(3, num_resonators=num_resonators, q_init=12.0)
        self.distance_q_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.azimuth_q_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.elevation_q_projection = nn.Linear(num_resonators, branch_hidden_dim)
        self.distance_q_gain = nn.Parameter(torch.tensor(-1.1))
        self.azimuth_q_gain = nn.Parameter(torch.tensor(-1.1))
        self.elevation_q_gain = nn.Parameter(torch.tensor(-1.1))

        self.ic_conv1 = nn.Conv2d(3, ic_channels, kernel_size=(3, 5), padding=(1, 2))
        self.ic_conv2 = nn.Conv2d(ic_channels, ic_channels, kernel_size=(3, 3), padding=1)
        self.ic_projection = nn.Linear(ic_channels * 2 * 4, branch_hidden_dim * 3)
        self.ic_distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.ic_azimuth_gain = nn.Parameter(torch.tensor(-1.2))
        self.ic_elevation_gain = nn.Parameter(torch.tensor(-1.2))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("CombinedAcceptedRound4Encoder requires spike inputs.")
        base = self.base_encoder(batch)

        transmit_onsets = collapse_spikes(onset_pathway(batch.transmit_spikes.float()))
        left_onsets = collapse_spikes(onset_pathway(batch.receive_spikes[:, 0].float()))
        right_onsets = collapse_spikes(onset_pathway(batch.receive_spikes[:, 1].float()))

        distance_left, distance_left_spikes, left_diag = self.distance_left_bank(
            transmit_onsets, left_onsets, self.distance_candidates
        )
        distance_right, distance_right_spikes, right_diag = self.distance_right_bank(
            transmit_onsets, right_onsets, self.distance_candidates
        )
        itd_scores, itd_spikes, itd_diag = self.itd_bank(left_onsets, right_onsets, self.itd_candidates)

        left_counts = batch.receive_spikes[:, 0].float().sum(dim=-1)
        right_counts = batch.receive_spikes[:, 1].float().sum(dim=-1)
        left_exc = F.softplus(self.left_exc_raw).view(1, -1) * left_counts
        right_exc = F.softplus(self.right_exc_raw).view(1, -1) * right_counts
        left_lso = F.relu(left_exc - F.softplus(self.left_inh_raw).view(1, -1) * right_exc)
        right_lso = F.relu(right_exc - F.softplus(self.right_inh_raw).view(1, -1) * left_exc)
        lso_compare = right_lso - left_lso
        lso_total = left_lso + right_lso
        normalized_compare = lso_compare / lso_total.clamp_min(1.0)
        bio_ild = torch.cat([left_lso, right_lso, lso_compare, normalized_compare], dim=-1)
        bio_ild_resized = resize_feature(bio_ild, self.num_delay_lines)

        distance_latent = F.relu(self.distance_projection(torch.cat([distance_left, distance_right], dim=-1)))
        azimuth_latent = F.relu(self.azimuth_projection(torch.cat([itd_scores, bio_ild_resized], dim=-1)))
        elevation_latent = base.elevation_latent

        left_total = batch.receive_spikes[:, 0].float().sum(dim=(1, 2))
        right_total = batch.receive_spikes[:, 1].float().sum(dim=(1, 2))
        total = left_total + right_total
        spike_sum_features = torch.stack(
            [
                left_total,
                right_total,
                total,
                left_total - right_total,
                torch.log1p(total),
            ],
            dim=-1,
        )
        distance_spike_scale = 0.25 * torch.sigmoid(self.distance_spike_gain)
        distance_latent = F.relu(
            distance_latent
            + distance_spike_scale * self.distance_spike_projection(F.layer_norm(spike_sum_features, (spike_sum_features.shape[-1],)))
        )

        transmit = batch.transmit_spikes.float().sum(dim=1)
        left = batch.receive_spikes[:, 0].float().sum(dim=1)
        right = batch.receive_spikes[:, 1].float().sum(dim=1)
        common = 0.5 * (left + right)
        distance_traces = torch.stack([transmit, common], dim=1)
        azimuth_traces = torch.stack([left, right, right - left], dim=1)
        elevation_traces = _split_elevation_bands(batch.receive_spikes)

        distance_pooled, distance_q_diag = self.distance_bank(distance_traces)
        azimuth_pooled, azimuth_q_diag = self.azimuth_bank(azimuth_traces)
        elevation_pooled, elevation_q_diag = self.elevation_bank(elevation_traces)
        distance_q_scale = 0.22 * torch.sigmoid(self.distance_q_gain)
        azimuth_q_scale = 0.22 * torch.sigmoid(self.azimuth_q_gain)
        elevation_q_scale = 0.22 * torch.sigmoid(self.elevation_q_gain)

        distance_latent = F.relu(distance_latent + distance_q_scale * self.distance_q_projection(distance_pooled))
        azimuth_latent = F.relu(azimuth_latent + azimuth_q_scale * self.azimuth_q_projection(azimuth_pooled))
        elevation_latent = F.relu(elevation_latent + elevation_q_scale * self.elevation_q_projection(elevation_pooled))

        distance_trace = base.diagnostics["post_pathway_distance_spikes"]
        azimuth_trace = base.diagnostics["post_pathway_azimuth_spikes"]
        elevation_trace = base.diagnostics["post_pathway_elevation_spikes"]
        ic_input = torch.stack([distance_trace, azimuth_trace, elevation_trace], dim=1)
        ic_map = F.relu(self.ic_conv1(ic_input))
        ic_map = F.relu(self.ic_conv2(ic_map))
        ic_pooled = F.adaptive_avg_pool2d(ic_map, (2, 4)).flatten(start_dim=1)
        ic_residual = self.ic_projection(ic_pooled).view(ic_input.shape[0], 3, self.branch_hidden_dim)

        ic_distance_scale = 0.25 * torch.sigmoid(self.ic_distance_gain)
        ic_azimuth_scale = 0.25 * torch.sigmoid(self.ic_azimuth_gain)
        ic_elevation_scale = 0.25 * torch.sigmoid(self.ic_elevation_gain)
        distance_latent = F.relu(distance_latent + ic_distance_scale * ic_residual[:, 0])
        azimuth_latent = F.relu(azimuth_latent + ic_azimuth_scale * ic_residual[:, 1])
        elevation_latent = F.relu(elevation_latent + ic_elevation_scale * ic_residual[:, 2])

        diagnostics = {
            **base.diagnostics,
            "lif_distance_left_spikes": distance_left_spikes,
            "lif_distance_right_spikes": distance_right_spikes,
            "lif_itd_spikes": itd_spikes,
            "lif_distance_left_beta": left_diag["beta"],
            "lif_distance_right_beta": right_diag["beta"],
            "lif_itd_beta": itd_diag["beta"],
            "bio_lso_left": left_lso.detach(),
            "bio_lso_right": right_lso.detach(),
            "bio_lso_compare": lso_compare.detach(),
            "bio_lso_normalized_compare": normalized_compare.detach(),
            "distance_spike_sum_features": spike_sum_features.detach(),
            "distance_spike_sum_scale": distance_spike_scale.detach(),
            "pathway_q_distance_frequency": distance_q_diag["frequency"],
            "pathway_q_distance_q": distance_q_diag["q_factor"],
            "pathway_q_distance_spikes": distance_q_diag["spikes"],
            "pathway_q_azimuth_frequency": azimuth_q_diag["frequency"],
            "pathway_q_azimuth_q": azimuth_q_diag["q_factor"],
            "pathway_q_azimuth_spikes": azimuth_q_diag["spikes"],
            "pathway_q_elevation_frequency": elevation_q_diag["frequency"],
            "pathway_q_elevation_q": elevation_q_diag["q_factor"],
            "pathway_q_elevation_spikes": elevation_q_diag["spikes"],
            "ic_conv_map": ic_map.detach(),
            "ic_distance_scale": ic_distance_scale.detach(),
            "ic_azimuth_scale": ic_azimuth_scale.detach(),
            "ic_elevation_scale": ic_elevation_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=distance_latent,
            azimuth_latent=azimuth_latent,
            elevation_latent=elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )
