from __future__ import annotations

import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_variants import BranchEncoding, ExperimentBatch
from models.round2_variants import AllRound2Encoder
from models.snn import collapse_spikes, onset_pathway


def _shift_time_bank(sequence: torch.Tensor, candidate_delays: torch.Tensor) -> torch.Tensor:
    total_steps = sequence.shape[-1]
    base_indices = torch.arange(total_steps, device=sequence.device).view(1, 1, total_steps)
    shifted_indices = base_indices + candidate_delays.view(1, -1, 1)
    valid = (shifted_indices >= 0) & (shifted_indices < total_steps)
    clamped = shifted_indices.clamp(0, total_steps - 1).expand(sequence.shape[0], -1, -1)
    shifted = sequence.unsqueeze(1).expand(-1, candidate_delays.numel(), -1).gather(-1, clamped)
    return shifted * valid.to(sequence.dtype)


class CoincidenceLIFBank(nn.Module):
    def __init__(
        self,
        num_detectors: int,
        *,
        threshold: float = 1.0,
        beta_bounds: tuple[float, float] = (0.70, 0.995),
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.beta_min = beta_bounds[0]
        self.beta_span = beta_bounds[1] - beta_bounds[0]
        self.reference_weight_raw = nn.Parameter(torch.zeros(num_detectors))
        self.target_weight_raw = nn.Parameter(torch.zeros(num_detectors))
        self.beta_raw = nn.Parameter(torch.zeros(num_detectors))

    def forward(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        candidate_delays: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        delayed_reference = _shift_time_bank(reference, candidate_delays)
        target_bank = target.unsqueeze(1).expand(-1, candidate_delays.numel(), -1)
        reference_weight = F.softplus(self.reference_weight_raw).view(1, -1, 1)
        target_weight = F.softplus(self.target_weight_raw).view(1, -1, 1)
        beta = (self.beta_min + self.beta_span * torch.sigmoid(self.beta_raw)).view(1, -1)

        current = reference_weight * delayed_reference + target_weight * target_bank
        membrane = torch.zeros(current.shape[0], current.shape[1], device=current.device, dtype=current.dtype)
        spike_trace = []
        for time_index in range(current.shape[-1]):
            membrane = beta * membrane + current[..., time_index]
            spikes = (membrane >= self.threshold).to(current.dtype)
            membrane = (membrane - spikes * self.threshold).clamp_min(0.0)
            spike_trace.append(spikes)
        spike_tensor = torch.stack(spike_trace, dim=-1)
        pooled = spike_tensor.mean(dim=-1)
        diagnostics = {
            "reference_weight": reference_weight.detach().squeeze(0).squeeze(-1),
            "target_weight": target_weight.detach().squeeze(0).squeeze(-1),
            "beta": beta.detach().squeeze(0),
        }
        return pooled, spike_tensor, diagnostics


class LIFCoincidenceRound3Encoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: AllRound2Encoder,
        branch_hidden_dim: int,
        distance_candidates: torch.Tensor,
        itd_candidates: torch.Tensor,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.register_buffer("distance_candidates", distance_candidates)
        self.register_buffer("itd_candidates", itd_candidates)

        self.distance_left_bank = CoincidenceLIFBank(distance_candidates.numel())
        self.distance_right_bank = CoincidenceLIFBank(distance_candidates.numel())
        self.itd_bank = CoincidenceLIFBank(itd_candidates.numel())

        self.distance_projection = nn.Linear(distance_candidates.numel() * 2, branch_hidden_dim)
        self.azimuth_projection = nn.Linear(itd_candidates.numel(), branch_hidden_dim)
        self.distance_gain = nn.Parameter(torch.tensor(-1.2))
        self.azimuth_gain = nn.Parameter(torch.tensor(-1.2))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.transmit_spikes is None or batch.receive_spikes is None:
            raise ValueError("LIFCoincidenceRound3Encoder requires spike inputs.")
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

        distance_residual = self.distance_projection(torch.cat([distance_left, distance_right], dim=-1))
        azimuth_residual = self.azimuth_projection(itd_scores)
        distance_scale = 0.25 * torch.sigmoid(self.distance_gain)
        azimuth_scale = 0.25 * torch.sigmoid(self.azimuth_gain)

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
            "lif_distance_scale": distance_scale.detach(),
            "lif_azimuth_scale": azimuth_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=F.relu(base.distance_latent + distance_scale * distance_residual),
            azimuth_latent=F.relu(base.azimuth_latent + azimuth_scale * azimuth_residual),
            elevation_latent=base.elevation_latent,
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


def _comb_response(spectral_norm: torch.Tensor, lags: tuple[int, ...]) -> torch.Tensor:
    response = torch.zeros_like(spectral_norm)
    for lag in lags:
        left = F.pad(spectral_norm[..., :-lag], (lag, 0))
        right = F.pad(spectral_norm[..., lag:], (0, lag))
        response = response + torch.abs(spectral_norm - 0.5 * (left + right))
    return response / max(1, len(lags))


class CombFilterElevationEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: AllRound2Encoder,
        branch_hidden_dim: int,
        num_frequency_channels: int,
        comb_lags: tuple[int, ...] = (2, 4, 6),
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_frequency_channels = num_frequency_channels
        self.comb_lags = comb_lags
        self.comb_projection = nn.Linear(num_frequency_channels * 3, branch_hidden_dim)
        self.comb_gain = nn.Parameter(torch.tensor(-1.1))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.receive_spikes is None:
            raise ValueError("CombFilterElevationEncoder requires receive spikes.")
        base = self.base_encoder(batch)

        left_counts = batch.receive_spikes[:, 0].float().sum(dim=-1)
        right_counts = batch.receive_spikes[:, 1].float().sum(dim=-1)
        spectral_counts = left_counts + right_counts
        spectral_norm = spectral_counts / spectral_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_mean = F.avg_pool1d(spectral_norm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        spectral_notches = F.relu(local_mean - spectral_norm)
        comb = _comb_response(spectral_norm, self.comb_lags)
        comb_features = torch.cat([spectral_norm, spectral_notches, comb], dim=-1)
        comb_residual = self.comb_projection(F.layer_norm(comb_features, (comb_features.shape[-1],)))
        comb_scale = 0.30 * torch.sigmoid(self.comb_gain)

        diagnostics = {
            **base.diagnostics,
            "comb_spectral_norm": spectral_norm.detach(),
            "comb_spectral_notches": spectral_notches.detach(),
            "comb_response": comb.detach(),
            "comb_scale": comb_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=base.distance_latent,
            azimuth_latent=base.azimuth_latent,
            elevation_latent=F.relu(base.elevation_latent + comb_scale * comb_residual),
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )


class NotchDetectorElevationEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_encoder: AllRound2Encoder,
        branch_hidden_dim: int,
        num_frequency_channels: int,
        num_detectors: int = 12,
        detector_width_channels: float = 2.5,
    ) -> None:
        super().__init__()
        self.branch_hidden_dim = branch_hidden_dim
        self.base_encoder = base_encoder
        self.fusion_resonance_projection = base_encoder.fusion_resonance_projection
        self.num_frequency_channels = num_frequency_channels

        channel_axis = torch.arange(num_frequency_channels, dtype=torch.float32)
        detector_centers = torch.linspace(0.0, float(num_frequency_channels - 1), steps=num_detectors)
        templates = torch.exp(
            -0.5 * ((channel_axis.unsqueeze(0) - detector_centers.unsqueeze(1)) / detector_width_channels).square()
        )
        templates = templates / templates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("detector_centers", detector_centers)
        self.register_buffer("detector_templates", templates)

        self.detector_projection = nn.Linear(num_detectors, branch_hidden_dim)
        self.detector_gain = nn.Parameter(torch.tensor(-1.0))

    def forward(self, batch: ExperimentBatch) -> BranchEncoding:
        if batch.receive_spikes is None:
            raise ValueError("NotchDetectorElevationEncoder requires receive spikes.")
        base = self.base_encoder(batch)

        left_counts = batch.receive_spikes[:, 0].float().sum(dim=-1)
        right_counts = batch.receive_spikes[:, 1].float().sum(dim=-1)
        spectral_counts = left_counts + right_counts
        spectral_norm = spectral_counts / spectral_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_mean = F.avg_pool1d(spectral_norm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        spectral_notches = F.relu(local_mean - spectral_norm)

        detector_response = spectral_notches @ self.detector_templates.transpose(0, 1)
        detector_residual = self.detector_projection(F.layer_norm(detector_response, (detector_response.shape[-1],)))
        detector_scale = 0.30 * torch.sigmoid(self.detector_gain)

        diagnostics = {
            **base.diagnostics,
            "notch_detector_response": detector_response.detach(),
            "notch_detector_centers": self.detector_centers.detach(),
            "notch_detector_scale": detector_scale.detach(),
        }
        return BranchEncoding(
            distance_latent=base.distance_latent,
            azimuth_latent=base.azimuth_latent,
            elevation_latent=F.relu(base.elevation_latent + detector_scale * detector_residual),
            spectral_source=base.spectral_source,
            spike_proxy=base.spike_proxy,
            diagnostics=diagnostics,
        )
