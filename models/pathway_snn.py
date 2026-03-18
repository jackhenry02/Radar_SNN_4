from __future__ import annotations

from dataclasses import dataclass

import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn import delay_bank_features, ild_features, itd_features


def resize_feature(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    if features.shape[-1] == target_dim:
        return features
    return F.interpolate(features.unsqueeze(1), size=target_dim, mode="linear", align_corners=False).squeeze(1)


@dataclass
class PathwayBatch:
    distance: torch.Tensor
    azimuth: torch.Tensor
    elevation: torch.Tensor
    spike_count: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "PathwayBatch":
        return PathwayBatch(
            distance=self.distance[indices],
            azimuth=self.azimuth[indices],
            elevation=self.elevation[indices],
            spike_count=self.spike_count[indices],
        )


def build_pathway_features(
    transmit_spikes: torch.Tensor,
    receive_spikes: torch.Tensor,
    distance_candidates: torch.Tensor,
    itd_candidates: torch.Tensor,
    num_delay_lines: int,
    num_frequency_channels: int,
) -> tuple[PathwayBatch, dict[str, torch.Tensor]]:
    left_spikes = receive_spikes[:, 0]
    right_spikes = receive_spikes[:, 1]

    distance_left = delay_bank_features(transmit_spikes, left_spikes, distance_candidates)
    distance_right = delay_bank_features(transmit_spikes, right_spikes, distance_candidates)
    distance_branch = torch.cat(
        [resize_feature(distance_left, num_delay_lines), resize_feature(distance_right, num_delay_lines)],
        dim=-1,
    )

    itd_scores = itd_features(left_spikes, right_spikes, itd_candidates)
    ild_scores = ild_features(left_spikes, right_spikes)
    azimuth_branch = torch.cat(
        [resize_feature(itd_scores, num_delay_lines), resize_feature(ild_scores, num_delay_lines)],
        dim=-1,
    )

    left_counts = left_spikes.sum(dim=-1)
    right_counts = right_spikes.sum(dim=-1)
    spectral_counts = left_counts + right_counts
    spectral_norm = spectral_counts / spectral_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    local_mean = F.avg_pool1d(spectral_norm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    spectral_notches = F.relu(local_mean - spectral_norm)
    spectral_slope = F.pad(spectral_norm[..., 1:] - spectral_norm[..., :-1], (1, 0))
    elevation_branch = torch.cat(
        [
            resize_feature(spectral_norm, num_frequency_channels),
            resize_feature(spectral_notches, num_frequency_channels),
            resize_feature(spectral_slope, num_frequency_channels),
        ],
        dim=-1,
    )

    spike_count = receive_spikes.sum(dim=(-1, -2, -3)).float() + transmit_spikes.sum(dim=(-1, -2)).float()
    pathway_batch = PathwayBatch(
        distance=distance_branch,
        azimuth=azimuth_branch,
        elevation=elevation_branch,
        spike_count=spike_count,
    )
    diagnostics = {
        "distance_left": distance_left,
        "distance_right": distance_right,
        "itd_scores": itd_scores,
        "spectral_norm": spectral_norm,
        "spectral_notches": spectral_notches,
    }
    return pathway_batch, diagnostics


class PathwayFusionSNN(nn.Module):
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
    ) -> None:
        super().__init__()
        self.num_steps = num_steps

        self.distance_branch = nn.Linear(distance_dim, branch_hidden_dim)
        self.azimuth_branch = nn.Linear(azimuth_dim, branch_hidden_dim)
        self.elevation_branch = nn.Linear(elevation_dim, branch_hidden_dim)

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

    def forward(self, pathway_batch: PathwayBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.fusion_lif.reset_mem()
        self.integration_lif.reset_mem()
        distance_latent = F.relu(self.distance_branch(pathway_batch.distance))
        azimuth_latent = F.relu(self.azimuth_branch(pathway_batch.azimuth))
        elevation_latent = F.relu(self.elevation_branch(pathway_batch.elevation))
        fused = torch.cat([distance_latent, azimuth_latent, elevation_latent], dim=-1)
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
            "distance_latent": distance_latent,
            "azimuth_latent": azimuth_latent,
            "elevation_latent": elevation_latent,
            "fusion_spikes": fusion_spikes,
            "integration_spikes": integration_spikes,
            "spike_rate": integration_spikes.mean(dim=(1, 2)),
        }
        return output, diagnostics


@dataclass
class PathwayTrainingResult:
    train_loss: list[float]
    val_loss: list[float]
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_loss: float
    diagnostics: dict[str, torch.Tensor]


def _batch_iterator(
    pathway_batch: PathwayBatch,
    targets: torch.Tensor,
    batch_size: int,
) -> tuple[PathwayBatch, torch.Tensor]:
    permutation = torch.randperm(targets.shape[0], device=targets.device)
    for start in range(0, targets.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield pathway_batch.index_select(indices), targets[indices]


def train_pathway_snn(
    model: PathwayFusionSNN,
    train_batch: PathwayBatch,
    train_targets: torch.Tensor,
    val_batch: PathwayBatch,
    val_targets: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    spike_weight: float,
    target_weights: torch.Tensor,
) -> PathwayTrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(reduction="none")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_loss = float("inf")
    best_diagnostics: dict[str, torch.Tensor] = {}
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_features, batch_targets in _batch_iterator(train_batch, train_targets, batch_size):
            optimizer.zero_grad(set_to_none=True)
            output, diagnostics = model(batch_features)
            localisation_loss = (criterion(output, batch_targets) * target_weights.view(1, -1)).mean()
            spike_penalty = diagnostics["spike_rate"].mean()
            loss = localisation_loss + spike_weight * spike_penalty
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss_history.append(float(sum(epoch_losses) / max(1, len(epoch_losses))))

        model.eval()
        with torch.no_grad():
            val_output, val_diagnostics = model(val_batch)
            val_localisation_loss = (criterion(val_output, val_targets) * target_weights.view(1, -1)).mean()
            val_loss = val_localisation_loss + spike_weight * val_diagnostics["spike_rate"].mean()
            val_loss_value = float(val_loss.item())
        val_loss_history.append(val_loss_value)

        if val_loss_value < best_loss:
            best_loss = val_loss_value
            best_epoch = epoch
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            best_diagnostics = {key: value.detach().clone() for key, value in val_diagnostics.items()}

    if best_state is None:
        best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}

    return PathwayTrainingResult(
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        best_state=best_state,
        best_epoch=best_epoch,
        best_loss=best_loss,
        diagnostics=best_diagnostics,
    )
