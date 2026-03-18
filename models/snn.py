from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return (inputs > 0.0).to(inputs.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + inputs.abs()).square()
        return grad_output * surrogate


def surrogate_spike(inputs: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(inputs)


def collapse_spikes(spikes: torch.Tensor) -> torch.Tensor:
    if spikes.ndim == 4:
        return spikes.sum(dim=-2)
    if spikes.ndim == 3:
        return spikes.sum(dim=-2)
    raise ValueError(f"Unsupported spike tensor rank: {spikes.ndim}")


def onset_pathway(spikes: torch.Tensor) -> torch.Tensor:
    difference = torch.diff(spikes.float(), dim=-1, prepend=torch.zeros_like(spikes[..., :1]))
    return F.relu(difference)


def sustained_pathway(spikes: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = torch.ones(1, 1, kernel_size, device=spikes.device, dtype=spikes.dtype) / kernel_size
    smoothed = F.conv1d(spikes.reshape(-1, 1, spikes.shape[-1]), kernel, padding=kernel_size // 2)
    return smoothed.reshape_as(spikes)


def delay_coincidence_1d(
    reference: torch.Tensor,
    target: torch.Tensor,
    candidate_delays: torch.Tensor,
) -> torch.Tensor:
    total_steps = reference.shape[-1]
    base_indices = torch.arange(total_steps, device=reference.device).view(1, 1, total_steps)
    shifted_indices = base_indices + candidate_delays.view(1, -1, 1)
    valid = (shifted_indices >= 0) & (shifted_indices < total_steps)
    clamped = shifted_indices.clamp(0, total_steps - 1).expand(reference.shape[0], -1, -1)
    shifted_target = target.unsqueeze(1).expand(-1, candidate_delays.numel(), -1).gather(-1, clamped)
    coincidence = (reference.unsqueeze(1) * shifted_target * valid).sum(dim=-1)
    return coincidence


def delay_bank_features(
    transmit_spikes: torch.Tensor,
    receive_spikes: torch.Tensor,
    candidate_delays: torch.Tensor,
) -> torch.Tensor:
    transmit_sum = collapse_spikes(onset_pathway(transmit_spikes))
    receive_sum = collapse_spikes(onset_pathway(receive_spikes))
    return delay_coincidence_1d(transmit_sum, receive_sum, candidate_delays)


def itd_features(
    left_spikes: torch.Tensor,
    right_spikes: torch.Tensor,
    candidate_delays: torch.Tensor,
) -> torch.Tensor:
    left_sum = collapse_spikes(onset_pathway(left_spikes))
    right_sum = collapse_spikes(onset_pathway(right_spikes))
    return delay_coincidence_1d(left_sum, right_sum, candidate_delays)


def ild_features(left_spikes: torch.Tensor, right_spikes: torch.Tensor) -> torch.Tensor:
    left_counts = left_spikes.sum(dim=-1)
    right_counts = right_spikes.sum(dim=-1)
    difference = left_counts - right_counts
    total = left_counts + right_counts
    normalized = difference / total.clamp_min(1.0)
    return torch.cat([difference, normalized], dim=-1)


def spectral_features(left_spikes: torch.Tensor, right_spikes: torch.Tensor) -> torch.Tensor:
    binaural_counts = left_spikes.sum(dim=-1) + right_spikes.sum(dim=-1)
    normalized = binaural_counts / binaural_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    channel_axis = torch.linspace(-1.0, 1.0, normalized.shape[-1], device=normalized.device)
    centroid = (normalized * channel_axis).sum(dim=-1, keepdim=True)
    spread = (normalized * (channel_axis - centroid).square()).sum(dim=-1, keepdim=True)
    return torch.cat([normalized, centroid, spread], dim=-1)


class StaticFeatureSNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_steps: int = 12,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = self.fc_in(features)
        membrane_1 = torch.zeros(features.shape[0], self.hidden_dim, device=features.device)
        membrane_2 = torch.zeros_like(membrane_1)
        spike_trace = []

        for _ in range(self.num_steps):
            membrane_1 = self.beta * membrane_1 + current
            spike_1 = surrogate_spike(membrane_1 - self.threshold)
            membrane_1 = membrane_1 - spike_1 * self.threshold

            hidden_current = self.fc_hidden(spike_1)
            membrane_2 = self.beta * membrane_2 + hidden_current
            spike_2 = surrogate_spike(membrane_2 - self.threshold)
            membrane_2 = membrane_2 - spike_2 * self.threshold
            spike_trace.append(spike_2)

        spike_tensor = torch.stack(spike_trace, dim=1)
        pooled = spike_tensor.mean(dim=1)
        output = self.readout(pooled)
        diagnostics = {"hidden_spikes": spike_tensor, "pooled": pooled}
        return output, diagnostics


@dataclass
class TrainingResult:
    train_loss: list[float]
    val_loss: list[float]
    train_metric: list[float]
    val_metric: list[float]
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_metric: float
    gradient_norm: float
    weight_delta: float
    diagnostics: dict[str, torch.Tensor]


def _batch_iterator(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(features.shape[0], device=features.device)
    for start in range(0, features.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        yield features[indices], targets[indices]


def train_snn(
    model: StaticFeatureSNN,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    task: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
        compare_bigger = True
        best_metric = float("-inf")
    else:
        criterion = nn.SmoothL1Loss()
        compare_bigger = False
        best_metric = float("inf")

    initial_readout = model.readout.weight.detach().clone()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    last_gradient_norm = 0.0
    best_diagnostics: dict[str, torch.Tensor] = {}

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    train_metric_history: list[float] = []
    val_metric_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        batch_metrics = []
        for batch_features, batch_targets in _batch_iterator(train_features, train_targets, batch_size):
            optimizer.zero_grad(set_to_none=True)
            output, diagnostics = model(batch_features)
            loss = criterion(output, batch_targets)
            loss.backward()

            grad_sq_sum = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq_sum += float(parameter.grad.detach().pow(2).sum().item())
            last_gradient_norm = grad_sq_sum**0.5
            optimizer.step()

            batch_losses.append(loss.item())
            if task == "classification":
                predictions = output.argmax(dim=-1)
                batch_metrics.append((predictions == batch_targets).float().mean().item())
            else:
                batch_metrics.append(torch.mean(torch.abs(output - batch_targets)).item())

        train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))
        train_metric_history.append(float(sum(batch_metrics) / max(1, len(batch_metrics))))

        model.eval()
        with torch.no_grad():
            val_output, val_diagnostics = model(val_features)
            val_loss = criterion(val_output, val_targets).item()
            if task == "classification":
                val_predictions = val_output.argmax(dim=-1)
                val_metric = (val_predictions == val_targets).float().mean().item()
            else:
                val_metric = torch.mean(torch.abs(val_output - val_targets)).item()

        val_loss_history.append(val_loss)
        val_metric_history.append(val_metric)

        improved = (val_metric > best_metric) if compare_bigger else (val_metric < best_metric)
        if improved:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            best_diagnostics = {key: value.detach().clone() for key, value in val_diagnostics.items()}

    if best_state is None:
        best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}

    weight_delta = torch.norm(model.readout.weight.detach() - initial_readout).item()
    return TrainingResult(
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        train_metric=train_metric_history,
        val_metric=val_metric_history,
        best_state=best_state,
        best_epoch=best_epoch,
        best_metric=best_metric,
        gradient_norm=last_gradient_norm,
        weight_delta=weight_delta,
        diagnostics=best_diagnostics,
    )
