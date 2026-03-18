from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn.functional as F

from models.acoustics import (
    balanced_distance_dataset,
    cochlea_to_spikes,
    matched_filter_distance,
    sample_uniform_positions,
    simulate_echo_batch,
    spike_density,
)
from models.snn import (
    StaticFeatureSNN,
    delay_bank_features,
    ild_features,
    itd_features,
    spectral_features,
    train_snn,
)
from stages.base import BaseStage, StageContext
from utils.common import (
    GlobalConfig,
    angular_mae,
    combined_localisation_error,
    distance_mae,
    format_float,
    radians_to_degrees,
    save_cochlea_plot,
    save_error_histogram,
    save_heatmap,
    save_json,
    save_loss_curve,
    save_prediction_scatter,
    save_text_figure,
    save_waveform_and_spectrogram,
)


def _distance_candidates(
    config: GlobalConfig,
    device: torch.device,
    num_bins: int | None = None,
) -> torch.Tensor:
    min_bin = int(math.floor((2.0 * config.min_range_m / config.speed_of_sound_m_s) * config.envelope_rate_hz))
    max_bin = int(math.ceil((2.0 * config.max_range_m / config.speed_of_sound_m_s) * config.envelope_rate_hz))
    if num_bins is None or num_bins >= (max_bin - min_bin + 1):
        return torch.arange(min_bin, max_bin + 1, device=device, dtype=torch.long)
    return torch.linspace(min_bin, max_bin, num_bins, device=device).round().to(torch.long).unique(sorted=True)


def _itd_candidates(config: GlobalConfig, device: torch.device) -> torch.Tensor:
    max_itd_s = config.ear_spacing_m / config.speed_of_sound_m_s
    max_bins = int(math.ceil(max_itd_s * config.envelope_rate_hz)) + 2
    return torch.arange(-max_bins, max_bins + 1, device=device, dtype=torch.long)


def _standardize(
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (train_tensor - mean) / std, (val_tensor - mean) / std, mean, std


def _ridge_fit(features: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    augmented = torch.cat([features, torch.ones(features.shape[0], 1, device=features.device)], dim=-1)
    identity = torch.eye(augmented.shape[-1], device=features.device, dtype=features.dtype)
    identity[-1, -1] = 0.0
    return torch.linalg.solve(augmented.T @ augmented + ridge * identity, augmented.T @ targets)


def _ridge_predict(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    augmented = torch.cat([features, torch.ones(features.shape[0], 1, device=features.device)], dim=-1)
    return augmented @ weights


def _compress_features(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    if features.shape[-1] == target_dim:
        return features
    return F.interpolate(features.unsqueeze(1), size=target_dim, mode="linear", align_corners=False).squeeze(1)


def _extract_front_end(acoustic_batch: Any, config: GlobalConfig) -> dict[str, torch.Tensor]:
    transmit_front = cochlea_to_spikes(acoustic_batch.transmit, config)
    receive_front = cochlea_to_spikes(acoustic_batch.receive, config)
    return {
        "transmit_cochlea": transmit_front["cochleagram"],
        "transmit_spikes": transmit_front["spikes"],
        "receive_cochlea": receive_front["cochleagram"],
        "receive_spikes": receive_front["spikes"],
        "center_frequencies": receive_front["center_frequencies"],
    }


def _distance_features(
    transmit_spikes: torch.Tensor,
    receive_spikes: torch.Tensor,
    config: GlobalConfig,
    num_bins: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = _distance_candidates(config, transmit_spikes.device, num_bins=num_bins)
    features = delay_bank_features(transmit_spikes, receive_spikes, candidates)
    return features, candidates


def _estimate_delay_from_scores(scores: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    indices = scores.argmax(dim=-1)
    return candidates[indices]


def _stage_note(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


class Model0Baseline(BaseStage):
    name = "model0_classical_baseline"
    max_attempts = 3

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [
            {"noise_scale": 1.0},
            {"noise_scale": 0.75},
            {"noise_scale": 0.55},
        ]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        config = context.config
        radii, _, _ = sample_uniform_positions(128, config, context.device, include_elevation=False)
        batch = simulate_echo_batch(
            config,
            radii_m=radii,
            binaural=False,
            add_noise=True,
        )
        batch.receive = batch.receive + (params["noise_scale"] - 1.0) * config.noise_std * torch.randn_like(batch.receive)
        estimate_m, correlation = matched_filter_distance(
            batch.receive.squeeze(1),
            batch.transmit[:, : config.chirp_samples],
            config.sample_rate_hz,
            config.speed_of_sound_m_s,
        )
        mae = distance_mae(estimate_m, radii)
        success_threshold = 0.05 * config.max_range_m
        success = mae < success_threshold

        front_end = _extract_front_end(batch, config)
        save_waveform_and_spectrogram(
            batch.receive[0, 0],
            config.sample_rate_hz,
            stage_dir / f"attempt_{attempt}_signal.png",
            "Model 0 Receive Waveform and Spectrogram",
        )
        save_cochlea_plot(
            front_end["receive_cochlea"][0, 0],
            front_end["receive_spikes"][0, 0],
            config.envelope_rate_hz,
            stage_dir / f"attempt_{attempt}_cochlea.png",
            "Model 0 Cochleagram and Spike Raster",
        )
        save_heatmap(
            correlation[:32],
            stage_dir / f"attempt_{attempt}_matched_filter.png",
            "Model 0 Matched Filter Population Activity",
            xlabel="Lag Bin",
            ylabel="Sample",
        )
        save_prediction_scatter(
            radii,
            estimate_m,
            stage_dir / f"attempt_{attempt}_distance_scatter.png",
            "Model 0 Distance Prediction",
            xlabel="True Distance (m)",
            ylabel="Predicted Distance (m)",
        )
        save_error_histogram(
            estimate_m - radii,
            stage_dir / f"attempt_{attempt}_distance_error.png",
            "Model 0 Distance Error Distribution",
            xlabel="Prediction Error (m)",
        )
        metrics = {
            "distance_mae_m": format_float(mae),
            "success_threshold_m": format_float(success_threshold),
        }
        return success, mae, metrics, _stage_note(metrics)


class Model1Coincidence(BaseStage):
    name = "model1_coincidence_detection"
    max_attempts = 2

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        device = context.device
        total_steps = 160
        candidate_delays = torch.arange(0, 41, device=device, dtype=torch.long)
        true_delays = torch.randint(4, 36, (64,), device=device)
        base = torch.zeros(64, total_steps, device=device)
        event_positions = torch.tensor([12, 26, 43, 75, 101, 133], device=device)
        base[:, event_positions] = 1.0
        target = torch.zeros_like(base)
        for sample_index in range(base.shape[0]):
            target[sample_index] = torch.roll(base[sample_index], shifts=int(true_delays[sample_index].item()))
            target[sample_index, : true_delays[sample_index]] = 0.0
        scores = delay_bank_features(base.unsqueeze(1), target.unsqueeze(1), candidate_delays)
        predicted_delays = _estimate_delay_from_scores(scores, candidate_delays)
        peak_error = torch.abs(predicted_delays - true_delays).float().max().item()
        success = peak_error < 1.0

        synthetic_signal = base[0] - target[0]
        synthetic_cochlea = torch.stack([base[0], target[0]], dim=0)
        save_waveform_and_spectrogram(
            synthetic_signal,
            sample_rate_hz=64_000,
            path=stage_dir / f"attempt_{attempt}_signal.png",
            title="Model 1 Synthetic Spike Pulse Train",
        )
        save_cochlea_plot(
            synthetic_cochlea,
            synthetic_cochlea,
            sample_rate_hz=64_000,
            path=stage_dir / f"attempt_{attempt}_cochlea.png",
            title="Model 1 Reference and Echo Spike Raster",
        )
        save_heatmap(
            scores[:32],
            stage_dir / f"attempt_{attempt}_coincidence_heatmap.png",
            "Model 1 Delay Sweep Coincidence Activity",
            xlabel="Candidate Delay",
            ylabel="Sample",
        )
        save_prediction_scatter(
            true_delays.float(),
            predicted_delays.float(),
            stage_dir / f"attempt_{attempt}_delay_scatter.png",
            "Model 1 Delay Prediction",
            xlabel="True Delay (bins)",
            ylabel="Predicted Delay (bins)",
        )
        save_error_histogram(
            (predicted_delays - true_delays).float(),
            stage_dir / f"attempt_{attempt}_delay_error.png",
            "Model 1 Delay Error Distribution",
            xlabel="Delay Error (bins)",
        )
        metrics = {"peak_delay_error_bins": format_float(peak_error)}
        return success, peak_error, metrics, _stage_note(metrics)


class Model2DelayBank(BaseStage):
    name = "model2_delay_bank"
    max_attempts = 2

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        device = context.device
        batch_size = 96
        channels = 12
        total_steps = 180
        candidate_delays = torch.arange(0, 61, device=device, dtype=torch.long)
        true_delays = torch.randint(3, 58, (batch_size,), device=device)

        transmit = torch.zeros(batch_size, channels, total_steps, device=device)
        base_events = torch.tensor([8, 21, 39, 57, 80, 109, 141], device=device)
        for channel in range(channels):
            shift = channel % 5
            transmit[:, channel, base_events + shift] = 1.0

        receive = torch.zeros_like(transmit)
        for sample_index in range(batch_size):
            delay = int(true_delays[sample_index].item())
            receive[sample_index] = torch.roll(transmit[sample_index], shifts=delay, dims=-1)
            receive[sample_index, :, :delay] = 0.0

        scores = delay_bank_features(transmit, receive, candidate_delays)
        predicted = _estimate_delay_from_scores(scores, candidate_delays)
        mean_error = torch.mean(torch.abs(predicted - true_delays).float()).item()
        success = mean_error < 2.0

        synthetic_signal = transmit[0].sum(dim=0) - receive[0].sum(dim=0)
        save_waveform_and_spectrogram(
            synthetic_signal,
            sample_rate_hz=64_000,
            path=stage_dir / f"attempt_{attempt}_signal.png",
            title="Model 2 Delay Bank Synthetic Pulse Train",
        )
        save_cochlea_plot(
            transmit[0],
            receive[0],
            sample_rate_hz=64_000,
            path=stage_dir / f"attempt_{attempt}_cochlea.png",
            title="Model 2 Multi-Channel Spike Raster",
        )
        save_heatmap(
            scores[:48],
            stage_dir / f"attempt_{attempt}_population_heatmap.png",
            "Model 2 Coincidence Neuron Population Activity",
            xlabel="Delay Candidate",
            ylabel="Sample",
        )
        save_prediction_scatter(
            true_delays.float(),
            predicted.float(),
            stage_dir / f"attempt_{attempt}_prediction.png",
            "Model 2 Delay Bank Prediction",
            xlabel="True Delay (bins)",
            ylabel="Predicted Delay (bins)",
        )
        save_error_histogram(
            (predicted - true_delays).float(),
            stage_dir / f"attempt_{attempt}_error.png",
            "Model 2 Delay Error Distribution",
            xlabel="Delay Error (bins)",
        )
        metrics = {"mean_delay_error_bins": format_float(mean_error)}
        return success, mean_error, metrics, _stage_note(metrics)


class Model3SignalToSpikes(BaseStage):
    name = "model3_signal_to_spikes"
    max_attempts = 3

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [
            {"spike_threshold": 0.42},
            {"spike_threshold": 0.36},
            {"spike_threshold": 0.30},
        ]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        config = context.config
        local_config = GlobalConfig(**{**config.__dict__, "spike_threshold": params["spike_threshold"]})
        radii, _, _ = sample_uniform_positions(96, local_config, context.device, include_elevation=False)
        batch = simulate_echo_batch(local_config, radii_m=radii, binaural=False, add_noise=True)
        front_end = _extract_front_end(batch, local_config)
        energy = front_end["receive_cochlea"].sum(dim=(-1, -2, -3))
        density = front_end["receive_spikes"].sum(dim=(-1, -2, -3))
        correlation = torch.corrcoef(torch.stack([energy, density]))[0, 1].item()
        success = correlation > 0.7

        save_waveform_and_spectrogram(
            batch.receive[0, 0],
            local_config.sample_rate_hz,
            stage_dir / f"attempt_{attempt}_signal.png",
            "Model 3 Receive Signal and Spectrogram",
        )
        save_cochlea_plot(
            front_end["receive_cochlea"][0, 0],
            front_end["receive_spikes"][0, 0],
            local_config.envelope_rate_hz,
            stage_dir / f"attempt_{attempt}_cochlea.png",
            "Model 3 Cochleagram and Spike Raster",
        )
        save_heatmap(
            spike_density(front_end["receive_spikes"][0, 0]),
            stage_dir / f"attempt_{attempt}_snn_activity.png",
            "Model 3 Spike Density Heatmap",
            xlabel="Time Bin",
            ylabel="Channel",
        )
        save_prediction_scatter(
            energy,
            density,
            stage_dir / f"attempt_{attempt}_prediction.png",
            "Model 3 Spike Density vs Signal Energy",
            xlabel="Signal Energy",
            ylabel="Spike Density",
        )
        save_error_histogram(
            density - energy,
            stage_dir / f"attempt_{attempt}_error.png",
            "Model 3 Density-Energy Difference",
            xlabel="Density - Energy",
        )
        metrics = {
            "energy_spike_correlation": format_float(correlation),
            "spike_threshold": params["spike_threshold"],
        }
        return success, -correlation, metrics, _stage_note(metrics)


class Model4Trainable(BaseStage):
    name = "model4_full_pipeline_trainable"
    max_attempts = 4

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [
            {"hidden_dim": 72, "lr": 0.010, "epochs": 20, "num_steps": 10, "num_bins": 96},
            {"hidden_dim": 96, "lr": 0.008, "epochs": 24, "num_steps": 12, "num_bins": 112},
            {"hidden_dim": 120, "lr": 0.006, "epochs": 28, "num_steps": 14, "num_bins": 128},
            {"hidden_dim": 144, "lr": 0.005, "epochs": 32, "num_steps": 16, "num_bins": 144},
        ]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        config = context.config
        num_classes = 4
        train_radii, train_labels = balanced_distance_dataset(80, config, context.device, num_classes=num_classes)
        val_radii, val_labels = balanced_distance_dataset(24, config, context.device, num_classes=num_classes)

        train_batch = simulate_echo_batch(config, radii_m=train_radii, binaural=False, add_noise=True)
        val_batch = simulate_echo_batch(config, radii_m=val_radii, binaural=False, add_noise=True)

        train_front = _extract_front_end(train_batch, config)
        val_front = _extract_front_end(val_batch, config)

        train_delay, _ = _distance_features(
            train_front["transmit_spikes"], train_front["receive_spikes"][:, 0], config, num_bins=params["num_bins"]
        )
        val_delay, _ = _distance_features(
            val_front["transmit_spikes"], val_front["receive_spikes"][:, 0], config, num_bins=params["num_bins"]
        )

        train_counts = train_front["receive_spikes"][:, 0].sum(dim=-1)
        val_counts = val_front["receive_spikes"][:, 0].sum(dim=-1)
        train_features = torch.cat([train_delay, train_counts], dim=-1)
        val_features = torch.cat([val_delay, val_counts], dim=-1)
        train_features, val_features, _, _ = _standardize(train_features, val_features)

        model = StaticFeatureSNN(
            input_dim=train_features.shape[-1],
            hidden_dim=params["hidden_dim"],
            output_dim=num_classes,
            num_steps=params["num_steps"],
            beta=0.9,
        ).to(context.device)

        training = train_snn(
            model,
            train_features,
            train_labels,
            val_features,
            val_labels,
            task="classification",
            epochs=params["epochs"],
            lr=params["lr"],
            batch_size=64,
        )
        model.load_state_dict(training.best_state)
        model.eval()
        with torch.no_grad():
            val_output, diagnostics = model(val_features)
            predictions = val_output.argmax(dim=-1)
            accuracy = (predictions == val_labels).float().mean().item()
            final_val_loss = torch.nn.CrossEntropyLoss()(val_output, val_labels).item()

        initial_loss = training.train_loss[0]
        best_train_loss = min(training.train_loss)
        loss_drop = (initial_loss - best_train_loss) / max(initial_loss, 1e-6)
        random_baseline = 1.0 / num_classes
        no_nans = float(torch.isfinite(val_output).all().item())
        success = (
            loss_drop > 0.30
            and accuracy > random_baseline + 0.20
            and training.gradient_norm > 0.0
            and training.weight_delta > 0.0
            and bool(no_nans)
        )

        save_waveform_and_spectrogram(
            val_batch.receive[0, 0],
            config.sample_rate_hz,
            stage_dir / f"attempt_{attempt}_signal.png",
            "Model 4 Receive Signal and Spectrogram",
        )
        save_cochlea_plot(
            val_front["receive_cochlea"][0, 0],
            val_front["receive_spikes"][0, 0],
            config.envelope_rate_hz,
            stage_dir / f"attempt_{attempt}_cochlea.png",
            "Model 4 Cochleagram and Spike Raster",
        )
        save_heatmap(
            diagnostics["hidden_spikes"][0].T,
            stage_dir / f"attempt_{attempt}_hidden_spikes.png",
            "Model 4 Hidden Layer Spike Activity",
            xlabel="Pseudo-Time Step",
            ylabel="Neuron",
        )
        save_prediction_scatter(
            val_labels.float(),
            predictions.float(),
            stage_dir / f"attempt_{attempt}_prediction.png",
            "Model 4 Distance Class Prediction",
            xlabel="True Class",
            ylabel="Predicted Class",
        )
        save_error_histogram(
            (predictions - val_labels).float(),
            stage_dir / f"attempt_{attempt}_error.png",
            "Model 4 Class Error Distribution",
            xlabel="Class Error",
        )
        save_loss_curve(
            training.train_loss,
            training.val_loss,
            stage_dir / f"attempt_{attempt}_loss.png",
            "Model 4 Training Loss",
        )
        metrics = {
            "val_accuracy": format_float(accuracy),
            "random_baseline": format_float(random_baseline),
            "loss_drop_fraction": format_float(loss_drop),
            "val_loss": format_float(final_val_loss),
            "gradient_norm": format_float(training.gradient_norm),
            "weight_delta": format_float(training.weight_delta),
            "no_nans": bool(no_nans),
        }
        context.shared["model4_best_accuracy"] = accuracy
        return success, -accuracy, metrics, _stage_note(metrics)


class Model5Binaural(BaseStage):
    name = "model5_binaural_localisation"
    max_attempts = 3

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [
            {"delay_bins": 128, "ridge": 1e-2},
            {"delay_bins": 160, "ridge": 5e-3},
            {"delay_bins": 192, "ridge": 1e-3},
        ]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        config = context.config
        train_r, train_az, _ = sample_uniform_positions(
            192,
            config,
            context.device,
            azimuth_limits_deg=(-45.0, 45.0),
            include_elevation=False,
        )
        val_r, val_az, _ = sample_uniform_positions(
            96,
            config,
            context.device,
            azimuth_limits_deg=(-45.0, 45.0),
            include_elevation=False,
        )
        train_batch = simulate_echo_batch(config, train_r, train_az, binaural=True, add_noise=True)
        val_batch = simulate_echo_batch(config, val_r, val_az, binaural=True, add_noise=True)
        train_front = _extract_front_end(train_batch, config)
        val_front = _extract_front_end(val_batch, config)

        delay_candidates = _distance_candidates(config, context.device, num_bins=params["delay_bins"])
        itd_candidates = _itd_candidates(config, context.device)

        train_left_delay = delay_bank_features(
            train_front["transmit_spikes"], train_front["receive_spikes"][:, 0], delay_candidates
        )
        train_right_delay = delay_bank_features(
            train_front["transmit_spikes"], train_front["receive_spikes"][:, 1], delay_candidates
        )
        val_left_delay = delay_bank_features(
            val_front["transmit_spikes"], val_front["receive_spikes"][:, 0], delay_candidates
        )
        val_right_delay = delay_bank_features(
            val_front["transmit_spikes"], val_front["receive_spikes"][:, 1], delay_candidates
        )
        train_left_bins = _estimate_delay_from_scores(train_left_delay, delay_candidates)
        train_right_bins = _estimate_delay_from_scores(train_right_delay, delay_candidates)
        val_left_bins = _estimate_delay_from_scores(val_left_delay, delay_candidates)
        val_right_bins = _estimate_delay_from_scores(val_right_delay, delay_candidates)

        val_distance_m = (
            config.speed_of_sound_m_s
            * (val_left_bins.float() + val_right_bins.float())
            / (4.0 * config.envelope_rate_hz)
        )

        train_itd_scores = itd_features(
            train_front["receive_spikes"][:, 0], train_front["receive_spikes"][:, 1], itd_candidates
        )
        val_itd_scores = itd_features(
            val_front["receive_spikes"][:, 0], val_front["receive_spikes"][:, 1], itd_candidates
        )
        train_itd_bins = _estimate_delay_from_scores(train_itd_scores, itd_candidates).float()
        val_itd_bins = _estimate_delay_from_scores(val_itd_scores, itd_candidates).float()
        train_itd_s = train_itd_bins / config.envelope_rate_hz
        val_itd_s = val_itd_bins / config.envelope_rate_hz

        train_ild_scalar = ild_features(
            train_front["receive_spikes"][:, 0], train_front["receive_spikes"][:, 1]
        ).mean(dim=-1, keepdim=True)
        val_ild_scalar = ild_features(
            val_front["receive_spikes"][:, 0], val_front["receive_spikes"][:, 1]
        ).mean(dim=-1, keepdim=True)

        train_az_features = torch.cat([train_itd_s.unsqueeze(-1), train_ild_scalar], dim=-1)
        val_az_features = torch.cat([val_itd_s.unsqueeze(-1), val_ild_scalar], dim=-1)
        az_weights = _ridge_fit(train_az_features, train_az.unsqueeze(-1), ridge=params["ridge"])
        val_azimuth_deg = _ridge_predict(val_az_features, az_weights).squeeze(-1).clamp(-60.0, 60.0)

        distance_error = distance_mae(val_distance_m, val_r)
        azimuth_error = angular_mae(val_azimuth_deg, val_az)
        success = distance_error < 0.10 * config.max_range_m and azimuth_error < 15.0

        save_waveform_and_spectrogram(
            val_batch.receive[0, 0],
            config.sample_rate_hz,
            stage_dir / f"attempt_{attempt}_signal.png",
            "Model 5 Left Ear Waveform and Spectrogram",
        )
        save_cochlea_plot(
            val_front["receive_cochlea"][0, 0],
            val_front["receive_spikes"][0, 0],
            config.envelope_rate_hz,
            stage_dir / f"attempt_{attempt}_cochlea.png",
            "Model 5 Left Ear Cochleagram and Spike Raster",
        )
        save_heatmap(
            torch.cat(
                [
                    _compress_features(val_left_delay[:32], 128),
                    _compress_features(val_itd_scores[:32], 128),
                ],
                dim=0,
            ),
            stage_dir / f"attempt_{attempt}_snn_activity.png",
            "Model 5 Distance and ITD Population Activity",
            xlabel="Delay Candidate",
            ylabel="Sample",
        )
        save_prediction_scatter(
            val_r,
            val_distance_m,
            stage_dir / f"attempt_{attempt}_distance_prediction.png",
            "Model 5 Distance Prediction",
            xlabel="True Distance (m)",
            ylabel="Predicted Distance (m)",
        )
        save_prediction_scatter(
            val_az,
            val_azimuth_deg,
            stage_dir / f"attempt_{attempt}_azimuth_prediction.png",
            "Model 5 Azimuth Prediction",
            xlabel="True Azimuth (deg)",
            ylabel="Predicted Azimuth (deg)",
        )
        save_error_histogram(
            val_azimuth_deg - val_az,
            stage_dir / f"attempt_{attempt}_azimuth_error.png",
            "Model 5 Azimuth Error Distribution",
            xlabel="Azimuth Error (deg)",
        )
        metrics = {
            "distance_mae_m": format_float(distance_error),
            "angular_mae_deg": format_float(azimuth_error),
        }
        context.shared["model5_distance_mae"] = distance_error
        context.shared["model5_angular_mae"] = azimuth_error
        return success, distance_error + azimuth_error / 180.0, metrics, _stage_note(metrics)


class Model6Full3D(BaseStage):
    name = "model6_full_3d_localisation"
    max_attempts = 4

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [
            {"hidden_dim": 80, "lr": 0.010, "epochs": 18, "num_steps": 10},
            {"hidden_dim": 112, "lr": 0.008, "epochs": 22, "num_steps": 12},
            {"hidden_dim": 144, "lr": 0.006, "epochs": 26, "num_steps": 14},
            {"hidden_dim": 176, "lr": 0.005, "epochs": 30, "num_steps": 16},
        ]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        config = context.config
        train_r, train_az, train_el = sample_uniform_positions(
            224,
            config,
            context.device,
            azimuth_limits_deg=(-45.0, 45.0),
            elevation_limits_deg=(-30.0, 30.0),
            include_elevation=True,
        )
        val_r, val_az, val_el = sample_uniform_positions(
            96,
            config,
            context.device,
            azimuth_limits_deg=(-45.0, 45.0),
            elevation_limits_deg=(-30.0, 30.0),
            include_elevation=True,
        )

        train_batch = simulate_echo_batch(
            config,
            train_r,
            train_az,
            train_el,
            binaural=True,
            add_noise=True,
            include_elevation_cues=True,
        )
        val_batch = simulate_echo_batch(
            config,
            val_r,
            val_az,
            val_el,
            binaural=True,
            add_noise=True,
            include_elevation_cues=True,
        )
        train_front = _extract_front_end(train_batch, config)
        val_front = _extract_front_end(val_batch, config)

        distance_candidates = _distance_candidates(config, context.device, num_bins=192)
        itd_candidates = _itd_candidates(config, context.device)

        train_distance_left = delay_bank_features(
            train_front["transmit_spikes"], train_front["receive_spikes"][:, 0], distance_candidates
        )
        train_distance_right = delay_bank_features(
            train_front["transmit_spikes"], train_front["receive_spikes"][:, 1], distance_candidates
        )
        val_distance_left = delay_bank_features(
            val_front["transmit_spikes"], val_front["receive_spikes"][:, 0], distance_candidates
        )
        val_distance_right = delay_bank_features(
            val_front["transmit_spikes"], val_front["receive_spikes"][:, 1], distance_candidates
        )

        train_itd = itd_features(train_front["receive_spikes"][:, 0], train_front["receive_spikes"][:, 1], itd_candidates)
        val_itd = itd_features(val_front["receive_spikes"][:, 0], val_front["receive_spikes"][:, 1], itd_candidates)
        train_ild = ild_features(train_front["receive_spikes"][:, 0], train_front["receive_spikes"][:, 1])
        val_ild = ild_features(val_front["receive_spikes"][:, 0], val_front["receive_spikes"][:, 1])
        train_spectral = spectral_features(train_front["receive_spikes"][:, 0], train_front["receive_spikes"][:, 1])
        val_spectral = spectral_features(val_front["receive_spikes"][:, 0], val_front["receive_spikes"][:, 1])

        train_features = torch.cat(
            [
                _compress_features(train_distance_left, 96),
                _compress_features(train_distance_right, 96),
                train_itd,
                _compress_features(train_ild, 24),
                train_spectral,
            ],
            dim=-1,
        )
        val_features = torch.cat(
            [
                _compress_features(val_distance_left, 96),
                _compress_features(val_distance_right, 96),
                val_itd,
                _compress_features(val_ild, 24),
                val_spectral,
            ],
            dim=-1,
        )

        train_features, val_features, _, _ = _standardize(train_features, val_features)
        train_targets = torch.stack([train_r, train_az / 45.0, train_el / 30.0], dim=-1)
        val_targets = torch.stack([val_r, val_az / 45.0, val_el / 30.0], dim=-1)
        train_targets, val_targets, target_mean, target_std = _standardize(train_targets, val_targets)

        model = StaticFeatureSNN(
            input_dim=train_features.shape[-1],
            hidden_dim=params["hidden_dim"],
            output_dim=3,
            num_steps=params["num_steps"],
            beta=0.9,
        ).to(context.device)

        training = train_snn(
            model,
            train_features,
            train_targets,
            val_features,
            val_targets,
            task="regression",
            epochs=params["epochs"],
            lr=params["lr"],
            batch_size=64,
        )
        model.load_state_dict(training.best_state)
        model.eval()
        with torch.no_grad():
            val_output, diagnostics = model(val_features)
            denormalized = val_output * target_std + target_mean
            predicted_distance = denormalized[:, 0]
            predicted_azimuth = denormalized[:, 1] * 45.0
            predicted_elevation = denormalized[:, 2] * 30.0

        distance_error = distance_mae(predicted_distance, val_r)
        azimuth_error = angular_mae(predicted_azimuth, val_az)
        elevation_error = angular_mae(predicted_elevation, val_el)
        total_error = combined_localisation_error(
            predicted_distance,
            val_r,
            predicted_azimuth,
            val_az,
            predicted_elevation,
            val_el,
            config.max_range_m,
        )

        delay_left_bins = _estimate_delay_from_scores(val_distance_left, distance_candidates).float()
        delay_right_bins = _estimate_delay_from_scores(val_distance_right, distance_candidates).float()
        model5_distance = config.speed_of_sound_m_s * (delay_left_bins + delay_right_bins) / (4.0 * config.envelope_rate_hz)
        itd_bins = _estimate_delay_from_scores(val_itd, itd_candidates).float()
        model5_azimuth = -radians_to_degrees(
            torch.asin((config.speed_of_sound_m_s * itd_bins / config.envelope_rate_hz / config.ear_spacing_m).clamp(-0.98, 0.98))
        )
        model5_total_error = combined_localisation_error(
            model5_distance,
            val_r,
            model5_azimuth,
            val_az,
            torch.zeros_like(val_el),
            val_el,
            config.max_range_m,
        )

        success = total_error < 0.18 and total_error < model5_total_error

        save_waveform_and_spectrogram(
            val_batch.receive[0, 0],
            config.sample_rate_hz,
            stage_dir / f"attempt_{attempt}_signal.png",
            "Model 6 Left Ear Signal and Spectrogram",
        )
        save_cochlea_plot(
            val_front["receive_cochlea"][0, 0],
            val_front["receive_spikes"][0, 0],
            config.envelope_rate_hz,
            stage_dir / f"attempt_{attempt}_cochlea.png",
            "Model 6 Left Ear Cochleagram and Spike Raster",
        )
        save_heatmap(
            diagnostics["hidden_spikes"][0].T,
            stage_dir / f"attempt_{attempt}_hidden_spikes.png",
            "Model 6 Multi-Branch Integration Activity",
            xlabel="Pseudo-Time Step",
            ylabel="Neuron",
        )
        save_prediction_scatter(
            val_r,
            predicted_distance,
            stage_dir / f"attempt_{attempt}_distance_prediction.png",
            "Model 6 Distance Prediction",
            xlabel="True Distance (m)",
            ylabel="Predicted Distance (m)",
        )
        save_prediction_scatter(
            val_el,
            predicted_elevation,
            stage_dir / f"attempt_{attempt}_elevation_prediction.png",
            "Model 6 Elevation Prediction",
            xlabel="True Elevation (deg)",
            ylabel="Predicted Elevation (deg)",
        )
        save_error_histogram(
            predicted_elevation - val_el,
            stage_dir / f"attempt_{attempt}_elevation_error.png",
            "Model 6 Elevation Error Distribution",
            xlabel="Elevation Error (deg)",
        )
        save_loss_curve(
            training.train_loss,
            training.val_loss,
            stage_dir / f"attempt_{attempt}_loss.png",
            "Model 6 Training Loss",
        )
        metrics = {
            "distance_mae_m": format_float(distance_error),
            "azimuth_mae_deg": format_float(azimuth_error),
            "elevation_mae_deg": format_float(elevation_error),
            "combined_error": format_float(total_error),
            "model5_comparison_error": format_float(model5_total_error),
        }
        context.shared["model6_baseline_error"] = total_error
        context.shared["model6_train_features"] = train_features.detach().clone()
        context.shared["model6_val_features"] = val_features.detach().clone()
        context.shared["model6_train_targets"] = train_targets.detach().clone()
        context.shared["model6_val_targets"] = val_targets.detach().clone()
        context.shared["model6_target_mean"] = target_mean.detach().clone()
        context.shared["model6_target_std"] = target_std.detach().clone()
        return success, total_error, metrics, _stage_note(metrics)


class Model7Optimisation(BaseStage):
    name = "model7_optimisation"
    max_attempts = 1

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        baseline_error = float(context.shared["model6_baseline_error"])
        train_features = context.shared["model6_train_features"]
        val_features = context.shared["model6_val_features"]
        train_targets = context.shared["model6_train_targets"]
        val_targets = context.shared["model6_val_targets"]
        target_mean = context.shared["model6_target_mean"]
        target_std = context.shared["model6_target_std"]

        trial_errors: list[float] = []

        def objective(trial: optuna.Trial) -> float:
            hidden_dim = trial.suggest_int("hidden_dim", 72, 168, step=16)
            num_steps = trial.suggest_int("num_steps", 8, 14, step=2)
            lr = trial.suggest_float("lr", 0.003, 0.012, log=True)
            epochs = trial.suggest_int("epochs", 12, 24, step=4)
            beta = trial.suggest_float("beta", 0.82, 0.95)
            model = StaticFeatureSNN(
                input_dim=train_features.shape[-1],
                hidden_dim=hidden_dim,
                output_dim=3,
                num_steps=num_steps,
                beta=beta,
            ).to(context.device)
            training = train_snn(
                model,
                train_features,
                train_targets,
                val_features,
                val_targets,
                task="regression",
                epochs=epochs,
                lr=lr,
                batch_size=48,
            )
            model.load_state_dict(training.best_state)
            model.eval()
            with torch.no_grad():
                val_output, _ = model(val_features)
                denormalized = val_output * target_std + target_mean
                predicted_distance = denormalized[:, 0]
                predicted_azimuth = denormalized[:, 1] * 45.0
                predicted_elevation = denormalized[:, 2] * 30.0
                true_distance = val_targets[:, 0] * target_std[:, 0] + target_mean[:, 0]
                true_azimuth = (val_targets[:, 1] * target_std[:, 1] + target_mean[:, 1]) * 45.0
                true_elevation = (val_targets[:, 2] * target_std[:, 2] + target_mean[:, 2]) * 30.0
                error = combined_localisation_error(
                    predicted_distance,
                    true_distance,
                    predicted_azimuth,
                    true_azimuth,
                    predicted_elevation,
                    true_elevation,
                    context.config.max_range_m,
                )
            trial_errors.append(error)
            return error

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=context.config.seed))
        study.optimize(objective, n_trials=4, show_progress_bar=False)

        best_error = float(study.best_value)
        improvement_fraction = (baseline_error - best_error) / max(baseline_error, 1e-6)
        success = improvement_fraction > 0.10

        save_text_figure(
            [
                "Optuna best parameters:",
                *[f"{key}: {value}" for key, value in study.best_params.items()],
                f"baseline_error: {baseline_error:.4f}",
                f"best_error: {best_error:.4f}",
                f"improvement_fraction: {improvement_fraction:.4f}",
            ],
            stage_dir / f"attempt_{attempt}_summary.png",
            "Model 7 Optuna Summary",
        )
        save_heatmap(
            torch.tensor(trial_errors, device=context.device).view(1, -1),
            stage_dir / f"attempt_{attempt}_trial_errors.png",
            "Model 7 Trial Error History",
            xlabel="Trial",
            ylabel="Combined Error",
        )
        save_json(stage_dir / "best_params.json", {"best_params": study.best_params, "best_value": best_error})
        metrics = {
            "baseline_error": format_float(baseline_error),
            "best_error": format_float(best_error),
            "improvement_fraction": format_float(improvement_fraction),
            "best_params": study.best_params,
        }
        return success, best_error, metrics, _stage_note(metrics)


def build_stages() -> list[BaseStage]:
    return [
        Model0Baseline(),
        Model1Coincidence(),
        Model2DelayBank(),
        Model3SignalToSpikes(),
        Model4Trainable(),
        Model5Binaural(),
        Model6Full3D(),
        Model7Optimisation(),
    ]


@dataclass
class PipelineSummary:
    stages: list[dict[str, Any]]


class PipelineRunner:
    def __init__(self, context: StageContext, start_stage: str | None = None) -> None:
        self.context = context
        stages = build_stages()
        if start_stage is None:
            self.stages = stages
            return

        filtered: list[BaseStage] = []
        include = False
        for stage in stages:
            if stage.name == start_stage:
                include = True
            if include:
                filtered.append(stage)
        if not filtered:
            raise ValueError(f"Unknown start stage: {start_stage}")
        self.stages = filtered

    def run(self) -> PipelineSummary:
        stage_payloads = []
        for stage in self.stages:
            result = stage.run(self.context)
            stage_payloads.append(result.to_dict())
            save_json(self.context.outputs.metrics_path, {"stages": stage_payloads})
        return PipelineSummary(stages=stage_payloads)
