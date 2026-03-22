from __future__ import annotations

import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.acoustics import (
    cochlea_filterbank_stages,
    lif_encode_stages,
    simulate_echo_batch,
)
from stages.base import StageContext
from stages.experiments import _baseline_reference_params
from stages.improved_experiments import _evaluate_improved_model, _prepare_target_bundle
from stages.round_2_combined_all import (
    _combined_all_spec,
    _instantiate_combined_all_model,
    _prepare_profiled_experiment_data,
    _train_combined_all_model,
)
from stages.round_2_experiments import _augment_with_cartesian_metrics
from stages.training_improved_experiments import EnhancedTrainingConfig
from utils.common import (
    GlobalConfig,
    OutputPaths,
    format_float,
    save_cochlea_plot,
    save_grouped_bar_chart,
    save_json,
    save_waveform_and_spectrogram,
    seed_everything,
)


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _finalize(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_transmit_receive_plot(
    transmit: torch.Tensor,
    receive: torch.Tensor,
    sample_rate_hz: int,
    path: Path,
) -> None:
    transmit_np = _to_numpy(transmit)
    receive_np = _to_numpy(receive)
    time_axis_ms = np.arange(transmit_np.shape[-1]) / sample_rate_hz * 1_000.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_axis_ms, transmit_np, linewidth=1.0, color="#355070")
    axes[0].set_title("Transmit Chirp")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(time_axis_ms, receive_np, linewidth=1.0, color="#b56576")
    axes[1].set_title("Left-Ear Echo Input To Cochlea")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude")
    _finalize(path)


def _save_center_frequency_plot(center_frequencies_hz: torch.Tensor, path: Path) -> None:
    center_np = _to_numpy(center_frequencies_hz)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(center_np.shape[0]), center_np / 1_000.0, marker="o", linewidth=1.5)
    ax.set_title("Cochlea Center Frequencies")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Center Frequency (kHz)")
    ax.grid(True, alpha=0.25)
    _finalize(path)


def _save_filter_response_plot(
    frequencies_hz: torch.Tensor,
    filters: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    path: Path,
    *,
    spacing_mode: str = "log",
) -> None:
    freq_np = _to_numpy(frequencies_hz) / 1_000.0
    filters_np = _to_numpy(filters)
    center_np = _to_numpy(center_frequencies_hz)
    channel_indices = np.linspace(0, center_np.shape[0] - 1, 6, dtype=int)
    fig, ax = plt.subplots(figsize=(10, 5))
    for channel_index in channel_indices:
        ax.plot(
            freq_np,
            filters_np[channel_index],
            linewidth=1.5,
            label=f"ch {channel_index} ({center_np[channel_index] / 1000.0:.1f} kHz)",
        )
    ax.set_title(f"Representative {spacing_mode.capitalize()}-Spaced Filter Responses")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Gain")
    ax.set_xlim(freq_np.min(), freq_np.max())
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    _finalize(path)


def _save_filter_heatmap(
    frequencies_hz: torch.Tensor,
    filters: torch.Tensor,
    path: Path,
    *,
    spacing_mode: str = "log",
) -> None:
    freq_np = _to_numpy(frequencies_hz) / 1_000.0
    filters_np = _to_numpy(filters)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        filters_np,
        aspect="auto",
        origin="lower",
        extent=[freq_np.min(), freq_np.max(), 0, filters_np.shape[0] - 1],
        cmap="viridis",
    )
    ax.set_title(f"Full {spacing_mode.capitalize()}-Spaced Filterbank Response Matrix")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Channel")
    _finalize(path)


def _save_channel_examples(
    filtered: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    sample_rate_hz: int,
    path: Path,
) -> None:
    filtered_np = _to_numpy(filtered)
    center_np = _to_numpy(center_frequencies_hz)
    time_axis_ms = np.arange(filtered_np.shape[-1]) / sample_rate_hz * 1_000.0
    channel_indices = np.linspace(0, filtered_np.shape[0] - 1, 3, dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis, channel_index in zip(axes, channel_indices, strict=True):
        axis.plot(time_axis_ms, filtered_np[channel_index], linewidth=1.0)
        axis.set_ylabel("Amp")
        axis.set_title(f"Filtered Channel {channel_index} ({center_np[channel_index] / 1000.0:.1f} kHz)")
    axes[-1].set_xlabel("Time (ms)")
    _finalize(path)


def _save_rectify_smooth_plot(
    filtered: torch.Tensor,
    rectified: torch.Tensor,
    smoothed: torch.Tensor,
    cochleagram: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    sample_rate_hz: int,
    downsample: int,
    channel_index: int,
    path: Path,
) -> None:
    filtered_np = _to_numpy(filtered[channel_index])
    rectified_np = _to_numpy(rectified[channel_index])
    smoothed_np = _to_numpy(smoothed[channel_index])
    cochlea_np = _to_numpy(cochleagram[channel_index])
    center_khz = float(center_frequencies_hz[channel_index].item()) / 1_000.0
    time_ms = np.arange(filtered_np.shape[-1]) / sample_rate_hz * 1_000.0
    cochlea_time_ms = np.arange(cochlea_np.shape[-1]) / (sample_rate_hz / downsample) * 1_000.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(time_ms, filtered_np, linewidth=1.0, color="#355070")
    axes[0].set_title(f"Representative Channel Pipeline ({center_khz:.1f} kHz)")
    axes[0].set_ylabel("Filtered")
    axes[1].plot(time_ms, rectified_np, linewidth=1.0, color="#6d597a")
    axes[1].set_ylabel("Rectified")
    axes[2].plot(time_ms, smoothed_np, linewidth=1.0, color="#b56576")
    axes[2].set_ylabel("Smoothed")
    axes[3].plot(cochlea_time_ms, cochlea_np, linewidth=1.0, color="#e56b6f")
    axes[3].set_ylabel("Downsampled")
    axes[3].set_xlabel("Time (ms)")
    _finalize(path)


def _save_lowpass_kernel_plot(kernel: torch.Tensor, sample_rate_hz: int, path: Path) -> None:
    kernel_np = _to_numpy(kernel)
    time_ms = np.arange(kernel_np.shape[0]) / sample_rate_hz * 1_000.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ms, kernel_np, linewidth=1.5)
    ax.set_title("Envelope Low-Pass Kernel")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.25)
    _finalize(path)


def _save_membrane_plot(
    scaled_envelope: torch.Tensor,
    membrane: torch.Tensor,
    spikes: torch.Tensor,
    center_frequencies_hz: torch.Tensor,
    envelope_rate_hz: int,
    channel_index: int,
    path: Path,
) -> None:
    scaled_np = _to_numpy(scaled_envelope[channel_index])
    membrane_np = _to_numpy(membrane[channel_index])
    spikes_np = _to_numpy(spikes[channel_index])
    time_axis_ms = np.arange(scaled_np.shape[-1]) / envelope_rate_hz * 1_000.0
    center_khz = float(center_frequencies_hz[channel_index].item()) / 1_000.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_axis_ms, scaled_np, linewidth=1.0, label="scaled envelope")
    axes[0].plot(time_axis_ms, membrane_np, linewidth=1.0, label="membrane")
    axes[0].set_title(f"LIF Dynamics For Representative Channel ({center_khz:.1f} kHz)")
    axes[0].set_ylabel("State")
    axes[0].legend()
    spike_indices = np.nonzero(spikes_np > 0.0)[0]
    if spike_indices.size:
        axes[1].scatter(time_axis_ms[spike_indices], np.ones_like(spike_indices), s=12, c="black")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_yticks([1.0])
    axes[1].set_yticklabels(["spike"])
    axes[1].set_xlabel("Time (ms)")
    _finalize(path)


def _load_saved_round2_combined_baseline(outputs: OutputPaths) -> dict:
    path = outputs.root / "round_2_combined_all" / "result.json"
    if not path.exists():
        raise FileNotFoundError("Saved round-2 combined-all baseline result not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _human_band_config(base: GlobalConfig) -> GlobalConfig:
    payload = {**base.__dict__}
    payload.update(
        {
            "sample_rate_hz": 64_000,
            "chirp_start_hz": 18_000.0,
            "chirp_end_hz": 2_000.0,
            "cochlea_low_hz": 20.0,
            "cochlea_high_hz": 20_000.0,
        }
    )
    return GlobalConfig(**payload)


def _matched_human_band_config(base: GlobalConfig) -> GlobalConfig:
    payload = {**base.__dict__}
    payload.update(
        {
            "sample_rate_hz": 64_000,
            "chirp_start_hz": 18_000.0,
            "chirp_end_hz": 2_000.0,
            "cochlea_low_hz": 2_000.0,
            "cochlea_high_hz": 20_000.0,
        }
    )
    return GlobalConfig(**payload)


def _matched_human_dense_channel_config(base: GlobalConfig) -> GlobalConfig:
    payload = {**_matched_human_band_config(base).__dict__}
    payload.update({"num_cochlea_channels": 700})
    return GlobalConfig(**payload)


def _matched_human_mel_config(base: GlobalConfig) -> GlobalConfig:
    payload = {**_matched_human_band_config(base).__dict__}
    payload.update({"cochlea_spacing_mode": "mel"})
    return GlobalConfig(**payload)


def _save_spacing_center_frequency_plot(
    center_frequency_sets: dict[str, torch.Tensor],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, centers in center_frequency_sets.items():
        centers_np = _to_numpy(centers) / 1_000.0
        channel_axis = np.linspace(0.0, 1.0, centers_np.shape[0])
        ax.plot(channel_axis, centers_np, linewidth=1.5, label=label)
    ax.set_title("Matched Human-Band Center Frequency Comparison")
    ax.set_xlabel("Normalized Channel Index")
    ax.set_ylabel("Center Frequency (kHz)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _finalize(path)


def _run_band_system_experiment(
    band_config: GlobalConfig,
    outputs: OutputPaths,
    figure_dir: Path,
    *,
    result_name: str,
    signal_name: str,
    cochlea_name: str,
    title_prefix: str,
    params_override: dict[str, float | int | str] | None = None,
) -> dict:
    params, baseline_label = _baseline_reference_params(StageContext(config=band_config, device=torch.device("cpu"), outputs=outputs))
    if params_override:
        params = {**params, **params_override}
    model_num_frequency_channels = int(params["num_frequency_channels"])
    front_end_num_frequency_channels = int(params.get("front_end_num_frequency_channels", model_num_frequency_channels))
    effective_config = GlobalConfig(**{**band_config.__dict__, "num_cochlea_channels": front_end_num_frequency_channels})

    result_path = figure_dir / result_name
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        config_payload = payload.setdefault("config", {})
        config_payload["num_cochlea_channels"] = front_end_num_frequency_channels
        config_payload["model_num_frequency_channels"] = model_num_frequency_channels
        config_payload.setdefault("cochlea_spacing_mode", effective_config.cochlea_spacing_mode)
        save_json(result_path, payload)
        return payload

    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    spec = _combined_all_spec()
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=10,
        scheduler_patience=3,
    )

    seed_everything(effective_config.seed)
    prep_start = time.perf_counter()
    data, prep_profile = _prepare_profiled_experiment_data(context, params, training_config.dataset_mode)
    target_bundle = _prepare_target_bundle(data)
    prep_seconds = time.perf_counter() - prep_start

    model = _instantiate_combined_all_model(data, spec, params)
    train_start = time.perf_counter()
    train_result, uncertainty_module, training_breakdown = _train_combined_all_model(
        model,
        data,
        target_bundle,
        spec,
        training_config,
        params,
        dict(spec.training_overrides),
    )
    training_seconds = time.perf_counter() - train_start

    model.load_state_dict(train_result.best_state)
    if uncertainty_module is not None and train_result.best_auxiliary_state is not None:
        uncertainty_module.load_state_dict(train_result.best_auxiliary_state)

    eval_start = time.perf_counter()
    val_eval = _augment_with_cartesian_metrics(
        _evaluate_improved_model(model, data.val_batch, data.val_targets_raw, target_bundle, data.local_config)
    )
    test_eval = _augment_with_cartesian_metrics(
        _evaluate_improved_model(model, data.test_batch, data.test_targets_raw, target_bundle, data.local_config)
    )
    evaluation_seconds = time.perf_counter() - eval_start
    total_seconds = prep_seconds + training_seconds + evaluation_seconds

    example_batch = simulate_echo_batch(
        effective_config,
        torch.tensor([1.4], device=context.device),
        torch.tensor([18.0], device=context.device),
        torch.tensor([12.0], device=context.device),
        binaural=True,
        add_noise=False,
        include_elevation_cues=True,
    )
    filter_stages = cochlea_filterbank_stages(
        example_batch.receive[0, 0].unsqueeze(0),
        sample_rate_hz=effective_config.sample_rate_hz,
        num_channels=effective_config.num_cochlea_channels,
        low_hz=effective_config.cochlea_low_hz,
        high_hz=effective_config.cochlea_high_hz,
        spacing_mode=effective_config.cochlea_spacing_mode,
        filter_bandwidth_sigma=effective_config.filter_bandwidth_sigma,
        envelope_lowpass_hz=effective_config.envelope_lowpass_hz,
        downsample=effective_config.envelope_downsample,
    )
    lif_stages = lif_encode_stages(
        filter_stages["cochleagram"],
        threshold=effective_config.spike_threshold,
        beta=effective_config.spike_beta,
    )
    echo_start_ms = float(example_batch.delays_s[0, 0].item() * 1_000.0)
    echo_end_ms = echo_start_ms + effective_config.chirp_duration_s * 1_000.0
    xlim_ms = (
        max(0.0, echo_start_ms - 1.0),
        min(effective_config.signal_duration_s * 1_000.0, echo_end_ms + 1.0),
    )
    save_waveform_and_spectrogram(
        example_batch.receive[0, 0],
        effective_config.sample_rate_hz,
        figure_dir / signal_name,
        f"{title_prefix} Receive Waveform And Spectrogram",
    )
    save_cochlea_plot(
        filter_stages["cochleagram"][0],
        lif_stages["spikes"][0],
        effective_config.envelope_rate_hz,
        figure_dir / cochlea_name,
        f"{title_prefix} Cochleagram And Spike Raster",
        xlim_ms=xlim_ms,
    )

    result = {
        "baseline_label": baseline_label,
        "config": {
            "sample_rate_hz": effective_config.sample_rate_hz,
            "chirp_start_hz": effective_config.chirp_start_hz,
            "chirp_end_hz": effective_config.chirp_end_hz,
            "num_cochlea_channels": effective_config.num_cochlea_channels,
            "model_num_frequency_channels": model_num_frequency_channels,
            "cochlea_low_hz": effective_config.cochlea_low_hz,
            "cochlea_high_hz": effective_config.cochlea_high_hz,
            "cochlea_spacing_mode": effective_config.cochlea_spacing_mode,
            "envelope_rate_hz": effective_config.envelope_rate_hz,
        },
        "training_config": {
            "dataset_mode": training_config.dataset_mode,
            "max_epochs": training_config.max_epochs,
            "scheduler_patience": training_config.scheduler_patience,
        },
        "training": {
            "executed_epochs": train_result.executed_epochs,
            "best_epoch": train_result.best_epoch + 1,
            "stopped_early": train_result.stopped_early,
        },
        "timings": {
            "data_prep_seconds": format_float(prep_profile["total_profiled_seconds"]),
            "training_seconds": format_float(training_seconds),
            "evaluation_seconds": format_float(evaluation_seconds),
            "total_seconds": format_float(total_seconds),
        },
        "prep_profile": {key: format_float(value) if isinstance(value, float) else value for key, value in prep_profile.items()},
        "training_breakdown": {
            key: [format_float(item) for item in value] if isinstance(value, list) else format_float(value)
            for key, value in training_breakdown.items()
        },
        "val_metrics": {key: format_float(value) for key, value in val_eval.metrics.items()},
        "test_metrics": {key: format_float(value) for key, value in test_eval.metrics.items()},
        "artifacts": {
            "example_signal": str(figure_dir / signal_name),
            "cochleagram_spikes": str(figure_dir / cochlea_name),
        },
    }
    save_json(result_path, result)
    return result


def _run_human_band_system_experiment(base_config: GlobalConfig, outputs: OutputPaths, figure_dir: Path) -> dict:
    return _run_band_system_experiment(
        _human_band_config(base_config),
        outputs,
        figure_dir,
        result_name="human_band_experiment.json",
        signal_name="human_example_signal.png",
        cochlea_name="human_cochleagram_spikes.png",
        title_prefix="Human-Band Example",
    )


def _run_matched_human_band_system_experiment(base_config: GlobalConfig, outputs: OutputPaths, figure_dir: Path) -> dict:
    return _run_band_system_experiment(
        _matched_human_band_config(base_config),
        outputs,
        figure_dir,
        result_name="human_matched_band_experiment.json",
        signal_name="human_matched_example_signal.png",
        cochlea_name="human_matched_cochleagram_spikes.png",
        title_prefix="Matched Human-Band Example",
    )


def _run_matched_human_dense_channel_experiment(base_config: GlobalConfig, outputs: OutputPaths, figure_dir: Path) -> dict:
    dense_config = _matched_human_dense_channel_config(base_config)
    return _run_band_system_experiment(
        dense_config,
        outputs,
        figure_dir,
        result_name="human_matched_dense_700_cochlea_only_experiment.json",
        signal_name="human_matched_dense_700_cochlea_only_example_signal.png",
        cochlea_name="human_matched_dense_700_cochlea_only_cochleagram_spikes.png",
        title_prefix="Matched Human-Band Dense 700-Channel Example",
        params_override={"front_end_num_frequency_channels": 700},
    )


def _run_matched_human_mel_experiment(base_config: GlobalConfig, outputs: OutputPaths, figure_dir: Path) -> dict:
    return _run_band_system_experiment(
        _matched_human_mel_config(base_config),
        outputs,
        figure_dir,
        result_name="human_matched_mel_spacing_experiment.json",
        signal_name="human_matched_mel_spacing_example_signal.png",
        cochlea_name="human_matched_mel_spacing_cochleagram_spikes.png",
        title_prefix="Matched Human-Band Mel-Spaced Example",
    )


def run_cochlea_explained(config: GlobalConfig, outputs: OutputPaths) -> dict[str, str]:
    seed_everything(config.seed)
    figure_dir = outputs.root / "cochlea_explained"
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    radii = torch.tensor([1.4], device=device)
    azimuth = torch.tensor([18.0], device=device)
    elevation = torch.tensor([12.0], device=device)
    acoustic_batch = simulate_echo_batch(
        config,
        radii,
        azimuth,
        elevation,
        binaural=True,
        add_noise=False,
        include_elevation_cues=True,
    )
    transmit = acoustic_batch.transmit[0]
    receive_left = acoustic_batch.receive[0, 0]
    filter_stages = cochlea_filterbank_stages(
        receive_left.unsqueeze(0),
        sample_rate_hz=config.sample_rate_hz,
        num_channels=config.num_cochlea_channels,
        low_hz=config.cochlea_low_hz,
        high_hz=config.cochlea_high_hz,
        spacing_mode=config.cochlea_spacing_mode,
        filter_bandwidth_sigma=config.filter_bandwidth_sigma,
        envelope_lowpass_hz=config.envelope_lowpass_hz,
        downsample=config.envelope_downsample,
    )
    lif_stages = lif_encode_stages(
        filter_stages["cochleagram"],
        threshold=config.spike_threshold,
        beta=config.spike_beta,
    )

    representative_channel = int(torch.argmin(torch.abs(filter_stages["center_frequencies_hz"] - 45_000.0)).item())
    left_echo_start_ms = float(acoustic_batch.delays_s[0, 0].item() * 1_000.0)
    left_echo_end_ms = left_echo_start_ms + config.chirp_duration_s * 1_000.0
    xlim_ms = (
        max(0.0, left_echo_start_ms - 1.0),
        min(config.signal_duration_s * 1_000.0, left_echo_end_ms + 1.0),
    )

    save_waveform_and_spectrogram(
        receive_left,
        config.sample_rate_hz,
        figure_dir / "example_signal.png",
        "Example Receive Waveform And Spectrogram",
    )
    _save_transmit_receive_plot(transmit, receive_left, config.sample_rate_hz, figure_dir / "transmit_receive.png")
    _save_center_frequency_plot(filter_stages["center_frequencies_hz"], figure_dir / "center_frequencies.png")
    _save_filter_response_plot(
        filter_stages["frequencies_hz"],
        filter_stages["filters"],
        filter_stages["center_frequencies_hz"],
        figure_dir / "filter_responses.png",
        spacing_mode=config.cochlea_spacing_mode,
    )
    _save_filter_heatmap(
        filter_stages["frequencies_hz"],
        filter_stages["filters"],
        figure_dir / "filter_heatmap.png",
        spacing_mode=config.cochlea_spacing_mode,
    )
    _save_channel_examples(
        filter_stages["filtered"][0],
        filter_stages["center_frequencies_hz"],
        config.sample_rate_hz,
        figure_dir / "filtered_channels.png",
    )
    _save_rectify_smooth_plot(
        filter_stages["filtered"][0],
        filter_stages["rectified"][0],
        filter_stages["smoothed"][0],
        filter_stages["cochleagram"][0],
        filter_stages["center_frequencies_hz"],
        config.sample_rate_hz,
        config.envelope_downsample,
        representative_channel,
        figure_dir / "channel_pipeline.png",
    )
    _save_lowpass_kernel_plot(filter_stages["lowpass_kernel"], config.sample_rate_hz, figure_dir / "lowpass_kernel.png")
    save_cochlea_plot(
        filter_stages["cochleagram"][0],
        lif_stages["spikes"][0],
        config.envelope_rate_hz,
        figure_dir / "cochleagram_spikes.png",
        "Final Cochleagram And Spike Raster",
        xlim_ms=xlim_ms,
    )
    _save_membrane_plot(
        lif_stages["scaled_envelope"][0],
        lif_stages["membrane"][0],
        lif_stages["spikes"][0],
        filter_stages["center_frequencies_hz"],
        config.envelope_rate_hz,
        representative_channel,
        figure_dir / "membrane_spikes.png",
    )

    baseline_system_result = _load_saved_round2_combined_baseline(outputs)
    human_band_result = _run_human_band_system_experiment(config, outputs, figure_dir)
    matched_human_band_result = _run_matched_human_band_system_experiment(config, outputs, figure_dir)
    matched_dense_700_result = _run_matched_human_dense_channel_experiment(config, outputs, figure_dir)
    matched_mel_result = _run_matched_human_mel_experiment(config, outputs, figure_dir)

    matched_baseline_center_frequencies = cochlea_filterbank_stages(
        torch.zeros(1, _matched_human_band_config(config).signal_samples, dtype=torch.float32),
        sample_rate_hz=_matched_human_band_config(config).sample_rate_hz,
        num_channels=int(matched_human_band_result["config"]["num_cochlea_channels"]),
        low_hz=_matched_human_band_config(config).cochlea_low_hz,
        high_hz=_matched_human_band_config(config).cochlea_high_hz,
        spacing_mode=str(matched_human_band_result["config"]["cochlea_spacing_mode"]),
        filter_bandwidth_sigma=_matched_human_band_config(config).filter_bandwidth_sigma,
        envelope_lowpass_hz=_matched_human_band_config(config).envelope_lowpass_hz,
        downsample=_matched_human_band_config(config).envelope_downsample,
    )["center_frequencies_hz"]
    matched_dense_center_frequencies = cochlea_filterbank_stages(
        torch.zeros(1, _matched_human_dense_channel_config(config).signal_samples, dtype=torch.float32),
        sample_rate_hz=_matched_human_dense_channel_config(config).sample_rate_hz,
        num_channels=int(matched_dense_700_result["config"]["num_cochlea_channels"]),
        low_hz=_matched_human_dense_channel_config(config).cochlea_low_hz,
        high_hz=_matched_human_dense_channel_config(config).cochlea_high_hz,
        spacing_mode=str(matched_dense_700_result["config"]["cochlea_spacing_mode"]),
        filter_bandwidth_sigma=_matched_human_dense_channel_config(config).filter_bandwidth_sigma,
        envelope_lowpass_hz=_matched_human_dense_channel_config(config).envelope_lowpass_hz,
        downsample=_matched_human_dense_channel_config(config).envelope_downsample,
    )["center_frequencies_hz"]
    matched_mel_center_frequencies = cochlea_filterbank_stages(
        torch.zeros(1, _matched_human_mel_config(config).signal_samples, dtype=torch.float32),
        sample_rate_hz=_matched_human_mel_config(config).sample_rate_hz,
        num_channels=int(matched_mel_result["config"]["num_cochlea_channels"]),
        low_hz=_matched_human_mel_config(config).cochlea_low_hz,
        high_hz=_matched_human_mel_config(config).cochlea_high_hz,
        spacing_mode=str(matched_mel_result["config"]["cochlea_spacing_mode"]),
        filter_bandwidth_sigma=_matched_human_mel_config(config).filter_bandwidth_sigma,
        envelope_lowpass_hz=_matched_human_mel_config(config).envelope_lowpass_hz,
        downsample=_matched_human_mel_config(config).envelope_downsample,
    )["center_frequencies_hz"]
    _save_spacing_center_frequency_plot(
        {
            "Matched baseline log": matched_baseline_center_frequencies,
            "Matched log 700 ch": matched_dense_center_frequencies,
            "Matched mel": matched_mel_center_frequencies,
        },
        figure_dir / "matched_channel_spacing_centers.png",
    )
    save_grouped_bar_chart(
        ["Prep", "Training", "Evaluation", "Total"],
        {
            "Ultrasonic baseline": [
                float(baseline_system_result["timings"]["data_prep_seconds"]),
                float(baseline_system_result["timings"]["training_seconds"]),
                float(baseline_system_result["timings"]["evaluation_seconds"]),
                float(baseline_system_result["timings"]["total_seconds"]),
            ],
            "Human-band analogue": [
                float(human_band_result["timings"]["data_prep_seconds"]),
                float(human_band_result["timings"]["training_seconds"]),
                float(human_band_result["timings"]["evaluation_seconds"]),
                float(human_band_result["timings"]["total_seconds"]),
            ],
            "Matched human-band": [
                float(matched_human_band_result["timings"]["data_prep_seconds"]),
                float(matched_human_band_result["timings"]["training_seconds"]),
                float(matched_human_band_result["timings"]["evaluation_seconds"]),
                float(matched_human_band_result["timings"]["total_seconds"]),
            ],
        },
        figure_dir / "bandwidth_runtime_comparison.png",
        "Bandwidth / Sample Rate Runtime Comparison",
        ylabel="Seconds",
    )
    save_grouped_bar_chart(
        ["Combined", "Distance", "Azimuth", "Elevation", "Euclidean"],
        {
            "Ultrasonic baseline": [
                float(baseline_system_result["test_metrics"]["combined_error"]),
                float(baseline_system_result["test_metrics"]["distance_mae_m"]),
                float(baseline_system_result["test_metrics"]["azimuth_mae_deg"]),
                float(baseline_system_result["test_metrics"]["elevation_mae_deg"]),
                float(baseline_system_result["test_metrics"]["euclidean_error_m"]),
            ],
            "Human-band analogue": [
                float(human_band_result["test_metrics"]["combined_error"]),
                float(human_band_result["test_metrics"]["distance_mae_m"]),
                float(human_band_result["test_metrics"]["azimuth_mae_deg"]),
                float(human_band_result["test_metrics"]["elevation_mae_deg"]),
                float(human_band_result["test_metrics"]["euclidean_error_m"]),
            ],
            "Matched human-band": [
                float(matched_human_band_result["test_metrics"]["combined_error"]),
                float(matched_human_band_result["test_metrics"]["distance_mae_m"]),
                float(matched_human_band_result["test_metrics"]["azimuth_mae_deg"]),
                float(matched_human_band_result["test_metrics"]["elevation_mae_deg"]),
                float(matched_human_band_result["test_metrics"]["euclidean_error_m"]),
            ],
        },
        figure_dir / "bandwidth_accuracy_comparison.png",
        "Bandwidth / Sample Rate Accuracy Comparison",
        ylabel="Error",
    )
    save_grouped_bar_chart(
        ["Prep", "Training", "Evaluation", "Total"],
        {
            "Matched log baseline": [
                float(matched_human_band_result["timings"]["data_prep_seconds"]),
                float(matched_human_band_result["timings"]["training_seconds"]),
                float(matched_human_band_result["timings"]["evaluation_seconds"]),
                float(matched_human_band_result["timings"]["total_seconds"]),
            ],
            "Matched log 700 ch": [
                float(matched_dense_700_result["timings"]["data_prep_seconds"]),
                float(matched_dense_700_result["timings"]["training_seconds"]),
                float(matched_dense_700_result["timings"]["evaluation_seconds"]),
                float(matched_dense_700_result["timings"]["total_seconds"]),
            ],
            "Matched mel": [
                float(matched_mel_result["timings"]["data_prep_seconds"]),
                float(matched_mel_result["timings"]["training_seconds"]),
                float(matched_mel_result["timings"]["evaluation_seconds"]),
                float(matched_mel_result["timings"]["total_seconds"]),
            ],
        },
        figure_dir / "matched_channel_spacing_runtime_comparison.png",
        "Matched Human-Band Channel / Spacing Runtime Comparison",
        ylabel="Seconds",
    )
    save_grouped_bar_chart(
        ["Combined", "Distance", "Azimuth", "Elevation", "Euclidean"],
        {
            "Matched log baseline": [
                float(matched_human_band_result["test_metrics"]["combined_error"]),
                float(matched_human_band_result["test_metrics"]["distance_mae_m"]),
                float(matched_human_band_result["test_metrics"]["azimuth_mae_deg"]),
                float(matched_human_band_result["test_metrics"]["elevation_mae_deg"]),
                float(matched_human_band_result["test_metrics"]["euclidean_error_m"]),
            ],
            "Matched log 700 ch": [
                float(matched_dense_700_result["test_metrics"]["combined_error"]),
                float(matched_dense_700_result["test_metrics"]["distance_mae_m"]),
                float(matched_dense_700_result["test_metrics"]["azimuth_mae_deg"]),
                float(matched_dense_700_result["test_metrics"]["elevation_mae_deg"]),
                float(matched_dense_700_result["test_metrics"]["euclidean_error_m"]),
            ],
            "Matched mel": [
                float(matched_mel_result["test_metrics"]["combined_error"]),
                float(matched_mel_result["test_metrics"]["distance_mae_m"]),
                float(matched_mel_result["test_metrics"]["azimuth_mae_deg"]),
                float(matched_mel_result["test_metrics"]["elevation_mae_deg"]),
                float(matched_mel_result["test_metrics"]["euclidean_error_m"]),
            ],
        },
        figure_dir / "matched_channel_spacing_accuracy_comparison.png",
        "Matched Human-Band Channel / Spacing Accuracy Comparison",
        ylabel="Error",
    )

    runtime_speedup = float(baseline_system_result["timings"]["total_seconds"]) / max(
        float(human_band_result["timings"]["total_seconds"]),
        1e-6,
    )
    prep_speedup = float(baseline_system_result["timings"]["data_prep_seconds"]) / max(
        float(human_band_result["timings"]["data_prep_seconds"]),
        1e-6,
    )
    training_speedup = float(baseline_system_result["timings"]["training_seconds"]) / max(
        float(human_band_result["timings"]["training_seconds"]),
        1e-6,
    )
    matched_runtime_speedup = float(baseline_system_result["timings"]["total_seconds"]) / max(
        float(matched_human_band_result["timings"]["total_seconds"]),
        1e-6,
    )
    matched_prep_speedup = float(baseline_system_result["timings"]["data_prep_seconds"]) / max(
        float(matched_human_band_result["timings"]["data_prep_seconds"]),
        1e-6,
    )
    matched_training_speedup = float(baseline_system_result["timings"]["training_seconds"]) / max(
        float(matched_human_band_result["timings"]["training_seconds"]),
        1e-6,
    )
    dense_runtime_multiplier = float(matched_dense_700_result["timings"]["total_seconds"]) / max(
        float(matched_human_band_result["timings"]["total_seconds"]),
        1e-6,
    )
    dense_prep_multiplier = float(matched_dense_700_result["timings"]["data_prep_seconds"]) / max(
        float(matched_human_band_result["timings"]["data_prep_seconds"]),
        1e-6,
    )
    dense_training_multiplier = float(matched_dense_700_result["timings"]["training_seconds"]) / max(
        float(matched_human_band_result["timings"]["training_seconds"]),
        1e-6,
    )
    mel_runtime_multiplier = float(matched_mel_result["timings"]["total_seconds"]) / max(
        float(matched_human_band_result["timings"]["total_seconds"]),
        1e-6,
    )
    mel_prep_multiplier = float(matched_mel_result["timings"]["data_prep_seconds"]) / max(
        float(matched_human_band_result["timings"]["data_prep_seconds"]),
        1e-6,
    )
    mel_training_multiplier = float(matched_mel_result["timings"]["training_seconds"]) / max(
        float(matched_human_band_result["timings"]["training_seconds"]),
        1e-6,
    )

    report_lines = [
        "# Cochlea Explained",
        "",
        "## Overview",
        "",
        "This document describes the current fixed cochlea front end used by the localisation system. The example figures are generated from one clean left-ear echo scene so the transformations are easy to inspect.",
        "",
        "Example scene:",
        f"- Distance: `{radii.item():.2f} m`",
        f"- Azimuth: `{azimuth.item():.1f} deg`",
        f"- Elevation: `{elevation.item():.1f} deg`",
        "- Binaural simulation: `on`",
        "- Noise: `off` for clarity",
        "",
        "## Pipeline",
        "",
        "```mermaid",
        "graph TD",
        "    A[Receive waveform] --> B[FFT]",
        "    B --> C[Log-spaced Gaussian filterbank]",
        "    C --> D[Inverse FFT per channel]",
        "    D --> E[Half-wave rectification]",
        "    E --> F[Low-pass envelope smoothing]",
        "    F --> G[Temporal downsampling]",
        "    G --> H[LIF spike encoder]",
        "    H --> I[Transmit / receive spike tensors]",
        "```",
        "",
        "## Current Fixed Parameters",
        "",
        "| Parameter | Value | Role |",
        "| --- | --- | --- |",
        f"| `sample_rate_hz` | `{config.sample_rate_hz}` | Raw waveform sampling rate |",
        f"| `num_cochlea_channels` | `{config.num_cochlea_channels}` | Number of frequency channels |",
        f"| `cochlea_low_hz` | `{config.cochlea_low_hz:.0f}` | Lowest cochlear center frequency |",
        f"| `cochlea_high_hz` | `{config.cochlea_high_hz:.0f}` | Highest cochlear center frequency |",
        f"| `filter_bandwidth_sigma` | `{config.filter_bandwidth_sigma:.3f}` | Width of the Gaussian log-frequency filters |",
        f"| `envelope_lowpass_hz` | `{config.envelope_lowpass_hz:.0f}` | Envelope smoothing cutoff proxy |",
        f"| `envelope_downsample` | `{config.envelope_downsample}` | Temporal downsampling factor before spiking |",
        f"| `spike_threshold` | `{config.spike_threshold:.2f}` | LIF firing threshold |",
        f"| `spike_beta` | `{config.spike_beta:.2f}` | LIF leak factor |",
        "",
        "## 1. Input Signal",
        "",
        "The cochlea receives the left-ear echo waveform. The transmitted chirp is shown alongside it for reference.",
        "",
        "![Input spectrogram](cochlea_explained/example_signal.png)",
        "![Transmit vs receive](cochlea_explained/transmit_receive.png)",
        "",
        "## 2. Log-Spaced Filterbank",
        "",
        "The raw waveform is transformed into the frequency domain, multiplied by a bank of Gaussian filters in log-frequency space, and returned to the time domain channel by channel.",
        "",
        "![Center frequencies](cochlea_explained/center_frequencies.png)",
        "![Filter responses](cochlea_explained/filter_responses.png)",
        "![Filter heatmap](cochlea_explained/filter_heatmap.png)",
        "",
        "## 3. Per-Channel Filtered Signals",
        "",
        "After inverse FFT, each channel contains a band-limited version of the original waveform. Low, middle, and high channels respond at different parts of the chirp.",
        "",
        "![Filtered channels](cochlea_explained/filtered_channels.png)",
        "",
        "## 4. Rectification, Smoothing, And Downsampling",
        "",
        "Each channel is half-wave rectified, smoothed with a Hann low-pass kernel, and then downsampled. The downsampled smoothed envelope is the actual input to the spike encoder.",
        "",
        "![Channel pipeline](cochlea_explained/channel_pipeline.png)",
        "![Low-pass kernel](cochlea_explained/lowpass_kernel.png)",
        "",
        "## 5. LIF Spike Encoding",
        "",
        "The smoothed envelope is normalized, integrated through a fixed LIF neuron per channel, thresholded, and reset by subtraction. Spikes are therefore driven by envelope peaks in each frequency band.",
        "",
        "![Membrane and spikes](cochlea_explained/membrane_spikes.png)",
        "",
        "## 6. Final Cochleagram And Spike Raster",
        "",
        "The final cochleagram is the smoothed, downsampled envelope across all channels. The spike raster is the binary output that the rest of the localisation system consumes. This figure is zoomed to the actual echo window so the short FM sweep is visible on the millisecond axis.",
        "",
        "![Cochleagram and spikes](cochlea_explained/cochleagram_spikes.png)",
        "",
        "## Interface To The Rest Of The Model",
        "",
        "The current barrier is after spike generation:",
        "",
        "- transmit spikes: shape `[batch, channel, time]`",
        "- receive spikes: shape `[batch, ear, channel, time]`",
        "",
        "Everything downstream assumes those spike tensors already exist. That makes the current cochlea replaceable, but the easiest swap is another cochlea that preserves the same spike-tensor contract and envelope-rate time base.",
        "",
        "## Current Interpretation",
        "",
        "This cochlea is fixed and hand-designed. It is not currently trainable. The expensive part is the fixed FFT filterbank plus spike conversion, not the later handcrafted pathway feature extraction.",
        "",
        "## Bandwidth And Sampling Experiment",
        "",
        "This comparison tests the effect of lowering the acoustic and cochlear bandwidth and reducing the sampling rate on the full short-data combined-all localisation system.",
        "",
        "Protocol:",
        "- Ultrasonic baseline: saved short-data round-2 combined-all result using the existing `20 kHz to 90 kHz` cochlea, `80 kHz to 20 kHz` chirp, and `256 kHz` sample rate.",
        "- Human-band analogue: fresh rerun of the same short-data combined-all model with cochlea range `20 Hz to 20 kHz`, sample rate `64 kHz`, and a practical downward FM chirp `18 kHz to 2 kHz`.",
        "- Matched human-band analogue: second fresh rerun with the same `64 kHz` sample rate and `18 kHz to 2 kHz` chirp, but with the cochlea restricted to the active signal band `2 kHz to 20 kHz`.",
        "- The lower chirp edge was not set literally to `20 Hz` because a `3 ms` chirp cannot meaningfully encode 20 Hz content; one 20 Hz period is `50 ms`.",
        "- Same dataset size and training budget: `700 / 150 / 150`, `10` epochs, one thread, no Optuna retuning.",
        "- The original wide human-band result is retained below for comparison; the new matched-band result is additional.",
        "",
        "Runtime comparison:",
        f"- Ultrasonic baseline total: `{float(baseline_system_result['timings']['total_seconds']):.2f} s`",
        f"- Human-band analogue total: `{float(human_band_result['timings']['total_seconds']):.2f} s`",
        f"- Overall speedup: `{runtime_speedup:.2f}x`",
        f"- Prep speedup: `{prep_speedup:.2f}x`",
        f"- Training speedup: `{training_speedup:.2f}x`",
        f"- Matched human-band total: `{float(matched_human_band_result['timings']['total_seconds']):.2f} s`",
        f"- Matched overall speedup: `{matched_runtime_speedup:.2f}x`",
        f"- Matched prep speedup: `{matched_prep_speedup:.2f}x`",
        f"- Matched training speedup: `{matched_training_speedup:.2f}x`",
        "",
        "Accuracy comparison:",
        f"- Ultrasonic combined error: `{float(baseline_system_result['test_metrics']['combined_error']):.4f}`",
        f"- Human-band combined error: `{float(human_band_result['test_metrics']['combined_error']):.4f}`",
        f"- Matched human-band combined error: `{float(matched_human_band_result['test_metrics']['combined_error']):.4f}`",
        f"- Ultrasonic distance / azimuth / elevation: `{float(baseline_system_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(baseline_system_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(baseline_system_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
        f"- Human-band distance / azimuth / elevation: `{float(human_band_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(human_band_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(human_band_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
        f"- Matched human-band distance / azimuth / elevation: `{float(matched_human_band_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(matched_human_band_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(matched_human_band_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
        f"- Ultrasonic Euclidean error: `{float(baseline_system_result['test_metrics']['euclidean_error_m']):.4f} m`",
        f"- Human-band Euclidean error: `{float(human_band_result['test_metrics']['euclidean_error_m']):.4f} m`",
        f"- Matched human-band Euclidean error: `{float(matched_human_band_result['test_metrics']['euclidean_error_m']):.4f} m`",
        "",
        "Interpretation:",
        "- This is not only a cochlea-bandwidth change. It also reduces raw waveform sampling resolution and moves the chirp into a much lower carrier band.",
        "- The comparison therefore measures the practical effect of a lower-bandwidth, lower-sample-rate auditory front end on the full localisation stack.",
        "- The matched human-band variant specifically tests whether excluding irrelevant sub-2 kHz channels helps once the signal itself only occupies `2 kHz to 18 kHz`.",
        "- Because the downstream model was not retuned for either human-band configuration, both should be treated as direct transfer tests rather than optimized redesigns.",
        "",
        "![Bandwidth runtime comparison](cochlea_explained/bandwidth_runtime_comparison.png)",
        "![Bandwidth accuracy comparison](cochlea_explained/bandwidth_accuracy_comparison.png)",
        "![Human-band example signal](cochlea_explained/human_example_signal.png)",
        "![Human-band cochleagram](cochlea_explained/human_cochleagram_spikes.png)",
        "![Matched human-band example signal](cochlea_explained/human_matched_example_signal.png)",
        "![Matched human-band cochleagram](cochlea_explained/human_matched_cochleagram_spikes.png)",
        "",
        "## Channel Count And Spacing Experiments",
        "",
        "This comparison keeps the matched human-band setup as the baseline and changes one cochlea design variable at a time.",
        "",
        "Protocol:",
        f"- Matched human-band baseline: `{int(matched_human_band_result['config']['num_cochlea_channels'])}` cochlea channels, `{matched_human_band_result['config']['cochlea_spacing_mode']}` spacing, `2 kHz to 20 kHz` cochlea range, downstream model width `{int(matched_human_band_result['config']['model_num_frequency_channels'])}`.",
        f"- Dense-channel variant: same matched human-band setup, but increase only the cochlea front end to `{int(matched_dense_700_result['config']['num_cochlea_channels'])}` channels while keeping the downstream model width fixed at `{int(matched_dense_700_result['config']['model_num_frequency_channels'])}` via channel-axis compression at the cochlea boundary.",
        "- Mel-spacing variant: same matched human-band setup and channel count, but replace the log-spaced cochlea with a mel-spaced cochlea.",
        "- Same dataset and training budget for all three: `700 / 150 / 150`, `10` epochs, one thread, no Optuna retuning.",
        "",
        "Runtime comparison against the matched human-band baseline:",
        f"- Matched log baseline total: `{float(matched_human_band_result['timings']['total_seconds']):.2f} s`",
        f"- Matched log 700-channel total: `{float(matched_dense_700_result['timings']['total_seconds']):.2f} s` (`{dense_runtime_multiplier:.2f}x` baseline)",
        f"- Matched log 700-channel prep / training multipliers: `{dense_prep_multiplier:.2f}x`, `{dense_training_multiplier:.2f}x`",
        f"- Matched mel total: `{float(matched_mel_result['timings']['total_seconds']):.2f} s` (`{mel_runtime_multiplier:.2f}x` baseline)",
        f"- Matched mel prep / training multipliers: `{mel_prep_multiplier:.2f}x`, `{mel_training_multiplier:.2f}x`",
        "",
        "Accuracy comparison against the matched human-band baseline:",
        f"- Matched log baseline combined / Euclidean: `{float(matched_human_band_result['test_metrics']['combined_error']):.4f}`, `{float(matched_human_band_result['test_metrics']['euclidean_error_m']):.4f} m`",
        f"- Matched log baseline distance / azimuth / elevation: `{float(matched_human_band_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(matched_human_band_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(matched_human_band_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
        f"- Matched log 700-channel combined / Euclidean: `{float(matched_dense_700_result['test_metrics']['combined_error']):.4f}`, `{float(matched_dense_700_result['test_metrics']['euclidean_error_m']):.4f} m`",
        f"- Matched log 700-channel distance / azimuth / elevation: `{float(matched_dense_700_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(matched_dense_700_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(matched_dense_700_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
        f"- Matched mel combined / Euclidean: `{float(matched_mel_result['test_metrics']['combined_error']):.4f}`, `{float(matched_mel_result['test_metrics']['euclidean_error_m']):.4f} m`",
        f"- Matched mel distance / azimuth / elevation: `{float(matched_mel_result['test_metrics']['distance_mae_m']):.4f} m`, `{float(matched_mel_result['test_metrics']['azimuth_mae_deg']):.4f} deg`, `{float(matched_mel_result['test_metrics']['elevation_mae_deg']):.4f} deg`",
        "",
        "Interpretation:",
        "- The `700`-channel test isolates the cost and benefit of much finer cochlear frequency resolution under the same matched human-band chirp and training budget, without also widening the downstream combined model.",
        "- The mel-spacing test isolates a change in channel placement along frequency while keeping the rest of the front end and downstream model structure fixed.",
        "- In this implementation, the mel-spaced bank uses mel-spaced centers with the same Gaussian FFT filter construction and a bandwidth rescaling so filter width stays comparable across the covered band.",
        "",
        "![Matched center frequencies](cochlea_explained/matched_channel_spacing_centers.png)",
        "![Matched runtime comparison](cochlea_explained/matched_channel_spacing_runtime_comparison.png)",
        "![Matched accuracy comparison](cochlea_explained/matched_channel_spacing_accuracy_comparison.png)",
    ]
    report_path = outputs.root / "cochlea_explained.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "report": str(report_path),
        "figure_dir": str(figure_dir),
    }
