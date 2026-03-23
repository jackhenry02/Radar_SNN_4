from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MAX_THREADS = os.environ.get("RADAR_SNN_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_MAX_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_MAX_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", DEFAULT_MAX_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", DEFAULT_MAX_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", DEFAULT_MAX_THREADS)
os.environ.setdefault("MPLCONFIGDIR", "outputs/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "outputs/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch


@dataclass
class GlobalConfig:
    sample_rate_hz: int = 256_000
    chirp_duration_s: float = 0.003
    signal_duration_s: float = 0.022
    chirp_start_hz: float = 80_000.0
    chirp_end_hz: float = 20_000.0
    speed_of_sound_m_s: float = 343.0
    max_range_m: float = 2.5
    min_range_m: float = 0.5
    ear_spacing_m: float = 0.03
    num_cochlea_channels: int = 24
    cochlea_low_hz: float = 20_000.0
    cochlea_high_hz: float = 90_000.0
    cochlea_spacing_mode: str = "log"
    filter_bandwidth_sigma: float = 0.16
    envelope_lowpass_hz: float = 1_800.0
    envelope_downsample: int = 4
    spike_threshold: float = 0.42
    spike_beta: float = 0.88
    normalize_spike_envelope: bool = True
    transmit_gain: float = 1.0
    noise_std: float = 0.008
    jitter_std_s: float = 2.5e-5
    head_shadow_strength: float = 0.32
    elevation_spectral_strength: float = 0.75
    elevation_cue_mode: str = "slope"
    elevation_notch_strength: float = 1.8
    elevation_notch_width: float = 0.065
    elevation_notch_center_min: float = 0.18
    elevation_notch_center_max: float = 0.82
    azimuth_cue_mode: str = "none"
    azimuth_spectral_strength: float = 0.65
    azimuth_notch_strength: float = 1.4
    azimuth_notch_width: float = 0.07
    azimuth_notch_center_min: float = 0.18
    azimuth_notch_center_max: float = 0.82
    azimuth_notch_mirror_across_band: bool = False
    seed: int = 7

    @property
    def chirp_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.chirp_duration_s))

    @property
    def signal_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.signal_duration_s))

    @property
    def envelope_rate_hz(self) -> int:
        return self.sample_rate_hz // self.envelope_downsample

    @property
    def max_delay_s(self) -> float:
        return 2.0 * self.max_range_m / self.speed_of_sound_m_s


@dataclass
class OutputPaths:
    root: Path
    figures: Path
    logs: Path
    metrics_path: Path

    @classmethod
    def create(cls, root: str | Path) -> "OutputPaths":
        root_path = Path(root)
        figures = root_path / "figures"
        logs = root_path / "logs"
        figures.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root_path,
            figures=figures,
            logs=logs,
            metrics_path=root_path / "metrics.json",
        )

    def stage_dir(self, stage_name: str) -> Path:
        stage_path = self.figures / stage_name
        stage_path.mkdir(parents=True, exist_ok=True)
        return stage_path


@dataclass
class AttemptRecord:
    attempt: int
    success: bool
    score: float
    metrics: dict[str, Any]
    notes: str = ""


@dataclass
class StageResult:
    name: str
    success: bool
    best_attempt: int
    best_score: float
    best_metrics: dict[str, Any]
    attempts: list[AttemptRecord] = field(default_factory=list)
    failure_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "best_attempt": self.best_attempt,
            "best_score": self.best_score,
            "best_metrics": tensor_to_python(self.best_metrics),
            "attempts": [tensor_to_python(asdict(record)) for record in self.attempts],
            "failure_report": tensor_to_python(self.failure_report),
        }


def tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: tensor_to_python(val) for key, val in value.items()}
    if isinstance(value, list):
        return [tensor_to_python(item) for item in value]
    if isinstance(value, tuple):
        return [tensor_to_python(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(tensor_to_python(payload), handle, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    forced_device = os.environ.get("RADAR_SNN_DEVICE")
    if forced_device:
        return torch.device(forced_device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def limit_backend_resources(max_threads: int | None = None) -> int:
    requested = max_threads or int(os.environ.get("RADAR_SNN_MAX_THREADS", "2"))
    bounded = max(1, requested)
    torch.set_num_threads(bounded)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return bounded


def distance_mae(prediction_m: torch.Tensor, target_m: torch.Tensor) -> float:
    return torch.mean(torch.abs(prediction_m - target_m)).item()


def angular_mae(prediction_deg: torch.Tensor, target_deg: torch.Tensor) -> float:
    return torch.mean(torch.abs(prediction_deg - target_deg)).item()


def combined_localisation_error(
    distance_pred_m: torch.Tensor,
    distance_true_m: torch.Tensor,
    azimuth_pred_deg: torch.Tensor,
    azimuth_true_deg: torch.Tensor,
    elevation_pred_deg: torch.Tensor,
    elevation_true_deg: torch.Tensor,
    max_range_m: float,
) -> float:
    distance_term = (distance_pred_m - distance_true_m) / max_range_m
    azimuth_term = (azimuth_pred_deg - azimuth_true_deg) / 180.0
    elevation_term = (elevation_pred_deg - elevation_true_deg) / 90.0
    combined = torch.sqrt(distance_term.square() + azimuth_term.square() + elevation_term.square())
    return combined.mean().item()


def _to_numpy(values: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _finalize_figure(path: str | Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_waveform_and_spectrogram(
    signal: torch.Tensor,
    sample_rate_hz: int,
    path: str | Path,
    title: str,
) -> None:
    waveform = _to_numpy(signal)
    time_axis_ms = np.arange(waveform.shape[-1]) / sample_rate_hz * 1_000.0
    signal_duration_s = waveform.shape[-1] / max(sample_rate_hz, 1)
    target_window_s = min(0.0010, max(0.0005, signal_duration_s / 32.0))
    desired_window_samples = max(32, int(round(sample_rate_hz * target_window_s)))
    if desired_window_samples >= waveform.shape[-1]:
        nfft = max(32, min(int(waveform.shape[-1]), desired_window_samples))
    else:
        nfft = 1 << int(math.ceil(math.log2(desired_window_samples)))
        nfft = min(nfft, int(waveform.shape[-1]))
    noverlap = min(nfft - 1, max(0, int(round(0.875 * nfft))))
    pad_to = max(256, 4 * nfft)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(time_axis_ms, waveform, linewidth=1.0)
    axes[0].set_title(title)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    _, _, _, image = axes[1].specgram(
        waveform,
        Fs=sample_rate_hz,
        NFFT=nfft,
        noverlap=noverlap,
        pad_to=pad_to,
        cmap="magma",
    )
    if hasattr(image, "set_interpolation"):
        image.set_interpolation("bilinear")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1_000.0:.1f}"))
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Frequency (Hz)")
    _finalize_figure(path)


def save_cochlea_plot(
    cochleagram: torch.Tensor,
    spikes: torch.Tensor,
    sample_rate_hz: int,
    path: str | Path,
    title: str,
    xlim_ms: tuple[float, float] | None = None,
) -> None:
    cochlea_np = _to_numpy(cochleagram)
    spikes_np = _to_numpy(spikes)
    time_axis_ms = np.arange(cochlea_np.shape[-1]) / sample_rate_hz * 1_000.0
    time_end_ms = cochlea_np.shape[-1] / sample_rate_hz * 1_000.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].imshow(
        cochlea_np,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0.0, time_end_ms, -0.5, cochlea_np.shape[0] - 0.5],
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("Channel")
    spike_y, spike_x = np.nonzero(spikes_np > 0.0)
    if spike_x.size:
        axes[1].scatter(time_axis_ms[spike_x], spike_y, s=4, c="black")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Spike Channel")
    if xlim_ms is not None:
        axes[1].set_xlim(*xlim_ms)
    _finalize_figure(path)


def save_heatmap(
    matrix: torch.Tensor,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(_to_numpy(matrix), aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finalize_figure(path)


def save_prediction_scatter(
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    target_np = _to_numpy(target)
    prediction_np = _to_numpy(prediction)
    minimum = min(float(target_np.min()), float(prediction_np.min()))
    maximum = max(float(target_np.max()), float(prediction_np.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target_np, prediction_np, s=18, alpha=0.75)
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finalize_figure(path)


def save_error_histogram(
    errors: torch.Tensor,
    path: str | Path,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(_to_numpy(errors), bins=25, color="#2d6a4f", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _finalize_figure(path)


def save_loss_curve(
    train_loss: list[float],
    val_loss: list[float],
    path: str | Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label="train", linewidth=2.0)
    ax.plot(val_loss, label="val", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    _finalize_figure(path)


def save_grouped_bar_chart(
    categories: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    x_axis = np.arange(len(categories))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(9, 4))
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2.0) * width
        ax.bar(x_axis + offset, values, width=width, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(categories)
    ax.legend()
    _finalize_figure(path)


def save_text_figure(lines: list[str], path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    _finalize_figure(path)


def format_float(value: float, digits: int = 4) -> float:
    return float(f"{value:.{digits}f}")


def stage_failure_report(
    issue: str,
    cause: str,
    evidence: dict[str, Any],
    suggested_fix: str,
    requires_next_model: bool,
) -> dict[str, Any]:
    return {
        "observed_issue": issue,
        "likely_cause": cause,
        "evidence": tensor_to_python(evidence),
        "suggested_fix": suggested_fix,
        "requires_next_model": requires_next_model,
    }


def percentile_clip(signal: torch.Tensor, percentile: float = 99.0) -> torch.Tensor:
    threshold = torch.quantile(signal.abs().flatten(), percentile / 100.0)
    return torch.clamp(signal, -threshold, threshold) / threshold.clamp_min(1e-6)


def radians_to_degrees(radians: torch.Tensor) -> torch.Tensor:
    return radians * (180.0 / math.pi)


def degrees_to_radians(degrees: torch.Tensor) -> torch.Tensor:
    return degrees * (math.pi / 180.0)
