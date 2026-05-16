from __future__ import annotations

"""Staged noise-robustness experiments for the distance pathway.

The clean distance pathway is highly accurate but fails under the 10 dB SNR
diagnostic noise condition. This experiment tests staged fixes:

1. Mild cochlea retuning for the spike-raster pathway.
2. VCN local multi-channel coincidence.
3. IC soft sweep facilitation.
"""

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments.full_distance_pathway_model import (
    AC_INHIBIT_GAIN,
    IC_LIF_BETA,
    IC_LIF_THRESHOLD,
    MAX_DISTANCE_M,
    MIN_DISTANCE_M,
    NOISE_ROBUSTNESS_JITTER_S,
    NOISE_ROBUSTNESS_SNR_DB,
    NUM_CHANNELS,
    NUM_DISTANCE_BINS,
    NUM_TEST_SAMPLES,
    RNG_SEED,
    VCN_LIF_THRESHOLD_FRACTION,
    _ac_topographic_map,
    _candidate_delay_samples,
    _candidate_distances,
    _chirp_channel_times,
    _load_channel_latency,
    _make_config,
    _make_noisy_config,
    _metrics,
    _run_cochlea_binaural,
    _simulate_scene,
    _vcn_input_tensor,
    _vcn_vnll_onset_detector,
)
from mini_models.common.plotting import ensure_dir
from mini_models.experiments.final_cochlea_model_analysis import _log_spaced_centers


OUTPUT_DIR = ROOT / "distance_pathway" / "outputs" / "distance_noise_robustness"
REPORT_PATH = ROOT / "distance_pathway" / "reports" / "distance_noise_robustness_experiments.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"
ROBUST_LATENCY_VECTOR_PATH = OUTPUT_DIR / "spike_tuned_consensus_facil_latency_samples.npy"

ROBUST_SPIKE_THRESHOLD_MULTIPLIER = 16.0
ROBUST_SPIKE_BETA = 0.50
CONSENSUS_CHANNEL_RADIUS = 2
CONSENSUS_TIME_RADIUS = 8
CONSENSUS_MIN_COUNT = 3
CONSENSUS_REFRACTORY_S = 0.0005
COCHLEAGRAM_CONSENSUS_THRESHOLD_FRACTION = 0.10
IC_FACIL_GAIN = 0.45
IC_FACIL_TAU_SAMPLES = 8.0
LATENCY_CALIBRATION_DISTANCES_M = np.linspace(MIN_DISTANCE_M, MAX_DISTANCE_M, 12)
VCN_MIN_RESPONSIVE_HZ = 4_000.0


@dataclass(frozen=True)
class RobustVariant:
    """Configuration for a staged robustness variant.

    Attributes:
        key: Stable identifier for JSON/report output.
        name: Human-readable variant name.
        vcn_input: `cochleagram` or `spikes`.
        spike_threshold_multiplier: Multiplier applied to cochlear spike threshold.
        spike_beta: Optional override for cochlear LIF beta.
        vcn_detector: `first` or `consensus`.
        ic_mode: `plain` or `facilitated`.
        note: Short explanation of the tested mechanism.
    """

    key: str
    name: str
    vcn_input: str
    spike_threshold_multiplier: float = 1.0
    spike_beta: float | None = None
    vcn_detector: str = "first"
    ic_mode: str = "plain"
    note: str = ""


@dataclass
class RobustPrediction:
    """Distance prediction for one sample.

    Attributes:
        distance_m: True distance.
        predicted_distance_m: Predicted distance.
    """

    distance_m: float
    predicted_distance_m: float


def _variant_config(base_config, variant: RobustVariant):
    """Apply cochlea tuning parameters for a variant.

    Args:
        base_config: Base clean or noisy acoustic configuration.
        variant: Robustness variant.

    Returns:
        Acoustic configuration for this variant.
    """
    kwargs = {
        "spike_threshold": float(base_config.spike_threshold) * variant.spike_threshold_multiplier,
    }
    if variant.spike_beta is not None:
        kwargs["spike_beta"] = float(variant.spike_beta)
    return replace(base_config, **kwargs)


def _combined_vcn_input(cochlea, variant: RobustVariant) -> torch.Tensor:
    """Return the bilateral activity tensor used by the VCN.

    Args:
        cochlea: Binaural cochlea output.
        variant: Robustness variant.

    Returns:
        Combined left/right VCN input `[channels, time]`.
    """
    left = _vcn_input_tensor(cochlea, "left", variant.vcn_input)
    right = _vcn_input_tensor(cochlea, "right", variant.vcn_input)
    return torch.maximum(left, right)


def _vcn_consensus_detector(activity_tensor: torch.Tensor, variant_config, source: str) -> np.ndarray:
    """Detect VCN events using local multi-channel coincidence.

    Args:
        activity_tensor: VCN input `[channels, time]`.
        variant_config: Acoustic configuration.
        source: `cochleagram` or `spikes`.

    Returns:
        Consensus onset raster `[channels, time]`.
    """
    activity = activity_tensor.detach().cpu().to(torch.float32)
    if source == "cochleagram":
        thresholds = COCHLEAGRAM_CONSENSUS_THRESHOLD_FRACTION * torch.clamp(activity.max(dim=1).values, min=1e-12)
        raw = activity >= thresholds[:, None]
    elif source == "spikes":
        raw = activity > 0.0
    else:
        raise ValueError(f"Unknown VCN source: {source}")

    kernel = torch.ones(
        1,
        1,
        2 * CONSENSUS_CHANNEL_RADIUS + 1,
        2 * CONSENSUS_TIME_RADIUS + 1,
        dtype=torch.float32,
    )
    counts = F.conv2d(
        raw.to(torch.float32)[None, None],
        kernel,
        padding=(CONSENSUS_CHANNEL_RADIUS, CONSENSUS_TIME_RADIUS),
    )[0, 0]
    candidates = raw & (counts >= CONSENSUS_MIN_COUNT)
    candidate_np = candidates.numpy()
    output = np.zeros_like(candidate_np, dtype=np.float32)
    refractory = int(round(CONSENSUS_REFRACTORY_S * variant_config.sample_rate_hz))
    for channel in range(candidate_np.shape[0]):
        event_times = np.flatnonzero(candidate_np[channel])
        last_emit = -refractory
        for event_time in event_times:
            if event_time - last_emit >= refractory:
                output[channel, int(event_time)] = 1.0
                last_emit = int(event_time)
                break
    return output


def _run_vcn(cochlea, variant_config, variant: RobustVariant) -> np.ndarray:
    """Run the selected VCN detector and return combined bilateral onsets.

    Args:
        cochlea: Binaural cochlea output.
        variant_config: Acoustic configuration.
        variant: Robustness variant.

    Returns:
        Combined VCN onset raster `[channels, time]`.
    """
    if variant.vcn_detector == "first":
        left = _vcn_vnll_onset_detector(_vcn_input_tensor(cochlea, "left", variant.vcn_input), variant_config)
        right = _vcn_vnll_onset_detector(_vcn_input_tensor(cochlea, "right", variant.vcn_input), variant_config)
        return _apply_vcn_frequency_mask(np.maximum(left, right), variant_config)
    if variant.vcn_detector == "consensus":
        return _apply_vcn_frequency_mask(
            _vcn_consensus_detector(_combined_vcn_input(cochlea, variant), variant_config, variant.vcn_input),
            variant_config,
        )
    raise ValueError(f"Unknown VCN detector: {variant.vcn_detector}")


def _apply_vcn_frequency_mask(vcn: np.ndarray, config) -> np.ndarray:
    """Silence VCN channels below the modelled call-relevant band.

    Args:
        vcn: VCN raster `[channels, time]`.
        config: Acoustic configuration.

    Returns:
        VCN raster with channels below `VCN_MIN_RESPONSIVE_HZ` silenced.
    """
    centers = _log_spaced_centers(config).detach().cpu().numpy()
    responsive = centers >= VCN_MIN_RESPONSIVE_HZ
    masked = vcn.copy()
    masked[~responsive, :] = 0.0
    return masked


def _responsive_channel_mask(config) -> np.ndarray:
    """Return channels allowed to drive the VCN distance pathway.

    Args:
        config: Acoustic configuration.

    Returns:
        Boolean mask over cochlea channels.
    """
    centers = _log_spaced_centers(config).detach().cpu().numpy()
    return centers >= VCN_MIN_RESPONSIVE_HZ


def _first_times(raster: np.ndarray) -> np.ndarray:
    """Return first event time per channel."""
    first = np.full(raster.shape[0], -1, dtype=np.int64)
    for channel in range(raster.shape[0]):
        event_times = np.flatnonzero(raster[channel] > 0.0)
        if event_times.size:
            first[channel] = int(event_times[0])
    return first


def _calibrate_latency(config, variant: RobustVariant) -> tuple[np.ndarray, dict[str, object]]:
    """Calibrate per-channel latency for a variant using clean echoes.

    Args:
        config: Clean acoustic configuration already adjusted for this variant.
        variant: Robustness variant.

    Returns:
        Pair of median per-channel latency samples and calibration stats.
    """
    cd_times = _chirp_channel_times(config)
    responsive = _responsive_channel_mask(config)
    rows = []
    for distance_m in LATENCY_CALIBRATION_DISTANCES_M:
        receive = _simulate_scene(config, float(distance_m), add_noise=False)
        cochlea = _run_cochlea_binaural(config, receive)
        vcn = _run_vcn(cochlea, config, variant)
        first = _first_times(vcn)
        round_trip = int(round((2.0 * float(distance_m) / config.speed_of_sound_m_s) * config.sample_rate_hz))
        row = np.full(NUM_CHANNELS, np.nan, dtype=np.float64)
        valid = (first >= 0) & responsive
        row[valid] = first[valid] - (cd_times[valid] + round_trip)
        rows.append(row)
    matrix = np.vstack(rows)
    median = np.zeros(NUM_CHANNELS, dtype=np.float64)
    responsive_matrix = matrix[:, responsive]
    responsive_medians = np.full(responsive_matrix.shape[1], np.nan, dtype=np.float64)
    calibrated_responsive = np.zeros(responsive_matrix.shape[1], dtype=bool)
    for channel_index in range(responsive_matrix.shape[1]):
        values = responsive_matrix[:, channel_index]
        values = values[np.isfinite(values)]
        if values.size:
            responsive_medians[channel_index] = np.rint(np.median(values))
            calibrated_responsive[channel_index] = True
    finite_medians = responsive_medians[np.isfinite(responsive_medians)]
    fallback = int(np.median(finite_medians)) if finite_medians.size else 0
    responsive_medians = np.where(np.isfinite(responsive_medians), responsive_medians, fallback)
    median[responsive] = responsive_medians
    latency_std = np.full(responsive_matrix.shape[1], np.nan, dtype=np.float64)
    for channel_index in range(responsive_matrix.shape[1]):
        values = responsive_matrix[:, channel_index]
        values = values[np.isfinite(values)]
        if values.size:
            latency_std[channel_index] = float(np.std(values))
    stats = {
        "responsive_channels": int(np.sum(responsive)),
        "silenced_channels": int(NUM_CHANNELS - np.sum(responsive)),
        "calibrated_responsive_channels": int(np.sum(calibrated_responsive)),
        "missing_responsive_channels": int(np.sum(~calibrated_responsive)),
        "latency_min_samples": int(np.min(median[responsive])) if np.any(responsive) else 0,
        "latency_max_samples": int(np.max(median[responsive])) if np.any(responsive) else 0,
        "latency_mean_std_samples": float(np.nanmean(latency_std)) if latency_std.size else 0.0,
        "latency_max_std_samples": float(np.nanmax(latency_std)) if latency_std.size else 0.0,
    }
    return median.astype(np.int64), stats


def _ic_plain(vcn: np.ndarray, config, latency_samples: np.ndarray) -> np.ndarray:
    """Compute plain closed-form IC LIF coincidence activation."""
    cd_times = _chirp_channel_times(config) + latency_samples.astype(np.int64)
    candidate_delays = _candidate_delay_samples(config)
    activation = np.zeros(NUM_DISTANCE_BINS, dtype=np.float64)
    for channel in range(NUM_CHANNELS):
        echo_times = np.flatnonzero(vcn[channel] > 0.0)
        if echo_times.size == 0:
            continue
        echo_time = int(echo_times[0])
        expected_times = cd_times[channel] + candidate_delays
        delta = np.abs(echo_time - expected_times)
        membrane_peak = 1.0 + np.power(IC_LIF_BETA, delta)
        activation += np.maximum(0.0, membrane_peak - IC_LIF_THRESHOLD)
    return activation


def _ic_facilitated(vcn: np.ndarray, config, latency_samples: np.ndarray) -> np.ndarray:
    """Compute IC activation with soft local sweep facilitation.

    Facilitation is not a hard gate. It boosts candidate delays that are
    locally consistent across neighbouring channels.
    """
    cd_times = _chirp_channel_times(config) + latency_samples.astype(np.int64)
    candidate_delays = _candidate_delay_samples(config)
    echo_first = _first_times(vcn)
    channel_scores = np.zeros((NUM_CHANNELS, NUM_DISTANCE_BINS), dtype=np.float64)
    facil_scores = np.zeros_like(channel_scores)

    for channel in range(NUM_CHANNELS):
        if echo_first[channel] < 0:
            continue
        expected_times = cd_times[channel] + candidate_delays
        delta = np.abs(int(echo_first[channel]) - expected_times)
        membrane_peak = 1.0 + np.power(IC_LIF_BETA, delta)
        channel_scores[channel] = np.maximum(0.0, membrane_peak - IC_LIF_THRESHOLD)

    for channel in range(NUM_CHANNELS):
        neighbours = [
            neighbour
            for neighbour in (channel - 2, channel - 1, channel + 1, channel + 2)
            if 0 <= neighbour < NUM_CHANNELS and echo_first[neighbour] >= 0
        ]
        if not neighbours:
            continue
        for neighbour in neighbours:
            expected_neighbour = cd_times[neighbour] + candidate_delays
            delta_neighbour = np.abs(int(echo_first[neighbour]) - expected_neighbour)
            facil_scores[channel] += np.exp(-delta_neighbour / IC_FACIL_TAU_SAMPLES)
        facil_scores[channel] /= len(neighbours)

    return np.sum(channel_scores * (1.0 + IC_FACIL_GAIN * facil_scores), axis=0)


def _predict_one(config, distance_m: float, variant: RobustVariant, latency_samples: np.ndarray, *, add_noise: bool) -> RobustPrediction:
    """Run one variant for one distance."""
    receive = _simulate_scene(config, distance_m, add_noise=add_noise)
    cochlea = _run_cochlea_binaural(config, receive)
    vcn = _run_vcn(cochlea, config, variant)
    ic = _ic_facilitated(vcn, config, latency_samples) if variant.ic_mode == "facilitated" else _ic_plain(vcn, config, latency_samples)
    ac = _ac_topographic_map(ic)
    distances = _candidate_distances()
    total = ac.sum()
    predicted = float(distances[len(distances) // 2]) if total <= 1e-12 else float(np.sum(ac * distances) / total)
    return RobustPrediction(distance_m=float(distance_m), predicted_distance_m=predicted)


def _variant_definitions() -> list[RobustVariant]:
    """Return all staged robustness variants."""
    return [
        RobustVariant(
            key="cochleagram_baseline",
            name="Cochleagram VCN baseline",
            vcn_input="cochleagram",
            note="Current high-accuracy clean model; VCN reads cochleagram.",
        ),
        RobustVariant(
            key="spike_baseline",
            name="Spike-raster VCN baseline",
            vcn_input="spikes",
            note="Strictly spike-raster VCN input.",
        ),
        RobustVariant(
            key="spike_tuned",
            name="Spike VCN + cochlea tuning",
            vcn_input="spikes",
            spike_threshold_multiplier=ROBUST_SPIKE_THRESHOLD_MULTIPLIER,
            spike_beta=ROBUST_SPIKE_BETA,
            note="Uses 16x cochlear threshold and beta=0.5.",
        ),
        RobustVariant(
            key="spike_tuned_consensus",
            name="Spike VCN + tuning + VCN consensus",
            vcn_input="spikes",
            spike_threshold_multiplier=ROBUST_SPIKE_THRESHOLD_MULTIPLIER,
            spike_beta=ROBUST_SPIKE_BETA,
            vcn_detector="consensus",
            note="Adds local multi-channel coincidence in VCN.",
        ),
        RobustVariant(
            key="spike_tuned_consensus_facil",
            name="Spike VCN + tuning + consensus + IC facilitation",
            vcn_input="spikes",
            spike_threshold_multiplier=ROBUST_SPIKE_THRESHOLD_MULTIPLIER,
            spike_beta=ROBUST_SPIKE_BETA,
            vcn_detector="consensus",
            ic_mode="facilitated",
            note="Adds soft sweep-consistency facilitation in IC.",
        ),
        RobustVariant(
            key="cochleagram_consensus_facil",
            name="Cochleagram VCN + consensus + IC facilitation",
            vcn_input="cochleagram",
            vcn_detector="consensus",
            ic_mode="facilitated",
            note="Tests whether the same pathway-level fixes work without spike-raster input.",
        ),
    ]


def _evaluate_variant(base_clean_config, base_noisy_config, distances: np.ndarray, variant: RobustVariant) -> dict[str, object]:
    """Evaluate one variant on clean and noisy distance sets."""
    clean_config = _variant_config(base_clean_config, variant)
    noisy_config = _variant_config(base_noisy_config, variant)
    if variant.key == "cochleagram_baseline":
        latency_samples = _load_channel_latency(clean_config)
        responsive = _responsive_channel_mask(clean_config)
        calibration_stats = {
            "responsive_channels": int(np.sum(responsive)),
            "silenced_channels": int(NUM_CHANNELS - np.sum(responsive)),
            "calibrated_responsive_channels": int(np.sum(responsive)),
            "missing_responsive_channels": 0,
            "latency_min_samples": int(latency_samples[responsive].min()),
            "latency_max_samples": int(latency_samples[responsive].max()),
            "latency_mean_std_samples": None,
            "latency_max_std_samples": None,
        }
    else:
        latency_samples, calibration_stats = _calibrate_latency(clean_config, variant)
    if variant.key == "spike_tuned_consensus_facil":
        ensure_dir(OUTPUT_DIR)
        np.save(ROBUST_LATENCY_VECTOR_PATH, latency_samples)

    torch.manual_seed(RNG_SEED + 30_000)
    clean_predictions = [
        _predict_one(clean_config, float(distance), variant, latency_samples, add_noise=False)
        for distance in distances
    ]
    torch.manual_seed(RNG_SEED + 40_000)
    noisy_predictions = [
        _predict_one(noisy_config, float(distance), variant, latency_samples, add_noise=True)
        for distance in distances
    ]
    return {
        "key": variant.key,
        "name": variant.name,
        "vcn_input": variant.vcn_input,
        "spike_threshold_multiplier": variant.spike_threshold_multiplier,
        "spike_beta": variant.spike_beta,
        "vcn_detector": variant.vcn_detector,
        "ic_mode": variant.ic_mode,
        "latency_min_samples": int(latency_samples.min()),
        "latency_max_samples": int(latency_samples.max()),
        "calibration_stats": calibration_stats,
        "clean_metrics": _metrics(clean_predictions),
        "noisy_metrics": _metrics(noisy_predictions),
        "note": variant.note,
    }


def _write_report(results: dict[str, object]) -> None:
    """Write the staged robustness experiment report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_noisy = min(results["variant_results"], key=lambda row: row["noisy_metrics"]["mae_m"])
    best_clean = min(results["variant_results"], key=lambda row: row["clean_metrics"]["mae_m"])
    selected = next(row for row in results["variant_results"] if row["key"] == "spike_tuned_consensus_facil")
    selected_stats = selected["calibration_stats"]
    lines = [
        "# Distance Noise Robustness Experiments",
        "",
        "This report tests staged fixes for the distance-pathway noise failure observed in `distance_noise_diagnostics.md`.",
        "",
        "## Noise Condition",
        "",
        f"- Noisy test: `{NOISE_ROBUSTNESS_SNR_DB:.1f} dB` SNR additive white receiver noise plus `jitter_std = {NOISE_ROBUSTNESS_JITTER_S:.6g} s`.",
        f"- Realised noisy config `noise_std`: `{results['noise_std']:.6g}`.",
        f"- Distances: `{MIN_DISTANCE_M} -> {MAX_DISTANCE_M} m`, `{NUM_TEST_SAMPLES}` samples.",
        "",
        "## Tested Mechanisms",
        "",
        f"- Cochlea tuning: spike threshold `x{ROBUST_SPIKE_THRESHOLD_MULTIPLIER:.0f}`, cochlear LIF beta `{ROBUST_SPIKE_BETA}`.",
        f"- VCN consensus: local window `±{CONSENSUS_CHANNEL_RADIUS}` channels and `±{CONSENSUS_TIME_RADIUS}` samples, requiring at least `{CONSENSUS_MIN_COUNT}` local events.",
        f"- IC facilitation: soft local sweep-consistency gain `{IC_FACIL_GAIN}`, tau `{IC_FACIL_TAU_SAMPLES}` samples.",
        f"- VCN frequency mask: channels below `{VCN_MIN_RESPONSIVE_HZ / 1_000.0:.1f} kHz` are silenced because the call-relevant sweep does not use them.",
        "",
        "## Results",
        "",
        "| Variant | VCN input | Cochlea tuning | VCN detector | IC mode | Clean MAE | Noisy MAE | Noisy RMSE | Noisy max error | Note |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results["variant_results"]:
        tuning = "default"
        if row["spike_threshold_multiplier"] != 1.0 or row["spike_beta"] is not None:
            tuning = f"x{row['spike_threshold_multiplier']:.0f}, beta={row['spike_beta']}"
        clean = row["clean_metrics"]
        noisy = row["noisy_metrics"]
        lines.append(
            "| "
            f"{row['name']} | "
            f"`{row['vcn_input']}` | "
            f"`{tuning}` | "
            f"`{row['vcn_detector']}` | "
            f"`{row['ic_mode']}` | "
            f"`{clean['mae_m'] * 100.0:.3f} cm` | "
            f"`{noisy['mae_m'] * 100.0:.3f} cm` | "
            f"`{noisy['rmse_m'] * 100.0:.3f} cm` | "
            f"`{noisy['max_abs_error_m'] * 100.0:.3f} cm` | "
            f"{row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- Best clean result: `{best_clean['name']}` with MAE `{best_clean['clean_metrics']['mae_m'] * 100.0:.3f} cm`.",
            f"- Best noisy result: `{best_noisy['name']}` with MAE `{best_noisy['noisy_metrics']['mae_m'] * 100.0:.3f} cm`.",
            "- The main robustness gain came from cochlea tuning on the spike-raster pathway: threshold `x16` and beta `0.5` reduced noisy MAE from metre-scale failure to approximately `5 cm`.",
            "- VCN consensus and IC facilitation did not materially improve this first tuned spike-raster result. They may need retuning now that the cochlear input is much sparser.",
            "- The cochleagram pathway remains excellent clean, but it is still highly noise-sensitive because it reads continuous low-threshold activity before the cochlear spike encoder.",
            "",
            "## Recalibrated Robust Latency Vector",
            "",
            "The selected robust model is `Spike VCN + tuning + consensus + IC facilitation`, with VCN channels below `4 kHz` silenced. Its latency vector is recalibrated after applying the spike-raster cochlea tuning, VCN consensus, and frequency mask.",
            "",
            "| Calibration property | Value |",
            "|---|---:|",
            f"| responsive channels | `{selected_stats['responsive_channels']}` |",
            f"| silenced channels below 4 kHz | `{selected_stats['silenced_channels']}` |",
            f"| calibrated responsive channels | `{selected_stats['calibrated_responsive_channels']}` |",
            f"| missing responsive channels | `{selected_stats['missing_responsive_channels']}` |",
            f"| latency range over responsive channels | `{selected_stats['latency_min_samples']} -> {selected_stats['latency_max_samples']}` samples |",
            f"| mean latency std across calibration distances | `{selected_stats['latency_mean_std_samples']:.3f}` samples |",
            f"| max latency std across calibration distances | `{selected_stats['latency_max_std_samples']:.3f}` samples |",
            f"| saved vector | `{ROBUST_LATENCY_VECTOR_PATH.relative_to(ROOT)}` |",
            "",
            f"With this recalibrated vector, the selected robust model gives clean MAE `{selected['clean_metrics']['mae_m'] * 100.0:.3f} cm` and noisy MAE `{selected['noisy_metrics']['mae_m'] * 100.0:.3f} cm`.",
            "",
            "## Interpretation",
            "",
            "- The spike-raster pathway is the relevant path for cochlea threshold/beta tuning, because cochleagram VCN bypasses cochlear spike generation.",
            "- VCN consensus is intended to reject isolated noisy events by requiring local channel agreement.",
            "- IC facilitation is deliberately soft: it boosts sweep-consistent candidate distances but does not hard-gate the response.",
            "- The VCN frequency mask is a biological/engineering assumption that sub-call-band channels should not drive the distance pathway.",
            "- If a variant improves noisy MAE but destroys clean MAE, it is not yet acceptable as a general distance pathway.",
            "",
            "## Generated Files",
            "",
            f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`",
            "",
            f"Runtime: `{results['elapsed_seconds']:.2f} s`.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run all staged robustness variants."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPORT_PATH.parent)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    clean_config = _make_config()
    noisy_config = _make_noisy_config(clean_config)
    rng = np.random.default_rng(RNG_SEED)
    distances = rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M, size=NUM_TEST_SAMPLES)
    variant_results = [
        _evaluate_variant(clean_config, noisy_config, distances, variant)
        for variant in _variant_definitions()
    ]
    results = {
        "experiment": "distance_noise_robustness_experiments",
        "elapsed_seconds": time.perf_counter() - start,
        "target_snr_db": NOISE_ROBUSTNESS_SNR_DB,
        "noise_std": noisy_config.noise_std,
        "jitter_std_s": noisy_config.jitter_std_s,
        "variant_results": variant_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(results)
    return results


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
