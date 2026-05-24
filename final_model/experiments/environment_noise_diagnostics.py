from __future__ import annotations

"""Environmental-noise and reverberation diagnostics for the fixed final model.

This experiment keeps the existing clean final-model reports untouched. It
tests a different acoustic noise convention: additive environmental noise is
inserted after propagation delay/attenuation, but before head shadow and
comb-filter spectral shaping. A second condition adds simple late returned
echoes to test reverberation/clutter sensitivity.
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azimuth_pathway.experiments import azimuth_ild_line_attractor as azc
from azimuth_pathway.experiments import azimuth_pathway_first_attempt as az
from distance_pathway.experiments import final_distance_pipeline_with_attractor as dist_cann
from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_line_attractor as elc
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from final_model.experiments import final_model_results as final
from mini_models.common.plotting import ensure_dir, save_figure
from models.acoustics import fractional_delay, generate_fm_chirp, pad_signal
from utils.common import GlobalConfig


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "environment_noise_diagnostics"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "final_model" / "reports" / "environment_noise_diagnostics.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

CHANNELS = 48
DISTANCE_BINS = 180
ANGULAR_BINS = 91
SAMPLES = 8
RNG_SEED = 9_401
ENVIRONMENT_SNR_DB_AT_CALL = 50.0
AZIMUTH_LIMIT_DEG = 45.0
ELEVATION_LIMIT_DEG = 45.0
REVERB_DELAYS_MS = (0.85, 1.75, 3.20, 5.10)
REVERB_GAINS = (0.34, 0.22, 0.13, 0.07)


@dataclass(frozen=True)
class AcousticCondition:
    """Environmental acoustic diagnostic condition."""

    key: str
    name: str
    add_environment_noise: bool
    add_reverb: bool


@dataclass(frozen=True)
class FixedPrediction:
    """Fixed-model prediction for one target and acoustic condition."""

    true_distance_m: float
    true_azimuth_deg: float
    true_elevation_deg: float
    direct_distance_m: float
    direct_azimuth_deg: float
    direct_elevation_deg: float
    cann_distance_m: float
    cann_azimuth_deg: float
    cann_elevation_deg: float


CONDITIONS = [
    AcousticCondition("clean", "Clean echo", False, False),
    AcousticCondition("env_noise_50db", "Environmental noise, 50 dB call-referenced SNR", True, False),
    AcousticCondition("env_noise_50db_reverb", "Environmental noise plus late echoes/reverb", True, True),
]


def _shift_with_zeros(signal: torch.Tensor, delay_samples: int) -> torch.Tensor:
    """Delay a signal by integer samples without circular wraparound."""
    if delay_samples <= 0:
        return signal.clone()
    shifted = torch.zeros_like(signal)
    if delay_samples < signal.shape[-1]:
        shifted[..., delay_samples:] = signal[..., :-delay_samples]
    return shifted


def _call_referenced_noise_std(config: GlobalConfig, snr_db: float) -> float:
    """Return noise std for a target SNR relative to emitted call RMS.

    The reference is the active emitted chirp at the source, after
    `transmit_gain`. The same resulting noise standard deviation is then added
    after echo attenuation as a fixed environmental floor.
    """
    chirp, _ = generate_fm_chirp(
        config,
        batch_size=1,
        device=torch.device("cpu"),
        transmit_gain=config.transmit_gain,
    )
    call_rms = float(torch.sqrt(torch.mean(chirp.square())).item())
    return call_rms / (10.0 ** (snr_db / 20.0))


def _simulate_environment_echo(
    config: GlobalConfig,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    *,
    condition: AcousticCondition,
    noise_std: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Simulate one binaural echo with environmental noise before filtering.

    Processing order:

    1. generate transmitted chirp;
    2. apply geometric delay and inverse-square attenuation;
    3. optionally add environmental white noise;
    4. optionally add delayed reverberant echo copies;
    5. apply multiplicative head shadow;
    6. apply the comb-filter elevation cue.
    """
    device = torch.device("cpu")
    radii = torch.tensor([distance_m], dtype=torch.float32, device=device)
    azimuth = torch.tensor([azimuth_deg], dtype=torch.float32, device=device)
    elevation = torch.tensor([elevation_deg], dtype=torch.float32, device=device)
    chirp, _ = generate_fm_chirp(config, batch_size=1, device=device, transmit_gain=config.transmit_gain)
    transmit = pad_signal(chirp, config.signal_samples)

    azimuth_rad = torch.deg2rad(azimuth)
    elevation_rad = torch.deg2rad(elevation)
    x_coord = radii * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y_coord = radii * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z_coord = radii * torch.sin(elevation_rad)

    ear_offsets = torch.tensor([-config.ear_spacing_m / 2.0, config.ear_spacing_m / 2.0], dtype=torch.float32)
    ear_y = y_coord[:, None] - ear_offsets[None, :]
    distance_to_ear = torch.sqrt(x_coord[:, None].square() + ear_y.square() + z_coord[:, None].square())
    path_lengths = radii[:, None] + distance_to_ear
    delays_s = path_lengths / config.speed_of_sound_m_s
    delay_samples = delays_s * config.sample_rate_hz

    base_echo = fractional_delay(transmit.unsqueeze(1).expand(-1, 2, -1), delay_samples)[0]
    amplitudes = 0.7 / path_lengths.square().clamp_min(0.25)
    attenuated = base_echo * amplitudes[0, :, None]

    returned = attenuated.clone()
    if condition.add_reverb:
        for delay_ms, gain in zip(REVERB_DELAYS_MS, REVERB_GAINS):
            delay = int(round(delay_ms * 1e-3 * config.sample_rate_hz))
            returned = returned + float(gain) * _shift_with_zeros(attenuated, delay)

    if condition.add_environment_noise:
        returned = returned + noise_std * torch.randn(returned.shape, generator=rng, dtype=returned.dtype)

    head_shadow = torch.exp(
        config.head_shadow_strength
        * torch.sin(azimuth_rad)[0]
        * torch.tensor([-1.0, 1.0], dtype=torch.float32)
    )
    filtered = returned * head_shadow[:, None]
    filtered[0] = elev.apply_comb_filter(filtered[0], config, elevation_deg, elev.DEEP_COMB_DELAYED_COPY_GAIN)
    filtered[1] = elev.apply_comb_filter(filtered[1], config, elevation_deg, elev.DEEP_COMB_DELAYED_COPY_GAIN)
    return filtered.detach()


def _distance_prediction_from_receive(
    config: GlobalConfig,
    variant: fdm.PathwayVariant,
    receive: torch.Tensor,
) -> tuple[float, float, np.ndarray]:
    """Return direct COM, CANN distance, and AC population for one waveform."""
    cochlea = fdm._run_cochlea_binaural(config, receive)
    vcn_left, vcn_right = fdm._run_vcn_for_variant(cochlea, config, variant)
    dnll = fdm._dnll_suppression(vcn_left, vcn_right, config)
    if variant.ic_mode == "facilitated":
        ic = fdm._ic_facilitated_coincidence(dnll, config, variant.latency_samples)
    else:
        ic = fdm._ic_lif_coincidence(dnll, config, variant.latency_samples)
    ac = fdm._ac_topographic_map(ic)
    direct = fdm._sc_center_of_mass(ac, config)
    bins = fdm._candidate_distances(config)
    cann, _, _, _ = dist_cann.run_line_attractor(
        ac[None, :],
        bins,
        dist_cann.SC_ATTRACTOR_VARIANTS[1],
        keep_history=False,
    )
    return direct, float(cann[0]), ac


def _azimuth_prediction_from_receive(
    config: GlobalConfig,
    receive: torch.Tensor,
) -> tuple[float, float, np.ndarray]:
    """Return direct ITD COM, CANN azimuth, and ITD population."""
    bins = az.azimuth_grid(AZIMUTH_LIMIT_DEG)
    cochlea = fdm._run_cochlea_binaural(config, receive)
    left_spikes, right_spikes = az.run_dynamic_cochlea_spikes(cochlea, config)
    vcn_left = az.vcn_consensus_single_ear(left_spikes, config)
    vcn_right = az.vcn_consensus_single_ear(right_spikes, config)
    itd = az.jeffress_lif_itd_activation(vcn_left, vcn_right, config, bins)
    direct = az.centre_of_mass(itd, bins)
    cann, _, _, _ = azc.run_cann_readout(itd[None, :], bins)
    return direct, float(cann[0]), itd


def _elevation_prediction_from_receive(
    config: GlobalConfig,
    baseline_profile: np.ndarray,
    calibration_params: dict[str, float],
    receive: torch.Tensor,
    azimuth_deg: float,
) -> tuple[float, float, np.ndarray]:
    """Return direct DCN COM, calibrated CANN elevation, and DCN population."""
    cochlea = fdm._run_cochlea_binaural(config, receive)
    selected = cochlea.right_cochleagram if azimuth_deg >= 0.0 else cochlea.left_cochleagram
    selected_spikes = fdm._dynamic_lif_encode(selected, config, fdm.DYNAMIC_COHLEA_SCHEDULE)
    profile = elev.dynamic_wideband_inhibited_profile(selected, selected_spikes, elev.DynamicInhibitionParams())
    equalized = profile / np.maximum(baseline_profile, 1e-4)
    equalized = equalized / np.maximum(equalized.max(), 1e-12)
    bins = elev.elevation_grid()
    centres_hz, gain_matrix, _ = elev.build_dcn_templates(config, bins, elev.DEEP_COMB_DELAYED_COPY_GAIN)
    population = elev.dcn_signal_weighted_transfer_response(equalized, baseline_profile, gain_matrix, centres_hz)
    direct = elev.centre_of_mass(population, bins)
    attractor = elc.run_attractor_variants(population[None, :], bins)
    raw = np.asarray(attractor["reflected"]["prediction"], dtype=np.float64)
    cann = elc.apply_tuned_inverse_sigmoid(raw, calibration_params)
    return direct, float(cann[0]), population


def _targets() -> list[dict[str, float]]:
    """Create the low-sample constrained 3D smoke-test target set."""
    rng = np.random.default_rng(RNG_SEED)
    return [
        {
            "distance_m": float(distance),
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
        }
        for distance, azimuth, elevation in zip(
            rng.uniform(0.25, 5.0, size=SAMPLES),
            rng.uniform(-AZIMUTH_LIMIT_DEG, AZIMUTH_LIMIT_DEG, size=SAMPLES),
            rng.uniform(-ELEVATION_LIMIT_DEG, ELEVATION_LIMIT_DEG, size=SAMPLES),
        )
    ]


def _angular_error_deg(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return wrapped angular error in degrees."""
    return (predicted - true + 180.0) % 360.0 - 180.0


def _spherical_to_cartesian(distance_m: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Convert spherical localisation coordinates to Cartesian coordinates."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.stack(
        [
            distance_m * np.cos(el) * np.cos(az),
            distance_m * np.cos(el) * np.sin(az),
            distance_m * np.sin(el),
        ],
        axis=-1,
    )


def _metrics(rows: list[FixedPrediction], prefix: str) -> dict[str, float]:
    """Compute coordinate and Euclidean metrics for direct or CANN predictions."""
    true_d = np.array([row.true_distance_m for row in rows])
    true_a = np.array([row.true_azimuth_deg for row in rows])
    true_e = np.array([row.true_elevation_deg for row in rows])
    pred_d = np.array([getattr(row, f"{prefix}_distance_m") for row in rows])
    pred_a = np.array([getattr(row, f"{prefix}_azimuth_deg") for row in rows])
    pred_e = np.array([getattr(row, f"{prefix}_elevation_deg") for row in rows])
    d_err = pred_d - true_d
    a_err = _angular_error_deg(pred_a, true_a)
    e_err = _angular_error_deg(pred_e, true_e)
    euclidean = np.linalg.norm(
        _spherical_to_cartesian(pred_d, pred_a, pred_e) - _spherical_to_cartesian(true_d, true_a, true_e),
        axis=1,
    )
    return {
        "distance_mae_m": float(np.mean(np.abs(d_err))),
        "azimuth_mae_deg": float(np.mean(np.abs(a_err))),
        "elevation_mae_deg": float(np.mean(np.abs(e_err))),
        "euclidean_mae_m": float(np.mean(euclidean)),
        "combined_normalised_error": float(
            np.mean(
                (
                    np.abs(d_err) / 5.0
                    + np.abs(a_err) / AZIMUTH_LIMIT_DEG
                    + np.abs(e_err) / ELEVATION_LIMIT_DEG
                )
                / 3.0
            )
        ),
    }


def _run_condition(
    condition: AcousticCondition,
    targets: list[dict[str, float]],
    *,
    acoustic_config: GlobalConfig,
    distance_config: GlobalConfig,
    distance_variant: fdm.PathwayVariant,
    azimuth_config: GlobalConfig,
    elevation_config: GlobalConfig,
    elevation_baseline: np.ndarray,
    elevation_params: dict[str, float],
    noise_std: float,
) -> dict[str, object]:
    """Run fixed direct and CANN readouts for one acoustic condition."""
    rows: list[FixedPrediction] = []
    runtime_s = []
    for index, target in enumerate(targets):
        rng = torch.Generator().manual_seed(RNG_SEED + index + 1000 * CONDITIONS.index(condition))
        start = time.perf_counter()
        receive = _simulate_environment_echo(
            acoustic_config,
            target["distance_m"],
            target["azimuth_deg"],
            target["elevation_deg"],
            condition=condition,
            noise_std=noise_std,
            rng=rng,
        )
        direct_d, cann_d, _ = _distance_prediction_from_receive(distance_config, distance_variant, receive)
        direct_a, cann_a, _ = _azimuth_prediction_from_receive(azimuth_config, receive)
        direct_e, cann_e, _ = _elevation_prediction_from_receive(
            elevation_config,
            elevation_baseline,
            elevation_params,
            receive,
            target["azimuth_deg"],
        )
        runtime_s.append(time.perf_counter() - start)
        rows.append(
            FixedPrediction(
                true_distance_m=target["distance_m"],
                true_azimuth_deg=target["azimuth_deg"],
                true_elevation_deg=target["elevation_deg"],
                direct_distance_m=direct_d,
                direct_azimuth_deg=direct_a,
                direct_elevation_deg=direct_e,
                cann_distance_m=cann_d,
                cann_azimuth_deg=cann_a,
                cann_elevation_deg=cann_e,
            )
        )
    return {
        "condition": condition.key,
        "name": condition.name,
        "num_samples": len(rows),
        "runtime_seconds_per_sample": float(np.mean(runtime_s)),
        "direct_metrics": _metrics(rows, "direct"),
        "cann_metrics": _metrics(rows, "cann"),
        "predictions": [row.__dict__ for row in rows],
    }


def _plot_waveform_examples(config: GlobalConfig, noise_std: float, path: Path) -> str:
    """Plot clean, environmental-noise, and reverb waveforms for examples."""
    examples = [
        {"distance_m": 1.0, "azimuth_deg": -30.0, "elevation_deg": -20.0},
        {"distance_m": 3.0, "azimuth_deg": 15.0, "elevation_deg": 0.0},
        {"distance_m": 5.0, "azimuth_deg": 35.0, "elevation_deg": 25.0},
    ]
    time_ms = np.arange(config.signal_samples) / config.sample_rate_hz * 1_000.0
    fig, axes = plt.subplots(len(examples), 1, figsize=(12.5, 8.8), sharex=True)
    colors = {"clean": "#111827", "env_noise_50db": "#2563eb", "env_noise_50db_reverb": "#dc2626"}
    for row, target in enumerate(examples):
        ax = axes[row]
        for condition in CONDITIONS:
            rng = torch.Generator().manual_seed(20_000 + row * 100 + CONDITIONS.index(condition))
            receive = _simulate_environment_echo(
                config,
                target["distance_m"],
                target["azimuth_deg"],
                target["elevation_deg"],
                condition=condition,
                noise_std=noise_std,
                rng=rng,
            )
            ax.plot(
                time_ms,
                receive[1].detach().cpu().numpy(),
                linewidth=1.1,
                alpha=0.86,
                color=colors[condition.key],
                label=condition.name if row == 0 else None,
            )
        expected_ms = 2.0 * target["distance_m"] / config.speed_of_sound_m_s * 1_000.0
        ax.axvline(expected_ms, color="#64748b", linestyle=":", linewidth=1.3)
        ax.set_title(
            f"Right ear waveform: d={target['distance_m']:.1f} m, "
            f"az={target['azimuth_deg']:.0f} deg, el={target['elevation_deg']:.0f} deg"
        )
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("time (ms)")
    axes[-1].set_xlim(0.0, 36.0)
    fig.tight_layout()
    return save_figure(fig, path)


def _format_metric_row(condition: str, readout: str, metrics: dict[str, float], runtime_s: float) -> str:
    """Return one markdown metric table row."""
    return (
        f"| {condition} | {readout} | `{metrics['distance_mae_m']:.4f} m` | "
        f"`{metrics['azimuth_mae_deg']:.3f} deg` | `{metrics['elevation_mae_deg']:.3f} deg` | "
        f"`{metrics['euclidean_mae_m']:.4f} m` | `{metrics['combined_normalised_error']:.4f}` | "
        f"`{runtime_s * 1_000.0:.1f} ms` |"
    )


def _write_report(payload: dict[str, object]) -> None:
    """Write the environmental-noise diagnostic report."""
    lines = [
        "# Final Model Environmental Noise Diagnostics",
        "",
        "This report tests a different noise convention from the previous receiver-noise runs. Here noise is inserted into the returning acoustic signal after propagation delay and inverse-square attenuation, but before the direction-dependent head-shadow and comb-filter stages.",
        "",
        "## Acoustic Definition",
        "",
        "The clean returned signal is first delayed and attenuated:",
        "",
        "$$",
        "x_{att}(t)=\\frac{0.7}{L^2}x(t-L/c).",
        "$$",
        "",
        "Environmental noise is then added as:",
        "",
        "$$",
        "x_{env}(t)=x_{att}(t)+\\eta(t),\\quad \\eta(t)\\sim\\mathcal{N}(0,\\sigma_n^2).",
        "$$",
        "",
        "The noise standard deviation is call-referenced, not echo-referenced:",
        "",
        "$$",
        "\\sigma_n=\\frac{\\mathrm{RMS}(x_{call})}{10^{\\mathrm{SNR}_{call}/20}}.",
        "$$",
        "",
        f"For this diagnostic, `SNR_call = {ENVIRONMENT_SNR_DB_AT_CALL:.1f} dB`, giving `noise_std = {payload['noise_std']:.6g}`. This is a fixed environmental floor, so far echoes naturally have lower effective SNR than near echoes.",
        "",
        "The reverberant condition adds delayed echo copies before the head-shadow and comb-filter stages:",
        "",
        "$$",
        "x_{rev}(t)=x_{att}(t)+\\sum_i g_i x_{att}(t-\\Delta_i).",
        "$$",
        "",
        f"Reverb delays are `{REVERB_DELAYS_MS}` ms with gains `{REVERB_GAINS}`.",
        "",
        "## Waveform Examples",
        "",
        "![Waveform examples](../outputs/environment_noise_diagnostics/figures/waveform_examples.png)",
        "",
        "## Low-Sample Fixed-Model Smoke Test",
        "",
        f"This smoke test uses `{SAMPLES}` random targets in the constrained `0.25-5 m`, `+/-45 deg` azimuth/elevation space. It compares direct feed-forward population COM readouts against the fixed CANN readouts. No trainable SNN is used.",
        "",
        "| Condition | Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error | Runtime/sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["condition_results"]:
        lines.append(_format_metric_row(result["name"], "Direct COM, no CANN", result["direct_metrics"], result["runtime_seconds_per_sample"]))
        lines.append(_format_metric_row(result["name"], "Fixed CANN", result["cann_metrics"], result["runtime_seconds_per_sample"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is intentionally a smoke test, not a statistically stable benchmark.",
            "- Environmental noise is harsher than the previous low ambient receiver-noise case because its amplitude is set relative to the emitted call and then held fixed after echo attenuation.",
            "- Reverb tests a different failure mode: later echoes can create extra onset/coincidence evidence at incorrect delays and can distort the spectral profile used by elevation.",
            "- If the reverb condition degrades strongly, the likely next fixes are stronger DNLL-style late suppression, echo-window gating, and multi-hypothesis tracking rather than only changing the final readout.",
            "",
            "## Generated Files",
            "",
            "- `waveform_examples`: `final_model/outputs/environment_noise_diagnostics/figures/waveform_examples.png`",
            "- `results`: `final_model/outputs/environment_noise_diagnostics/results.json`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the environmental-noise diagnostics."""
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    start = time.perf_counter()
    with final.temporary_model_constants(
        channels=CHANNELS,
        distance_bins=DISTANCE_BINS,
        angular_bins=ANGULAR_BINS,
        elevation_limit_deg=ELEVATION_LIMIT_DEG,
    ):
        acoustic_config = final.make_elevation_config(CHANNELS, 0.25, 5.0)
        distance_config = final.make_distance_config(CHANNELS, 0.25, 5.0)
        azimuth_config = final.make_azimuth_config(CHANNELS, 5.0)
        elevation_config = final.make_elevation_config(CHANNELS, 0.25, 5.0)
        distance_variant = final.make_distance_variant(distance_config, CHANNELS, 0.25, 5.0)
        elevation_params, elevation_baseline = final.tune_elevation_calibration(elevation_config)
        noise_std = _call_referenced_noise_std(acoustic_config, ENVIRONMENT_SNR_DB_AT_CALL)
        targets = _targets()
        waveform_path = _plot_waveform_examples(acoustic_config, noise_std, FIGURE_DIR / "waveform_examples.png")
        condition_results = [
            _run_condition(
                condition,
                targets,
                acoustic_config=acoustic_config,
                distance_config=distance_config,
                distance_variant=distance_variant,
                azimuth_config=azimuth_config,
                elevation_config=elevation_config,
                elevation_baseline=elevation_baseline,
                elevation_params=elevation_params,
                noise_std=noise_std,
            )
            for condition in CONDITIONS
        ]
    payload = {
        "experiment": "environment_noise_diagnostics",
        "elapsed_seconds": time.perf_counter() - start,
        "noise_snr_db_at_call": ENVIRONMENT_SNR_DB_AT_CALL,
        "noise_std": noise_std,
        "reverb_delays_ms": REVERB_DELAYS_MS,
        "reverb_gains": REVERB_GAINS,
        "targets": targets,
        "condition_results": condition_results,
        "artifacts": {"waveform_examples": waveform_path},
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload)
    return payload


if __name__ == "__main__":
    main()
    print(REPORT_PATH)
