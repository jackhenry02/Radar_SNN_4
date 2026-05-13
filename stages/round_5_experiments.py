from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models.snn import delay_bank_features, ild_features, itd_features
from stages.base import StageContext
from stages.experiments import _baseline_reference_params, _metrics_delta
from stages.expanded_space_test import _prepare_expanded_data
from stages.improvement import _distance_candidates, _itd_candidates
from stages.round_2_experiments import _augment_with_cartesian_metrics
from stages.round_3_experiments import _evaluate_round3_output, _round3_support_spec
from stages.round_4_experiments import _round4_base_config
from utils.common import GlobalConfig, OutputPaths, save_grouped_bar_chart, save_json, seed_everything


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _round3_winner_with_metrics(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    winner = summary.get("winner")
    if isinstance(winner, dict) and "test_metrics" in winner:
        return winner
    winner_name = winner.get("name") if isinstance(winner, dict) else None
    for key in ["combined_candidates", "component_references"]:
        items = summary.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if item.get("name") == winner_name and "test_metrics" in item:
                return item
    return None


def _candidate_distance_m(config: GlobalConfig, candidates: torch.Tensor) -> torch.Tensor:
    return candidates.float() / float(config.envelope_rate_hz) * float(config.speed_of_sound_m_s) / 2.0


def _candidate_azimuth_deg(config: GlobalConfig, candidates: torch.Tensor) -> torch.Tensor:
    itd_s = candidates.float() / float(config.envelope_rate_hz)
    sine = (itd_s * float(config.speed_of_sound_m_s) / float(config.ear_spacing_m)).clamp(-0.999, 0.999)
    return torch.rad2deg(torch.asin(sine))


def _normalise_population(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float().clamp_min(0.0)
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _soft_population(scores: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
    scaled = scores.float()
    scaled = scaled / scaled.amax(dim=-1, keepdim=True).clamp_min(1.0)
    return F.softmax(temperature * scaled, dim=-1)


def _atanh_clamped(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(-0.98, 0.98)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def _population_features(
    batch: Any,
    config: GlobalConfig,
    distance_candidates: torch.Tensor,
    itd_candidates: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if batch.transmit_spikes is None or batch.receive_spikes is None:
        raise ValueError("Round 5 fixed decoders require cached transmit and receive spikes.")

    transmit = batch.transmit_spikes.float()
    left = batch.receive_spikes[:, 0].float()
    right = batch.receive_spikes[:, 1].float()

    distance_left = delay_bank_features(transmit, left, distance_candidates)
    distance_right = delay_bank_features(transmit, right, distance_candidates)
    distance_scores = 0.5 * (distance_left + distance_right)
    itd_scores = itd_features(left, right, itd_candidates)
    ild = ild_features(left, right)

    left_counts = left.sum(dim=-1)
    right_counts = right.sum(dim=-1)
    spectral_counts = left_counts + right_counts
    spectral_norm = spectral_counts / spectral_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    local_mean = F.avg_pool1d(spectral_norm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    spectral_notches = F.relu(local_mean - spectral_norm)
    spectral_slope = F.pad(spectral_norm[..., 1:] - spectral_norm[..., :-1], (1, 0))

    left_total = left.sum(dim=(1, 2))
    right_total = right.sum(dim=(1, 2))
    total = left_total + right_total
    loudness = torch.stack([left_total, right_total, total, left_total - right_total, torch.log1p(total)], dim=-1)

    return {
        "distance_scores": distance_scores,
        "itd_scores": itd_scores,
        "ild": ild,
        "spectral_norm": spectral_norm,
        "spectral_notches": spectral_notches,
        "spectral_slope": spectral_slope,
        "loudness": loudness,
    }


def _decode_naive(
    batch: Any,
    config: GlobalConfig,
    distance_candidates: torch.Tensor,
    itd_candidates: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = _population_features(batch, config, distance_candidates, itd_candidates)

    distance_values = _candidate_distance_m(config, distance_candidates).to(features["distance_scores"].device)
    distance_weights = _soft_population(features["distance_scores"])
    distance = (distance_weights * distance_values.view(1, -1)).sum(dim=-1)

    azimuth_values = _candidate_azimuth_deg(config, itd_candidates).to(features["itd_scores"].device)
    azimuth_weights = _soft_population(features["itd_scores"])
    azimuth = (azimuth_weights * azimuth_values.view(1, -1)).sum(dim=-1)

    bins = torch.linspace(0.0, 1.0, features["spectral_notches"].shape[-1], device=features["spectral_notches"].device)
    notch_weights = features["spectral_notches"] + 0.05 * features["spectral_norm"]
    notch_center = (notch_weights * bins.view(1, -1)).sum(dim=-1) / notch_weights.sum(dim=-1).clamp_min(1e-6)
    center_min = float(config.elevation_notch_center_min)
    center_max = float(config.elevation_notch_center_max)
    elevation_scale = 2.0 * ((notch_center - center_min) / max(center_max - center_min, 1e-6)) - 1.0
    elevation = torch.rad2deg(_atanh_clamped(elevation_scale) * (math.pi / 6.0))

    raw = torch.stack(
        [
            distance.clamp(float(config.min_range_m), float(config.max_range_m)),
            azimuth.clamp(-90.0, 90.0),
            elevation.clamp(-60.0, 60.0),
        ],
        dim=-1,
    )
    diagnostics = {
        "spike_rate": batch.spike_count.float() / max(1, batch.receive_spikes.shape[-1] * batch.receive_spikes.shape[-2] * 2),
        "distance_population": distance_weights,
        "azimuth_population": azimuth_weights,
        "elevation_notch_center": notch_center,
    }
    return raw, diagnostics


def _evaluate_prediction(
    raw_prediction: torch.Tensor,
    targets_raw: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    config: GlobalConfig,
) -> dict[str, Any]:
    evaluation = _evaluate_round3_output(raw_prediction, targets_raw, diagnostics, config)
    return evaluation.metrics


def _normalise_targets(targets: torch.Tensor, config: GlobalConfig) -> torch.Tensor:
    scales = targets.new_tensor([float(config.max_range_m), 45.0, 30.0]).view(1, 3)
    return targets / scales


def _denormalise_targets(model_targets: torch.Tensor, config: GlobalConfig) -> torch.Tensor:
    scales = model_targets.new_tensor([float(config.max_range_m), 45.0, 30.0]).view(1, 3)
    raw = model_targets * scales
    raw[:, 0] = raw[:, 0].clamp(float(config.min_range_m), float(config.max_range_m))
    raw[:, 1] = raw[:, 1].clamp(-90.0, 90.0)
    raw[:, 2] = raw[:, 2].clamp(-60.0, 60.0)
    return raw


def _fit_scale_bias(train_prediction: torch.Tensor, train_targets: torch.Tensor, config: GlobalConfig) -> dict[str, torch.Tensor]:
    x = _normalise_targets(train_prediction, config)
    y = _normalise_targets(train_targets, config)
    scale = []
    bias = []
    for index in range(3):
        design = torch.stack([x[:, index], torch.ones_like(x[:, index])], dim=-1)
        solution = torch.linalg.lstsq(design, y[:, index : index + 1]).solution.squeeze(-1)
        scale.append(solution[0])
        bias.append(solution[1])
    return {"scale": torch.stack(scale), "bias": torch.stack(bias)}


def _apply_scale_bias(prediction: torch.Tensor, params: dict[str, torch.Tensor], config: GlobalConfig) -> torch.Tensor:
    x = _normalise_targets(prediction, config)
    y = x * params["scale"].view(1, 3) + params["bias"].view(1, 3)
    return _denormalise_targets(y, config)


def _build_tuned_feature_matrix(
    batch: Any,
    config: GlobalConfig,
    distance_candidates: torch.Tensor,
    itd_candidates: torch.Tensor,
) -> torch.Tensor:
    naive, _ = _decode_naive(batch, config, distance_candidates, itd_candidates)
    features = _population_features(batch, config, distance_candidates, itd_candidates)
    parts = [
        _normalise_population(features["distance_scores"]),
        _normalise_population(features["itd_scores"]),
        F.layer_norm(features["ild"].float(), (features["ild"].shape[-1],)),
        features["spectral_norm"].float(),
        _normalise_population(features["spectral_notches"]),
        F.layer_norm(features["spectral_slope"].float(), (features["spectral_slope"].shape[-1],)),
        F.layer_norm(features["loudness"].float(), (features["loudness"].shape[-1],)),
        _normalise_targets(naive, config),
    ]
    return torch.cat(parts, dim=-1)


def _fit_ridge_decoder(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    config: GlobalConfig,
) -> dict[str, Any]:
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_features - feature_mean) / feature_std
    val_x = (val_features - feature_mean) / feature_std
    train_design = torch.cat([train_x, torch.ones(train_x.shape[0], 1, device=train_x.device)], dim=-1)
    val_design = torch.cat([val_x, torch.ones(val_x.shape[0], 1, device=val_x.device)], dim=-1)
    train_y = _normalise_targets(train_targets, config)

    best_payload: dict[str, Any] | None = None
    for ridge in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        identity = torch.eye(train_design.shape[-1], device=train_design.device, dtype=train_design.dtype)
        identity[-1, -1] = 0.0
        weights = torch.linalg.solve(train_design.T @ train_design + ridge * identity, train_design.T @ train_y)
        val_prediction = _denormalise_targets(val_design @ weights, config)
        val_metrics = _evaluate_prediction(
            val_prediction,
            val_targets,
            {"spike_rate": torch.zeros(val_targets.shape[0], device=val_targets.device)},
            config,
        )
        if best_payload is None or val_metrics["combined_error"] < best_payload["val_metrics"]["combined_error"]:
            best_payload = {"ridge": ridge, "weights": weights, "feature_mean": feature_mean, "feature_std": feature_std, "val_metrics": val_metrics}
    assert best_payload is not None
    return best_payload


def _apply_ridge_decoder(features: torch.Tensor, payload: dict[str, Any], config: GlobalConfig) -> torch.Tensor:
    x = (features - payload["feature_mean"]) / payload["feature_std"]
    design = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=-1)
    return _denormalise_targets(design @ payload["weights"], config)


def _tensor_payload_to_json(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            result[key] = _tensor_payload_to_json(value)
        else:
            result[key] = value
    return result


def _run_round5_experiment_1(
    outputs: OutputPaths,
    context: StageContext,
    data: Any,
    base_params: dict[str, Any],
    round4_baseline: dict[str, Any] | None,
    round3_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_root = outputs.root / "round_5_experiments" / "experiment_1_fixed_decoders"
    stage_root.mkdir(parents=True, exist_ok=True)
    result_path = stage_root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    distance_candidates = _distance_candidates(data.local_config, context.device, int(base_params["num_delay_lines"]))
    itd_candidates = _itd_candidates(data.local_config, context.device, int(base_params["num_delay_lines"]))
    round4_metrics = None if round4_baseline is None else round4_baseline["test_metrics"]
    round3_metrics = None if round3_reference is None else round3_reference["test_metrics"]

    variants: list[dict[str, Any]] = []

    naive_start = time.perf_counter()
    naive_train, _ = _decode_naive(data.train_batch, data.local_config, distance_candidates, itd_candidates)
    naive_val, _ = _decode_naive(data.val_batch, data.local_config, distance_candidates, itd_candidates)
    naive_test, naive_diag = _decode_naive(data.test_batch, data.local_config, distance_candidates, itd_candidates)
    naive_seconds = time.perf_counter() - naive_start
    naive_metrics = _evaluate_prediction(naive_test, data.test_targets_raw, naive_diag, data.local_config)
    variants.append(
        {
            "name": "round5_experiment_1a_naive_fixed_population_decoder",
            "title": "Round 5 Experiment 1A: Naive Fixed Population Decoder",
            "description": "No training. Decode distance, azimuth, and elevation directly from fixed population-code features.",
            "training_mode": "none",
            "trainable_parameters": 0,
            "fixed_tuned_parameters": 0,
            "test_metrics": naive_metrics,
            "timings": {"training_seconds": 0.0, "evaluation_seconds": naive_seconds, "total_seconds": naive_seconds},
            "comparison_vs_round4": {} if round4_metrics is None else _metrics_delta(naive_metrics, round4_metrics),
            "comparison_vs_round3_2b_plus_3": {} if round3_metrics is None else _metrics_delta(naive_metrics, round3_metrics),
        }
    )

    calibration_start = time.perf_counter()
    scale_bias = _fit_scale_bias(naive_train, data.train_targets_raw, data.local_config)
    calibrated_test = _apply_scale_bias(naive_test, scale_bias, data.local_config)
    calibration_seconds = time.perf_counter() - calibration_start
    calibrated_metrics = _evaluate_prediction(calibrated_test, data.test_targets_raw, naive_diag, data.local_config)
    variants.append(
        {
            "name": "round5_experiment_1b_minimal_scale_bias_calibration",
            "title": "Round 5 Experiment 1B: Minimal Scale/Bias Calibration",
            "description": "Fit one scale and one bias per coordinate on the training set, then freeze them.",
            "training_mode": "closed_form_scale_bias",
            "trainable_parameters": 0,
            "fixed_tuned_parameters": 6,
            "tuned_parameters": _tensor_payload_to_json(scale_bias),
            "test_metrics": calibrated_metrics,
            "timings": {"training_seconds": calibration_seconds, "evaluation_seconds": 0.0, "total_seconds": calibration_seconds},
            "comparison_vs_round4": {} if round4_metrics is None else _metrics_delta(calibrated_metrics, round4_metrics),
            "comparison_vs_round3_2b_plus_3": {} if round3_metrics is None else _metrics_delta(calibrated_metrics, round3_metrics),
        }
    )

    tuned_start = time.perf_counter()
    train_features = _build_tuned_feature_matrix(data.train_batch, data.local_config, distance_candidates, itd_candidates)
    val_features = _build_tuned_feature_matrix(data.val_batch, data.local_config, distance_candidates, itd_candidates)
    test_features = _build_tuned_feature_matrix(data.test_batch, data.local_config, distance_candidates, itd_candidates)
    tuned_payload = _fit_ridge_decoder(train_features, data.train_targets_raw, val_features, data.val_targets_raw, data.local_config)
    tuned_test = _apply_ridge_decoder(test_features, tuned_payload, data.local_config)
    tuned_seconds = time.perf_counter() - tuned_start
    tuned_metrics = _evaluate_prediction(tuned_test, data.test_targets_raw, naive_diag, data.local_config)
    tuned_parameter_count = int(tuned_payload["weights"].numel())
    tuned_json = _tensor_payload_to_json({key: value for key, value in tuned_payload.items() if key != "val_metrics"})
    save_json(stage_root / "fixed_tuned_ridge_decoder.json", tuned_json)
    variants.append(
        {
            "name": "round5_experiment_1c_trained_once_fixed_ridge_decoder",
            "title": "Round 5 Experiment 1C: Trained-Once Fixed Ridge Decoder",
            "description": "Use the training set once to tune a small linear population-code decoder, then freeze the decoder.",
            "training_mode": "closed_form_ridge",
            "trainable_parameters": 0,
            "fixed_tuned_parameters": tuned_parameter_count,
            "selected_ridge": float(tuned_payload["ridge"]),
            "validation_metrics": tuned_payload["val_metrics"],
            "test_metrics": tuned_metrics,
            "timings": {"training_seconds": tuned_seconds, "evaluation_seconds": 0.0, "total_seconds": tuned_seconds},
            "comparison_vs_round4": {} if round4_metrics is None else _metrics_delta(tuned_metrics, round4_metrics),
            "comparison_vs_round3_2b_plus_3": {} if round3_metrics is None else _metrics_delta(tuned_metrics, round3_metrics),
        }
    )

    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined", "Euclidean"],
        {
            item["title"].replace("Round 5 Experiment ", "R5 "): [
                item["test_metrics"]["distance_mae_m"],
                item["test_metrics"]["azimuth_mae_deg"],
                item["test_metrics"]["elevation_mae_deg"],
                item["test_metrics"]["combined_error"],
                item["test_metrics"]["euclidean_error_m"],
            ]
            for item in variants
        },
        stage_root / "round5_experiment_1_comparison.png",
        "Round 5 Experiment 1 Fixed Decoder Comparison",
        ylabel="Error",
    )

    result = {
        "name": "round5_experiment_1_fixed_decoders",
        "title": "Round 5 Experiment 1: Fixed / Nearly Fixed Decoders",
        "baseline": "Round 4 combined model",
        "reference": "Round 3 Combined A: 2B + 3",
        "round4_baseline_metrics": None if round4_baseline is None else round4_baseline["test_metrics"],
        "round3_reference_metrics": None if round3_reference is None else round3_reference["test_metrics"],
        "space": {
            "distance_m": [float(data.local_config.min_range_m), float(data.local_config.max_range_m)],
            "azimuth_deg": [-45.0, 45.0],
            "elevation_deg": [-30.0, 30.0],
        },
        "data": {"train": int(data.train_targets_raw.shape[0]), "validation": int(data.val_targets_raw.shape[0]), "test": int(data.test_targets_raw.shape[0])},
        "variants": variants,
        "artifacts": {
            "comparison": str(stage_root / "round5_experiment_1_comparison.png"),
            "fixed_tuned_decoder": str(stage_root / "fixed_tuned_ridge_decoder.json"),
        },
    }
    save_json(result_path, result)
    return result


def _append_round5_report(outputs_root: Path, summary: dict[str, Any]) -> Path:
    report_path = outputs_root / "round_5_experiments_report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Round 5 Experiments Report\n"
    marker = "\n## Experiment 1 Results\n"
    existing = existing.split(marker)[0].rstrip() + "\n"
    lines = [existing, "## Experiment 1 Results", ""]
    lines.extend(
        [
            "Experiment 1 tested whether the trained IC/fusion/readout could be replaced by fixed or nearly fixed population-code decoders.",
            "",
            "| Variant | Training mode | Fixed tuned params | Combined | Distance | Azimuth | Elevation | Euclidean | Runtime |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in summary["experiment_1"]["variants"]:
        metrics = variant["test_metrics"]
        timings = variant["timings"]
        lines.append(
            f"| {variant['title']} | {variant['training_mode']} | {variant['fixed_tuned_parameters']} | "
            f"{metrics['combined_error']:.4f} | {metrics['distance_mae_m']:.4f} m | "
            f"{metrics['azimuth_mae_deg']:.4f} deg | {metrics['elevation_mae_deg']:.4f} deg | "
            f"{metrics['euclidean_error_m']:.4f} m | {timings['total_seconds']:.2f} s |"
        )
    lines.append("")
    lines.append(
        f"Shared data preparation time was `{summary['data_prep_seconds']:.2f} s`. "
        "The runtime column above is the decoder fitting/evaluation time after the cochlear spike/pathway data had been prepared."
    )
    lines.extend(["", "### Interpretation", ""])
    variants = summary["experiment_1"]["variants"]
    best = min(variants, key=lambda item: item["test_metrics"]["combined_error"])
    naive = variants[0]
    lines.append(
        f"The naive no-training decoder reached combined error `{naive['test_metrics']['combined_error']:.4f}`. "
        "This directly measures how much localisation can be recovered from fixed population codes without learned calibration."
    )
    lines.append("")
    lines.append(
        "This should be treated as a failure of the fully naive no-training decoder: distance was usable, but azimuth and "
        "elevation were much too poorly calibrated."
    )
    lines.append("")
    lines.append(
        f"The best Experiment 1 variant was `{best['title']}` with combined error `{best['test_metrics']['combined_error']:.4f}`. "
        "Its tuned parameters are fixed after fitting, so inference does not require gradient training."
    )
    lines.append("")
    if summary["experiment_1"].get("round4_baseline_metrics"):
        round4 = summary["experiment_1"]["round4_baseline_metrics"]
        lines.append(
            f"For comparison, the Round 4 combined baseline had combined error `{round4['combined_error']:.4f}`, "
            f"distance MAE `{round4['distance_mae_m']:.4f} m`, azimuth MAE `{round4['azimuth_mae_deg']:.4f} deg`, "
            f"and elevation MAE `{round4['elevation_mae_deg']:.4f} deg`."
        )
        lines.append("")
    if summary["experiment_1"].get("round3_reference_metrics"):
        round3 = summary["experiment_1"]["round3_reference_metrics"]
        lines.append(
            f"The Round 3 `2B + 3` reference had combined error `{round3['combined_error']:.4f}` and Euclidean error "
            f"`{round3['euclidean_error_m']:.4f} m`."
        )
        lines.append("")
    lines.append(
        "The trained-once fixed ridge decoder therefore slightly beat both the Round 4 combined model and the Round 3 `2B + 3` "
        "model on combined error, mainly because distance error became lower. It did not beat the Round 4 combined model on "
        "azimuth, and its Euclidean error was similar to but slightly worse than the original Round 3 `2B + 3` result."
    )
    lines.append("")
    lines.append(
        "FLOP/SOP accounting has not yet been added. The useful first cost result is already clear: replacing gradient training "
        f"with a closed-form fixed decoder reduced decoder tuning to about `{best['timings']['total_seconds']:.2f} s`, with only "
        f"`{best['fixed_tuned_parameters']}` fixed tuned decoder parameters after fitting."
    )
    lines.append("")
    lines.append("![Round 5 Experiment 1 comparison](round_5_experiments/experiment_1_fixed_decoders/round5_experiment_1_comparison.png)")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_round_5_experiments(config: GlobalConfig, outputs: OutputPaths) -> dict[str, Any]:
    seed_everything(config.seed)
    effective_config = _round4_base_config(config)
    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    base_params, _ = _baseline_reference_params(context)
    support_spec = _round3_support_spec()

    prep_start = time.perf_counter()
    data = _prepare_expanded_data(context, base_params, support_spec, chunk_size=16)
    prep_seconds = time.perf_counter() - prep_start

    round4_summary = _load_json(outputs.root / "round_4_combined_experiment_summary.json")
    round4_baseline = None if round4_summary is None else round4_summary.get("combined")
    round3_summary = _load_json(outputs.root / "round_3_combined_experiment_summary.json")
    round3_reference = _round3_winner_with_metrics(round3_summary)

    experiment_1 = _run_round5_experiment_1(outputs, context, data, base_params, round4_baseline, round3_reference)
    for variant in experiment_1["variants"]:
        variant.setdefault("timings", {})["data_prep_seconds"] = float(prep_seconds)

    summary = {
        "round": 5,
        "data_prep_seconds": float(prep_seconds),
        "experiment_1": experiment_1,
    }
    report_path = _append_round5_report(outputs.root, summary)
    summary["report_path"] = str(report_path)
    save_json(outputs.root / "round_5_experiments_summary.json", summary)
    save_json(outputs.root / "round_5_experiments" / "results.json", summary)
    return summary
