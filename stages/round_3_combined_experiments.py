from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from models.round2_variants import AllRound2CombinedModel
from models.round3_variants import LIFCoincidenceRound3Encoder, NotchDetectorElevationEncoder, OrthogonalNotchCombinedEncoder
from stages.base import StageContext
from stages.combined_experiment import _save_coordinate_error_profiles, _save_prediction_cache
from stages.expanded_space_test import _prepare_expanded_data
from stages.experiments import _metrics_delta
from stages.improved_experiments import _is_accepted, _prepare_target_bundle
from stages.round_3_experiments import (
    Round3ExperimentSpec,
    _evaluate_round3_output,
    _instantiate_base_encoder,
    _prepare_distance01_target_bundle,
    _prepare_sincos_target_bundle,
    _round3_base_config,
    _round3_loss_and_decode,
    _round3_support_spec,
    _round3_variant_config,
    _save_round3_outputs,
)
from stages.training_improved_experiments import EnhancedTrainingConfig
from utils.common import GlobalConfig, save_grouped_bar_chart, save_json, seed_everything


def _combined_specs() -> list[Round3ExperimentSpec]:
    return [
        Round3ExperimentSpec(
            name="round3_combined_experiment_2b_plus_3",
            title="Round 3 Combined A: 2B + 3",
            description=(
                "Combine the accepted elevation moving-notch detector model from 2B with the sine/cosine angle output "
                "coding from Experiment 3."
            ),
            rationale=(
                "This tests whether the best accepted elevation cue model benefits further from the better angular output code."
            ),
            variant="combined_2b_3",
            output_mode="sincos",
            implemented_steps=[
                "Use the moving-notch elevation cue and elevation notch-detector residual from 2B.",
                "Keep the rest of the round-2 combined architecture unchanged.",
                "Replace direct angle regression with sine/cosine output coding and unit-circle regularization from 3.",
            ],
            analysis_focus=[
                "Whether adding sine/cosine output coding improves the already-strong 2B angular model.",
                "Whether the combined model beats both 2B and 3 individually.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35, "unit_penalty_weight": 0.1},
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round3_combined_experiment_3c_plus_3",
            title="Round 3 Combined B: 3C + 3",
            description=(
                "Combine the accepted orthogonal azimuth/elevation notch model from 3C with the sine/cosine angle "
                "output coding from Experiment 3."
            ),
            rationale=(
                "This tests whether the orthogonal cue separation from 3C benefits further from the better angular output code."
            ),
            variant="combined_3c_3",
            output_mode="sincos",
            implemented_steps=[
                "Use the orthogonal combined azimuth/elevation notch cues and branch-specific detectors from 3C.",
                "Keep distance and the fusion stack unchanged.",
                "Replace direct angle regression with sine/cosine output coding and unit-circle regularization from 3.",
            ],
            analysis_focus=[
                "Whether sine/cosine output coding helps the orthogonal notch model improve both angles further.",
                "Whether the combined model beats both 3C and 3 individually.",
            ],
            training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35, "unit_penalty_weight": 0.1},
            data_variant="orthogonal_combined_notches",
        ),
    ]


def _full_combined_spec(winner_variant: str, winner_title: str, data_variant: str) -> Round3ExperimentSpec:
    return Round3ExperimentSpec(
        name="round3_combined_experiment_full_winner_plus_1",
        title="Round 3 Combined C: Winner + 1",
        description=(
            f"Take the better of the first two combined models ({winner_title}) and add the accepted trainable LIF "
            "coincidence detectors from Experiment 1."
        ),
        rationale=(
            "This tests whether the best combined angular model also benefits from explicitly trainable spike-domain "
            "coincidence detection in the timing pathways."
        ),
        variant=f"full_{winner_variant}_plus_1",
        output_mode="sincos",
        implemented_steps=[
            f"Start from the winning combined model: {winner_title}.",
            "Add the trainable LIF distance and ITD coincidence banks from Experiment 1 on top of that encoder.",
            "Keep the sine/cosine output coding from Experiment 3.",
        ],
        analysis_focus=[
            "Whether adding Experiment 1 improves the winning combined angular model further.",
            "Whether the full combined model beats the round-3 control and all relevant component experiments.",
        ],
        training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35, "unit_penalty_weight": 0.1},
        data_variant=data_variant,
    )


def _load_round3_references(outputs_root: Path) -> dict[str, dict[str, Any]]:
    summary_path = outputs_root / "round_3_experiments_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    references = {summary["control"]["name"]: summary["control"]}
    for item in summary["experiments"]:
        references[item["name"]] = item
    return references


def _instantiate_combined_model(
    data: Any,
    params: dict[str, Any],
    spec: Round3ExperimentSpec,
) -> AllRound2CombinedModel:
    base_encoder = _instantiate_base_encoder(data, params)

    if spec.variant == "combined_2b_3":
        encoder = NotchDetectorElevationEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    elif spec.variant == "combined_3c_3":
        encoder = OrthogonalNotchCombinedEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    elif spec.variant == "full_combined_2b_3_plus_1":
        elevation_encoder = NotchDetectorElevationEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
        encoder = LIFCoincidenceRound3Encoder(
            base_encoder=elevation_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            distance_candidates=base_encoder.distance_candidates,
            itd_candidates=base_encoder.itd_candidates,
        )
    elif spec.variant == "full_combined_3c_3_plus_1":
        orthogonal_encoder = OrthogonalNotchCombinedEncoder(
            base_encoder=base_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
        encoder = LIFCoincidenceRound3Encoder(
            base_encoder=orthogonal_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            distance_candidates=base_encoder.distance_candidates,
            itd_candidates=base_encoder.itd_candidates,
        )
    else:
        raise ValueError(f"Unsupported combined experiment variant '{spec.variant}'.")

    return AllRound2CombinedModel(
        encoder=encoder,
        hidden_dim=int(params["hidden_dim"]),
        output_dim=5,
        num_steps=int(params["num_steps"]),
        beta=float(params["membrane_beta"]),
        threshold=float(params["fusion_threshold"]),
        reset_mechanism=str(params["reset_mechanism"]),
    ).to(data.train_targets_raw.device)


def _reference_labels_for_spec(spec: Round3ExperimentSpec, winner_name: str | None = None) -> list[str]:
    if spec.name == "round3_combined_experiment_2b_plus_3":
        return [
            "round3_control_round2_combined_140db_unnormalized",
            "round3_experiment_2b_moving_notch_plus_detectors",
            "round3_experiment_3_sincos_angle_regression",
        ]
    if spec.name == "round3_combined_experiment_3c_plus_3":
        return [
            "round3_control_round2_combined_140db_unnormalized",
            "round3_experiment_3c_orthogonal_combined_notches",
            "round3_experiment_3_sincos_angle_regression",
        ]
    if spec.name == "round3_combined_experiment_full_winner_plus_1":
        labels = [
            "round3_control_round2_combined_140db_unnormalized",
            "round3_experiment_1_lif_coincidence_detectors",
            "round3_experiment_3_sincos_angle_regression",
        ]
        if winner_name is not None:
            labels.append(winner_name)
            if winner_name == "round3_combined_experiment_2b_plus_3":
                labels.append("round3_experiment_2b_moving_notch_plus_detectors")
            if winner_name == "round3_combined_experiment_3c_plus_3":
                labels.append("round3_experiment_3c_orthogonal_combined_notches")
        return labels
    return ["round3_control_round2_combined_140db_unnormalized"]


def _run_single_spec(
    outputs_root: Path,
    stage_root: Path,
    context: StageContext,
    training_config: EnhancedTrainingConfig,
    base_params: dict[str, Any],
    spec: Round3ExperimentSpec,
    references: dict[str, dict[str, Any]],
    data_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result_path = stage_root / spec.name / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    if spec.data_variant not in data_cache:
        variant_config = _round3_variant_config(context.config, spec.data_variant)
        variant_context = StageContext(config=variant_config, device=context.device, outputs=context.outputs, shared=context.shared)
        prep_start = time.perf_counter()
        data = _prepare_expanded_data(variant_context, base_params, _round3_support_spec(), chunk_size=16)
        data_cache[spec.data_variant] = {
            "data": data,
            "baseline_bundle": _prepare_target_bundle(data),
            "sincos_bundle": _prepare_sincos_target_bundle(data),
            "distance01_bundle": _prepare_distance01_target_bundle(data, data.local_config),
            "data_prep_seconds": time.perf_counter() - prep_start,
        }

    payload = data_cache[spec.data_variant]
    data = payload["data"]
    baseline_bundle = payload["baseline_bundle"]
    sincos_bundle = payload["sincos_bundle"]
    distance01_bundle = payload["distance01_bundle"]
    data_prep_seconds = float(payload["data_prep_seconds"])

    seed_everything(context.config.seed + hash(spec.name) % 10_000)
    print(f"[round_3_combined] running {spec.name} on cpu", flush=True)

    model = _instantiate_combined_model(data, base_params, spec)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(base_params["learning_rate"]) * float(spec.training_overrides.get("learning_rate_scale", 1.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        threshold=training_config.scheduler_threshold,
        min_lr=training_config.scheduler_min_lr,
    )

    batch_size = int(spec.training_overrides.get("batch_size", 8))
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    best_val_combined = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    total_start = time.perf_counter()
    training_start = time.perf_counter()
    for epoch in range(training_config.max_epochs):
        model.train()
        permutation = torch.randperm(data.train_targets_raw.shape[0], device=data.train_targets_raw.device)
        batch_losses: list[float] = []
        for start in range(0, data.train_targets_raw.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = data.train_batch.index_select(indices)
            batch_targets_raw = data.train_targets_raw[indices]
            batch_baseline_model = baseline_bundle.train_model[indices]
            batch_sincos_distance_model = sincos_bundle.train_distance_model[indices]
            batch_distance01 = distance01_bundle.train_distance01[indices]
            batch_distance01_angle_model = distance01_bundle.train_angle_model[indices]

            optimizer.zero_grad(set_to_none=True)
            output_model, diagnostics = model(batch_features)
            loss, _, _ = _round3_loss_and_decode(
                spec,
                output_model,
                batch_targets_raw,
                diagnostics,
                data.local_config,
                base_params,
                baseline_bundle,
                batch_baseline_model,
                sincos_bundle,
                batch_sincos_distance_model,
                distance01_bundle,
                batch_distance01,
                batch_distance01_angle_model,
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss_history.append(float(sum(batch_losses) / max(1, len(batch_losses))))

        model.eval()
        with torch.no_grad():
            val_output_model, val_diagnostics = model(data.val_batch)
            val_loss, val_raw_prediction, _ = _round3_loss_and_decode(
                spec,
                val_output_model,
                data.val_targets_raw,
                val_diagnostics,
                data.local_config,
                base_params,
                baseline_bundle,
                baseline_bundle.val_model,
                sincos_bundle,
                sincos_bundle.val_distance_model,
                distance01_bundle,
                distance01_bundle.val_distance01,
                distance01_bundle.val_angle_model,
            )
            val_eval = _evaluate_round3_output(val_raw_prediction, data.val_targets_raw, val_diagnostics, data.local_config)
        val_loss_history.append(float(val_loss.item()))
        scheduler.step(float(val_eval.metrics["combined_error"]))

        if float(val_eval.metrics["combined_error"]) < best_val_combined - training_config.early_stopping_min_delta:
            best_val_combined = float(val_eval.metrics["combined_error"])
            best_epoch = epoch
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.early_stopping_patience:
            break

    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best_state)

    evaluation_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        test_output_model, test_diagnostics = model(data.test_batch)
        _, test_raw_prediction, test_summary = _round3_loss_and_decode(
            spec,
            test_output_model,
            data.test_targets_raw,
            test_diagnostics,
            data.local_config,
            base_params,
            baseline_bundle,
            baseline_bundle.test_model,
            sincos_bundle,
            sincos_bundle.test_distance_model,
            distance01_bundle,
            distance01_bundle.test_distance01,
            distance01_bundle.test_angle_model,
        )
        if spec.output_mode == "sincos":
            from stages.round_3_experiments import _decode_sincos_output

            _, decoded_aux = _decode_sincos_output(test_output_model, sincos_bundle)
            test_diagnostics = {
                **test_diagnostics,
                "output_az_raw_norm": decoded_aux["az_raw_norm"].detach(),
                "output_el_raw_norm": decoded_aux["el_raw_norm"].detach(),
            }
        test_eval = _evaluate_round3_output(test_raw_prediction, data.test_targets_raw, test_diagnostics, data.local_config)
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    control_metrics = references["round3_control_round2_combined_140db_unnormalized"]["test_metrics"]
    comparison = _metrics_delta(test_eval.metrics, control_metrics)
    cartesian_delta = {
        "x_mae_delta": float(test_eval.metrics["x_mae_m"] - control_metrics["x_mae_m"]),
        "y_mae_delta": float(test_eval.metrics["y_mae_m"] - control_metrics["y_mae_m"]),
        "z_mae_delta": float(test_eval.metrics["z_mae_m"] - control_metrics["z_mae_m"]),
        "euclidean_error_delta": float(test_eval.metrics["euclidean_error_m"] - control_metrics["euclidean_error_m"]),
    }
    accepted = _is_accepted(test_eval.metrics, control_metrics)

    artifacts = _save_round3_outputs(
        outputs_root / "round_3_combined_experiments",
        spec,
        train_loss_history,
        val_loss_history,
        test_eval,
        control_metrics,
        test_diagnostics,
        data.local_config,
    )
    stage_dir = stage_root / spec.name
    prediction_cache = _save_prediction_cache(stage_dir, test_eval.predictions, data.test_targets_raw)
    coordinate_profile = _save_coordinate_error_profiles(
        Path(prediction_cache),
        stage_dir / "coordinate_error_profile.png",
        f"{spec.title} Coordinate Error Profile",
    )

    result = {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "rationale": spec.rationale,
        "implemented_steps": spec.implemented_steps,
        "analysis_focus": spec.analysis_focus,
        "output_mode": spec.output_mode,
        "data_variant": spec.data_variant,
        "accepted": accepted,
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "test_metrics": {key: float(value) for key, value in test_eval.metrics.items()},
        "comparison": {key: float(value) for key, value in comparison.items()},
        "cartesian_delta": {key: float(value) for key, value in cartesian_delta.items()},
        "training": {"best_epoch": best_epoch + 1, "executed_epochs": len(train_loss_history)},
        "timings": {
            "data_prep_seconds": data_prep_seconds,
            "training_seconds": float(training_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "total_seconds": float(total_seconds),
        },
        "loss_summary": test_summary,
        "artifacts": {**artifacts, "prediction_cache": prediction_cache, "coordinate_error_profile": coordinate_profile},
    }
    save_json(result_path, result)
    return result


def _round3_combined_report(
    outputs_root: Path,
    control: dict[str, Any],
    references: dict[str, dict[str, Any]],
    first_results: list[dict[str, Any]],
    winner: dict[str, Any],
    final_result: dict[str, Any],
) -> Path:
    comparison_names = [
        "round3_control_round2_combined_140db_unnormalized",
        "round3_experiment_1_lif_coincidence_detectors",
        "round3_experiment_2b_moving_notch_plus_detectors",
        "round3_experiment_3_sincos_angle_regression",
        "round3_experiment_3c_orthogonal_combined_notches",
        first_results[0]["name"],
        first_results[1]["name"],
        final_result["name"],
    ]
    rows = [control] + [references[name] for name in comparison_names[1:5]] + first_results + [final_result]

    lines = [
        "# Round 3 Combined Experiments",
        "",
        "## Overview",
        "",
        "This report combines the strongest round-3 building blocks rather than testing them in isolation. The protocol is the same as round 3: matched-human front end, 140 dB under the current convention, unnormalized spike encoding, 0.5-5.0 m range, and the short 700/150/150 split.",
        "",
        "## Reference Table",
        "",
        "| Model | Combined Error | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Euclidean (m) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        lines.append(
            f"| {item['title']} | {item['test_metrics']['combined_error']:.4f} | "
            f"{item['test_metrics']['distance_mae_m']:.4f} | {item['test_metrics']['azimuth_mae_deg']:.4f} | "
            f"{item['test_metrics']['elevation_mae_deg']:.4f} | {item['test_metrics']['euclidean_error_m']:.4f} |"
        )

    lines.extend(["", "## Candidate Combined Models", ""])
    for item in first_results:
        component_names = _reference_labels_for_spec(item["spec"])
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Description: {item['description']}",
                f"- Decision vs round-3 control: `{item['decision']}`",
                "",
                "Metrics:",
                f"- Combined error: `{item['test_metrics']['combined_error']:.4f}`",
                f"- Distance MAE: `{item['test_metrics']['distance_mae_m']:.4f} m`",
                f"- Azimuth MAE: `{item['test_metrics']['azimuth_mae_deg']:.4f} deg`",
                f"- Elevation MAE: `{item['test_metrics']['elevation_mae_deg']:.4f} deg`",
                f"- Euclidean error: `{item['test_metrics']['euclidean_error_m']:.4f} m`",
                "",
                "Comparisons:",
            ]
        )
        for ref_name in component_names:
            ref = control if ref_name == control["name"] else references[ref_name]
            lines.append(
                f"- vs {ref['title']}: combined delta `{item['test_metrics']['combined_error'] - ref['test_metrics']['combined_error']:.4f}`, "
                f"distance delta `{item['test_metrics']['distance_mae_m'] - ref['test_metrics']['distance_mae_m']:.4f} m`, "
                f"azimuth delta `{item['test_metrics']['azimuth_mae_deg'] - ref['test_metrics']['azimuth_mae_deg']:.4f} deg`, "
                f"elevation delta `{item['test_metrics']['elevation_mae_deg'] - ref['test_metrics']['elevation_mae_deg']:.4f} deg`"
            )
        lines.extend(
            [
                "",
                f"![{item['title']} loss](round_3_combined_experiments/{item['name']}/loss.png)",
                f"![{item['title']} comparison](round_3_combined_experiments/{item['name']}/comparison.png)",
                f"![{item['title']} distance](round_3_combined_experiments/{item['name']}/test_distance_prediction.png)",
                f"![{item['title']} coordinate profile](round_3_combined_experiments/{item['name']}/coordinate_error_profile.png)",
            ]
        )
        if item["artifacts"].get("moving_notch_cue"):
            lines.append(f"![{item['title']} moving notch cue](round_3_combined_experiments/{item['name']}/moving_notch_cue.png)")
        if item["artifacts"].get("azimuth_moving_notch_cue"):
            lines.append(f"![{item['title']} azimuth moving notch cue](round_3_combined_experiments/{item['name']}/azimuth_moving_notch_cue.png)")
        if item["artifacts"].get("notch_detector_response"):
            lines.append(f"![{item['title']} elevation notch detectors](round_3_combined_experiments/{item['name']}/notch_detector_response.png)")
        if item["artifacts"].get("orthogonal_elevation_response"):
            lines.append(f"![{item['title']} orthogonal elevation detectors](round_3_combined_experiments/{item['name']}/orthogonal_elevation_response.png)")
            lines.append(f"![{item['title']} orthogonal azimuth detectors](round_3_combined_experiments/{item['name']}/orthogonal_azimuth_response.png)")
        if item["artifacts"].get("angle_norms"):
            lines.append(f"![{item['title']} angle norms](round_3_combined_experiments/{item['name']}/angle_norms.png)")
        lines.append("")

    lines.extend(
        [
            "## Winner Selection",
            "",
            f"- Winner by combined error: `{winner['title']}`",
            f"- Winner combined error: `{winner['test_metrics']['combined_error']:.4f}`",
            f"- Winner Euclidean error: `{winner['test_metrics']['euclidean_error_m']:.4f} m`",
            "",
            "## Full Combined Model",
            "",
            f"### {final_result['title']}",
            "",
            f"- Decision vs round-3 control: `{final_result['decision']}`",
            "",
            "Metrics:",
            f"- Combined error: `{final_result['test_metrics']['combined_error']:.4f}`",
            f"- Distance MAE: `{final_result['test_metrics']['distance_mae_m']:.4f} m`",
            f"- Azimuth MAE: `{final_result['test_metrics']['azimuth_mae_deg']:.4f} deg`",
            f"- Elevation MAE: `{final_result['test_metrics']['elevation_mae_deg']:.4f} deg`",
            f"- Euclidean error: `{final_result['test_metrics']['euclidean_error_m']:.4f} m`",
            "",
            "Comparisons:",
        ]
    )
    for ref_name in _reference_labels_for_spec(final_result["spec"], winner["name"]):
        ref = control if ref_name == control["name"] else (winner if ref_name == winner["name"] else references[ref_name])
        lines.append(
            f"- vs {ref['title']}: combined delta `{final_result['test_metrics']['combined_error'] - ref['test_metrics']['combined_error']:.4f}`, "
            f"distance delta `{final_result['test_metrics']['distance_mae_m'] - ref['test_metrics']['distance_mae_m']:.4f} m`, "
            f"azimuth delta `{final_result['test_metrics']['azimuth_mae_deg'] - ref['test_metrics']['azimuth_mae_deg']:.4f} deg`, "
            f"elevation delta `{final_result['test_metrics']['elevation_mae_deg'] - ref['test_metrics']['elevation_mae_deg']:.4f} deg`"
        )
    lines.extend(
        [
            "",
            f"![{final_result['title']} loss](round_3_combined_experiments/{final_result['name']}/loss.png)",
            f"![{final_result['title']} comparison](round_3_combined_experiments/{final_result['name']}/comparison.png)",
            f"![{final_result['title']} distance](round_3_combined_experiments/{final_result['name']}/test_distance_prediction.png)",
            f"![{final_result['title']} coordinate profile](round_3_combined_experiments/{final_result['name']}/coordinate_error_profile.png)",
        ]
    )
    if final_result["artifacts"].get("moving_notch_cue"):
        lines.append(f"![{final_result['title']} moving notch cue](round_3_combined_experiments/{final_result['name']}/moving_notch_cue.png)")
    if final_result["artifacts"].get("azimuth_moving_notch_cue"):
        lines.append(f"![{final_result['title']} azimuth moving notch cue](round_3_combined_experiments/{final_result['name']}/azimuth_moving_notch_cue.png)")
    if final_result["artifacts"].get("notch_detector_response"):
        lines.append(f"![{final_result['title']} elevation notch detectors](round_3_combined_experiments/{final_result['name']}/notch_detector_response.png)")
    if final_result["artifacts"].get("lif_distance_left_spikes"):
        lines.append(f"![{final_result['title']} LIF distance spikes](round_3_combined_experiments/{final_result['name']}/lif_distance_left_spikes.png)")
    if final_result["artifacts"].get("lif_itd_spikes"):
        lines.append(f"![{final_result['title']} LIF ITD spikes](round_3_combined_experiments/{final_result['name']}/lif_itd_spikes.png)")
    if final_result["artifacts"].get("angle_norms"):
        lines.append(f"![{final_result['title']} angle norms](round_3_combined_experiments/{final_result['name']}/angle_norms.png)")

    report_path = outputs_root / "round_3_combined_experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_round_3_combined_experiments(config: GlobalConfig, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=4,
        scheduler_patience=2,
    )
    effective_config = _round3_base_config(config)
    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    from stages.experiments import _baseline_reference_params

    base_params, _ = _baseline_reference_params(context)
    seed_everything(effective_config.seed)

    references = _load_round3_references(outputs.root)
    control = references["round3_control_round2_combined_140db_unnormalized"]

    stage_root = outputs.root / "round_3_combined_experiments"
    stage_root.mkdir(parents=True, exist_ok=True)

    data_cache: dict[str, dict[str, Any]] = {}
    first_results: list[dict[str, Any]] = []
    for spec in _combined_specs():
        result = _run_single_spec(outputs.root, stage_root, context, training_config, base_params, spec, references, data_cache)
        result["spec"] = spec
        first_results.append(result)

    winner = min(first_results, key=lambda item: (float(item["test_metrics"]["combined_error"]), float(item["test_metrics"]["euclidean_error_m"])))
    winner_data_variant = winner["data_variant"]
    winner_title = winner["title"]
    if winner["name"] == "round3_combined_experiment_2b_plus_3":
        winner_variant = "combined_2b_3"
    else:
        winner_variant = "combined_3c_3"
    final_spec = _full_combined_spec(winner_variant, winner_title, winner_data_variant)
    final_result = _run_single_spec(outputs.root, stage_root, context, training_config, base_params, final_spec, references, data_cache)
    final_result["spec"] = final_spec

    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error", "Euclidean"],
        {
            "Control": [
                control["test_metrics"]["distance_mae_m"],
                control["test_metrics"]["azimuth_mae_deg"],
                control["test_metrics"]["elevation_mae_deg"],
                control["test_metrics"]["combined_error"],
                control["test_metrics"]["euclidean_error_m"],
            ],
            "2B": [
                references["round3_experiment_2b_moving_notch_plus_detectors"]["test_metrics"]["distance_mae_m"],
                references["round3_experiment_2b_moving_notch_plus_detectors"]["test_metrics"]["azimuth_mae_deg"],
                references["round3_experiment_2b_moving_notch_plus_detectors"]["test_metrics"]["elevation_mae_deg"],
                references["round3_experiment_2b_moving_notch_plus_detectors"]["test_metrics"]["combined_error"],
                references["round3_experiment_2b_moving_notch_plus_detectors"]["test_metrics"]["euclidean_error_m"],
            ],
            "3": [
                references["round3_experiment_3_sincos_angle_regression"]["test_metrics"]["distance_mae_m"],
                references["round3_experiment_3_sincos_angle_regression"]["test_metrics"]["azimuth_mae_deg"],
                references["round3_experiment_3_sincos_angle_regression"]["test_metrics"]["elevation_mae_deg"],
                references["round3_experiment_3_sincos_angle_regression"]["test_metrics"]["combined_error"],
                references["round3_experiment_3_sincos_angle_regression"]["test_metrics"]["euclidean_error_m"],
            ],
            "3C": [
                references["round3_experiment_3c_orthogonal_combined_notches"]["test_metrics"]["distance_mae_m"],
                references["round3_experiment_3c_orthogonal_combined_notches"]["test_metrics"]["azimuth_mae_deg"],
                references["round3_experiment_3c_orthogonal_combined_notches"]["test_metrics"]["elevation_mae_deg"],
                references["round3_experiment_3c_orthogonal_combined_notches"]["test_metrics"]["combined_error"],
                references["round3_experiment_3c_orthogonal_combined_notches"]["test_metrics"]["euclidean_error_m"],
            ],
            first_results[0]["title"]: [
                first_results[0]["test_metrics"]["distance_mae_m"],
                first_results[0]["test_metrics"]["azimuth_mae_deg"],
                first_results[0]["test_metrics"]["elevation_mae_deg"],
                first_results[0]["test_metrics"]["combined_error"],
                first_results[0]["test_metrics"]["euclidean_error_m"],
            ],
            first_results[1]["title"]: [
                first_results[1]["test_metrics"]["distance_mae_m"],
                first_results[1]["test_metrics"]["azimuth_mae_deg"],
                first_results[1]["test_metrics"]["elevation_mae_deg"],
                first_results[1]["test_metrics"]["combined_error"],
                first_results[1]["test_metrics"]["euclidean_error_m"],
            ],
            final_result["title"]: [
                final_result["test_metrics"]["distance_mae_m"],
                final_result["test_metrics"]["azimuth_mae_deg"],
                final_result["test_metrics"]["elevation_mae_deg"],
                final_result["test_metrics"]["combined_error"],
                final_result["test_metrics"]["euclidean_error_m"],
            ],
        },
        stage_root / "overall_comparison.png",
        "Round 3 Combined Experiment Comparison",
        ylabel="Error",
    )

    report_path = _round3_combined_report(outputs.root, control, references, first_results, winner, final_result)

    def _strip_spec(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "spec"}

    summary = {
        "control": control,
        "component_references": {
            key: references[key]
            for key in [
                "round3_experiment_1_lif_coincidence_detectors",
                "round3_experiment_2b_moving_notch_plus_detectors",
                "round3_experiment_3_sincos_angle_regression",
                "round3_experiment_3c_orthogonal_combined_notches",
            ]
        },
        "combined_candidates": [_strip_spec(item) for item in first_results],
        "winner": {"name": winner["name"], "title": winner["title"]},
        "full_combined": _strip_spec(final_result),
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_3_combined_experiment_summary.json", summary)
    save_json(stage_root / "results.json", summary)
    return summary
