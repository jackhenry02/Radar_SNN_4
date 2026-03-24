from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.round2_variants import AllRound2CombinedModel
from models.round3_variants import NotchDetectorElevationEncoder
from models.round4_variants import (
    BioILDLSOEncoder,
    CombinedAcceptedRound4Encoder,
    DistanceSpikeSumEncoder,
    FullReplacementLIFTimingEncoder,
    PathwayQResonanceEncoder,
    PostPathwayICConvEncoder,
    SharedBackbonePrePathwayEncoder,
)
from stages.base import StageContext
from stages.experiments import _baseline_reference_params, _metrics_delta
from stages.improved_experiments import _is_accepted, _prepare_target_bundle
from stages.round_3_experiments import (
    Distance01TargetBundle,
    Round3ExperimentSpec,
    SinCosTargetBundle,
    _evaluate_round3_output,
    _instantiate_base_encoder,
    _prepare_distance01_target_bundle,
    _prepare_sincos_target_bundle,
    _round3_loss_and_decode,
    _round3_support_spec,
    _round3_variant_config,
    _save_round3_outputs,
)
from stages.training_improved_experiments import EnhancedTrainingConfig
from stages.expanded_space_test import _prepare_expanded_data
from utils.common import GlobalConfig, save_grouped_bar_chart, save_json, seed_everything


def _round4_base_config(base: GlobalConfig) -> GlobalConfig:
    from stages.round_3_experiments import _round3_base_config

    config = _round3_base_config(base)
    return _round3_variant_config(config, "moving_notch")


def _round4_specs() -> list[Round3ExperimentSpec]:
    shared_training = {"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35, "unit_penalty_weight": 0.1}
    return [
        Round3ExperimentSpec(
            name="round4_control_baseline_2b_plus_3",
            title="Round 4 Experiment 0: 2B + 3 Baseline",
            description="Re-run the accepted round-3 best model, 2B + 3, as the fixed baseline for round 4.",
            rationale="This keeps all round-4 comparisons anchored to the current best working model under the same short matched-human 140 dB unnormalized setup.",
            variant="baseline_2b_3",
            output_mode="sincos",
            implemented_steps=[
                "Use the moving-notch elevation cue from 2B.",
                "Use the elevation notch-detector residual from 2B.",
                "Use sine/cosine angle regression from 3.",
            ],
            analysis_focus=[
                "Establish the round-4 control under the exact same training loop as the new variants.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round4_experiment_1_full_lif_timing_replacement",
            title="Round 4 Experiment 1: Full LIF Timing Replacement",
            description="Replace the distance pathway and the ITD part of azimuth with explicit LIF coincidence banks, rather than adding them as residuals.",
            rationale="This tests whether a full spike-domain timing pathway is stronger than the current fixed overlap features once 2B + 3 is already in place.",
            variant="full_lif_timing_replacement",
            output_mode="sincos",
            implemented_steps=[
                "Keep the 2B + 3 elevation system unchanged.",
                "Discard the baseline distance latent and replace it with a trainable LIF distance coincidence latent.",
                "Discard the baseline ITD-derived azimuth component and replace it with a trainable LIF ITD latent, while retaining a fixed ILD contribution.",
            ],
            analysis_focus=[
                "Whether full replacement of the fixed timing code improves distance and azimuth.",
                "Whether removing the old timing latent hurts the fused model stability.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round4_experiment_2a_shared_prepathway_conv",
            title="Round 4 Experiment 2A: Shared Pre-Pathway Conv Backbone",
            description="Add a shared 2D convolutional preprocessing backbone before the three pathways.",
            rationale="This tests whether a light shared spectral-temporal backbone can improve the pathway inputs before they are split into distance, azimuth, and elevation.",
            variant="shared_prepathway_conv",
            output_mode="sincos",
            implemented_steps=[
                "Apply a learned shared backbone to the transmit and receive spike tensors before pathway feature extraction.",
                "Rebuild the three pathway feature sets from the transformed spikes.",
                "Feed the transformed pathway batch into the existing 2B + 3 architecture.",
            ],
            analysis_focus=[
                "Whether a shared front-end backbone improves all pathways jointly.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round4_experiment_2b_postpathway_ic_conv",
            title="Round 4 Experiment 2B: Post-Pathway IC Conv",
            description="Add a 2D convolutional integration stage after the pathway latents, inspired by an inferior-colliculus-like shared integration area.",
            rationale="This tests whether a shared pathway interaction map can refine the three branch latents after pathway-specific processing is already complete.",
            variant="postpathway_ic_conv",
            output_mode="sincos",
            implemented_steps=[
                "Take the post-pathway spike traces from the three pathway latents.",
                "Stack them into a branch-time-latent map.",
                "Apply a small 2D convolutional integration block and project the result back into all three latents.",
            ],
            analysis_focus=[
                "Whether cross-pathway latent interaction improves final decoding.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round4_experiment_3_bio_ild_lso_mntb",
            title="Round 4 Experiment 3: LSO/MNTB ILD System",
            description="Replace the simple ILD calculation with a more biologically inspired LSO/MNTB style opponent system.",
            rationale="This tests whether explicitly modeling excitatory ipsilateral drive and contralateral inhibitory drive improves azimuth, especially near the midline.",
            variant="bio_ild_lso_mntb",
            output_mode="sincos",
            implemented_steps=[
                "Keep the 2B + 3 elevation and ITD systems unchanged.",
                "Compute per-ear spike-count drives, convert contralateral drive into inhibition through an MNTB-like stage, and form left and right LSO responses.",
                "Compare the LSO outputs to form a replacement ILD representation for the azimuth pathway.",
            ],
            analysis_focus=[
                "Whether the LSO/MNTB style ILD improves azimuth without disturbing elevation.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round4_experiment_4_distance_spike_sum",
            title="Round 4 Experiment 4: Distance Spike-Sum Cue",
            description="Add a simple receive-spike summing cue to the distance pathway so that overall echo strength can contribute to range estimation.",
            rationale="This tests whether the unnormalized 140 dB setup still contains useful loudness information that the current distance pathway is not using directly.",
            variant="distance_spike_sum",
            output_mode="sincos",
            implemented_steps=[
                "Compute simple left, right, total, and contrast spike-count summaries from the receive spikes.",
                "Project them into a distance-only residual latent.",
                "Add that residual to the 2B + 3 distance pathway only.",
            ],
            analysis_focus=[
                "Whether explicit spike-sum distance information improves range estimation.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
        Round3ExperimentSpec(
            name="round4_experiment_5_per_pathway_q_resonance",
            title="Round 4 Experiment 5: Per-Pathway Q-Tunable Resonance Banks",
            description="Add separate resonance banks to distance, azimuth, and elevation, each with its own trainable Q factor and task-specific initialization.",
            rationale="This tests whether task-specific resonance is more useful than the current shared resonance branch.",
            variant="per_pathway_q_resonance",
            output_mode="sincos",
            implemented_steps=[
                "Add separate resonance banks for distance, azimuth, and elevation.",
                "Initialize azimuth with lower Q, distance with moderate Q, and elevation with higher Q.",
                "Project each bank back only into its own pathway latent.",
            ],
            analysis_focus=[
                "Whether pathway-specific resonance improves timing and spectral discrimination.",
            ],
            training_overrides=shared_training,
            data_variant="moving_notch",
        ),
    ]


def _instantiate_round4_model(data: Any, params: dict[str, Any], spec: Round3ExperimentSpec) -> nn.Module:
    base_encoder = _instantiate_base_encoder(data, params)
    baseline_encoder = NotchDetectorElevationEncoder(
        base_encoder=base_encoder,
        branch_hidden_dim=int(params["branch_hidden_dim"]),
        num_frequency_channels=int(params["num_frequency_channels"]),
    )
    if spec.variant == "baseline_2b_3":
        encoder = baseline_encoder
    elif spec.variant == "full_lif_timing_replacement":
        encoder = FullReplacementLIFTimingEncoder(
            base_encoder=baseline_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            distance_candidates=base_encoder.distance_candidates,
            itd_candidates=base_encoder.itd_candidates,
            num_delay_lines=int(params["num_delay_lines"]),
        )
    elif spec.variant == "shared_prepathway_conv":
        encoder = SharedBackbonePrePathwayEncoder(
            base_encoder=baseline_encoder,
            num_frequency_channels=int(params["num_frequency_channels"]),
            num_delay_lines=int(params["num_delay_lines"]),
            distance_candidates=base_encoder.distance_candidates,
            itd_candidates=base_encoder.itd_candidates,
        )
    elif spec.variant == "postpathway_ic_conv":
        encoder = PostPathwayICConvEncoder(
            base_encoder=baseline_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
        )
    elif spec.variant == "bio_ild_lso_mntb":
        encoder = BioILDLSOEncoder(
            base_encoder=baseline_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
            num_delay_lines=int(params["num_delay_lines"]),
        )
    elif spec.variant == "distance_spike_sum":
        encoder = DistanceSpikeSumEncoder(
            base_encoder=baseline_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
        )
    elif spec.variant == "per_pathway_q_resonance":
        encoder = PathwayQResonanceEncoder(
            base_encoder=baseline_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
        )
    elif spec.variant == "combined_all_accepted":
        encoder = CombinedAcceptedRound4Encoder(
            base_encoder=baseline_encoder,
            branch_hidden_dim=int(params["branch_hidden_dim"]),
            num_frequency_channels=int(params["num_frequency_channels"]),
            num_delay_lines=int(params["num_delay_lines"]),
            distance_candidates=base_encoder.distance_candidates,
            itd_candidates=base_encoder.itd_candidates,
        )
    else:
        raise ValueError(f"Unsupported round-4 variant '{spec.variant}'.")

    return AllRound2CombinedModel(
        encoder=encoder,
        hidden_dim=int(params["hidden_dim"]),
        output_dim=5,
        num_steps=int(params["num_steps"]),
        beta=float(params["membrane_beta"]),
        threshold=float(params["fusion_threshold"]),
        reset_mechanism=str(params["reset_mechanism"]),
    ).to(data.train_targets_raw.device)


def _run_single_round4_experiment(
    outputs_root: Path,
    stage_root: Path,
    context: StageContext,
    training_config: EnhancedTrainingConfig,
    data: Any,
    base_params: dict[str, Any],
    spec: Round3ExperimentSpec,
    baseline_bundle: Any,
    sincos_bundle: SinCosTargetBundle,
    distance01_bundle: Distance01TargetBundle,
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    result_path = stage_root / spec.name / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    seed_everything(context.config.seed + hash(spec.name) % 10_000)
    print(f"[round_4] running {spec.name} on cpu", flush=True)
    model = _instantiate_round4_model(data, base_params, spec)
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
        train_loss_history.append(sum(batch_losses) / max(1, len(batch_losses)))

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
        val_loss_history.append(float(val_loss.item()))
        val_evaluation = _evaluate_round3_output(val_raw_prediction, data.val_targets_raw, val_diagnostics, data.local_config)
        scheduler.step(float(val_loss.item()))

        val_combined = float(val_evaluation.metrics["combined_error"])
        if val_combined < best_val_combined - 1e-6:
            best_val_combined = val_combined
            best_epoch = epoch + 1
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
        test_loss, test_raw_prediction, loss_summary = _round3_loss_and_decode(
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
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    evaluation = _evaluate_round3_output(test_raw_prediction, data.test_targets_raw, test_diagnostics, data.local_config)
    if baseline_metrics is None:
        comparison = {key: 0.0 for key in ["combined_error_delta", "distance_mae_delta", "azimuth_mae_delta", "elevation_mae_delta", "spike_rate_delta"]}
        accepted = False
        decision = "CONTROL"
        baseline_reference = evaluation.metrics
    else:
        comparison = _metrics_delta(evaluation.metrics, baseline_metrics)
        accepted = _is_accepted(evaluation.metrics, baseline_metrics)
        decision = "ACCEPTED" if accepted else "REJECTED"
        baseline_reference = baseline_metrics

    artifacts = _save_round3_outputs(stage_root, spec, train_loss_history, val_loss_history, evaluation, baseline_reference, test_diagnostics, data.local_config)
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
        "decision": decision,
        "baseline_label": "Round 4 baseline is 2B + 3",
        "test_metrics": evaluation.metrics,
        "comparison": comparison,
        "training": {
            "best_epoch": best_epoch,
            "executed_epochs": len(train_loss_history),
        },
        "timings": {
            "data_prep_seconds": float(0.0),
            "training_seconds": float(training_seconds),
            "evaluation_seconds": float(evaluation_seconds),
            "total_seconds": float(total_seconds),
        },
        "loss_summary": loss_summary,
        "artifacts": artifacts,
        "test_loss": float(test_loss.item()),
    }
    save_json(result_path, result)
    return result


def _round4_report(outputs_root: Path, control: dict[str, Any], results: list[dict[str, Any]]) -> Path:
    report_path = outputs_root / "round_4_experiments_report.md"
    lines = [
        "# Round 4 Experiments",
        "",
        "Round 4 uses the accepted round-3 combined model `2B + 3` as the fixed baseline.",
        "",
        "| Experiment | Combined | Distance | Azimuth | Elevation | Euclidean | Accepted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        f"| {control['title']} | {control['test_metrics']['combined_error']:.4f} | {control['test_metrics']['distance_mae_m']:.4f} | {control['test_metrics']['azimuth_mae_deg']:.4f} | {control['test_metrics']['elevation_mae_deg']:.4f} | {control['test_metrics']['euclidean_error_m']:.4f} | Control |",
    ]
    for result in results:
        lines.append(
            f"| {result['title']} | {result['test_metrics']['combined_error']:.4f} | {result['test_metrics']['distance_mae_m']:.4f} | "
            f"{result['test_metrics']['azimuth_mae_deg']:.4f} | {result['test_metrics']['elevation_mae_deg']:.4f} | "
            f"{result['test_metrics']['euclidean_error_m']:.4f} | {'Yes' if result['accepted'] else 'No'} |"
        )
    lines.extend(["", "## Baseline", ""])
    lines.extend([f"- Combined error: `{control['test_metrics']['combined_error']:.4f}`", f"- Distance MAE: `{control['test_metrics']['distance_mae_m']:.4f} m`", f"- Azimuth MAE: `{control['test_metrics']['azimuth_mae_deg']:.4f} deg`", f"- Elevation MAE: `{control['test_metrics']['elevation_mae_deg']:.4f} deg`", f"- Euclidean error: `{control['test_metrics']['euclidean_error_m']:.4f} m`", ""])
    for result in results:
        lines.extend([f"## {result['title']}", ""])
        lines.append(f"- Decision: `{result['decision']}`")
        lines.append(f"- Change: {result['description']}")
        lines.append(f"- Combined error: `{result['test_metrics']['combined_error']:.4f}`")
        lines.append(f"- Distance MAE: `{result['test_metrics']['distance_mae_m']:.4f} m`")
        lines.append(f"- Azimuth MAE: `{result['test_metrics']['azimuth_mae_deg']:.4f} deg`")
        lines.append(f"- Elevation MAE: `{result['test_metrics']['elevation_mae_deg']:.4f} deg`")
        lines.append(f"- Euclidean error: `{result['test_metrics']['euclidean_error_m']:.4f} m`")
        lines.append(
            f"- Delta vs baseline: combined `{result['comparison']['combined_error_delta']:.4f}`, "
            f"distance `{result['comparison']['distance_mae_delta']:.4f} m`, "
            f"azimuth `{result['comparison']['azimuth_mae_delta']:.4f} deg`, "
            f"elevation `{result['comparison']['elevation_mae_delta']:.4f} deg`"
        )
        lines.append(f"- Runtime: `{result['timings']['total_seconds']:.2f} s`")
        lines.append("")
        loss_path = result["artifacts"].get("loss")
        comparison_path = result["artifacts"].get("comparison")
        distance_path = result["artifacts"].get("test_distance_prediction")
        coordinate_path = result["artifacts"].get("coordinate_error_profile")
        if loss_path:
            lines.append(f"![{result['title']} loss]({Path(loss_path).relative_to(outputs_root)})")
        if comparison_path:
            lines.append(f"![{result['title']} comparison]({Path(comparison_path).relative_to(outputs_root)})")
        if distance_path:
            lines.append(f"![{result['title']} distance]({Path(distance_path).relative_to(outputs_root)})")
        if coordinate_path:
            lines.append(f"![{result['title']} coordinate profile]({Path(coordinate_path).relative_to(outputs_root)})")
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_round_4_experiments(config: GlobalConfig, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=4,
        scheduler_patience=2,
    )
    effective_config = _round4_base_config(config)
    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    base_params, _ = _baseline_reference_params(context)
    support_spec = _round3_support_spec()
    seed_everything(effective_config.seed)

    stage_root = outputs.root / "round_4_experiments"
    stage_root.mkdir(parents=True, exist_ok=True)

    prep_start = time.perf_counter()
    data = _prepare_expanded_data(context, base_params, support_spec, chunk_size=16)
    baseline_bundle = _prepare_target_bundle(data)
    sincos_bundle = _prepare_sincos_target_bundle(data)
    distance01_bundle = _prepare_distance01_target_bundle(data, data.local_config)
    data_prep_seconds = time.perf_counter() - prep_start

    specs = _round4_specs()
    control_result: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        seed_everything(effective_config.seed + index)
        baseline_metrics = None if control_result is None else control_result["test_metrics"]
        result = _run_single_round4_experiment(
            outputs.root,
            stage_root,
            context,
            training_config,
            data,
            base_params,
            spec,
            baseline_bundle,
            sincos_bundle,
            distance01_bundle,
            baseline_metrics,
        )
        result["timings"]["data_prep_seconds"] = float(data_prep_seconds)
        if control_result is None:
            control_result = result
        else:
            results.append(result)

    assert control_result is not None
    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error", "Euclidean"],
        {
            control_result["title"]: [
                control_result["test_metrics"]["distance_mae_m"],
                control_result["test_metrics"]["azimuth_mae_deg"],
                control_result["test_metrics"]["elevation_mae_deg"],
                control_result["test_metrics"]["combined_error"],
                control_result["test_metrics"]["euclidean_error_m"],
            ],
            **{
                result["title"]: [
                    result["test_metrics"]["distance_mae_m"],
                    result["test_metrics"]["azimuth_mae_deg"],
                    result["test_metrics"]["elevation_mae_deg"],
                    result["test_metrics"]["combined_error"],
                    result["test_metrics"]["euclidean_error_m"],
                ]
                for result in results
            },
        },
        stage_root / "overall_comparison.png",
        "Round 4 Experiment Comparison",
        ylabel="Error",
    )
    report_path = _round4_report(outputs.root, control_result, results)
    summary = {
        "baseline": control_result,
        "experiments": results,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_4_experiments_summary.json", summary)
    save_json(stage_root / "results.json", summary)
    return summary
