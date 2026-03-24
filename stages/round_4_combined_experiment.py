from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from stages.base import StageContext
from stages.experiments import _baseline_reference_params
from stages.improved_experiments import _prepare_target_bundle
from stages.round_3_experiments import (
    Round3ExperimentSpec,
    _prepare_distance01_target_bundle,
    _prepare_sincos_target_bundle,
    _round3_support_spec,
)
from stages.round_4_experiments import _round4_base_config, _run_single_round4_experiment
from stages.training_improved_experiments import EnhancedTrainingConfig
from stages.expanded_space_test import _prepare_expanded_data
from utils.common import GlobalConfig, save_grouped_bar_chart, save_json, seed_everything


def _combined_spec() -> Round3ExperimentSpec:
    return Round3ExperimentSpec(
        name="round4_combined_experiment_all_but_2a",
        title="Round 4 Combined: 1 + 2B + 3 + 4 + 5",
        description="Combine all accepted round-4 architectural additions except 2A on top of the 2B + 3 baseline.",
        rationale="This tests whether the accepted round-4 improvements are complementary when combined, or whether they interfere once stacked into one model.",
        variant="combined_all_accepted",
        output_mode="sincos",
        implemented_steps=[
            "Start from the 2B + 3 baseline.",
            "Use full LIF timing replacement from Experiment 1.",
            "Use post-pathway IC convolution from Experiment 2B.",
            "Use the LSO/MNTB ILD replacement from Experiment 3.",
            "Use the distance spike-sum cue from Experiment 4.",
            "Use the per-pathway Q-tunable resonance banks from Experiment 5.",
        ],
        analysis_focus=[
            "Whether the accepted round-4 improvements are complementary or mutually interfering.",
            "Whether the combined model beats the best individual round-4 experiment.",
        ],
        training_overrides={"batch_size": 8, "learning_rate_scale": 0.9, "cartesian_mix_weight": 0.35, "unit_penalty_weight": 0.1},
        data_variant="moving_notch",
    )


def _load_round4_references(outputs_root: Path) -> dict[str, Any]:
    summary = json.loads((outputs_root / "round_4_experiments_summary.json").read_text(encoding="utf-8"))
    references = {summary["baseline"]["name"]: summary["baseline"]}
    for item in summary["experiments"]:
        references[item["name"]] = item
    return references


def _write_report(outputs_root: Path, combined: dict[str, Any], references: dict[str, Any]) -> Path:
    report_path = outputs_root / "round_4_combined_experiment_report.md"
    baseline = references["round4_control_baseline_2b_plus_3"]
    compare_keys = [
        "round4_control_baseline_2b_plus_3",
        "round4_experiment_1_full_lif_timing_replacement",
        "round4_experiment_2b_postpathway_ic_conv",
        "round4_experiment_3_bio_ild_lso_mntb",
        "round4_experiment_4_distance_spike_sum",
        "round4_experiment_5_per_pathway_q_resonance",
    ]
    lines = [
        "# Round 4 Combined Experiment",
        "",
        "This run combines all accepted round-4 additions except the rejected `2A` shared pre-pathway convolution.",
        "",
        "| Model | Combined | Distance | Azimuth | Elevation | Euclidean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in compare_keys:
        item = references[key]
        metrics = item["test_metrics"]
        lines.append(
            f"| {item['title']} | {metrics['combined_error']:.4f} | {metrics['distance_mae_m']:.4f} | "
            f"{metrics['azimuth_mae_deg']:.4f} | {metrics['elevation_mae_deg']:.4f} | {metrics['euclidean_error_m']:.4f} |"
        )
    metrics = combined["test_metrics"]
    lines.append(
        f"| {combined['title']} | {metrics['combined_error']:.4f} | {metrics['distance_mae_m']:.4f} | "
        f"{metrics['azimuth_mae_deg']:.4f} | {metrics['elevation_mae_deg']:.4f} | {metrics['euclidean_error_m']:.4f} |"
    )
    lines.extend([
        "",
        "## Result",
        "",
        f"- Decision vs round-4 baseline: `{combined['decision']}`",
        f"- Combined error: `{metrics['combined_error']:.4f}`",
        f"- Distance MAE: `{metrics['distance_mae_m']:.4f} m`",
        f"- Azimuth MAE: `{metrics['azimuth_mae_deg']:.4f} deg`",
        f"- Elevation MAE: `{metrics['elevation_mae_deg']:.4f} deg`",
        f"- Euclidean error: `{metrics['euclidean_error_m']:.4f} m`",
        "",
        "## Deltas",
        "",
        f"- vs baseline: combined `{metrics['combined_error'] - baseline['test_metrics']['combined_error']:.4f}`, distance `{metrics['distance_mae_m'] - baseline['test_metrics']['distance_mae_m']:.4f} m`, azimuth `{metrics['azimuth_mae_deg'] - baseline['test_metrics']['azimuth_mae_deg']:.4f} deg`, elevation `{metrics['elevation_mae_deg'] - baseline['test_metrics']['elevation_mae_deg']:.4f} deg`",
        f"- vs best individual combined (Experiment 3): combined `{metrics['combined_error'] - references['round4_experiment_3_bio_ild_lso_mntb']['test_metrics']['combined_error']:.4f}`",
        f"- vs best azimuth individual (Experiment 4): azimuth `{metrics['azimuth_mae_deg'] - references['round4_experiment_4_distance_spike_sum']['test_metrics']['azimuth_mae_deg']:.4f} deg`",
        f"- vs best distance individual (Experiment 2B): distance `{metrics['distance_mae_m'] - references['round4_experiment_2b_postpathway_ic_conv']['test_metrics']['distance_mae_m']:.4f} m`",
        "",
    ])
    for artifact_key in ["loss", "comparison", "test_distance_prediction", "coordinate_error_profile"]:
        artifact = combined["artifacts"].get(artifact_key)
        if artifact:
            lines.append(f"![{combined['title']} {artifact_key}]({Path(artifact).relative_to(outputs_root)})")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_round_4_combined_experiment(config: GlobalConfig, outputs: Any) -> dict[str, Any]:
    training_config = EnhancedTrainingConfig(
        dataset_mode="combined_small",
        max_epochs=10,
        early_stopping_patience=4,
        scheduler_patience=2,
    )
    effective_config = _round4_base_config(config)
    context = StageContext(config=effective_config, device=torch.device("cpu"), outputs=outputs)
    base_params, _ = _baseline_reference_params(context)
    seed_everything(effective_config.seed)
    references = _load_round4_references(outputs.root)

    stage_root = outputs.root / "round_4_combined_experiment"
    stage_root.mkdir(parents=True, exist_ok=True)

    prep_start = time.perf_counter()
    data = _prepare_expanded_data(context, base_params, _round3_support_spec(), chunk_size=16)
    baseline_bundle = _prepare_target_bundle(data)
    sincos_bundle = _prepare_sincos_target_bundle(data)
    distance01_bundle = _prepare_distance01_target_bundle(data, data.local_config)
    data_prep_seconds = time.perf_counter() - prep_start

    spec = _combined_spec()
    combined = _run_single_round4_experiment(
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
        references["round4_control_baseline_2b_plus_3"]["test_metrics"],
    )
    combined["timings"]["data_prep_seconds"] = float(data_prep_seconds)

    save_grouped_bar_chart(
        ["Distance MAE", "Azimuth MAE", "Elevation MAE", "Combined Error", "Euclidean"],
        {
            references["round4_control_baseline_2b_plus_3"]["title"]: [
                references["round4_control_baseline_2b_plus_3"]["test_metrics"]["distance_mae_m"],
                references["round4_control_baseline_2b_plus_3"]["test_metrics"]["azimuth_mae_deg"],
                references["round4_control_baseline_2b_plus_3"]["test_metrics"]["elevation_mae_deg"],
                references["round4_control_baseline_2b_plus_3"]["test_metrics"]["combined_error"],
                references["round4_control_baseline_2b_plus_3"]["test_metrics"]["euclidean_error_m"],
            ],
            references["round4_experiment_1_full_lif_timing_replacement"]["title"]: [
                references["round4_experiment_1_full_lif_timing_replacement"]["test_metrics"]["distance_mae_m"],
                references["round4_experiment_1_full_lif_timing_replacement"]["test_metrics"]["azimuth_mae_deg"],
                references["round4_experiment_1_full_lif_timing_replacement"]["test_metrics"]["elevation_mae_deg"],
                references["round4_experiment_1_full_lif_timing_replacement"]["test_metrics"]["combined_error"],
                references["round4_experiment_1_full_lif_timing_replacement"]["test_metrics"]["euclidean_error_m"],
            ],
            references["round4_experiment_2b_postpathway_ic_conv"]["title"]: [
                references["round4_experiment_2b_postpathway_ic_conv"]["test_metrics"]["distance_mae_m"],
                references["round4_experiment_2b_postpathway_ic_conv"]["test_metrics"]["azimuth_mae_deg"],
                references["round4_experiment_2b_postpathway_ic_conv"]["test_metrics"]["elevation_mae_deg"],
                references["round4_experiment_2b_postpathway_ic_conv"]["test_metrics"]["combined_error"],
                references["round4_experiment_2b_postpathway_ic_conv"]["test_metrics"]["euclidean_error_m"],
            ],
            references["round4_experiment_3_bio_ild_lso_mntb"]["title"]: [
                references["round4_experiment_3_bio_ild_lso_mntb"]["test_metrics"]["distance_mae_m"],
                references["round4_experiment_3_bio_ild_lso_mntb"]["test_metrics"]["azimuth_mae_deg"],
                references["round4_experiment_3_bio_ild_lso_mntb"]["test_metrics"]["elevation_mae_deg"],
                references["round4_experiment_3_bio_ild_lso_mntb"]["test_metrics"]["combined_error"],
                references["round4_experiment_3_bio_ild_lso_mntb"]["test_metrics"]["euclidean_error_m"],
            ],
            references["round4_experiment_4_distance_spike_sum"]["title"]: [
                references["round4_experiment_4_distance_spike_sum"]["test_metrics"]["distance_mae_m"],
                references["round4_experiment_4_distance_spike_sum"]["test_metrics"]["azimuth_mae_deg"],
                references["round4_experiment_4_distance_spike_sum"]["test_metrics"]["elevation_mae_deg"],
                references["round4_experiment_4_distance_spike_sum"]["test_metrics"]["combined_error"],
                references["round4_experiment_4_distance_spike_sum"]["test_metrics"]["euclidean_error_m"],
            ],
            references["round4_experiment_5_per_pathway_q_resonance"]["title"]: [
                references["round4_experiment_5_per_pathway_q_resonance"]["test_metrics"]["distance_mae_m"],
                references["round4_experiment_5_per_pathway_q_resonance"]["test_metrics"]["azimuth_mae_deg"],
                references["round4_experiment_5_per_pathway_q_resonance"]["test_metrics"]["elevation_mae_deg"],
                references["round4_experiment_5_per_pathway_q_resonance"]["test_metrics"]["combined_error"],
                references["round4_experiment_5_per_pathway_q_resonance"]["test_metrics"]["euclidean_error_m"],
            ],
            combined["title"]: [
                combined["test_metrics"]["distance_mae_m"],
                combined["test_metrics"]["azimuth_mae_deg"],
                combined["test_metrics"]["elevation_mae_deg"],
                combined["test_metrics"]["combined_error"],
                combined["test_metrics"]["euclidean_error_m"],
            ],
        },
        stage_root / "overall_comparison.png",
        "Round 4 Combined Comparison",
        ylabel="Error",
    )

    report_path = _write_report(outputs.root, combined, references)
    summary = {
        "baseline": references["round4_control_baseline_2b_plus_3"],
        "components": {
            key: references[key]
            for key in [
                "round4_experiment_1_full_lif_timing_replacement",
                "round4_experiment_2b_postpathway_ic_conv",
                "round4_experiment_3_bio_ild_lso_mntb",
                "round4_experiment_4_distance_spike_sum",
                "round4_experiment_5_per_pathway_q_resonance",
            ]
        },
        "combined": combined,
        "report_path": str(report_path),
    }
    save_json(outputs.root / "round_4_combined_experiment_summary.json", summary)
    save_json(stage_root / "result.json", summary)
    return summary
