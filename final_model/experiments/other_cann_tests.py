from __future__ import annotations

"""Extra CANN input-structure checks for cached final-model pathway populations.

The test reuses cached raw pathway populations and compares:

* raw population centre of mass;
* two-block diagonal excitatory-only input, `[I; 0]`;
* two-block diagonal opponent input, `[I; -I]`;
* the implemented FI diagonal two-block input;
* the implemented FI reflected Gaussian two-block input.

This checks whether the reflected FI input is worth using as the final generic
SC-like readout, without rerunning the expensive acoustic/pathway simulation.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import final_distance_pipeline_with_attractor as cann
from final_model.experiments import final_model_results as final
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "other_cann_tests"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "final_model" / "reports" / "other_CANN_tests.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"

DISTANCE_MIN_M = 0.25
DISTANCE_MAX_M = 5.0
AZ_EL_LIMIT_DEG = 45.0

CACHE_SPECS = {
    "clean": ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "cache_constrained_0p25_5m_pm45_train2000_val400_test400.npz",
    "environment_noise": ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "cache_constrained_0p25_5m_pm45_train2000_val400_test400_envnoise50dB.npz",
    "environment_noise_reverb": ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "cache_constrained_0p25_5m_pm45_train2000_val400_test400_envnoise50dB_reverb.npz",
}


@dataclass(frozen=True)
class Variant:
    """One CANN/readout variant."""

    key: str
    label: str
    mode: str
    input_family: str = "identity"
    beta: float = 1.0


VARIANTS = [
    Variant("raw_com", "Raw population COM", "raw"),
    Variant("two_block_excitatory_only", "Two-block [I; 0] CANN", "two_block", beta=0.0),
    Variant("two_block_identity", "Two-block [I; -I] CANN", "two_block", beta=1.0),
    Variant("fi_diagonal", "FI diagonal 2-block CANN", "existing", beta=cann.DIAGONAL_ATTRACTOR_BETA),
    Variant("fi_reflected", "FI reflected Gaussian 2-block CANN", "existing", input_family="reflected", beta=cann.REFLECTED_ATTRACTOR_BETA),
]


def population_center_of_mass(populations: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Decode non-negative population codes by centre of mass."""
    values = np.maximum(populations, 0.0)
    totals = values.sum(axis=1)
    fallback = float(bins[len(bins) // 2])
    decoded = np.full(values.shape[0], fallback, dtype=np.float64)
    valid = totals > 1e-12
    decoded[valid] = (values[valid] * bins[None, :]).sum(axis=1) / totals[valid]
    return decoded


def two_block_identity_cann(
    populations: np.ndarray,
    bins: np.ndarray,
    beta: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a non-FI two-block diagonal-input line attractor."""
    variant = cann.AttractorVariant(
        key=f"two_block_identity_beta_{beta:g}",
        label=label,
        input_family="identity",
        input_width_bins=0.0,
        beta=beta,
    )
    return cann.run_line_attractor(populations, bins, variant, keep_history=False)[:3]


def existing_cann(populations: np.ndarray, bins: np.ndarray, variant: Variant) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one of the existing FI CANN variants."""
    existing = cann.AttractorVariant(
        key=variant.key,
        label=variant.label,
        input_family=variant.input_family,
        input_width_bins=cann.ATTRACTOR_INPUT_WIDTH_BINS if variant.input_family == "reflected" else 0.0,
        beta=variant.beta,
    )
    return cann.run_line_attractor(populations, bins, existing, keep_history=False)[:3]


def run_variant(populations: np.ndarray, bins: np.ndarray, variant: Variant) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Run a readout variant on a population matrix."""
    if variant.mode == "raw":
        return population_center_of_mass(populations, bins), None, 0.0
    if variant.mode == "two_block":
        return two_block_identity_cann(populations, bins, variant.beta, variant.label)
    if variant.mode == "existing":
        return existing_cann(populations, bins, variant)
    raise ValueError(f"Unknown variant mode: {variant.mode}")


def angular_abs_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return absolute wrapped angular error in degrees."""
    return np.abs(final.angular_error_deg(pred, true))


def scalar_metrics(pred: np.ndarray, true: np.ndarray, coordinate: str) -> dict[str, float]:
    """Compute scalar coordinate metrics."""
    if coordinate == "distance":
        error = pred - true
        abs_error = np.abs(error)
    else:
        error = final.angular_error_deg(pred, true)
        abs_error = np.abs(error)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(abs_error)),
    }


def trajectory_mae(trajectory: np.ndarray, true: np.ndarray, coordinate: str) -> np.ndarray:
    """Compute MAE over CANN time."""
    if coordinate == "distance":
        return np.mean(np.abs(trajectory - true[:, None]), axis=0)
    return np.mean(np.abs(final.angular_error_deg(trajectory, true[:, None])), axis=0)


def load_test_populations(cache_path: Path) -> dict[str, np.ndarray]:
    """Load cached test split populations and true coordinates."""
    loaded = np.load(cache_path, allow_pickle=True)
    features = np.asarray(loaded["features"], dtype=np.float64)
    true_coordinates = np.asarray(loaded["true_coordinates"], dtype=np.float64)
    split_names = np.asarray(loaded["split_names"])
    groups = loaded["feature_groups"].item()
    mask = split_names == "test"
    test_features = features[mask]
    test_true = true_coordinates[mask]

    def group(name: str) -> np.ndarray:
        left, right = groups[name]
        return test_features[:, left:right]

    return {
        "true": test_true,
        "distance": group("raw_distance_population"),
        "azimuth": group("raw_azimuth_itd_population"),
        "elevation": group("raw_elevation_population"),
    }


def plot_trajectory_mae(results: dict[str, object], condition: str, coordinate: str, path: Path) -> str:
    """Plot CANN MAE over time for one condition/coordinate."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    time_ms = np.arange(int(round(cann.ATTRACTOR_SIM_TIME_S / cann.ATTRACTOR_DT_S)) + 1) * cann.ATTRACTOR_DT_S * 1_000.0
    coord_results = results[condition]["coordinates"][coordinate]
    for variant in VARIANTS:
        if variant.key == "raw_com":
            raw_mae = coord_results[variant.key]["metrics"]["mae"]
            ax.axhline(raw_mae, color="#6b7280", linestyle="--", linewidth=1.4, label="raw COM")
            continue
        mae = np.asarray(coord_results[variant.key]["trajectory_mae"], dtype=np.float64)
        ax.plot(time_ms, mae, linewidth=2.0, label=variant.label)
    ax.axvline(cann.ATTRACTOR_READOUT_TIME_S * 1_000.0, color="#111827", linestyle=":", linewidth=1.2, label="1 ms readout")
    ax.axvline(5.0, color="#dc2626", linestyle=":", linewidth=1.2, label="5 ms")
    unit = "m" if coordinate == "distance" else "deg"
    ax.set_xlabel("CANN time (ms)")
    ax.set_ylabel(f"{coordinate} MAE ({unit})")
    ax.set_title(f"{condition}: {coordinate} readout error over time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return save_figure(fig, path)


def run_condition(condition: str, cache_path: Path) -> dict[str, object]:
    """Run all CANN variants on one cached condition."""
    data = load_test_populations(cache_path)
    bins = {
        "distance": np.linspace(DISTANCE_MIN_M, DISTANCE_MAX_M, data["distance"].shape[1], dtype=np.float64),
        "azimuth": np.linspace(-AZ_EL_LIMIT_DEG, AZ_EL_LIMIT_DEG, data["azimuth"].shape[1], dtype=np.float64),
        "elevation": np.linspace(-AZ_EL_LIMIT_DEG, AZ_EL_LIMIT_DEG, data["elevation"].shape[1], dtype=np.float64),
    }
    true = {
        "distance": data["true"][:, 0],
        "azimuth": data["true"][:, 1],
        "elevation": data["true"][:, 2],
    }
    populations = {
        "distance": data["distance"],
        "azimuth": data["azimuth"],
        "elevation": data["elevation"],
    }
    output: dict[str, object] = {"n": int(data["true"].shape[0]), "coordinates": {}}
    for coordinate in ["distance", "azimuth", "elevation"]:
        output["coordinates"][coordinate] = {}
        for variant in VARIANTS:
            pred, trajectory, seconds_per_sample = run_variant(populations[coordinate], bins[coordinate], variant)
            item = {
                "metrics": scalar_metrics(pred, true[coordinate], coordinate),
                "seconds_per_sample": seconds_per_sample,
            }
            if trajectory is not None:
                item["trajectory_mae"] = trajectory_mae(trajectory, true[coordinate], coordinate).tolist()
            output["coordinates"][coordinate][variant.key] = item
    return output


def write_report(results: dict[str, object], artifacts: dict[str, str], elapsed_s: float) -> None:
    """Write markdown report."""
    lines = [
        "# Other CANN Tests",
        "",
        "This report tests whether the FI reflected Gaussian CANN input is worthwhile compared with simpler diagonal CANN inputs. The expensive acoustic and pathway stages are not rerun. Instead, the test reuses cached raw pathway populations from the final trainable-readout caches and applies alternative SC/CANN readouts to the distance, azimuth ITD, and elevation populations.",
        "",
        "## Tested Readouts",
        "",
        "- `Raw population COM`: direct centre of mass of the cached pathway population.",
        "- `Two-block [I; 0] CANN`: two-population line attractor with diagonal excitatory input and no explicit inhibitory input (`beta=0`).",
        "- `Two-block [I; -I] CANN`: balanced E/I line attractor with diagonal opponent input and fixed `beta=1`.",
        f"- `FI diagonal 2-block CANN`: balanced E/I diagonal input with analytic beta `{cann.DIAGONAL_ATTRACTOR_BETA:.3f}`.",
        f"- `FI reflected Gaussian 2-block CANN`: implemented final CANN input with reflected Gaussian `M`, width `{cann.ATTRACTOR_INPUT_WIDTH_BINS:.0f}` bins, and analytic beta `{cann.REFLECTED_ATTRACTOR_BETA:.3f}`.",
        "",
        "All CANN variants use the same recurrent width, gain, rate cap, simulation time, and centre-of-mass readout. The reported CANN scalar metrics use the existing `1 ms` readout time. The error-over-time figures show whether another readout time would change the conclusion.",
        "",
    ]
    for condition, condition_results in results.items():
        lines.extend(
            [
                f"## {condition}",
                "",
                f"Test samples: `{condition_results['n']}`.",
                "",
            ]
        )
        for coordinate in ["distance", "azimuth", "elevation"]:
            unit = "m" if coordinate == "distance" else "deg"
            lines.extend(
                [
                    f"### {coordinate.capitalize()}",
                    "",
                    f"| Readout | MAE ({unit}) | RMSE ({unit}) | Bias ({unit}) | Max abs error ({unit}) | runtime/sample (ms) |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            coord_results = condition_results["coordinates"][coordinate]
            for variant in VARIANTS:
                metric = coord_results[variant.key]["metrics"]
                runtime_ms = coord_results[variant.key]["seconds_per_sample"] * 1_000.0
                lines.append(
                    f"| {variant.label} | "
                    f"`{metric['mae']:.4f}` | "
                    f"`{metric['rmse']:.4f}` | "
                    f"`{metric['bias']:.4f}` | "
                    f"`{metric['max_abs_error']:.4f}` | "
                    f"`{runtime_ms:.4f}` |"
                )
            figure_key = f"{condition}_{coordinate}"
            lines.extend(["", f"![{condition} {coordinate} MAE over time]({artifacts[figure_key]})", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "This is a cached-population readout test rather than a full acoustic rerun. It directly answers whether the final CANN input transformation improves the already-computed pathway populations. If reflected Gaussian and diagonal variants are close, the main value of the FI work should be described as a principled boundary-aware readout design rather than as a large empirical accuracy gain.",
            "",
            f"Runtime: `{elapsed_s:.2f} s`.",
            f"Results JSON: `{RESULTS_PATH.relative_to(ROOT)}`.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    """Run all cached CANN tests."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    results = {}
    artifacts = {}
    for condition, cache_path in CACHE_SPECS.items():
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        results[condition] = run_condition(condition, cache_path)
        for coordinate in ["distance", "azimuth", "elevation"]:
            path = FIGURE_DIR / f"{condition}_{coordinate}_mae_over_time.png"
            artifacts[f"{condition}_{coordinate}"] = Path(
                os.path.relpath(plot_trajectory_mae(results, condition, coordinate, path), REPORT_PATH.parent)
            ).as_posix()
    elapsed_s = time.perf_counter() - start
    payload = {
        "experiment": "other_cann_tests",
        "elapsed_seconds": elapsed_s,
        "cache_specs": {key: str(path.relative_to(ROOT)) for key, path in CACHE_SPECS.items()},
        "variants": [variant.__dict__ for variant in VARIANTS],
        "results": results,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    write_report(results, artifacts, elapsed_s)
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
