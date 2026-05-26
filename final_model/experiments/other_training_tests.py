from __future__ import annotations

"""Small comparison tests for alternative final-readout training targets.

This script reuses the cached pathway feature matrix produced by
trainable_final_readout.py and trains the same small snnTorch readout with a
different output encoding. The first test replaces sine/cosine angle targets
with directly normalised azimuth and elevation scalars.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_model.experiments import final_model_results as final
from final_model.experiments import trainable_final_readout as base


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "other_training_tests"
REPORT_PATH = ROOT / "final_model" / "reports" / "other_training_tests.md"
DEFAULT_CACHE = (
    ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "cache_constrained_0p25_5m_pm45_train2000_val400_test400_envnoise50dB_reverb.npz"
)
DEFAULT_REFERENCE_RESULTS = (
    ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "results_train2000_val400_test400_envnoise50dB_reverb.json"
)


def encode_scalar_targets(coordinates: np.ndarray) -> np.ndarray:
    """Encode `[distance_m, azimuth_deg, elevation_deg]` as three scalars."""
    distance = (coordinates[:, 0] - base.DISTANCE_MIN_M) / (base.DISTANCE_MAX_M - base.DISTANCE_MIN_M)
    azimuth = (coordinates[:, 1] + base.AZIMUTH_LIMIT_DEG) / (2.0 * base.AZIMUTH_LIMIT_DEG)
    elevation = (coordinates[:, 2] + base.ELEVATION_LIMIT_DEG) / (2.0 * base.ELEVATION_LIMIT_DEG)
    return np.stack([distance, azimuth, elevation], axis=1).astype(np.float32)


def decode_scalar_outputs(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode three scalar outputs back to metres/degrees."""
    clipped = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0)
    distance = clipped[:, 0] * (base.DISTANCE_MAX_M - base.DISTANCE_MIN_M) + base.DISTANCE_MIN_M
    azimuth = clipped[:, 1] * (2.0 * base.AZIMUTH_LIMIT_DEG) - base.AZIMUTH_LIMIT_DEG
    elevation = clipped[:, 2] * (2.0 * base.ELEVATION_LIMIT_DEG) - base.ELEVATION_LIMIT_DEG
    return distance, azimuth, elevation


class ScalarUncertaintyLoss(nn.Module):
    """Uncertainty-weighted loss for `[distance_norm, az_norm, el_norm]`."""

    def __init__(self) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = torch.mean((prediction - target) ** 2, dim=0)
        precision = torch.exp(-self.log_vars)
        return torch.sum(0.5 * precision * losses + 0.5 * self.log_vars)


def make_loader(features: np.ndarray, targets: np.ndarray, baseline: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a dataloader for cached features."""
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(baseline.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def model_output(model: base.SmallSNNReadout, features: torch.Tensor, baseline: torch.Tensor, mode: str, residual_scale: float) -> torch.Tensor:
    """Return direct or residual scalar readout output."""
    raw = model(features)
    if mode == "direct":
        return raw
    if mode == "residual":
        return baseline + residual_scale * raw
    raise ValueError(f"Unknown mode: {mode}")


def train_readout(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_baseline: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    val_baseline: np.ndarray,
    *,
    mode: str,
    hidden: int,
    num_steps: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    residual_scale: float,
    seed: int,
) -> tuple[base.SmallSNNReadout, ScalarUncertaintyLoss, dict[str, object]]:
    """Train one scalar-angle SNN readout."""
    torch.manual_seed(seed + (0 if mode == "residual" else 10_000))
    model = base.SmallSNNReadout(train_features.shape[1], hidden, 3, num_steps)
    if mode == "residual":
        nn.init.zeros_(model.fc_out.weight)
        nn.init.zeros_(model.fc_out.bias)

    criterion = ScalarUncertaintyLoss()
    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = make_loader(train_features, train_targets, train_baseline, batch_size, shuffle=True)
    val_loader = make_loader(val_features, val_targets, val_baseline, batch_size, shuffle=False)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        criterion.train()
        train_losses = []
        for features, targets, baseline in train_loader:
            prediction = model_output(model, features, baseline, mode, residual_scale)
            loss = criterion(prediction, targets)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 2.0)
            optimiser.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        criterion.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets, baseline in val_loader:
                prediction = model_output(model, features, baseline, mode, residual_scale)
                val_losses.append(float(criterion(prediction, targets).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
                "criterion": {name: value.detach().clone() for name, value in criterion.state_dict().items()},
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        criterion.load_state_dict(best_state["criterion"])

    return model, criterion, {
        **history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "training_seconds": time.perf_counter() - start,
        "learned_log_vars": criterion.log_vars.detach().cpu().numpy().tolist(),
    }


def train_sinecos_mse_readout(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_baseline: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    val_baseline: np.ndarray,
    *,
    mode: str,
    hidden: int,
    num_steps: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    residual_scale: float,
    seed: int,
) -> tuple[base.SmallSNNReadout, dict[str, object]]:
    """Train the standard sine/cosine readout with plain evenly weighted MSE."""
    torch.manual_seed(seed + (20_000 if mode == "residual" else 30_000))
    model = base.SmallSNNReadout(train_features.shape[1], hidden, 5, num_steps)
    if mode == "residual":
        nn.init.zeros_(model.fc_out.weight)
        nn.init.zeros_(model.fc_out.bias)

    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = make_loader(train_features, train_targets, train_baseline, batch_size, shuffle=True)
    val_loader = make_loader(val_features, val_targets, val_baseline, batch_size, shuffle=False)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for features, targets, baseline in train_loader:
            prediction = model_output(model, features, baseline, mode, residual_scale)
            loss = criterion(prediction, targets)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets, baseline in val_loader:
                prediction = model_output(model, features, baseline, mode, residual_scale)
                val_losses.append(float(criterion(prediction, targets).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        **history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "training_seconds": time.perf_counter() - start,
    }


def predict(model: base.SmallSNNReadout, features: np.ndarray, baseline: np.ndarray, mode: str, residual_scale: float, batch_size: int) -> np.ndarray:
    """Predict scalar-encoded outputs."""
    model.eval()
    loader = make_loader(features, np.zeros((features.shape[0], 3), dtype=np.float32), baseline, batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for batch_features, _, batch_baseline in loader:
            outputs.append(model_output(model, batch_features, batch_baseline, mode, residual_scale).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def coordinate_metrics(true_coordinates: np.ndarray, scalar_prediction: np.ndarray) -> dict[str, float]:
    """Compute localisation metrics for scalar-encoded predictions."""
    pred_distance, pred_azimuth, pred_elevation = decode_scalar_outputs(scalar_prediction)
    rows = []
    for true, distance, azimuth, elevation in zip(true_coordinates, pred_distance, pred_azimuth, pred_elevation):
        rows.append(
            {
                "true_distance_m": float(true[0]),
                "true_azimuth_deg": float(true[1]),
                "true_elevation_deg": float(true[2]),
                "pred_distance_m": float(distance),
                "pred_azimuth_deg": float(azimuth),
                "pred_elevation_deg": float(elevation),
            }
        )
    return final.localisation_metrics(rows, base.DISTANCE_MAX_M, base.AZIMUTH_LIMIT_DEG, base.ELEVATION_LIMIT_DEG)


def sinecos_coordinate_metrics(true_coordinates: np.ndarray, encoded_prediction: np.ndarray) -> dict[str, float]:
    """Compute localisation metrics for standard sine/cosine encoded predictions."""
    return base.coordinate_metrics(true_coordinates, encoded_prediction)


def load_reference_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Load existing sine/cosine readout metrics if available."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("test_metrics", {})


def write_report(results: dict[str, object], reference: dict[str, dict[str, float]]) -> None:
    """Write the other-training-tests markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scalar = results["test_metrics"]
    sinecos_mse = results["sinecos_mse_test_metrics"]
    lines = [
        "# Other Training Tests",
        "",
        "This report records small alternative training tests for the final SNN fusion readout. These tests reuse the cached pathway features and therefore do not regenerate the acoustic simulation or pathway outputs.",
        "",
        "## Normalised Angle Target Test",
        "",
        "The main trainable readout predicts `[distance_norm, az_sin, az_cos, el_sin, el_cos]`. This test keeps the same cached feature vector and the same small snnTorch readout architecture, but changes the target to `[distance_norm, az_norm, el_norm]`, where azimuth and elevation are linearly mapped from `[-45, 45] deg` to `[0, 1]`.",
        "",
        "This is a fair quick test inside the constrained space because there is no angular wrap-around within `+/-45 deg`. It would be less appropriate for a full `+/-180 deg` or circular-angle task.",
        "",
        "## Reproducible Setup",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| cache | `{results['cache_path']}` |",
        f"| train / val / test | `{results['splits']['train']} / {results['splits']['val']} / {results['splits']['test']}` |",
        f"| hidden neurons | `{results['hyperparameters']['hidden']}` |",
        f"| timesteps | `{results['hyperparameters']['num_steps']}` |",
        f"| epochs | `{results['hyperparameters']['epochs']}` |",
        f"| batch size | `{results['hyperparameters']['batch_size']}` |",
        f"| learning rate | `{results['hyperparameters']['learning_rate']}` |",
        f"| weight decay | `{results['hyperparameters']['weight_decay']}` |",
        f"| residual scale | `{results['hyperparameters']['residual_scale']}` |",
        "",
        "## Results",
        "",
        "| Encoding | Readout | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Euclidean MAE (m) | Combined error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for readout in ["direct", "residual"]:
        metrics = scalar[readout]
        lines.append(
            f"| normalised scalar angles | {readout} | "
            f"{metrics['distance_mae_m']:.4f} | "
            f"{metrics['azimuth_mae_deg']:.3f} | "
            f"{metrics['elevation_mae_deg']:.3f} | "
            f"{metrics['euclidean_mae_m']:.4f} | "
            f"{metrics['combined_normalised_error']:.4f} |"
        )
    for readout in ["direct", "residual"]:
        if readout in reference:
            metrics = reference[readout]
            lines.append(
                f"| sine/cosine angles | {readout} | "
                f"{metrics['distance_mae_m']:.4f} | "
                f"{metrics['azimuth_mae_deg']:.3f} | "
                f"{metrics['elevation_mae_deg']:.3f} | "
                f"{metrics['euclidean_mae_m']:.4f} | "
                f"{metrics['combined_normalised_error']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if reference:
        scalar_residual = scalar["residual"]["combined_normalised_error"]
        reference_residual = reference["residual"]["combined_normalised_error"]
        if scalar_residual < reference_residual:
            lines.append(
                f"The normalised scalar-angle residual readout is better on this quick test (`{scalar_residual:.4f}` vs `{reference_residual:.4f}` combined error). This is worth a follow-up run or repeat over multiple seeds before changing the main report."
            )
        else:
            lines.append(
                f"The normalised scalar-angle residual readout is not better on this quick test (`{scalar_residual:.4f}` vs `{reference_residual:.4f}` combined error for the sine/cosine residual). The existing sine/cosine target remains the safer default."
            )
    else:
        lines.append("No reference sine/cosine result file was found, so this report only records the scalar-angle run.")
    lines.extend(
        [
            "",
            "The scalar-angle target is simpler and may be adequate in the constrained `+/-45 deg` range because the target is not circular. The sine/cosine target remains more general because it handles angular wrap-around and avoids imposing a discontinuity at the boundary.",
            "",
            "## Regular MSE Loss Test",
            "",
            "The main trainable readout uses the sine/cosine angle target with learned uncertainty weighting across distance, azimuth, and elevation. This second test keeps the same `[distance_norm, az_sin, az_cos, el_sin, el_cos]` target but replaces the uncertainty-weighted objective with plain evenly weighted MSE over the five output components.",
            "",
            "| Encoding | Loss | Readout | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Euclidean MAE (m) | Combined error |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for readout in ["direct", "residual"]:
        metrics = sinecos_mse[readout]
        lines.append(
            f"| sine/cosine angles | regular MSE | {readout} | "
            f"{metrics['distance_mae_m']:.4f} | "
            f"{metrics['azimuth_mae_deg']:.3f} | "
            f"{metrics['elevation_mae_deg']:.3f} | "
            f"{metrics['euclidean_mae_m']:.4f} | "
            f"{metrics['combined_normalised_error']:.4f} |"
        )
    for readout in ["direct", "residual"]:
        if readout in reference:
            metrics = reference[readout]
            lines.append(
                f"| sine/cosine angles | uncertainty weighted | {readout} | "
                f"{metrics['distance_mae_m']:.4f} | "
                f"{metrics['azimuth_mae_deg']:.3f} | "
                f"{metrics['elevation_mae_deg']:.3f} | "
                f"{metrics['euclidean_mae_m']:.4f} | "
                f"{metrics['combined_normalised_error']:.4f} |"
            )
    lines.extend(
        [
            "",
        ]
    )
    if reference:
        mse_residual = sinecos_mse["residual"]["combined_normalised_error"]
        reference_residual = reference["residual"]["combined_normalised_error"]
        if mse_residual < reference_residual:
            lines.append(
                f"The regular-MSE residual readout is better on this quick test (`{mse_residual:.4f}` vs `{reference_residual:.4f}` combined error for the uncertainty-weighted residual). This would justify repeating the comparison over multiple seeds."
            )
        else:
            lines.append(
                f"The regular-MSE residual readout is not better on this quick test (`{mse_residual:.4f}` vs `{reference_residual:.4f}` combined error for the uncertainty-weighted residual). The uncertainty-weighted loss remains the safer default."
            )
    lines.extend(
        [
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reference-results", type=Path, default=DEFAULT_REFERENCE_RESULTS)
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--hidden", type=int, default=base.HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=base.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=base.WEIGHT_DECAY)
    parser.add_argument("--num-steps", type=int, default=base.NUM_STEPS)
    parser.add_argument("--residual-scale", type=float, default=base.RESIDUAL_SCALE)
    parser.add_argument("--training-seed", type=int, default=base.TRAINING_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.cache, allow_pickle=True)
    features = np.asarray(loaded["features"], dtype=np.float32)
    true_coordinates = np.asarray(loaded["true_coordinates"], dtype=np.float32)
    baseline_coordinates = np.asarray(loaded["baseline_coordinates"], dtype=np.float32)
    split_names = np.asarray(loaded["split_names"])

    scalar_targets = encode_scalar_targets(true_coordinates)
    scalar_baseline = encode_scalar_targets(baseline_coordinates)
    sinecos_targets = np.asarray(loaded["targets_encoded"], dtype=np.float32)
    sinecos_baseline = np.asarray(loaded["baseline_encoded"], dtype=np.float32)
    train_mask = split_names == "train"
    val_mask = split_names == "val"
    test_mask = split_names == "test"

    train_features = features[train_mask]
    val_features = features[val_mask]
    test_features = features[test_mask]
    mean, std = base.fit_normaliser(train_features)
    train_features = base.apply_normaliser(train_features, mean, std)
    val_features = base.apply_normaliser(val_features, mean, std)
    test_features = base.apply_normaliser(test_features, mean, std)

    train_targets = scalar_targets[train_mask]
    val_targets = scalar_targets[val_mask]
    test_true = true_coordinates[test_mask]
    train_baseline = scalar_baseline[train_mask]
    val_baseline = scalar_baseline[val_mask]
    test_baseline = scalar_baseline[test_mask]
    train_sinecos_targets = sinecos_targets[train_mask]
    val_sinecos_targets = sinecos_targets[val_mask]
    train_sinecos_baseline = sinecos_baseline[train_mask]
    val_sinecos_baseline = sinecos_baseline[val_mask]
    test_sinecos_baseline = sinecos_baseline[test_mask]

    histories = {}
    metrics = {}
    for mode in ["direct", "residual"]:
        model, criterion, history = train_readout(
            train_features,
            train_targets,
            train_baseline,
            val_features,
            val_targets,
            val_baseline,
            mode=mode,
            hidden=args.hidden,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            residual_scale=args.residual_scale,
            seed=args.training_seed,
        )
        prediction = predict(model, test_features, test_baseline, mode, args.residual_scale, args.batch_size)
        metrics[mode] = coordinate_metrics(test_true, prediction)
        histories[mode] = history

    sinecos_mse_histories = {}
    sinecos_mse_metrics = {}
    for mode in ["direct", "residual"]:
        model, history = train_sinecos_mse_readout(
            train_features,
            train_sinecos_targets,
            train_sinecos_baseline,
            val_features,
            val_sinecos_targets,
            val_sinecos_baseline,
            mode=mode,
            hidden=args.hidden,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            residual_scale=args.residual_scale,
            seed=args.training_seed,
        )
        prediction = predict(model, test_features, test_sinecos_baseline, mode, args.residual_scale, args.batch_size)
        sinecos_mse_metrics[mode] = sinecos_coordinate_metrics(test_true, prediction)
        sinecos_mse_histories[mode] = history

    results = {
        "experiment": "normalised_scalar_angle_targets",
        "elapsed_seconds": time.perf_counter() - start,
        "cache_path": str(args.cache.relative_to(ROOT) if args.cache.is_relative_to(ROOT) else args.cache),
        "splits": {
            "train": int(np.sum(train_mask)),
            "val": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask)),
        },
        "hyperparameters": {
            "hidden": args.hidden,
            "num_steps": args.num_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "residual_scale": args.residual_scale,
            "training_seed": args.training_seed,
        },
        "histories": histories,
        "test_metrics": metrics,
        "sinecos_mse_histories": sinecos_mse_histories,
        "sinecos_mse_test_metrics": sinecos_mse_metrics,
    }
    results_path = OUTPUT_DIR / "normalised_angle_targets_envnoise50dB_reverb.json"
    results_path.write_text(json.dumps(results, indent=2))
    reference = load_reference_metrics(args.reference_results)
    write_report(results, reference)
    print(f"Wrote {results_path}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
