from __future__ import annotations

"""Test parameter-free attention on delayed-echo localisation readouts.

This post-submission experiment reuses the cached 50 dB environmental-noise
plus delayed-echo datasets. It intentionally leaves the report-backed model
code unchanged and trains only:

1. the structured residual pathway-feature SNN with PfA; and
2. the direct time-binned cochlear-spike-raster SNN with PfA.

The PfA block is a plain-PyTorch vector adaptation of the supplied
``atten_pfa`` implementation. It is applied to the hidden input currents after
each linear layer and before the corresponding LIF neuron.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STRUCTURED_CACHE = (
    ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "cache_constrained_0p25_5m_pm45_train2000_val400_test400_envnoise50dB_reverb.npz"
)
DEFAULT_CSR_CACHE = (
    ROOT
    / "final_model"
    / "outputs"
    / "trainable_input_baselines"
    / "cache_train2000_val400_test400_envnoise50dB_reverb.npz"
)
DEFAULT_STRUCTURED_REFERENCE = (
    ROOT
    / "final_model"
    / "outputs"
    / "trainable_readout"
    / "results_train2000_val400_test400_envnoise50dB_reverb.json"
)
DEFAULT_CSR_REFERENCE = (
    ROOT
    / "final_model"
    / "outputs"
    / "trainable_input_baselines"
    / "results_train2000_val400_test400_envnoise50dB_reverb.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "PfA_post_submission_tests" / "outputs"
DISTANCE_MIN_M = 0.25
DISTANCE_MAX_M = 5.0
AZIMUTH_LIMIT_DEG = 45.0
ELEVATION_LIMIT_DEG = 45.0
HIDDEN_DIM = 96
NUM_STEPS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAINING_SEED = 8_202
RESIDUAL_SCALE = 0.15


class UncertaintyWeightedLoss(nn.Module):
    """Match the three-task learned uncertainty weighting used in the report."""

    def __init__(self) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute distance, azimuth, and elevation task losses."""
        losses = torch.stack(
            [
                torch.mean((prediction[:, 0] - target[:, 0]) ** 2),
                torch.mean((prediction[:, 1:3] - target[:, 1:3]) ** 2),
                torch.mean((prediction[:, 3:5] - target[:, 3:5]) ** 2),
            ]
        )
        precision = torch.exp(-self.log_vars)
        return torch.sum(0.5 * precision * losses + 0.5 * self.log_vars)


class SmallSNNReadout(nn.Module):
    """Local copy of the report-backed cached-feature SNN readout."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_steps: int) -> None:
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()
        beta_in = torch.full((hidden_dim,), 0.85)
        beta_hidden = torch.full((hidden_dim,), 0.88)
        beta_out = torch.full((output_dim,), 0.92)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=0.9, learn_beta=True, spike_grad=spike_grad)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=0.9, learn_beta=True, spike_grad=spike_grad)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.li_out = snn.Leaky(
            beta=beta_out,
            threshold=1.0,
            learn_beta=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )


class VectorPfA(nn.Module):
    """Parameter-free attention over a vector of hidden currents."""

    def __init__(self, e_lambda: float = 1e-4, theta: float = 1.0) -> None:
        super().__init__()
        self.e_lambda = e_lambda
        self.theta = theta

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        """Reweight hidden currents relative to their within-sample mean."""
        mean = currents.mean(dim=-1, keepdim=True)
        centered = currents - mean
        centered_square = centered.square()
        weight = self.theta * centered / (centered_square + 2.0 * self.e_lambda + 2.0 * centered_square)
        bias = (self.theta - weight * (currents + mean)) / 2.0
        gate = torch.sigmoid(currents * weight + bias)
        return currents * gate


class PfASNNReadout(SmallSNNReadout):
    """Existing two-layer SNN readout with PfA before each hidden LIF layer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_steps: int) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, num_steps)
        self.pfa_in = VectorPfA()
        self.pfa_hidden = VectorPfA()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the PfA-weighted readout for a fixed number of timesteps."""
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_out = self.li_out.init_leaky()
        outputs = []
        for _ in range(self.num_steps):
            current_1 = self.pfa_in(self.fc_in(features))
            spikes_1, mem_1 = self.lif_in(current_1, mem_1)
            current_2 = self.pfa_hidden(self.fc_hidden(spikes_1))
            spikes_2, mem_2 = self.lif_hidden(current_2, mem_2)
            current_out = self.fc_out(spikes_2)
            _, mem_out = self.li_out(current_out, mem_out)
            outputs.append(mem_out)
        return torch.stack(outputs, dim=0).mean(dim=0)


def parse_args() -> argparse.Namespace:
    """Parse experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-cache", type=Path, default=DEFAULT_STRUCTURED_CACHE)
    parser.add_argument("--csr-cache", type=Path, default=DEFAULT_CSR_CACHE)
    parser.add_argument("--structured-reference", type=Path, default=DEFAULT_STRUCTURED_REFERENCE)
    parser.add_argument("--csr-reference", type=Path, default=DEFAULT_CSR_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--max-train", type=int, default=0, help="Use the first N cached training samples; 0 uses all.")
    parser.add_argument("--max-val", type=int, default=0, help="Use the first N cached validation samples; 0 uses all.")
    parser.add_argument("--max-test", type=int, default=0, help="Use the first N cached test samples; 0 uses all.")
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, object]:
    """Load one NumPy cache."""
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}")
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key].item() if loaded[key].shape == () else loaded[key] for key in loaded.files}


def limited_mask(split_names: np.ndarray, split: str, limit: int) -> np.ndarray:
    """Return a split mask limited to the first N matching samples."""
    indices = np.flatnonzero(np.asarray(split_names) == split)
    if limit > 0:
        indices = indices[:limit]
    mask = np.zeros(len(split_names), dtype=bool)
    mask[indices] = True
    return mask


def split(
    cache: dict[str, object],
    feature_key: str,
    split_name: str,
    limit: int,
    *,
    baseline_key: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one cache split."""
    mask = limited_mask(np.asarray(cache["split_names"]), split_name, limit)
    features = np.asarray(cache[feature_key], dtype=np.float32)[mask]
    targets = np.asarray(cache["targets_encoded"], dtype=np.float32)[mask]
    true_coordinates = np.asarray(cache["true_coordinates"], dtype=np.float32)[mask]
    if baseline_key is None:
        baseline = np.zeros_like(targets)
    else:
        baseline = np.asarray(cache[baseline_key], dtype=np.float32)[mask]
    return features, targets, baseline, true_coordinates


def loader(
    features: np.ndarray,
    targets: np.ndarray,
    baseline: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a readout dataloader."""
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(baseline),
    )
    return DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=shuffle)


def model_output(model: PfASNNReadout, features: torch.Tensor, baseline: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a direct or residual encoded prediction."""
    raw = model(features)
    if mode == "residual":
        return baseline + RESIDUAL_SCALE * raw
    if mode == "direct":
        return raw
    raise ValueError(f"Unknown readout mode: {mode}")


def fit_normaliser(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the feature-wise standardisation used by the original readouts."""
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def apply_normaliser(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply feature-wise standardisation."""
    return ((features - mean) / std).astype(np.float32)


def decode_outputs(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode normalised distance and angle sine/cosine pairs."""
    distance = np.clip(encoded[:, 0], 0.0, 1.0) * (DISTANCE_MAX_M - DISTANCE_MIN_M) + DISTANCE_MIN_M
    azimuth = np.rad2deg(np.arctan2(encoded[:, 1], encoded[:, 2]))
    elevation = np.rad2deg(np.arctan2(encoded[:, 3], encoded[:, 4]))
    return distance, azimuth, elevation


def angular_error_deg(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return wrapped angular error in degrees."""
    return (predicted - true + 180.0) % 360.0 - 180.0


def spherical_to_cartesian(distance_m: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Convert spherical localisation coordinates to Cartesian coordinates."""
    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.deg2rad(elevation_deg)
    x = distance_m * np.cos(el_rad) * np.cos(az_rad)
    y = distance_m * np.cos(el_rad) * np.sin(az_rad)
    z = distance_m * np.sin(el_rad)
    return np.stack([x, y, z], axis=-1)


def coordinate_metrics(true_coordinates: np.ndarray, encoded_prediction: np.ndarray) -> dict[str, float]:
    """Compute the same localisation metrics as the report-backed scripts."""
    pred_distance, pred_azimuth, pred_elevation = decode_outputs(encoded_prediction)
    true_distance = true_coordinates[:, 0].astype(np.float64)
    true_azimuth = true_coordinates[:, 1].astype(np.float64)
    true_elevation = true_coordinates[:, 2].astype(np.float64)
    distance_error = pred_distance - true_distance
    azimuth_error = angular_error_deg(pred_azimuth, true_azimuth)
    elevation_error = angular_error_deg(pred_elevation, true_elevation)
    euclidean = np.linalg.norm(
        spherical_to_cartesian(pred_distance, pred_azimuth, pred_elevation)
        - spherical_to_cartesian(true_distance, true_azimuth, true_elevation),
        axis=1,
    )
    combined = (
        np.abs(distance_error) / DISTANCE_MAX_M
        + np.abs(azimuth_error) / AZIMUTH_LIMIT_DEG
        + np.abs(elevation_error) / ELEVATION_LIMIT_DEG
    ) / 3.0
    return {
        "distance_mae_m": float(np.mean(np.abs(distance_error))),
        "distance_rmse_m": float(np.sqrt(np.mean(distance_error**2))),
        "distance_bias_m": float(np.mean(distance_error)),
        "azimuth_mae_deg": float(np.mean(np.abs(azimuth_error))),
        "azimuth_rmse_deg": float(np.sqrt(np.mean(azimuth_error**2))),
        "azimuth_bias_deg": float(np.mean(azimuth_error)),
        "elevation_mae_deg": float(np.mean(np.abs(elevation_error))),
        "elevation_rmse_deg": float(np.sqrt(np.mean(elevation_error**2))),
        "elevation_bias_deg": float(np.mean(elevation_error)),
        "euclidean_mae_m": float(np.mean(euclidean)),
        "euclidean_rmse_m": float(np.sqrt(np.mean(euclidean**2))),
        "euclidean_max_m": float(np.max(euclidean)),
        "combined_normalised_error": float(np.mean(combined)),
    }


def train_variant(
    *,
    cache: dict[str, object],
    feature_key: str,
    baseline_key: str | None,
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Train and evaluate one PfA readout."""
    train_x, train_y, train_base, _ = split(cache, feature_key, "train", args.max_train, baseline_key=baseline_key)
    val_x, val_y, val_base, _ = split(cache, feature_key, "val", args.max_val, baseline_key=baseline_key)
    test_x, _, test_base, test_true = split(cache, feature_key, "test", args.max_test, baseline_key=baseline_key)

    mean, std = fit_normaliser(train_x)
    train_x = apply_normaliser(train_x, mean, std)
    val_x = apply_normaliser(val_x, mean, std)
    test_x = apply_normaliser(test_x, mean, std)

    torch.manual_seed(seed)
    model = PfASNNReadout(train_x.shape[1], args.hidden, 5, args.num_steps)
    if mode == "residual":
        nn.init.zeros_(model.fc_out.weight)
        nn.init.zeros_(model.fc_out.bias)
    criterion = UncertaintyWeightedLoss()
    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = loader(train_x, train_y, train_base, batch_size=args.batch_size, shuffle=True)
    val_loader = loader(val_x, val_y, val_base, batch_size=args.batch_size, shuffle=False)
    history: dict[str, object] = {"train_loss": [], "val_loss": []}
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        criterion.train()
        train_losses = []
        for features, targets, baseline in train_loader:
            prediction = model_output(model, features, baseline, mode)
            loss = criterion(prediction, targets)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 2.0)
            optimiser.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        criterion.eval()
        with torch.no_grad():
            val_losses = [
                float(criterion(model_output(model, features, baseline, mode), targets).item())
                for features, targets, baseline in val_loader
            ]
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
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"  epoch {epoch + 1:>3}/{args.epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        criterion.load_state_dict(best_state["criterion"])

    test_loader = loader(
        test_x,
        np.zeros((test_x.shape[0], 5), dtype=np.float32),
        test_base,
        batch_size=args.batch_size,
        shuffle=False,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for features, _, baseline in test_loader:
            predictions.append(model_output(model, features, baseline, mode).cpu().numpy())
    encoded = np.concatenate(predictions, axis=0)
    metrics = coordinate_metrics(test_true, encoded)
    history.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "training_seconds": time.perf_counter() - start,
            "learned_log_vars": criterion.log_vars.detach().cpu().numpy().tolist(),
            "input_dim": int(train_x.shape[1]),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
                + sum(parameter.numel() for parameter in criterion.parameters())
            ),
            "splits": {
                "train": int(train_x.shape[0]),
                "val": int(val_x.shape[0]),
                "test": int(test_x.shape[0]),
            },
        }
    )
    return {"history": history, "metrics": metrics}, encoded, test_true


def reference_metrics(path: Path, key: str) -> dict[str, float] | None:
    """Load an existing unmodified-model metric row."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test_metrics", {}).get(key)
    return dict(metrics) if isinstance(metrics, dict) else None


def metric_row(name: str, metrics: dict[str, float]) -> str:
    """Return one concise Markdown metric row."""
    return (
        f"| {name} | {metrics['distance_mae_m']:.4f} m | {metrics['azimuth_mae_deg']:.3f} deg | "
        f"{metrics['elevation_mae_deg']:.3f} deg | {metrics['euclidean_mae_m']:.4f} m | "
        f"{metrics['combined_normalised_error']:.4f} |"
    )


def write_report(path: Path, results: dict[str, object]) -> None:
    """Write a short Markdown summary."""
    metrics = results["metrics"]
    references = results["references"]
    lines = [
        "# PfA Delayed-Echo Readout Test",
        "",
        "This post-submission experiment reuses the cached 50 dB environmental-noise plus delayed-echo datasets. It does not regenerate the acoustic simulation or modify the report-backed pipeline.",
        "",
        "PfA is applied after each hidden linear layer and before the corresponding LIF neuron. The structured model uses residual fusion; the cochlear-spike-raster (CSR) model uses the direct time-binned raster input without the fixed Gaussian projection. No raw-waveform variant is included.",
        "",
        "## Results",
        "",
        "| Model | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if references["structured_residual"] is not None:
        lines.append(metric_row("structured residual reference", references["structured_residual"]))
    lines.append(metric_row("structured residual + PfA", metrics["structured_residual_pfa"]))
    if references["csr_direct"] is not None:
        lines.append(metric_row("direct CSR reference", references["csr_direct"]))
    lines.append(metric_row("direct CSR + PfA", metrics["csr_direct_pfa"]))
    lines.extend(
        [
            "",
            "## Run Setup",
            "",
            f"- epochs: `{results['setup']['epochs']}`",
            f"- hidden neurons: `{results['setup']['hidden']}`",
            f"- SNN timesteps: `{results['setup']['num_steps']}`",
            f"- batch size: `{results['setup']['batch_size']}`",
            f"- structured splits: `{results['runs']['structured_residual_pfa']['history']['splits']}`",
            f"- direct CSR splits: `{results['runs']['csr_direct_pfa']['history']['splits']}`",
            "",
            "The saved JSON contains the full metric dictionaries and training histories.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run both delayed-echo PfA comparisons."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structured_cache = load_cache(args.structured_cache)
    csr_cache = load_cache(args.csr_cache)

    print("Training structured residual + PfA")
    structured, structured_predictions, structured_true = train_variant(
        cache=structured_cache,
        feature_key="features",
        baseline_key="baseline_encoded",
        mode="residual",
        seed=args.training_seed,
        args=args,
    )
    print("Training direct time-binned CSR + PfA")
    csr, csr_predictions, csr_true = train_variant(
        cache=csr_cache,
        feature_key="cochlear_raster_direct_features",
        baseline_key=None,
        mode="direct",
        seed=args.training_seed + 11_000,
        args=args,
    )

    results = {
        "experiment": "pfa_delayed_echo_readouts",
        "setup": {
            "acoustic_condition": "50 dB environmental noise plus controlled delayed echoes",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "training_seed": args.training_seed,
            "attention_location": "after each hidden linear layer and before its LIF neuron",
            "raw_waveform_variant_included": False,
            "structured_cache": str(args.structured_cache),
            "csr_cache": str(args.csr_cache),
        },
        "references": {
            "structured_residual": reference_metrics(args.structured_reference, "residual"),
            "csr_direct": reference_metrics(args.csr_reference, "cochlear_raster_direct"),
        },
        "metrics": {
            "structured_residual_pfa": structured["metrics"],
            "csr_direct_pfa": csr["metrics"],
        },
        "runs": {
            "structured_residual_pfa": structured,
            "csr_direct_pfa": csr,
        },
    }
    json_path = args.output_dir / "pfa_delayed_echo_results.json"
    report_path = args.output_dir / "pfa_delayed_echo_results.md"
    predictions_path = args.output_dir / "pfa_delayed_echo_test_predictions.npz"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(
        predictions_path,
        structured_true_coordinates=structured_true,
        structured_residual_pfa_encoded=structured_predictions,
        csr_true_coordinates=csr_true,
        csr_direct_pfa_encoded=csr_predictions,
    )
    write_report(report_path, results)
    print(report_path)
    print(json_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
