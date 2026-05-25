from __future__ import annotations

"""Train raw-waveform and cochlear-raster SNN baselines.

These baselines test whether the crafted pathway feature extraction is useful by
replacing it with minimally processed binaural inputs. The downstream readout is
kept as close as possible to the final trainable readout: same target encoding,
same snnTorch LIF architecture, same optimiser, and same uncertainty-weighted
loss. Projected and non-projected versions are included to show the accuracy and
parameter-count motivation for the fixed projection.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_pathway.experiments import full_distance_pathway_model as fdm
from elevation_pathway.experiments import elevation_pathway_first_attempt as elev
from final_model.experiments import environment_noise_diagnostics as envdiag
from final_model.experiments import final_model_results as final
from final_model.experiments import trainable_final_readout as readout
from mini_models.common.plotting import ensure_dir, save_figure


OUTPUT_DIR = ROOT / "final_model" / "outputs" / "trainable_input_baselines"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "final_model" / "reports" / "trainable_input_baselines.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"
CACHE_PATH = OUTPUT_DIR / "cache.npz"

DISTANCE_MIN_M = readout.DISTANCE_MIN_M
DISTANCE_MAX_M = readout.DISTANCE_MAX_M
AZIMUTH_LIMIT_DEG = readout.AZIMUTH_LIMIT_DEG
ELEVATION_LIMIT_DEG = readout.ELEVATION_LIMIT_DEG
CHANNELS = readout.CHANNELS
FEATURE_DIM = readout.build_feature_spec().input_dim
COCHLEAR_TIME_BINS = 10
PROJECTION_SEED = 42_404
DATASET_SEED = readout.DATASET_SEED
TRAINING_SEED = readout.TRAINING_SEED
HIDDEN_DIM = readout.HIDDEN_DIM
NUM_STEPS = readout.NUM_STEPS
BATCH_SIZE = readout.BATCH_SIZE
EPOCHS = readout.EPOCHS
LEARNING_RATE = readout.LEARNING_RATE
WEIGHT_DECAY = readout.WEIGHT_DECAY
NOISE_STD = 0.0
NOISE_LABEL = "clean"
ACOUSTIC_MODE = "clean"
ENVIRONMENT_NOISE_DB = 0.0
ENVIRONMENT_REVERB = False
FORCE_CACHE = False
SPLITS = {"train": 48, "val": 16, "test": 16}
RUN_LABEL = "smoke"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=48, help="Number of training samples.")
    parser.add_argument("--val", type=int, default=16, help="Number of validation samples.")
    parser.add_argument("--test", type=int, default=16, help="Number of held-out test samples.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs for each baseline.")
    parser.add_argument("--hidden", type=int, default=96, help="Hidden neurons in the SNN readout.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--dataset-seed", type=int, default=DATASET_SEED, help="Random seed for target sampling.")
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED, help="Random seed for SNN training.")
    parser.add_argument("--projection-seed", type=int, default=PROJECTION_SEED, help="Fixed random projection seed.")
    parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM, help="Projected input dimension.")
    parser.add_argument("--cochlear-time-bins", type=int, default=COCHLEAR_TIME_BINS, help="Temporal bins per ear/channel for cochlear rasters.")
    parser.add_argument("--force-cache", action="store_true", help="Regenerate projected inputs even if the cache exists.")
    parser.add_argument("--noise-db-spl", type=float, default=None, help="Add receiver noise using the old dB-SPL convention.")
    parser.add_argument("--noise-std", type=float, default=None, help="Add receiver noise with this waveform standard deviation.")
    parser.add_argument("--environment-noise-db", type=float, default=None, help="Add call-referenced environmental noise before head-shadow/elevation filtering.")
    parser.add_argument("--environment-reverb", action="store_true", help="Add late echo copies in the environmental-noise simulator.")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    """Apply CLI settings to module globals."""
    global SPLITS, DATASET_SEED, TRAINING_SEED, PROJECTION_SEED, FEATURE_DIM, COCHLEAR_TIME_BINS
    global HIDDEN_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, FORCE_CACHE
    global NOISE_STD, NOISE_LABEL, ACOUSTIC_MODE, ENVIRONMENT_NOISE_DB, ENVIRONMENT_REVERB
    global RUN_LABEL, FIGURE_DIR, CACHE_PATH, RESULTS_PATH

    SPLITS = {"train": args.train, "val": args.val, "test": args.test}
    DATASET_SEED = args.dataset_seed
    TRAINING_SEED = args.training_seed
    PROJECTION_SEED = args.projection_seed
    FEATURE_DIM = args.feature_dim
    COCHLEAR_TIME_BINS = args.cochlear_time_bins
    HIDDEN_DIM = args.hidden
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    FORCE_CACHE = args.force_cache

    if args.environment_noise_db is not None and (args.noise_std is not None or args.noise_db_spl is not None):
        raise ValueError("Use either receiver-noise flags or --environment-noise-db, not both.")
    if args.environment_reverb and args.environment_noise_db is None:
        raise ValueError("--environment-reverb requires --environment-noise-db.")
    if args.noise_std is not None and args.noise_db_spl is not None:
        raise ValueError("Use either --noise-std or --noise-db-spl, not both.")

    if args.environment_noise_db is not None:
        NOISE_STD = 0.0
        ACOUSTIC_MODE = "environment_noise_reverb" if args.environment_reverb else "environment_noise"
        ENVIRONMENT_NOISE_DB = float(args.environment_noise_db)
        ENVIRONMENT_REVERB = bool(args.environment_reverb)
        base = f"envnoise{int(round(ENVIRONMENT_NOISE_DB))}dB"
        NOISE_LABEL = f"{base}_reverb" if ENVIRONMENT_REVERB else base
    elif args.noise_db_spl is not None:
        NOISE_STD = fdm._noise_std_from_db(float(args.noise_db_spl))
        NOISE_LABEL = f"noise{int(round(float(args.noise_db_spl)))}dB"
        ACOUSTIC_MODE = "receiver_noise"
        ENVIRONMENT_NOISE_DB = 0.0
        ENVIRONMENT_REVERB = False
    elif args.noise_std is not None:
        NOISE_STD = float(args.noise_std)
        NOISE_LABEL = f"noise_std_{str(args.noise_std).replace('.', 'p')}"
        ACOUSTIC_MODE = "receiver_noise"
        ENVIRONMENT_NOISE_DB = 0.0
        ENVIRONMENT_REVERB = False
    else:
        NOISE_STD = 0.0
        NOISE_LABEL = "clean"
        ACOUSTIC_MODE = "clean"
        ENVIRONMENT_NOISE_DB = 0.0
        ENVIRONMENT_REVERB = False

    base_label = f"train{args.train}_val{args.val}_test{args.test}"
    RUN_LABEL = base_label if NOISE_LABEL == "clean" else f"{base_label}_{NOISE_LABEL}"
    FIGURE_DIR = OUTPUT_DIR / "figures" / RUN_LABEL
    CACHE_PATH = OUTPUT_DIR / f"cache_{RUN_LABEL}.npz"
    RESULTS_PATH = OUTPUT_DIR / f"results_{RUN_LABEL}.json"


def _fixed_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    """Return a deterministic Gaussian projection matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / math.sqrt(max(input_dim, 1)), size=(input_dim, output_dim)).astype(np.float32)


def _project(vectors: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Project high-dimensional inputs into the SNN feature dimension."""
    return (vectors.astype(np.float32) @ _fixed_projection(vectors.shape[1], output_dim, seed)).astype(np.float32)


def _time_bin_spikes(spikes: np.ndarray, bins: int) -> np.ndarray:
    """Convert an ear/channel/time spike raster into channel-by-time-bin counts."""
    chunks = np.array_split(spikes.astype(np.float32), bins, axis=1)
    return np.concatenate([chunk.sum(axis=1) for chunk in chunks], axis=0)


def _simulate_receive(config: fdm.GlobalConfig, distance_m: float, azimuth_deg: float, elevation_deg: float, index: int) -> torch.Tensor:
    """Generate the shared binaural waveform for one target."""
    if ACOUSTIC_MODE in {"environment_noise", "environment_noise_reverb"}:
        noise_std = envdiag._call_referenced_noise_std(config, ENVIRONMENT_NOISE_DB)
        rng = torch.Generator().manual_seed(DATASET_SEED + 70_000 + index)
        condition = envdiag.AcousticCondition(
            key=ACOUSTIC_MODE,
            name=NOISE_LABEL,
            add_environment_noise=True,
            add_reverb=ENVIRONMENT_REVERB,
        )
        return envdiag._simulate_environment_echo(
            config,
            distance_m,
            azimuth_deg,
            elevation_deg,
            condition=condition,
            noise_std=noise_std,
            rng=rng,
        )

    receive = elev.simulate_full_3d_scene(
        config,
        distance_m,
        azimuth_deg,
        elevation_deg,
        elev.DEEP_COMB_DELAYED_COPY_GAIN,
    )
    if ACOUSTIC_MODE == "receiver_noise" and NOISE_STD > 0.0:
        generator = torch.Generator().manual_seed(DATASET_SEED + 80_000 + index)
        receive = receive + NOISE_STD * torch.randn(receive.shape, generator=generator, dtype=receive.dtype)
    return receive


def build_cache() -> dict[str, object]:
    """Generate or load raw/direct and projected waveform/raster inputs."""
    if CACHE_PATH.exists() and not FORCE_CACHE:
        print(f"Loading cached baseline inputs: {CACHE_PATH}")
        loaded = np.load(CACHE_PATH, allow_pickle=True)
        return {key: loaded[key].item() if loaded[key].shape == () else loaded[key] for key in loaded.files}

    ensure_dir(OUTPUT_DIR)
    total = sum(SPLITS.values())
    distance, azimuth, elevation = readout.random_targets(total, DATASET_SEED)
    split_names = np.array(["train"] * SPLITS["train"] + ["val"] * SPLITS["val"] + ["test"] * SPLITS["test"])
    raw_vectors = []
    cochlear_vectors = []
    feature_seconds = []

    with final.temporary_model_constants(
        channels=CHANNELS,
        distance_bins=readout.DISTANCE_BINS,
        angular_bins=readout.ANGULAR_BINS,
        elevation_limit_deg=ELEVATION_LIMIT_DEG,
    ):
        config = replace(final.make_distance_config(CHANNELS, DISTANCE_MIN_M, DISTANCE_MAX_M), noise_std=NOISE_STD)
        for index, (distance_m, azimuth_deg, elevation_deg) in enumerate(zip(distance, azimuth, elevation)):
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == total:
                print(f"  cached {index + 1}/{total} baseline samples")
            start = time.perf_counter()
            receive = _simulate_receive(config, float(distance_m), float(azimuth_deg), float(elevation_deg), index)
            raw_vectors.append(receive.detach().cpu().numpy().reshape(-1).astype(np.float32))

            cochlea = fdm._run_cochlea_binaural(config, receive)
            left = _time_bin_spikes(cochlea.left_spikes.detach().cpu().numpy(), COCHLEAR_TIME_BINS)
            right = _time_bin_spikes(cochlea.right_spikes.detach().cpu().numpy(), COCHLEAR_TIME_BINS)
            cochlear_vectors.append(np.concatenate([left, right], axis=0).astype(np.float32))
            feature_seconds.append(time.perf_counter() - start)

    raw_high = np.stack(raw_vectors, axis=0)
    cochlear_high = np.stack(cochlear_vectors, axis=0)
    raw_features = _project(raw_high, FEATURE_DIM, PROJECTION_SEED)
    cochlear_features = _project(cochlear_high, FEATURE_DIM, PROJECTION_SEED + 1)
    targets_encoded = readout.encode_targets(distance, azimuth, elevation)
    true_coordinates = np.stack([distance, azimuth, elevation], axis=1).astype(np.float32)
    payload = {
        "raw_waveform_projected_features": raw_features,
        "raw_waveform_direct_features": raw_high.astype(np.float32),
        "cochlear_raster_projected_features": cochlear_features,
        "cochlear_raster_direct_features": cochlear_high.astype(np.float32),
        "targets_encoded": targets_encoded,
        "true_coordinates": true_coordinates,
        "split_names": split_names,
        "feature_seconds_per_sample": float(np.mean(feature_seconds)),
        "setup": {
            "splits": SPLITS,
            "dataset_seed": DATASET_SEED,
            "projection_seed": PROJECTION_SEED,
            "feature_dim": FEATURE_DIM,
            "cochlear_time_bins": COCHLEAR_TIME_BINS,
            "raw_high_dim": int(raw_high.shape[1]),
            "cochlear_high_dim": int(cochlear_high.shape[1]),
            "acoustic_mode": ACOUSTIC_MODE,
            "noise_label": NOISE_LABEL,
            "noise_std": NOISE_STD,
            "environment_noise_db": ENVIRONMENT_NOISE_DB,
            "environment_reverb": ENVIRONMENT_REVERB,
        },
    }
    np.savez_compressed(CACHE_PATH, **payload)
    return payload


def _split(cache: dict[str, object], key: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features, targets, and true coordinates for one split."""
    mask = np.asarray(cache["split_names"]) == split
    return (
        np.asarray(cache[key], dtype=np.float32)[mask],
        np.asarray(cache["targets_encoded"], dtype=np.float32)[mask],
        np.asarray(cache["true_coordinates"], dtype=np.float32)[mask],
    )


def _make_loader(features: np.ndarray, targets: np.ndarray, shuffle: bool) -> DataLoader:
    """Create a PyTorch dataloader."""
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(targets.astype(np.float32)))
    return DataLoader(dataset, batch_size=min(BATCH_SIZE, max(1, len(dataset))), shuffle=shuffle)


def _parameter_count(input_dim: int) -> int:
    """Return trainable parameter count for the SNN readout at this input width."""
    return sum(
        parameter.numel()
        for parameter in readout.SmallSNNReadout(input_dim, HIDDEN_DIM, 5, NUM_STEPS).parameters()
    ) + readout.UncertaintyWeightedLoss().log_vars.numel()


def train_baseline(name: str, cache: dict[str, object]) -> tuple[readout.SmallSNNReadout, readout.UncertaintyWeightedLoss, dict[str, object], np.ndarray, np.ndarray]:
    """Train one direct SNN baseline."""
    train_x, train_y, _ = _split(cache, name, "train")
    val_x, val_y, _ = _split(cache, name, "val")
    test_x, _, test_true = _split(cache, name, "test")
    mean, std = readout.fit_normaliser(train_x)
    train_x = readout.apply_normaliser(train_x, mean, std)
    val_x = readout.apply_normaliser(val_x, mean, std)
    test_x = readout.apply_normaliser(test_x, mean, std)

    seed_offsets = {
        "raw_waveform_projected_features": 0,
        "raw_waveform_direct_features": 1_000,
        "cochlear_raster_projected_features": 10_000,
        "cochlear_raster_direct_features": 11_000,
    }
    torch.manual_seed(TRAINING_SEED + seed_offsets[name])
    model = readout.SmallSNNReadout(train_x.shape[1], HIDDEN_DIM, 5, NUM_STEPS)
    criterion = readout.UncertaintyWeightedLoss()
    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_loader = _make_loader(train_x, train_y, shuffle=True)
    val_loader = _make_loader(val_x, val_y, shuffle=False)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        criterion.train()
        train_losses = []
        for features, targets in train_loader:
            prediction = model(features)
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
            for features, targets in val_loader:
                val_losses.append(float(criterion(model(features), targets).item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
                "criterion": {key: value.detach().clone() for key, value in criterion.state_dict().items()},
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        criterion.load_state_dict(best_state["criterion"])

    prediction = _predict(model, test_x)
    history.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "training_seconds": time.perf_counter() - start,
            "learned_log_vars": criterion.log_vars.detach().cpu().numpy().tolist(),
            "input_dim": int(train_x.shape[1]),
            "parameter_count": _parameter_count(int(train_x.shape[1])),
        }
    )
    return model, criterion, history, prediction, test_true


def _predict(model: readout.SmallSNNReadout, features: np.ndarray) -> np.ndarray:
    """Predict encoded coordinates for a feature array."""
    model.eval()
    outputs = []
    loader = _make_loader(features, np.zeros((features.shape[0], 5), dtype=np.float32), shuffle=False)
    with torch.no_grad():
        for batch_features, _ in loader:
            outputs.append(model(batch_features).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def plot_training(histories: dict[str, dict[str, object]], path: Path) -> str:
    """Plot baseline training curves."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = {
        "raw_waveform_projected": "raw waveform projected",
        "raw_waveform_direct": "raw waveform direct",
        "cochlear_raster_projected": "cochlear raster projected",
        "cochlear_raster_direct": "cochlear raster direct",
    }
    for name, history in histories.items():
        ax.plot(history["train_loss"], label=f"{labels[name]} train")
        ax.plot(history["val_loss"], linestyle="--", label=f"{labels[name]} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("uncertainty-weighted loss")
    ax.set_title("Input baseline SNN training")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def run_report_path() -> Path:
    """Return the run-labelled report path."""
    return REPORT_PATH.parent / f"trainable_input_baselines_{RUN_LABEL}.md"


def write_report(results: dict[str, object], artifacts: dict[str, str], report_path: Path | None = None) -> None:
    """Write the markdown experiment report."""
    report_path = run_report_path() if report_path is None else report_path
    lines = [
        "# Trainable Input Baselines",
        "",
        "This report tests control inputs for the same small SNN readout used in the final trainable model. The goal is to check whether the crafted brain-inspired pathway features are useful, rather than simply asking whether any small SNN can learn the constrained localisation task.",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| distance range | `{DISTANCE_MIN_M}-{DISTANCE_MAX_M} m` |",
        f"| azimuth/elevation range | `+/-{AZIMUTH_LIMIT_DEG} deg` |",
        f"| train / val / test samples | `{SPLITS['train']} / {SPLITS['val']} / {SPLITS['test']}` |",
        f"| acoustic mode | `{ACOUSTIC_MODE}` |",
        f"| noise label | `{NOISE_LABEL}` |",
        f"| feature dimension after projection | `{FEATURE_DIM}` |",
        f"| raw waveform dimension before projection | `{results['cache']['raw_high_dim']}` |",
        f"| cochlear raster dimension before projection | `{results['cache']['cochlear_high_dim']}` |",
        f"| cochlear temporal bins | `{COCHLEAR_TIME_BINS}` |",
        f"| projection seed | `{PROJECTION_SEED}` |",
        f"| hidden neurons | `{HIDDEN_DIM}` |",
        f"| SNN timesteps | `{NUM_STEPS}` |",
        f"| batch size | `{BATCH_SIZE}` |",
        f"| epochs | `{EPOCHS}` |",
        f"| optimiser | `AdamW`, lr `{LEARNING_RATE}`, weight decay `{WEIGHT_DECAY}` |",
        "",
        "## Projection Method",
        "",
        "The raw waveform baseline flattens the binaural waveform as `[left waveform, right waveform]`. The cochlear baseline first converts each binaural cochlear spike raster into channel-by-time-bin spike counts, then flattens `[left counts, right counts]`.",
        "",
        "The projected variants compress these high-dimensional vectors by a fixed deterministic Gaussian projection:",
        "",
        "$$",
        "z = xR, \\qquad R_{ij}\\sim\\mathcal{N}\\left(0,\\frac{1}{D}\\right),",
        "$$",
        "",
        "where $D$ is the input dimensionality before projection. The projection is not learned. This keeps the trainable capacity concentrated in the same small snnTorch readout rather than giving the raw/cochlear baselines a large extra trainable front end.",
        "",
        "The direct variants skip this projection and feed the flattened waveform or time-binned cochlear raster directly into the same SNN architecture. They therefore have a larger first linear layer. This is included as a control to show whether projection is only a convenience or whether it changes the baseline behaviour.",
        "",
        "## Model Size",
        "",
        "| Input baseline | Input dimension | Trainable parameters |",
        "|---|---:|---:|",
    ]
    for name, history in results["histories"].items():
        lines.append(f"| {name} | `{history['input_dim']}` | `{history['parameter_count']:,}` |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Input baseline | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metric in results["test_metrics"].items():
        lines.append(
            f"| {name} | `{metric['distance_mae_m']:.4f} m` | `{metric['azimuth_mae_deg']:.3f} deg` | "
            f"`{metric['elevation_mae_deg']:.3f} deg` | `{metric['euclidean_mae_m']:.4f} m` | "
            f"`{metric['combined_normalised_error']:.4f}` |"
        )
    lines.extend(
        [
            "",
            f"Mean cache generation time was `{results['cache']['feature_seconds_per_sample']:.3f} s/sample`.",
            "",
            f"![Training curves]({Path(os.path.relpath(artifacts['training_curves'], report_path.parent)).as_posix()})",
            "",
            f"![Prediction scatter]({Path(os.path.relpath(artifacts['prediction_scatter'], report_path.parent)).as_posix()})",
            "",
            "## Interpretation Template",
            "",
            "- If the raw waveform baseline performs similarly to the crafted-feature SNN, then much of the task can be learned directly from waveform samples in this constrained space.",
            "- If the cochlear raster baseline performs well but the raw waveform baseline does not, then the cochlear front end is doing useful representation work, but the later hand-designed pathways may be less essential.",
            "- If both baselines are worse than the crafted-feature SNN, this supports the argument that the distance, azimuth, elevation, and CANN feature extraction stages improve sample efficiency and interpretability.",
            "- A strong baseline result should still be treated carefully: the projection is fixed and non-biological, and the test space is constrained to the same range used for training.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{Path(path).relative_to(ROOT)}`")
    lines.append(f"- `cache`: `{CACHE_PATH.relative_to(ROOT)}`")
    lines.append(f"- `results`: `{RESULTS_PATH.relative_to(ROOT)}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict[str, object]:
    """Run the input baseline experiment."""
    start = time.perf_counter()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_PATH.parent)
    cache = build_cache()
    histories = {}
    predictions = {}
    test_true = None
    for key, label in [
        ("raw_waveform_projected_features", "raw_waveform_projected"),
        ("raw_waveform_direct_features", "raw_waveform_direct"),
        ("cochlear_raster_projected_features", "cochlear_raster_projected"),
        ("cochlear_raster_direct_features", "cochlear_raster_direct"),
    ]:
        print(f"Training {label} baseline for {EPOCHS} epochs")
        _, _, history, prediction, true_coordinates = train_baseline(key, cache)
        histories[label] = history
        predictions[label] = prediction
        test_true = true_coordinates

    assert test_true is not None
    test_metrics = {name: readout.coordinate_metrics(test_true, pred) for name, pred in predictions.items()}
    artifacts = {
        "training_curves": plot_training(histories, FIGURE_DIR / "training_curves.png"),
        "prediction_scatter": readout.plot_prediction_scatter(test_true, predictions, FIGURE_DIR / "prediction_scatter.png"),
    }
    setup = dict(cache["setup"].item() if hasattr(cache["setup"], "item") else cache["setup"])
    results = {
        "experiment": "trainable_input_baselines",
        "elapsed_seconds": time.perf_counter() - start,
        "setup": {
            **setup,
            "run_label": RUN_LABEL,
            "training_seed": TRAINING_SEED,
            "hidden_dim": HIDDEN_DIM,
            "num_steps": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "cache": {
            "path": str(CACHE_PATH),
            "feature_seconds_per_sample": float(cache["feature_seconds_per_sample"]),
            "raw_high_dim": int(setup["raw_high_dim"]),
            "cochlear_high_dim": int(setup["cochlear_high_dim"]),
            "feature_dim": FEATURE_DIM,
        },
        "histories": histories,
        "test_metrics": test_metrics,
        "artifacts": artifacts,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, artifacts)
    write_report(results, artifacts, report_path=REPORT_PATH)
    return results


if __name__ == "__main__":
    configure(parse_args())
    main()
    print(REPORT_PATH)
    print(run_report_path())
    print(RESULTS_PATH)
