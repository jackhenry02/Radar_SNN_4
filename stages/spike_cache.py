from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from stages.base import StageContext
from stages.improvement import _sample_dataset_split, _slice_acoustic_batch
from utils.common import GlobalConfig, OutputPaths, save_json, seed_everything


SPIKE_CACHE_SPLITS: dict[str, int] = {
    "train": 7_000,
    "val": 1_500,
    "test": 1_500,
}

SPIKE_CACHE_CHUNK_SIZE = 64
SPIKE_CACHE_VERSION = 1


def _matched_human_700_config(base: GlobalConfig) -> GlobalConfig:
    payload = {**base.__dict__}
    payload.update(
        {
            "sample_rate_hz": 64_000,
            "chirp_start_hz": 18_000.0,
            "chirp_end_hz": 2_000.0,
            "cochlea_low_hz": 2_000.0,
            "cochlea_high_hz": 20_000.0,
            "num_cochlea_channels": 700,
            "cochlea_spacing_mode": "log",
        }
    )
    return GlobalConfig(**payload)


def _pack_spikes(spikes: torch.Tensor) -> np.ndarray:
    spikes_np = spikes.to(torch.bool).cpu().numpy().astype(np.uint8, copy=False)
    return np.packbits(spikes_np, axis=-1, bitorder="little")


def _create_split_datasets(
    group: h5py.Group,
    count: int,
    *,
    channels: int,
    time_steps: int,
    chunk_size: int,
) -> dict[str, h5py.Dataset]:
    packed_steps = math.ceil(time_steps / 8)
    datasets = {
        "transmit_spikes_packed": group.create_dataset(
            "transmit_spikes_packed",
            shape=(count, channels, packed_steps),
            dtype=np.uint8,
            chunks=(min(chunk_size, count), channels, packed_steps),
        ),
        "receive_spikes_packed": group.create_dataset(
            "receive_spikes_packed",
            shape=(count, 2, channels, packed_steps),
            dtype=np.uint8,
            chunks=(min(chunk_size, count), 2, channels, packed_steps),
        ),
        "targets": group.create_dataset(
            "targets",
            shape=(count, 3),
            dtype=np.float32,
            chunks=(min(chunk_size, count), 3),
        ),
        "transmit_spike_count": group.create_dataset(
            "transmit_spike_count",
            shape=(count,),
            dtype=np.float32,
            chunks=(min(chunk_size, count),),
        ),
        "receive_spike_count": group.create_dataset(
            "receive_spike_count",
            shape=(count,),
            dtype=np.float32,
            chunks=(min(chunk_size, count),),
        ),
    }
    group.attrs["packed_bitorder"] = "little"
    group.attrs["original_time_steps"] = time_steps
    group.attrs["packed_time_steps"] = packed_steps
    group.attrs["channels"] = channels
    return datasets


def _write_split(
    h5_group: h5py.Group,
    context: StageContext,
    split_name: str,
    count: int,
    *,
    split_seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    from stages.improvement import _extract_front_end

    generation_start = time.perf_counter()
    acoustic_batch, targets = _sample_dataset_split(context.config, context.device, count, split_seed=split_seed)
    generation_seconds = time.perf_counter() - generation_start

    time_steps = context.config.signal_samples // context.config.envelope_downsample
    datasets = _create_split_datasets(
        h5_group,
        count,
        channels=context.config.num_cochlea_channels,
        time_steps=time_steps,
        chunk_size=chunk_size,
    )

    front_end_seconds = 0.0
    write_seconds = 0.0
    num_chunks = 0
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        num_chunks += 1
        chunk_batch = _slice_acoustic_batch(acoustic_batch, slice(start, stop))

        front_start = time.perf_counter()
        front = _extract_front_end(chunk_batch, context.config, include_cochlea=False)
        front_end_seconds += time.perf_counter() - front_start

        write_start = time.perf_counter()
        transmit_spikes = front["transmit_spikes"]
        receive_spikes = front["receive_spikes"]
        datasets["transmit_spikes_packed"][start:stop] = _pack_spikes(transmit_spikes)
        datasets["receive_spikes_packed"][start:stop] = _pack_spikes(receive_spikes)
        datasets["targets"][start:stop] = targets[start:stop].cpu().numpy().astype(np.float32, copy=False)
        datasets["transmit_spike_count"][start:stop] = (
            transmit_spikes.sum(dim=(-1, -2)).cpu().numpy().astype(np.float32, copy=False)
        )
        datasets["receive_spike_count"][start:stop] = (
            receive_spikes.sum(dim=(-1, -2, -3)).cpu().numpy().astype(np.float32, copy=False)
        )
        write_seconds += time.perf_counter() - write_start

    h5_group.attrs["split_seed"] = split_seed
    h5_group.attrs["chunk_size"] = chunk_size
    h5_group.attrs["count"] = count
    return {
        "count": count,
        "split_seed": split_seed,
        "chunk_size": chunk_size,
        "chunks": num_chunks,
        "dataset_generation_seconds": round(generation_seconds, 4),
        "front_end_seconds": round(front_end_seconds, 4),
        "write_seconds": round(write_seconds, 4),
        "total_seconds": round(generation_seconds + front_end_seconds + write_seconds, 4),
    }


def run_build_matched_human_700_spike_cache(config: GlobalConfig, outputs: OutputPaths) -> dict[str, Any]:
    cache_root = outputs.root / "caches"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "matched_human_700_spikes_10000.h5"
    summary_path = cache_root / "matched_human_700_spikes_10000_summary.json"

    matched_config = _matched_human_700_config(config)
    expected_metadata = {
        "cache_version": SPIKE_CACHE_VERSION,
        "splits": SPIKE_CACHE_SPLITS,
        "sample_rate_hz": matched_config.sample_rate_hz,
        "chirp_start_hz": matched_config.chirp_start_hz,
        "chirp_end_hz": matched_config.chirp_end_hz,
        "cochlea_low_hz": matched_config.cochlea_low_hz,
        "cochlea_high_hz": matched_config.cochlea_high_hz,
        "cochlea_spacing_mode": matched_config.cochlea_spacing_mode,
        "num_cochlea_channels": matched_config.num_cochlea_channels,
        "envelope_rate_hz": matched_config.envelope_rate_hz,
        "signal_samples": matched_config.signal_samples,
        "seed": matched_config.seed,
        "split_seed_offsets": {"train": 70_001, "val": 70_002, "test": 70_003},
    }

    if cache_path.exists() and summary_path.exists():
        summary = h5py.File(cache_path, "r")
        try:
            if all(
                summary.attrs.get(key) == value
                for key, value in expected_metadata.items()
                if key not in {"splits", "split_seed_offsets"}
            ):
                return {
                    **expected_metadata,
                    **json.loads(summary_path.read_text(encoding="utf-8")),
                    "cache_path": str(cache_path),
                    "summary_path": str(summary_path),
                    "reused": True,
                }
        finally:
            summary.close()

    total_start = time.perf_counter()
    stage_context = StageContext(config=matched_config, device=torch.device("cpu"), outputs=outputs)
    seed_everything(matched_config.seed)

    split_seed_offsets = expected_metadata["split_seed_offsets"]
    split_profiles: dict[str, Any] = {}
    with h5py.File(cache_path, "w") as handle:
        for key, value in expected_metadata.items():
            if isinstance(value, dict):
                continue
            handle.attrs[key] = value
        handle.attrs["total_samples"] = sum(SPIKE_CACHE_SPLITS.values())
        handle.attrs["target_names"] = np.asarray(["distance_m", "azimuth_deg", "elevation_deg"], dtype="S16")
        handle.attrs["spike_storage"] = "packbits_last_axis_uint8"

        metadata_group = handle.create_group("metadata")
        metadata_group.attrs["split_seed_offsets_json"] = str(split_seed_offsets)
        metadata_group.attrs["split_counts_json"] = str(SPIKE_CACHE_SPLITS)

        for split_name, count in SPIKE_CACHE_SPLITS.items():
            split_profiles[split_name] = _write_split(
                handle.create_group(split_name),
                stage_context,
                split_name,
                count,
                split_seed=matched_config.seed + int(split_seed_offsets[split_name]),
                chunk_size=SPIKE_CACHE_CHUNK_SIZE,
            )

    total_seconds = time.perf_counter() - total_start
    cache_size_bytes = cache_path.stat().st_size
    summary_payload = {
        "cache_version": SPIKE_CACHE_VERSION,
        "cache_path": str(cache_path),
        "summary_path": str(summary_path),
        "reused": False,
        "total_samples": sum(SPIKE_CACHE_SPLITS.values()),
        "splits": SPIKE_CACHE_SPLITS,
        "config": {
            "sample_rate_hz": matched_config.sample_rate_hz,
            "chirp_start_hz": matched_config.chirp_start_hz,
            "chirp_end_hz": matched_config.chirp_end_hz,
            "cochlea_low_hz": matched_config.cochlea_low_hz,
            "cochlea_high_hz": matched_config.cochlea_high_hz,
            "cochlea_spacing_mode": matched_config.cochlea_spacing_mode,
            "num_cochlea_channels": matched_config.num_cochlea_channels,
            "envelope_rate_hz": matched_config.envelope_rate_hz,
            "signal_samples": matched_config.signal_samples,
            "signal_duration_s": matched_config.signal_duration_s,
            "chirp_duration_s": matched_config.chirp_duration_s,
        },
        "split_seed_offsets": split_seed_offsets,
        "storage": {
            "file_format": "hdf5",
            "packing": "np.packbits on the last time axis with little-endian bit order",
            "transmit_dataset": "/<split>/transmit_spikes_packed",
            "receive_dataset": "/<split>/receive_spikes_packed",
            "targets_dataset": "/<split>/targets",
            "transmit_spike_count_dataset": "/<split>/transmit_spike_count",
            "receive_spike_count_dataset": "/<split>/receive_spike_count",
            "cache_size_bytes": cache_size_bytes,
            "cache_size_mb": round(cache_size_bytes / (1024.0 * 1024.0), 2),
        },
        "timings": {
            "total_seconds": round(total_seconds, 4),
            "split_profiles": split_profiles,
        },
    }
    save_json(summary_path, summary_payload)
    return summary_payload
