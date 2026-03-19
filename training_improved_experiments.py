from __future__ import annotations

import os

from stages.training_improved_experiments import run_training_improved_suite
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    include_experiments = os.environ.get("RADAR_SNN_TRAINING_IMPROVED_RUN_EXPERIMENTS", "0") == "1"
    summary = run_training_improved_suite(config, outputs, include_experiments=include_experiments)
    print(
        f"Training-improved suite complete with max_threads={max_threads}, "
        f"dataset_mode={summary['dataset_mode']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
