from __future__ import annotations

from stages.combined_experiment import run_combined_experiment
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    summary = run_combined_experiment(config, outputs)
    print(
        f"Combined experiment complete with max_threads={max_threads}, "
        f"dataset_mode={summary['dataset_mode']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
