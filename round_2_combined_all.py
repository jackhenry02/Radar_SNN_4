from __future__ import annotations

from stages.round_2_combined_all import run_round_2_combined_all
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    summary = run_round_2_combined_all(config, outputs)
    print(
        f"Round 2 combined-all experiment complete with max_threads={max_threads}, "
        f"dataset_mode={summary['result']['dataset_mode']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
