from __future__ import annotations

from stages.expanded_space_test import run_expanded_space_control_test
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    summary = run_expanded_space_control_test(config, outputs)
    print(
        f"Original-range control test complete with max_threads={max_threads} at {summary['report_path']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
