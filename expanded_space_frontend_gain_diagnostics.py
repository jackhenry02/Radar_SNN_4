from __future__ import annotations

from stages.expanded_space_test import run_expanded_space_frontend_gain_diagnostics
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    summary = run_expanded_space_frontend_gain_diagnostics(config, outputs)
    print(
        f"Expanded-space frontend gain diagnostics complete with max_threads={max_threads} at "
        f"outputs/expanded_space_frontend_gain_diagnostics/summary.json for {len(summary['diagnostics'])} cases.",
        flush=True,
    )


if __name__ == "__main__":
    main()
