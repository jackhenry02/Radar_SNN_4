from __future__ import annotations

from stages.round_3_combined_experiments import run_round_3_combined_experiments
from utils.common import GlobalConfig, OutputPaths


def main() -> None:
    config = GlobalConfig()
    outputs = OutputPaths.create("outputs")
    summary = run_round_3_combined_experiments(config, outputs)
    print(summary["report_path"])


if __name__ == "__main__":
    main()
