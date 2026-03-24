from __future__ import annotations

from stages.round_4_combined_experiment import run_round_4_combined_experiment
from utils.common import GlobalConfig, OutputPaths


if __name__ == "__main__":
    outputs = OutputPaths.create("outputs")
    summary = run_round_4_combined_experiment(GlobalConfig(), outputs)
    print(summary["report_path"])
