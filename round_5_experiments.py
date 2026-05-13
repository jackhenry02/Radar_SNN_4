from __future__ import annotations

from stages.round_5_experiments import run_round_5_experiments
from utils.common import GlobalConfig, OutputPaths


if __name__ == "__main__":
    outputs = OutputPaths.create("outputs")
    summary = run_round_5_experiments(GlobalConfig(), outputs)
    print(summary["report_path"])

