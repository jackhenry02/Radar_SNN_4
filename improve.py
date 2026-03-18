from __future__ import annotations

import os

from stages.base import StageContext
from stages.improvement import ImprovementRunner
from utils.common import GlobalConfig, OutputPaths, get_device, limit_backend_resources, save_json, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    if "RADAR_SNN_DEVICE" not in os.environ:
        os.environ["RADAR_SNN_DEVICE"] = "cpu"
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    context = StageContext(config=config, device=get_device(), outputs=outputs)
    summary = ImprovementRunner(context).run()
    save_json(
        outputs.root / "improvement_summary.json",
        {"max_threads": max_threads, "device": str(context.device), "stages": summary.stages, "report_path": summary.report_path},
    )
    print(f"Improvement phase complete with max_threads={max_threads}, device={context.device}.")


if __name__ == "__main__":
    main()
