from __future__ import annotations

import os

from stages.base import StageContext
from stages.pipeline import PipelineRunner
from utils.common import GlobalConfig, OutputPaths, get_device, limit_backend_resources, save_json, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources()
    device = get_device()
    outputs = OutputPaths.create("outputs")
    context = StageContext(config=config, device=device, outputs=outputs)
    start_stage = os.environ.get("RADAR_SNN_START_STAGE")
    summary = PipelineRunner(context, start_stage=start_stage).run()
    save_json(
        outputs.metrics_path,
        {
            "device": str(device),
            "max_threads": max_threads,
            "start_stage": start_stage,
            "stages": summary.stages,
        },
    )
    print(f"Completed {len(summary.stages)} stages on device={device} with max_threads={max_threads}.")


if __name__ == "__main__":
    main()
