from __future__ import annotations

from stages.base import StageContext
from stages.improved_experiments import run_improved_experiments
from utils.common import GlobalConfig, OutputPaths, get_device, limit_backend_resources, save_json, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    context = StageContext(config=config, device=get_device(), outputs=outputs)
    summary = run_improved_experiments(context)
    save_json(
        outputs.root / "improved_experiments_summary.json",
        {
            "max_threads": max_threads,
            "device": str(context.device),
            **summary,
        },
    )
    print(
        f"Improved experiments complete with max_threads={max_threads}, device={context.device}, "
        f"dataset_mode={summary['dataset_mode']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
