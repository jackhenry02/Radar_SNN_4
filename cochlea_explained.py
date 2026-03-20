from __future__ import annotations

from stages.cochlea_explained import run_cochlea_explained
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    artifacts = run_cochlea_explained(config, outputs)
    print(
        f"Cochlea explanation generated with max_threads={max_threads} at {artifacts['report']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
