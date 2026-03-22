from __future__ import annotations

from stages.spike_cache import run_build_matched_human_700_spike_cache
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    summary = run_build_matched_human_700_spike_cache(config, outputs)
    print(
        f"Spike cache ready with max_threads={max_threads} at {summary['cache_path']} "
        f"(reused={summary['reused']}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
