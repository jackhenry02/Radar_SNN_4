from __future__ import annotations

import os

from stages.expanded_space_test import run_expanded_space_140db_test
from utils.common import GlobalConfig, OutputPaths, limit_backend_resources, seed_everything


def main() -> None:
    config = GlobalConfig()
    seed_everything(config.seed)
    max_threads = limit_backend_resources(1)
    outputs = OutputPaths.create("outputs")
    normalize = os.environ.get("RADAR_SNN_140DB_NORMALIZE", "1").lower() not in {"0", "false", "no"}
    result = run_expanded_space_140db_test(config, outputs, normalize_spike_envelope=normalize)
    print(
        f"Expanded-space 140 dB test complete with max_threads={max_threads} at "
        f"outputs/{result['name']}/result.json (normalize={normalize}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
