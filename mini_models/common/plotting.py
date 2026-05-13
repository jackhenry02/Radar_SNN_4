from __future__ import annotations

"""Small plotting helpers shared by mini-model scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The created path as a `Path` object.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_figure(fig: plt.Figure, path: str | Path) -> str:
    """Save and close a matplotlib figure.

    Args:
        fig: Figure to save.
        path: Output image path.

    Returns:
        String path to the saved figure.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
