from __future__ import annotations

"""Signal helpers for mini-model acoustic experiments."""

from dataclasses import replace

from utils.common import GlobalConfig


def matched_human_signal_config(base: GlobalConfig | None = None) -> GlobalConfig:
    """Return the matched-human front-end configuration used in recent tests.

    Args:
        base: Optional base configuration. If omitted, a fresh `GlobalConfig`
            is used.

    Returns:
        Configuration with 64 kHz sampling, 18 kHz to 2 kHz chirp, and a
        2 kHz to 20 kHz cochlear band.
    """
    config = GlobalConfig() if base is None else base
    return replace(
        config,
        sample_rate_hz=64_000,
        chirp_start_hz=18_000.0,
        chirp_end_hz=2_000.0,
        cochlea_low_hz=2_000.0,
        cochlea_high_hz=20_000.0,
    )


def moving_notch_signal_config(base: GlobalConfig | None = None) -> GlobalConfig:
    """Return matched-human config with the elevation moving-notch cue enabled.

    Args:
        base: Optional base configuration. If omitted, a fresh `GlobalConfig`
            is used.

    Returns:
        Configuration suitable for signal-analysis plots of the elevation
        spectral notch.
    """
    config = matched_human_signal_config(base)
    return replace(
        config,
        elevation_cue_mode="slope_notch",
        elevation_notch_strength=1.8,
        elevation_notch_width=0.065,
        normalize_spike_envelope=False,
        transmit_gain=1_000.0,
    )

