from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "presentation_draft_assets"
MARKDOWN_PATH = ROOT / "outputs" / "presentation_draft.md"


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _render_frame(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return Image.fromarray(rgba.reshape(height, width, 4), mode="RGBA")


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 90) -> None:
    if not frames:
        raise ValueError("No frames to save.")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _lif_sim(input_current: np.ndarray, beta: float, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    membrane = np.zeros_like(input_current, dtype=float)
    spikes = np.zeros_like(input_current, dtype=float)
    state = 0.0
    for index, current in enumerate(input_current):
        state = beta * state + current
        if state >= threshold:
            spikes[index] = 1.0
            state = max(0.0, state - threshold)
        membrane[index] = state
    return membrane, spikes


def _resonator_sim(
    input_current: np.ndarray,
    omega: float,
    decay: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = 0.0
    velocity = 0.0
    state_trace = np.zeros_like(input_current, dtype=float)
    velocity_trace = np.zeros_like(input_current, dtype=float)
    spikes = np.zeros_like(input_current, dtype=float)
    for index, current in enumerate(input_current):
        velocity = decay * velocity + current - omega * state
        state = state + omega * velocity
        if state >= threshold:
            spikes[index] = 1.0
            state -= threshold
        state_trace[index] = state
        velocity_trace[index] = velocity
    return state_trace, velocity_trace, spikes


def _save_lif_animation() -> Path:
    path = OUTPUT_DIR / "lif_neuron.gif"
    steps = 70
    t = np.arange(steps)
    current = np.zeros(steps)
    current[8:17] = 0.22
    current[30:38] = 0.16
    current[38:45] = 0.28
    current[54:60] = 0.22
    membrane, spikes = _lif_sim(current, beta=0.90, threshold=1.0)

    frames: list[Image.Image] = []
    for frame in range(steps):
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        fig.suptitle("Leaky Integrate-and-Fire Neuron", fontsize=16, fontweight="bold")

        axes[0].plot(t[: frame + 1], current[: frame + 1], color="#0b7285", lw=2.5)
        axes[0].axvline(frame, color="#999999", ls="--", lw=1)
        axes[0].set_ylabel("Input")
        axes[0].set_xlim(0, steps - 1)
        axes[0].set_ylim(0, 0.34)
        axes[0].set_title("Input current")
        _style_axes(axes[0])

        axes[1].plot(t[: frame + 1], membrane[: frame + 1], color="#1d4ed8", lw=2.5)
        axes[1].axhline(1.0, color="#dc2626", ls="--", lw=1.5, label="threshold")
        spike_times = np.where(spikes[: frame + 1] > 0)[0]
        if spike_times.size:
            axes[1].scatter(spike_times, np.full_like(spike_times, 1.05), color="#dc2626", s=35, zorder=4)
        axes[1].axvline(frame, color="#999999", ls="--", lw=1)
        axes[1].set_ylabel("Membrane")
        axes[1].set_xlim(0, steps - 1)
        axes[1].set_ylim(0, 1.15)
        axes[1].set_title("Membrane integration and reset")
        _style_axes(axes[1])

        axes[2].stem(t[: frame + 1], spikes[: frame + 1], linefmt="#dc2626", markerfmt=" ", basefmt=" ")
        axes[2].axvline(frame, color="#999999", ls="--", lw=1)
        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Spike")
        axes[2].set_xlim(0, steps - 1)
        axes[2].set_ylim(0, 1.2)
        axes[2].set_title("Output spikes")
        _style_axes(axes[2])

        fig.text(
            0.53,
            0.03,
            r"$m[t+1] = \beta m[t] + I[t] - \theta s[t],\quad s[t] = H(m[t]-\theta)$",
            ha="center",
            fontsize=12,
        )
        frames.append(_render_frame(fig))
        plt.close(fig)

    _save_gif(frames, path)
    return path


def _save_resonant_animation() -> Path:
    path = OUTPUT_DIR / "resonant_neuron.gif"
    steps = 90
    t = np.arange(steps)
    current = np.zeros(steps)
    current[10] = 1.2
    current[46] = 1.0
    state, velocity, spikes = _resonator_sim(current, omega=0.42, decay=0.96, threshold=1.0)

    frames: list[Image.Image] = []
    for frame in range(steps):
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        fig.suptitle("Resonant Spiking Neuron", fontsize=16, fontweight="bold")

        axes[0].stem(t[: frame + 1], current[: frame + 1], linefmt="#0f766e", markerfmt=" ", basefmt=" ")
        axes[0].axvline(frame, color="#999999", ls="--", lw=1)
        axes[0].set_xlim(0, steps - 1)
        axes[0].set_ylim(0, 1.35)
        axes[0].set_ylabel("Drive")
        axes[0].set_title("Impulse input")
        _style_axes(axes[0])

        axes[1].plot(t[: frame + 1], state[: frame + 1], color="#7c3aed", lw=2.5, label="state")
        axes[1].plot(t[: frame + 1], velocity[: frame + 1], color="#f59e0b", lw=1.8, alpha=0.85, label="velocity")
        axes[1].axhline(1.0, color="#dc2626", ls="--", lw=1.5)
        spike_times = np.where(spikes[: frame + 1] > 0)[0]
        if spike_times.size:
            axes[1].scatter(spike_times, np.full_like(spike_times, 1.05), color="#dc2626", s=35, zorder=4)
        axes[1].axvline(frame, color="#999999", ls="--", lw=1)
        axes[1].set_xlim(0, steps - 1)
        axes[1].set_ylim(-1.2, 1.2)
        axes[1].set_ylabel("State")
        axes[1].set_title("Oscillatory state")
        _style_axes(axes[1])

        axes[2].stem(t[: frame + 1], spikes[: frame + 1], linefmt="#dc2626", markerfmt=" ", basefmt=" ")
        axes[2].axvline(frame, color="#999999", ls="--", lw=1)
        axes[2].set_xlim(0, steps - 1)
        axes[2].set_ylim(0, 1.2)
        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Spike")
        axes[2].set_title("Output spikes")
        _style_axes(axes[2])

        fig.text(
            0.53,
            0.03,
            r"$v[t+1] = \alpha v[t] + u[t] - \omega z[t],\quad z[t+1] = z[t] + \omega v[t]$",
            ha="center",
            fontsize=12,
        )
        frames.append(_render_frame(fig))
        plt.close(fig)

    _save_gif(frames, path)
    return path


def _save_coincidence_animation() -> Path:
    path = OUTPUT_DIR / "coincidence_bank.gif"
    steps = 60
    t = np.arange(steps)
    tx_start = 4
    echo_time = 24
    arrival_times = [16, 24, 32]
    labels = ["Too early", "Just right", "Too late"]
    taus = [12, 20, 28]
    colors = ["#64748b", "#16a34a", "#64748b"]
    mems = []
    outs = []
    for arrival in arrival_times:
        current = np.zeros(steps)
        current[arrival] += 0.95
        current[echo_time] += 0.95
        membrane, spikes = _lif_sim(current, beta=0.68, threshold=1.55)
        mems.append(membrane)
        outs.append(spikes)

    frames: list[Image.Image] = []
    for frame in range(steps):
        fig = plt.figure(figsize=(11, 7), constrained_layout=True)
        subfigs = fig.subfigures(1, 2, width_ratios=[1.2, 1.0])
        left_axes = subfigs[0].subplots(3, 1)
        right_axes = subfigs[1].subplots(3, 1)
        fig.patch.set_facecolor("white")
        fig.suptitle("Delay-Tuned Coincidence Detection", fontsize=17, fontweight="bold")

        for idx, (ax_l, ax_r) in enumerate(zip(left_axes, right_axes)):
            ax_l.set_xlim(0, 10)
            ax_l.set_ylim(0, 2)
            ax_l.axis("off")
            ax_l.text(0.1, 1.78, f"{labels[idx]}  ($\\tau_{idx+1}={taus[idx]}$)", fontsize=12, weight="bold")
            ax_l.plot([1, 8], [1.25, 1.25], color="#334155", lw=2)
            ax_l.add_patch(FancyArrowPatch((8, 0.25), (8, 1.0), arrowstyle="-|>", mutation_scale=16, lw=2, color="#ef4444"))
            ax_l.text(0.2, 1.28, "Transmit delay line", fontsize=10)
            ax_l.text(8.25, 0.55, "Echo", fontsize=10, color="#ef4444")

            if frame >= tx_start:
                progress = min(1.0, (frame - tx_start) / max(1, arrival_times[idx] - tx_start))
                x_pos = 1 + 7 * progress
                ax_l.scatter([x_pos], [1.25], s=90, color="#0f766e", zorder=4)
            if frame <= echo_time:
                echo_progress = max(0.0, min(1.0, frame / max(1, echo_time)))
                y_pos = 0.25 + 1.0 * echo_progress
                ax_l.scatter([8], [y_pos], s=90, color="#ef4444", zorder=4)
            elif frame > echo_time:
                ax_l.scatter([8], [1.25], s=90, color="#ef4444", zorder=4)

            neuron_color = colors[idx] if outs[idx][frame] > 0 else "#ffffff"
            circle = Circle((8.9, 1.25), 0.35, edgecolor=colors[idx], facecolor=neuron_color, lw=2.5)
            ax_l.add_patch(circle)
            ax_l.text(8.9, 1.25, "LIF", ha="center", va="center", fontsize=10, color="#111827")
            if outs[idx][frame] > 0:
                ax_l.scatter([9.7], [1.25], marker="*", s=180, color="#dc2626", zorder=5)
                ax_l.text(9.25, 1.62, "spike", color="#dc2626", fontsize=10)

            ax_r.plot(t[: frame + 1], mems[idx][: frame + 1], color=colors[idx], lw=2.5)
            ax_r.axhline(1.55, color="#dc2626", ls="--", lw=1.2)
            spike_times = np.where(outs[idx][: frame + 1] > 0)[0]
            if spike_times.size:
                ax_r.scatter(spike_times, np.full_like(spike_times, 1.62), color="#dc2626", s=30, zorder=4)
            ax_r.axvline(frame, color="#999999", ls="--", lw=1)
            ax_r.set_xlim(0, steps - 1)
            ax_r.set_ylim(0, 1.75)
            ax_r.set_ylabel("Membrane")
            ax_r.set_title("Detector membrane")
            _style_axes(ax_r)

        fig.text(
            0.53,
            0.03,
            r"$m_i[t+1] = \beta_i m_i[t] + w_{\mathrm{tx}}x[t-\tau_i] + w_{\mathrm{echo}}y[t] - \theta s_i[t]$",
            ha="center",
            fontsize=12,
        )
        fig.text(
            0.53,
            0.005,
            r"Cross-correlation analogy: $c[\tau] = \sum_t x[t-\tau]\,y[t]$",
            ha="center",
            fontsize=11,
        )
        frames.append(_render_frame(fig))
        plt.close(fig)

    _save_gif(frames, path, duration_ms=110)
    return path


def _save_resonance_bank_animation() -> Path:
    path = OUTPUT_DIR / "resonance_bank.gif"
    steps = 90
    t = np.arange(steps)
    tx_time = 8
    echo_time = 46
    current = np.zeros(steps)
    current[tx_time] = 1.0
    current[echo_time] = 1.0
    freqs = [0.25, 0.39, 0.58]
    labels = ["Low freq", "Matched", "High freq"]
    colors = ["#64748b", "#16a34a", "#64748b"]
    thresholds = [1.35, 1.05, 1.35]
    traces = []
    spikes = []
    for omega, threshold in zip(freqs, thresholds):
        state, _velocity, spike = _resonator_sim(current, omega=omega, decay=0.97, threshold=threshold)
        traces.append(state)
        spikes.append(spike)

    frames: list[Image.Image] = []
    for frame in range(steps):
        fig, axes = plt.subplots(4, 1, figsize=(9, 7), constrained_layout=True)
        fig.patch.set_facecolor("white")
        fig.suptitle("Resonance Bank as Frequency-Selective Timing Analysis", fontsize=17, fontweight="bold")

        axes[0].stem(t[: frame + 1], current[: frame + 1], linefmt="#0f766e", markerfmt=" ", basefmt=" ")
        axes[0].axvline(frame, color="#999999", ls="--", lw=1)
        axes[0].set_xlim(0, steps - 1)
        axes[0].set_ylim(0, 1.15)
        axes[0].set_ylabel("Input")
        axes[0].set_title("Transmit spike excites the bank, echo spike probes resonance")
        _style_axes(axes[0])

        for idx in range(3):
            ax = axes[idx + 1]
            ax.plot(t[: frame + 1], traces[idx][: frame + 1], color=colors[idx], lw=2.5)
            ax.axhline(thresholds[idx], color="#dc2626", ls="--", lw=1.2)
            spike_times = np.where(spikes[idx][: frame + 1] > 0)[0]
            if spike_times.size:
                ax.scatter(spike_times, np.full_like(spike_times, thresholds[idx] + 0.04, dtype=float), color="#dc2626", s=30)
            ax.axvline(frame, color="#999999", ls="--", lw=1)
            ax.set_xlim(0, steps - 1)
            ax.set_ylim(-1.2, 1.55)
            ax.set_ylabel("State")
            ax.set_title(labels[idx])
            _style_axes(ax)

        fig.text(
            0.52,
            0.03,
            r"$v_i[t+1]=\alpha_i v_i[t] + u[t] - \omega_i z_i[t],\quad z_i[t+1]=z_i[t]+\omega_i v_i[t]$",
            ha="center",
            fontsize=12,
        )
        fig.text(
            0.52,
            0.005,
            r"DFT analogy: $X[k]=\sum_t x[t]e^{-j2\pi kt/N}$  (frequency-selective temporal decomposition)",
            ha="center",
            fontsize=11,
        )
        frames.append(_render_frame(fig))
        plt.close(fig)

    _save_gif(frames, path, duration_ms=100)
    return path


def _draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str = "#f8fafc") -> None:
    x, y = xy
    w, h = wh
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor="#1f2937", lw=1.6, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def _draw_arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = "#334155") -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.8, color=color))


def _save_jeffress_diagram() -> Path:
    path = OUTPUT_DIR / "jeffress_delay_line_diagram.png"
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Jeffress-Style Delay-Line Coincidence Detection", fontsize=15, fontweight="bold")

    ax.text(0.4, 3.15, "Transmit / reference spikes", fontsize=11, weight="bold", color="#0f766e")
    ax.text(0.4, 0.55, "Echo / target spikes", fontsize=11, weight="bold", color="#ef4444")
    for y in [3.0, 2.1, 1.2]:
        ax.plot([1.5, 8.0], [y, y], color="#0f766e", lw=2)
    ax.plot([8.0, 8.0], [0.8, 3.2], color="#94a3b8", ls=":", lw=1.6)
    ax.scatter([1.7], [3.0], s=80, color="#0f766e")
    ax.scatter([1.7], [2.1], s=80, color="#0f766e")
    ax.scatter([1.7], [1.2], s=80, color="#0f766e")
    ax.scatter([8.0], [0.9], s=90, color="#ef4444")
    ax.text(2.8, 3.18, r"$\tau_1$", fontsize=12)
    ax.text(4.2, 2.28, r"$\tau_2$", fontsize=12)
    ax.text(5.8, 1.38, r"$\tau_3$", fontsize=12)
    for y, label in zip([3.0, 2.1, 1.2], ["too early", "match", "too late"]):
        circle = Circle((8.9, y), 0.32, edgecolor="#1f2937", facecolor="#dcfce7" if label == "match" else "#ffffff", lw=2)
        ax.add_patch(circle)
        ax.text(8.9, y, "CD", ha="center", va="center", fontsize=10)
        ax.text(9.35, y, label, va="center", fontsize=10)
    _draw_arrow(ax, (8.0, 0.95), (8.55, 2.1), color="#ef4444")
    fig.text(
        0.5,
        0.03,
        r"Interpretation: each neuron tests one delay hypothesis $\tau_i$ by comparing $x[t-\tau_i]$ with $y[t]$.",
        ha="center",
        fontsize=11,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_lso_mntb_diagram() -> Path:
    path = OUTPUT_DIR / "lso_mntb_diagram.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Biological ILD Sketch: LSO and MNTB", fontsize=15, fontweight="bold")

    _draw_box(ax, (0.7, 4.0), (1.8, 1.0), "Left ear spikes", "#ecfeff")
    _draw_box(ax, (0.7, 1.0), (1.8, 1.0), "Right ear spikes", "#fef2f2")
    _draw_box(ax, (4.0, 4.0), (1.7, 1.0), "Left LSO", "#ecfccb")
    _draw_box(ax, (4.0, 1.0), (1.7, 1.0), "Right LSO", "#ecfccb")
    _draw_box(ax, (4.0, 2.4), (1.7, 1.0), "Left MNTB", "#fee2e2")
    _draw_box(ax, (4.0, 0.0), (1.7, 1.0), "Right MNTB", "#fee2e2")
    _draw_box(ax, (8.0, 2.0), (2.1, 1.4), "Opponent compare\n(IC / azimuth readout)", "#ede9fe")

    _draw_arrow(ax, (2.5, 4.5), (4.0, 4.5), color="#16a34a")
    _draw_arrow(ax, (2.5, 1.5), (4.0, 1.5), color="#16a34a")
    _draw_arrow(ax, (2.5, 4.35), (4.0, 2.95), color="#dc2626")
    _draw_arrow(ax, (2.5, 1.35), (4.0, 0.55), color="#dc2626")
    _draw_arrow(ax, (5.7, 2.95), (4.85, 1.95), color="#dc2626")
    _draw_arrow(ax, (5.7, 0.55), (4.85, 4.45), color="#dc2626")
    _draw_arrow(ax, (5.7, 4.5), (8.0, 2.9), color="#16a34a")
    _draw_arrow(ax, (5.7, 1.5), (8.0, 2.5), color="#16a34a")
    ax.text(3.1, 4.82, "ipsi excit.", color="#16a34a", fontsize=10)
    ax.text(2.9, 2.9, "contra via\nMNTB", color="#dc2626", fontsize=10)
    ax.text(6.15, 2.8, "inhib.", color="#dc2626", fontsize=10)
    ax.text(10.5, 2.55, r"$\mathrm{ILD}\ \approx \mathrm{LSO}_R - \mathrm{LSO}_L$", fontsize=11)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_elevation_diagram() -> Path:
    path = OUTPUT_DIR / "elevation_notch_pathway_diagram.png"
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Elevation Pathway: Spectral Notches", fontsize=15, fontweight="bold")

    _draw_box(ax, (0.7, 1.5), (1.8, 1.0), "Echo spectrum", "#eff6ff")
    _draw_box(ax, (3.0, 1.5), (2.0, 1.0), "Slope + moving notch\n(simulator cue)", "#ecfccb")
    _draw_box(ax, (5.7, 1.5), (1.8, 1.0), "Cochlea + spikes", "#f5f3ff")
    _draw_box(ax, (8.1, 1.5), (2.1, 1.0), "Notch detector bank", "#fef3c7")
    _draw_box(ax, (10.8, 1.5), (1.8, 1.0), "Elevation latent", "#fee2e2")
    for x0, x1 in [(2.5, 3.0), (5.0, 5.7), (7.5, 8.1), (10.2, 10.8)]:
        _draw_arrow(ax, (x0, 2.0), (x1, 2.0))
    ax.text(8.55, 0.8, "Responses strongest when notch\nposition matches detector centre", fontsize=10, ha="center")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_full_pipeline_diagram() -> Path:
    path = OUTPUT_DIR / "bat_brain_pipeline_diagram.png"
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Full Bat-Inspired Localization Pipeline", fontsize=16, fontweight="bold")

    _draw_box(ax, (0.5, 3.0), (1.7, 1.0), "Transmit chirp", "#ecfeff")
    _draw_box(ax, (2.8, 3.0), (2.1, 1.0), "Echo physics\n(delay, attenuation,\nITD/ILD, elevation cue)", "#fef3c7")
    _draw_box(ax, (5.6, 3.0), (1.9, 1.0), "Cochlea", "#ede9fe")
    _draw_box(ax, (8.2, 3.0), (1.9, 1.0), "Spike encoding", "#fce7f3")
    _draw_box(ax, (10.9, 5.0), (2.0, 0.9), "Distance pathway", "#dbeafe")
    _draw_box(ax, (10.9, 3.0), (2.0, 0.9), "Azimuth pathway", "#dcfce7")
    _draw_box(ax, (10.9, 1.0), (2.0, 0.9), "Elevation pathway", "#fee2e2")
    _draw_box(ax, (13.8, 3.0), (1.8, 1.0), "Fusion SNN", "#f3e8ff")
    _draw_box(ax, (13.8, 1.2), (1.8, 1.0), "Readout", "#fef2f2")

    _draw_arrow(ax, (2.2, 3.5), (2.8, 3.5))
    _draw_arrow(ax, (4.9, 3.5), (5.6, 3.5))
    _draw_arrow(ax, (7.5, 3.5), (8.2, 3.5))
    _draw_arrow(ax, (10.1, 3.5), (10.9, 5.45))
    _draw_arrow(ax, (10.1, 3.5), (10.9, 3.45))
    _draw_arrow(ax, (10.1, 3.5), (10.9, 1.45))
    _draw_arrow(ax, (12.9, 5.45), (13.8, 3.75))
    _draw_arrow(ax, (12.9, 3.45), (13.8, 3.55))
    _draw_arrow(ax, (12.9, 1.45), (13.8, 3.35))
    _draw_arrow(ax, (14.7, 3.0), (14.7, 2.2))

    ax.text(8.55, 5.7, "distance: coincidence", fontsize=10)
    ax.text(8.45, 5.15, "azimuth: ITD + ILD", fontsize=10)
    ax.text(8.35, 4.6, "elevation: spectral notches", fontsize=10)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_milestones_plot() -> Path:
    path = OUTPUT_DIR / "milestones_summary.png"
    labels = [
        "Matched-human\nbaseline",
        "140 dB\nunnormalized",
        "Round 3\nbest (2B+3)",
        "Round 4\nbest ILD",
        "Round 4\ncombined",
        "Expanded 20 m\n140 dB",
    ]
    combined = np.array([0.1221, 0.0522, 0.0394, 0.0407, 0.0435, 0.2344])
    euclidean = np.array([0.3964, 0.1459, 0.2043, 0.2211, 0.2264, 3.2815])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))

    axes[0].bar(x, combined, color=["#94a3b8", "#16a34a", "#0ea5e9", "#7c3aed", "#f59e0b", "#ef4444"])
    axes[0].set_title("Combined Error")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Error")
    _style_axes(axes[0])

    axes[1].bar(x, euclidean, color=["#94a3b8", "#16a34a", "#0ea5e9", "#7c3aed", "#f59e0b", "#ef4444"])
    axes[1].set_title("Euclidean Localization Error (m)")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Metres")
    _style_axes(axes[1])

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_markdown(assets: dict[str, Path]) -> None:
    text = f"""# Presentation Draft

This is a first-pass presentation draft for a non-specialist audience. It is structured slide-by-slide, with suggested visuals and a short speaking goal for each slide.

## Slide 1: Title

**Title:** Building a Bat-Inspired Spiking Neural Network for 3D Sound Localisation

- Goal: state the problem in one sentence.
- Key message: the project uses bat-like sensing and biologically inspired spiking computation to estimate distance, azimuth, and elevation from echoes.

Visual:
- ![Full pipeline]({assets["pipeline"].relative_to(ROOT / "outputs")})

## Slide 2: Why Spiking Neural Networks?

- Standard neural networks are good at static pattern recognition.
- Echolocation is event-driven, time-critical, and naturally sparse.
- Spiking neural networks give a natural language for:
  - timing
  - coincidence
  - oscillation / resonance
  - biological interpretability

Suggested points:
- spikes represent events rather than continuous activations
- delays and synchrony are central to localization
- the architecture can be interpreted in neural terms rather than just as a black-box regressor

## Slide 3: What A Spike-Based Neuron Does

### LIF neuron

- integrates input current over time
- leaks back toward rest
- spikes when threshold is crossed

Visual:
- ![LIF animation]({assets["lif"].relative_to(ROOT / "outputs")})

Equation:

```text
m[t+1] = beta * m[t] + I[t] - theta * s[t]
s[t]   = H(m[t] - theta)
```

Speaking note:
- this is the simplest timing-sensitive building block in the project

## Slide 4: Resonant Neuron

- a resonant neuron does not just integrate
- it prefers certain temporal rhythms
- this makes it useful for echo timing and frequency-selective temporal structure

Visual:
- ![Resonant animation]({assets["resonant"].relative_to(ROOT / "outputs")})

Equation:

```text
v[t+1] = alpha * v[t] + u[t] - omega * z[t]
z[t+1] = z[t] + omega * v[t]
```

Speaking note:
- this is roughly RLC-like in spirit: damped, oscillatory, and frequency-tuned

## Slide 5: From Algorithms To Circuits

This is the key bridge in the talk: show that familiar algorithms can be reinterpreted as neuronal structures.

- coincidence bank -> delay-swept cross-correlation
- resonance bank -> frequency-selective temporal decomposition
- LSO/MNTB opponent coding -> signed binaural level comparison
- spike summing -> intensity / range cue

## Slide 6: Coincidence Detection Animation

Key message:
- distance and ITD estimation can be framed as testing multiple delay hypotheses in parallel

Visual:
- ![Coincidence animation]({assets["coincidence"].relative_to(ROOT / "outputs")})
- ![Jeffress diagram]({assets["jeffress"].relative_to(ROOT / "outputs")})

Equations:

```text
m_i[t+1] = beta_i * m_i[t] + w_tx * x[t-tau_i] + w_echo * y[t] - theta * s_i[t]
c[tau]   = sum_t x[t-tau] * y[t]
```

Speaking note:
- each tuned neuron tests one delay hypothesis
- the best-matching delay is the one that receives transmit and echo input together

## Slide 7: Resonance Bank Animation

Key message:
- a bank of resonators is not a literal DFT, but it behaves like a frequency-selective decomposition of temporal structure

Visual:
- ![Resonance bank animation]({assets["resonance_bank"].relative_to(ROOT / "outputs")})

Equation link:

```text
resonator bank: state evolves with tuned omega_i
DFT analogy: X[k] = sum_t x[t] * exp(-j 2 pi k t / N)
```

Suggested wording:
- "This is Fourier-like rather than exactly Fourier."

## Slide 8: Biological Azimuth Coding: LSO / MNTB

Key message:
- azimuth is not only about timing
- level differences between ears can also be coded by opponent circuits

Visual:
- ![LSO MNTB diagram]({assets["lso_mntb"].relative_to(ROOT / "outputs")})

Talking points:
- ipsilateral excitation
- contralateral inhibition via MNTB
- opponent comparison sharpens lateralization cues

## Slide 9: Elevation As Spectral Pattern Analysis

Key message:
- elevation is fundamentally different from distance and ITD
- it depends on spectral shaping rather than coincidence timing

Visual:
- ![Elevation notch pathway]({assets["elevation"].relative_to(ROOT / "outputs")})
- ![Moving notch cue](round_3_experiments/round3_experiment_2b_moving_notch_plus_detectors/moving_notch_cue.png)

Talking points:
- early elevation cue was a simple slope
- adding a moving notch made the cue richer
- explicit notch detectors improved elevation further

## Slide 10: The Life Of A Sound Signal

Walk the audience through one sound end-to-end:

1. transmit chirp
2. echo simulator adds delay, attenuation, azimuth asymmetry, and elevation cue
3. cochlea filterbank converts waveform into channel activity
4. spike encoder converts activity into spikes
5. distance / azimuth / elevation pathways compute different cue families
6. fusion SNN combines them into a 3D estimate

Visuals:
- ![Matched-human cochleagram and spikes](cochlea_explained/human_matched_cochleagram_spikes.png)
- ![Current pipeline diagram]({assets["pipeline"].relative_to(ROOT / "outputs")})

## Slide 11: Building A Bat Brain

Use this slide to map the computational parts to biological interpretations:

- cochlea -> peripheral filtering
- delay-line coincidence bank -> Jeffress-like timing circuit
- ILD opponent coding -> LSO/MNTB-like binaural comparison
- spectral notch detectors -> elevation / pinna-like cue decoding
- resonance bank -> tuned temporal feature extraction
- fusion SNN -> higher integration area

## Slide 12: What Actually Worked

Key experimental findings:

- the front end mattered more than expected
- unnormalized high-amplitude spikes recovered long-range behavior
- richer elevation cues helped a lot
- sine/cosine angle outputs stabilized angle regression
- biologically inspired ILD improved overall performance
- adding trainability helped, but stacking everything did not always help

Visual:
- ![Milestones summary]({assets["milestones"].relative_to(ROOT / "outputs")})

Numbers to cite:
- matched-human baseline combined error: `0.1221`
- 140 dB unnormalized short-range combined error: `0.0522`
- best round-3 combined model (`2B + 3`): combined error `0.0394`
- best round-4 individual model (LSO/MNTB ILD): combined error `0.0407`

## Slide 13: A Useful Failure Story

This is worth including because it shows real scientific debugging.

Observation:
- expanded-space tests initially collapsed badly

Diagnosis:
- per-sample envelope normalization made the front end almost level-invariant
- weak long-range returns were being renormalized upward
- that destroyed amplitude information and could create noisy spike patterns

Fix:
- use a much stronger source level (`140 dB` under the current convention)
- disable the front-end normalization

Result:
- expanded 20 m test improved from combined error `0.6315` to `0.2344`

Useful visuals:
- [Direct-drive spike count vs level](cochlea_explained/direct_drive_gain_sweep_700_spike_count_vs_level.png)
- [140 dB unnormalized cochleagram](cochlea_explained/human_matched_140db_unnormalized_cochleagram_spikes.png)

## Slide 14: Current Best Model

Recommended model to present as the current best overall story:

- Round 3 combined model `2B + 3`
  - moving-notch elevation cue + notch detectors
  - sine/cosine angle regression

If you want the most biologically trainable later variant:
- Round 4 combined model
  - explicit LIF timing replacement
  - LSO/MNTB ILD system
  - spike-sum distance cue
  - per-pathway resonance banks

Suggested message:
- the best pure accuracy and the best biological decomposition are not always exactly the same model

## Slide 15: Next Steps

### Improving the bat model
- better long-range amplitude calibration
- more realistic elevation cues / HRTF-like filtering
- cleaner separation of azimuth and elevation spectral codes
- larger cached datasets with the fixed cochlea front end

### Generalising the model
- replace hand-designed cue modules with learnable but constrained spiking modules
- test broader spatial domains
- test other sensing tasks where timing matters

## Slide 16: Conclusion

Suggested closing message:

- SNNs were useful here not just as a fashionable model class, but because the task itself is about timing, coincidence, oscillation, and sparse events.
- The work showed that biologically inspired structure can genuinely help localization.
- The strongest improvements came from understanding the sensory front end and cue design, not just adding depth.

## Suggested Backup Slides

- exhaustive experiment table: [experiments_summary.md](experiments_summary.md)
- cochlea walkthrough: [cochlea_explained.md](cochlea_explained.md)
- current system explanation: [current_system_explained.md](current_system_explained.md)
- round 3 results: [round_3_experiments_report.md](round_3_experiments_report.md)
- round 4 results: [round_4_experiments_report.md](round_4_experiments_report.md)
"""
    MARKDOWN_PATH.write_text(text)


def main() -> None:
    _prepare_output_dir()
    assets = {
        "lif": _save_lif_animation(),
        "resonant": _save_resonant_animation(),
        "coincidence": _save_coincidence_animation(),
        "resonance_bank": _save_resonance_bank_animation(),
        "jeffress": _save_jeffress_diagram(),
        "lso_mntb": _save_lso_mntb_diagram(),
        "elevation": _save_elevation_diagram(),
        "pipeline": _save_full_pipeline_diagram(),
        "milestones": _save_milestones_plot(),
    }
    _write_markdown(assets)

