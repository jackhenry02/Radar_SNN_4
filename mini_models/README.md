# Mini Models

This folder is for isolated mini models and ablation studies used to design the next model from first principles.

The old round-based model code and outputs should remain untouched. Mini-model work should live here and generate its own reports, figures, and results.

## Structure

- `common/`: reusable signal, neuron, plotting, timing, and metric utilities.
- `experiments/`: standalone experiment scripts.
- `reports/`: generated markdown reports.
- `outputs/`: generated figures, JSON results, and animations.
- `notebooks/scratch/`: optional disposable notebooks for exploration only.

## Current Experiments

- `experiments/neuron_analysis.py`: micro and macro behaviour of LIF, resonate-and-fire, and level-crossing neurons.
- `experiments/signal_analysis.py`: emitted call, received echoes, binaural attenuation/head shadow, noise/jitter, and elevation spectral notch.
- `experiments/cochlea_analysis.py`: compares original FFT/IFFT cochlea, Conv1D + LIF, Conv1D + level crossing, and RF-bank cochlea candidates.
