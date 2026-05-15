# Distance Pathway 2: Accuracy And Optimisation Testing

This report compares LIF, RF, and binary delay-line coincidence detectors under four pulse conditions: clean, added noise, added jitter, and noise plus jitter.

## Signal Conditions

| Condition | True echo jitter | Spurious onset noise |
|---|---:|---:|
| Clean perfect | `False` | `False` |
| Added noise | `False` | `True` |
| Added jitter | `True` | `False` |
| Noise + jitter | `True` | `True` |

Noise here means extra spurious onset pulses in the echo window. Jitter means Gaussian timing jitter on the true echo pulse. This is still a simplified pulse model, not full waveform noise.

## Detector Equations

For all detectors, the mismatch between each observed pulse and candidate delay is:

```text
delta_p,k = abs(delay_observed[p] - delay_candidate[k])
```

The LIF and RF detectors score all observed pulses and use the strongest response. The binary detector checks whether any pulse lands inside a small timing window.

```text
LIF:    score_k = max_p amplitude_p * w * (1 + beta^delta_p,k)
RF:     score_k = max_p amplitude_p * w * (1 + exp(-delta_p,k/tau_rf) * cos(omega_rf * delta_p,k))
Binary: match_k = any_p(delta_p,k <= tolerance)
```

## Benchmark Setup

| Parameter | Value |
|---|---:|
| sample rate | `64000 Hz` |
| speed of sound | `343.0 m/s` |
| distance range | `0.25 -> 5.0 m` |
| test samples per condition | `1000` |
| delay lines | `160` |
| jitter std | `35.0 us` |
| noise pulses | `3` |
| noise amplitude range | `0.25 -> 1.1` |

## Accuracy Across Conditions

![Condition MAE](../outputs/accuracy_optimisation/figures/condition_mae.png)

The detailed numeric results are:

| Condition | Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean perfect | LIF detector | 0.77 | 0.89 | 1.44 | 1.60 | 1.076 | 1,280,000 | 320,000 |
| Clean perfect | RF detector | 2.20 | 3.06 | 4.98 | 5.13 | 1.300 | 2,240,000 | 320,000 |
| Clean perfect | Binary detector | 0.77 | 0.89 | 1.44 | 1.60 | 0.566 | 160,000 | 160,000 |
| Added noise | LIF detector | 46.71 | 102.98 | 258.37 | 452.73 | 5.553 | 5,120,000 | 1,280,000 |
| Added noise | RF detector | 59.60 | 115.94 | 279.53 | 452.73 | 8.145 | 8,960,000 | 1,280,000 |
| Added noise | Binary detector | 91.73 | 146.00 | 335.34 | 464.92 | 2.392 | 640,000 | 640,000 |
| Added jitter | LIF detector | 0.88 | 1.06 | 1.93 | 3.10 | 1.155 | 1,280,000 | 320,000 |
| Added jitter | RF detector | 2.22 | 3.03 | 5.40 | 6.39 | 1.322 | 2,240,000 | 320,000 |
| Added jitter | Binary detector | 0.88 | 1.06 | 1.93 | 3.10 | 0.672 | 160,000 | 160,000 |
| Noise + jitter | LIF detector | 41.63 | 94.94 | 251.83 | 429.46 | 5.681 | 5,120,000 | 1,280,000 |
| Noise + jitter | RF detector | 56.12 | 112.86 | 293.48 | 429.46 | 8.272 | 8,960,000 | 1,280,000 |
| Noise + jitter | Binary detector | 96.03 | 150.21 | 327.56 | 429.46 | 2.254 | 640,000 | 640,000 |

## Hardest-Condition Plots

The scatter, histogram, and cost plots below use the hardest condition, `Noise + jitter`.

![Accuracy scatter](../outputs/accuracy_optimisation/figures/accuracy_scatter_noise_jitter.png)

![Error histogram](../outputs/accuracy_optimisation/figures/error_histogram_noise_jitter.png)

![Cost comparison](../outputs/accuracy_optimisation/figures/cost_comparison.png)

## Interpretation

- Clean perfect signals are essentially a delay quantisation problem, so LIF and binary should be close.
- Jitter tests timing tolerance. LIF remains a useful soft detector because the membrane trace decays smoothly with timing mismatch.
- Noise tests false-onset robustness. Binary is cheap, but can be fooled if a strong false onset lands near another candidate delay.
- RF remains biologically interesting, but its oscillatory side lobes are a weakness for this specific pure-delay task.
- These results still assume onset pulses have already been extracted; the next hard problem is robust onset extraction from real cochlear spike rasters.

## Generated Files

- `condition_mae`: `distance_pathway/outputs/accuracy_optimisation/figures/condition_mae.png`
- `accuracy_scatter_noise_jitter`: `distance_pathway/outputs/accuracy_optimisation/figures/accuracy_scatter_noise_jitter.png`
- `error_histogram_noise_jitter`: `distance_pathway/outputs/accuracy_optimisation/figures/error_histogram_noise_jitter.png`
- `cost_comparison`: `distance_pathway/outputs/accuracy_optimisation/figures/cost_comparison.png`
- `results`: `distance_pathway/outputs/distance_pathway_results.json`

Runtime: `6.03 s`.
