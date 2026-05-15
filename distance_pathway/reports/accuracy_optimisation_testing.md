# Distance Pathway 2: Accuracy And Optimisation Testing

This report compares LIF, RF, and binary delay-line coincidence detectors under four pulse conditions: clean, added noise, added jitter, and noise plus jitter.

## Signal Conditions

| Condition | True echo jitter | Additive white noise |
|---|---:|---:|
| Clean perfect | `False` | `False` |
| Added noise | `False` | `True` |
| Added jitter | `True` | `False` |
| Noise + jitter | `True` | `True` |

Noise here means additive Gaussian white noise on the synthetic echo waveform, with an approximate SNR of `10.0 dB` relative to the unit echo pulse. The echo itself is a narrow Gaussian pulse with sigma `2.0` samples, so nearby delay lines still see a graded amplitude. Jitter means Gaussian timing jitter on the true echo pulse.

## Detector Equations

For all detectors, the candidate delay lines sample the echo waveform at their expected echo-arrival time:

```text
a_k = max(0, waveform[call_time + delay_candidate[k]])
delta_k = abs(delay_echo - delay_candidate[k])
```

The LIF and RF detectors use the sampled amplitude as their input drive. The binary detector checks whether the candidate sample crosses a fixed amplitude threshold and is close enough in time.

```text
LIF:    score_k = a_k * w * (1 + beta^delta_k)
RF:     score_k = a_k * w * (1 + exp(-delta_k/tau_rf) * cos(omega_rf * delta_k))
Binary: match_k = 1 if waveform[call_time + delay_candidate[k]] >= threshold and delta_k <= tolerance
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
| white noise SNR | `10.0 dB` |
| echo pulse sigma | `2.0 samples` |

## Accuracy Across Conditions

![Condition MAE](../outputs/accuracy_optimisation/figures/condition_mae.png)

The detailed numeric results are:

| Condition | Detector | MAE (cm) | RMSE (cm) | p95 abs error (cm) | max abs error (cm) | runtime (ms) | FLOPs | SOPs / bit ops |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean perfect | LIF detector | 0.77 | 0.89 | 1.44 | 1.60 | 1.906 | 1,280,000 | 320,000 |
| Clean perfect | RF detector | 0.77 | 0.89 | 1.44 | 1.60 | 2.017 | 2,240,000 | 320,000 |
| Clean perfect | Binary detector | 0.77 | 0.89 | 1.44 | 1.60 | 1.023 | 160,000 | 160,000 |
| Added noise | LIF detector | 50.35 | 103.55 | 267.11 | 450.41 | 1.679 | 1,280,000 | 320,000 |
| Added noise | RF detector | 96.90 | 149.54 | 330.10 | 450.41 | 2.041 | 2,240,000 | 320,000 |
| Added noise | Binary detector | 103.44 | 155.65 | 339.15 | 464.02 | 0.936 | 160,000 | 160,000 |
| Added jitter | LIF detector | 0.88 | 1.06 | 1.93 | 3.10 | 1.762 | 1,280,000 | 320,000 |
| Added jitter | RF detector | 0.88 | 1.06 | 1.93 | 3.10 | 2.024 | 2,240,000 | 320,000 |
| Added jitter | Binary detector | 0.88 | 1.06 | 1.93 | 3.10 | 0.984 | 160,000 | 160,000 |
| Noise + jitter | LIF detector | 50.98 | 105.39 | 276.16 | 459.65 | 1.755 | 1,280,000 | 320,000 |
| Noise + jitter | RF detector | 91.64 | 145.93 | 327.44 | 459.65 | 2.082 | 2,240,000 | 320,000 |
| Noise + jitter | Binary detector | 100.78 | 153.51 | 336.27 | 459.65 | 1.055 | 160,000 | 160,000 |

## Hardest-Condition Plots

The scatter, histogram, and cost plots below use the hardest condition, `Noise + jitter`.

![Accuracy scatter](../outputs/accuracy_optimisation/figures/accuracy_scatter_noise_jitter.png)

![Error histogram](../outputs/accuracy_optimisation/figures/error_histogram_noise_jitter.png)

![Cost comparison](../outputs/accuracy_optimisation/figures/cost_comparison.png)

## Interpretation

- Clean perfect signals are essentially a delay quantisation problem, so LIF and binary should be close.
- Jitter tests timing tolerance. LIF remains a useful soft detector because the membrane trace decays smoothly with timing mismatch.
- Noise tests robustness to additive waveform fluctuations. In this simplified setup, delay lines sample the noisy waveform at candidate arrival times.
- RF remains biologically interesting, but its oscillatory side lobes are a weakness for this specific pure-delay task.
- These results still assume onset pulses have already been extracted; the next hard problem is robust onset extraction from real cochlear spike rasters.

## Generated Files

- `condition_mae`: `distance_pathway/outputs/accuracy_optimisation/figures/condition_mae.png`
- `accuracy_scatter_noise_jitter`: `distance_pathway/outputs/accuracy_optimisation/figures/accuracy_scatter_noise_jitter.png`
- `error_histogram_noise_jitter`: `distance_pathway/outputs/accuracy_optimisation/figures/error_histogram_noise_jitter.png`
- `cost_comparison`: `distance_pathway/outputs/accuracy_optimisation/figures/cost_comparison.png`
- `results`: `distance_pathway/outputs/distance_pathway_results.json`

Runtime: `5.35 s`.
