# Distance Noise Diagnostics

This report diagnoses why the full distance pathway fails under the harsh noise condition used in the signal-analysis mini model.

Two variants are compared on the same noisy echo:

- Cochleagram VCN: the VCN onset LIF reads the rectified cochleagram.
- Spike-raster VCN: the VCN onset LIF reads the cochlear spike raster.

## Noise Condition

- Distance: `3.00 m`
- Additive white receiver noise: `10.0 dB` SNR over the active echo window
- Propagation jitter: `jitter_std = 0.00025 s`
- Realised `noise_std`: `2.96442`

The goal is not to prove final robustness. The goal is to see where the failure enters the pathway.

## Shared Noisy Cochlea Output

Both VCN variants receive the same noisy simulated echo and the same cochlea front end.

![Noisy cochleagram](../outputs/distance_noise_diagnostics/figures/noisy_cochleagram.png)

![Noisy cochlear spike raster](../outputs/distance_noise_diagnostics/figures/noisy_spike_raster.png)

## VCN Outputs

![VCN outputs](../outputs/distance_noise_diagnostics/figures/noisy_vcn_outputs.png)

![Expected vs VCN](../outputs/distance_noise_diagnostics/figures/expected_vs_vcn.png)

## Timing Summary

| Model | VCN input | Active channels | First global onset | Mean abs timing error | Median timing error | Error range |
|---|---|---:|---:|---:|---:|---:|
| Cochleagram VCN | `cochleagram` | `48` | `0.016 ms` | `1131.0` samples | `-1130.5` samples | `-1142 -> -1116` samples |
| Spike-raster VCN | `spikes` | `48` | `0.062 ms` | `1128.4` samples | `-1128.5` samples | `-1138 -> -1114` samples |

## Cochlear Threshold Sweep

The first attempted fix is to raise the cochlear spike threshold while keeping the same noisy waveform. This tests whether the noisy cochlear spike raster can be cleaned before changing the downstream VCN logic.

![Threshold spike rasters](../outputs/distance_noise_diagnostics/figures/threshold_spike_rasters.png)

![Threshold summary](../outputs/distance_noise_diagnostics/figures/threshold_summary.png)

| Threshold multiplier | Spike threshold | Total cochlear spikes | Active channels | First global spike |
|---:|---:|---:|---:|---:|
| `x1` | `0.42` | `73585` | `48` | `0.062 ms` |
| `x2` | `0.84` | `42909` | `48` | `0.125 ms` |
| `x4` | `1.68` | `22050` | `48` | `0.188 ms` |
| `x8` | `3.36` | `9728` | `48` | `0.375 ms` |
| `x16` | `6.72` | `2950` | `48` | `0.641 ms` |

## Interpretation

- The cochleagram-driven VCN is very sensitive because it uses a low adaptive threshold on continuous cochleagram activity.
- Under strong white noise, early noise energy can cross that low threshold before the real echo onset.
- The spike-raster VCN is slightly more conservative because it waits for the cochlear spike encoder, but it still fails when noise creates false or shifted cochlear spikes.
- The clean 0.32 cm result is therefore a clean-timing result, not yet a robust-noise result.
- Raising the cochlear spike threshold can reduce spike density, but the important question is whether it removes the early false events without deleting the real echo.
- The next fix should be a more robust VCN onset rule, such as multi-channel agreement, matched sweep gating, higher/refractory adaptive thresholds, or a pre-onset denoising/gain-control stage.

## Generated Files

- `noisy_cochleagram`: `distance_pathway/outputs/distance_noise_diagnostics/figures/noisy_cochleagram.png`
- `noisy_spike_raster`: `distance_pathway/outputs/distance_noise_diagnostics/figures/noisy_spike_raster.png`
- `noisy_vcn_outputs`: `distance_pathway/outputs/distance_noise_diagnostics/figures/noisy_vcn_outputs.png`
- `expected_vs_vcn`: `distance_pathway/outputs/distance_noise_diagnostics/figures/expected_vs_vcn.png`
- `threshold_spike_rasters`: `distance_pathway/outputs/distance_noise_diagnostics/figures/threshold_spike_rasters.png`
- `threshold_summary`: `distance_pathway/outputs/distance_noise_diagnostics/figures/threshold_summary.png`
- `results`: `distance_pathway/outputs/distance_noise_diagnostics/results.json`

Runtime: `6.44 s`.