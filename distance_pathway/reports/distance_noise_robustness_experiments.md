# Distance Noise Robustness Experiments

This report tests staged fixes for the distance-pathway noise failure observed in `distance_noise_diagnostics.md`.

## Noise Condition

- Noisy test: `10.0 dB` SNR additive white receiver noise plus `jitter_std = 0.00025 s`.
- Realised noisy config `noise_std`: `2.96442`.
- Distances: `0.25 -> 5.0 m`, `80` samples.

## Tested Mechanisms

- Cochlea tuning: spike threshold `x16`, cochlear LIF beta `0.5`.
- Dynamic cochlea tuning: threshold `x16 -> x2.5`, beta `0.20 -> 0.60`.
- VCN consensus: local window `±2` channels and `±8` samples, requiring at least `3` local events.
- IC facilitation: soft local sweep-consistency gain `0.45`, tau `8.0` samples.
- VCN frequency mask: channels below `4.0 kHz` are silenced before VCN detection and after VCN output, because the call-relevant sweep does not use them.

## Results

| Variant | VCN input | Cochlea tuning | VCN detector | IC mode | Clean MAE | Noisy MAE | Noisy RMSE | Noisy max error | Note |
|---|---|---|---|---|---:|---:|---:|---:|---|
| Cochleagram VCN baseline | `cochleagram` | `default` | `first` | `plain` | `0.342 cm` | `110.203 cm` | `132.685 cm` | `359.425 cm` | Current high-accuracy clean model; VCN reads cochleagram. |
| Spike-raster VCN baseline | `spikes` | `default` | `first` | `plain` | `2.154 cm` | `114.850 cm` | `130.619 cm` | `233.981 cm` | Strictly spike-raster VCN input. |
| Spike VCN + cochlea tuning | `spikes` | `x16, beta=0.5` | `first` | `plain` | `4.960 cm` | `5.239 cm` | `6.585 cm` | `23.532 cm` | Uses 16x cochlear threshold and beta=0.5. |
| Spike VCN + dynamic cochlea | `spikes` | `dynamic x16->x2.5, beta=0.20->0.60` | `first` | `plain` | `3.554 cm` | `6.550 cm` | `8.488 cm` | `23.860 cm` | Uses dynamic cochlear LIF: threshold x16->x2.5, beta 0.20->0.60. |
| Spike VCN + tuning + VCN consensus | `spikes` | `x16, beta=0.5` | `consensus` | `plain` | `4.977 cm` | `5.239 cm` | `6.566 cm` | `23.415 cm` | Adds local multi-channel coincidence in VCN. |
| Spike VCN + tuning + consensus + IC facilitation | `spikes` | `x16, beta=0.5` | `consensus` | `facilitated` | `5.010 cm` | `5.251 cm` | `6.576 cm` | `23.407 cm` | Adds soft sweep-consistency facilitation in IC. |
| Spike VCN + dynamic cochlea + consensus + IC facilitation | `spikes` | `dynamic x16->x2.5, beta=0.20->0.60` | `consensus` | `facilitated` | `3.571 cm` | `5.626 cm` | `7.215 cm` | `23.050 cm` | Tests the selected dynamic cochlea schedule inside the robust pathway. |
| Cochleagram VCN + consensus + IC facilitation | `cochleagram` | `default` | `consensus` | `facilitated` | `0.340 cm` | `117.381 cm` | `164.196 cm` | `402.419 cm` | Tests whether the same pathway-level fixes work without spike-raster input. |

## Main Findings

- Best clean result: `Cochleagram VCN + consensus + IC facilitation` with MAE `0.340 cm`.
- Best noisy result: `Spike VCN + cochlea tuning` with MAE `5.239 cm`.
- The main robustness gain came from cochlea tuning on the spike-raster pathway: threshold `x16` and beta `0.5` reduced noisy MAE from metre-scale failure to approximately `5 cm`.
- The dynamic cochlea variants test the selected diagnostic setting: threshold `x16 -> x2.5`, beta `0.20 -> 0.60`.
- VCN consensus and IC facilitation did not materially improve this first tuned spike-raster result. They may need retuning now that the cochlear input is much sparser.
- The cochleagram pathway remains excellent clean, but it is still highly noise-sensitive because it reads continuous low-threshold activity before the cochlear spike encoder.

## Recalibrated Robust Latency Vector

The selected robust model is `Spike VCN + tuning + consensus + IC facilitation`, with VCN channels below `4 kHz` silenced before detection. Its latency vector is recalibrated after applying the spike-raster cochlea tuning, VCN consensus, and frequency mask.

| Calibration property | Value |
|---|---:|
| responsive channels | `33` |
| silenced channels below 4 kHz | `15` |
| calibrated responsive channels | `33` |
| missing responsive channels | `0` |
| latency range over responsive channels | `-62 -> 25` samples |
| mean latency std across calibration distances | `31.376` samples |
| max latency std across calibration distances | `53.363` samples |
| saved vector | `distance_pathway/outputs/distance_noise_robustness/spike_tuned_consensus_facil_latency_samples.npy` |

With this recalibrated vector, the selected robust model gives clean MAE `5.010 cm` and noisy MAE `5.251 cm`.

## Dynamic Cochlea Robust Latency Vector

The dynamic robust model uses the selected dynamic cochlea schedule, VCN consensus, IC facilitation, and the same pre-VCN `4 kHz` mask.

| Dynamic calibration property | Value |
|---|---:|
| responsive channels | `33` |
| silenced channels below 4 kHz | `15` |
| calibrated responsive channels | `33` |
| missing responsive channels | `0` |
| latency range over responsive channels | `-43 -> 31` samples |
| mean latency std across calibration distances | `27.884` samples |
| max latency std across calibration distances | `52.137` samples |
| saved vector | `distance_pathway/outputs/distance_noise_robustness/spike_dynamic_consensus_facil_latency_samples.npy` |

With this recalibrated vector, the dynamic robust model gives clean MAE `3.571 cm` and noisy MAE `5.626 cm`.

## Interpretation

- The spike-raster pathway is the relevant path for cochlea threshold/beta tuning, because cochleagram VCN bypasses cochlear spike generation.
- VCN consensus is intended to reject isolated noisy events by requiring local channel agreement.
- IC facilitation is deliberately soft: it boosts sweep-consistent candidate distances but does not hard-gate the response.
- The VCN frequency mask is a biological/engineering assumption that sub-call-band channels should not drive the distance pathway or contribute to consensus counts.
- If a variant improves noisy MAE but destroys clean MAE, it is not yet acceptable as a general distance pathway.
- The dynamic cochlea schedule should be judged on full pathway distance error, not just visual raster cleanliness, because stricter spike cleanup can also remove useful timing events.

## Generated Files

- `results`: `distance_pathway/outputs/distance_noise_robustness/results.json`

Runtime: `45.85 s`.
