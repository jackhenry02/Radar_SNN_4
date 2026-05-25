# Trainable Input Baseline Comparison

This report aggregates the raw-waveform and cochlear-raster SNN control baselines. These controls use the same target encoding, snnTorch readout class, optimiser, uncertainty-weighted loss, hidden size, and timestep count as the final trainable readout, but remove the crafted distance/azimuth/elevation pathway features.

The projected variants compress the flattened input to the crafted-feature dimension with a fixed non-learned Gaussian projection. The direct variants skip the projection and therefore have larger first-layer parameter counts.

## Primary Full-Run Summary

| Run | Acoustic mode | Noise | Input baseline | Input dim | Params | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error | Results |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train2000_val400_test400` | `clean` | `clean` | `raw_waveform_projected` | `474` | `55,597` | `0.2954 m` | `21.160 deg` | `20.221 deg` | `1.4882 m` | `0.3262` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400.json) |
| `train2000_val400_test400` | `clean` | `clean` | `raw_waveform_direct` | `4608` | `452,461` | `0.8053 m` | `21.580 deg` | `22.855 deg` | `1.8587 m` | `0.3828` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400.json) |
| `train2000_val400_test400` | `clean` | `clean` | `cochlear_raster_projected` | `474` | `55,597` | `0.0677 m` | `4.292 deg` | `6.282 deg` | `0.2580 m` | `0.0828` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400.json) |
| `train2000_val400_test400` | `clean` | `clean` | `cochlear_raster_direct` | `960` | `102,253` | `0.0651 m` | `6.033 deg` | `7.178 deg` | `0.3873 m` | `0.1022` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400.json) |
| `train2000_val400_test400_envnoise50dB` | `environment_noise` | `envnoise50dB` | `raw_waveform_projected` | `474` | `55,597` | `0.4113 m` | `23.106 deg` | `22.667 deg` | `1.7410 m` | `0.3665` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB.json) |
| `train2000_val400_test400_envnoise50dB` | `environment_noise` | `envnoise50dB` | `raw_waveform_direct` | `4608` | `452,461` | `0.9794 m` | `23.847 deg` | `23.836 deg` | `2.0059 m` | `0.4185` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB.json) |
| `train2000_val400_test400_envnoise50dB` | `environment_noise` | `envnoise50dB` | `cochlear_raster_projected` | `474` | `55,597` | `0.0733 m` | `6.235 deg` | `4.987 deg` | `0.2977 m` | `0.0880` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB.json) |
| `train2000_val400_test400_envnoise50dB` | `environment_noise` | `envnoise50dB` | `cochlear_raster_direct` | `960` | `102,253` | `0.0699 m` | `5.991 deg` | `4.678 deg` | `0.2830 m` | `0.0837` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB.json) |
| `train2000_val400_test400_envnoise50dB_reverb` | `environment_noise_reverb` | `envnoise50dB_reverb` | `raw_waveform_projected` | `474` | `55,597` | `0.3945 m` | `22.349 deg` | `22.307 deg` | `1.7023 m` | `0.3571` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB_reverb.json) |
| `train2000_val400_test400_envnoise50dB_reverb` | `environment_noise_reverb` | `envnoise50dB_reverb` | `raw_waveform_direct` | `4608` | `452,461` | `0.9414 m` | `21.990 deg` | `22.471 deg` | `1.9033 m` | `0.3921` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB_reverb.json) |
| `train2000_val400_test400_envnoise50dB_reverb` | `environment_noise_reverb` | `envnoise50dB_reverb` | `cochlear_raster_projected` | `474` | `55,597` | `0.0772 m` | `5.415 deg` | `3.811 deg` | `0.3001 m` | `0.0735` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB_reverb.json) |
| `train2000_val400_test400_envnoise50dB_reverb` | `environment_noise_reverb` | `envnoise50dB_reverb` | `cochlear_raster_direct` | `960` | `102,253` | `0.0632 m` | `5.900 deg` | `4.704 deg` | `0.2685 m` | `0.0828` | [json](../outputs/trainable_input_baselines/results_train2000_val400_test400_envnoise50dB_reverb.json) |

## Interpretation

- In the clean full run, the best input-only baseline is `cochlear_raster_projected` with combined error `0.0828`.
- In environmental noise, the best input-only baseline is `cochlear_raster_direct` with combined error `0.0837`.
- With environmental noise plus late echoes/reverb, the best input-only baseline is `cochlear_raster_projected` with combined error `0.0735`.
- Compare these rows against `trainable_final_readout_comparison.md`. The crafted-feature residual SNN keeps the same small readout but starts from structured pathway populations, CANN readouts, and confidence features rather than raw waveform/raster samples.
- The direct waveform baseline has many more trainable parameters because its first linear layer sees the full flattened waveform. If it does not outperform the projected/crafted versions, that supports the value of structured feature extraction rather than simply increasing input dimensionality.

## Per-Run Reports

- `train2000_val400_test400`: [trainable_input_baselines_train2000_val400_test400.md](trainable_input_baselines_train2000_val400_test400.md)
- `train2000_val400_test400_envnoise50dB`: [trainable_input_baselines_train2000_val400_test400_envnoise50dB.md](trainable_input_baselines_train2000_val400_test400_envnoise50dB.md)
- `train2000_val400_test400_envnoise50dB_reverb`: [trainable_input_baselines_train2000_val400_test400_envnoise50dB_reverb.md](trainable_input_baselines_train2000_val400_test400_envnoise50dB_reverb.md)
- `train2_val1_test1`: [trainable_input_baselines_train2_val1_test1.md](trainable_input_baselines_train2_val1_test1.md)

## Smoke-Test Runs

These rows are retained for reproducibility only. They used fewer than 100 total samples and should not be used for model comparison.

| Run | Acoustic mode | Noise | Input baseline | Input dim | Params | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error | Results |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train2_val1_test1` | `clean` | `clean` | `raw_waveform_projected` | `474` | `55,597` | `1.0017 m` | `26.807 deg` | `48.505 deg` | `1.1262 m` | `0.6246` | [json](../outputs/trainable_input_baselines/results_train2_val1_test1.json) |
| `train2_val1_test1` | `clean` | `clean` | `raw_waveform_direct` | `4608` | `452,461` | `0.3806 m` | `169.943 deg` | `116.163 deg` | `2.0986 m` | `2.1447` | [json](../outputs/trainable_input_baselines/results_train2_val1_test1.json) |
| `train2_val1_test1` | `clean` | `clean` | `cochlear_raster_projected` | `474` | `55,597` | `0.5053 m` | `137.850 deg` | `64.381 deg` | `2.7399 m` | `1.5317` | [json](../outputs/trainable_input_baselines/results_train2_val1_test1.json) |
| `train2_val1_test1` | `clean` | `clean` | `cochlear_raster_direct` | `960` | `102,253` | `1.0017 m` | `85.650 deg` | `33.939 deg` | `1.2182 m` | `0.9526` | [json](../outputs/trainable_input_baselines/results_train2_val1_test1.json) |