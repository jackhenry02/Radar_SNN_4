# Trainable Final SNN Readout Comparison

This report is regenerated automatically from available `results_*.json` files when `final_model/experiments/trainable_final_readout.py` is run. It allows clean and noisy cached runs to coexist without overwriting each other.

## Summary Table

| Run | Noise | Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error | Results |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `train2000_val400_test400` | `clean` | `baseline` | `0.0366 m` | `4.925 deg` | `0.910 deg` | `0.2435 m` | `0.0457` | [json](../outputs/trainable_readout/results_train2000_val400_test400.json) |
| `train2000_val400_test400` | `clean` | `residual` | `0.0139 m` | `2.619 deg` | `0.267 deg` | `0.1247 m` | `0.0223` | [json](../outputs/trainable_readout/results_train2000_val400_test400.json) |
| `train2000_val400_test400` | `clean` | `direct` | `0.0624 m` | `2.449 deg` | `1.133 deg` | `0.1612 m` | `0.0307` | [json](../outputs/trainable_readout/results_train2000_val400_test400.json) |

## Per-Run Reports

The next full script run will create a run-labelled report, for example:

- `trainable_final_readout_train2000_val400_test400_noise50dB.md`

The original clean report currently remains available as:

- [trainable_final_readout.md](trainable_final_readout.md)
