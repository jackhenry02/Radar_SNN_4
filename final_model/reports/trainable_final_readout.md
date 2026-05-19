# Trainable Final SNN Readout

This report tests whether the fixed biologically structured pathways can be followed by a small trainable snnTorch readout. The default command is a smoke test; larger runs use the same cache, training, evaluation, and report-writing pipeline.

## Reproducible Setup

| Item | Value |
|---|---:|
| distance range | `0.25-5.0 m` |
| azimuth range | `+/-45.0 deg` |
| elevation range | `+/-45.0 deg` |
| train / val / test samples | `4 / 2 / 2` |
| run label | `train4_val2_test2` |
| dataset seed | `8101` |
| training seed | `8202` |
| cochlear channels | `48` |
| distance bins | `180` |
| angular bins | `91` |
| SNN hidden neurons | `96` |
| SNN timesteps | `12` |
| optimiser | `AdamW`, lr `0.001`, weight decay `1e-05` |
| batch size | `16` |
| epochs | `1` |
| residual scale | `0.15` |

The cached input vector is:

```text
raw distance population
raw azimuth ITD population
raw azimuth ILD population
raw elevation population
CANN distance/azimuth/elevation readouts
confidence features and spike counts
```

The target vector is:

```text
[distance_norm, az_sin, az_cos, el_sin, el_cos]
```

Distance is normalised as `(distance - 0.25) / 4.75`. Angles are trained as sine/cosine pairs to avoid discontinuities.

## Loss Function

The smoke test uses learned uncertainty weighting over three tasks:

$$
L=\frac{L_d}{2\sigma_d^2}+\log\sigma_d+\frac{L_a}{2\sigma_a^2}+\log\sigma_a+\frac{L_e}{2\sigma_e^2}+\log\sigma_e.
$$

Here `Ld` is distance MSE, `La` is azimuth sine/cosine MSE, and `Le` is elevation sine/cosine MSE.

## Smoke-Test Results

| Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error |
|---|---:|---:|---:|---:|---:|
| baseline | `0.0605 m` | `1.170 deg` | `1.755 deg` | `0.1163 m` | `0.0257` |
| residual | `0.0590 m` | `1.223 deg` | `1.543 deg` | `0.1028 m` | `0.0244` |
| direct | `2.1532 m` | `116.815 deg` | `53.735 deg` | `2.6500 m` | `1.4069` |

Mean feature-cache generation time was `1.404 s/sample` for this smoke test.

![Training curves](../outputs/trainable_readout/figures/train4_val2_test2/training_curves.png)

![Prediction scatter](../outputs/trainable_readout/figures/train4_val2_test2/test_prediction_scatter.png)

## Feature Importance

The first diagnostic sums the absolute first-layer weights by feature group. The second zeroes each normalised feature group on the test set and measures the increase in combined error. These are not perfect causal explanations, but they show whether the trained SNN is using the CANN readouts or mostly ignoring them.

![Feature importance](../outputs/trainable_readout/figures/train4_val2_test2/residual_feature_importance.png)

| Feature group | First-layer share | Ablation delta |
|---|---:|---:|
| raw_distance_population | `0.1659` | `-0.0011` |
| raw_azimuth_itd_population | `0.1681` | `0.0000` |
| raw_azimuth_ild_population | `0.1674` | `0.0000` |
| raw_elevation_population | `0.1664` | `0.0000` |
| cann_readouts | `0.1641` | `0.0000` |
| confidence_features | `0.1682` | `0.0000` |

## Biological Interpretation

This is best interpreted as a higher-level contextual integration layer. The sensory pathways still produce structured population codes. The small SNN receives raw cue populations, stabilised CANN readouts, and confidence signals, then learns how to combine them when cues are distorted by distance, azimuth, elevation, and pathway confidence.

The residual variant is especially biologically defensible because it keeps the hand-designed pathway answer as the main estimate and learns a small context-dependent correction.

## Generated Files

- `training_curves`: `final_model/outputs/trainable_readout/figures/train4_val2_test2/training_curves.png`
- `test_prediction_scatter`: `final_model/outputs/trainable_readout/figures/train4_val2_test2/test_prediction_scatter.png`
- `residual_feature_importance`: `final_model/outputs/trainable_readout/figures/train4_val2_test2/residual_feature_importance.png`
- `cache`: `final_model/outputs/trainable_readout/cache_constrained_0p25_5m_pm45_train4_val2_test2.npz`
- `results`: `final_model/outputs/trainable_readout/results_train4_val2_test2.json`