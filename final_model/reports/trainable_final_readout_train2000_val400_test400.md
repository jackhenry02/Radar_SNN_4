# Trainable Final SNN Readout

This report tests whether the fixed biologically structured pathways can be followed by a small trainable snnTorch readout. The default command is a smoke test; larger runs use the same cache, training, evaluation, and report-writing pipeline.

## Reproducible Setup

| Item | Value |
|---|---:|
| distance range | `0.25-5.0 m` |
| azimuth range | `+/-45.0 deg` |
| elevation range | `+/-45.0 deg` |
| acoustic mode | `clean` |
| noise label | `clean` |
| receiver noise std | `0` |
| environmental noise | `0.0 dB`, reverb `False` |
| train / val / test samples | `2000 / 400 / 400` |
| run label | `train2000_val400_test400` |
| dataset seed | `8101` |
| training seed | `8202` |
| cochlear channels | `48` |
| distance bins | `180` |
| angular bins | `91` |
| SNN hidden neurons | `96` |
| SNN timesteps | `12` |
| optimiser | `AdamW`, lr `0.001`, weight decay `1e-05` |
| batch size | `16` |
| epochs | `80` |
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

The readout uses learned uncertainty weighting over three tasks:

$$
L=\frac{L_d}{2\sigma_d^2}+\log\sigma_d+\frac{L_a}{2\sigma_a^2}+\log\sigma_a+\frac{L_e}{2\sigma_e^2}+\log\sigma_e.
$$

Here `Ld` is distance MSE, `La` is azimuth sine/cosine MSE, and `Le` is elevation sine/cosine MSE.

## Cached Training Results

| Readout | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error |
|---|---:|---:|---:|---:|---:|
| raw | `0.0365 m` | `4.580 deg` | `2.986 deg` | `0.2999 m` | `0.0585` |
| baseline | `0.0366 m` | `4.925 deg` | `0.910 deg` | `0.2435 m` | `0.0457` |
| residual | `0.0139 m` | `2.619 deg` | `0.267 deg` | `0.1247 m` | `0.0223` |
| direct | `0.0624 m` | `2.449 deg` | `1.133 deg` | `0.1612 m` | `0.0307` |

Mean feature-cache generation time was `0.628 s/sample` for this run.

![Training curves](../outputs/trainable_readout/figures/train2000_val400_test400/training_curves.png)

![Prediction scatter](../outputs/trainable_readout/figures/train2000_val400_test400/test_prediction_scatter.png)

The following diagnostic isolates whether the fixed readout collapse is already present in the raw pathway populations or is introduced by the CANN stage.

![Raw vs baseline scatter](../outputs/trainable_readout/figures/train2000_val400_test400/raw_vs_baseline_scatter.png)

## Feature Importance

The first diagnostic sums the absolute first-layer weights by feature group. The second zeroes each normalised feature group on the test set and measures the increase in combined error. These are not perfect causal explanations, but they show whether the trained SNN is using the CANN readouts or mostly ignoring them.

### Residual SNN

![Residual feature importance](../outputs/trainable_readout/figures/train2000_val400_test400/residual_feature_importance.png)

| Feature group | First-layer share | Ablation delta |
|---|---:|---:|
| raw_distance_population | `0.1602` | `0.0040` |
| raw_azimuth_itd_population | `0.1451` | `0.0115` |
| raw_azimuth_ild_population | `0.1493` | `0.0107` |
| raw_elevation_population | `0.1768` | `0.0053` |
| cann_readouts | `0.1617` | `0.0000` |
| confidence_features | `0.2067` | `0.0011` |

### Direct SNN

![Direct feature importance](../outputs/trainable_readout/figures/train2000_val400_test400/direct_feature_importance.png)

| Feature group | First-layer share | Ablation delta |
|---|---:|---:|
| raw_distance_population | `0.1564` | `0.0356` |
| raw_azimuth_itd_population | `0.1183` | `0.0242` |
| raw_azimuth_ild_population | `0.1334` | `0.0239` |
| raw_elevation_population | `0.1806` | `0.1139` |
| cann_readouts | `0.1923` | `-0.0001` |
| confidence_features | `0.2190` | `0.0051` |

## Biological Interpretation

This is best interpreted as a higher-level contextual integration layer. The sensory pathways still produce structured population codes. The small SNN receives raw cue populations, stabilised CANN readouts, and confidence signals, then learns how to combine them when cues are distorted by distance, azimuth, elevation, and pathway confidence.

The residual variant is especially biologically defensible because it keeps the hand-designed pathway answer as the main estimate and learns a small context-dependent correction.

## Generated Files

- `training_curves`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400/training_curves.png`
- `test_prediction_scatter`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400/test_prediction_scatter.png`
- `raw_vs_baseline_scatter`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400/raw_vs_baseline_scatter.png`
- `residual_feature_importance`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400/residual_feature_importance.png`
- `direct_feature_importance`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400/direct_feature_importance.png`
- `test_predictions`: `final_model/outputs/trainable_readout/test_predictions_train2000_val400_test400.npz`
- `cache`: `final_model/outputs/trainable_readout/cache_constrained_0p25_5m_pm45_train2000_val400_test400.npz`
- `results`: `final_model/outputs/trainable_readout/results_train2000_val400_test400.json`