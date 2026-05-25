# Trainable Final SNN Readout

This report tests whether the fixed biologically structured pathways can be followed by a small trainable snnTorch readout. The default command is a smoke test; larger runs use the same cache, training, evaluation, and report-writing pipeline.

## Reproducible Setup

| Item | Value |
|---|---:|
| distance range | `0.25-5.0 m` |
| azimuth range | `+/-45.0 deg` |
| elevation range | `+/-45.0 deg` |
| acoustic mode | `environment_noise` |
| noise label | `envnoise50dB` |
| receiver noise std | `0` |
| environmental noise | `50.0 dB`, reverb `False` |
| train / val / test samples | `2000 / 400 / 400` |
| run label | `train2000_val400_test400_envnoise50dB` |
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
| raw | `0.0786 m` | `7.691 deg` | `29.980 deg` | `1.7054 m` | `0.2843` |
| baseline | `0.0789 m` | `8.094 deg` | `26.642 deg` | `1.5772 m` | `0.2626` |
| residual | `0.0213 m` | `3.705 deg` | `3.505 deg` | `0.2968 m` | `0.0548` |
| direct | `0.0690 m` | `3.831 deg` | `4.018 deg` | `0.3214 m` | `0.0627` |

Mean feature-cache generation time was `0.529 s/sample` for this run.

![Training curves](../outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/training_curves.png)

![Prediction scatter](../outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/test_prediction_scatter.png)

The following diagnostic isolates whether the fixed readout collapse is already present in the raw pathway populations or is introduced by the CANN stage.

![Raw vs baseline scatter](../outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/raw_vs_baseline_scatter.png)

## Feature Importance

The first diagnostic sums the absolute first-layer weights by feature group. The second zeroes each normalised feature group on the test set and measures the increase in combined error. These are not perfect causal explanations, but they show whether the trained SNN is using the CANN readouts or mostly ignoring them.

![Feature importance](../outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/residual_feature_importance.png)

| Feature group | First-layer share | Ablation delta |
|---|---:|---:|
| raw_distance_population | `0.1432` | `0.0471` |
| raw_azimuth_itd_population | `0.1114` | `0.0215` |
| raw_azimuth_ild_population | `0.1113` | `0.0252` |
| raw_elevation_population | `0.2041` | `0.1102` |
| cann_readouts | `0.1807` | `0.0003` |
| confidence_features | `0.2493` | `0.0200` |

## Biological Interpretation

This is best interpreted as a higher-level contextual integration layer. The sensory pathways still produce structured population codes. The small SNN receives raw cue populations, stabilised CANN readouts, and confidence signals, then learns how to combine them when cues are distorted by distance, azimuth, elevation, and pathway confidence.

The residual variant is especially biologically defensible because it keeps the hand-designed pathway answer as the main estimate and learns a small context-dependent correction.

## Generated Files

- `raw_vs_baseline_scatter`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/raw_vs_baseline_scatter.png`

- `training_curves`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/training_curves.png`
- `test_prediction_scatter`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/test_prediction_scatter.png`
- `residual_feature_importance`: `final_model/outputs/trainable_readout/figures/train2000_val400_test400_envnoise50dB/residual_feature_importance.png`
- `cache`: `final_model/outputs/trainable_readout/cache_constrained_0p25_5m_pm45_train2000_val400_test400_envnoise50dB.npz`
- `results`: `final_model/outputs/trainable_readout/results_train2000_val400_test400_envnoise50dB.json`