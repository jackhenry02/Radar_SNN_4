# Other Training Tests

This report records small alternative training tests for the final SNN fusion readout. These tests reuse the cached pathway features and therefore do not regenerate the acoustic simulation or pathway outputs.

## Normalised Angle Target Test

The main trainable readout predicts `[distance_norm, az_sin, az_cos, el_sin, el_cos]`. This test keeps the same cached feature vector and the same small snnTorch readout architecture, but changes the target to `[distance_norm, az_norm, el_norm]`, where azimuth and elevation are linearly mapped from `[-45, 45] deg` to `[0, 1]`.

This is a fair quick test inside the constrained space because there is no angular wrap-around within `+/-45 deg`. It would be less appropriate for a full `+/-180 deg` or circular-angle task.

## Reproducible Setup

| Item | Value |
|---|---:|
| cache | `final_model/outputs/trainable_readout/cache_constrained_0p25_5m_pm45_train2000_val400_test400_envnoise50dB_reverb.npz` |
| train / val / test | `2000 / 400 / 400` |
| hidden neurons | `96` |
| timesteps | `12` |
| epochs | `80` |
| batch size | `16` |
| learning rate | `0.001` |
| weight decay | `1e-05` |
| residual scale | `0.15` |

## Results

| Encoding | Readout | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Euclidean MAE (m) | Combined error |
|---|---|---:|---:|---:|---:|---:|
| normalised scalar angles | direct | 0.0759 | 3.945 | 4.266 | 0.3495 | 0.0659 |
| normalised scalar angles | residual | 0.0244 | 3.781 | 3.791 | 0.3161 | 0.0577 |
| sine/cosine angles | direct | 0.0731 | 3.883 | 4.028 | 0.3431 | 0.0635 |
| sine/cosine angles | residual | 0.0202 | 3.898 | 3.341 | 0.3103 | 0.0550 |

## Interpretation

The normalised scalar-angle residual readout is not better on this quick test (`0.0577` vs `0.0550` combined error for the sine/cosine residual). The existing sine/cosine target remains the safer default.

The scalar-angle target is simpler and may be adequate in the constrained `+/-45 deg` range because the target is not circular. The sine/cosine target remains more general because it handles angular wrap-around and avoids imposing a discontinuity at the boundary.

## Regular MSE Loss Test

The main trainable readout uses the sine/cosine angle target with learned uncertainty weighting across distance, azimuth, and elevation. This second test keeps the same `[distance_norm, az_sin, az_cos, el_sin, el_cos]` target but replaces the uncertainty-weighted objective with plain evenly weighted MSE over the five output components.

| Encoding | Loss | Readout | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Euclidean MAE (m) | Combined error |
|---|---|---|---:|---:|---:|---:|---:|
| sine/cosine angles | regular MSE | direct | 0.0983 | 4.038 | 4.136 | 0.3654 | 0.0671 |
| sine/cosine angles | regular MSE | residual | 0.0257 | 3.780 | 3.347 | 0.3018 | 0.0545 |
| sine/cosine angles | uncertainty weighted | direct | 0.0731 | 3.883 | 4.028 | 0.3431 | 0.0635 |
| sine/cosine angles | uncertainty weighted | residual | 0.0202 | 3.898 | 3.341 | 0.3103 | 0.0550 |

The regular-MSE residual readout is better on this quick test (`0.0545` vs `0.0550` combined error for the uncertainty-weighted residual). This would justify repeating the comparison over multiple seeds.
