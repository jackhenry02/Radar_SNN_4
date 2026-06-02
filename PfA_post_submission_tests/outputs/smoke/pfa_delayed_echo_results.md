# PfA Delayed-Echo Readout Test

This post-submission experiment reuses the cached 50 dB environmental-noise plus delayed-echo datasets. It does not regenerate the acoustic simulation or modify the report-backed pipeline.

PfA is applied after each hidden linear layer and before the corresponding LIF neuron. The structured model uses residual fusion; the cochlear-spike-raster (CSR) model uses the direct time-binned raster input without the fixed Gaussian projection. No raw-waveform variant is included.

## Results

| Model | Distance MAE | Azimuth MAE | Elevation MAE | Euclidean MAE | Combined error |
|---|---:|---:|---:|---:|---:|
| structured residual reference | 0.0202 m | 3.898 deg | 3.341 deg | 0.3103 m | 0.0550 |
| structured residual + PfA | 0.0533 m | 6.553 deg | 23.254 deg | 1.4187 m | 0.2243 |
| direct CSR reference | 0.0632 m | 5.900 deg | 4.704 deg | 0.2685 m | 0.0828 |
| direct CSR + PfA | 0.7875 m | 19.423 deg | 9.206 deg | 1.4415 m | 0.2646 |

## Run Setup

- epochs: `2`
- hidden neurons: `96`
- SNN timesteps: `12`
- batch size: `16`
- structured splits: `{'train': 64, 'val': 24, 'test': 24}`
- direct CSR splits: `{'train': 64, 'val': 24, 'test': 24}`

The saved JSON contains the full metric dictionaries and training histories.