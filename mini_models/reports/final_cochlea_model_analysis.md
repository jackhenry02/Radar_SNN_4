# Final Cochlea Model And Analysis

This document consolidates the selected cochlea front end for the next model design. It uses the optimized IIR resonator filterbank, TorchScript LIF spike encoding, and active-window gating.

## Final Model

```mermaid
flowchart LR
    A[received waveform] --> B[active window detector]
    B --> C[crop active echo with padding]
    C --> D[torchaudio lfilter IIR resonator bank]
    D --> E[half-wave rectification]
    E --> F[TorchScript LIF with subtractive reset]
    F --> G[full-length spike raster]
```

| Parameter | Value |
|---|---:|
| sample rate | `64000 Hz` |
| chirp | `18000 -> 2000 Hz` |
| cochlea band | `2000 -> 20000 Hz` |
| channels for final example | `48` |
| final IIR Q factor | `12.0` |
| spike threshold | `0.42` |
| spike beta | `0.88` |
| active-window threshold | `0.02 * max(abs(signal))` |
| active-window padding | `1.0 ms` |

The Q factor was increased from the earlier exploratory value of `5` to `12` to make the resonator channels more frequency selective. This improves separation between neighbouring channels, at the cost of longer ringing.

## Final Output

- Runtime for the `48`-channel final example: `2.532 ms`
- Estimated FLOPs for active window: `170,496`
- Output spike count: `4212`
- Active samples processed: `296` / `1408`
- Active fraction: `0.210`

![Final cochlea output](../outputs/final_cochlea_model/figures/final_cochlea_output.png)

## IIR Filter Mathematics

Each channel is implemented as a second-order resonator. For centre frequency `f_c`, sampling rate `f_s`, and quality factor `Q`:

```text
bandwidth_c = f_c / Q
theta_c = 2*pi*f_c / f_s
r_c = exp(-pi*bandwidth_c / f_s)
```

The time-domain difference equation is:

```text
y_c[n] = b0_c*x[n] + 2*r_c*cos(theta_c)*y_c[n-1] - r_c^2*y_c[n-2]
b0_c = 1 - r_c
```

The transfer function is:

```text
H_c(z) = b0_c / (1 - 2*r_c*cos(theta_c)*z^-1 + r_c^2*z^-2)
```

The poles are:

```text
z = r_c * exp(+/-j*theta_c)
```

For the final model, pole radii range from `0.9214` to `0.9919`. Since all pole radii are less than `1`, the IIR filters are stable.

![Q selectivity](../outputs/final_cochlea_model/figures/q_selectivity.png)

![Final frequency response](../outputs/final_cochlea_model/figures/final_frequency_response.png)

![IIR stability](../outputs/final_cochlea_model/figures/iir_stability.png)

## Channel Scaling

The plot below compares runtime as the channel count increases from `10` to `1000`. The final IIR model uses active-window gating, so its operation count depends on active samples rather than full waveform length.

![Runtime scaling](../outputs/final_cochlea_model/figures/channel_scaling_runtime.png)

![FLOP scaling](../outputs/final_cochlea_model/figures/channel_scaling_flops.png)

| Channels | Final IIR time (ms) | FFT time (ms) | Final IIR FLOPs | FFT FLOPs |
|---:|---:|---:|---:|---:|
| 10 | 2.645 | 3.959 | 35,520 | 1,939,958 |
| 24 | 2.683 | 4.136 | 85,248 | 4,552,812 |
| 48 | 2.679 | 4.345 | 170,496 | 9,031,990 |
| 100 | 3.101 | 7.033 | 355,200 | 18,736,874 |
| 200 | 3.498 | 10.268 | 710,400 | 37,400,114 |
| 500 | 4.754 | 21.504 | 1,776,000 | 93,389,834 |
| 1000 | 7.834 | 40.512 | 3,552,000 | 186,706,033 |

## Interpretation

- Increasing Q from `5` to `12` narrows each IIR channel and improves theoretical frequency selectivity.
- Stability is guaranteed by construction as long as `Q > 0` and `f_c > 0`, because `r = exp(-pi*f_c/(Q*f_s))` lies inside the unit circle.
- The final model is still not truly sparse inside the active window. It is a gated dense computation: silence is skipped, but the echo window is processed by all channels.
- The scaling curves are the key test for whether this front end remains practical as channel count increases.

## Generated Files

- `final_cochlea_output`: `mini_models/outputs/final_cochlea_model/figures/final_cochlea_output.png`
- `q_selectivity`: `mini_models/outputs/final_cochlea_model/figures/q_selectivity.png`
- `final_frequency_response`: `mini_models/outputs/final_cochlea_model/figures/final_frequency_response.png`
- `iir_stability`: `mini_models/outputs/final_cochlea_model/figures/iir_stability.png`
- `channel_scaling_runtime`: `mini_models/outputs/final_cochlea_model/figures/channel_scaling_runtime.png`
- `channel_scaling_flops`: `mini_models/outputs/final_cochlea_model/figures/channel_scaling_flops.png`
- `results`: `mini_models/outputs/final_cochlea_model/results.json`

Runtime: `1.63 s`.
