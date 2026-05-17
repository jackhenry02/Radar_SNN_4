# Azimuth ILD Pathway With SC Line Attractor

This report adds the same optimised SC line-attractor readout used in the distance pathway to the calibrated azimuth ILD population. The lower azimuth pathway is unchanged: binaural cochlea, multi-threshold ILD coding, MNTB/LSO opponent comparison, then inverse-sigmoid synaptic mapping.

![Pipeline diagram](../outputs/ild_line_attractor/figures/pipeline_diagram.png)

## Model

The no-attractor baseline decodes the inverse-sigmoid ILD population directly by centre of mass:

$$
\hat\theta_{COM}=\frac{\sum_k A_{ILD}^{inv}(\theta_k)\theta_k}{\sum_k A_{ILD}^{inv}(\theta_k)}.
$$

The CANN version injects that same population into the FI reflected Gaussian two-block balanced E/I line attractor:

$$
x(0)=s\begin{bmatrix}M \\ -\beta M\end{bmatrix}A_{ILD}^{inv},
$$

$$
\tau\dot{x}=-x+Wx,
\qquad
W=\begin{bmatrix}W_0&-W_0\\W_0&-W_0\end{bmatrix}.
$$

The final readout is centre of mass over the rectified excitatory half of the SC state at the selected readout time.

## Parameters

| Parameter | Value |
|---|---:|
| sample rate | `64000 Hz` |
| cochlea channels | `48` |
| inverse-sigmoid gain `k` | `3.750` |
| inverse-sigmoid sigma | `0.120` |
| SC attractor variant | `FI reflected Gaussian 2-block` |
| input width | `3` bins |
| recurrent width | `4` bins |
| beta | `0.897` |
| alpha prime | `4.0` |
| tau | `20.0 ms` |
| readout time | `1.0 ms` |

![Attractor matrices](../outputs/ild_line_attractor/figures/attractor_matrices.png)

## Example Dynamics

The example below shows the inverse-sigmoid ILD population being injected into the SC attractor, followed by the excitatory rate dynamics, decoded trajectory, and illustrative output spikes.

![Example dynamics](../outputs/ild_line_attractor/figures/example_dynamics.png)

## Accuracy

| Readout | MAE | RMSE | Max error | Bias |
|---|---:|---:|---:|---:|
| +/-45 direct inverse-sigmoid ILD | `0.982 deg` | `1.480 deg` | `6.577 deg` | `-0.039 deg` |
| +/-45 SC CANN | `1.019 deg` | `1.535 deg` | `6.981 deg` | `-0.038 deg` |
| +/-90 direct inverse-sigmoid ILD | `7.733 deg` | `10.699 deg` | `26.658 deg` | `1.110 deg` |
| +/-90 SC CANN | `7.658 deg` | `10.547 deg` | `25.789 deg` | `1.037 deg` |

![Prediction scatter](../outputs/ild_line_attractor/figures/prediction_scatter.png)

![Error over time](../outputs/ild_line_attractor/figures/error_over_time.png)

## Interpretation

The attractor is tested as a reversible SC readout module: it receives exactly the same inverse-sigmoid ILD population as the direct COM baseline. If the CANN improves accuracy, it is sharpening or stabilising the population readout. If it does not, then the calibrated ILD population is already close to the useful decoded statistic and recurrence mainly adds smoothing/bias.

This distinction matters biologically: the CANN is not a replacement for the LSO/MNTB cue computation. It is a candidate superior-colliculus style population stabiliser placed after the cue has already been mapped into azimuth space.

## Runtime

| Quantity | Value |
|---|---:|
| full experiment runtime | `18.33 s` |
| CANN seconds per sample, +/-45 | `0.001187` |
| CANN seconds per sample, +/-90 | `0.001437` |

## Generated Files

- `pipeline_diagram`: `azimuth_pathway/outputs/ild_line_attractor/figures/pipeline_diagram.png`
- `attractor_matrices`: `azimuth_pathway/outputs/ild_line_attractor/figures/attractor_matrices.png`
- `prediction_scatter`: `azimuth_pathway/outputs/ild_line_attractor/figures/prediction_scatter.png`
- `error_over_time`: `azimuth_pathway/outputs/ild_line_attractor/figures/error_over_time.png`
- `example_dynamics`: `azimuth_pathway/outputs/ild_line_attractor/figures/example_dynamics.png`
- `results`: `azimuth_pathway/outputs/ild_line_attractor/results.json`
