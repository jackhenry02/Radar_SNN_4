# Final Distance Pipeline With SC Line Attractor

This report is the final distance-pipeline summary before separate failure-case analysis. It keeps the current primary distance pathway unchanged through the AC distance map, then compares two SC readouts:

- `simple COM`: the existing centre-of-mass readout directly from the AC map.
- `SC line attractor`: a balanced E/I continuous-attractor readout added after AC.

The comparison is controlled because both readouts receive the same AC activity. Any difference is caused by the final SC readout only.

![Pipeline diagram](../outputs/final_distance_pipeline_with_attractor/figures/pipeline_diagram.png)

## Acoustic And Pathway Setup

| Parameter | Value |
|---|---:|
| small-space distance range | `0.25 -> 5.00 m` |
| full-space distance range | `0.25 -> 10.00 m` |
| full-space azimuth range | `±90 deg` |
| full-space elevation range | `±45 deg` |
| sample rate | `64000 Hz` |
| call sweep | `18000 -> 2000 Hz` |
| call duration | `3.0 ms` |
| small-space signal duration | `36.0 ms` |
| full-space signal duration | `70.0 ms` |
| cochlea channels | `48` |
| distance bins | `180` |
| primary variant | `Primary: dynamic spike VCN + consensus + IC facilitation` |

## Full Pipeline

### 1. Echo Waveform

The simulator generates a received binaural echo from the emitted FM call. In the full-space tests, azimuth changes the binaural timing/level cues and elevation applies spectral filtering before cochlear processing.

### 2. Cochlea

The cochlea is the optimised IIR front end from the cochlea mini-model work. It filters the waveform into frequency channels and converts activity to spikes. The current distance model uses the dynamic spike schedule:

```text
threshold(t): 16x -> 2.5x baseline threshold
beta(t):      0.20 -> 0.60
```

This makes early high-amplitude noise less likely to spike while still allowing weaker later echoes to pass.

### 3. VCN/VNLL

The VCN/VNLL stage is simplified as a causal onset detector. It ignores channels below `4 kHz`, then requires local frequency-time consensus before emitting a first onset spike for each channel:

```text
count[c,t] = sum spikes in local channel-time window
VCN[c,t] = first spike where count[c,t] >= 3
```

This replaces the much harder biological VCN/VNLL constant-latency mechanism with a robust engineering approximation.

### 4. DNLL

DNLL is modelled as delayed inhibition. It suppresses later onsets after the first echo sweep window, helping the pathway focus on the primary echo:

```text
suppress_after = first_onset + chirp_duration + padding
```

### 5. Corollary Discharge

The corollary discharge is an internal sweep-shaped copy of the expected call response. Per-channel cochlea/VCN latency is added to the CD timing rather than subtracting latency from echo spikes, keeping the whole pathway causal.

### 6. IC

The IC compares the VCN/DNLL echo onset against delayed corollary-discharge spikes for every distance bin. For a two-spike LIF coincidence detector:

```text
delta[c,k] = abs(t_echo[c] - (t_CD[c] + delay[k]))
m[c,k] = 1 + beta_IC^delta[c,k]
score[c,k] = relu(m[c,k] - threshold_IC)
```

Neighbouring frequency channels add a soft facilitation term when they support the same candidate delay. This helps a sweep-consistent echo dominate isolated noisy events without hard-gating the system.

### 7. AC

The AC stage organises the IC coincidence scores into a sharper topographic distance map with a Mexican-hat interaction:

```text
AC = relu(IC + IC * K_mexican_hat)
K = Gaussian(sigma_exc) - g_inh Gaussian(sigma_inh)
```

### 8. SC Readouts

The baseline SC readout is the original centre of mass:

```text
d_hat_COM = sum_k AC[k] d[k] / sum_k AC[k]
```

The upgraded SC readout is the finite-line balanced E/I attractor:

```text
r = [r_E, r_I]
B = [M; -beta M] / sqrt(1 + beta^2)
W = [[W0, -W0], [W0, -W0]]
tau dr/dt = -r + W r
```

The final decoded distance is centre of mass over the rectified excitatory population at `60 ms`.

## Selected SC Attractor Parameters

These parameters come from the finite-line input theory work. The model uses the reflected finite-line input because it handles boundaries better than a raw Toeplitz matrix. The recurrent matrix is the balanced two-block E/I model. The alpha choice is deliberately lower than the uncapped mathematical optimum because the capped-alpha analysis showed that very large alpha requires unrealistic peak firing rates.

| SC attractor parameter | Value |
|---|---:|
| input family | `reflected` |
| input width | `3` bins |
| recurrent width | `4` bins |
| opponent beta | `0.897` |
| alpha prime | `8.0` |
| recurrent local max eigenvalue | `8.000` |
| tau | `20.0 ms` |
| simulation step | `1.0 ms` |
| readout time | `60.0 ms` |
| rate cap | `55.0 Hz` |

## Example Spike Processing Path

The example below uses a target at `2.51 m`. Frequency is shown on a log axis; the dotted horizontal line marks the `4 kHz` VCN cut-off.

![Frequency-time rasters](../outputs/final_distance_pipeline_with_attractor/figures/frequency_time_rasters.png)

The next figure shows the conversion from IC coincidence scores to AC topographic activity, then into the final horizontal SC attractor activity over represented distance.

![Distance population stages](../outputs/final_distance_pipeline_with_attractor/figures/distance_population_stages.png)

The attractor dynamics are shown as a horizontal distance activation through SC time. This is the key visualisation of the line attractor: the activity bump evolves over time but remains organised along the distance line.

![Line attractor dynamics](../outputs/final_distance_pipeline_with_attractor/figures/line_attractor_dynamics.png)

The line attractor itself is simulated as a rate model. The spike raster below is an illustrative deterministic spike conversion from the excitatory rate state, included so the final SC output can be visualised as spikes.

![Line attractor output spikes](../outputs/final_distance_pipeline_with_attractor/figures/line_attractor_output_spikes.png)

## Readout Comparison

The simple readout and attractor readout are compared on the same AC activations.

![Readout MAE comparison](../outputs/final_distance_pipeline_with_attractor/figures/readout_mae_comparison.png)

![Readout scatter](../outputs/final_distance_pipeline_with_attractor/figures/readout_scatter.png)

| Condition | Subset | N | Simple MAE | Attractor MAE | Simple RMSE | Attractor RMSE | Simple max error | Attractor max error | Attractor runtime/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small clean 0.25-5m | all | `80` | `3.571 cm` | `3.818 cm` | `4.364 cm` | `4.717 cm` | `11.710 cm` | `13.864 cm` | `1.496 ms` |
| Small noisy 10dB+jitter | all | `80` | `7.127 cm` | `7.965 cm` | `13.797 cm` | `14.927 cm` | `104.423 cm` | `104.465 cm` | `1.866 ms` |
| Full 3D clean 0.25-10m | <=5 m | `33` | `2.831 cm` | `2.080 cm` | `4.167 cm` | `2.758 cm` | `11.500 cm` | `7.233 cm` | `1.202 ms` |
| Full 3D clean 0.25-10m | <=10 m | `80` | `34.041 cm` | `33.963 cm` | `110.014 cm` | `110.776 cm` | `464.131 cm` | `466.854 cm` | `1.202 ms` |
| Full 3D 50dB floor | <=5 m | `33` | `2.826 cm` | `2.075 cm` | `4.164 cm` | `2.753 cm` | `11.500 cm` | `7.233 cm` | `1.678 ms` |
| Full 3D 50dB floor | <=10 m | `80` | `34.040 cm` | `33.962 cm` | `110.014 cm` | `110.776 cm` | `464.131 cm` | `466.854 cm` | `1.678 ms` |

## Interpretation

- The final SC line attractor is now attached as a reversible readout module after AC.
- The comparison is controlled: cochlea, VCN/VNLL, DNLL, IC, and AC are identical for both readouts.
- The line attractor gives a biologically motivated recurrent readout and a clear population-bump visualisation.
- If the attractor does not improve a condition, that means the AC map already contains the relevant bias or ambiguity; the SC cannot recover information that was lost upstream.
- Failure-case analysis is intentionally deferred to the next report.

## Generated Files

- `pipeline_diagram`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/pipeline_diagram.png`
- `frequency_time_rasters`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/frequency_time_rasters.png`
- `distance_population_stages`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/distance_population_stages.png`
- `line_attractor_dynamics`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/line_attractor_dynamics.png`
- `line_attractor_output_spikes`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/line_attractor_output_spikes.png`
- `readout_mae_comparison`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/readout_mae_comparison.png`
- `readout_scatter`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/figures/readout_scatter.png`
- `results`: `distance_pathway/outputs/final_distance_pipeline_with_attractor/results.json`

Runtime: `18.03 s`.
