# Final Integrated Model Explained

This document defines the final integrated model in signal order, from target coordinate to predicted coordinate. It is intended as a parameter reference and conceptual walkthrough.

## Coordinate Convention

A target is represented by spherical coordinates `(r, a, e)`: distance `r` in metres, azimuth `a` in degrees, and elevation `e` in degrees. Cartesian error is computed using:

$$
x=r\cos e\cos a,\quad y=r\cos e\sin a,\quad z=r\sin e.
$$

## Acoustic Signal

| Quantity | Value / definition |
|---|---|
| sample rate | `64 kHz` |
| chirp duration | `3 ms` |
| chirp sweep | `18 kHz -> 2 kHz` linear down-sweep |
| transmit gain | `1000x`, treated as 140 dB relative to an 80 dB amplitude-1 reference |
| attenuation | `0.7 / path_length^2` in the simulator |
| binaural cue | path-length ITD plus multiplicative head shadow |
| elevation cue | deep comb filter with delayed-copy gain `0.99` |

The emitted chirp is:

$$
s(t)=A w(t)\sin\{2\pi[f_0t+\tfrac{1}{2}kt^2]\}.
$$

The elevation comb cue is:

$$
y(t)=x(t)+a x(t-\tau),\qquad |H(f)|=\frac{\sqrt{1+a^2+2a\cos(2\pi f\tau)}}{1+a}.
$$

## Shared Cochlea

Each pathway currently runs its own copy of the final IIR cochlea. This is intentionally conservative for timing; the production model should share this stage.

| Parameter | Value |
|---|---:|
| default cochlear channels | `48` |
| cochlea centre spacing | logarithmic over the matched-human band |
| final Q factor | `12.0` |
| dynamic threshold | `16x -> 2.5x` spike threshold |
| dynamic beta | `0.20 -> 0.60` |
| VCN low-frequency silence | channels below `4 kHz` ignored for distance/onset processing |

The dynamic cochlear LIF is:

$$
v_c[t]=\beta(t)v_c[t-1]+I_c[t],\quad S_c[t]=\mathbb{1}[v_c[t]\geq \theta(t)].
$$

## Distance Pathway

The distance pathway estimates range from echo delay. It uses dynamic cochlear spikes, VCN consensus, DNLL late suppression, IC coincidence, AC sharpening, and SC CANN readout.

$$
C_k=\sum_c \max(0, 1+\beta^{|t^{echo}_c-(t^{CD}_c+d_k)|}-\theta).
$$

Neighbouring-channel sweep facilitation boosts candidates consistent across the FM sweep. The AC applies a Mexican-hat kernel. The SC uses the reflected FI two-block line attractor and decodes centre of mass over distance bins.

## Azimuth Pathway

The azimuth pathway uses the ITD branch as the final selected branch. The lower pathway runs binaural cochleae, VCN onset extraction for each ear, and a Jeffress-style candidate-delay coincidence population.

$$
\Delta t_k \approx \frac{d_{ear}}{c}\sin a_k,\qquad A_k=\sum_c \max(0,1+\beta^{|\Delta n_c-\Delta n_k|}-\theta).
$$

This ITD population is injected into the reflected FI two-block SC line attractor. The ILD inverse-sigmoid pathway remains useful in the isolated azimuth sweep, but the ITD branch was selected here because it was more stable in the constrained full-3D test.

## Elevation Pathway

The elevation pathway is monaural after selected-ear gating by azimuth sign. The DCN template compares the observed spectrum with the expected comb transfer function.

$$
m_k(f_c)=P_0(f_c)(1-H_k(f_c))^2,
$$

$$
r_k=\exp\left[-\frac{\sum_c m_k(f_c)(\tilde p_c-H_k(f_c))^2}{2\sigma^2}\right].
$$

This DCN population is injected into the reflected FI two-block SC line attractor at a 5 ms readout time. A final inverse-sigmoid calibration corrects the stable monotonic distortion in the raw elevation readout.

## SC Line Attractor

All three pathway readouts use the same generic finite-line attractor form:

$$
\tau\dot{x}=-x+Wx,\qquad x(0)=s\begin{bmatrix}Mu\\-\beta Mu\end{bmatrix}.
$$

This FI-optimised input is a common readout geometry, not a pathway-specific sensory optimisation.

| Attractor parameter | Value |
|---|---:|
| alpha prime | `4.0` |
| input width | `3 bins` |
| recurrent width | `4 bins` |
| tau | `20 ms` |
| dt | `1 ms` |
| simulation time | `60 ms` |
| rate cap | `55 Hz` |

## Current Integrated Performance

The constrained integrated test used `24` samples. Its main metrics were: distance MAE `0.0310 m`, azimuth MAE `4.896 deg`, elevation MAE `0.810 deg`, and Euclidean MAE `0.2394 m`.

## Important Implementation Caveats

- The three pathways currently simulate or process cochlear activity separately; sharing the cochlea should reduce runtime.
- The azimuth branch uses an untuned ITD CANN readout; the elevation calibration is tuned on a controlled sweep and then reused in full 3D.
- The expanded 0-10 m, +/-90 degree test is intentionally a stress test outside the main tuned operating range.
- Exact zero range is numerically replaced by `0.02 m` to avoid a singular path length; the expanded test should be read as a near-zero-to-10 m stress test.
- FLOPs and SOPs in the results report are analytical estimates, not hardware profiler counts.
