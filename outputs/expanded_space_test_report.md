# Expanded Space Quick Test

## Control Check

The same matched-human / round-2 combined-all harness was rerun on the original task limits before this expanded-space run. That control did not collapse, so the flat behavior in the expanded test is not a generic harness failure.

- Control combined error: `0.1150`
- Control distance / azimuth / elevation: `0.0720 m`, `5.4033 deg`, `8.6680 deg`
- Control prediction spread: distance std `0.5402`, azimuth std `24.1809`, elevation std `14.8356`
- Control target spread: distance std `0.5632`, azimuth std `24.2894`, elevation std `16.5835`

## Overview

This run reuses the matched-human round-2 combined-all architecture and tests it on a different spatial support. The goal is not a fair benchmark against the original domain, but a quick check of whether the current system still trains and produces structured predictions when the range and angle support are changed. This scenario is: `Expanded Space Quick Test`.

Important note:
- The existing matched-human spike cache was not reused here.
- That cache is for a different domain and also uses a different cochlea width, so it cannot support this test directly.
- Direct metric comparison against the original short-domain matched-human run is therefore only contextual, not like-for-like.

## Test Setup

- Model architecture: `combined_all`
- Reference architecture source: saved round-2 combined-all matched-human run
- Dataset counts: `train 700 / val 150 / test 150`
- Distance support: `0.5 to 20.0 m`
- Azimuth support: `-90 to 90 deg`
- Elevation support: `-90 to 90 deg`
- Signal duration increased to: `139.6 ms`
- Sample rate: `64000` (`matched-human front end`)
- Chirp: `18000 Hz -> 2000 Hz`
- Cochlea range: `2000 Hz -> 20000 Hz`
- Cochlea channels actually used in front end: `48`
- Downstream model frequency width: `48`
- Delay lines: `64`
- Branch hidden dim: `40`
- Fusion hidden dim: `160`
- SNN time steps: `12`
- Loss mode: `mixed_cartesian_expanded`
- Max epochs: `10`, batch size: `6`

## Results

- Validation combined error: `0.6332`
- Test combined error: `0.6315`
- Test distance MAE: `4.1299 m`
- Test azimuth MAE: `41.2251 deg`
- Test elevation MAE: `43.7670 deg`
- Test Euclidean error: `10.7116 m`
- Mean spike rate: `0.0354`

## Timing

- Data preparation: `46.71 s`
- Training: `1972.71 s`
- Evaluation: `13.24 s`
- Total: `2032.67 s`
- Best epoch: `3`

## Reference Comparison

The saved reference below is the original short-domain matched-human round-2 combined-all run (`0.5 to 2.5 m`, `-45 to 45 deg`, `-30 to 30 deg`). It is included only as context.

- Reference combined error: `0.1221`
- Reference distance / azimuth / elevation: `0.0946 m`, `7.8027 deg`, `8.4785 deg`
- Scenario combined error delta vs reference: `0.5094`
- Scenario distance MAE delta vs reference: `4.0353 m`
- Scenario azimuth MAE delta vs reference: `33.4224 deg`
- Scenario elevation MAE delta vs reference: `35.2885 deg`

## Interpretation

- When the support is widened, the range expansion is especially severe because the echo delay support grows from a few milliseconds to over 100 ms, forcing a much longer receive window even at the cheaper 64 kHz front end.
- The angular task becomes harder when the model has to cover the full front hemisphere for both azimuth and elevation.
- This rerun increases delay-bank and latent capacity relative to the earlier failed expanded attempt, so it is testing whether the previous collapse was partly a model-capacity mismatch rather than only a data-domain mismatch.
- The round-2 combined-all architecture is still the base model; the main changes are support-aware delay-bank and latent-capacity scaling.
- If performance degrades sharply but predictions still show non-trivial spread, that suggests the pipeline remains functional but is out of its previously tuned operating regime.

## Failure Analysis

The current evidence points away from a generic code or report bug and toward a front-end failure mode under the expanded-space acoustics:

- The control run uses the same matched-human front end and the same round-2 combined-all harness, and it does not collapse.
- Increasing delay lines from the short-range setting to `64`, and increasing latent capacity, only changed the expanded result slightly. That means the collapse is not primarily caused by the old short-range delay-bank width.
- The `1/r^2` attenuation was intentionally left unchanged. At `20 m`, the return is much weaker than in the original task, so the effective cue SNR is much worse.
- The strongest new clue is in the spike encoder itself: `lif_encode_stages()` in [models/acoustics.py](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/models/acoustics.py) scales every sample by its own maximum envelope before thresholding. That means weak long-range echoes are renormalized upward, so noise and low-level background structure can dominate the spike raster instead of simply disappearing.
- That combination is a plausible explanation for the observed behavior: the model gets less trustworthy cue structure at long range, then regresses toward mean-like predictions in azimuth and elevation.

## Front-End Distance Sweep

To make that visible, I generated matched-human receive-waveform, cochleagram, and spike-raster diagnostics at several fixed distances with `azimuth = 0`, `elevation = 0`, and `add_noise = True`. The figures are in [expanded_space_frontend_diagnostics](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_diagnostics), with the summary in [summary.json](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_diagnostics/summary.json).

Important note:
- `0 m` is not included because the current simulator uses inverse-square attenuation and a two-way delay model, so `0.5 m` is the nearest practical diagnostic point.

Key observations:
- `0.5 m`: receive peak `0.695879`, cochleagram peak `0.189507`, spike count `484`
- `2.5 m`: receive peak `0.045582`, cochleagram peak `0.007484`, spike count `31025`
- `5.0 m`: receive peak `0.034623`, cochleagram peak `0.003716`, spike count `64364`
- `10.0 m`: receive peak `0.032740`, cochleagram peak `0.002717`, spike count `80493`
- `15.0 m`: receive peak `0.034823`, cochleagram peak `0.003047`, spike count `75561`
- `20.0 m`: receive peak `0.031619`, cochleagram peak `0.002756`, spike count `80071`

The important pattern is not just that the waveform/cochleagram amplitudes fall with distance. It is that the spike count explodes once the return becomes weak. That strongly suggests the current normalization-plus-thresholding front end is turning weak long-range/noisy inputs into broad, noise-dominated spike activity instead of preserving clean range-dependent structure.

Diagnostic figures:
![0.5 m cochleagram and spikes](expanded_space_frontend_diagnostics/distance_0p5_cochleagram_spikes.png)
![2.5 m cochleagram and spikes](expanded_space_frontend_diagnostics/distance_2p5_cochleagram_spikes.png)
![5.0 m cochleagram and spikes](expanded_space_frontend_diagnostics/distance_5p0_cochleagram_spikes.png)
![10.0 m cochleagram and spikes](expanded_space_frontend_diagnostics/distance_10p0_cochleagram_spikes.png)
![15.0 m cochleagram and spikes](expanded_space_frontend_diagnostics/distance_15p0_cochleagram_spikes.png)
![20.0 m cochleagram and spikes](expanded_space_frontend_diagnostics/distance_20p0_cochleagram_spikes.png)

## Plots

![Loss](expanded_space_test/expanded_space_quick_test/loss.png)
![Distance prediction](expanded_space_test/expanded_space_quick_test/test_distance_prediction.png)
![Azimuth prediction](expanded_space_test/expanded_space_quick_test/test_azimuth_prediction.png)
![Elevation prediction](expanded_space_test/expanded_space_quick_test/test_elevation_prediction.png)
![Coordinate error profile](expanded_space_test/expanded_space_quick_test/coordinate_error_profile.png)
![Reference vs scenario](expanded_space_test/reference_vs_expanded.png)

## Appendix: No-Normalization Spike Raster Test

This second diagnostic repeats the same fixed-distance cochleagram and spike-raster sweep, but disables the per-sample envelope normalization before the LIF stage. The original normalized figures above are kept unchanged; this section is an additional comparison only.

- Diagnostic summary: [summary.json](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_diagnostics_no_norm/summary.json)
- Diagnostic figures directory: [expanded_space_frontend_diagnostics_no_norm](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_diagnostics_no_norm)

Key observations:
- `0.5 m`: receive peak `0.695879`, cochleagram peak `0.189507`, spike count `55`
- `2.5 m`: receive peak `0.045582`, cochleagram peak `0.007484`, spike count `0`
- `5.0 m`: receive peak `0.034623`, cochleagram peak `0.003716`, spike count `0`
- `10.0 m`: receive peak `0.032740`, cochleagram peak `0.002717`, spike count `0`
- `15.0 m`: receive peak `0.034823`, cochleagram peak `0.003047`, spike count `0`
- `20.0 m`: receive peak `0.031619`, cochleagram peak `0.002756`, spike count `0`

Interpretation:
- Turning normalization off removes the spike-count explosion seen in the original diagnostic.
- But with the current threshold left unchanged, almost all returns beyond `0.5 m` produce no spikes at all.
- So the current front end is bracketed between two bad regimes:
  - with per-sample normalization, weak long-range returns become dense, noise-dominated spike rasters
  - without normalization, weak long-range returns fall below threshold and vanish
- That strongly suggests the real issue is not simply "normalization on or off", but the lack of a range-stable amplitude calibration between the cochleagram and the LIF threshold.

No-normalization diagnostic figures:
![0.5 m no-norm cochleagram and spikes](expanded_space_frontend_diagnostics_no_norm/distance_0p5_cochleagram_spikes.png)
![2.5 m no-norm cochleagram and spikes](expanded_space_frontend_diagnostics_no_norm/distance_2p5_cochleagram_spikes.png)
![5.0 m no-norm cochleagram and spikes](expanded_space_frontend_diagnostics_no_norm/distance_5p0_cochleagram_spikes.png)
![10.0 m no-norm cochleagram and spikes](expanded_space_frontend_diagnostics_no_norm/distance_10p0_cochleagram_spikes.png)
![15.0 m no-norm cochleagram and spikes](expanded_space_frontend_diagnostics_no_norm/distance_15p0_cochleagram_spikes.png)
![20.0 m no-norm cochleagram and spikes](expanded_space_frontend_diagnostics_no_norm/distance_20p0_cochleagram_spikes.png)

## Appendix: High-Amplitude Spike Raster Test

This diagnostic keeps the original normalized spike encoder, but increases transmit gain before propagation and before additive noise is applied. It is intended to test whether stronger signal amplitude restores cleaner long-range spike structure without changing the attenuation law.
The gain labels below are also shown in relative dB, computed as `20 * log10(gain)` with `1x = 0 dB` referenced to the baseline simulated chirp amplitude. They are not absolute SPL values.

- Diagnostic summary: [summary.json](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_gain_diagnostics/summary.json)
- Diagnostic figures directory: [expanded_space_frontend_gain_diagnostics](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_gain_diagnostics)

Key observations:
- gain `1.0x` (`+0.0 dB re 1x`), distance `2.5 m`: receive peak `0.042572`, cochleagram peak `0.007435`, spike count `31505`
- gain `1.0x` (`+0.0 dB re 1x`), distance `10.0 m`: receive peak `0.032314`, cochleagram peak `0.002783`, spike count `79858`
- gain `1.0x` (`+0.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `75797`
- gain `4.0x` (`+12.0 dB re 1x`), distance `2.5 m`: receive peak `0.118146`, cochleagram peak `0.029307`, spike count `2699`
- gain `4.0x` (`+12.0 dB re 1x`), distance `10.0 m`: receive peak `0.032314`, cochleagram peak `0.002783`, spike count `79898`
- gain `4.0x` (`+12.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `75802`
- gain `10.0x` (`+20.0 dB re 1x`), distance `2.5 m`: receive peak `0.287450`, cochleagram peak `0.073052`, spike count `499`
- gain `10.0x` (`+20.0 dB re 1x`), distance `10.0 m`: receive peak `0.032314`, cochleagram peak `0.005215`, spike count `47152`
- gain `10.0x` (`+20.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `75865`
- gain `20.0x` (`+26.0 dB re 1x`), distance `2.5 m`: receive peak `0.569624`, cochleagram peak `0.145959`, spike count `497`
- gain `20.0x` (`+26.0 dB re 1x`), distance `10.0 m`: receive peak `0.042151`, cochleagram peak `0.009821`, spike count `21890`
- gain `20.0x` (`+26.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `75920`

Interpretation:
- At `2.5 m`, increasing transmit gain strongly reduces the pathological spike-count explosion. By `10x` to `20x`, the raster becomes much cleaner and the spike count falls from about `31k` to about `500`.
- At `10.0 m`, higher gain helps partially. The spike count stays pathological at `4x`, improves somewhat at `10x`, and improves more at `20x`, but is still far above a clean sparse raster.
- At `20.0 m`, increasing transmit gain has almost no visible effect in this diagnostic. The receive peak, cochleagram peak, and spike count remain almost unchanged across gains, which suggests the additive noise floor is still dominating what reaches the encoder at that range.
- So "make the signal louder" is not useless here. It does help at shorter and mid expanded ranges. But by `20 m`, the current front end is still effectively noise-limited under the unchanged attenuation law.

Representative high-amplitude diagnostic figures:
![2.5 m gain 4x](expanded_space_frontend_gain_diagnostics/gain_4p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 4x](expanded_space_frontend_gain_diagnostics/gain_4p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 4x](expanded_space_frontend_gain_diagnostics/gain_4p0_distance_20p0_cochleagram_spikes.png)
![2.5 m gain 10x](expanded_space_frontend_gain_diagnostics/gain_10p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 10x](expanded_space_frontend_gain_diagnostics/gain_10p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 10x](expanded_space_frontend_gain_diagnostics/gain_10p0_distance_20p0_cochleagram_spikes.png)
![2.5 m gain 20x](expanded_space_frontend_gain_diagnostics/gain_20p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 20x](expanded_space_frontend_gain_diagnostics/gain_20p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 20x](expanded_space_frontend_gain_diagnostics/gain_20p0_distance_20p0_cochleagram_spikes.png)

## Appendix: High-Amplitude No-Normalization Spike Raster Test

This diagnostic repeats the same transmit-gain sweep, but also disables per-sample envelope normalization before the LIF stage. It is the direct comparison to the normalized high-amplitude sweep above.
The gain labels below are also shown in relative dB, computed as `20 * log10(gain)` with `1x = 0 dB` referenced to the baseline simulated chirp amplitude. They are not absolute SPL values.

- Diagnostic summary: [summary.json](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_gain_no_norm_diagnostics/summary.json)
- Diagnostic figures directory: [expanded_space_frontend_gain_no_norm_diagnostics](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_gain_no_norm_diagnostics)

Key observations:
- gain `1.0x` (`+0.0 dB re 1x`), distance `2.5 m`: receive peak `0.042572`, cochleagram peak `0.007435`, spike count `0`
- gain `1.0x` (`+0.0 dB re 1x`), distance `10.0 m`: receive peak `0.032314`, cochleagram peak `0.002783`, spike count `0`
- gain `1.0x` (`+0.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `0`
- gain `4.0x` (`+12.0 dB re 1x`), distance `2.5 m`: receive peak `0.118146`, cochleagram peak `0.029307`, spike count `0`
- gain `4.0x` (`+12.0 dB re 1x`), distance `10.0 m`: receive peak `0.032314`, cochleagram peak `0.002783`, spike count `0`
- gain `4.0x` (`+12.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `0`
- gain `10.0x` (`+20.0 dB re 1x`), distance `2.5 m`: receive peak `0.287450`, cochleagram peak `0.073052`, spike count `7`
- gain `10.0x` (`+20.0 dB re 1x`), distance `10.0 m`: receive peak `0.032314`, cochleagram peak `0.005215`, spike count `0`
- gain `10.0x` (`+20.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `0`
- gain `20.0x` (`+26.0 dB re 1x`), distance `2.5 m`: receive peak `0.569624`, cochleagram peak `0.145959`, spike count `38`
- gain `20.0x` (`+26.0 dB re 1x`), distance `10.0 m`: receive peak `0.042151`, cochleagram peak `0.009821`, spike count `0`
- gain `20.0x` (`+26.0 dB re 1x`), distance `20.0 m`: receive peak `0.034623`, cochleagram peak `0.002961`, spike count `0`

Interpretation:
- Without normalization, the gain sweep confirms the opposite failure mode very strongly.
- Even large transmit gains barely recover spikes beyond `2.5 m`.
- At `2.5 m`, very large gains (`10x` to `20x`) begin to recover a small number of spikes.
- At `10.0 m` and `20.0 m`, even `20x` gain still produces zero spikes in this setup.
- So the current front end is not just overly normalized. It is also under-sensitive when normalization is removed and the threshold is left unchanged.
- Taken together with the normalized gain sweep, this strongly suggests the missing piece is a better amplitude calibration or adaptive gain-control mechanism between cochleagram output and spike threshold, rather than a simple binary switch.

Representative high-amplitude no-normalization diagnostic figures:
![2.5 m gain 4x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_4p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 4x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_4p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 4x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_4p0_distance_20p0_cochleagram_spikes.png)
![2.5 m gain 10x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_10p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 10x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_10p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 10x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_10p0_distance_20p0_cochleagram_spikes.png)
![2.5 m gain 20x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_20p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 20x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_20p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 20x no-norm](expanded_space_frontend_gain_no_norm_diagnostics/gain_20p0_distance_20p0_cochleagram_spikes.png)

## Appendix: Extreme-Amplitude No-Normalization Spike Raster Test

This diagnostic extends the no-normalization comparison to much larger transmit amplitudes. It is intended to test whether the unnormalized front end can be driven back into a useful spiking regime by very large source levels alone.
The gain labels below are also shown in relative dB, computed as `20 * log10(gain)` with `1x = 0 dB` referenced to the baseline simulated chirp amplitude. They are not absolute SPL values.

- Diagnostic summary: [summary.json](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_gain_no_norm_high_diagnostics/summary.json)
- Diagnostic figures directory: [expanded_space_frontend_gain_no_norm_high_diagnostics](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/expanded_space_frontend_gain_no_norm_high_diagnostics)

Useful mapping if `1x` were interpreted as `80 dB SPL`:
- `100x` = `+40 dB re 1x` = `120 dB SPL`
- `500x` = `+54.0 dB re 1x` = `134.0 dB SPL`
- `1000x` = `+60.0 dB re 1x` = `140 dB SPL`

Key observations:
- gain `100.0x` (`+40.0 dB re 1x`), distance `2.5 m`: receive peak `2.827017`, cochleagram peak `0.729218`, spike count `311`
- gain `100.0x` (`+40.0 dB re 1x`), distance `10.0 m`: receive peak `0.180975`, cochleagram peak `0.047300`, spike count `0`
- gain `100.0x` (`+40.0 dB re 1x`), distance `20.0 m`: receive peak `0.050307`, cochleagram peak `0.011263`, spike count `0`
- gain `500.0x` (`+54.0 dB re 1x`), distance `2.5 m`: receive peak `14.113980`, cochleagram peak `3.645511`, spike count `833`
- gain `500.0x` (`+54.0 dB re 1x`), distance `10.0 m`: receive peak `0.878807`, cochleagram peak `0.235481`, spike count `78`
- gain `500.0x` (`+54.0 dB re 1x`), distance `20.0 m`: receive peak `0.226245`, cochleagram peak `0.057505`, spike count `0`
- gain `1000.0x` (`+60.0 dB re 1x`), distance `2.5 m`: receive peak `28.222683`, cochleagram peak `7.290877`, spike count `1116`
- gain `1000.0x` (`+60.0 dB re 1x`), distance `10.0 m`: receive peak `1.752803`, cochleagram peak `0.470707`, spike count `197`
- gain `1000.0x` (`+60.0 dB re 1x`), distance `20.0 m`: receive peak `0.447518`, cochleagram peak `0.115309`, spike count `23`

Interpretation:
- Very large source levels do eventually recover spikes in the unnormalized front end.
- At `2.5 m`, the recovered spike counts become substantial by `100x` and then continue rising at `500x` and `1000x`.
- At `10.0 m`, there are still no spikes at `100x`, but `500x` and `1000x` finally begin to produce a non-zero raster.
- At `20.0 m`, even `500x` is still silent, and `1000x` only produces `23` spikes, so the front end remains extremely under-sensitive at the far end of the expanded domain when normalization is disabled.
- This is a useful contrast with the normalized case: removing normalization can preserve the attenuation cue, but only if the source level is made unrealistically large relative to the current thresholding scheme.

Representative extreme-amplitude no-normalization diagnostic figures:
![2.5 m gain 100x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_100p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 100x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_100p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 100x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_100p0_distance_20p0_cochleagram_spikes.png)
![2.5 m gain 500x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_500p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 500x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_500p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 500x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_500p0_distance_20p0_cochleagram_spikes.png)
![2.5 m gain 1000x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_1000p0_distance_2p5_cochleagram_spikes.png)
![10.0 m gain 1000x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_1000p0_distance_10p0_cochleagram_spikes.png)
![20.0 m gain 1000x no-norm](expanded_space_frontend_gain_no_norm_high_diagnostics/gain_1000p0_distance_20p0_cochleagram_spikes.png)
