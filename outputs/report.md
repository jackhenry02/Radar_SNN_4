# Bat-Inspired SNN Localisation Report

## System Overview

The repository implements an echolocation pipeline that starts from a downward FM chirp, propagates the pulse through a synthetic 3D scene, models binaural echoes, transforms the waveform into cochlear spike trains, and predicts object distance, azimuth, and elevation. The final localisation stage uses an explicit three-pathway architecture so the temporal, binaural, and spectral cues are processed separately before fusion.

Core stages:

- FM chirp synthesis and echo simulation in Cartesian space expressed as range, azimuth, and elevation.
- Binaural reception with interaural time and level differences plus simple elevation-dependent spectral shaping.
- Cochlea front end with log-spaced filters, half-wave rectification, envelope extraction, and spike encoding.
- Parallel pathway SNN with distance delay tuning, azimuth ITD/ILD processing, and elevation spectral cue extraction.
- Fusion and readout layers that predict object location and report spike-rate efficiency.

## Mermaid Flowcharts

### Full System Pipeline

```mermaid
graph TD
    A[FM Chirp] --> B[3D Echo Simulation]
    B --> C[Binaural Reception]
    C --> D[Cochlea Filterbank]
    D --> E[Spike Encoding]
    E --> F[Distance Pathway]
    E --> G[Azimuth Pathway]
    E --> H[Elevation Pathway]
    F --> I[Fusion Layer]
    G --> I
    H --> I
    I --> J[Localisation Output]
```

### SNN Architecture

```mermaid
graph TD
    A[Cochlea Spikes] --> B1[Distance Delay Bank]
    A --> B2[ITD and ILD Branch]
    A --> B3[Spectral Notch Branch]
    B1 --> C1[Branch Projection]
    B2 --> C2[Branch Projection]
    B3 --> C3[Branch Projection]
    C1 --> D[Fusion Spiking Layer]
    C2 --> D
    C3 --> D
    D --> E[Integration Spiking Layer]
    E --> F[Linear Readout]
```

### Pathway Split Diagram

```mermaid
graph TD
    A[Cochlea Spikes]
    A --> B[Distance Pathway]
    A --> C[Azimuth Pathway]
    A --> D[Elevation Pathway]
    B --> E[Delay Tuning Features]
    C --> F[ITD and ILD Features]
    D --> G[Spectral Notch Features]
    E --> H[Fusion]
    F --> H
    G --> H
    H --> I[Distance, Azimuth, Elevation]
```

## Biological Interpretation

- Chirp generation and echo simulation approximate bat vocalisation and pulse-echo acoustics.
- The cochlear filterbank and spike encoder approximate basilar membrane channelisation and auditory nerve spiking.
- The distance pathway maps to delay-sensitive processing for pulse-echo timing.
- The azimuth pathway maps to superior-olive style ITD and ILD processing.
- The elevation pathway maps to pinna-driven spectral cue analysis.
- The fusion spiking layer acts as a compact inferior-colliculus and cortex-like integration stage.

## Pathway Split Details

- Distance pathway: matched pulse-echo timing is converted into vectorised delay-bank features and coincidence-style activations that emphasise round-trip delay.
- Azimuth pathway: binaural spikes are transformed into ITD features through signed delay sweeps and augmented with ILD rate contrasts.
- Elevation pathway: per-channel spike counts, local spectral notches, and spectral slope features capture pinna-like frequency shaping without explicit delay lines.
- Fusion stage: each pathway is projected into its own latent space, concatenated, then passed through two spiking integration layers before linear readout.

## Model-by-Model Summary

| Model | Status | Key Result |
| --- | --- | --- |
| Model 0 | Pass | Distance MAE 0.0231 m |
| Model 1 | Pass | Peak delay error 0.0 bins |
| Model 2 | Pass | Mean delay error 0.0 bins |
| Model 3 | Pass | Energy/spike correlation 0.9865 |
| Model 4 | Pass | Validation accuracy 0.9896 |
| Model 5 | Pass | Distance MAE 0.1404 m, angular MAE 13.5678 deg |
| Model 6 (previous) | Pass | Combined error 0.1228 |
| Model 6 (pathway split) | Pass | Combined error 0.0921 |
| Model 7 (initial) | Fail | Improvement fraction 0.0531 |
| Model 7 (enhanced) | Fail | Improvement fraction 0.0334 |

## Visualisations

### Signal and Cochlea

![Model 0 waveform](figures/model0_classical_baseline/attempt_1_signal.png)
![Model 3 cochlea](figures/model3_signal_to_spikes/attempt_1_cochlea.png)
![Model 5 binaural cochlea](figures/model5_binaural_localisation/attempt_1_cochlea.png)
![Pathway split cochlea](figures/model6_pathway_split/attempt_2_cochlea.png)

### Spiking and Pathway Activity

![Model 4 hidden spikes](figures/model4_full_pipeline_trainable/attempt_1_hidden_spikes.png)
![Pathway distance tuning](figures/model6_pathway_split/attempt_2_distance_pathway.png)
![Pathway azimuth tuning](figures/model6_pathway_split/attempt_2_azimuth_pathway.png)
![Pathway elevation activity](figures/model6_pathway_split/attempt_2_elevation_pathway.png)
![Pathway fusion spikes](figures/model6_pathway_split/attempt_2_fusion_spikes.png)

### Predictions and Optimisation

![Model 6 baseline elevation](figures/model6_full_3d_localisation/attempt_3_elevation_prediction.png)
![Pathway split comparison](figures/model6_pathway_split/attempt_2_comparison.png)
![Pathway distance prediction](figures/model6_pathway_split/attempt_2_distance_prediction.png)
![Pathway azimuth prediction](figures/model6_pathway_split/attempt_2_azimuth_prediction.png)
![Pathway elevation prediction](figures/model6_pathway_split/attempt_2_elevation_prediction.png)
![Enhanced Optuna history](figures/model7_enhanced_optuna/optimization_history.png)
![Enhanced Optuna importance](figures/model7_enhanced_optuna/parameter_importance.png)
![Enhanced Optuna summary](figures/model7_enhanced_optuna/summary.png)

## Results Analysis

- The explicit pathway split improved biological clarity by separating pulse-echo timing, binaural directional processing, and spectral elevation cues.
- The original Model 6 baseline achieved combined error 0.1228.
- The pathway-split Model 6 achieved combined error 0.0921, distance MAE 0.0643, azimuth MAE 4.3465, and elevation MAE 6.8544.
- The new architecture therefore improved the previous Model 6 combined error.
- The pathway split reduced combined error by approximately 25.00% relative to the previous Model 6.
- The enhanced Optuna run used persistent SQLite storage at `optuna_study.db`, selected the best completed trial by combined localisation error, and retained a dashboard command in `outputs/run_optuna_dashboard.sh`.
- The best enhanced study result was combined error 0.089, distance MAE n/a, azimuth MAE n/a, elevation MAE n/a, with spike rate 0.2889.

## Optuna Configuration

- Storage: `sqlite:///optuna_study.db`
- Study name: `pathway_split_enhanced`
- Dashboard command: `optuna-dashboard optuna_study.db`
- Selection rule: best completed trial by combined localisation error, while Optuna objective includes spike-rate regularisation.

## Failure Analysis

- The initial Model 7 optimisation plateaued at about 5.31% improvement because the baseline Model 6 configuration was already strong and the search space was too narrow.
- The enhanced Optuna study expanded channel count, delay line count, membrane parameters, reset mechanism, encoding parameters, hidden size, and loss weighting.
- The enhanced study did not meet the >10% improvement target, with improvement fraction 0.0334.
- The remaining plateau suggests the feature extractor and readout now dominate performance more than the coarse hyperparameters; further gains likely require richer elevation cues, longer training, or a more expressive fusion head rather than only wider sweeps.

## Future Work

- Replace the current mixed regression head with full multi-task uncertainty-aware regression.
- Add resonant or adaptive neurons to better model delay selectivity.
- Use measured HRTFs or bat pinna impulse responses for richer elevation cues.
- Deploy the cochlea and pathway split model in an online real-time localisation loop.
