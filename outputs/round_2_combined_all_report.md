# Round 2 Combined-All Experiment

## Overview

This run combines all of the short-data architectural additions into one model and uses the mixed Cartesian-plus-polar loss. It is intentionally a stress test: if performance improves, the features are stacking constructively; if it degrades, the gains from the individual experiments are not additive.

## Fixed Protocol

- Dataset mode: `combined_small`
- Split: `700 train / 150 validation / 150 test`
- Max epochs: `10`
- Scheduler: `ReduceLROnPlateau` with patience `3` and factor `0.5`
- Device: `cpu`
- Thread cap: `1`

## Reference Models

- Fixed short-data combined baseline: combined `0.0894`, Euclidean `0.2659 m`
- Best individual round-2 model: `Round 2 Experiment 5B: Mixed Cartesian And Polar Loss` with combined `0.0816` and Euclidean `0.2517 m`

## Combined-All Design

Architectural additions active together:
- Adaptive cue tuning from Experiment 1
- Shared corollary-discharge resonance routed both per-pathway and at fusion from Experiments 2A and 2B
- Pre-pathway LIF residual from Experiment 3
- Post-pathway LIF residual from Experiment 4
- Mixed Cartesian-plus-polar loss from Experiment 5B

## Result

- Decision vs fixed short-data baseline: `ACCEPTED`
- Better than best individual round-2 model: `YES`

Polar metrics:
- Combined error: `0.0789`
- Distance MAE: `0.0636 m`
- Azimuth MAE: `3.5316 deg`
- Elevation MAE: `5.6846 deg`

Cartesian metrics:
- Euclidean error: `0.2332 m`
- X / Y / Z MAE: `0.0778`, `0.1001`, `0.1649 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0105`
- Distance MAE delta: `-0.0361`
- Azimuth MAE delta: `-0.6857`
- Elevation MAE delta: `-0.0215`
- Euclidean error delta: `-0.0327 m`

Delta vs best individual round-2 model:
- Combined error delta: `-0.0027`
- Distance MAE delta: `-0.0040`
- Azimuth MAE delta: `-0.7617`
- Elevation MAE delta: `-0.0854`
- Euclidean error delta: `-0.0185 m`

Timing:
- Data prep: `135.98 s`
- Training: `946.07 s`
- Evaluation: `5.06 s`
- Total: `1087.14 s`

## Plots

![Combined-all distance](round_2_combined_all/test_distance_prediction.png)
![Combined-all comparison](round_2_combined_all/comparison.png)
![Combined-all cartesian comparison](round_2_combined_all/cartesian_comparison.png)
![Combined-all coordinate profile](round_2_combined_all/coordinate_error_profile.png)
![Combined-all adaptive delays](round_2_combined_all/adaptive_delay_offsets.png)
![Combined-all adaptive gains](round_2_combined_all/adaptive_gains.png)
![Combined-all resonant tuning](round_2_combined_all/resonant_tuning.png)
![Combined-all resonant spikes](round_2_combined_all/resonant_spikes.png)
![Combined-all pre-pathway spikes](round_2_combined_all/pre_pathway_left_spikes.png)
![Combined-all post-pathway spikes](round_2_combined_all/post_pathway_distance_spikes.png)

## Interpretation

If this model improves on the best individual result, the round-2 changes are largely complementary. If it only beats the fixed baseline but not the best individual variant, then the additions help in isolation but partly compete when stacked. If it loses to both, the short-data improvements are not additive and the combined model is over-complex for this regime.
