# Training Improved Experiments Report

## Scope

This report covers the baseline-only training-improvement pass requested before rerunning any non-baseline experiments.
- Baseline model: `pathway_split_enhanced trial 10`
- Dataset mode: `training_improved`
- Dataset split: `3500 / 750 / 750` synthetic scenes (`70% / 15% / 15%` of 5000 total)
- Max epochs: `50`
- Early stopping patience: `10` epochs
- Scheduler: `ReduceLROnPlateau` on validation combined error
- Scheduler patience/factor: `4` / `0.5`
- Backend threads: `1`

The CPU baseline remains the fixed reference for every improved experiment in this report.

## CPU Baseline

- Status: `SUCCESS`
- Executed epochs: `35`
- Early stopped: `True`
- Best epoch: `25`
- Test combined error: `0.0648`
- Test distance MAE: `0.0350 m`
- Test azimuth MAE: `2.5537 deg`
- Test elevation MAE: `5.1750 deg`
- Total runtime: `729.60 s`
- Training runtime: `9.99 s`

![CPU loss](training_improved_experiments/cpu/baseline/loss.png)
![CPU summary](training_improved_experiments/cpu/run_summary.png)

## MPS Baseline

- Status: `FAILED`

```text
NotImplementedError: aten::logspace.out is not currently implemented for the MPS device during cochlea_filterbank center-frequency construction.
```

## Comparison

![Runtime comparison](training_improved_experiments/runtime_comparison.png)

![Metric comparison](training_improved_experiments/metric_comparison.png)

- The MPS pass did not complete, so all non-baseline experiments in this report remain CPU-only comparisons against the CPU baseline.

## CPU Improved Experiments

Each experiment below used the same `3500 / 750 / 750` split, the same `50`-epoch maximum, early stopping, and `ReduceLROnPlateau`, and was judged only against the fixed CPU baseline above.

| Experiment | Combined Error | Distance MAE | Azimuth MAE | Elevation MAE | Epochs | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| Improved Experiment 1: Residual Learned Elevation | 0.0639 | 0.0338 | 2.4254 | 5.1221 | 34 | Yes |
| Improved Experiment 2: Corrected Compound Loss | 0.0648 | 0.0383 | 2.3782 | 5.1312 | 34 | Yes |
| Improved Experiment 3: Corrected Uncertainty Weighting | 0.0642 | 0.0270 | 2.2791 | 5.2784 | 34 | Yes |
| Improved Experiment 4: Distance-Only Resonance | 0.0736 | 0.0745 | 2.8157 | 5.1595 | 27 | No |
| Improved Experiment 5: Elevation SConv Residual | 0.0635 | 0.0400 | 2.5799 | 4.9465 | 24 | Yes |

![CPU experiment comparison](training_improved_experiments/cpu/overall_experiment_comparison.png)

![CPU experiment runtime](training_improved_experiments/cpu/experiment_training_runtime.png)

## Experiment Details

### Improved Experiment 1: Residual Learned Elevation

- Decision: `ACCEPTED`
- Change: Keep handcrafted distance/azimuth features and replace only the elevation branch with a residual learned spectral module.
- Rationale: This targets the weakest cue branch first while preserving the baseline timing and binaural inductive bias.
- Executed epochs: `34`
- Best epoch: `24`
- Early stopped: `True`
- Training time: `3332.87 s`
- Initial learning rate: `0.002818`
- Final learning rate: `0.000352`
- Test combined error: `0.0639`
- Distance MAE: `0.0338 m`
- Azimuth MAE: `2.4254 deg`
- Elevation MAE: `5.1221 deg`
- Delta vs CPU baseline: `-0.0009`
- Delta vs previous improved-experiment run: `-0.0199`

Implemented steps:
- Step 1: keep handcrafted distance and azimuth pathways unchanged.
- Step 2: replace only the elevation branch with a learned residual spectral CNN.
- Step 3: keep the learned residual contribution small at initialization so the baseline pathway still dominates early training.

Remaining follow-up steps:
- Add explicit bandpass and smoothness regularization to the learned spectral filters.
- Warm-start from the stable dataset split if the dev split is still too small.

![Improved Experiment 1: Residual Learned Elevation loss](training_improved_experiments/cpu/improved_experiment_1_learned_features/loss.png)
![Improved Experiment 1: Residual Learned Elevation comparison](training_improved_experiments/cpu/improved_experiment_1_learned_features/comparison.png)
![Improved Experiment 1: Residual Learned Elevation azimuth](training_improved_experiments/cpu/improved_experiment_1_learned_features/test_azimuth_prediction.png)
![Improved Experiment 1: Residual Learned Elevation elevation](training_improved_experiments/cpu/improved_experiment_1_learned_features/test_elevation_prediction.png)

### Improved Experiment 2: Corrected Compound Loss

- Decision: `ACCEPTED`
- Change: Use a compound loss with distance, azimuth, and elevation normalized by their actual sampled ranges rather than by global angular bounds.
- Rationale: The previous compound loss likely collapsed angle learning because its angular terms were too weak relative to distance.
- Executed epochs: `34`
- Best epoch: `24`
- Early stopped: `True`
- Training time: `22.27 s`
- Initial learning rate: `0.003132`
- Final learning rate: `0.000391`
- Test combined error: `0.0648`
- Distance MAE: `0.0383 m`
- Azimuth MAE: `2.3782 deg`
- Elevation MAE: `5.1312 deg`
- Delta vs CPU baseline: `-0.0000`
- Delta vs previous improved-experiment run: `-0.0209`

Implemented steps:
- Step 1: keep the baseline architecture unchanged.
- Step 2: normalize distance by max range, azimuth by 45 deg, and elevation by 30 deg.
- Step 3: reuse the manual task weights only after correcting the term scales.

Remaining follow-up steps:
- Tune the task weights after confirming that the corrected scaling no longer collapses angles.
- Log per-task gradient norms to verify balance directly.

![Improved Experiment 2: Corrected Compound Loss loss](training_improved_experiments/cpu/improved_experiment_2_compound_loss/loss.png)
![Improved Experiment 2: Corrected Compound Loss comparison](training_improved_experiments/cpu/improved_experiment_2_compound_loss/comparison.png)
![Improved Experiment 2: Corrected Compound Loss azimuth](training_improved_experiments/cpu/improved_experiment_2_compound_loss/test_azimuth_prediction.png)
![Improved Experiment 2: Corrected Compound Loss elevation](training_improved_experiments/cpu/improved_experiment_2_compound_loss/test_elevation_prediction.png)

### Improved Experiment 3: Corrected Uncertainty Weighting

- Decision: `ACCEPTED`
- Change: Apply uncertainty weighting only after correcting the task normalization and initialize the uncertainty terms near the successful manual weighting.
- Rationale: This tests uncertainty weighting under a fairer objective instead of asking it to rescue a mis-scaled loss.
- Executed epochs: `34`
- Best epoch: `24`
- Early stopped: `True`
- Training time: `22.12 s`
- Initial learning rate: `0.002818`
- Final learning rate: `0.000352`
- Test combined error: `0.0642`
- Distance MAE: `0.0270 m`
- Azimuth MAE: `2.2791 deg`
- Elevation MAE: `5.2784 deg`
- Delta vs CPU baseline: `-0.0006`
- Delta vs previous improved-experiment run: `-0.0177`

Implemented steps:
- Step 1: use the corrected per-task normalization from Improved Experiment 2.
- Step 2: initialize the uncertainty parameters from the baseline manual weights.
- Step 3: freeze the uncertainty parameters during a short warm-up so the task head stabilizes first.

Remaining follow-up steps:
- Add regularization toward the manual weighting if the learned sigmas drift too early.
- Test a longer warm-up period on the stable split.

![Improved Experiment 3: Corrected Uncertainty Weighting loss](training_improved_experiments/cpu/improved_experiment_3_uncertainty_weighting/loss.png)
![Improved Experiment 3: Corrected Uncertainty Weighting comparison](training_improved_experiments/cpu/improved_experiment_3_uncertainty_weighting/comparison.png)
![Improved Experiment 3: Corrected Uncertainty Weighting azimuth](training_improved_experiments/cpu/improved_experiment_3_uncertainty_weighting/test_azimuth_prediction.png)
![Improved Experiment 3: Corrected Uncertainty Weighting elevation](training_improved_experiments/cpu/improved_experiment_3_uncertainty_weighting/test_elevation_prediction.png)

### Improved Experiment 4: Distance-Only Resonance

- Decision: `REJECTED`
- Change: Confine resonant dynamics to the distance pathway and keep the baseline fusion stage for azimuth and elevation.
- Rationale: The original resonant experiment changed too much at once and likely destabilized all three outputs.
- Executed epochs: `27`
- Best epoch: `17`
- Early stopped: `True`
- Training time: `22.40 s`
- Initial learning rate: `0.002818`
- Final learning rate: `0.000352`
- Test combined error: `0.0736`
- Distance MAE: `0.0745 m`
- Azimuth MAE: `2.8157 deg`
- Elevation MAE: `5.1595 deg`
- Delta vs CPU baseline: `0.0088`
- Delta vs previous improved-experiment run: `-0.0161`

Implemented steps:
- Step 1: keep the handcrafted branch encoder and the baseline fusion head.
- Step 2: apply resonant dynamics only to the distance branch before fusion.
- Step 3: constrain resonance frequency and damping to a narrow range and increase the spike penalty slightly.

Remaining follow-up steps:
- Sweep the resonance band against the echo envelope timescale rather than using a single constrained range.
- Test the resonant block in the distance pathway only on the stable split.

![Improved Experiment 4: Distance-Only Resonance loss](training_improved_experiments/cpu/improved_experiment_4_resonant_neurons/loss.png)
![Improved Experiment 4: Distance-Only Resonance comparison](training_improved_experiments/cpu/improved_experiment_4_resonant_neurons/comparison.png)
![Improved Experiment 4: Distance-Only Resonance azimuth](training_improved_experiments/cpu/improved_experiment_4_resonant_neurons/test_azimuth_prediction.png)
![Improved Experiment 4: Distance-Only Resonance elevation](training_improved_experiments/cpu/improved_experiment_4_resonant_neurons/test_elevation_prediction.png)

### Improved Experiment 5: Elevation SConv Residual

- Decision: `ACCEPTED`
- Change: Move SConv2dLSTM into the elevation pathway as a residual spectral-temporal correction instead of a global fusion addition.
- Rationale: This keeps the baseline timing path intact while giving the elevation branch extra spectral-temporal capacity.
- Executed epochs: `24`
- Best epoch: `14`
- Early stopped: `True`
- Training time: `428.33 s`
- Initial learning rate: `0.002662`
- Final learning rate: `0.000333`
- Test combined error: `0.0635`
- Distance MAE: `0.0400 m`
- Azimuth MAE: `2.5799 deg`
- Elevation MAE: `4.9465 deg`
- Delta vs CPU baseline: `-0.0013`
- Delta vs previous improved-experiment run: `-0.0178`

Implemented steps:
- Step 1: keep the baseline handcrafted pathway encoder as the main route.
- Step 2: add an SConv2dLSTM branch only inside the elevation pathway.
- Step 3: inject the recurrent output as a small residual correction rather than as a new dominant fusion feature.

Remaining follow-up steps:
- Reduce temporal pooling further if CPU budget allows and timing detail is still being blurred.
- Try an ear-specific spectral branch before the recurrent block for stronger elevation cues.

![Improved Experiment 5: Elevation SConv Residual loss](training_improved_experiments/cpu/improved_experiment_5_sconv2dlstm/loss.png)
![Improved Experiment 5: Elevation SConv Residual comparison](training_improved_experiments/cpu/improved_experiment_5_sconv2dlstm/comparison.png)
![Improved Experiment 5: Elevation SConv Residual azimuth](training_improved_experiments/cpu/improved_experiment_5_sconv2dlstm/test_azimuth_prediction.png)
![Improved Experiment 5: Elevation SConv Residual elevation](training_improved_experiments/cpu/improved_experiment_5_sconv2dlstm/test_elevation_prediction.png)

## Experiment Summary

- Accepted experiments under the training-improved regime: Improved Experiment 1: Residual Learned Elevation, Improved Experiment 2: Corrected Compound Loss, Improved Experiment 3: Corrected Uncertainty Weighting, Improved Experiment 5: Elevation SConv Residual
- The baseline remains the best reference unless an experiment beats it on combined error and at least one individual metric.

## Next Step

This report now contains the training-improved baseline and the training-improved CPU experiment results. The next step is to decide whether any of the rejected variants are worth another architectural fix pass.
