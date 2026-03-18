# Improved Experimental Pipeline

- Dataset mode: `dev`
- Fixed baseline reference for every experiment: `pathway_split_enhanced trial 10`

## Protocol

Every improved experiment was compared directly against the same baseline control rather than against the previous improved experiment. Acceptance therefore means that the modified variant beat the fixed baseline on combined error and on at least one individual metric.

## Experiment 0 Control

- Combined error: `0.0803`
- Distance MAE: `0.0527 m`
- Azimuth MAE: `3.3531 deg`
- Elevation MAE: `6.1172 deg`
- Predicted azimuth std: `24.9491`
- Predicted elevation std: `16.7983`

![Improved baseline comparison](improved_experiments/overall_test_comparison.png)
![Prediction spread comparison](improved_experiments/prediction_spread_comparison.png)

## Results Table

| Experiment | Combined Error | Distance MAE | Azimuth MAE | Elevation MAE | Accepted |
| --- | --- | --- | --- | --- | --- |
| Improved Experiment 1: Residual Learned Elevation | 0.0838 | 0.0606 | 3.4263 | 6.3473 | No |
| Improved Experiment 2: Corrected Compound Loss | 0.0857 | 0.0700 | 4.9850 | 6.0337 | No |
| Improved Experiment 3: Corrected Uncertainty Weighting | 0.0819 | 0.0605 | 3.8316 | 6.1289 | No |
| Improved Experiment 4: Distance-Only Resonance | 0.0897 | 0.0727 | 4.0226 | 6.5029 | No |
| Improved Experiment 5: Elevation SConv Residual | 0.0813 | 0.0614 | 3.6317 | 6.0184 | No |

## Detailed Analysis

### Improved Experiment 1: Residual Learned Elevation

- Change: Keep handcrafted distance/azimuth features and replace only the elevation branch with a residual learned spectral module.
- Rationale: This targets the weakest cue branch first while preserving the baseline timing and binaural inductive bias.
- Decision against fixed baseline: `REJECTED`
- Combined error: `0.0838`
- Distance MAE: `0.0606 m`
- Azimuth MAE: `3.4263 deg`
- Elevation MAE: `6.3473 deg`
- Predicted azimuth std: `23.5551`
- Predicted elevation std: `15.7347`
- Delta vs fixed baseline: `0.0035`
- Delta vs original failed version: `-0.1442`

Implemented steps:
- Step 1: keep handcrafted distance and azimuth pathways unchanged.
- Step 2: replace only the elevation branch with a learned residual spectral CNN.
- Step 3: keep the learned residual contribution small at initialization so the baseline pathway still dominates early training.

Remaining follow-up steps:
- Add explicit bandpass and smoothness regularization to the learned spectral filters.
- Warm-start from the stable dataset split if the dev split is still too small.

![Improved Experiment 1: Residual Learned Elevation loss](improved_experiments/improved_experiment_1_learned_features/loss.png)
![Improved Experiment 1: Residual Learned Elevation comparison](improved_experiments/improved_experiment_1_learned_features/comparison.png)
![Improved Experiment 1: Residual Learned Elevation azimuth](improved_experiments/improved_experiment_1_learned_features/test_azimuth_prediction.png)
![Improved Experiment 1: Residual Learned Elevation elevation](improved_experiments/improved_experiment_1_learned_features/test_elevation_prediction.png)

### Improved Experiment 2: Corrected Compound Loss

- Change: Use a compound loss with distance, azimuth, and elevation normalized by their actual sampled ranges rather than by global angular bounds.
- Rationale: The previous compound loss likely collapsed angle learning because its angular terms were too weak relative to distance.
- Decision against fixed baseline: `REJECTED`
- Combined error: `0.0857`
- Distance MAE: `0.0700 m`
- Azimuth MAE: `4.9850 deg`
- Elevation MAE: `6.0337 deg`
- Predicted azimuth std: `23.4237`
- Predicted elevation std: `16.7615`
- Delta vs fixed baseline: `0.0053`
- Delta vs original failed version: `-0.1399`

Implemented steps:
- Step 1: keep the baseline architecture unchanged.
- Step 2: normalize distance by max range, azimuth by 45 deg, and elevation by 30 deg.
- Step 3: reuse the manual task weights only after correcting the term scales.

Remaining follow-up steps:
- Tune the task weights after confirming that the corrected scaling no longer collapses angles.
- Log per-task gradient norms to verify balance directly.

![Improved Experiment 2: Corrected Compound Loss loss](improved_experiments/improved_experiment_2_compound_loss/loss.png)
![Improved Experiment 2: Corrected Compound Loss comparison](improved_experiments/improved_experiment_2_compound_loss/comparison.png)
![Improved Experiment 2: Corrected Compound Loss azimuth](improved_experiments/improved_experiment_2_compound_loss/test_azimuth_prediction.png)
![Improved Experiment 2: Corrected Compound Loss elevation](improved_experiments/improved_experiment_2_compound_loss/test_elevation_prediction.png)

### Improved Experiment 3: Corrected Uncertainty Weighting

- Change: Apply uncertainty weighting only after correcting the task normalization and initialize the uncertainty terms near the successful manual weighting.
- Rationale: This tests uncertainty weighting under a fairer objective instead of asking it to rescue a mis-scaled loss.
- Decision against fixed baseline: `REJECTED`
- Combined error: `0.0819`
- Distance MAE: `0.0605 m`
- Azimuth MAE: `3.8316 deg`
- Elevation MAE: `6.1289 deg`
- Predicted azimuth std: `23.3834`
- Predicted elevation std: `15.8912`
- Delta vs fixed baseline: `0.0016`
- Delta vs original failed version: `-0.1439`

Implemented steps:
- Step 1: use the corrected per-task normalization from Improved Experiment 2.
- Step 2: initialize the uncertainty parameters from the baseline manual weights.
- Step 3: freeze the uncertainty parameters during a short warm-up so the task head stabilizes first.

Remaining follow-up steps:
- Add regularization toward the manual weighting if the learned sigmas drift too early.
- Test a longer warm-up period on the stable split.

![Improved Experiment 3: Corrected Uncertainty Weighting loss](improved_experiments/improved_experiment_3_uncertainty_weighting/loss.png)
![Improved Experiment 3: Corrected Uncertainty Weighting comparison](improved_experiments/improved_experiment_3_uncertainty_weighting/comparison.png)
![Improved Experiment 3: Corrected Uncertainty Weighting azimuth](improved_experiments/improved_experiment_3_uncertainty_weighting/test_azimuth_prediction.png)
![Improved Experiment 3: Corrected Uncertainty Weighting elevation](improved_experiments/improved_experiment_3_uncertainty_weighting/test_elevation_prediction.png)

### Improved Experiment 4: Distance-Only Resonance

- Change: Confine resonant dynamics to the distance pathway and keep the baseline fusion stage for azimuth and elevation.
- Rationale: The original resonant experiment changed too much at once and likely destabilized all three outputs.
- Decision against fixed baseline: `REJECTED`
- Combined error: `0.0897`
- Distance MAE: `0.0727 m`
- Azimuth MAE: `4.0226 deg`
- Elevation MAE: `6.5029 deg`
- Predicted azimuth std: `24.5807`
- Predicted elevation std: `15.9299`
- Delta vs fixed baseline: `0.0093`
- Delta vs original failed version: `-0.1099`

Implemented steps:
- Step 1: keep the handcrafted branch encoder and the baseline fusion head.
- Step 2: apply resonant dynamics only to the distance branch before fusion.
- Step 3: constrain resonance frequency and damping to a narrow range and increase the spike penalty slightly.

Remaining follow-up steps:
- Sweep the resonance band against the echo envelope timescale rather than using a single constrained range.
- Test the resonant block in the distance pathway only on the stable split.

![Improved Experiment 4: Distance-Only Resonance loss](improved_experiments/improved_experiment_4_resonant_neurons/loss.png)
![Improved Experiment 4: Distance-Only Resonance comparison](improved_experiments/improved_experiment_4_resonant_neurons/comparison.png)
![Improved Experiment 4: Distance-Only Resonance azimuth](improved_experiments/improved_experiment_4_resonant_neurons/test_azimuth_prediction.png)
![Improved Experiment 4: Distance-Only Resonance elevation](improved_experiments/improved_experiment_4_resonant_neurons/test_elevation_prediction.png)

### Improved Experiment 5: Elevation SConv Residual

- Change: Move SConv2dLSTM into the elevation pathway as a residual spectral-temporal correction instead of a global fusion addition.
- Rationale: This keeps the baseline timing path intact while giving the elevation branch extra spectral-temporal capacity.
- Decision against fixed baseline: `REJECTED`
- Combined error: `0.0813`
- Distance MAE: `0.0614 m`
- Azimuth MAE: `3.6317 deg`
- Elevation MAE: `6.0184 deg`
- Predicted azimuth std: `23.9038`
- Predicted elevation std: `17.2480`
- Delta vs fixed baseline: `0.0010`
- Delta vs original failed version: `-0.0247`

Implemented steps:
- Step 1: keep the baseline handcrafted pathway encoder as the main route.
- Step 2: add an SConv2dLSTM branch only inside the elevation pathway.
- Step 3: inject the recurrent output as a small residual correction rather than as a new dominant fusion feature.

Remaining follow-up steps:
- Reduce temporal pooling further if CPU budget allows and timing detail is still being blurred.
- Try an ear-specific spectral branch before the recurrent block for stronger elevation cues.

![Improved Experiment 5: Elevation SConv Residual loss](improved_experiments/improved_experiment_5_sconv2dlstm/loss.png)
![Improved Experiment 5: Elevation SConv Residual comparison](improved_experiments/improved_experiment_5_sconv2dlstm/comparison.png)
![Improved Experiment 5: Elevation SConv Residual azimuth](improved_experiments/improved_experiment_5_sconv2dlstm/test_azimuth_prediction.png)
![Improved Experiment 5: Elevation SConv Residual elevation](improved_experiments/improved_experiment_5_sconv2dlstm/test_elevation_prediction.png)

## Summary

- Accepted improved experiments: none
- Because every comparison used the same fixed baseline, the acceptance decisions are directly comparable across experiments.
- If an experiment improved on its previous failed version but still lost to baseline, it should still be treated as a rejected baseline replacement and a useful partial fix only.

## So why is the baseline better?

Mostly because the baseline is the best match to the current synthetic problem.

- The simulator injects cues in a fairly structured way in [acoustics.py](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/models/acoustics.py): distance is driven by echo delay, azimuth by ITD/ILD-like binaural differences, and elevation by a simple spectral shaping cue.
- The baseline in [pathway_snn.py](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/models/pathway_snn.py) is almost a hand-engineered decoder for exactly those cues: delay-bank features for distance, ITD/ILD features for azimuth, and spectral-normalization/notch features for elevation.
- On the `dev` dataset size and short training budget, that strong inductive bias beats higher-capacity variants. The new models had more freedom, but not enough data or tuning to learn those same cue transforms reliably.
- The baseline was also already tuned by the earlier Optuna work, while the new variants were mostly tested under baseline-like training budgets. That is favorable to the baseline.
- The control result matters: its prediction spread stayed close to the target spread, so the harness was not flattening outputs. The degradation came from the added models/losses, not from the evaluation code.

So the strongest interpretation is:

The baseline is best for this repository’s current simulator, dataset size, and training regime.

That is not the same as saying the baseline is universally best. It means the current handcrafted pathway split is the most data-efficient and best matched to the cues we are generating right now. If we move to richer elevation cues, larger datasets, or longer training, that conclusion may change.


## Data

The improved experiments were run on the `dev` dataset split: `512` training samples, `256` validation samples, and `256` test samples, so `1024` synthetic scenes in total. That split definition is in [stages/improvement.py:36](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/stages/improvement.py#L36), and the improved runner defaults to `dev` unless `RADAR_SNN_IMPROVED_EXPERIMENT_DATASET_MODE` is changed in [stages/improved_experiments.py:769](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/stages/improved_experiments.py#L769).

The saved run confirms that exact mode: [outputs/improved_experiments/results.json](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/IIB%20Project%20work/Radar_SNN_4/outputs/improved_experiments/results.json) starts with `"dataset_mode": "dev"`.

So for each improved experiment:
- training used `512` samples
- model selection used `256` validation samples
- final comparison used `256` test samples

All variants used that same fixed split, so the comparisons were like-for-like.

They were trained for 14 epochs only as well!

Adam with no scheduler were used for training as well.

For such little data, it is no surprise the hand tuned baseline did better, but these have very strong results.