# Controlled Experimental Pipeline

- Dataset mode: `dev`
- Baseline reference: `pathway_split_enhanced trial 10`
- Final selected model: `baseline`

## Overview

The experiments used one fixed dataset split and one shared evaluation protocol. Each candidate was compared against the current accepted model. A change was accepted only if it improved combined localisation error and at least one individual metric.

## Implementation Details

The controlled pipeline was added without replacing the working Model 6 / Model 7 path. The new entrypoint is `experiments.py`, the shared protocol and report generation live in `stages/experiments.py`, and the new model variants live in `models/experimental_variants.py`.

- Experiment 0 control: the tuned pathway-split baseline is evaluated through the same fixed dataset split and the same held-out test metrics used for the later variants.
- Shared dataset protocol: one cached `train/val/test` split is prepared once, using the same synthetic scene generator, the same cochlea/spike front end, and the same target ranges for every experiment.
- Shared acceptance rule: a variant is accepted only if its test combined error is lower than the current accepted model and at least one of distance, azimuth, or elevation MAE is also lower.
- Experiment 1 implementation: `LearnedBranchEncoder` replaces handcrafted pathway extraction with a learnable Conv1d cochlear bank, learnable temporal delay kernels for distance and azimuth, and a Conv2d spectral branch for elevation.
- Experiment 2 implementation: `loss_mode="compound"` keeps the same model but replaces the baseline weighted SmoothL1 objective with separate distance, azimuth, and elevation losses plus the spike penalty.
- Experiment 3 implementation: `TaskUncertaintyWeights` adds three learnable `log_sigma` parameters and uses them to reweight the per-task losses during training.
- Experiment 4 implementation: `use_resonant=True` swaps the standard leaky fusion dynamics for a second-order resonant fusion block with damped oscillatory state updates.
- Experiment 5 implementation: `use_sconv=True` adds an `snn.SConv2dLSTM` branch that reads spectral-temporal source frames and contributes a pooled recurrent context vector at fusion time.
- Training protocol for variants: all new variants are trained from scratch on the same cached split, with the same number of epochs unless an experiment-specific override is stated in the config block.
- Interpretation of flat responses: if Experiment 0 shows broad prediction spread and good error while a candidate collapses, the issue is in the added feature/loss/dynamics change rather than the report or metric code.

## Experiments

| Experiment | Change | Result | Accepted |
| --- | --- | --- | --- |
| Experiment 0: Control (Model 6) | Tuned pathway-split baseline evaluated inside the experiment harness. | combined 0.0803, distance 0.0527 m | Reference |
| Experiment 1: Learned Feature Extraction | Replace handcrafted pathway features with learnable cochlear, delay, and spectral modules. | combined 0.2280, distance 0.0719 m | No |
| Experiment 2: Compound Loss Function | Switch from the baseline weighted SmoothL1 loss to pathway-aligned per-task losses plus spike penalty. | combined 0.2256, distance 0.0400 m | No |
| Experiment 3: Uncertainty Weighting | Replace manual task weights with learnable task uncertainty parameters. | combined 0.2258, distance 0.0456 m | No |
| Experiment 4: Resonant Neurons | Introduce damped resonant spiking dynamics in the fusion stage. | combined 0.1996, distance 0.1199 m | No |
| Experiment 5: SConv2dLSTM Augmentation | Augment the current model with an snnTorch SConv2dLSTM spectral-temporal context path. | combined 0.1060, distance 0.0982 m | No |

## Experiment 0 Control

- Test combined error: `0.0803`
- Test distance MAE: `0.0527 m`
- Test azimuth MAE: `3.3531 deg`
- Test elevation MAE: `6.1172 deg`
- Test spike rate: `0.2557`
- Predicted distance std: `0.5992`
- Predicted azimuth std: `24.9491`
- Predicted elevation std: `16.7983`
- Target distance std: `0.5976`
- Target azimuth std: `24.9943`
- Target elevation std: `17.6456`

The control is included to check whether the edited experimental harness itself is collapsing to flat predictions. If the control retains substantial prediction variance and low error, then the flat-response issue is specific to the modified variants rather than the evaluation stack.

Proposed solution:
- Keep Experiment 0 as a required sanity check for every future run.
- Add automatic guards that flag a candidate when prediction standard deviation collapses far below the target standard deviation.

![Experiment 0 loss](experiments/baseline/loss.png)
![Experiment 0 distance](experiments/baseline/test_distance_prediction.png)
![Experiment 0 azimuth](experiments/baseline/test_azimuth_prediction.png)
![Experiment 0 elevation](experiments/baseline/test_elevation_prediction.png)
![Experiment 0 azimuth error](experiments/baseline/test_azimuth_error.png)
![Experiment 0 elevation error](experiments/baseline/test_elevation_error.png)

![Overall test comparison](experiments/overall_test_comparison.png)
![Overall spike rate](experiments/overall_spike_rate.png)

## Detailed Analysis

### Experiment 1: Learned Feature Extraction

- Change: Replace handcrafted pathway features with learnable cochlear, delay, and spectral modules.
- Why it should help: Learnable filters and temporal kernels may adapt better to the synthetic echo statistics than fixed heuristics.
- Compared against: `baseline`
- Decision: `REJECTED`
- Test combined error: `0.2280`
- Test distance MAE: `0.0719 m`
- Test azimuth MAE: `21.3432 deg`
- Test elevation MAE: `15.2887 deg`
- Test spike rate: `0.1371`
- Combined delta vs reference: `0.1477`

Failure analysis:
- Likely cause: the learnable front end removed the handcrafted binaural and spectral inductive bias all at once, while keeping a small dataset and short training schedule. The Conv1d/Conv2d replacement is therefore likely learning coarse energy structure rather than stable ITD and spectral-elevation cues.
- Assessment: mostly an implementation issue. This is not strong evidence that learned feature extraction is a bad idea in principle; it is strong evidence that this specific low-data replacement strategy degrades localisation.

Proposed solution:
- Keep the pathway split but replace only one handcrafted block at a time, starting with the elevation branch rather than all branches simultaneously.
- Initialize the learned cochlear and delay filters from the handcrafted templates, freeze them for a short warm-up period, then unfreeze gradually.
- Add structural regularizers so the learned filters stay bandpass and the delay kernels stay smooth and localized.
- Re-run with the `stable` dataset split and longer training, because the current learned front end is too data-hungry for the `dev` setup.

![Experiment 1: Learned Feature Extraction loss](experiments/experiment_1_learned_features/loss.png)
![Experiment 1: Learned Feature Extraction comparison](experiments/experiment_1_learned_features/comparison.png)
![Experiment 1: Learned Feature Extraction distance](experiments/experiment_1_learned_features/test_distance_prediction.png)
![Experiment 1: Learned Feature Extraction azimuth](experiments/experiment_1_learned_features/test_azimuth_prediction.png)
![Experiment 1: Learned Feature Extraction elevation](experiments/experiment_1_learned_features/test_elevation_prediction.png)

#### Learned Feature Analysis

The learned front end is compared against the log-spaced initialization, which acts as the handcrafted reference template.

![Experiment 1: Learned Feature Extraction filter kernels](experiments/experiment_1_learned_features/filter_kernels.png)
![Experiment 1: Learned Feature Extraction frequency response](experiments/experiment_1_learned_features/filter_frequency_response.png)
![Experiment 1: Learned Feature Extraction delay kernels](experiments/experiment_1_learned_features/delay_kernels.png)

### Experiment 2: Compound Loss Function

- Change: Switch from the baseline weighted SmoothL1 loss to pathway-aligned per-task losses plus spike penalty.
- Why it should help: Separating distance, azimuth, and elevation losses should align optimization with the architecture split.
- Compared against: `baseline`
- Decision: `REJECTED`
- Test combined error: `0.2256`
- Test distance MAE: `0.0400 m`
- Test azimuth MAE: `21.3522 deg`
- Test elevation MAE: `15.2982 deg`
- Test spike rate: `0.1958`
- Combined delta vs reference: `0.1453`

Failure analysis:
- Likely cause: the compound loss underweighted angular terms. In the current implementation both angles were normalized by `180 deg`, even though the actual sampled ranges are much narrower, so the optimizer could improve distance while largely ignoring azimuth and elevation.
- Assessment: implementation issue. This is not a strong negative finding against compound losses; it is a strong finding that this particular scaling choice damages angular learning.

Proposed solution:
- Rescale the angular losses by the actual sampled target ranges, for example azimuth by `45 deg` and elevation by `30 deg`, instead of `180 deg`.
- Tune `lambda_d`, `lambda_a`, and `lambda_e` explicitly after the normalization is corrected.
- Log per-task gradient magnitudes during training to verify that azimuth and elevation are still receiving meaningful updates.

![Experiment 2: Compound Loss Function loss](experiments/experiment_2_compound_loss/loss.png)
![Experiment 2: Compound Loss Function comparison](experiments/experiment_2_compound_loss/comparison.png)
![Experiment 2: Compound Loss Function distance](experiments/experiment_2_compound_loss/test_distance_prediction.png)
![Experiment 2: Compound Loss Function azimuth](experiments/experiment_2_compound_loss/test_azimuth_prediction.png)
![Experiment 2: Compound Loss Function elevation](experiments/experiment_2_compound_loss/test_elevation_prediction.png)

### Experiment 3: Uncertainty Weighting

- Change: Replace manual task weights with learnable task uncertainty parameters.
- Why it should help: Automatically learned task weighting may improve balance across distance and angular errors.
- Compared against: `baseline`
- Decision: `REJECTED`
- Test combined error: `0.2258`
- Test distance MAE: `0.0456 m`
- Test azimuth MAE: `21.4002 deg`
- Test elevation MAE: `15.2586 deg`
- Test spike rate: `0.2064`
- Combined delta vs reference: `0.1455`

Failure analysis:
- Likely cause: uncertainty weighting sat on top of the same loss scaling used in Experiment 2. The learned task uncertainties converged to similar values instead of correcting the imbalance, so the angular tasks remained weakly driven.
- Assessment: mostly an implementation issue. This does not strongly reject uncertainty weighting as a method; it shows that the current uncertainty formulation did not rescue the mis-scaled objective.

Proposed solution:
- Apply uncertainty weighting only after fixing the base task normalization used in Experiment 2.
- Initialize the uncertainty parameters near the current successful manual weights rather than starting all tasks equally.
- Delay learning of the uncertainty terms for a few epochs or regularize them more strongly so they do not settle into a weak but balanced solution too early.

![Experiment 3: Uncertainty Weighting loss](experiments/experiment_3_uncertainty_weighting/loss.png)
![Experiment 3: Uncertainty Weighting comparison](experiments/experiment_3_uncertainty_weighting/comparison.png)
![Experiment 3: Uncertainty Weighting distance](experiments/experiment_3_uncertainty_weighting/test_distance_prediction.png)
![Experiment 3: Uncertainty Weighting azimuth](experiments/experiment_3_uncertainty_weighting/test_azimuth_prediction.png)
![Experiment 3: Uncertainty Weighting elevation](experiments/experiment_3_uncertainty_weighting/test_elevation_prediction.png)

### Experiment 4: Resonant Neurons

- Change: Introduce damped resonant spiking dynamics in the fusion stage.
- Why it should help: A second-order resonant fusion layer may sharpen temporal selectivity for pulse-echo timing.
- Compared against: `baseline`
- Decision: `REJECTED`
- Test combined error: `0.1996`
- Test distance MAE: `0.1199 m`
- Test azimuth MAE: `11.0119 deg`
- Test elevation MAE: `15.2186 deg`
- Test spike rate: `0.4094`
- Combined delta vs reference: `0.1193`

Failure analysis:
- Likely cause: the resonant block replaced a stable leaky fusion stage with an uncalibrated second-order dynamical system. The much higher spike rate suggests that it became over-excitable and distorted the precise timing information needed for distance and angle estimation.
- Assessment: mixed. This is a strong negative result for this exact resonant implementation, but not a strong biological conclusion that resonant neurons are unsuitable in general.

Proposed solution:
- Constrain resonance frequency and damping to a narrower biologically plausible range matched to the echo envelope timescale.
- Insert the resonant dynamics only in the distance pathway first, instead of replacing the whole fusion stage.
- Add a stronger spike-rate penalty or explicit stability constraint, since the current resonant block became over-active.

![Experiment 4: Resonant Neurons loss](experiments/experiment_4_resonant_neurons/loss.png)
![Experiment 4: Resonant Neurons comparison](experiments/experiment_4_resonant_neurons/comparison.png)
![Experiment 4: Resonant Neurons distance](experiments/experiment_4_resonant_neurons/test_distance_prediction.png)
![Experiment 4: Resonant Neurons azimuth](experiments/experiment_4_resonant_neurons/test_azimuth_prediction.png)
![Experiment 4: Resonant Neurons elevation](experiments/experiment_4_resonant_neurons/test_elevation_prediction.png)

### Experiment 5: SConv2dLSTM Augmentation

- Change: Augment the current model with an snnTorch SConv2dLSTM spectral-temporal context path.
- Why it should help: An explicit spatiotemporal recurrent branch may capture spectral-temporal cues that the simple fusion head misses.
- Compared against: `baseline`
- Decision: `REJECTED`
- Test combined error: `0.1060`
- Test distance MAE: `0.0982 m`
- Test azimuth MAE: `5.1508 deg`
- Test elevation MAE: `7.4746 deg`
- Test spike rate: `0.2063`
- Combined delta vs reference: `0.0257`

Failure analysis:
- Likely cause: the added `SConv2dLSTM` branch appears to preserve some broad spectral-temporal structure, but its pooled context likely blurs the timing precision needed for distance. The earlier state-reset bug was fixed, and the model still underperformed, so the remaining drop is not just a broken run.
- Assessment: moderate negative result for the current integration strategy. This does not reject `SConv2dLSTM` in principle, but it is good evidence that this way of inserting it into fusion hurts accuracy.

Proposed solution:
- Move `SConv2dLSTM` into the elevation or spectral branch instead of fusing its pooled output directly into the global head.
- Preserve more timing detail by avoiding aggressive temporal pooling before the recurrent layer.
- Use a residual integration strategy where the baseline fusion head remains dominant and the recurrent branch only adds a correction term.

![Experiment 5: SConv2dLSTM Augmentation loss](experiments/experiment_5_sconv2dlstm/loss.png)
![Experiment 5: SConv2dLSTM Augmentation comparison](experiments/experiment_5_sconv2dlstm/comparison.png)
![Experiment 5: SConv2dLSTM Augmentation distance](experiments/experiment_5_sconv2dlstm/test_distance_prediction.png)
![Experiment 5: SConv2dLSTM Augmentation azimuth](experiments/experiment_5_sconv2dlstm/test_azimuth_prediction.png)
![Experiment 5: SConv2dLSTM Augmentation elevation](experiments/experiment_5_sconv2dlstm/test_elevation_prediction.png)

## Key Insights

- Accepted changes: none
- Rejected changes: Experiment 1: Learned Feature Extraction, Experiment 2: Compound Loss Function, Experiment 3: Uncertainty Weighting, Experiment 4: Resonant Neurons, Experiment 5: SConv2dLSTM Augmentation
- The main limiting factor remains front-end cue quality rather than readout capacity alone.
- Loss shaping can help balance distance and angle errors, but only when the feature extractor remains stable.
- More expressive temporal layers increase cost quickly, so they need clear metric gains to justify acceptance.

## Updated Best Model

The current best experiment-selected model is `baseline`.

## Recommendations

- Re-run the accepted stack on the `stable` dataset split before treating the gain as robust.
- Improve elevation realism with ear-specific spectral filtering before further expanding the fusion stack.
- If the learned front end is promising, add a regularizer that keeps the filters bandpass and delay kernels smooth.
