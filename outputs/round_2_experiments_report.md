# Round 2 Experiments

## Overview

This report tests new ideas against the accepted combined model using only the short-data protocol. The combined small-data run is treated as the fixed control for every comparison, so all acceptance decisions are against the same reference model under the same reduced-data budget.

## Fixed Protocol

- Dataset mode: `combined_small`
- Split: `700 train / 150 validation / 150 test`
- Max epochs: `10`
- Early stopping patience: `10`
- Scheduler: `ReduceLROnPlateau` with patience `3` and factor `0.5`
- Device: `cpu`
- Thread cap: `1`

## Experiment 0 Control

- Source: saved short-data combined run at `outputs/combined_experiment/short_data_1000_result.json`
- Polar combined error: `0.0894`
- Distance MAE: `0.0997 m`
- Azimuth MAE: `4.2173 deg`
- Elevation MAE: `5.7061 deg`
- Cartesian Euclidean error: `0.2659 m`
- X / Y / Z MAE: `0.1118`, `0.1149`, `0.1647 m`
- Runtime: `367.10 s` total, `220.85 s` training

![Baseline distance](combined_experiment/combined_experiment_1235_small_data/test_distance_prediction.png)
![Baseline coordinate error profile](combined_experiment/combined_experiment_1235_small_data/coordinate_error_profile.png)

## Results Table

| Experiment | Combined Error | Euclidean Error (m) | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Accepted | Cartesian Improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Round 2 Experiment 1: Adaptive Delay And Spectral Filters | 0.0882 | 0.2645 | 0.0912 | 3.8537 | 5.9913 | Yes | Yes |
| Round 2 Experiment 2A: Resonant Branch At Fusion | 0.0868 | 0.2600 | 0.0900 | 3.9982 | 5.7440 | Yes | Yes |
| Round 2 Experiment 2B: Resonant Branch Per Pathway | 0.0831 | 0.2569 | 0.0838 | 3.9352 | 5.5873 | Yes | Yes |
| Round 2 Experiment 3: Pre-Pathway LIF Residual | 0.0862 | 0.2542 | 0.0744 | 3.8629 | 6.0904 | Yes | Yes |
| Round 2 Experiment 4: Post-Pathway LIF Residual | 0.0843 | 0.2566 | 0.0718 | 3.4273 | 6.1480 | Yes | Yes |
| Round 2 Experiment 5A: Pure Cartesian Loss | 0.0832 | 0.2524 | 0.0772 | 3.7977 | 5.8794 | Yes | Yes |
| Round 2 Experiment 5B: Mixed Cartesian And Polar Loss | 0.0816 | 0.2517 | 0.0676 | 4.2933 | 5.7700 | Yes | Yes |

## Detailed Experiments

### Round 2 Experiment 1: Adaptive Delay And Spectral Filters

- Change: Keep the accepted combined model intact and add constrained trainable delay offsets, delay weights, and spectral-channel gains/offsets as small residual corrections.
- Rationale: This tests the recommendation to learn small deviations around the fixed biologically motivated delays and spectral channels rather than replacing them with fully free filters.
- Loss mode: `corrected_uncertainty`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Use the combined model as the base path.
- Add learnable offset-and-gain transforms to the fixed distance, ITD, and spectral feature axes.
- Keep all residual gains small at initialization so the fixed cue geometry still dominates early training.
- Log the learned offsets and gains so they can be compared directly against the original fixed values.

Analysis focus:
- Whether learned delay offsets stay near zero or move substantially.
- Whether spectral channel gains become strongly non-uniform.
- Whether the constrained adaptive version improves on the saved short-data combined baseline.

Polar metrics:
- Combined error: `0.0882`
- Distance MAE: `0.0912 m`
- Azimuth MAE: `3.8537 deg`
- Elevation MAE: `5.9913 deg`

Cartesian metrics:
- Euclidean error: `0.2645 m`
- X / Y / Z MAE: `0.1002`, `0.1107`, `0.1796 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0012`
- Distance MAE delta: `-0.0085`
- Azimuth MAE delta: `-0.3636`
- Elevation MAE delta: `0.2852`
- Euclidean error delta: `-0.0014 m`

Timing:
- Data prep: `159.23 s`
- Training: `231.21 s`
- Evaluation: `2.01 s`
- Total: `233.23 s`

![Round 2 Experiment 1: Adaptive Delay And Spectral Filters loss](round_2_experiments/round2_experiment_1_adaptive_filters_delays/loss.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters polar comparison](round_2_experiments/round2_experiment_1_adaptive_filters_delays/comparison.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters cartesian comparison](round_2_experiments/round2_experiment_1_adaptive_filters_delays/cartesian_comparison.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters distance](round_2_experiments/round2_experiment_1_adaptive_filters_delays/test_distance_prediction.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters coordinate profile](round_2_experiments/round2_experiment_1_adaptive_filters_delays/coordinate_error_profile.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters adaptive delays](round_2_experiments/round2_experiment_1_adaptive_filters_delays/adaptive_delay_offsets.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters adaptive spectral offsets](round_2_experiments/round2_experiment_1_adaptive_filters_delays/adaptive_spectral_offsets.png)
![Round 2 Experiment 1: Adaptive Delay And Spectral Filters adaptive gains](round_2_experiments/round2_experiment_1_adaptive_filters_delays/adaptive_gains.png)

### Round 2 Experiment 2A: Resonant Branch At Fusion

- Change: Add a corollary-discharge resonant branch in parallel with the current distance/azimuth/elevation latents and fuse it only at the final fusion stage.
- Rationale: This is the least invasive resonance test: the existing pathways stay unchanged while the SNN gets an extra resonant timing summary built from negative transmit drive plus positive echo drive.
- Loss mode: `corrected_uncertainty`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Build signed resonance drives using positive receive activity and negative transmit activity.
- Run the signed drives through a learnable bank of damped oscillatory resonators.
- Project the pooled resonant spikes into one extra latent block and concatenate that at fusion.

Analysis focus:
- Whether the resonant branch improves distance or angular metrics without destabilising the existing latents.
- Whether the learned resonance frequencies cluster in a useful range rather than saturating.

Polar metrics:
- Combined error: `0.0868`
- Distance MAE: `0.0900 m`
- Azimuth MAE: `3.9982 deg`
- Elevation MAE: `5.7440 deg`

Cartesian metrics:
- Euclidean error: `0.2600 m`
- X / Y / Z MAE: `0.0958`, `0.1159`, `0.1719 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0026`
- Distance MAE delta: `-0.0097`
- Azimuth MAE delta: `-0.2191`
- Elevation MAE delta: `0.0379`
- Euclidean error delta: `-0.0059 m`

Timing:
- Data prep: `159.23 s`
- Training: `193.59 s`
- Evaluation: `1.83 s`
- Total: `195.42 s`

![Round 2 Experiment 2A: Resonant Branch At Fusion loss](round_2_experiments/round2_experiment_2a_resonance_fusion/loss.png)
![Round 2 Experiment 2A: Resonant Branch At Fusion polar comparison](round_2_experiments/round2_experiment_2a_resonance_fusion/comparison.png)
![Round 2 Experiment 2A: Resonant Branch At Fusion cartesian comparison](round_2_experiments/round2_experiment_2a_resonance_fusion/cartesian_comparison.png)
![Round 2 Experiment 2A: Resonant Branch At Fusion distance](round_2_experiments/round2_experiment_2a_resonance_fusion/test_distance_prediction.png)
![Round 2 Experiment 2A: Resonant Branch At Fusion coordinate profile](round_2_experiments/round2_experiment_2a_resonance_fusion/coordinate_error_profile.png)
![Round 2 Experiment 2A: Resonant Branch At Fusion resonant tuning](round_2_experiments/round2_experiment_2a_resonance_fusion/resonant_tuning.png)
![Round 2 Experiment 2A: Resonant Branch At Fusion resonant spikes](round_2_experiments/round2_experiment_2a_resonance_fusion/resonant_spikes.png)

### Round 2 Experiment 2B: Resonant Branch Per Pathway

- Change: Add one shared resonant bank but project its output separately into residual corrections for the distance, azimuth, and elevation pathways.
- Rationale: This tests whether the resonance features are more useful when each pathway can consume them with its own learned projection rather than only at the final fusion layer.
- Loss mode: `corrected_uncertainty`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Reuse the same signed resonant input construction as Experiment 2A.
- Project the pooled resonant spikes separately into distance, azimuth, and elevation residuals.
- Inject the resonant pathway residuals with small learned gains.

Analysis focus:
- Whether pathway-specific resonance routing is better than fusion-only resonance.
- Whether any one pathway benefits disproportionately from the resonant information.

Polar metrics:
- Combined error: `0.0831`
- Distance MAE: `0.0838 m`
- Azimuth MAE: `3.9352 deg`
- Elevation MAE: `5.5873 deg`

Cartesian metrics:
- Euclidean error: `0.2569 m`
- X / Y / Z MAE: `0.1111`, `0.1018`, `0.1681 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0063`
- Distance MAE delta: `-0.0159`
- Azimuth MAE delta: `-0.2821`
- Elevation MAE delta: `-0.1188`
- Euclidean error delta: `-0.0090 m`

Timing:
- Data prep: `159.23 s`
- Training: `224.19 s`
- Evaluation: `1.88 s`
- Total: `226.07 s`

![Round 2 Experiment 2B: Resonant Branch Per Pathway loss](round_2_experiments/round2_experiment_2b_resonance_per_pathway/loss.png)
![Round 2 Experiment 2B: Resonant Branch Per Pathway polar comparison](round_2_experiments/round2_experiment_2b_resonance_per_pathway/comparison.png)
![Round 2 Experiment 2B: Resonant Branch Per Pathway cartesian comparison](round_2_experiments/round2_experiment_2b_resonance_per_pathway/cartesian_comparison.png)
![Round 2 Experiment 2B: Resonant Branch Per Pathway distance](round_2_experiments/round2_experiment_2b_resonance_per_pathway/test_distance_prediction.png)
![Round 2 Experiment 2B: Resonant Branch Per Pathway coordinate profile](round_2_experiments/round2_experiment_2b_resonance_per_pathway/coordinate_error_profile.png)
![Round 2 Experiment 2B: Resonant Branch Per Pathway resonant tuning](round_2_experiments/round2_experiment_2b_resonance_per_pathway/resonant_tuning.png)
![Round 2 Experiment 2B: Resonant Branch Per Pathway resonant spikes](round_2_experiments/round2_experiment_2b_resonance_per_pathway/resonant_spikes.png)

### Round 2 Experiment 3: Pre-Pathway LIF Residual

- Change: Insert an extra learnable LIF preprocessing stage on the spike trains before rebuilding the fixed pathway features, then add the resulting features back as small residuals.
- Rationale: This tests whether another spiking preprocessing step can improve feature extraction without replacing the strong handcrafted cue pathways.
- Loss mode: `corrected_uncertainty`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Process transmit and receive spike trains with learned 1x1 channel mixing and LIF neurons.
- Rebuild distance, azimuth, and elevation pathway features from the processed spikes.
- Project the rebuilt features into residual pathway corrections rather than replacing the baseline pathways.

Analysis focus:
- Whether early spiking preprocessing sharpens the pathways or blurs their timing structure.
- Whether the learned residual stays small or dominates the baseline cues.

Polar metrics:
- Combined error: `0.0862`
- Distance MAE: `0.0744 m`
- Azimuth MAE: `3.8629 deg`
- Elevation MAE: `6.0904 deg`

Cartesian metrics:
- Euclidean error: `0.2542 m`
- X / Y / Z MAE: `0.0892`, `0.1047`, `0.1754 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0032`
- Distance MAE delta: `-0.0253`
- Azimuth MAE delta: `-0.3544`
- Elevation MAE delta: `0.3843`
- Euclidean error delta: `-0.0117 m`

Timing:
- Data prep: `159.23 s`
- Training: `911.54 s`
- Evaluation: `4.56 s`
- Total: `916.10 s`

![Round 2 Experiment 3: Pre-Pathway LIF Residual loss](round_2_experiments/round2_experiment_3_pre_pathway_lif/loss.png)
![Round 2 Experiment 3: Pre-Pathway LIF Residual polar comparison](round_2_experiments/round2_experiment_3_pre_pathway_lif/comparison.png)
![Round 2 Experiment 3: Pre-Pathway LIF Residual cartesian comparison](round_2_experiments/round2_experiment_3_pre_pathway_lif/cartesian_comparison.png)
![Round 2 Experiment 3: Pre-Pathway LIF Residual distance](round_2_experiments/round2_experiment_3_pre_pathway_lif/test_distance_prediction.png)
![Round 2 Experiment 3: Pre-Pathway LIF Residual coordinate profile](round_2_experiments/round2_experiment_3_pre_pathway_lif/coordinate_error_profile.png)
![Round 2 Experiment 3: Pre-Pathway LIF Residual pre-pathway spikes](round_2_experiments/round2_experiment_3_pre_pathway_lif/pre_pathway_left_spikes.png)

### Round 2 Experiment 4: Post-Pathway LIF Residual

- Change: Add one extra LIF processing block on each pathway latent after the current branch encoders and before fusion so the model can do deeper branch-specific processing.
- Rationale: This is the safer depth experiment because it preserves the current pathway feature extraction and only adds extra processing after those cues already exist.
- Loss mode: `corrected_uncertainty`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Keep the accepted combined encoder untouched.
- Add a branch-specific linear-plus-LIF residual block after each pathway latent.
- Feed the updated latents into the existing fusion SNN head.

Analysis focus:
- Whether extra branch-specific processing helps without flattening the learned representation.
- Whether post-pathway depth is safer than pre-pathway depth on the short run.

Polar metrics:
- Combined error: `0.0843`
- Distance MAE: `0.0718 m`
- Azimuth MAE: `3.4273 deg`
- Elevation MAE: `6.1480 deg`

Cartesian metrics:
- Euclidean error: `0.2566 m`
- X / Y / Z MAE: `0.0926`, `0.0942`, `0.1850 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0051`
- Distance MAE delta: `-0.0279`
- Azimuth MAE delta: `-0.7900`
- Elevation MAE delta: `0.4419`
- Euclidean error delta: `-0.0094 m`

Timing:
- Data prep: `159.23 s`
- Training: `219.68 s`
- Evaluation: `1.87 s`
- Total: `221.55 s`

![Round 2 Experiment 4: Post-Pathway LIF Residual loss](round_2_experiments/round2_experiment_4_post_pathway_lif/loss.png)
![Round 2 Experiment 4: Post-Pathway LIF Residual polar comparison](round_2_experiments/round2_experiment_4_post_pathway_lif/comparison.png)
![Round 2 Experiment 4: Post-Pathway LIF Residual cartesian comparison](round_2_experiments/round2_experiment_4_post_pathway_lif/cartesian_comparison.png)
![Round 2 Experiment 4: Post-Pathway LIF Residual distance](round_2_experiments/round2_experiment_4_post_pathway_lif/test_distance_prediction.png)
![Round 2 Experiment 4: Post-Pathway LIF Residual coordinate profile](round_2_experiments/round2_experiment_4_post_pathway_lif/coordinate_error_profile.png)
![Round 2 Experiment 4: Post-Pathway LIF Residual post-pathway spikes](round_2_experiments/round2_experiment_4_post_pathway_lif/post_pathway_distance_spikes.png)

### Round 2 Experiment 5A: Pure Cartesian Loss

- Change: Keep the accepted combined architecture fixed and train it with a pure Cartesian-position loss plus the usual spike penalty.
- Rationale: This tests whether optimizing directly for physical position improves localization in Euclidean space even if the polar-coordinate metrics shift differently.
- Loss mode: `pure_cartesian`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Decode the model output to polar coordinates as usual.
- Convert both prediction and target to Cartesian coordinates.
- Optimize only the normalized Cartesian error and the spike penalty.
- Still report both Cartesian and polar metrics at evaluation time.

Analysis focus:
- Whether Euclidean position error improves relative to the short combined baseline.
- Whether pure Cartesian optimization hurts the angular metrics even if position improves.

Polar metrics:
- Combined error: `0.0832`
- Distance MAE: `0.0772 m`
- Azimuth MAE: `3.7977 deg`
- Elevation MAE: `5.8794 deg`

Cartesian metrics:
- Euclidean error: `0.2524 m`
- X / Y / Z MAE: `0.0946`, `0.1004`, `0.1760 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0062`
- Distance MAE delta: `-0.0225`
- Azimuth MAE delta: `-0.4196`
- Elevation MAE delta: `0.1733`
- Euclidean error delta: `-0.0135 m`

Timing:
- Data prep: `159.23 s`
- Training: `221.90 s`
- Evaluation: `1.79 s`
- Total: `223.68 s`

![Round 2 Experiment 5A: Pure Cartesian Loss loss](round_2_experiments/round2_experiment_5a_cartesian_loss/loss.png)
![Round 2 Experiment 5A: Pure Cartesian Loss polar comparison](round_2_experiments/round2_experiment_5a_cartesian_loss/comparison.png)
![Round 2 Experiment 5A: Pure Cartesian Loss cartesian comparison](round_2_experiments/round2_experiment_5a_cartesian_loss/cartesian_comparison.png)
![Round 2 Experiment 5A: Pure Cartesian Loss distance](round_2_experiments/round2_experiment_5a_cartesian_loss/test_distance_prediction.png)
![Round 2 Experiment 5A: Pure Cartesian Loss coordinate profile](round_2_experiments/round2_experiment_5a_cartesian_loss/coordinate_error_profile.png)

### Round 2 Experiment 5B: Mixed Cartesian And Polar Loss

- Change: Keep the accepted combined architecture fixed and train with a mixed loss made of Cartesian error, polar error regularization, and the spike penalty.
- Rationale: This checks whether a mixed objective can improve physical position without giving up the stable distance, azimuth, and elevation behavior learned by the current polar formulation.
- Loss mode: `mixed_cartesian`
- Decision against fixed short-data baseline: `ACCEPTED`
- Cartesian improvement: `YES`

Implementation details:
- Compute the same normalized Cartesian error as Experiment 5A.
- Add a smaller corrected polar loss term as a regularizer.
- Keep all evaluation outputs in both coordinate systems for a fair comparison to the pure Cartesian run.

Analysis focus:
- Whether mixed optimization is a safer compromise than pure Cartesian optimization.
- Whether the mixed loss retains the baseline angular behavior better than the pure Cartesian loss.

Polar metrics:
- Combined error: `0.0816`
- Distance MAE: `0.0676 m`
- Azimuth MAE: `4.2933 deg`
- Elevation MAE: `5.7700 deg`

Cartesian metrics:
- Euclidean error: `0.2517 m`
- X / Y / Z MAE: `0.0928`, `0.1164`, `0.1646 m`

Delta vs fixed short-data baseline:
- Combined error delta: `-0.0078`
- Distance MAE delta: `-0.0321`
- Azimuth MAE delta: `0.0760`
- Elevation MAE delta: `0.0639`
- Euclidean error delta: `-0.0142 m`

Timing:
- Data prep: `159.23 s`
- Training: `223.48 s`
- Evaluation: `1.90 s`
- Total: `225.38 s`

![Round 2 Experiment 5B: Mixed Cartesian And Polar Loss loss](round_2_experiments/round2_experiment_5b_mixed_cartesian_loss/loss.png)
![Round 2 Experiment 5B: Mixed Cartesian And Polar Loss polar comparison](round_2_experiments/round2_experiment_5b_mixed_cartesian_loss/comparison.png)
![Round 2 Experiment 5B: Mixed Cartesian And Polar Loss cartesian comparison](round_2_experiments/round2_experiment_5b_mixed_cartesian_loss/cartesian_comparison.png)
![Round 2 Experiment 5B: Mixed Cartesian And Polar Loss distance](round_2_experiments/round2_experiment_5b_mixed_cartesian_loss/test_distance_prediction.png)
![Round 2 Experiment 5B: Mixed Cartesian And Polar Loss coordinate profile](round_2_experiments/round2_experiment_5b_mixed_cartesian_loss/coordinate_error_profile.png)

## Summary

- Accepted experiments by the existing polar acceptance rule: Round 2 Experiment 1: Adaptive Delay And Spectral Filters, Round 2 Experiment 2A: Resonant Branch At Fusion, Round 2 Experiment 2B: Resonant Branch Per Pathway, Round 2 Experiment 3: Pre-Pathway LIF Residual, Round 2 Experiment 4: Post-Pathway LIF Residual, Round 2 Experiment 5A: Pure Cartesian Loss, Round 2 Experiment 5B: Mixed Cartesian And Polar Loss
- Cartesian-only improvements are reported separately so the Cartesian-loss runs can be interpreted even if they do not beat the baseline on the original polar rule.
- This round uses the short-data protocol only. Any promising experiment should be rerun later with the longer training regime before it is treated as a genuine model replacement.
