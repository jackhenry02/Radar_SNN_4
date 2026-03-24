# Round 4 Experiments

Round 4 uses the accepted round-3 combined model `2B + 3` as the fixed baseline.

| Experiment | Combined | Distance | Azimuth | Elevation | Euclidean | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Round 4 Experiment 0: 2B + 3 Baseline | 0.0542 | 0.0785 | 3.8763 | 3.6179 | 0.2839 | Control |
| Round 4 Experiment 1: Full LIF Timing Replacement | 0.0458 | 0.0786 | 4.1350 | 2.4979 | 0.2462 | Yes |
| Round 4 Experiment 2A: Shared Pre-Pathway Conv Backbone | 0.1847 | 0.1256 | 9.4365 | 14.8391 | 0.8900 | No |
| Round 4 Experiment 2B: Post-Pathway IC Conv | 0.0523 | 0.0715 | 3.6499 | 3.5883 | 0.2824 | Yes |
| Round 4 Experiment 3: LSO/MNTB ILD System | 0.0407 | 0.0724 | 3.1207 | 2.4737 | 0.2211 | Yes |
| Round 4 Experiment 4: Distance Spike-Sum Cue | 0.0421 | 0.0763 | 2.9143 | 2.7600 | 0.2132 | Yes |
| Round 4 Experiment 5: Per-Pathway Q-Tunable Resonance Banks | 0.0488 | 0.0818 | 3.1792 | 3.3205 | 0.2429 | Yes |

## Baseline

- Combined error: `0.0542`
- Distance MAE: `0.0785 m`
- Azimuth MAE: `3.8763 deg`
- Elevation MAE: `3.6179 deg`
- Euclidean error: `0.2839 m`

## Round 4 Experiment 1: Full LIF Timing Replacement

- Decision: `ACCEPTED`
- Change: Replace the distance pathway and the ITD part of azimuth with explicit LIF coincidence banks, rather than adding them as residuals.
- Combined error: `0.0458`
- Distance MAE: `0.0786 m`
- Azimuth MAE: `4.1350 deg`
- Elevation MAE: `2.4979 deg`
- Euclidean error: `0.2462 m`
- Delta vs baseline: combined `-0.0084`, distance `0.0002 m`, azimuth `0.2587 deg`, elevation `-1.1200 deg`
- Runtime: `912.38 s`

![Round 4 Experiment 1: Full LIF Timing Replacement loss](round_4_experiments/round4_experiment_1_full_lif_timing_replacement/loss.png)
![Round 4 Experiment 1: Full LIF Timing Replacement comparison](round_4_experiments/round4_experiment_1_full_lif_timing_replacement/comparison.png)
![Round 4 Experiment 1: Full LIF Timing Replacement distance](round_4_experiments/round4_experiment_1_full_lif_timing_replacement/test_distance_prediction.png)

## Round 4 Experiment 2A: Shared Pre-Pathway Conv Backbone

- Decision: `REJECTED`
- Change: Add a shared 2D convolutional preprocessing backbone before the three pathways.
- Combined error: `0.1847`
- Distance MAE: `0.1256 m`
- Azimuth MAE: `9.4365 deg`
- Elevation MAE: `14.8391 deg`
- Euclidean error: `0.8900 m`
- Delta vs baseline: combined `0.1305`, distance `0.0472 m`, azimuth `5.5602 deg`, elevation `11.2212 deg`
- Runtime: `761.39 s`

![Round 4 Experiment 2A: Shared Pre-Pathway Conv Backbone loss](round_4_experiments/round4_experiment_2a_shared_prepathway_conv/loss.png)
![Round 4 Experiment 2A: Shared Pre-Pathway Conv Backbone comparison](round_4_experiments/round4_experiment_2a_shared_prepathway_conv/comparison.png)
![Round 4 Experiment 2A: Shared Pre-Pathway Conv Backbone distance](round_4_experiments/round4_experiment_2a_shared_prepathway_conv/test_distance_prediction.png)

## Round 4 Experiment 2B: Post-Pathway IC Conv

- Decision: `ACCEPTED`
- Change: Add a 2D convolutional integration stage after the pathway latents, inspired by an inferior-colliculus-like shared integration area.
- Combined error: `0.0523`
- Distance MAE: `0.0715 m`
- Azimuth MAE: `3.6499 deg`
- Elevation MAE: `3.5883 deg`
- Euclidean error: `0.2824 m`
- Delta vs baseline: combined `-0.0019`, distance `-0.0070 m`, azimuth `-0.2264 deg`, elevation `-0.0295 deg`
- Runtime: `621.96 s`

![Round 4 Experiment 2B: Post-Pathway IC Conv loss](round_4_experiments/round4_experiment_2b_postpathway_ic_conv/loss.png)
![Round 4 Experiment 2B: Post-Pathway IC Conv comparison](round_4_experiments/round4_experiment_2b_postpathway_ic_conv/comparison.png)
![Round 4 Experiment 2B: Post-Pathway IC Conv distance](round_4_experiments/round4_experiment_2b_postpathway_ic_conv/test_distance_prediction.png)

## Round 4 Experiment 3: LSO/MNTB ILD System

- Decision: `ACCEPTED`
- Change: Replace the simple ILD calculation with a more biologically inspired LSO/MNTB style opponent system.
- Combined error: `0.0407`
- Distance MAE: `0.0724 m`
- Azimuth MAE: `3.1207 deg`
- Elevation MAE: `2.4737 deg`
- Euclidean error: `0.2211 m`
- Delta vs baseline: combined `-0.0135`, distance `-0.0061 m`, azimuth `-0.7557 deg`, elevation `-1.1442 deg`
- Runtime: `700.37 s`

![Round 4 Experiment 3: LSO/MNTB ILD System loss](round_4_experiments/round4_experiment_3_bio_ild_lso_mntb/loss.png)
![Round 4 Experiment 3: LSO/MNTB ILD System comparison](round_4_experiments/round4_experiment_3_bio_ild_lso_mntb/comparison.png)
![Round 4 Experiment 3: LSO/MNTB ILD System distance](round_4_experiments/round4_experiment_3_bio_ild_lso_mntb/test_distance_prediction.png)

## Round 4 Experiment 4: Distance Spike-Sum Cue

- Decision: `ACCEPTED`
- Change: Add a simple receive-spike summing cue to the distance pathway so that overall echo strength can contribute to range estimation.
- Combined error: `0.0421`
- Distance MAE: `0.0763 m`
- Azimuth MAE: `2.9143 deg`
- Elevation MAE: `2.7600 deg`
- Euclidean error: `0.2132 m`
- Delta vs baseline: combined `-0.0121`, distance `-0.0021 m`, azimuth `-0.9620 deg`, elevation `-0.8578 deg`
- Runtime: `695.97 s`

![Round 4 Experiment 4: Distance Spike-Sum Cue loss](round_4_experiments/round4_experiment_4_distance_spike_sum/loss.png)
![Round 4 Experiment 4: Distance Spike-Sum Cue comparison](round_4_experiments/round4_experiment_4_distance_spike_sum/comparison.png)
![Round 4 Experiment 4: Distance Spike-Sum Cue distance](round_4_experiments/round4_experiment_4_distance_spike_sum/test_distance_prediction.png)

## Round 4 Experiment 5: Per-Pathway Q-Tunable Resonance Banks

- Decision: `ACCEPTED`
- Change: Add separate resonance banks to distance, azimuth, and elevation, each with its own trainable Q factor and task-specific initialization.
- Combined error: `0.0488`
- Distance MAE: `0.0818 m`
- Azimuth MAE: `3.1792 deg`
- Elevation MAE: `3.3205 deg`
- Euclidean error: `0.2429 m`
- Delta vs baseline: combined `-0.0054`, distance `0.0033 m`, azimuth `-0.6972 deg`, elevation `-0.2973 deg`
- Runtime: `733.62 s`

![Round 4 Experiment 5: Per-Pathway Q-Tunable Resonance Banks loss](round_4_experiments/round4_experiment_5_per_pathway_q_resonance/loss.png)
![Round 4 Experiment 5: Per-Pathway Q-Tunable Resonance Banks comparison](round_4_experiments/round4_experiment_5_per_pathway_q_resonance/comparison.png)
![Round 4 Experiment 5: Per-Pathway Q-Tunable Resonance Banks distance](round_4_experiments/round4_experiment_5_per_pathway_q_resonance/test_distance_prediction.png)

