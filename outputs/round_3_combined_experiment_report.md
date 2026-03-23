# Round 3 Combined Experiments

## Overview

This report combines the strongest round-3 building blocks rather than testing them in isolation. The protocol is the same as round 3: matched-human front end, 140 dB under the current convention, unnormalized spike encoding, 0.5-5.0 m range, and the short 700/150/150 split.

## Reference Table

| Model | Combined Error | Distance MAE (m) | Azimuth MAE (deg) | Elevation MAE (deg) | Euclidean (m) |
| --- | --- | --- | --- | --- | --- |
| Round 3 Experiment 0: 140 dB Unnormalized Control | 0.0630 | 0.1377 | 3.4642 | 3.9121 | 0.3145 |
| Round 3 Experiment 1: Trainable LIF Coincidence Detectors | 0.0531 | 0.1075 | 2.3063 | 3.6960 | 0.2482 |
| Round 3 Experiment 2B: Moving-Notch Cue Plus Notch Detectors | 0.0396 | 0.1038 | 2.8270 | 1.9386 | 0.2127 |
| Round 3 Experiment 3: Sine/Cosine Angle Regression | 0.0593 | 0.0770 | 3.3350 | 4.2824 | 0.2706 |
| Round 3 Experiment 3C: Orthogonal Combined Azimuth/Elevation Notches | 0.0469 | 0.1351 | 3.3489 | 2.1144 | 0.2391 |
| Round 3 Combined A: 2B + 3 | 0.0394 | 0.0646 | 2.8595 | 2.5258 | 0.2043 |
| Round 3 Combined B: 3C + 3 | 0.0415 | 0.0680 | 3.1475 | 2.5775 | 0.2177 |
| Round 3 Combined C: Winner + 1 | 0.0483 | 0.0785 | 3.0023 | 3.3191 | 0.2470 |

## Candidate Combined Models

### Round 3 Combined A: 2B + 3

- Description: Combine the accepted elevation moving-notch detector model from 2B with the sine/cosine angle output coding from Experiment 3.
- Decision vs round-3 control: `ACCEPTED`

Metrics:
- Combined error: `0.0394`
- Distance MAE: `0.0646 m`
- Azimuth MAE: `2.8595 deg`
- Elevation MAE: `2.5258 deg`
- Euclidean error: `0.2043 m`

Comparisons:
- vs Round 3 Experiment 0: 140 dB Unnormalized Control: combined delta `-0.0236`, distance delta `-0.0731 m`, azimuth delta `-0.6047 deg`, elevation delta `-1.3863 deg`
- vs Round 3 Experiment 2B: Moving-Notch Cue Plus Notch Detectors: combined delta `-0.0002`, distance delta `-0.0391 m`, azimuth delta `0.0325 deg`, elevation delta `0.5872 deg`
- vs Round 3 Experiment 3: Sine/Cosine Angle Regression: combined delta `-0.0199`, distance delta `-0.0124 m`, azimuth delta `-0.4755 deg`, elevation delta `-1.7566 deg`

![Round 3 Combined A: 2B + 3 loss](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/loss.png)
![Round 3 Combined A: 2B + 3 comparison](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/comparison.png)
![Round 3 Combined A: 2B + 3 distance](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/test_distance_prediction.png)
![Round 3 Combined A: 2B + 3 coordinate profile](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/coordinate_error_profile.png)
![Round 3 Combined A: 2B + 3 moving notch cue](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/moving_notch_cue.png)
![Round 3 Combined A: 2B + 3 elevation notch detectors](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/notch_detector_response.png)
![Round 3 Combined A: 2B + 3 angle norms](round_3_combined_experiments/round3_combined_experiment_2b_plus_3/angle_norms.png)

### Round 3 Combined B: 3C + 3

- Description: Combine the accepted orthogonal azimuth/elevation notch model from 3C with the sine/cosine angle output coding from Experiment 3.
- Decision vs round-3 control: `ACCEPTED`

Metrics:
- Combined error: `0.0415`
- Distance MAE: `0.0680 m`
- Azimuth MAE: `3.1475 deg`
- Elevation MAE: `2.5775 deg`
- Euclidean error: `0.2177 m`

Comparisons:
- vs Round 3 Experiment 0: 140 dB Unnormalized Control: combined delta `-0.0216`, distance delta `-0.0697 m`, azimuth delta `-0.3167 deg`, elevation delta `-1.3346 deg`
- vs Round 3 Experiment 3C: Orthogonal Combined Azimuth/Elevation Notches: combined delta `-0.0054`, distance delta `-0.0671 m`, azimuth delta `-0.2014 deg`, elevation delta `0.4631 deg`
- vs Round 3 Experiment 3: Sine/Cosine Angle Regression: combined delta `-0.0178`, distance delta `-0.0090 m`, azimuth delta `-0.1875 deg`, elevation delta `-1.7050 deg`

![Round 3 Combined B: 3C + 3 loss](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/loss.png)
![Round 3 Combined B: 3C + 3 comparison](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/comparison.png)
![Round 3 Combined B: 3C + 3 distance](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/test_distance_prediction.png)
![Round 3 Combined B: 3C + 3 coordinate profile](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/coordinate_error_profile.png)
![Round 3 Combined B: 3C + 3 moving notch cue](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/moving_notch_cue.png)
![Round 3 Combined B: 3C + 3 azimuth moving notch cue](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/azimuth_moving_notch_cue.png)
![Round 3 Combined B: 3C + 3 orthogonal elevation detectors](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/orthogonal_elevation_response.png)
![Round 3 Combined B: 3C + 3 orthogonal azimuth detectors](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/orthogonal_azimuth_response.png)
![Round 3 Combined B: 3C + 3 angle norms](round_3_combined_experiments/round3_combined_experiment_3c_plus_3/angle_norms.png)

## Winner Selection

- Winner by combined error: `Round 3 Combined A: 2B + 3`
- Winner combined error: `0.0394`
- Winner Euclidean error: `0.2043 m`

## Full Combined Model

### Round 3 Combined C: Winner + 1

- Decision vs round-3 control: `ACCEPTED`

Metrics:
- Combined error: `0.0483`
- Distance MAE: `0.0785 m`
- Azimuth MAE: `3.0023 deg`
- Elevation MAE: `3.3191 deg`
- Euclidean error: `0.2470 m`

Comparisons:
- vs Round 3 Experiment 0: 140 dB Unnormalized Control: combined delta `-0.0147`, distance delta `-0.0592 m`, azimuth delta `-0.4619 deg`, elevation delta `-0.5929 deg`
- vs Round 3 Experiment 1: Trainable LIF Coincidence Detectors: combined delta `-0.0048`, distance delta `-0.0290 m`, azimuth delta `0.6960 deg`, elevation delta `-0.3769 deg`
- vs Round 3 Experiment 3: Sine/Cosine Angle Regression: combined delta `-0.0110`, distance delta `0.0015 m`, azimuth delta `-0.3327 deg`, elevation delta `-0.9633 deg`
- vs Round 3 Combined A: 2B + 3: combined delta `0.0089`, distance delta `0.0139 m`, azimuth delta `0.1428 deg`, elevation delta `0.7934 deg`
- vs Round 3 Experiment 2B: Moving-Notch Cue Plus Notch Detectors: combined delta `0.0087`, distance delta `-0.0253 m`, azimuth delta `0.1753 deg`, elevation delta `1.3806 deg`

![Round 3 Combined C: Winner + 1 loss](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/loss.png)
![Round 3 Combined C: Winner + 1 comparison](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/comparison.png)
![Round 3 Combined C: Winner + 1 distance](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/test_distance_prediction.png)
![Round 3 Combined C: Winner + 1 coordinate profile](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/coordinate_error_profile.png)
![Round 3 Combined C: Winner + 1 moving notch cue](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/moving_notch_cue.png)
![Round 3 Combined C: Winner + 1 elevation notch detectors](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/notch_detector_response.png)
![Round 3 Combined C: Winner + 1 LIF distance spikes](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/lif_distance_left_spikes.png)
![Round 3 Combined C: Winner + 1 LIF ITD spikes](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/lif_itd_spikes.png)
![Round 3 Combined C: Winner + 1 angle norms](round_3_combined_experiments/round3_combined_experiment_full_winner_plus_1/angle_norms.png)
