# Experiment And Report Handover Summary

This document is a navigation aid for the main experimental folders in this repository. It summarises what each experiment/report is for, what was analysed, and how it fits into the final project. It does not replace the individual reports; use those for equations, parameters, figures, and full numerical results.

The main project development path is:

1. `mini_models`: isolated signal, neuron, and cochlea tests.
2. `distance_pathway`, `azimuth_pathway`, `elevation_pathway`: cue-specific pathway prototypes.
3. `final_model`: integrated 3D localisation pipeline, fixed readouts, trainable fusion, and input-only baselines.

## `mini_models`

Small self-contained experiments used to choose primitives before building the pathway models.

### `mini_models/experiments/neuron_analysis.py`

Report: `mini_models/reports/neuron_analysis.md`

- Compares candidate event/neuron encoders: leaky integrate-and-fire (LIF), resonate-and-fire (RF), and level-crossing encoders.
- Analyses mathematical definitions, micro temporal dynamics, frequency response, spike-rate behaviour, and phase locking.
- Establishes the modelling role of LIF neurons as low-pass coincidence/evidence accumulators, RF neurons as resonant detectors, and level-crossing encoders as change detectors.

### `mini_models/experiments/signal_analysis.py`

Report: `mini_models/reports/signal_analysis.md`

- Visualises the acoustic signal before neural processing: emitted FM call, delayed echo, attenuation, binaural head-shadow, noise, jitter, and elevation filtering.
- Justifies the three main localisation cues used later: time-of-flight for distance, binaural differences for azimuth, and spectral shaping for elevation.
- Compares early elevation-cue options, including simple notches and comb-interference notches.

### `mini_models/experiments/cochlea_analysis.py`

Report: `mini_models/reports/cochlea_analysis.md`

- Compares several cochlear front-end candidates, including FFT/IFFT filtering, time-domain convolutional filterbanks, IIR resonator filterbanks, RF banks, LIF encoding, and level-crossing encoding.
- Analyses runtime, approximate FLOPs, output spike/event counts, spike density, and qualitative cochleagram/raster structure.
- Motivates the final IIR resonator plus LIF cochlear front end, with active-window gating as the main runtime optimisation.

### `mini_models/experiments/final_cochlea_model_analysis.py`

Report: `mini_models/reports/final_cochlea_model_analysis.md`

- Consolidates the selected cochlea model: active-window detection, IIR resonator filterbank, half-wave rectification, and TorchScript LIF spike encoding.
- Documents the filter mathematics, stability, Q-factor behaviour, frequency response, channel scaling, and runtime.
- Includes discussion of why the engineered IIR filterbank is a functional abstraction rather than a full biophysical cochlea.

### `mini_models/common`

- Shared helper code for the mini-model experiments.
- `neurons.py` contains reusable neuron/encoder functions.
- `signals.py` contains signal-generation and acoustic helper functions.
- `plotting.py` contains common plotting utilities.

## `distance_pathway`

Experiments developing the pulse-echo distance pathway, from simple delay coincidence to the final full distance pipeline with SC-style line-attractor readout.

### Report-only / legacy: `simple_coincidence_model.md`

Report: `distance_pathway/reports/simple_coincidence_model.md`

- Introduces the basic distance-pathway idea: compare a corollary-discharge pulse with a delayed echo using a bank of candidate delays.
- Shows simple detector-bank responses and explanatory amplitude-versus-time examples.
- Useful as conceptual background, but not the final implemented distance pathway.

### Report-only / legacy: `accuracy_optimisation_testing.md`

Report: `distance_pathway/reports/accuracy_optimisation_testing.md`

- Benchmarks simplified LIF-inspired, RF-inspired, and binary coincidence-detector scores under clean, noisy, jittered, and combined noise/jitter conditions.
- Separates waveform-input, spiking-input, FM-sweep spiking, and sustained-pitch spiking cases.
- Important caveat: the LIF/RF labels here are detector scores, not full dynamic spiking neuron simulations.

### `distance_pathway/experiments/binary_clean_optimisation.py`

Report: `distance_pathway/reports/binary_clean_optimisation.md`

- Replaces earlier analytic-delay comparisons with a fair raster/event benchmark where methods receive only corollary-discharge rasters, echo rasters, and candidate delay banks.
- Compares raster soft scores, binary detectors, coordinate event accumulators, direct coordinate inputs, bit-packed binary methods, and sparse-stack methods.
- Identifies a practical optimised coincidence detector for clean sweep-like spike inputs.

### `distance_pathway/experiments/cochlea_latency_experiment.py`

Report: `distance_pathway/reports/cochlea_latency_experiment.md`

- Measures latency introduced by the cochlear spike encoder and onset detector.
- Tests refractory LIF onset detection and recalibrates robust latency offsets.
- Provides selected dynamic cochlear LIF settings for later distance-pathway use.

### `distance_pathway/experiments/distance_noise_diagnostics.py`

Report: `distance_pathway/reports/distance_noise_diagnostics.md`

- Diagnoses why the distance pathway behaves under environmental noise.
- Examines shared noisy cochlea outputs, VCN outputs, timing summaries, threshold sweeps, decay sweeps, and per-distance examples.
- Helps tune dynamic threshold and LIF beta choices for robust onset extraction.

### `distance_pathway/experiments/distance_noise_robustness_experiments.py`

Report: `distance_pathway/reports/distance_noise_robustness_experiments.md`

- Tests proposed robustness mechanisms for distance estimation under noisy input.
- Compares recalibrated latency vectors and dynamic cochlea settings.
- Supports the final choice of onset/latency handling used in the distance pathway.

### `distance_pathway/experiments/full_distance_pathway_model.py`

Report: `distance_pathway/reports/full_distance_pathway_model.md`

- Builds the full distance pathway: cochlea, VCN/VNLL onset cleaning, DNLL-style late suppression, corollary discharge, IC coincidence bank, AC topographic map, and SC readout.
- Analyses example stage progressions, clean accuracy, noise robustness, ablations, full-test behaviour, and comparison with older models.
- This is the main standalone distance-pathway implementation before final model integration.

### `distance_pathway/experiments/line_attractor_analysis.py`

Report: `distance_pathway/reports/line_attractor_model.md`

- Develops the finite-line attractor concept used as an SC-style population readout.
- Analyses ring-to-line conversion, boundary correction, Fisher information, recurrent dynamics, E/I balancing, stability, bump dynamics, and noisy decoding.
- Provides the theory base for later CANN/line-attractor readouts.

### `distance_pathway/experiments/finite_line_input_theory.py`

Report: `distance_pathway/reports/finite_line_input_theory.md`

- Studies input mappings into the finite-line attractor, especially Fisher-information-balanced inputs.
- Compares one-block and two-block input forms, diagonal versus reflected Gaussian input matrices, alpha sweeps, rate caps, and local population-vector decoding.
- Motivates the reflected/opponent CANN input family used in the final readouts.

### `distance_pathway/experiments/sc_line_attractor_integration.py`

Report: `distance_pathway/reports/sc_line_attractor_integration.md`

- Integrates the balanced E/I line-attractor readout into the existing distance pathway without changing upstream stages.
- Compares baseline COM readout with CANN/SC readout, including input timing, bump dynamics, alpha sweep, and full-3D failure cases.
- Important implementation point: the attractor receives the completed AC population as its initial input, then runs recurrent readout dynamics.

### `distance_pathway/experiments/final_distance_pipeline_with_attractor.py`

Report: `distance_pathway/reports/final_distance_pipeline_with_attractor.md`

- Consolidates the final distance pipeline with the SC line-attractor readout.
- Documents each stage from echo waveform through cochlea, VCN/VNLL, DNLL, corollary discharge, IC, AC, and SC.
- Provides the final standalone distance-pathway readout comparison and selected attractor parameters.

## `azimuth_pathway`

Experiments developing horizontal localisation from binaural timing and level cues.

### `azimuth_pathway/experiments/azimuth_pathway_first_attempt.py`

Report: `azimuth_pathway/reports/azimuth_pathway_first_attempt.md`

- Builds the first standalone azimuth pathway using ITD and ILD branches.
- ITD branch uses binaural timing/coincidence ideas; ILD branch uses opponent level comparison and a warped synaptic mapping layer.
- Analyses acoustic setup, branch outputs, IC/SC readout, example processing stages, accuracy, and why wider angle ranges are harder.

### `azimuth_pathway/experiments/azimuth_ild_line_attractor.py`

Report: `azimuth_pathway/reports/azimuth_ild_line_attractor.md`

- Tests an ILD-based azimuth pathway with an SC-style line-attractor readout.
- Analyses isolated accuracy, example attractor dynamics, full-3D/noise tests, fixed-distance trends, and ITD-swap comparisons.
- Important conclusion for final results: ILD can work well in isolated sweeps after calibration, but transfers less cleanly to full 3D because level cues vary with distance.

## `elevation_pathway`

Experiments developing vertical localisation from spectral-shape cues.

### `elevation_pathway/experiments/elevation_pathway_first_attempt.py`

Report: `elevation_pathway/reports/elevation_pathway_first_attempt.md`

- Develops the first elevation pathway from comb-filter spectral cues and DCN-style template matching.
- Analyses received-signal PSDs, comb-depth and signal-weighting improvements, lateral inhibition, dynamic wideband inhibition, full-3D tuning, and DCN disinhibitory notch detection.
- Establishes elevation as the most noise-sensitive fixed pathway because it depends on spectral shape rather than direct timing.

### `elevation_pathway/experiments/elevation_line_attractor.py`

Report: `elevation_pathway/reports/elevation_line_attractor.md`

- Adds an SC-style line-attractor readout to the elevation population.
- Compares direct COM and CANN-like readouts in isolated elevation and full-3D tests.
- Adds inverse-sigmoid calibration to correct stable monotonic distortion in the elevation scalar readout.

### `elevation_pathway/experiments/noisy_inhibition_comparison.py`

Report: `elevation_pathway/reports/noisy_inhibition_comparison.md`

- Compares elevation-pathway inhibition/noise settings under noise-free, environmental-noise, and noise-plus-delayed-echo conditions.
- Summarises best parameter settings and summary metrics for each condition.
- Supports the final treatment of noise-vulnerable elevation cues and reliability weighting.

## `final_model`

Integrated 3D localisation experiments combining the acoustic simulator, cochlea, distance/azimuth/elevation pathways, CANN readouts, trainable SNN fusion, and input-only baselines.

### `final_model/experiments/final_model_results.py`

Reports:

- `final_model/reports/final_model_results.md`
- `final_model/reports/final_model_explained.md`

- Runs and explains the integrated fixed-pathway model.
- Documents final pathway choices: distance with cochlea/VCN/DNLL/IC/AC/SC, azimuth with ITD/ILD branches and ITD CANN readout, and elevation with DCN/template population plus calibrated CANN readout.
- Analyses full 3D tests, old-model comparisons, frequency-channel scaling, readout-neuron scaling, and key implementation caveats.

### `final_model/experiments/environment_noise_diagnostics.py`

Report: `final_model/reports/environment_noise_diagnostics.md`

- Documents the environmental-noise definition used in the final experiments.
- Shows waveform examples and low-sample smoke tests of the fixed model under noisy conditions.
- Useful for clarifying that the 50 dB noise floor is call-referenced and inserted before binaural/elevation transformations.

### `final_model/experiments/other_cann_tests.py`

Report: `final_model/reports/other_CANN_tests.md`

- Tests alternative CANN/readout variants on cached final-model pathway populations.
- Compares two-block `[I;0]`, two-block `[I;-I]`, FI diagonal, and FI reflected Gaussian CANN inputs across clean, environmental-noise, and noise-plus-delayed-echo conditions.
- Shows that CANN dynamics are pathway-dependent: useful for some smoothing/calibration cases, but not a universal optimiser.

### `final_model/experiments/other_training_tests.py`

Report: `final_model/reports/other_training_tests.md`

- Tests training variants around the final SNN readout, including normalised angle targets and regular MSE loss.
- Compares whether alternative target/loss choices improve or degrade training behaviour.
- Supports the final encoded-output and loss choices used in the trainable readout.

### `final_model/experiments/trainable_final_readout.py`

Reports:

- `final_model/reports/trainable_final_readout.md`
- `final_model/reports/trainable_final_readout_comparison.md`
- `final_model/reports/trainable_final_readout_train2000_val400_test400.md`
- `final_model/reports/trainable_final_readout_train2000_val400_test400_envnoise50dB.md`
- `final_model/reports/trainable_final_readout_train2000_val400_test400_envnoise50dB_reverb.md`
- smoke reports: `trainable_final_readout_train1_val1_test1_envnoise50dB*.md`

- Trains the final pathway-feature SNN readouts using cached pathway features.
- Compares fixed raw/CANN readouts, direct SNN coordinate prediction, and residual SNN correction from the fixed pathway estimate.
- Analyses training curves, clean/noisy/noise-plus-delayed-echo results, feature importance, and zero-ablation sensitivity.
- This is one of the central final-result experiments: the residual SNN is the strongest overall model in the report.

### `final_model/experiments/trainable_input_baselines.py`

Reports:

- `final_model/reports/trainable_input_baselines.md`
- `final_model/reports/trainable_input_baselines_comparison.md`
- `final_model/reports/trainable_input_baselines_train2000_val400_test400.md`
- `final_model/reports/trainable_input_baselines_train2000_val400_test400_envnoise50dB.md`
- `final_model/reports/trainable_input_baselines_train2000_val400_test400_envnoise50dB_reverb.md`
- smoke report: `trainable_input_baselines_train2_val1_test1.md`

- Trains input-only SNN baselines using raw binaural waveform features and cochlear-raster features.
- Includes both projected and direct input variants to separate feature usefulness from input dimensionality.
- Provides the main control experiment for the report: the structured pathway-feature SNN is compared against SNNs without cue-specific hand-designed pathways.

### `final_model/experiments/report_results_redraft.py`

Report: `final_model/reports/report_results_redraft.md`

- Regenerates and collects figures/tables used in the redrafted Results chapter.
- Summarises clean, environmental-noise, and noise-plus-delayed-echo final metrics for fixed pathways, trainable readouts, and input baselines.
- Useful as the bridge between raw experiment outputs and report-ready figures.

## Suggested Reading Order For A New Student

1. Start with `final_model/reports/final_model_explained.md` for the full architecture.
2. Read `final_model/reports/trainable_final_readout_comparison.md` and `final_model/reports/trainable_input_baselines_comparison.md` for the final claim.
3. Read `mini_models/reports/final_cochlea_model_analysis.md` to understand the cochlear front end.
4. Read the pathway reports:
   - `distance_pathway/reports/final_distance_pipeline_with_attractor.md`
   - `azimuth_pathway/reports/azimuth_pathway_first_attempt.md`
   - `elevation_pathway/reports/elevation_pathway_first_attempt.md`
5. Read `final_model/reports/report_results_redraft.md` for the report-ready figures and result summaries.
6. Use the older/legacy reports only when tracing why a design choice was made.

## Important Caveats

- Several early reports use simplified scoring functions labelled as LIF/RF detectors; these are not always full dynamic spiking neuron simulations.
- The final CANN/readout receives completed pathway population vectors and then runs recurrent dynamics; it is not yet a continuously driven temporal tracker.
- The final trainable SNN uses cached pathway features repeated across its internal timesteps, not raw temporally aligned pathway spike trains.
- The raw waveform and cochlear-raster baselines are controls, not intended final architectures.
- Smoke-test reports with tiny train/validation/test sets exist for reproducibility/debugging and should not be used for performance claims.
