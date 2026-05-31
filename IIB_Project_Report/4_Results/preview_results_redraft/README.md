# Results Redraft Figure Previews

These files are inspection previews only. The report source has not been changed.

## Figures

- `01_fixed_pathway_scatter.png`: Fixed pathway raw COM, calibrated COM, and calibrated CANN readouts. Rows compare noise-free and full-noise conditions.

- `02_calibration_context.png`: ILD calibration changes with distance, while the elevation calibration transfers from an isolated sweep to a clean full-3D stress sweep.

- `03_fixed_pathway_failure_curves.png`: Coordinate-dependent error structure of the raw fixed pathways for all three acoustic conditions.

- `04_trained_fusion_delayed_echo_scatter.png`: Overlay of fixed, direct-SNN, and residual-SNN predictions with environmental noise and delayed echo copies.

- `05_residual_feature_ablation.png`: Existing final-model zero-ablation results for all three acoustic conditions.

- `06_training_loss_curves.png`: Training and validation losses combined into one three-panel figure. The baseline panel includes direct inputs only; dots mark retained validation epochs.

## Calibration Figure Interpretation

- Left panel: the calibrated ILD mapping changes with target range. ILD is therefore not a pure azimuth cue in this simulator. This motivates using ITD for the fixed azimuth scalar while retaining the ILD population for trainable fusion.
- Right panel: the inverse-sigmoid elevation calibration fitted on the isolated sweep transfers closely to the clean full-3D stress sweep. This is an empirical result within the tested synthetic cue model; it does not establish environmental-noise robustness or general distance independence.

## Suggested Distance-Pathway Follow-Up

A future causal ablation should compare the full distance pathway, a version without DNLL suppression, a version without sweep facilitation, and a version without either mechanism using shared simulated waveforms. Before interpreting DNLL, revise or bypass the current first-onset VCN output: it removes late events upstream of DNLL and prevents the present implementation from isolating the suppression stage cleanly.

## Training-Loss Availability

The saved result JSON files contain both training and validation loss histories. Figure `06_training_loss_curves.png` combines the direct input-only baselines, direct fusion SNN, and residual fusion SNN. Dashed lines show training loss, solid lines show validation loss, and dots mark retained validation epochs. Only the noise-free and full-noise conditions are plotted for the fusion models.

## Resolution Reference

- Distance population: 180 bins over 0.25--5 m, giving approximately 2.65 cm spacing.
- Azimuth and elevation populations: 91 bins over -45--45 degrees, giving 1 degree spacing.
- Use the phrase `sub-bin precision` for errors below these spacings; this does not establish biological hyperacuity.

## Combined Error Check

- Noise-free: raw `0.0585`, fixed CANN `0.0457`, direct SNN `0.0307`, residual SNN `0.0223`.
- Environmental noise: raw `0.2843`, fixed CANN `0.2626`, direct SNN `0.0627`, residual SNN `0.0548`.
- Noise + delayed echoes: raw `0.2771`, fixed CANN `0.2581`, direct SNN `0.0635`, residual SNN `0.0550`.