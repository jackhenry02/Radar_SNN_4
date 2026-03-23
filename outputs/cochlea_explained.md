# Cochlea Explained

## Overview

This document describes the current fixed cochlea front end used by the localisation system. The example figures are generated from one clean left-ear echo scene so the transformations are easy to inspect.

Example scene:
- Distance: `1.40 m`
- Azimuth: `18.0 deg`
- Elevation: `12.0 deg`
- Binaural simulation: `on`
- Noise: `off` for clarity

## Pipeline

```mermaid
graph TD
    A[Receive waveform] --> B[FFT]
    B --> C[Log-spaced Gaussian filterbank]
    C --> D[Inverse FFT per channel]
    D --> E[Half-wave rectification]
    E --> F[Low-pass envelope smoothing]
    F --> G[Temporal downsampling]
    G --> H[LIF spike encoder]
    H --> I[Transmit / receive spike tensors]
```

## Current Fixed Parameters

| Parameter | Value | Role |
| --- | --- | --- |
| `sample_rate_hz` | `256000` | Raw waveform sampling rate |
| `num_cochlea_channels` | `24` | Number of frequency channels |
| `cochlea_low_hz` | `20000` | Lowest cochlear center frequency |
| `cochlea_high_hz` | `90000` | Highest cochlear center frequency |
| `filter_bandwidth_sigma` | `0.160` | Width of the Gaussian log-frequency filters |
| `envelope_lowpass_hz` | `1800` | Envelope smoothing cutoff proxy |
| `envelope_downsample` | `4` | Temporal downsampling factor before spiking |
| `spike_threshold` | `0.42` | LIF firing threshold |
| `spike_beta` | `0.88` | LIF leak factor |

## 1. Input Signal

The cochlea receives the left-ear echo waveform. The transmitted chirp is shown alongside it for reference.

![Input spectrogram](cochlea_explained/example_signal.png)
![Transmit vs receive](cochlea_explained/transmit_receive.png)

## 1A. Upstream Elevation Spectral Cue

Before the waveform reaches the cochlea, the simulator applies a fixed elevation-dependent spectral tilt to the echo. This is not part of the cochlea itself, but it is part of the current elevation cue model and is already baked into the cached spike datasets.

Positive elevation boosts higher frequencies relative to lower ones, negative elevation does the opposite, and `0 deg` is flat.

![Elevation spectral cue](cochlea_explained/elevation_spectral_cue.png)

## 2. Log-Spaced Filterbank

The raw waveform is transformed into the frequency domain, multiplied by a bank of Gaussian filters in log-frequency space, and returned to the time domain channel by channel.

![Center frequencies](cochlea_explained/center_frequencies.png)
![Filter responses](cochlea_explained/filter_responses.png)
![Filter heatmap](cochlea_explained/filter_heatmap.png)

## 3. Per-Channel Filtered Signals

After inverse FFT, each channel contains a band-limited version of the original waveform. Low, middle, and high channels respond at different parts of the chirp.

![Filtered channels](cochlea_explained/filtered_channels.png)

## 4. Rectification, Smoothing, And Downsampling

Each channel is half-wave rectified, smoothed with a Hann low-pass kernel, and then downsampled. The downsampled smoothed envelope is the actual input to the spike encoder.

![Channel pipeline](cochlea_explained/channel_pipeline.png)
![Low-pass kernel](cochlea_explained/lowpass_kernel.png)

## 5. LIF Spike Encoding

The smoothed envelope is normalized, integrated through a fixed LIF neuron per channel, thresholded, and reset by subtraction. Spikes are therefore driven by envelope peaks in each frequency band.

![Membrane and spikes](cochlea_explained/membrane_spikes.png)

## 6. Final Cochleagram And Spike Raster

The final cochleagram is the smoothed, downsampled envelope across all channels. The spike raster is the binary output that the rest of the localisation system consumes. This figure is zoomed to the actual echo window so the short FM sweep is visible on the millisecond axis.

![Cochleagram and spikes](cochlea_explained/cochleagram_spikes.png)

## 7. Direct-Drive Gain Sweep (24-Channel Reference)

This diagnostic removes propagation and echo attenuation entirely. It drives the matched human-band cochlea directly with the padded transmit chirp and sweeps an effective source level from `0 dB SPL` to `140 dB SPL`.

Assumption used for the x-axis:
- `1x` simulator chirp amplitude is treated as `80 dB SPL`
- each plotted level is converted to amplitude with `gain = 10^((level_dB - 80) / 20)`
- this is only a labeling convention for the direct-drive experiment, not a full physical calibration of the simulator

Both the current normalized spike encoder and the unnormalized variant are shown, so the effect of normalization on level sensitivity is visible directly.

- 24-channel reference first level with spikes, normalized encoder: `0 dB SPL`
- 24-channel reference first level with spikes, unnormalized encoder: `70 dB SPL`

Interpretation:
- If the normalized curve is nearly flat, that means the current encoder is mostly insensitive to absolute input level under direct drive.
- If the unnormalized curve rises only at higher levels, that shows the thresholding regime required for spikes without per-sample renormalization.
- This direct-drive test isolates the cochlea and spike encoder from propagation, attenuation, and additive noise.

![Direct-drive spike count vs level, 24-channel reference](cochlea_explained/direct_drive_gain_sweep_spike_count_vs_level.png)

## 8. Source Level Needed vs Distance (24-Channel Reference)

Using the unnormalized direct-drive threshold above, the figure below projects the source level needed to reach first-spike conditions at the receiver after the simulator attenuation law is applied.

Projection assumptions:
- unnormalized direct-drive first-spike threshold: `70 dB SPL` at the cochlea input
- attenuation model: `0.7 / path_length^2`
- ear spacing: `0.030 m`
- centerline geometry: `azimuth = 0 deg`, `elevation = 0 deg`, binaural path length to one ear
- plotted empirical points are the coarse first-spike source levels observed in the no-normalization distance sweeps

24-channel reference cochlea configuration used for this threshold projection:
- sample rate: `64000 Hz`
- chirp: `18000 Hz -> 2000 Hz` over `0.003 s`
- signal duration: `0.022 s`
- cochlea channels: `24`
- cochlea band: `2000 Hz -> 20000 Hz`
- spacing: `log`
- filter bandwidth sigma: `0.160`
- envelope low-pass: `1800 Hz`
- downsample: `4`
- envelope rate: `16000 Hz`
- spike threshold: `0.42`
- spike beta: `0.88`

Interpretation:
- The smooth curve is the attenuation-only prediction, so it is a lower-complexity estimate rather than a full noisy simulation.
- The empirical points tend to sit on or above the curve because the actual sweeps include noise, waveform structure, and a coarse tested gain grid.
- This makes the graph useful for intuition about how quickly source-level demands rise with range under the current front end.

![Required source level vs distance, 24-channel reference](cochlea_explained/attenuation_threshold_projection.png)

## 9. Direct-Drive Gain Sweep (700-Channel Model)

This reruns the same direct-drive test, but with the matched human-band `700`-channel cochlea used in the dense front-end experiment.

- 700-channel first level with spikes, normalized encoder: `0 dB SPL`
- 700-channel first level with spikes, unnormalized encoder: `70 dB SPL`

Interpretation:
- This is the cleaner threshold result to use if the `700`-channel front end is the intended model of interest.
- Comparing this plot to the 24-channel reference shows whether increasing cochlear resolution changes absolute level sensitivity or mainly changes spike-count scale.

![Direct-drive spike count vs level, 700-channel](cochlea_explained/direct_drive_gain_sweep_700_spike_count_vs_level.png)

## 10. Source Level Needed vs Distance (700-Channel Model)

Using the unnormalized direct-drive threshold from the `700`-channel model, the figure below projects the source level needed to reach first-spike conditions at the receiver after attenuation.

Projection assumptions:
- unnormalized direct-drive first-spike threshold: `70 dB SPL` at the cochlea input
- attenuation model: `0.7 / path_length^2`
- ear spacing: `0.030 m`
- centerline geometry: `azimuth = 0 deg`, `elevation = 0 deg`, binaural path length to one ear
- this curve is attenuation-only; the expanded-space distance sweeps were not rerun at 700 channels, so there is no empirical point overlay here

700-channel cochlea configuration used for this threshold projection:
- sample rate: `64000 Hz`
- chirp: `18000 Hz -> 2000 Hz` over `0.003 s`
- signal duration: `0.022 s`
- cochlea channels: `700`
- cochlea band: `2000 Hz -> 20000 Hz`
- spacing: `log`
- filter bandwidth sigma: `0.160`
- envelope low-pass: `1800 Hz`
- downsample: `4`
- envelope rate: `16000 Hz`
- spike threshold: `0.42`
- spike beta: `0.88`

Interpretation:
- This is the direct attenuation projection for the `700`-channel front end.
- If the threshold is lower than the 24-channel reference, the dense cochlea is effectively more sensitive to weak direct-drive inputs under the same LIF parameters.
- If the threshold is unchanged, then the extra frequency resolution is mostly changing representation richness rather than minimum spike-onset level.

![Required source level vs distance, 700-channel](cochlea_explained/attenuation_threshold_projection_700.png)

## Interface To The Rest Of The Model

The current barrier is after spike generation:

- transmit spikes: shape `[batch, channel, time]`
- receive spikes: shape `[batch, ear, channel, time]`

Everything downstream assumes those spike tensors already exist. That makes the current cochlea replaceable, but the easiest swap is another cochlea that preserves the same spike-tensor contract and envelope-rate time base.

## Current Interpretation

This cochlea is fixed and hand-designed. It is not currently trainable. The expensive part is the fixed FFT filterbank plus spike conversion, not the later handcrafted pathway feature extraction.

## Bandwidth And Sampling Experiment

This comparison tests the effect of lowering the acoustic and cochlear bandwidth and reducing the sampling rate on the full short-data combined-all localisation system.

Protocol:
- Ultrasonic baseline: saved short-data round-2 combined-all result using the existing `20 kHz to 90 kHz` cochlea, `80 kHz to 20 kHz` chirp, and `256 kHz` sample rate.
- Human-band analogue: fresh rerun of the same short-data combined-all model with cochlea range `20 Hz to 20 kHz`, sample rate `64 kHz`, and a practical downward FM chirp `18 kHz to 2 kHz`.
- Matched human-band analogue: second fresh rerun with the same `64 kHz` sample rate and `18 kHz to 2 kHz` chirp, but with the cochlea restricted to the active signal band `2 kHz to 20 kHz`.
- The lower chirp edge was not set literally to `20 Hz` because a `3 ms` chirp cannot meaningfully encode 20 Hz content; one 20 Hz period is `50 ms`.
- Same dataset size and training budget: `700 / 150 / 150`, `10` epochs, one thread, no Optuna retuning.
- The original wide human-band result is retained below for comparison; the new matched-band result is additional.

Runtime comparison:
- Ultrasonic baseline total: `1115.77 s`
- Human-band analogue total: `143.36 s`
- Overall speedup: `7.78x`
- Prep speedup: `20.68x`
- Training speedup: `7.05x`
- Matched human-band total: `142.92 s`
- Matched overall speedup: `7.81x`
- Matched prep speedup: `29.47x`
- Matched training speedup: `6.95x`

Accuracy comparison:
- Ultrasonic combined error: `0.0789`
- Human-band combined error: `0.1288`
- Matched human-band combined error: `0.1221`
- Ultrasonic distance / azimuth / elevation: `0.0636 m`, `3.5316 deg`, `5.6846 deg`
- Human-band distance / azimuth / elevation: `0.0890 m`, `7.9520 deg`, `9.2961 deg`
- Matched human-band distance / azimuth / elevation: `0.0946 m`, `7.8027 deg`, `8.4785 deg`
- Ultrasonic Euclidean error: `0.2332 m`
- Human-band Euclidean error: `0.4231 m`
- Matched human-band Euclidean error: `0.3964 m`

Interpretation:
- This is not only a cochlea-bandwidth change. It also reduces raw waveform sampling resolution and moves the chirp into a much lower carrier band.
- The comparison therefore measures the practical effect of a lower-bandwidth, lower-sample-rate auditory front end on the full localisation stack.
- The matched human-band variant specifically tests whether excluding irrelevant sub-2 kHz channels helps once the signal itself only occupies `2 kHz to 18 kHz`.
- Because the downstream model was not retuned for either human-band configuration, both should be treated as direct transfer tests rather than optimized redesigns.

![Bandwidth runtime comparison](cochlea_explained/bandwidth_runtime_comparison.png)
![Bandwidth accuracy comparison](cochlea_explained/bandwidth_accuracy_comparison.png)
![Human-band example signal](cochlea_explained/human_example_signal.png)
![Human-band cochleagram](cochlea_explained/human_cochleagram_spikes.png)
![Matched human-band example signal](cochlea_explained/human_matched_example_signal.png)
![Matched human-band cochleagram](cochlea_explained/human_matched_cochleagram_spikes.png)

## Channel Count And Spacing Experiments

This comparison keeps the matched human-band setup as the baseline and changes one cochlea design variable at a time.

Protocol:
- Matched human-band baseline: `48` cochlea channels, `log` spacing, `2 kHz to 20 kHz` cochlea range, downstream model width `48`.
- Dense-channel variant: same matched human-band setup, but increase only the cochlea front end to `700` channels while keeping the downstream model width fixed at `48` via channel-axis compression at the cochlea boundary.
- Mel-spacing variant: same matched human-band setup and channel count, but replace the log-spaced cochlea with a mel-spaced cochlea.
- Same dataset and training budget for all three: `700 / 150 / 150`, `10` epochs, one thread, no Optuna retuning.

Runtime comparison against the matched human-band baseline:
- Matched log baseline total: `142.92 s`
- Matched log 700-channel total: `336.95 s` (`2.36x` baseline)
- Matched log 700-channel prep / training multipliers: `32.66x`, `1.13x`
- Matched mel total: `170.06 s` (`1.19x` baseline)
- Matched mel prep / training multipliers: `1.01x`, `1.20x`

Accuracy comparison against the matched human-band baseline:
- Matched log baseline combined / Euclidean: `0.1221`, `0.3964 m`
- Matched log baseline distance / azimuth / elevation: `0.0946 m`, `7.8027 deg`, `8.4785 deg`
- Matched log 700-channel combined / Euclidean: `0.1088`, `0.3585 m`
- Matched log 700-channel distance / azimuth / elevation: `0.0859 m`, `7.1216 deg`, `7.5448 deg`
- Matched mel combined / Euclidean: `0.1222`, `0.4004 m`
- Matched mel distance / azimuth / elevation: `0.0812 m`, `7.2090 deg`, `8.8032 deg`

Interpretation:
- The `700`-channel test isolates the cost and benefit of much finer cochlear frequency resolution under the same matched human-band chirp and training budget, without also widening the downstream combined model.
- The mel-spacing test isolates a change in channel placement along frequency while keeping the rest of the front end and downstream model structure fixed.
- In this implementation, the mel-spaced bank uses mel-spaced centers with the same Gaussian FFT filter construction and a bandwidth rescaling so filter width stays comparable across the covered band.

![Matched center frequencies](cochlea_explained/matched_channel_spacing_centers.png)
![Matched runtime comparison](cochlea_explained/matched_channel_spacing_runtime_comparison.png)
![Matched accuracy comparison](cochlea_explained/matched_channel_spacing_accuracy_comparison.png)

## 140 dB Training Tests

These two runs test the original matched human-band round-2 combined-all baseline under a much larger source level, using the same short-data training setup as the saved baseline. The `140 dB` label uses the same convention as above: `1x = 80 dB SPL`, so `140 dB` corresponds to `1000x` transmit gain.

Tested variants:
- Baseline reference: saved matched human-band combined-all run at `1x`, normalized cochlea envelope, combined error `0.1221`
- `140 dB` normalized: transmit gain `1000x`, normalize envelope `True`, combined error `0.1198`
- `140 dB` unnormalized: transmit gain `1000x`, normalize envelope `False`, combined error `0.0522`

Metric comparison:
- Baseline distance / azimuth / elevation / Euclidean: `0.0946 m`, `7.8027 deg`, `8.4785 deg`, `0.3964 m`
- `140 dB` normalized distance / azimuth / elevation / Euclidean: `0.0908 m`, `7.1150 deg`, `8.3037 deg`, `0.4066 m`
- `140 dB` unnormalized distance / azimuth / elevation / Euclidean: `0.0641 m`, `2.5352 deg`, `3.2643 deg`, `0.1459 m`

Runtime comparison:
- Baseline total runtime: `142.92 s`
- `140 dB` normalized total runtime: `153.91 s`
- `140 dB` unnormalized total runtime: `171.20 s`

Interpretation:
- Better `140 dB` variant by combined error: `unnormalized`
- The normalized and unnormalized runs can be compared directly to the saved baseline because they reuse the same dataset size, model family, and training budget.
- This comparison isolates the effect of source level and cochlea-envelope normalization on end-to-end localization, rather than only on front-end spike rasters.

