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
- The lower chirp edge was not set literally to `20 Hz` because a `3 ms` chirp cannot meaningfully encode 20 Hz content; one 20 Hz period is `50 ms`.
- Same dataset size and training budget: `700 / 150 / 150`, `10` epochs, one thread, no Optuna retuning.

Runtime comparison:
- Ultrasonic baseline total: `1115.77 s`
- Human-band analogue total: `143.36 s`
- Overall speedup: `7.78x`
- Prep speedup: `20.68x`
- Training speedup: `7.05x`

Accuracy comparison:
- Ultrasonic combined error: `0.0789`
- Human-band combined error: `0.1288`
- Ultrasonic distance / azimuth / elevation: `0.0636 m`, `3.5316 deg`, `5.6846 deg`
- Human-band distance / azimuth / elevation: `0.0890 m`, `7.9520 deg`, `9.2961 deg`
- Ultrasonic Euclidean error: `0.2332 m`
- Human-band Euclidean error: `0.4231 m`

Interpretation:
- This is not only a cochlea-bandwidth change. It also reduces raw waveform sampling resolution and moves the chirp into a much lower carrier band.
- The comparison therefore measures the practical effect of a lower-bandwidth, lower-sample-rate auditory front end on the full localisation stack.
- Because the downstream model was not retuned for the human-band configuration, the result should be treated as a direct transfer test rather than an optimized redesign.

![Bandwidth runtime comparison](cochlea_explained/bandwidth_runtime_comparison.png)
![Bandwidth accuracy comparison](cochlea_explained/bandwidth_accuracy_comparison.png)
![Human-band example signal](cochlea_explained/human_example_signal.png)
![Human-band cochleagram](cochlea_explained/human_cochleagram_spikes.png)
