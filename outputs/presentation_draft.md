# Presentation Draft

This is a first-pass presentation draft for a non-specialist audience. It is structured slide-by-slide, with suggested visuals and a short speaking goal for each slide.

## Slide 1: Title

**Title:** Building a Bat-Inspired Spiking Neural Network for 3D Sound Localisation

- Goal: state the problem in one sentence.
- Key message: the project uses bat-like sensing and biologically inspired spiking computation to estimate distance, azimuth, and elevation from echoes.

Visual:
- ![Full pipeline](presentation_draft_assets/bat_brain_pipeline_diagram.png)

## Slide 2: Why Spiking Neural Networks?

- Standard neural networks are good at static pattern recognition.
- Echolocation is event-driven, time-critical, and naturally sparse.
- Spiking neural networks give a natural language for:
  - timing
  - coincidence
  - oscillation / resonance
  - biological interpretability

Suggested points:
- spikes represent events rather than continuous activations
- delays and synchrony are central to localization
- the architecture can be interpreted in neural terms rather than just as a black-box regressor

## Slide 3: What A Spike-Based Neuron Does

### LIF neuron

- integrates input current over time
- leaks back toward rest
- spikes when threshold is crossed

Visual:
- ![LIF animation](presentation_draft_assets/lif_neuron.gif)

Equation:

```text
m[t+1] = beta * m[t] + I[t] - theta * s[t]
s[t]   = H(m[t] - theta)
```

Speaking note:
- this is the simplest timing-sensitive building block in the project

## Slide 4: Resonant Neuron

- a resonant neuron does not just integrate
- it prefers certain temporal rhythms
- this makes it useful for echo timing and frequency-selective temporal structure

Visual:
- ![Resonant animation](presentation_draft_assets/resonant_neuron.gif)

Equation:

```text
v[t+1] = alpha * v[t] + u[t] - omega * z[t]
z[t+1] = z[t] + omega * v[t]
```

Speaking note:
- this is roughly RLC-like in spirit: damped, oscillatory, and frequency-tuned

## Slide 5: From Algorithms To Circuits

This is the key bridge in the talk: show that familiar algorithms can be reinterpreted as neuronal structures.

- coincidence bank -> delay-swept cross-correlation
- resonance bank -> frequency-selective temporal decomposition
- LSO/MNTB opponent coding -> signed binaural level comparison
- spike summing -> intensity / range cue

## Slide 6: Coincidence Detection Animation

Key message:
- distance and ITD estimation can be framed as testing multiple delay hypotheses in parallel

Visual:
- ![Coincidence animation](presentation_draft_assets/coincidence_bank.gif)
- ![Jeffress diagram](presentation_draft_assets/jeffress_delay_line_diagram.png)

Equations:

```text
m_i[t+1] = beta_i * m_i[t] + w_tx * x[t-tau_i] + w_echo * y[t] - theta * s_i[t]
c[tau]   = sum_t x[t-tau] * y[t]
```

Speaking note:
- each tuned neuron tests one delay hypothesis
- the best-matching delay is the one that receives transmit and echo input together

## Slide 7: Resonance Bank Animation

Key message:
- a bank of resonators is not a literal DFT, but it behaves like a frequency-selective decomposition of temporal structure

Visual:
- ![Resonance bank animation](presentation_draft_assets/resonance_bank.gif)

Equation link:

```text
resonator bank: state evolves with tuned omega_i
DFT analogy: X[k] = sum_t x[t] * exp(-j 2 pi k t / N)
```

Suggested wording:
- "This is Fourier-like rather than exactly Fourier."

## Slide 8: Biological Azimuth Coding: LSO / MNTB

Key message:
- azimuth is not only about timing
- level differences between ears can also be coded by opponent circuits

Visual:
- ![LSO MNTB diagram](presentation_draft_assets/lso_mntb_diagram.png)

Talking points:
- ipsilateral excitation
- contralateral inhibition via MNTB
- opponent comparison sharpens lateralization cues

## Slide 9: Elevation As Spectral Pattern Analysis

Key message:
- elevation is fundamentally different from distance and ITD
- it depends on spectral shaping rather than coincidence timing

Visual:
- ![Elevation notch pathway](presentation_draft_assets/elevation_notch_pathway_diagram.png)
- ![Moving notch cue](round_3_experiments/round3_experiment_2b_moving_notch_plus_detectors/moving_notch_cue.png)

Talking points:
- early elevation cue was a simple slope
- adding a moving notch made the cue richer
- explicit notch detectors improved elevation further

## Slide 10: The Life Of A Sound Signal

Walk the audience through one sound end-to-end:

1. transmit chirp
2. echo simulator adds delay, attenuation, azimuth asymmetry, and elevation cue
3. cochlea filterbank converts waveform into channel activity
4. spike encoder converts activity into spikes
5. distance / azimuth / elevation pathways compute different cue families
6. fusion SNN combines them into a 3D estimate

Visuals:
- ![Matched-human cochleagram and spikes](cochlea_explained/human_matched_cochleagram_spikes.png)
- ![Current pipeline diagram](presentation_draft_assets/bat_brain_pipeline_diagram.png)

## Slide 11: Building A Bat Brain

Use this slide to map the computational parts to biological interpretations:

- cochlea -> peripheral filtering
- delay-line coincidence bank -> Jeffress-like timing circuit
- ILD opponent coding -> LSO/MNTB-like binaural comparison
- spectral notch detectors -> elevation / pinna-like cue decoding
- resonance bank -> tuned temporal feature extraction
- fusion SNN -> higher integration area

## Slide 12: What Actually Worked

Key experimental findings:

- the front end mattered more than expected
- unnormalized high-amplitude spikes recovered long-range behavior
- richer elevation cues helped a lot
- sine/cosine angle outputs stabilized angle regression
- biologically inspired ILD improved overall performance
- adding trainability helped, but stacking everything did not always help

Visual:
- ![Milestones summary](presentation_draft_assets/milestones_summary.png)

Numbers to cite:
- matched-human baseline combined error: `0.1221`
- 140 dB unnormalized short-range combined error: `0.0522`
- best round-3 combined model (`2B + 3`): combined error `0.0394`
- best round-4 individual model (LSO/MNTB ILD): combined error `0.0407`

## Slide 13: A Useful Failure Story

This is worth including because it shows real scientific debugging.

Observation:
- expanded-space tests initially collapsed badly

Diagnosis:
- per-sample envelope normalization made the front end almost level-invariant
- weak long-range returns were being renormalized upward
- that destroyed amplitude information and could create noisy spike patterns

Fix:
- use a much stronger source level (`140 dB` under the current convention)
- disable the front-end normalization

Result:
- expanded 20 m test improved from combined error `0.6315` to `0.2344`

Useful visuals:
- [Direct-drive spike count vs level](cochlea_explained/direct_drive_gain_sweep_700_spike_count_vs_level.png)
- [140 dB unnormalized cochleagram](cochlea_explained/human_matched_140db_unnormalized_cochleagram_spikes.png)

## Slide 14: Current Best Model

Recommended model to present as the current best overall story:

- Round 3 combined model `2B + 3`
  - moving-notch elevation cue + notch detectors
  - sine/cosine angle regression

If you want the most biologically trainable later variant:
- Round 4 combined model
  - explicit LIF timing replacement
  - LSO/MNTB ILD system
  - spike-sum distance cue
  - per-pathway resonance banks

Suggested message:
- the best pure accuracy and the best biological decomposition are not always exactly the same model

## Slide 15: Next Steps

### Improving the bat model
- better long-range amplitude calibration
- more realistic elevation cues / HRTF-like filtering
- cleaner separation of azimuth and elevation spectral codes
- larger cached datasets with the fixed cochlea front end

### Generalising the model
- replace hand-designed cue modules with learnable but constrained spiking modules
- test broader spatial domains
- test other sensing tasks where timing matters

## Slide 16: Conclusion

Suggested closing message:

- SNNs were useful here not just as a fashionable model class, but because the task itself is about timing, coincidence, oscillation, and sparse events.
- The work showed that biologically inspired structure can genuinely help localization.
- The strongest improvements came from understanding the sensory front end and cue design, not just adding depth.

## Suggested Backup Slides

- exhaustive experiment table: [experiments_summary.md](experiments_summary.md)
- cochlea walkthrough: [cochlea_explained.md](cochlea_explained.md)
- current system explanation: [current_system_explained.md](current_system_explained.md)
- round 3 results: [round_3_experiments_report.md](round_3_experiments_report.md)
- round 4 results: [round_4_experiments_report.md](round_4_experiments_report.md)
