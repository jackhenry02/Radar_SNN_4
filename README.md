# Biologically Informed SNN for Active Acoustic Localisation

This repository contains the simulation, pathway experiments, trained readouts,
generated results, and report source for an IIB project on three-dimensional
active acoustic localisation. The project investigates whether simplified
biological auditory computations can provide useful intermediate
representations for a small spiking neural network (SNN).

The simulated system emits a frequency-modulated acoustic call and estimates
target distance, azimuth, and elevation from the returned binaural echo. The
final architecture combines a cochlear front end, interpretable cue-specific
pathways, population-code readouts, and a trainable residual SNN fusion stage.

## Report-Backed Work

The work presented in the final report is primarily developed in:

- `mini_models/`: isolated experiments used to select and explain the cochlear,
  acoustic, and neuron abstractions.
- `distance_pathway/`: time-of-flight pathway, onset processing,
  coincidence detection, suppression, facilitation, and distance population
  readouts.
- `azimuth_pathway/`: interaural time-difference (ITD) and interaural
  level-difference (ILD) pathway experiments and azimuth readouts.
- `elevation_pathway/`: pinna-inspired comb filtering, cochlear spectral
  profiles, DCN-style template populations, and elevation readouts.
- `final_model/`: integrated three-dimensional evaluation, acoustic stress
  tests, trainable fusion readouts, input-only SNN baselines, and final result
  generation.
- `IIB_Project_Report/`: the current LaTeX report, references, report figures,
  and scripts used to regenerate selected figures.

These folders are the best starting point for understanding the submitted
project. Each pathway folder separates experiment scripts, generated outputs,
and generated Markdown reports where applicable.

## Repository Map

### Core Project Folders

- `mini_models/`: standalone component studies and shared utilities for early
  design decisions.
- `distance_pathway/`: development and evaluation of the range-estimation
  pathway.
- `azimuth_pathway/`: development and evaluation of binaural horizontal-angle
  pathways.
- `elevation_pathway/`: development and evaluation of the vertical-angle
  spectral pathway.
- `final_model/`: integrated model experiments and trained baseline
  comparisons.
- `IIB_Project_Report/`: final report source. The redraft chapter files and
  `main.tex` form the current report.

### Supporting and Earlier Development Folders

- `models/`: reusable acoustic and SNN components from earlier development
  rounds.
- `stages/`: earlier staged pipeline implementations and experiment helpers.
- `utils/`: shared helpers used by earlier scripts.
- `outputs/`: generated outputs from earlier root-level experiments.
- `latex_report/`: an earlier report draft and associated figures.
- `trainable_readout/`: preliminary notebook-based readout exploration.
- `notes/`: editing notes and project feedback.
- `figures2/`: additional exploratory figures not central to the final report.

### Archived or Duplicate Material

- `IIB_Project_Report_version_1/`: older report version retained for reference.
- `IIB_Project_Report copy/`: duplicate report working copy retained for
  reference.
- Root-level Python scripts: earlier round-based experiments retained for
  provenance. The final report should be traced through the pathway folders and
  `final_model/` instead.

## Typical Workflow

The repository uses Python 3.11 or later. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`.

Install the Python environment with:

```bash
uv sync
```

Experiments are generally run as individual modules or scripts. For example:

```bash
uv run python distance_pathway/experiments/full_distance_pathway_model.py
uv run python azimuth_pathway/experiments/azimuth_pathway_first_attempt.py
uv run python elevation_pathway/experiments/elevation_pathway_first_attempt.py
uv run python final_model/experiments/trainable_final_readout.py
```

Some experiments are computationally expensive because they simulate acoustic
scenes, generate pathway features, or train SNN readouts. Generated reports and
outputs are retained within the relevant pathway folder where possible.

## Report

The current report is located in `IIB_Project_Report/`. To compile it with a
standard LaTeX installation:

```bash
cd IIB_Project_Report
latexmk -pdf main.tex
```

The report is the authoritative explanation of the final architecture,
evaluation conditions, limitations, and interpretation of the results. The
repository also retains exploratory work to show how the final abstractions
were selected, but not every retained experiment contributes directly to the
submitted model.

## Scope

The final system is a simulated proof of principle rather than a complete
biophysical model of bat hearing or a hardware deployment. It is designed to
test a modelling approach: use known biological computations as inspectable
feature extraction, then apply a small trainable SNN where cue interactions are
difficult to calibrate manually.

## Related Repositories

Multiple code repositories were used during the project. The main pipeline code
is available in [Radar_SNN_4](https://github.com/jackhenry02/Radar_SNN_4).
Complementary repositories are available under the
[jackhenry02 GitHub profile](https://github.com/jackhenry02), including:

- [Radar_SNN_2](https://github.com/jackhenry02/Radar_SNN_2)
- [SNN_testing](https://github.com/jackhenry02/SNN_testing)
- [Radar_SNN](https://github.com/jackhenry02/Radar_SNN)
- [Radar_simulation](https://github.com/jackhenry02/Radar_simulation)
- [Bat_model](https://github.com/jackhenry02/Bat_model)
- [radarsim](https://github.com/jackhenry02/radarsim)
