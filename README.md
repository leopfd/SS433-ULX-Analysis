# SS433 Main Pipeline

End-to-end analysis for SS433 HRC observations: fit spatial models, track jet components, fit kinematic models, and compare ejection times with Swift/XRT light curves.

**Pipeline Stages**
1. Fit images: 2D Gaussian centroid + source + multi-component Sherpa fits, optional MCMC.
2. Track components: parse fit logs, re-center on the core, compute PA/radius, and generate tracking plots.
3. Kinematics: fit jet model to component positions and derive ejection times and beta.
4. Swift comparison: compare derived ejection times against Swift/XRT rates.

**Quick Start**
1. Activate a CIAO/Sherpa-enabled Python environment.
2. Update `config.py` to point at your Chandra data and Swift data files (or pass `--base-dir`).
3. Run the pipeline from this directory.

```bash
python pipeline.py
python pipeline.py --obs 26569-26572 --comps 4 --sigma 1 --bin 0.25 --steps 2000
python pipeline.py --no-mcmc --skip-stages swift
```

Run `python pipeline.py --help` for the full CLI.

**Configuration**
- `config.py` sets default paths, ephemeris parameters, and output naming.
- `--base-dir` overrides `BASE_DIR` in `config.py`.
- `--ephem simple|full` selects the ephemeris model.
- `--skip-stages` accepts stage names or numbers: `fit`, `track`, `kinematics`, `swift`.
- `--obs` restricts processing to specific observation IDs or ranges.

**Expected Data Layout**
The pipeline expects spline-corrected event files under the base directory:
- `BASE_DIR/<OBSID>/repro/*splinecorr.fits`

Outputs are written under:
- `BASE_DIR/2Dfits/`

**Outputs**
- Fit logs: `2Dfits/fit results/fit-results-<ID>.txt`
- Multi-component logs: `2Dfits/multi comp fit results/multi-comp-fit-results-<ID>.txt`
- Fit PDFs: `2Dfits/fit plots/fit-plots-<ID>.pdf`
- Multi-component PDFs: `2Dfits/multi comp fit plots/multi-comp-plots-<ID>.pdf`
- Component tracker table: `2Dfits/comp tracker tables/comp-tracker-table-<ID>.csv`
- Tracker plots: `2Dfits/comp tracker plots/comp-tracker-plots-<ID>.pdf`
- Kinematic plots: `2Dfits/jet plots/ss433-jet-fit-results-<ID>.pdf`
- Swift comparison: `2Dfits/jet plots/swift-comparison-<ID>.pdf`

`<ID>` is derived from component count, sigma, bin size, and signifiers.

**Repo Structure**
- `pipeline.py` orchestrates the four analysis stages.
- `fit_images.py` runs Sherpa fits in parallel and compiles PDFs.
- `track_components.py` parses log output, re-centers components, and plots tracking diagnostics.
- `model_kinematics.py` fits the jet model and generates kinematic plots.
- `lib/arguments.py` defines the CLI.
- `lib/sherpa_fit.py` contains Sherpa + CIAO fitting logic and optional MCMC.
- `lib/log_parser.py` parses Sherpa log files into a tracker table.
- `lib/physics.py` implements the SS433 kinematic model and fitting utilities.
- `lib/plotting.py` generates kinematic model plots.
- `lib/image_utils.py` provides centroiding and WCS helpers.
- `lib/swift_compare.py` builds Swift/XRT comparison plots.

**Dependencies**
The code assumes a CIAO/Sherpa Python environment plus common scientific packages.
- CIAO tools: `sherpa`, `ciao_contrib.runtool` (dmcopy, reproject_image, dmhedit)
- Python packages: `numpy`, `pandas`, `scipy`, `astropy`, `matplotlib`, `tqdm`, `multiprocess`, `Pillow`, `emcee`, `corner`
- A `coords` package providing `coords.format.ra2deg` and `coords.format.dec2deg`

If you hit import errors, confirm you are running inside the CIAO environment and update `config.py` paths.

**Analysis Sandbox**
Exploratory notebooks live in the sibling folder `../analysis-sandbox/`. They are not part of the automated pipeline but document intermediate analyses:
- `interactive_jet_plot.ipynb`: interactive widget-driven jet plots with adjustable betas and view options.
- `ss433-phases-2023.ipynb`: compute and plot precession/orbital/nutation phases, redshifts, and proper motions.
- `component-tracking-plots.ipynb`: standalone log parser and component-tracking plots (PA vs time, count rates, polar map).
- `hrc-jitter-analysis.ipynb`: centroid-based jitter diagnostics, spline fits, and jitter-corrected event files.
- `sherpa-image-fit-pipeline.ipynb`: notebook version of the Sherpa fitting pipeline with MCMC and PDF outputs.
- `srcflux.ipynb`: batch CIAO `srcflux` runs using source/background regions.
- `process-hrc.ipynb`: centroiding via 1D Gaussian histogram fits and diagnostic PDFs.
- `jets_overlayed_smooth.ipynb`: CIAO deconvolution/smoothing and jet overlays on images.
- `kinematic-model-fits.ipynb`: manual kinematic model fitting and plot generation.
- `fit-comparisons.ipynb`: compare multi-component fit statistics and p-values across model choices.
