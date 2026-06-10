# Archive vs. qdot Module — Functionality Comparison and Porting Plan

This document compares the original research scripts under `archive/` with the
`qdot` package, function by function, with particular attention to the
experimental simulation scripts and graph generation. It then sets out a plan
for porting the missing capabilities into `qdot`.

The archive is historical reference only. `qdot/` supersedes it for all new work.

Conventions used below: "full" means qdot reproduces the capability; "partial"
means the computation or data exists but the driver/figure does not; "none"
means nothing in qdot covers it.

---

## 1. What qdot adds that the archive lacks

| Addition | Location |
|---|---|
| Importable package with clean public API | `qdot/__init__.py`, `pyproject.toml` |
| Explicit `data_dir` arguments — the archive hardcodes `/home/will/...` paths in every file | all modules |
| Vectorised EFG calculation (`calculate_efg_vectorised`) | `qdot/efg.py` |
| Vectorised strain energy and tensor calculations | `qdot/strain.py` |
| Type hints throughout; tests (139); logging instead of `print()` | all modules, `tests/` |
| Physics corrections: `transition_rate` \|M\|→\|M\|², ρ normalisation, trace-preserving Kraus operators (√γ·I vs the archive's γ·I), valid initial density matrix in NFF, FFT plotted as magnitude rather than real part, improper-rotation fix in EFG Euler angles, `pinv` in strain tensor | various |
| Separation of computation from plotting | `qdot/plot.py` |
| `save_efg` overwrite control and skip warnings; archive-name normalisation | `qdot/io.py` |

Note on the NFF physics corrections: the archive's `DephasingKrausKreator` and
its initial density matrix `[[1,1],[0,1]]` differ from `qdot/nff.py` by design —
the qdot versions are the corrected forms. Any numerical comparison against
archive NFF output will differ for this reason.

---

## 2. Fully ported (no action needed)

The shared engine (`archive/common_modules/backbone_quadrupolar_functions.py`,
757 lines) is fully covered: data loaders (`load_strain_data`,
`load_concentration_data`, `load_mirrored_data`), EFG calculation and archive
save/load, Euler angle extraction, both Hamiltonian constructors,
`rf_hamiltonian`, `transition_rate`, and the serial/parallel spin correlators.
`isotope_parameters.py` (three duplicate copies in the archive) is consolidated
into `qdot/isotopes.py`.

Also fully ported: the whole strain toy model (`neat_strain_toy_model.py` →
`qdot/strain.py` + the two strain plots in `qdot/plot.py`), the state-vector
machine gun (`machine_gun_basics.py`/`backbone_functions.py` →
`qdot/machine_gun.py`), the correlator calculation drivers
(`parallel_correlator.py` → `qdot/correlators.py`), the single-site NMR
absorption spectrum, the correlator time-domain and FFT plots, the
equivalent-B-field *plotters*, and the concentration map plots
(`plot_concentration`, `plot_concentration_with_regions`).

---

## 3. Gaps, by capability

### 3.1 NMR: 2D field–frequency absorption maps (`archive/NMR/spectra.py`, 2774 lines)

The single-site spectrum (`absorption_spectrum`) is the only part of the
archive's NMR machinery that qdot has. Missing:

- **Multi-site parallel computation** — `many_location_parallel_calculation`
  farms the per-site spectrum over a `multiprocessing.Pool` for each applied
  field, sums over sites, and produces a (fields × frequencies) array. This is
  the NMR analogue of `run_correlator_series`.
- **Disk caching** — results saved as parameter-encoded `.npy` files,
  recomputed only on `recalc=True` (`load_many_location_data`,
  `many_NMR_spectra_data_calculation` batch driver over species × geometries).
- **Map plotting** — log-scaled `imshow` heatmaps (x = RF frequency in MHz,
  y = B in T), including named experiment presets (`large_hd_spectra`,
  `region_characterisation_spectra`, `atomic_region_spectra`,
  `gigahertz_range_spectra`) that fix the grids used for publication figures.
- **Comparison figures** — Faraday−Voigt difference maps (diverging colormap),
  side-by-side two-panel geometry comparison, side-by-side and difference maps
  for Checkhovich vs Sundfors GET values with relabelled colorbars.
- **Site sampling** — `random_locations_list_generator`,
  `all_locations_list_generator`, and `find_best_locations`
  (`random_point_searcher.py`), which snaps random seeds to local maxima of the
  summed strain map so sampled sites sit on strongly-strained nuclei.

### 3.2 NMR: concentration-weighted experimental simulation

`experimental_NMR_sim` produces the figure closest to what an experimentalist
measures: it loads the In concentration map, splits the sampled sites into
In115/Ga69/As75 populations according to mean In concentration (As fixed at
50%, the remainder split between Ga and In), computes each species' 2D map in
parallel, and renders either the log-summed total or per-species transparent
layers (`experimental_NMR_sim_transparent_version`, Greys/Blues/Reds with
decreasing alpha). Nothing in qdot covers species mixing, and the species-split
heuristic is a physics decision that needs confirming before porting.

### 3.3 NMR: 1D line spectra and RF pulse selectivity analysis

A second family of figures absent from qdot:

- `single_field_line_spectra` — multi-site summed, normalised spectrum at one field.
- `applied_field_comparison_line_spectra` — stacked subplots, one per field.
- `RF_comparison_line_spectra`, `number_of_nuclei_comparison` — overlay curves
  with an inset zoom on the 69–91 MHz window (ensemble-convergence studies).
- Combined-species analysis: `combined_species_single_spectra_data` sums all
  four isotopes over a shared site list at fixed B; the plot variants mark
  `scipy.signal.find_peaks` peaks or shade a target frequency window.
- Pulse selectivity: `create_lorentzian_pulse` models an RF pulse's frequency
  profile; `overlaid_pulse_graphs` and `pulse_graph_area_calculations` /
  `comparison_area_plots` multiply pulse × spectrum and Simpson-integrate to
  report the fraction of total absorption captured in target vs off-target
  windows as a function of pulse width. This was used to assess whether a
  pulse can address one species without touching others.

### 3.4 Energy-level diagrams (`archive/quadrupolar/splitting_graphs.py`)

Entirely absent from qdot. The archive computes eigenenergies (MHz) of the
site Hamiltonian swept over applied field (0–3 T) or biaxiality η (0–1), and
plots:

- single-site level diagrams (anti-crossings), per species and geometry;
- two-panel Faraday | Voigt comparisons with shared y-limits;
- ensemble versions overlaying all sites or n random sites in a region, which
  visualise inhomogeneous level broadening across the dot;
- mirrored-dot variants (`mirrored_data_plots.py`).

`qdot/hamiltonians.py` provides the constructors; what is missing is the sweep
helper (build H per field/η value, collect `eigenenergies()`) and the plots.

### 3.5 EFG direction quiver maps (`archive/quadrupolar/plotting_QI_direction.py`)

`qdot/plot.py` has no quiver function at all. The archive draws the EFG
principal-axis direction at each site (arrow angle from Euler γ, optionally
scaled by V_ZZ) over a choice of backgrounds:

- biaxiality η heatmap;
- In concentration heatmap;
- quadrupole frequency magnitude heatmap (single species and 2×2 all-species
  grid with a shared colour scale);
- a two-panel Checkhovich vs Sundfors comparison with a shared colorbar.

### 3.6 Quadrupolar statistics: heatmaps and histograms (`quadrupolar_strength_maps.py`, `energy_histograms.py`)

- Spatial heatmaps of quadrupole frequency K·V_ZZ and of η (fixed 0–1 scale).
  qdot's `plot_equivalent_b_field` is the same imshow pattern but no dedicated
  maps exist, and the **computation** of the equivalent-B-field array — the
  partner to qdot's existing plotters — was never ported either
  (`equivalent_B_field_calculation` in `correlator_calc.py`).
- Histograms of per-site quadrupole frequencies with fitted distributions
  (Gaussian, Maxwell-Boltzmann, gamma), per species and 2×2 all-species grids,
  with fit parameters annotated on the figure and an `.npz` cache layer.
- Layered comparisons: all species in one plot; multiple regions distinguished
  by linestyle; Checkhovich vs Sundfors comparison limited to the two species
  whose GET values changed (Ga69, As75).

### 3.7 Measured (Sokolov) strain plotting (`archive/strain/strain_grapher.py`)

qdot can load the Sokolov strain data but cannot plot it.
`plot_strain_tensors` only accepts the toy model's (n, m, 2, 2) tensor array
with xx/xy/yy components. The archive has:

- `sokolov_strain_grapher` / `sokolov_strain_vertical_grapher` — the
  paper-recreation figures: three panels (ε_xx, ε_zz, ε_xz), horizontal with
  fixed ±0.02 scale or vertical with a shared data-driven RdBu scale.
- `strain_histograms` — shear strain distribution under two definitions.
- `single_row_strain_grapher` — line cut of all three components along a row.
- `many_rows_strain_grapher_3d` — 3D surface of one component (low value).
- `finding_atomic_sites` — peak detection over the strain map via
  `skimage.feature.peak_local_max` (only archive code needing scikit-image).

### 3.8 Mirrored (symmetric) dataset support

qdot has `load_mirrored_data` for strain but:

- the **creation** functions are missing (`create_mirrored_data.py`):
  `mirror_array_left_to_right`/`right_to_left` and the four writers that save
  mirrored strain `.npz` and mirrored concentration `.npy` files — without
  them `load_mirrored_data` only works on files produced by the archive;
- there is no loader for mirrored **concentration** data at all;
- `absorption_spectrum` does not pass `real_strain`/`mirror_type` through to
  `load_efg`, so NMR on mirrored datasets is not reachable through qdot even
  though `load_efg` itself supports the flags.

### 3.9 Concentration dataset creation (`archive/concentration/`)

`create_nice_conc_dataset.py` builds the file every concentration loader
depends on: it reads the raw Sokolov `.mat` scatter data (15,011 points),
clips negatives, grids it onto the 1600×1600 strain-pixel grid with
`scipy.interpolate.griddata` (cubic), and writes
`conc_data_to_scale_cubic_interpolation.npy`.
`recreate_sokolov_conc_graphs.py` renders the nearest/linear/cubic variants
for comparison against the published figure. qdot only consumes the finished
`.npy`. Also unported: `region_bound_finder`, which converts a rectangle drawn
on the concentration map into `region_bounds` for the other functions, and the
named-region catalogue `QD_regions_dict` (nine named pixel regions such as
`entire_dot`, `central_high_In`) — qdot has only the single
`SOKOLOV_DOT_REGION` constant.

### 3.10 Machine gun: density-matrix formulation with error channels (`density_matrix_machine_gun.py`)

`qdot/machine_gun.py` covers only the ideal state-vector protocol; its unused
scaffolding (`state_to_density_matrix`, `cnot_array`) was written for this
extension. Missing:

- `PerfectMachineGunDensityMatrices` — ideal protocol on vectorised ρ via
  superoperators;
- `DotDephasingKrausKreator` / `DotDampingKrausKreator` — phase-damping and
  amplitude-damping Kraus pairs on the dot qubit, embedded into the full
  (n_photons+1)-qubit space;
- `DephasingMachineGun` / `AmplitudeDampingMachineGun` — protocol runs with
  the error channel applied each cycle;
- the analysis/figure family: fidelity and trace-distance vs error strength,
  multi-photon-count overlays, combined fidelity+trace-distance figures, a 2D
  (photon count × dephasing) fidelity heatmap, and a tabulation of maximal
  dephasing fidelity against candidate analytic scalings;
- `ErrorWrapper` — converts the readable `[[photon, "X"], ...]` error list
  into the numeric array `machine_gun_with_pauli_errors` takes.

### 3.11 NFF: small gaps

- X and Y polarisation finders (qdot has Z only).
- No plotting function for the dephasing polarisation curve (the archive plots
  it in a module-level script; `qdot/plot.py` has no NFF figure).
- `phase_damping_graph.py` — the analytic γ vs photon-scattering-probability
  relation.
- `NFF_no_nuclear_spin_steady_state.py` — closed-form 4×4 process matrix for
  one pulse-plus-precession cycle (Greilich 2007 parameters) and a scan for
  eigenvalue-1 steady states over (q0, φ). Print-only in the archive.

### 3.12 Correlator: minor gaps

The computation and plots are ported. Missing: a load-and-plot helper for the
`.npz` archives that `run_log_correlator_simulation` /
`run_linear_correlator_simulation` write, and the batch drivers that loop
calculation + figures over a list of B fields.

---

## 4. Not worth porting

Scratch, superseded, or one-off code; listed so future reviews don't rediscover it:

| Item | Reason |
|---|---|
| `errors_testing.py`, `CNOT_testing.py`, `perfect_machine_gun_testing.py`, `single_qubit_operation_testing.py` | debug duplicates of ported functions with print statements |
| `correlator/testing_functions.py` | scrapbook with undefined references; the chunksize/step-size benchmarks belong in `benchmarks/` if ever needed |
| `edge_counting_test.py`, `toy_model_timer.py` | verify things now covered by tests; timing harness trivial to rewrite |
| `tersoff_cutoff_func_test.py` | explores a Tersoff potential never adopted by the model |
| `changing_eigen_vectors_test.py`, `ratio_of_QI_strengths.py` | scratch; the latter is a self-contained analytic curiosity (xkcd-styled plot) |
| `comparing_sokolov_and_checkhovich_QI_graphs.py` | marked outdated in-file (2020); superseded by `use_sundfors` + the quiver plots |
| `realistic_ham_testing.py` Hinton diagrams and console dumps | collaborator one-offs (`EigenvalueDataforDara`); the eigenvector-overlap sweep figures are the only candidate worth reconsidering later |
| `spectra.py` module-level "control panel" boolean blocks | replaced by `examples/` scripts in qdot's structure |
| Duplicate loaders in `conc_maps.py`, `create_mirrored_data.py` | already consolidated in `qdot/io.py` |

---

## 5. Porting plan

Ordered by value per unit of effort. Each phase is a self-contained PR-sized
unit with tests; plotting functions follow `qdot/plot.py` conventions (return
the Figure, `save_path` to save instead of show). All cache files take an
explicit directory argument and reuse the `save_efg` pattern (skip-with-warning
unless `overwrite=True`).

### Phase 1 — NMR multi-site engine and maps (highest value)

The largest experimental capability gap. Target: reproduce
`many_location_parallel_calculation` → `NMR_plot` end to end.

New in `qdot/nmr.py`:

```python
def summed_absorption_spectrum(nuclear_species, applied_field, field_geometry,
                               rf_freq_list, locations, data_dir, *,
                               region_bounds=None, step_size=1, rf_field=5e-3,
                               use_sundfors=False, n_processes=None) -> np.ndarray
    # Pool.starmap over locations (pattern from run_correlator_series),
    # summed over sites. One row of the 2D map.

def absorption_map(nuclear_species, applied_field_list, field_geometry,
                   rf_freq_list, locations, data_dir, ...) -> np.ndarray
    # (n_fields, n_freqs); loops summed_absorption_spectrum over fields,
    # loading EFG data once.
```

New in `qdot/io.py`: `save_nmr_map` / `load_nmr_map` with a parameter-encoded
filename (species, geometry, n_locs, bounds, field and frequency grids), and a
`SOKOLOV_REGIONS` dict porting `QD_regions_dict` (keep `SOKOLOV_DOT_REGION` as
an alias for the `entire_dot` entry).

New site-sampling helpers (in `qdot/nmr.py` or a small `qdot/sites.py`):
`random_locations(n, region_bounds, shape_source)`, `all_locations(...)`, and
`find_best_locations(...)` (strain-local-maximum snap; the "summed strain
components" metric should be flagged to the physicist before relying on it).

New in `qdot/plot.py`: `plot_nmr_map` (log-scaled imshow, MHz × T extents),
`plot_nmr_map_pair` (two-panel side-by-side, used for both Faraday|Voigt and
Checkhovich|Sundfors), `plot_nmr_map_difference` (diverging colormap; covers
both comparison figures via arguments rather than four near-identical
functions).

Also in this phase: pass `real_strain`/`mirror_type` through
`absorption_spectrum` and `varied_field_spectra` (one-line plumbing that
unblocks mirrored-dot NMR).

Effort: the largest phase — engine + io + three plot functions + tests.
Tests can run on tiny synthetic EFG archives as `tests/test_nmr.py` already does.

### Phase 2 — Energy-level diagrams

New in `qdot/hamiltonians.py` (or a new `qdot/levels.py` if hamiltonians.py
should stay constructor-only):

```python
def energy_levels_vs_field(applied_fields, eta, V_ZZ, euler_angles,
                           nuclear_species, field_geometry) -> np.ndarray  # (n_levels, n_fields)
def energy_levels_vs_eta(etas, applied_field, V_ZZ, euler_angles, ...) -> np.ndarray
```

taking site parameters directly (callers fetch them via `load_efg`), so the
functions stay decoupled from the archive format. New in `qdot/plot.py`:
`plot_energy_levels` (single panel) and `plot_energy_levels_comparison`
(two-panel Faraday|Voigt with shared y-limits). Ensemble (all-sites /
random-sites) figures fall out by looping the sweep and overplotting — provide
this as an example script rather than a library function.

Effort: small. The physics is three lines per sweep point.

### Phase 3 — Quadrupolar maps, quiver plots, and histograms

Computation side:

- factor the quadrupole coupling constant `3eQ/(2hI(2I−1))` — currently
  duplicated in `correlators.py` and `nmr.py` — into a function in
  `qdot/isotopes.py` (e.g. `quadrupole_coupling(species_dict_entry)`), and add
  `quadrupole_frequency_map(nuclear_species, V_ZZ)` and
  `equivalent_b_field(nuclear_species, V_ZZ)` (the missing partner to the
  existing plotters) somewhere appropriate (`qdot/efg.py`).

Plot side (`qdot/plot.py`):

- `plot_site_map(array, ...)` — the generic imshow+colorbar pattern used by
  η maps, frequency maps, and the existing equivalent-B-field plots; dedicated
  thin wrappers `plot_biaxiality` and `plot_quadrupole_frequency` for the
  standard colour scales (η fixed 0–1).
- `plot_efg_directions(euler_angles, background, ...)` — the quiver overlay
  with a background array argument instead of three near-copies; an
  `all_species` 2×2 variant with shared normalisation.
- `plot_frequency_histogram(frequencies, fit="gamma"|"maxwell"|"gauss"|None)`
  — step histogram with optional fitted pdf and annotated fit parameters;
  `plot_frequency_histograms_grid` (2×2 all species);
  layered overlays accept a list of (label, frequencies) pairs, covering both
  the multi-region and Checkhovich-vs-Sundfors comparisons with one function.

The histogram `.npz` cache from the archive is probably unnecessary once the
inputs come from already-cached EFG archives; drop it unless the physicist
wants it.

Effort: moderate; all plotting, no new physics. The distribution-fitting
choice (which distributions are meaningful) is a physics question — port all
three fits, let the physicist prune.

### Phase 4 — Sokolov strain plots and mirrored datasets

- `plot_measured_strain(xx, xz, zz, layout="horizontal"|"vertical", ...)` in
  `qdot/plot.py` covering both paper-recreation figures (fixed ±0.02 scale
  horizontal; shared RdBu scale vertical); `plot_strain_row_cut`; the shear
  histogram folds into the Phase 3 histogram function. Skip the 3D surface and
  `finding_atomic_sites` (scikit-image dependency, exploratory) unless asked.
- `qdot/io.py`: `mirror_array(array, direction)`,
  `create_mirrored_strain_data(data_dir, region_bounds, direction)`,
  `create_mirrored_concentration_data(...)`,
  `load_mirrored_concentration_data(...)` — completing the mirrored pipeline
  whose loader already exists.

Effort: small-moderate.

### Phase 5 — Experimental NMR simulation and pulse analysis

Builds on Phase 1.

- `experimental_nmr_simulation(...)` in `qdot/nmr.py`: species populations from
  the concentration map, per-species `absorption_map`, returns the per-species
  maps so the caller chooses summed or layered rendering;
  `plot_nmr_map_layered` in `qdot/plot.py` (Greys/Blues/Reds, decreasing alpha).
  **Blocked on a physics decision:** the 50%-As/concentration-split site
  heuristic should be confirmed (add to `human-todo.` when the port starts).
- Combined-species spectrum + pulse selectivity: `combined_spectrum(...)`
  (all four species summed over shared sites), `lorentzian_pulse(freqs, centre,
  width)`, `pulse_capture_fraction(spectrum, pulse, windows)` (Simpson
  integration), and plots for the peak-marked spectrum and pulse-width sweep.
  The hardcoded target windows (77.8–82.3 MHz etc.) become arguments.

Effort: moderate, mostly plumbing once Phase 1 exists.

### Phase 6 — Machine gun density-matrix extension

Completes what `qdot/machine_gun.py`'s docstring already promises:

```python
def initial_density_matrix(n_photons) -> qutip.Qobj
def dot_dephasing_superoperator(dephasing, n_photons) -> qutip.Qobj
def dot_damping_superoperator(damping, n_photons) -> qutip.Qobj
def perfect_machine_gun_density_matrix(n_photons) -> qutip.Qobj
def noisy_machine_gun(n_photons, channel, strength) -> qutip.Qobj   # dephasing | damping
def error_list_to_array(errors: list) -> np.ndarray                  # ErrorWrapper
```

plus fidelity/trace-distance sweep helpers returning arrays, and one plot
function for metric-vs-error-strength curves (multi-photon-count overlay).
The existing `cnot_array` and `state_to_density_matrix` scaffolding gets used.
Note the archive's analytic-scaling tabulation (`MaximumDephasingFidelityTest`)
is analysis, not library code — example script material.

Effort: moderate; Kraus embedding needs care, and tests should check channel
trace preservation and the perfect-channel limit against the state-vector path.

### Phase 7 — One-time dataset tooling and NFF extras (lowest priority)

- Concentration dataset creation from the Sokolov `.mat` (`scripts/` or an
  example, not the library — it runs once per dataset): clip, `griddata`
  interpolation, method-comparison figures. Requires the raw `.mat` file,
  which is not in the repo — confirm it still exists before porting.
- `region_bound_finder` (rectangle → region_bounds converter) alongside
  `SOKOLOV_REGIONS` in `qdot/io.py`.
- NFF: `x_polarisation`/`y_polarisation`, `plot_polarisation_curve`, the γ vs
  scattering-probability relation, and (optionally, as an example) the
  steady-state process-matrix scan with a real figure instead of the archive's
  print-only output.
- Correlator: `load_correlator_archive(...)` + batch plot driver for the
  multi-field figure sets.

---

## 6. Physics questions raised by the port (for human-todo. when work starts)

- Species-split heuristic in `experimental_NMR_sim` (As fixed at 50%).
- `find_best_locations` ranks sites by ε_xx + ε_xz + ε_zz — is a plain sum the
  right strain metric?
- Which distribution fits (Gaussian / Maxwell / gamma) of the quadrupole
  frequency histograms are physically meaningful.
- Whether archive figures that depend on the corrected NFF/transition-rate
  physics should be regenerated for comparison before the archive is retired.
