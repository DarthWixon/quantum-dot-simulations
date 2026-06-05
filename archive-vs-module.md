# Archive vs. qdot Module — Functionality Gaps

This document records what the original research scripts under `archive/` do that
the `qdot` package does not yet cover, what `qdot` added that the archive lacks,
and where something was ported but the interface changed significantly.

The archive is historical reference only. `qdot/` supersedes it for all new work.

---

## What qdot adds (not in the archive)

| Addition | Location |
|---|---|
| Importable package with clean public API | `qdot/__init__.py`, `pyproject.toml` |
| Explicit `data_dir` arguments — no hardcoded paths | all modules |
| Vectorised EFG calculation (`calculate_efg_vectorised`) | `qdot/efg.py` |
| Vectorised strain energy and tensor (`_potential_energy_vectorised`, `strain_tensor_vectorised`) | `qdot/strain.py` |
| Type hints throughout | all modules |
| Test suite (137 tests across all modules) | `tests/` |
| Physics bug fixes: `transition_rate` |M|→|M|², ρ normalisation, Kraus trace-preserving form, EFG improper-rotation fix, `pinv` in strain tensor | various |
| Separation of computation from plotting (`qdot/plot.py`) | `qdot/plot.py` |
| `logging` instead of `print()` | `qdot/io.py`, `qdot/correlators.py` |
| `SOKOLOV_DOT_REGION` constant | `qdot/io.py` |
| `load_mirrored_data` for reading symmetric datasets | `qdot/io.py` |

---

## Gaps: archive functionality absent from qdot

### NMR spectra (`archive/NMR/spectra.py`)

The archive's NMR module is far larger than `qdot/nmr.py`. The main missing pieces:

**Multi-site parallel computation**
- `many_location_parallel_calculation` — runs `absorption_spectrum` across a list of
  lattice sites using `multiprocessing.Pool`, summing contributions at each field value.
  This is the NMR equivalent of `run_correlator_series`; it has no counterpart in `qdot`.
- `load_many_location_data` / `many_location_parallel_plot` — load cached multi-site
  absorption arrays and display as a 2D (field × frequency) heatmap.
- `many_NMR_spectra_data_calculation` — batch driver that calls the above for all four
  species and both geometries in one pass.

**Full-experiment simulation**
- `experimental_NMR_sim` — concentration-weighted multi-species, multi-site simulation.
  Uses `load_In_concentration_data` to determine the In/Ga/As site counts, randomly
  samples real lattice locations for each species, computes parallel absorption, sums
  all species, and plots a log-scaled 2D absorption map that matches what an experimentalist
  would record. No equivalent exists in `qdot`.
- `experimental_NMR_sim_transparent_version` — same output with per-species transparent
  overlaid panels for direct comparison.

**Combined and analysed spectra**
- `combined_species_single_spectra_data` / `combined_species_single_spectra_plot` —
  computes the sum of all species' spectra at a single applied field, then detects and
  marks peaks. Useful for identifying overlapping resonances.
- `combined_species_single_spectra_plot_subregion` — as above but plots a frequency
  sub-window with optional shaded highlight.
- `pulse_graph_area_calculations` — integrates area under Lorentzian pulses as a
  function of RF frequency width.

**Site sampling utilities**
- `random_locations_list_generator` — draws N random (x, y) lattice indices from a
  named region. Used to create representative site samples without computing at every site.
- `all_locations_list_generator` — enumerates all lattice positions in a region.

**Parameter-set and geometry comparisons**
- `Faraday_Voigt_spectra_comparison`, `side_by_side_comparison_plotter` — side-by-side
  Faraday vs. Voigt geometry NMR plots at matching sites.
- `Sundfors_Checkovich_comparison` / `GET_values_comparison_spectra` — runs spectra with
  both Sundfors (1974) and Checkhovich GET values and plots them together for direct
  parameter-set comparison.

---

### Correlator Fourier analysis (`archive/correlator/parallel_correlator.py`)

`qdot/correlators.py` covers the core time-domain computation and the save/load functions.
What is missing is everything in the frequency domain:

- `single_species_correlator_to_fourier` — loads a stored correlator archive, computes
  the FFT along the time axis, and saves the result. Produces the spectral density
  S(ω) from C(t).
- `single_species_fourier_transform_grapher` — plots the FFT result, optionally alongside
  the original time-domain curve.
- `all_species_correlator_to_fourier` — batch version of the above across all four species
  and a list of applied fields.
- `all_species_logtime_correlator_grapher` — generates log-time correlator plots for all
  species at multiple B-field values, intended for publication-style figure production.

There is also no load-and-plot function in `qdot` for the `.npz` archives that
`run_log_correlator_simulation` writes.

---

### Energy level diagrams (`archive/quadrupolar/splitting_graphs.py`)

This entire category is absent from `qdot`. The archive has functions to:

- `single_field_energy_levels` / `many_fields_energy_levels_calc` — compute eigenstate
  energies (in MHz) for a real lattice site as the applied field is swept from 0 to
  some maximum, in either Faraday or Voigt geometry.
- `single_eta_energy_levels` / `many_eta_energy_levels_calc` — same sweep but over
  biaxiality η rather than B field, useful for understanding the quadrupolar effect in
  isolation.
- `plot_energy_level_diagram` / `plot_and_save_all_species_EL_diagrams` — anti-crossing
  energy level plots, one curve per eigenstate, for all four species.
- `all_sites_energy_level_diagram` / `random_sites_energy_level_diagram` — overlay
  energy levels from many sites to show the spread due to inhomogeneous strain.
- `random_sites_geometry_comparison_EL_diagram` — Faraday and Voigt side-by-side on
  the same axes.

`qdot/hamiltonians.py` provides the Hamiltonians these functions build on, but there
is no function in `qdot` that diagonalises them over a parameter sweep and returns or
plots the eigenvalue trajectories.

---

### Quadrupolar interaction visualisation (`archive/quadrupolar/`)

**`quadrupolar_strength_maps.py`**
- `quadrupolar_interaction_strength_plotter` — computes the quadrupolar frequency
  K·V_ZZ at every site in the dot region and plots a 2D heatmap. Highlights where
  the quadrupolar interaction is strongest.
- `biaxiality_plotter` — 2D heatmap of η, with optional overlay of the Sundfors vs.
  Checkhovich difference.

**`plotting_QI_direction.py`**
- `plot_arrows_with_biax_background` — quiver plot of the EFG principal-axis direction
  at each site, overlaid on an η heatmap. Shows where the EFG points across the dot.
- `comparison_plot_with_biax_background` — side-by-side Sundfors vs. Checkhovich
  quiver comparison using a shared axis grid.
- `plot_arrows_with_conc_background` — same quiver plot with In concentration as
  background instead of η.
- `plot_arrows_with_strength_background` / `plot_arrows_with_strength_background_all_species`
  — quiver plots with quadrupolar frequency magnitude as background, all four species
  in a 2×2 grid.

None of these appear in `qdot/plot.py`, which has heatmap and correlator FFT plotting
but no quiver or vector-field visualisation.

---

### Machine gun error channels (`archive/machine_gun/density_matrix_machine_gun.py`)

`qdot/machine_gun.py` implements the cluster-state machine-gun (CSMG) protocol using
state vectors and unitary operators. The archive has a parallel density-matrix
formulation using QuTiP superoperators that is entirely absent from `qdot`:

- `DotDephasingKrausKreator` — builds the dephasing superoperator from Kraus operators.
- `DotDampingKrausKreator` — builds the amplitude-damping superoperator.
- `DephasingMachineGun` — runs the CSMG protocol with a dephasing error channel
  applied at each step.
- `AmplitudeDampingMachineGun` — same with amplitude damping.
- `PerfectMachineGunDensityMatrices` — density-matrix version of the ideal (error-free)
  protocol, using QuTiP's `spre`/`spost` superoperator infrastructure.

The practical consequence is that `qdot` can only simulate the ideal CSMG and cannot
model the effect of photon loss or pure dephasing noise on the output state.

---

### Symmetric dot datasets (`archive/mirrored_dots/create_mirrored_data.py`)

`qdot/io.py` has `load_mirrored_data` to read pre-made symmetric datasets, but the
creation functions that produce those files are not ported:

- `mirror_array_left_to_right` / `mirror_array_right_to_left` — reflect a strain or
  concentration array about its centre column.
- `create_left_right_strain_data` / `create_right_left_strain_data` — applies the
  mirror, packages all three strain components, and saves to `.npz`.
- `create_left_right_conc_data` / `create_right_left_conc_data` — same for the
  In115 concentration map.

Without these, `load_mirrored_data` can only be used if the files already exist from
a previous run of the archive scripts.

---

## What was ported with significant interface changes

| Archive function | `qdot` equivalent | Key difference |
|---|---|---|
| `calculate_EFG` (nested Python loops) | `calculate_efg` + `calculate_efg_vectorised` | Vectorised via batched NumPy; Euler angle sign-convention differences documented |
| `one_species_time_series_correlator_calculator` | `run_correlator_series` | Returns array directly; does not save to disk |
| `log_spaced_correlator_simulation` | `run_log_correlator_simulation` | Saves all species in one `.npz`; same structure |
| `calculate_absorbtion_single_slice_data` | `absorption_spectrum` | Takes explicit `data_dir`; no global paths; step_size defaults to 1 |
| `varied_B_field_spectra` | `varied_field_spectra` | Returns list of dicts rather than writing files |
| `phase_damping_polarisation` (NFF) | `dephasing_polarisation_curve` | Split into computation and plotting |
| Strain toy model | `run_strain_simulation` | Vectorised energy and tensor; returns tuple not separate variables |
| `EFG_data_creator` | `calculate_efg` + `save_efg` | Separated into two explicit calls |

---

## Priority assessment

The gaps vary significantly in how useful they would be to port:

**High value, moderate effort:**
- Multi-site parallel NMR + 2D absorption heatmap (`experimental_NMR_sim` pattern) —
  directly comparable to experimental data; the architecture already exists in the
  correlator module.
- FFT of correlator + spectral density plots — connects time-domain results to
  frequency-domain measurements.

**Moderate value, moderate effort:**
- Energy level diagrams — useful for understanding individual sites; straightforward
  given existing Hamiltonian functions.
- Mirrored dataset creation — needed to make `load_mirrored_data` independently usable.

**High effort, specialist use:**
- Density-matrix machine gun with error channels — requires QuTiP superoperator
  infrastructure; only needed for noise modelling studies.
- EFG direction quiver plots — visualisation detail; useful for papers but not
  core computation.
