# qdot — Nuclear Spin Simulations for InGaAs Quantum Dots

A Python library for simulating nuclear spin dynamics in InGaAs self-assembled quantum dots. It covers the full pipeline from experimental strain data through to NMR spectra, spin correlators, and Nuclear Frequency Focussing (NFF) curves.

The four nuclear species modelled are **Ga69, Ga71, As75, and In115**.

---

## Installation

Python 3.10 or later is required. Install the package in editable mode from the repository root:

```bash
pip install -e ".[dev]"
```

> **Note:** Earlier versions of the strain simulation required `scipy<1.11` because of `scipy.optimize` changes in 1.11. The current vectorised energy minimisation works on modern scipy (the test suite passes on 1.17), and the package now requires `scipy>=1.11`. Dependencies are declared in `pyproject.toml`; there is no separate requirements file.

---

## Quick start

All public functions are importable directly from the `qdot` namespace:

```python
import qdot

# Look up physical constants for a species
params = qdot.species_dict["In115"]
print(params["zeeman_frequency_per_tesla"])  # 9.33e6 Hz/T
print(params["particle_spin"])               # 4.5
```

---

## The simulation pipeline

Most workflows follow this sequence:

```
Strain data  →  EFG tensors  →  Hamiltonians  →  Observable (correlator / NMR / NFF)
```

### Step 1 — Load strain data

The primary experimental source is the Sokolov dataset (Phys. Rev. B 93, 045301, 2016). You need three plain-text files: `full_epsilon_xx.txt`, `full_epsilon_xy.txt`, `full_epsilon_yy.txt`, each a 1600×1600 array.

```python
import pathlib
import qdot

DATA_DIR = pathlib.Path("/path/to/sokolov/data")

# [left, right, top, bottom] pixel bounds within the 1600×1600 source
DOT_REGION = qdot.SOKOLOV_DOT_REGION  # [100, 1200, 439, 880]

xx, xz, zz = qdot.load_strain_data(DATA_DIR, DOT_REGION)
# Each array has shape (n_rows, n_cols) — the pixel crop of the dot region.

# step_size=10 loads every 10th pixel, useful for quick tests
xx, xz, zz = qdot.load_strain_data(DATA_DIR, DOT_REGION, step_size=10)
```

### Step 2 — Calculate EFG tensors

Given the strain components, compute the electric field gradient at every lattice site. This returns the biaxiality η, the three principal EFG components, and the Euler angles to the principal axis frame (PAF).

```python
# Use the vectorised version — same output, much faster for large arrays
eta, V_XX, V_YY, V_ZZ, euler_angles = qdot.calculate_efg_vectorised(
    "In115", xx, xz, zz
)
# eta, V_XX, V_YY, V_ZZ:  shape (n_rows, n_cols)
# euler_angles:            shape (n_rows, n_cols, 3) — (alpha, beta, gamma) per site

# Use the Sundfors (1974) parameters instead of the default Checkhovich values:
eta, V_XX, V_YY, V_ZZ, euler_angles = qdot.calculate_efg_vectorised(
    "In115", xx, xz, zz, use_sundfors=True
)
```

The scalar `calculate_efg` has the same interface but uses a Python loop — use it only for debugging or small arrays.

### Step 3 — Save and reload EFG archives

EFG calculation for the full dot takes minutes. Save the results so you can skip the calculation on subsequent runs.

```python
qdot.save_efg(
    data_dir=DATA_DIR,
    nuclear_species="In115",
    region_bounds=DOT_REGION,
    step_size=1,
    eta=eta,
    V_XX=V_XX,
    V_YY=V_YY,
    V_ZZ=V_ZZ,
    euler_angles=euler_angles,
)
# Saves a .npz file. Silently skips if the file already exists.

# Reload later:
eta, V_XX, V_YY, V_ZZ, euler_angles = qdot.load_efg(
    DATA_DIR, "In115", DOT_REGION, step_size=1
)
# euler_angles is reshaped to (n_sites, 3) on load.
```

### Step 4 — Build Hamiltonians

Hamiltonians are QuTiP `Qobj` objects. The Zeeman and quadrupolar terms both carry units of Hz.

```python
import scipy.constants as const
import qdot

species = qdot.species_dict["Ga69"]
spin         = species["particle_spin"]           # 1.5
zeeman_hz    = species["zeeman_frequency_per_tesla"] * 5.0  # 5 T field

# Quadrupolar coupling constant (Hz per V/m²)
Q   = species["quadrupole_moment"]
qcc = (3 * const.e * Q) / (2 * const.h * spin * (2 * spin - 1))

# Pick a single site (e.g. site [50, 50] in the EFG arrays)
alpha, beta, gamma = euler_angles[50, 50]
eta_site  = eta[50, 50]
V_ZZ_site = V_ZZ[50, 50]

H = qdot.faraday_hamiltonian(
    zeeman_term=zeeman_hz,
    quadrupolar_term=qcc * V_ZZ_site,
    biaxiality=eta_site,
    particle_spin=spin,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)
# H is a (2I+1) × (2I+1) QuTiP Qobj.

# Voigt geometry (B along x instead of z):
H_voigt = qdot.voigt_hamiltonian(zeeman_hz, qcc * V_ZZ_site, eta_site, spin, alpha, beta, gamma)
```

---

## Computing observables

### Spin correlators

The spin correlator ⟨Iα(t) Iα(0)⟩ decays as the nuclear spin dephases. `run_correlator_series` averages over all sites in the dot and evaluates all three axes.

Pre-computed EFG archives must exist before calling this (see Step 3 above).

```python
import numpy as np
import qdot

# Log-spaced time axis from 10⁻⁷ to 10⁻³ s, 50 points
timerange = np.logspace(-7, -3, 50)

correlators = qdot.run_correlator_series(
    data_dir=DATA_DIR,
    timerange=timerange,
    applied_field=5.0,          # Tesla
    nuclear_species="Ga69",
    region_bounds=DOT_REGION,
    step_size=100,              # subsample 1 in 100 sites to keep runtime short
)
# correlators has shape (3, 50) — rows are x, y, z axes.
print(correlators.shape)  # (3, 50)
```

For a full run over all four species with data saved to disk, use `run_log_correlator_simulation` or `run_linear_correlator_simulation` (in `qdot.correlators`). These write a `.npz` archive automatically.

```python
from qdot.correlators import run_log_correlator_simulation

run_log_correlator_simulation(
    data_dir=DATA_DIR,
    save_dir=pathlib.Path("output/"),
    min_time_exp=-7,
    max_time_exp=-3,
    n_times=50,
    applied_field=5.0,
    step_size=100,
)
# Writes output/log_time_correlator_data_B5.0T_50pts_1e-07_1e-03s_region[...].npz
```

### NMR absorption spectra

Compute the NMR absorption spectrum at a single lattice site. The result is the total transition rate as a function of RF frequency.

```python
import numpy as np
import qdot

rf_freqs = np.linspace(40e6, 70e6, 500)   # Hz

rates = qdot.absorption_spectrum(
    nuclear_species="Ga69",
    applied_field=5.0,            # Tesla
    field_geometry="Faraday",     # or "Voigt"
    rf_freq_list=rf_freqs,
    location=(50, 50),            # (x, y) index into the EFG arrays
    data_dir=DATA_DIR,
    region_bounds=DOT_REGION,
)
# rates has shape (500,)

# Sweep over several field values:
results = qdot.varied_field_spectra(
    nuclear_species="Ga69",
    applied_field_list=[3.0, 5.0, 7.0],
    field_geometry="Faraday",
    rf_freq_list=rf_freqs,
    location=(50, 50),
    data_dir=DATA_DIR,
)
# results is a list of dicts, each with keys "applied_field", "rf_freq_list", "data"
for r in results:
    print(r["applied_field"], r["data"].max())
```

### NFF polarisation curves

Nuclear Frequency Focussing models the electron spin polarisation after repeated optical pulses under dephasing.

```python
import qdot

# Polarisation as a function of dephasing strength, for one pulse
dephasing_values, z_polarisation = qdot.dephasing_polarisation_curve(
    q0=0.9,       # pulse coherence (1 = perfect, 0 = incoherent)
    phase=0.5,    # pulse phase in radians
    n_gammas=100, # number of dephasing values to evaluate
)
# dephasing_values: 100 points from 1.0 (no dephasing) to 0.5 (maximum)
# z_polarisation:   real array, same length

# Polarisation with no dephasing at all:
pol = qdot.non_dephased_polarisation(q0=0.9, phase=0.5)
```

---

## Generating and saving graphs

All plot functions are in `qdot.plot`. Each function returns the `matplotlib.Figure` object and accepts an optional `save_path`. Pass `save_path` to write the file to disk; omit it to display the figure interactively.

```python
import pathlib
import numpy as np
import qdot
from qdot import plot as qdplot

OUTPUT_DIR = pathlib.Path("output/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Correlator plot

```python
timerange = np.logspace(-7, -3, 50)
correlators = qdot.run_correlator_series(DATA_DIR, timerange, 5.0, "Ga69", DOT_REGION)

fig = qdplot.plot_correlator(
    timerange=timerange,
    correlator_data=correlators,
    nuclear_species="Ga69",
    applied_field=5.0,
    log_time=True,
    save_path=OUTPUT_DIR / "correlator_Ga69_5T.png",
)
# fig is returned even when save_path is set, so you can further customise it.
```

### Fourier transform of a correlator (linear time axis only)

```python
timestep = 1e-7
timerange = np.arange(1e-7, 5e-5, timestep)
# run correlator with the same timerange...

fig = qdplot.plot_fourier_transform(
    timerange=timerange,
    correlator_data=correlators,
    timestep=timestep,
    nuclear_species="Ga69",
    applied_field=5.0,
    save_path=OUTPUT_DIR / "fft_Ga69_5T.png",
)
```

### EFG spatial maps

```python
# Equivalent B-field for one species
b_field = V_ZZ * qcc / species["zeeman_frequency_per_tesla"]

fig = qdplot.plot_equivalent_b_field(
    nuclear_species="Ga69",
    b_field_array=b_field,
    save_path=OUTPUT_DIR / "bfield_Ga69.png",
)

# All four species in one 2×2 figure
fig = qdplot.plot_all_equivalent_b_fields(
    b_field_arrays=[b_Ga69, b_Ga71, b_As75, b_In115],
    region_bounds=DOT_REGION,
    save_path=OUTPUT_DIR / "bfield_all_species.png",
)
```

### Indium concentration map

```python
conc = qdot.load_concentration_data(DATA_DIR, DOT_REGION)

fig = qdplot.plot_concentration(
    conc_data=conc,
    save_path=OUTPUT_DIR / "concentration.png",
)
```

### Strain tensor maps (toy model)

```python
from qdot.plot import plot_strain_lattice, plot_strain_tensors

species, unstrained, strained, *_ = qdot.run_strain_simulation(15, 15, lattice_type=1)
tensors = qdot.strain_tensor(strained, unstrained)

plot_strain_lattice(species, unstrained, strained, save_path=OUTPUT_DIR / "lattice.png")
plot_strain_tensors(tensors, save_path=OUTPUT_DIR / "tensors.png")
```

---

## Running the example scripts

The `examples/` directory contains ready-to-run scripts. All can be called from the repository root without any arguments.

| Script | What it does |
|---|---|
| `examples/strain_toy_model.py` | Spring-mass strain model for pure GaAs, single In atom, and a 5×5 In block. Saves six PNG files. |
| `examples/strain_sokolov_data.py` | Loads and visualises the experimental Sokolov strain data. Requires the external data files. |
| `examples/nmr_spectrum.py` | NMR spectrum for Ga69 at a single site, no data files needed. |
| `examples/spin_correlator.py` | Spin correlator time series, no data files needed. |
| `examples/nff_polarisation.py` | NFF dephasing polarisation curve. |

```bash
# No external data required:
python examples/strain_toy_model.py
python examples/nmr_spectrum.py
python examples/spin_correlator.py
python examples/nff_polarisation.py

# Requires Sokolov data files — set DATA_DIR inside the script first:
python examples/strain_sokolov_data.py
```

All example scripts save their output to `claude-test-graphs/` in the repository root.

---

## The toy strain model

The `qdot.strain` module provides a 2D spring-mass model that can generate synthetic strain fields without the Sokolov dataset.

```python
from qdot.strain import run_strain_simulation, strain_tensor, block_in_positions
import qdot

# Pure GaAs — no strain expected
species, unstrained, strained, time_s, row_err, col_err = run_strain_simulation(
    n_rows=15, n_cols=15, lattice_type=0
)

# Single In atom at the centre
species, unstrained, strained, *_ = run_strain_simulation(15, 15, lattice_type=1)

# Custom In block
in_pos = block_in_positions(bottom_left=[5, 5], top_right=[10, 10])
species, unstrained, strained, *_ = run_strain_simulation(
    15, 15, lattice_type=2, in_positions=in_pos
)

# Calculate strain tensor from the relaxed positions
tensors = qdot.strain_tensor(strained, unstrained)
# tensors has shape (n_rows, n_cols, 2, 2)
print(tensors[7, 7])  # 2×2 strain tensor at site (7, 7)
```

The species encoding is `0 = Ga, 1 = As, 2 = In`. Bond parameters come from Vurgaftman 2001 (DOI: 10.1063/1.1368156).

---

## Isotope parameters

`qdot.species_dict` and `qdot.old_species_dict` are plain Python dicts. Each entry is a dict with these keys:

| Key | Units | Description |
|---|---|---|
| `particle_spin` | — | Nuclear spin quantum number I |
| `zeeman_frequency_per_tesla` | Hz/T | Zeeman splitting per unit field |
| `quadrupole_moment` | m² | Nuclear quadrupole moment Q |
| `S11` | V/m² | Gradient-elastic tensor component |
| `S44` | V/m² | Gradient-elastic tensor component |

`species_dict` uses the Checkhovich values (current best). `old_species_dict` uses the older Sundfors (1974) values, which are useful when comparing with older literature. Pass `use_sundfors=True` to `calculate_efg` and `absorption_spectrum` to switch.

```python
for name, params in qdot.species_dict.items():
    print(f"{name}: I={params['particle_spin']}, "
          f"γ={params['zeeman_frequency_per_tesla']/1e6:.2f} MHz/T")
```

---

## Running tests

```bash
pytest tests/
```

---

## Key references

- Sokolov et al., Phys. Rev. B **93**, 045301 (2016) — experimental strain dataset. DOI: 10.1103/PhysRevB.93.045301
- Checkhovich et al. — gradient-elastic tensor values used in `species_dict`.
- Sundfors (1974) — older values in `old_species_dict`.
- Vurgaftman et al., J. Appl. Phys. **89**, 5815 (2001) — bond stiffnesses in the strain toy model. DOI: 10.1063/1.1368156
