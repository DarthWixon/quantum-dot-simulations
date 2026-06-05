# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reviewing examples

To review an example script, use the `/review-example` skill with the filename as argument (e.g. `/review-example efg_species_comparison.py`). This runs a physics reviewer and a code reviewer in parallel, then synthesises their findings into a single ranked report.

The two reviewers are defined in `.claude/agents/`:
- `physics-reviewer` — checks physical correctness against known parameter ranges, symmetries, and limiting cases
- `code-reviewer` — checks Python quality, style, type hints, imports, and qdot API usage; does not comment on physics

A session restart is needed to pick up newly created agent definitions.

## Known limitations

`human-todo.` in the repo root lists issues that require the physicist's judgement — weak strain magnitudes in the toy model, boundary artefacts, architectural decisions about examples. Do not flag or attempt to fix items listed there.

## Git workflow

All changes must be made on a separate branch, never directly on `main`. Create a new branch before starting any task:

```bash
git checkout -b <branch-name>
```

## Setup

Install the package in editable mode (requires Python 3.9+):

```bash
pip install -e ".[dev]"
```

Key constraint: `scipy<1.11` — changes to `scipy.optimize` in 1.11 break the strain simulation's energy minimisation.

Run tests:

```bash
pytest tests/
```

Format code:

```bash
black .
```

**Always run `black <file>` on any file you have edited before committing.** A pre-commit hook also enforces this, but running it immediately avoids surprises at commit time.

## Architecture

This is a Python library simulating nuclear spin dynamics in InGaAs quantum dots, targeting other researchers in the field.

### Package layout (`qdot/`)

| Module | Contents |
|---|---|
| `isotopes.py` | Physical constants for Ga69, Ga71, As75, In115 — the single canonical source |
| `io.py` | Data loading/saving: Sokolov strain data, concentration data, EFG archives. All functions take an explicit `data_dir` path argument. |
| `efg.py` | EFG tensor calculation from strain tensors; Euler angle utilities |
| `hamiltonians.py` | Nuclear spin Hamiltonian constructors: `faraday_hamiltonian`, `voigt_hamiltonian`, `rf_hamiltonian`, `transition_rate` |
| `correlators.py` | Spin correlator functions (serial + parallel); `run_correlator_series` for full simulations |
| `strain.py` | Toy spring-mass strain model: lattice generators, energy minimisation, strain tensor calculation |
| `nmr.py` | NMR absorption spectra via transition rate sums |
| `nff.py` | Nuclear Frequency Focussing: Kraus operators, dephasing polarisation curves |
| `machine_gun.py` | CSMG / "machine gun" cluster-state protocol |
| `plot.py` | All matplotlib visualisation, separated from computation |

### Simulation pipeline

1. **Load strain data** (`qdot.io.load_sokolov_data`) — ε_xx, ε_xz, ε_zz from the Sokolov dataset (DOI: 10.1103/PhysRevB.93.045301).
2. **Calculate EFG tensors** (`qdot.efg.calculate_efg`) — strain → η (biaxiality), V_XX/V_YY/V_ZZ, Euler angles per site. Save with `qdot.io.save_efg`.
3. **Build Hamiltonians** (`qdot.hamiltonians`) — Zeeman + quadrupolar terms in Faraday or Voigt geometry, rotated to the lab frame via Euler angles.
4. **Calculate correlators** (`qdot.correlators.run_correlator_series`) — spin-spin time correlator averaged over the dot, parallelised over sites.
5. **NMR spectra** (`qdot.nmr.absorption_spectrum`) — transition rates summed over eigenstates as a function of RF frequency.

### Physics context

- Four nuclear species: Ga69, Ga71, As75, In115.
- Faraday geometry: static B along z; Zeeman term on I_z.
- Voigt geometry: static B along x; Zeeman term on I_x.
- Hamiltonians are QuTiP `Qobj` objects.
- `use_sundfors=True` selects the older Sundfors (1974) gradient-elastic tensor values; default is Checkhovich.

### Archive

The original research scripts live under `archive/`. They are historical reference only — do not review, modify, or flag issues in them. The `qdot/` package supersedes all of them.
