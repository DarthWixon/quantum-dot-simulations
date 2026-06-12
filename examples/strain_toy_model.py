"""
Strain toy model examples.

Demonstrates the spring-mass strain model from qdot.strain across three
lattice configurations. Run from the repo root:

    python examples/strain_toy_model.py

Output: six PNG files in outputs/examples/

Physics background
------------------
The model places atoms on a 2D grid connected by springs. Each bond has a
species-dependent stiffness (from Vurgaftman 2001) and natural length. When
Indium replaces Gallium its natural bond length to As is 6% longer (1.06 vs
1.00), so the surrounding lattice is forced to stretch — this is the origin of
the strain field around an InAs inclusion in GaAs.

scipy.optimize.minimize finds the lowest-energy atomic positions, then
qdot.strain.strain_tensor computes the symmetric Cauchy–Green strain tensor at
each site from the deformation gradient.

Three scenarios
---------------
1. Pure GaAs      — baseline; strain should be negligible everywhere.
2. Single In atom — local radial distortion around the In site.
3. 5×5 In block   — strong interface strain where InAs meets GaAs.
"""

import matplotlib

matplotlib.use("Agg")

import pathlib
import numpy as np

from qdot.strain import run_strain_simulation, strain_tensor, block_in_positions
from qdot.plot import plot_strain_lattice, plot_strain_tensors

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 15×15 gives good visual resolution in a few seconds.
N = 15


def run_and_save(label, lattice_type, in_positions=None):
    print(f"Running: {label}...")
    species, unstrained, strained, t, row_err, col_err = run_strain_simulation(
        N, N, lattice_type=lattice_type, in_positions=in_positions or []
    )
    tensors = strain_tensor(strained, unstrained)

    lattice_path = OUTPUT_DIR / f"strain_toy_{label}_lattice.png"
    tensor_path = OUTPUT_DIR / f"strain_toy_{label}_tensor.png"

    plot_strain_lattice(
        species, unstrained, strained, real_atoms=True, save_path=lattice_path
    )
    plot_strain_tensors(tensors, save_path=tensor_path)

    print(f"  minimisation: {t}s  |  row error: {row_err}%  |  col error: {col_err}%")
    print(f"  strain range: [{tensors.min():.4f}, {tensors.max():.4f}]")
    print(f"  saved: {lattice_path.name}, {tensor_path.name}")


# ---------------------------------------------------------------------------
# Scenario 1: Pure GaAs — no Indium, no strain
# ---------------------------------------------------------------------------
run_and_save("gaas", lattice_type=0)

# ---------------------------------------------------------------------------
# Scenario 2: Single In atom at the lattice centre
# ---------------------------------------------------------------------------
run_and_save("single_in", lattice_type=1)

# ---------------------------------------------------------------------------
# Scenario 3: 5×5 block of In atoms centred in the lattice
# ---------------------------------------------------------------------------
in_pos = block_in_positions([5, 5], [10, 10])
run_and_save("in_block", lattice_type=2, in_positions=in_pos)

print(f"\nAll graphs saved to {OUTPUT_DIR.resolve()}")
