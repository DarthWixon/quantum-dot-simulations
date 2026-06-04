"""
EFG species comparison example.

Runs the toy spring-mass strain model with a single indium atom at the centre
of a 12×12 GaAs lattice, then computes the EFG tensors for all four nuclear
species and plots V_ZZ, η, and β side-by-side.

Run from the repo root:

    python examples/efg_species_comparison.py

Output: claude-test-graphs/efg_species_comparison.png
"""

import matplotlib
matplotlib.use("Agg")

import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from qdot.strain import run_strain_simulation, strain_tensor
from qdot.efg import calculate_efg_vectorised

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = pathlib.Path("claude-test-graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

SPECIES = ["Ga69", "Ga71", "As75", "In115"]
N = 12


def main() -> None:
    log.info("Running strain simulation (%dx%d, single In centre)...", N, N)
    species_array, unstrained, strained, t, row_err, col_err = run_strain_simulation(
        N, N, lattice_type=1
    )
    log.info("  minimisation: %ss  |  row error: %s%%  |  col error: %s%%",
             t, row_err, col_err)

    tensors = strain_tensor(strained, unstrained)
    xx = tensors[:, :, 0, 0]
    xz = tensors[:, :, 0, 1]
    zz = tensors[:, :, 1, 1]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(
        "EFG components across nuclear species — single In atom in 12×12 GaAs",
        fontsize=13,
    )

    row_labels = ["$V_{ZZ}$", r"$\eta$ (biaxiality)", r"$\beta$ (Euler angle, rad)"]

    for col, sp in enumerate(SPECIES):
        log.info("Computing EFG for %s...", sp)
        eta, V_XX, V_YY, V_ZZ, euler_angles = calculate_efg_vectorised(
            sp, xx, xz, zz
        )
        beta = euler_angles[:, :, 1]

        data_rows = [V_ZZ, eta, beta]

        for row, data in enumerate(data_rows):
            ax = axes[row, col]
            im = ax.imshow(data, origin="lower", aspect="equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(sp, fontsize=11)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    out_path = OUTPUT_DIR / "efg_species_comparison.png"
    fig.savefig(out_path, dpi=150)
    log.info("Saved: %s", out_path.resolve())


if __name__ == "__main__":
    main()
