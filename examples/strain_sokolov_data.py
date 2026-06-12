"""
Sokolov strain data examples.

Loads and visualises the experimental strain field from the Sokolov dataset:
    Sokolov et al., Phys. Rev. B 93, 045301 (2016)
    DOI: 10.1103/PhysRevB.93.045301

Run from the repo root:

    python examples/strain_sokolov_data.py

Required data files
-------------------
Set DATA_DIR below to the directory containing:
    full_epsilon_xx.txt   — ε_xx component, 1600×1600 floats
    full_epsilon_xy.txt   — ε_xy component, 1600×1600 floats
    full_epsilon_yy.txt   — ε_yy component, 1600×1600 floats

These files are not included in the repository. Contact the authors of the
paper or the original researcher for access.

Output
------
Three PNG files in outputs/examples/ (if data is present):
    strain_sokolov_full_region.png   — ε_xx, ε_xz, ε_zz across the full dot
    strain_sokolov_dot_only.png      — same components cropped to the dot
    strain_sokolov_shear_hist.png    — shear strain distribution in the dot
"""

import matplotlib

matplotlib.use("Agg")

import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from qdot.io import load_strain_data

# ---------------------------------------------------------------------------
# Configuration — set this to the directory containing the Sokolov .txt files
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("path/to/sokolov/data")

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Two regions used throughout the original analysis.
# [left, right, top, bottom] in pixel coordinates of the 1600×1600 source.
FULL_REGION = [100, 1200, 200, 1000]
DOT_REGION = [100, 1200, 439, 880]

# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------
required = [
    DATA_DIR / f
    for f in ("full_epsilon_xx.txt", "full_epsilon_xy.txt", "full_epsilon_yy.txt")
]
missing = [p for p in required if not p.exists()]
if missing:
    print("Sokolov data files not found. Set DATA_DIR in this script.")
    print("Missing:")
    for p in missing:
        print(f"  {p}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading Sokolov strain data (large files — this may take a moment)...")
full_xx, full_xz, full_zz = load_strain_data(DATA_DIR, FULL_REGION)
dot_xx, dot_xz, dot_zz = load_strain_data(DATA_DIR, DOT_REGION)
print(f"  full region shape: {full_xx.shape}")
print(f"  dot region shape:  {dot_xx.shape}")


def _strain_map(xx, xz, zz, title, save_path):
    """Three-panel strain component map matching the Sokolov paper layout."""
    components = [
        (xx, r"$\epsilon_{xx}$"),
        (zz, r"$\epsilon_{zz}$"),
        (xz, r"$\epsilon_{xz}$"),
    ]

    all_vals = np.concatenate([a.ravel() for a, _ in components])
    vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)

    cmap = cm.RdBu
    norm = Normalize(vmin, vmax)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (data, label) in zip(axs, components):
        ax.imshow(data, cmap=cmap, norm=norm, origin="upper")
        ax.axis("off")
        ax.set_title(label, fontsize=14)

    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
        label="Strain",
    )
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Graph 1: Full dot region strain map
# ---------------------------------------------------------------------------
print("Plotting full region strain map...")
_strain_map(
    full_xx,
    full_xz,
    full_zz,
    "Sokolov strain — full region",
    OUTPUT_DIR / "strain_sokolov_full_region.png",
)

# ---------------------------------------------------------------------------
# Graph 2: Dot-only strain map
# ---------------------------------------------------------------------------
print("Plotting dot-only strain map...")
_strain_map(
    dot_xx,
    dot_xz,
    dot_zz,
    "Sokolov strain — dot region",
    OUTPUT_DIR / "strain_sokolov_dot_only.png",
)

# ---------------------------------------------------------------------------
# Graph 3: Shear strain histogram
# ---------------------------------------------------------------------------
print("Plotting shear strain histogram...")
fig, ax = plt.subplots(figsize=(8, 5))

# The Sokolov paper distinguishes ε_xz as measured from the symmetric
# shear strain 2|ε_xz|.
ax.hist(
    dot_xz.ravel(),
    bins="auto",
    histtype="step",
    density=True,
    label=r"$\epsilon_{xz}$ (as measured)",
)
ax.hist(
    2 * np.abs(dot_xz).ravel(),
    bins="auto",
    histtype="step",
    density=True,
    label=r"$2|\epsilon_{xz}|$ (symmetric shear)",
)
ax.set_xlabel("Shear strain")
ax.set_ylabel("Probability density")
ax.set_title("Shear strain distribution in dot region")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "strain_sokolov_shear_hist.png")
plt.close(fig)
print("  saved: strain_sokolov_shear_hist.png")

print(f"\nAll graphs saved to {OUTPUT_DIR.resolve()}")
