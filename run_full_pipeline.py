"""
Full pipeline simulation using the Sokolov experimental strain dataset.
Sokolov et al., Phys. Rev. B 93, 045301 (2016).

Steps
-----
1. Load strain data (dot region, full resolution)
2. Calculate EFG tensors for all four nuclear species
3. Plot strain field maps
4. Plot EFG spatial maps and histograms
5. NMR absorption spectra (Faraday geometry, centre site)
6. Spin–spin time correlator for Ga69, site-averaged

Outputs are saved to outputs/full_pipeline/.
"""

import multiprocessing
import pathlib
import time

# macOS defaults to 'spawn' for new processes, which re-imports this script in
# each worker and triggers another Pool() call. 'fork' copies memory directly,
# avoiding the re-import.
multiprocessing.set_start_method("fork")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qdot.correlators import run_correlator_series
from qdot.efg import calculate_efg_vectorised
from qdot.io import SOKOLOV_DOT_REGION, load_strain_data, save_efg
from qdot.isotopes import species_dict
from qdot.nmr import absorption_spectrum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO / "real_data"
OUTPUT_DIR = REPO / "outputs" / "full_pipeline"
EFG_DIR = OUTPUT_DIR / "efg"
EFG_DIR.mkdir(parents=True, exist_ok=True)

DOT_REGION = SOKOLOV_DOT_REGION  # [100, 1200, 439, 880]
ALL_SPECIES = ["Ga69", "Ga71", "As75", "In115"]
APPLIED_FIELD = 5.0  # T

# step_size=1  → full dot (441×1100) for EFG maps and NMR
# CORR_STEP=50 → sparse grid (9×22=198 sites) for the correlator
EFG_STEP = 1
CORR_STEP = 50
N_CORR_TIMES = 30

SPECIES_COLOURS = {
    "Ga69": "steelblue",
    "Ga71": "firebrick",
    "As75": "forestgreen",
    "In115": "darkorange",
}
NMR_FREQ_SPAN = {sp: 10e6 for sp in ALL_SPECIES}
NMR_FREQ_SPAN["In115"] = 20e6  # spin-9/2 has wider satellite structure

timings: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Step 1 — Load strain data
# ---------------------------------------------------------------------------
print("Step 1 — loading strain data...")
t0 = time.perf_counter()
xx, xz, zz = load_strain_data(DATA_DIR, DOT_REGION, step_size=EFG_STEP)
timings["1. Load strain"] = time.perf_counter() - t0
n_rows, n_cols = xx.shape
print(f"  {n_rows}×{n_cols} sites  ({timings['1. Load strain']:.1f} s)")


# ---------------------------------------------------------------------------
# Step 2 — Calculate EFG for all species; save archives
# ---------------------------------------------------------------------------
efg_data: dict[str, tuple] = {}
print("Step 2 — calculating EFG tensors...")
for sp in ALL_SPECIES:
    print(f"  {sp}...", end="", flush=True)
    t0 = time.perf_counter()
    eta, V_XX, V_YY, V_ZZ, euler = calculate_efg_vectorised(sp, xx, xz, zz)
    efg_data[sp] = (eta, V_XX, V_YY, V_ZZ, euler)
    save_efg(EFG_DIR, sp, DOT_REGION, EFG_STEP, eta, V_XX, V_YY, V_ZZ, euler)
    elapsed = time.perf_counter() - t0
    timings[f"2. EFG {sp}"] = elapsed
    print(f"  {elapsed:.1f} s")

# Save subsampled archives for the correlator step (CORR_STEP).
# Subsampling eta[::S, ::S] picks the same sites as loading strain at step_size=S,
# so the physics is identical without a second disk read.
for sp in ALL_SPECIES:
    eta, V_XX, V_YY, V_ZZ, euler = efg_data[sp]
    save_efg(
        EFG_DIR,
        sp,
        DOT_REGION,
        CORR_STEP,
        eta[::CORR_STEP, ::CORR_STEP],
        V_XX[::CORR_STEP, ::CORR_STEP],
        V_YY[::CORR_STEP, ::CORR_STEP],
        V_ZZ[::CORR_STEP, ::CORR_STEP],
        euler[::CORR_STEP, ::CORR_STEP, :],
    )
corr_n_sites = len(range(0, n_rows, CORR_STEP)) * len(range(0, n_cols, CORR_STEP))


# ---------------------------------------------------------------------------
# Step 3 — Strain field maps
# ---------------------------------------------------------------------------
print("Step 3 — plotting strain maps...")
t0 = time.perf_counter()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
cmap = "RdBu_r"
for ax, (data, label) in zip(
    axes,
    [
        (xx, r"$\varepsilon_{xx}$"),
        (xz, r"$\varepsilon_{xz}$"),
        (zz, r"$\varepsilon_{zz}$"),
    ],
):
    vmax = np.percentile(np.abs(data), 98)
    im = ax.imshow(
        data, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper", aspect="auto"
    )
    ax.set_title(label, fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="strain")
fig.suptitle("Sokolov dot — strain components", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "strain_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
timings["3. Plot strain"] = time.perf_counter() - t0
print(f"  saved strain_maps.png  ({timings['3. Plot strain']:.1f} s)")


# ---------------------------------------------------------------------------
# Step 4 — EFG spatial maps and histograms
# ---------------------------------------------------------------------------
print("Step 4 — plotting EFG distributions...")
t0 = time.perf_counter()

# 4a: spatial maps (η and |V_ZZ| for each species)
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for col, sp in enumerate(ALL_SPECIES):
    eta, _, _, V_ZZ, _ = efg_data[sp]
    colour = SPECIES_COLOURS[sp]

    ax_eta = axes[0, col]
    eta_vmax = np.nanpercentile(np.abs(eta), 98)
    im_e = ax_eta.imshow(
        eta, cmap="RdBu_r", vmin=-eta_vmax, vmax=eta_vmax, origin="upper", aspect="auto"
    )
    ax_eta.set_title(sp, fontsize=12)
    ax_eta.axis("off")
    plt.colorbar(im_e, ax=ax_eta, fraction=0.046, pad=0.04)

    ax_vzz = axes[1, col]
    vzz_vmax = np.percentile(np.abs(V_ZZ), 98)
    im_v = ax_vzz.imshow(
        V_ZZ / 1e20,
        cmap="RdBu_r",
        vmin=-vzz_vmax / 1e20,
        vmax=vzz_vmax / 1e20,
        origin="upper",
        aspect="auto",
    )
    ax_vzz.axis("off")
    plt.colorbar(im_v, ax=ax_vzz, fraction=0.046, pad=0.04)

axes[0, 0].set_ylabel(r"$\eta$", fontsize=12)
axes[1, 0].set_ylabel(r"$V_{ZZ}$ ($\times 10^{20}$ V/m²)", fontsize=12)
fig.text(0.01, 0.72, r"Biaxiality $\eta$", va="center", rotation=90, fontsize=11)
fig.text(
    0.01,
    0.28,
    r"$V_{ZZ}$ ($\times 10^{20}$ V/m²)",
    va="center",
    rotation=90,
    fontsize=11,
)
fig.suptitle("EFG spatial maps — Sokolov dot", fontsize=13, y=1.01)
plt.tight_layout(rect=[0.03, 0, 1, 1])
fig.savefig(OUTPUT_DIR / "efg_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4b: histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey="row")
for col, sp in enumerate(ALL_SPECIES):
    eta, _, _, V_ZZ, _ = efg_data[sp]
    colour = SPECIES_COLOURS[sp]

    axes[0, col].hist(
        eta[~np.isnan(eta)], bins=80, color=colour, alpha=0.8, density=True
    )
    axes[0, col].set_xlabel(r"$\eta$")
    axes[0, col].set_title(sp, fontsize=12)

    axes[1, col].hist(
        V_ZZ.ravel() / 1e20, bins=80, color=colour, alpha=0.8, density=True
    )
    axes[1, col].set_xlabel(r"$V_{ZZ}$ ($\times 10^{20}$ V/m²)")

axes[0, 0].set_ylabel("Density")
axes[1, 0].set_ylabel("Density")
fig.suptitle("EFG distributions — Sokolov dot", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "efg_histograms.png", dpi=150)
plt.close(fig)

timings["4. Plot EFG"] = time.perf_counter() - t0
print(f"  saved efg_maps.png + efg_histograms.png  ({timings['4. Plot EFG']:.1f} s)")


# ---------------------------------------------------------------------------
# Step 5 — NMR absorption spectra (centre site, Faraday geometry)
# ---------------------------------------------------------------------------
print("Step 5 — computing NMR spectra...")
t0 = time.perf_counter()

centre = (n_rows // 2, n_cols // 2)
N_FREQ = 500

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, sp in zip(axes, ALL_SPECIES):
    sp_info = species_dict[sp]
    larmor = sp_info["zeeman_frequency_per_tesla"] * APPLIED_FIELD
    span = NMR_FREQ_SPAN[sp]
    rf_freqs = np.linspace(larmor - span, larmor + span, N_FREQ)

    rates = absorption_spectrum(
        sp,
        APPLIED_FIELD,
        "Faraday",
        rf_freqs,
        centre,
        EFG_DIR,
        region_bounds=DOT_REGION,
    )

    ax.plot((rf_freqs - larmor) / 1e6, rates, lw=1.0, color=SPECIES_COLOURS[sp])
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_title(sp, fontsize=12)
    ax.set_xlabel("Offset from Larmor (MHz)")

axes[0].set_ylabel("Absorption (arb. units)")
fig.suptitle(
    f"NMR absorption — Faraday, {APPLIED_FIELD:.0f} T, "
    f"centre site ({centre[0]}, {centre[1]})",
    fontsize=12,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "nmr_spectra.png", dpi=150)
plt.close(fig)

timings["5. NMR spectra"] = time.perf_counter() - t0
print(f"  saved nmr_spectra.png  ({timings['5. NMR spectra']:.1f} s)")


# ---------------------------------------------------------------------------
# Step 6 — Spin correlator for Ga69 (site-averaged, Faraday)
# ---------------------------------------------------------------------------
print(
    f"Step 6 — computing Ga69 correlator ({corr_n_sites} sites, "
    f"{N_CORR_TIMES} times, 3 axes)..."
)
t0 = time.perf_counter()

ga69 = species_dict["Ga69"]
larmor_period = 1.0 / (ga69["zeeman_frequency_per_tesla"] * APPLIED_FIELD)
timerange = np.linspace(0, 8 * larmor_period, N_CORR_TIMES)

corr = run_correlator_series(
    EFG_DIR,
    timerange,
    APPLIED_FIELD,
    "Ga69",
    DOT_REGION,
    step_size=CORR_STEP,
)
timings["6. Correlator Ga69"] = time.perf_counter() - t0

times_ns = timerange * 1e9
axis_labels = [
    r"$\langle I_x(t)\,I_x(0)\rangle$",
    r"$\langle I_y(t)\,I_y(0)\rangle$",
    r"$\langle I_z(t)\,I_z(0)\rangle$",
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, label, row in zip(axes, axis_labels, corr):
    ax.plot(times_ns, row, lw=1.2, color="steelblue")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Time (ns)")
    ax.set_title(label, fontsize=12)

axes[0].set_ylabel("Correlator value")
fig.suptitle(
    f"Ga69 spin correlator — Faraday, {APPLIED_FIELD:.0f} T, "
    f"{corr_n_sites} sites (step={CORR_STEP})",
    fontsize=12,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "correlator_Ga69.png", dpi=150)
plt.close(fig)
print(f"  saved correlator_Ga69.png  ({timings['6. Correlator Ga69']:.1f} s)")


# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------
print()
print("=" * 52)
print(f"{'Step':<34} {'Time (s)':>8}  {'Time (min)':>9}")
print("-" * 52)
for step, t in timings.items():
    print(f"{step:<34} {t:>8.1f}  {t/60:>9.2f}")
print("=" * 52)
total = sum(timings.values())
print(f"{'TOTAL':<34} {total:>8.1f}  {total/60:>9.2f}")
print()
print(f"Outputs written to: {OUTPUT_DIR.resolve()}")
