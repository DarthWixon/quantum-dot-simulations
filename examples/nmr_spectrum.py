"""
NMR absorption spectrum for Ga69 in Faraday geometry.

Demonstrates computing a single-site NMR spectrum directly from physical
constants, without pre-computed EFG files. Two panels are produced:

    Left  — spectrum at 5 T: intensity vs RF frequency.
    Right — 2D colour map across 1–8 T: each row is one spectrum, colour
            encodes intensity. This is the type of plot experimentalists use
            to identify and track resonance lines as a function of field.

Output: claude-test-graphs/nmr_spectrum.png

Physics
-------
Ga69 has spin I = 3/2 and a Zeeman splitting of 10.22 MHz/T. In a strained
quantum dot the quadrupolar interaction shifts the three allowed transitions
(Δm = ±1) away from the bare Larmor frequency by ±ν_Q, where ν_Q depends on
the local electric field gradient.

We use η = 0.1 (mild biaxiality) and V_ZZ = 1.0 (normalised units), then
scale qcc·V_ZZ to set a representative quadrupolar coupling of ~3 MHz — in
the middle of the range measured in GaAs dots.

Euler angles (0, 0, 0) place the EFG principal axis along z, so the site is
exactly on-axis.
"""

import matplotlib

matplotlib.use("Agg")

import logging
import pathlib
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from qdot.isotopes import species_dict
from qdot.hamiltonians import faraday_hamiltonian, rf_hamiltonian, transition_rate

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
SPECIES = "Ga69"
ga69 = species_dict[SPECIES]

SPIN = ga69["particle_spin"]  # 3/2
ZEEMAN_PER_TESLA = ga69["zeeman_frequency_per_tesla"]  # 10.22e6 Hz/T

# Quadrupolar coupling parameter Q passed directly to the Hamiltonian.
# For spin-3/2 the two satellites appear at ±6Q from the Larmor frequency,
# so Q = 0.5 MHz gives a satellite separation of ±3 MHz — representative of
# lightly strained GaAs quantum dots.
QUAD_COUPLING_HZ = 0.5e6  # Hz

ETA = 0.1  # biaxiality
ALPHA, BETA, GAMMA = 0.0, 0.0, 0.0  # on-axis site
RF_FIELD = 5e-3  # Tesla — RF field amplitude

SINGLE_FIELD = 5.0  # T — for the left panel
FIELD_SWEEP = np.linspace(1.0, 8.0, 10)  # T — for the right panel

FREQ_HALF_SPAN = 5.0e6  # Hz each side of the Larmor frequency
N_FREQ = 500

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Spectrum calculation
# ---------------------------------------------------------------------------


def _rf_freq_range(applied_field: float, n_points: int) -> np.ndarray:
    """Return RF frequency array centred on the Larmor frequency for SPECIES."""
    larmor = ZEEMAN_PER_TESLA * applied_field
    return np.linspace(larmor - FREQ_HALF_SPAN, larmor + FREQ_HALF_SPAN, n_points)


def compute_spectrum(applied_field: float, rf_freq_list: np.ndarray) -> np.ndarray:
    """
    Compute the NMR absorption spectrum at a single lattice site.

    Builds the Faraday Hamiltonian from physical constants directly,
    diagonalises it, and sums Lorentzian transition rates over all
    state pairs for each RF frequency.

    Args:
        applied_field: Static magnetic field in Tesla.
        rf_freq_list: RF frequencies to evaluate (Hz).

    Returns:
        Absorption intensity at each RF frequency, shape (len(rf_freq_list),).
    """
    zeeman_term = ZEEMAN_PER_TESLA * applied_field
    quad_term = QUAD_COUPLING_HZ

    H = faraday_hamiltonian(zeeman_term, quad_term, ETA, SPIN, ALPHA, BETA, GAMMA)
    H = H.tidyup()
    H_rf = rf_hamiltonian(SPIN, RF_FIELD, 0.0, 0.0)

    eigenenergies, eigenvectors = H.eigenstates()
    eigenenergies = np.real_if_close(eigenenergies)
    index_list = np.arange(len(eigenenergies))

    rates = np.zeros(len(rf_freq_list))
    for r, rf_freq in enumerate(rf_freq_list):
        for pair in permutations(index_list, 2):
            rates[r] += transition_rate(
                H_rf,
                eigenvectors[pair[0]],
                eigenvectors[pair[1]],
                eigenenergies[pair[0]],
                eigenenergies[pair[1]],
                rf_freq,
            )
    return rates


# ---------------------------------------------------------------------------
# Single-field spectrum
# ---------------------------------------------------------------------------
logger.info("Computing single-field spectrum at %.1f T ...", SINGLE_FIELD)
rf_single = _rf_freq_range(SINGLE_FIELD, N_FREQ)
intensity_single = compute_spectrum(SINGLE_FIELD, rf_single)

# ---------------------------------------------------------------------------
# Field sweep (2D colour map)
# ---------------------------------------------------------------------------
logger.info("Computing field sweep (%d steps, 1–8 T) ...", len(FIELD_SWEEP))
sweep_matrix = np.zeros((len(FIELD_SWEEP), N_FREQ))

for i, B in enumerate(FIELD_SWEEP):
    logger.info("  B = %.2f T", B)
    rf_row = _rf_freq_range(B, N_FREQ)
    sweep_matrix[i] = compute_spectrum(B, rf_row)

# For the colour map each row is computed at a different Larmor frequency, so
# we use frequency offset from the Larmor frequency as the x-axis.
offset_mhz = np.linspace(-FREQ_HALF_SPAN, FREQ_HALF_SPAN, N_FREQ) / 1e6

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5))

# Left panel: single spectrum at SINGLE_FIELD
rf_mhz = rf_single / 1e6
ax_left.plot(rf_mhz, intensity_single, linewidth=1.0, color="steelblue")
ax_left.set_xlabel("RF frequency (MHz)")
ax_left.set_ylabel("Absorption intensity (arb. units)")
ax_left.set_title(f"Ga69 Faraday spectrum at {SINGLE_FIELD:.0f} T")

# Right panel: 2D colour map — rows = field steps, columns = frequency offset
img = ax_right.imshow(
    sweep_matrix,
    origin="lower",
    aspect="auto",
    extent=[offset_mhz[0], offset_mhz[-1], FIELD_SWEEP[0], FIELD_SWEEP[-1]],
    cmap="viridis",
)
cbar = fig.colorbar(img, ax=ax_right, label="Absorption intensity (arb. units)")
ax_right.set_xlabel("RF frequency offset from Larmor (MHz)")
ax_right.set_ylabel("Applied field (T)")
ax_right.set_title("Ga69 Faraday — field sweep")

plt.tight_layout()
save_path = OUTPUT_DIR / "nmr_spectrum.png"
plt.savefig(save_path, dpi=150)
logger.info("Saved: %s", save_path.resolve())
