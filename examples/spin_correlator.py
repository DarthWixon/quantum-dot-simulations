"""
Spin-spin time correlator <I_z(t) I_z(0)> for Ga69 in InGaAs.

Two contrasting lattice sites are compared:
  - weakly strained: small quadrupolar coupling, eta ~ 0
  - strongly strained: larger quadrupolar coupling, eta ~ 0.3

Left panel: Faraday geometry (B along z).
Right panel: Voigt geometry (B along x).

Output: claude-test-graphs/spin_correlator.png
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip
import scipy.constants as const

from qdot.correlators import site_correlator, spin_correlator
from qdot.hamiltonians import voigt_hamiltonian
from qdot.isotopes import species_dict

# --- Ga69 physical parameters ---
species = species_dict["Ga69"]
spin = species["particle_spin"]
zeeman_per_tesla = species["zeeman_frequency_per_tesla"]
Q = species["quadrupole_moment"]
qcc = (3 * const.e * Q) / (2 * const.h * spin * (2 * spin - 1))

# --- Simulation parameters ---
applied_field = 5.0  # T
euler_angles = (0.0, 0.0, 0.0)

# Larmor period ~ 1 / (zeeman_per_tesla * field)
larmor_period = 1.0 / (zeeman_per_tesla * applied_field)
t_max = 8 * larmor_period
times = np.linspace(0, t_max, 200)
times_ns = times * 1e9

# --- Two site configurations ---
# Weakly strained: V_ZZ chosen so quadrupolar << Zeeman; eta ~ 0
# Strongly strained: quadrupolar coupling ~30% of Zeeman; eta ~ 0.3
sites = {
    "weakly strained": dict(V_ZZ=1e12, biaxiality=0.0),
    "strongly strained": dict(V_ZZ=8e12, biaxiality=0.3),
}

colors = {"weakly strained": "steelblue", "strongly strained": "firebrick"}
linestyles = {"weakly strained": "-", "strongly strained": "--"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for label, params in sites.items():
    V_ZZ = params["V_ZZ"]
    biaxiality = params["biaxiality"]

    # Faraday: site_correlator handles Hamiltonian construction internally
    faraday_vals = [
        site_correlator(
            t, zeeman_per_tesla, qcc, spin,
            biaxiality, euler_angles, V_ZZ, applied_field, "z",
        )
        for t in times
    ]

    # Voigt: build Hamiltonian manually, then call spin_correlator
    alpha, beta, gamma = euler_angles
    zeeman_term = zeeman_per_tesla * applied_field
    quadrupolar_term = qcc * V_ZZ
    H_voigt = voigt_hamiltonian(
        zeeman_term, quadrupolar_term, biaxiality, spin, alpha, beta, gamma
    )
    voigt_vals = [spin_correlator(t, H_voigt, "z") for t in times]

    axes[0].plot(
        times_ns, faraday_vals,
        color=colors[label], ls=linestyles[label], label=label,
    )
    axes[1].plot(
        times_ns, voigt_vals,
        color=colors[label], ls=linestyles[label], label=label,
    )

for ax, title in zip(axes, ["Faraday geometry", "Voigt geometry"]):
    ax.set_xlabel("Time (ns)")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

axes[0].set_ylabel(r"$\langle I_z(t)\, I_z(0) \rangle$")

plt.tight_layout()

output_dir = pathlib.Path("claude-test-graphs")
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / "spin_correlator.png", dpi=150)
