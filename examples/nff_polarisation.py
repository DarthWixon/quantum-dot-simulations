"""
Nuclear Frequency Focussing (NFF) polarisation curves.

Two panels:
  Left  — z-polarisation vs dephasing strength γ for three pulse phases.
           γ=1 is no dephasing; γ=0.5 is maximum dephasing.
  Right — z-polarisation in the zero-dephasing limit as the pulse phase
           sweeps 0 → 2π.

Output: examples/output/nff_polarisation.png
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qdot.nff import dephasing_polarisation_curve, non_dephased_polarisation

Q0 = 0.8

phases_left = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
phase_labels = [r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"]

phases_right = np.linspace(0, 2 * np.pi, 100)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

for phase, label in zip(phases_left, phase_labels):
    gamma_values, pol_values = dephasing_polarisation_curve(Q0, phase)
    ax_left.plot(gamma_values, np.real(pol_values), label=f"phase = {label}")

ax_left.set_xlabel(r"Dephasing parameter $\gamma$")
ax_left.set_ylabel("z-polarisation")
ax_left.set_title("Dephasing polarisation curves")
ax_left.legend()

pol_right = np.real([non_dephased_polarisation(Q0, ph) for ph in phases_right])

ax_right.plot(phases_right, pol_right)
ax_right.set_xlabel("Pulse phase (rad)")
ax_right.set_ylabel("z-polarisation")
ax_right.set_title("No-dephasing limit")
ax_right.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax_right.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

plt.tight_layout()

output_dir = pathlib.Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / "nff_polarisation.png", dpi=150)
