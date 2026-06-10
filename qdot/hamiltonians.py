"""
Hamiltonian constructors for nuclear spins in InGaAs quantum dots.

All Hamiltonians are QuTiP Qobj objects. Frequencies are in Hz throughout;
the Zeeman and quadrupolar terms should be passed in the same units.
"""

import numpy as np
import qutip


def spin_rotator(
    alpha: float, beta: float, gamma: float, initial_spin: qutip.Qobj
) -> qutip.Qobj:
    """
    Rotate a quantum spin operator using Euler angles (X-Y-Z convention).

    Args:
        alpha (float): Rotation about X axis, in radians.
        beta (float): Rotation about Y axis, in radians.
        gamma (float): Rotation about Z axis, in radians.
        initial_spin (Qobj): Spin operator to rotate.

    Returns:
        Qobj: Rotated spin operator.
    """
    spin = (initial_spin.dims[0][0] - 1) / 2
    R = (
        (-1j * alpha * qutip.jmat(spin, "x")).expm()
        * (-1j * beta * qutip.jmat(spin, "y")).expm()
        * (-1j * gamma * qutip.jmat(spin, "z")).expm()
    )
    return R * initial_spin * R.dag()


def _nuclear_hamiltonian(
    zeeman_term: float,
    quadrupolar_term: float,
    biaxiality: float,
    particle_spin: float,
    alpha: float,
    beta: float,
    gamma: float,
    x_axis: str,
    y_axis: str,
    z_axis: str,
) -> qutip.Qobj:
    """
    Shared Zeeman + quadrupolar Hamiltonian body.

    The axis labels map the geometry's lab-frame x/y/z onto qutip.jmat axes;
    Faraday uses the identity mapping, Voigt swaps x and z.
    """
    I_x = qutip.jmat(particle_spin, x_axis)
    I_y = qutip.jmat(particle_spin, y_axis)
    I_z = qutip.jmat(particle_spin, z_axis)

    I_x_q = spin_rotator(alpha, beta, gamma, I_x)
    I_y_q = spin_rotator(alpha, beta, gamma, I_y)
    I_z_q = spin_rotator(alpha, beta, gamma, I_z)

    return zeeman_term * I_z + quadrupolar_term * (
        2 * I_z_q**2 + (biaxiality - 1) * I_x_q**2 - (biaxiality + 1) * I_y_q**2
    )


def faraday_hamiltonian(
    zeeman_term: float,
    quadrupolar_term: float,
    biaxiality: float,
    particle_spin: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> qutip.Qobj:
    """
    Nuclear spin Hamiltonian in Faraday geometry (static B along z).

    H = Z·Iz + Q·(2·Iz'² + (η-1)·Ix'² - (η+1)·Iy'²)

    Unprimed operators are in the lab frame; primed operators are in the
    principal axis frame of the nucleus, reached via Euler angles (alpha, beta, gamma).

    Args:
        zeeman_term (float): Zeeman interaction strength Z (Hz).
        quadrupolar_term (float): Quadrupolar interaction strength Q (Hz).
        biaxiality (float): Nuclear biaxiality η.
        particle_spin (float): Nuclear spin quantum number.
        alpha, beta, gamma (float): Euler angles (radians) rotating lab → PAF.

    Returns:
        Qobj: Nuclear spin Hamiltonian.
    """
    return _nuclear_hamiltonian(
        zeeman_term,
        quadrupolar_term,
        biaxiality,
        particle_spin,
        alpha,
        beta,
        gamma,
        "x",
        "y",
        "z",
    )


def voigt_hamiltonian(
    zeeman_term: float,
    quadrupolar_term: float,
    biaxiality: float,
    particle_spin: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> qutip.Qobj:
    """
    Nuclear spin Hamiltonian in Voigt geometry (static B along x).

    H = Z·Ix + Q·(2·Iz'² + (η-1)·Ix'² - (η+1)·Iy'²)

    Args:
        zeeman_term (float): Zeeman interaction strength Z (Hz).
        quadrupolar_term (float): Quadrupolar interaction strength Q (Hz).
        biaxiality (float): Nuclear biaxiality η.
        particle_spin (float): Nuclear spin quantum number.
        alpha, beta, gamma (float): Euler angles (radians) rotating lab → PAF.

    Returns:
        Qobj: Nuclear spin Hamiltonian.
    """
    # In Voigt geometry, x ↔ z labels are swapped relative to Faraday
    return _nuclear_hamiltonian(
        zeeman_term,
        quadrupolar_term,
        biaxiality,
        particle_spin,
        alpha,
        beta,
        gamma,
        "z",
        "y",
        "x",
    )


def rf_hamiltonian(
    particle_spin: float,
    B_x: float,
    B_y: float,
    B_z: float,
    gamma_n: float = 1,
) -> qutip.Qobj:
    """
    RF perturbation Hamiltonian for NMR transition rate calculations.

    H_rf = -γ·(Bx·Ix + By·Iy + Bz·Iz)

    Note: this does not include the cos(ωt) time dependence; it gives the
    coupling structure used to compute transition matrix elements.

    Args:
        particle_spin (float): Nuclear spin quantum number.
        B_x, B_y, B_z (float): RF field components.
        gamma_n (float): Nuclear gyromagnetic ratio (arbitrary units). Default 1.

    Returns:
        Qobj: RF Hamiltonian.
    """
    I_x = qutip.jmat(particle_spin, "x")
    I_y = qutip.jmat(particle_spin, "y")
    I_z = qutip.jmat(particle_spin, "z")
    return -gamma_n * (B_x * I_x + B_y * I_y + B_z * I_z)


def transition_rate(
    mixing_hamiltonian: qutip.Qobj,
    init_state: qutip.Qobj,
    final_state: qutip.Qobj,
    E_init: float,
    E_final: float,
    omega_rf: float | np.ndarray,
    delta: float = 10e3,
) -> float | np.ndarray:
    """
    Transition rate between two eigenstates under an RF perturbation.

    Uses a Lorentzian lineshape centred at (E_final - E_init).

    Args:
        mixing_hamiltonian (Qobj): Typically the RF Hamiltonian.
        init_state (Qobj): Initial eigenstate ket.
        final_state (Qobj): Final eigenstate ket.
        E_init, E_final (float): Eigenenergies (Hz).
        omega_rf (float or ndarray): RF frequency (Hz). An array evaluates the
            rate at every frequency with a single matrix-element computation.
        delta (float): Lorentzian half-width (Hz). Default 10 kHz.

    Returns:
        float or ndarray: Transition rate, matching the shape of omega_rf.
    """
    prob = np.abs(mixing_hamiltonian.matrix_element(final_state, init_state)) ** 2
    lorentzian = (2 * delta) / ((E_final - E_init - omega_rf) ** 2 + delta**2)
    return np.real_if_close(prob * lorentzian)
