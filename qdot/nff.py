"""
Nuclear Frequency Focussing (NFF) simulations.

Models dephasing under repeated optical pulse sequences using Kraus operators
and the QuTiP superoperator formalism.
"""

import numpy as np
import qutip


def dephasing_kraus_operator(dephasing_value):
    """
    Superoperator for phase damping with a given dephasing strength.

    Args:
        dephasing_value (float): Dephasing parameter in [0.5, 1].
            1 = no dephasing, 0.5 = maximum dephasing.

    Returns:
        Qobj: Superoperator representing phase damping.
    """
    K_0 = qutip.Qobj(dephasing_value * np.array([[1, 0], [0, 1]]))
    K_1 = qutip.Qobj(np.sqrt(1 - dephasing_value) * np.array([[1, 0], [0, -1]]))
    return qutip.superop_reps.kraus_to_super([K_0, K_1])


def pulse_kraus_operator(q0, phase):
    """
    Superoperator for an optical pulse with coherence q0 and phase shift.

    Args:
        q0 (float): Pulse coherence parameter in [0, 1]. 1 = perfect pulse.
        phase (float): Phase of the pulse in radians.

    Returns:
        Qobj: Superoperator representing the pulse.
    """
    E_0 = qutip.Qobj(np.array([[1, 0], [0, q0 * np.exp(1j * phase)]]))
    E_1 = qutip.Qobj(np.array([[0, np.sqrt((1 - q0**2) / 2)], [0, 0]]))
    E_2 = qutip.Qobj(np.array([[0, 0], [0, np.sqrt((1 - q0**2) / 2)]]))
    return qutip.superop_reps.kraus_to_super([E_0, E_1, E_2])


def z_polarisation(density_matrix):
    """Z-axis polarisation of a density matrix (in X basis)."""
    Z_in_X = qutip.Qobj(np.array([[0, 1], [1, 0]]))
    return (density_matrix * Z_in_X).tr()


def dephasing_polarisation_curve(q0, phase, n_gammas=100):
    """
    Z polarisation as a function of dephasing strength after one pulse.

    Args:
        q0 (float): Pulse coherence parameter.
        phase (float): Pulse phase in radians.
        n_gammas (int): Number of dephasing values to evaluate.

    Returns:
        dephasing_list (ndarray): Dephasing values from 1 (none) to 0.5 (max).
        z_pol_list (ndarray): Z polarisation at each dephasing value.
    """
    initial_dm = qutip.Qobj(np.array([[1, 1], [0, 1]]))
    initial_vec = qutip.superoperator.operator_to_vector(initial_dm)
    pulse_op = pulse_kraus_operator(q0, phase)

    dephasing_list = np.linspace(1, 0.5, n_gammas)
    z_pol_list = np.zeros(n_gammas, dtype="complex")

    for d, deph in enumerate(dephasing_list):
        deph_op = dephasing_kraus_operator(deph)
        final_dm = qutip.superoperator.vector_to_operator(deph_op * pulse_op * initial_vec)
        z_pol_list[d] = z_polarisation(final_dm)

    return dephasing_list, np.real_if_close(z_pol_list)


def non_dephased_polarisation(q0, phase):
    """
    Z polarisation after a single perfect (undephased) pulse.

    Args:
        q0 (float): Pulse coherence parameter.
        phase (float): Pulse phase in radians.

    Returns:
        float: Z polarisation.
    """
    initial_dm = qutip.Qobj(np.array([[1, 1], [0, 1]]))
    initial_vec = qutip.superoperator.operator_to_vector(initial_dm)
    pulse_op = pulse_kraus_operator(q0, phase)
    final_dm = qutip.superoperator.vector_to_operator(pulse_op * initial_vec)
    return np.real_if_close(z_polarisation(final_dm))
