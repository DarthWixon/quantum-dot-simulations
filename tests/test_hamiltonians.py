import numpy as np
import pytest
import qutip
from qdot.hamiltonians import (
    faraday_hamiltonian,
    rf_hamiltonian,
    transition_rate,
    voigt_hamiltonian,
    spin_rotator,
)


def test_faraday_hamiltonian_is_hermitian():
    H = faraday_hamiltonian(1.0, 0.1, 0.4, 1.5, 0.3, -0.5, 1.2)
    assert (H - H.dag()).norm() < 1e-10


def test_voigt_hamiltonian_is_hermitian():
    H = voigt_hamiltonian(1.0, 0.1, 0.4, 1.5, 0.3, -0.5, 1.2)
    assert (H - H.dag()).norm() < 1e-10


def test_faraday_pure_zeeman_eigenvalues():
    # No quadrupolar term, no rotation: eigenvalues should be m * zeeman_term
    # for m in {-3/2, -1/2, +1/2, +3/2} at spin-3/2
    H = faraday_hamiltonian(1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    eigs = np.sort(np.real(H.eigenenergies()))
    expected = np.array([-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(eigs, expected, atol=1e-10)


def test_faraday_and_voigt_differ_with_zeeman():
    # With only a Zeeman term they should give different eigenstructures
    H_f = faraday_hamiltonian(1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    H_v = voigt_hamiltonian(1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    assert (H_f - H_v).norm() > 1e-10


def test_spin_rotator_zero_rotation_is_identity():
    op = qutip.jmat(1.5, "z")
    result = spin_rotator(0.0, 0.0, 0.0, op)
    assert (result - op).norm() < 1e-10


def test_spin_rotator_2pi_is_identity():
    # Rotating a spin-3/2 operator by 2π should return the original (up to sign)
    op = qutip.jmat(1.5, "z")
    result = spin_rotator(2 * np.pi, 0.0, 0.0, op)
    # For integer representation, 2π gives +identity; for half-integer, -identity.
    # jmat(3/2) lives in a 4×4 space; the rotation acts on operators so sign cancels.
    assert (result - op).norm() < 1e-10


# ---------------------------------------------------------------------------
# voigt_hamiltonian
# ---------------------------------------------------------------------------


def test_voigt_pure_zeeman_eigenvalues():
    # Pure Zeeman in Voigt geometry uses I_x; eigenvalues are the same set as
    # Faraday (±1/2, ±3/2 for spin-3/2) regardless of which axis carries the field.
    H = voigt_hamiltonian(1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    eigs = np.sort(np.real(H.eigenenergies()))
    expected = np.array([-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(eigs, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# rf_hamiltonian
# ---------------------------------------------------------------------------


def test_rf_hamiltonian_is_hermitian():
    H = rf_hamiltonian(1.5, B_x=1.0, B_y=0.5, B_z=0.0)
    assert (H - H.dag()).norm() < 1e-10


def test_rf_hamiltonian_correct_dims():
    H = rf_hamiltonian(1.5, B_x=1.0, B_y=0.0, B_z=0.0)
    assert H.dims == [[4], [4]]


def test_rf_hamiltonian_zero_field_is_zero():
    H = rf_hamiltonian(1.5, B_x=0.0, B_y=0.0, B_z=0.0)
    assert H.norm() < 1e-10


def test_rf_hamiltonian_scales_with_field():
    H1 = rf_hamiltonian(1.5, B_x=1.0, B_y=0.0, B_z=0.0)
    H2 = rf_hamiltonian(1.5, B_x=2.0, B_y=0.0, B_z=0.0)
    assert (H2 - 2 * H1).norm() < 1e-10


# ---------------------------------------------------------------------------
# transition_rate
# ---------------------------------------------------------------------------


def _eigenstates_spin32():
    """Return eigenstates and energies for a simple spin-3/2 Hamiltonian."""
    H = faraday_hamiltonian(1.0, 0.1, 0.4, 1.5, 0.0, 0.0, 0.0)
    energies, states = H.eigenstates()
    return energies, states


def test_transition_rate_is_nonnegative():
    energies, states = _eigenstates_spin32()
    H_rf = rf_hamiltonian(1.5, B_x=1.0, B_y=0.0, B_z=0.0)
    rate = transition_rate(H_rf, states[0], states[1], energies[0], energies[1], 0.0)
    assert rate >= 0


def test_transition_rate_returns_scalar():
    energies, states = _eigenstates_spin32()
    H_rf = rf_hamiltonian(1.5, B_x=1.0, B_y=0.0, B_z=0.0)
    rate = transition_rate(H_rf, states[0], states[1], energies[0], energies[1], 0.0)
    assert np.ndim(rate) == 0


def test_transition_rate_on_resonance_exceeds_off_resonance():
    energies, states = _eigenstates_spin32()
    H_rf = rf_hamiltonian(1.5, B_x=1.0, B_y=0.0, B_z=0.0)
    gap = energies[1] - energies[0]
    rate_on = transition_rate(H_rf, states[0], states[1], energies[0], energies[1], gap)
    rate_off = transition_rate(
        H_rf, states[0], states[1], energies[0], energies[1], gap * 1000
    )
    assert rate_on > rate_off
