import numpy as np
import pytest
import qutip
from qdot.hamiltonians import faraday_hamiltonian, voigt_hamiltonian, spin_rotator


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
