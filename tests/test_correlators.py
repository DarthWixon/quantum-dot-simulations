import numpy as np
import pytest
import qutip
from qdot.hamiltonians import faraday_hamiltonian
from qdot.correlators import spin_correlator, site_correlator


def _simple_hamiltonian():
    return faraday_hamiltonian(1.0, 0.1, 0.4, 1.5, 0.0, 0.0, 0.0)


def test_spin_correlator_invalid_axis_returns_none():
    H = _simple_hamiltonian()
    assert spin_correlator(0.0, H, "q") is None


def test_spin_correlator_t0_equals_iz_squared_trace():
    # At t=0, <I_z(0)I_z(0)> = Tr(rho * I_z^2) for rho = I/d
    H = _simple_hamiltonian()
    result = spin_correlator(0.0, H, "z")

    spin = 1.5
    I_z = qutip.jmat(spin, "z")
    dim = int(2 * spin + 1)
    rho = qutip.qeye(dim).unit()
    expected = float(np.real((I_z * I_z * rho).tr()))
    assert abs(result - expected) < 1e-10


def test_spin_correlator_all_axes_return_float():
    H = _simple_hamiltonian()
    for axis in ("x", "y", "z"):
        result = spin_correlator(0.0, H, axis)
        assert isinstance(float(result), float)


def test_spin_correlator_decays_over_time():
    # The correlator should not be constant over time (for a non-trivial Hamiltonian)
    H = _simple_hamiltonian()
    times = np.linspace(0, 1e-6, 5)
    values = [spin_correlator(t, H, "z") for t in times]
    assert not np.allclose(values, values[0])


def test_site_correlator_matches_spin_correlator():
    # site_correlator with known parameters should match spin_correlator directly
    import scipy.constants as const
    from qdot.isotopes import species_dict

    species = species_dict["Ga69"]
    spin = species["particle_spin"]
    zeeman_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]
    h, ec = const.h, const.e
    qcc = (3 * ec * Q) / (2 * h * spin * (2 * spin - 1))

    applied_field = 0.5
    biaxiality = 0.3
    euler = [0.1, 0.2, 0.3]
    V_ZZ = 1e13
    t = 1e-8

    result = site_correlator(
        t, zeeman_per_tesla, qcc, spin, biaxiality, euler, V_ZZ, applied_field, "z"
    )

    H = faraday_hamiltonian(
        zeeman_per_tesla * applied_field,
        qcc * V_ZZ,
        biaxiality, spin, *euler,
    )
    expected = spin_correlator(t, H, "z")
    assert abs(result - expected) < 1e-10
