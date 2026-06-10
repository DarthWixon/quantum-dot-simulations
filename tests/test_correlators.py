import numpy as np
import pytest
import qutip
from qdot.hamiltonians import faraday_hamiltonian
from qdot.correlators import spin_correlator, site_correlator, run_correlator_series
from qdot.io import save_efg


def _simple_hamiltonian():
    return faraday_hamiltonian(1.0, 0.1, 0.4, 1.5, 0.0, 0.0, 0.0)


def test_spin_correlator_invalid_axis_raises():
    H = _simple_hamiltonian()
    with pytest.raises(ValueError, match="spin_axis"):
        spin_correlator(0.0, H, "q")


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
    # The Hamiltonian has zeeman_term=1.0 Hz and a quadrupolar term that does not
    # commute with I_z, so <I_z(t)I_z(0)> must oscillate. The dominant frequency
    # gap is ~3 Hz (eigenvalue spread), period ~0.33 s — use times up to 1 second.
    H = _simple_hamiltonian()
    times = np.linspace(0, 1.0, 20)
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
        biaxiality,
        spin,
        *euler,
    )
    expected = spin_correlator(t, H, "z")
    assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# run_correlator_series
# ---------------------------------------------------------------------------

_BOUNDS = [0, 2, 0, 2]
_RNG = np.random.default_rng(7)


@pytest.fixture
def efg_dir(tmp_path):
    """Write a tiny (2×2) synthetic EFG archive for Ga69 to tmp_path."""
    shape = (2, 2)
    eta = _RNG.uniform(0.0, 1.0, shape)
    V_XX = _RNG.uniform(-1e14, 1e14, shape)
    V_YY = _RNG.uniform(-1e14, 1e14, shape)
    V_ZZ = _RNG.uniform(1e13, 1e14, shape)
    euler_angles = _RNG.uniform(0, 2 * np.pi, (*shape, 3))
    save_efg(tmp_path, "Ga69", _BOUNDS, 1, eta, V_XX, V_YY, V_ZZ, euler_angles)
    return tmp_path


def test_run_correlator_series_output_shape(efg_dir):
    timerange = np.array([0.0, 1e-9])
    result = run_correlator_series(
        efg_dir, timerange, 1.0, "Ga69", _BOUNDS, step_size=1
    )
    assert result.shape == (3, 2)


def test_run_correlator_series_values_are_finite(efg_dir):
    timerange = np.array([0.0, 1e-9])
    result = run_correlator_series(
        efg_dir, timerange, 1.0, "Ga69", _BOUNDS, step_size=1
    )
    assert np.all(np.isfinite(result))


def test_run_correlator_series_invalid_species_raises(efg_dir):
    with pytest.raises(ValueError, match="nuclear_species"):
        run_correlator_series(efg_dir, np.array([0.0]), 1.0, "Xx99", _BOUNDS)


# ---------------------------------------------------------------------------
# load_correlator_archive
# ---------------------------------------------------------------------------


def test_load_correlator_archive_round_trip(tmp_path):
    from qdot.correlators import load_correlator_archive

    rng = np.random.default_rng(0)
    timerange = np.logspace(-6, -3, 8)
    bounds = [0, 2, 0, 2]
    species_data = {
        s: rng.random((3, len(timerange))) for s in ("Ga69", "Ga71", "As75", "In115")
    }
    path = tmp_path / "correlator_archive.npz"
    np.savez(
        path,
        timerange=timerange,
        region_bounds=bounds,
        **{f"{s}_data": d for s, d in species_data.items()},
    )

    result = load_correlator_archive(path)
    np.testing.assert_allclose(result["timerange"], timerange)
    np.testing.assert_array_equal(result["region_bounds"], bounds)
    for s, d in species_data.items():
        np.testing.assert_allclose(result["data"][s], d)
