import numpy as np
import pytest
import qutip
from qdot.nff import (
    dephasing_kraus_operator,
    dephasing_polarisation_curve,
    non_dephased_polarisation,
    pulse_kraus_operator,
    z_polarisation,
)

_IDENTITY_SUPEROP = qutip.superop_reps.to_super(qutip.qeye(2))


# ---------------------------------------------------------------------------
# dephasing_kraus_operator
# ---------------------------------------------------------------------------


def test_dephasing_kraus_operator_correct_dims():
    op = dephasing_kraus_operator(0.8)
    assert op.issuper
    assert op.dims == [[[2], [2]], [[2], [2]]]


def test_dephasing_kraus_operator_identity_at_no_dephasing():
    op = dephasing_kraus_operator(1.0)
    assert (op - _IDENTITY_SUPEROP).norm() < 1e-10


def test_dephasing_kraus_operator_dephasing_value_affects_output():
    op_none = dephasing_kraus_operator(1.0)
    op_max = dephasing_kraus_operator(0.5)
    assert (op_none - op_max).norm() > 1e-10


# ---------------------------------------------------------------------------
# pulse_kraus_operator
# ---------------------------------------------------------------------------


def test_pulse_kraus_operator_correct_dims():
    op = pulse_kraus_operator(0.8, 0.5)
    assert op.issuper
    assert op.dims == [[[2], [2]], [[2], [2]]]


def test_pulse_kraus_operator_identity_at_perfect_pulse():
    # q0=1, phase=0: E_0 = I, E_1 = E_2 = 0 → identity channel
    op = pulse_kraus_operator(1.0, 0.0)
    assert (op - _IDENTITY_SUPEROP).norm() < 1e-10


def test_pulse_kraus_operator_q0_affects_output():
    op_perfect = pulse_kraus_operator(1.0, 0.5)
    op_imperfect = pulse_kraus_operator(0.5, 0.5)
    assert (op_perfect - op_imperfect).norm() > 1e-10


def test_pulse_kraus_operator_phase_affects_output():
    op_zero = pulse_kraus_operator(0.8, 0.0)
    op_pi = pulse_kraus_operator(0.8, np.pi)
    assert (op_zero - op_pi).norm() > 1e-10


# ---------------------------------------------------------------------------
# z_polarisation
# ---------------------------------------------------------------------------


def test_z_polarisation_maximally_coherent_state():
    # |+⟩ state: ρ = [[1,1],[1,1]]/2 has full Z polarisation in X basis
    rho = qutip.Qobj(np.array([[1, 1], [1, 1]]) / 2)
    assert abs(z_polarisation(rho) - 1.0) < 1e-10


def test_z_polarisation_incoherent_state():
    # |0⟩ state: ρ = [[1,0],[0,0]] has no off-diagonal coherence
    rho = qutip.Qobj(np.array([[1, 0], [0, 0]], dtype=complex))
    assert abs(z_polarisation(rho)) < 1e-10


# ---------------------------------------------------------------------------
# dephasing_polarisation_curve
# ---------------------------------------------------------------------------


def test_dephasing_polarisation_curve_output_shapes():
    deph_list, z_list = dephasing_polarisation_curve(0.9, 0.3, n_gammas=10)
    assert deph_list.shape == (10,)
    assert z_list.shape == (10,)


def test_dephasing_polarisation_curve_dephasing_list_values():
    deph_list, _ = dephasing_polarisation_curve(0.9, 0.3, n_gammas=10)
    np.testing.assert_allclose(deph_list, np.linspace(1, 0.5, 10))


def test_dephasing_polarisation_curve_z_pol_is_real():
    _, z_list = dephasing_polarisation_curve(0.9, 0.3, n_gammas=10)
    assert z_list.dtype.kind == "f"


def test_dephasing_polarisation_curve_z_pol_is_finite():
    _, z_list = dephasing_polarisation_curve(0.9, 0.3, n_gammas=10)
    assert np.all(np.isfinite(z_list))


def test_dephasing_polarisation_curve_first_point_matches_non_dephased():
    q0, phase = 0.8, 1.2
    _, z_list = dephasing_polarisation_curve(q0, phase, n_gammas=5)
    expected = non_dephased_polarisation(q0, phase)
    assert abs(z_list[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# non_dephased_polarisation
# ---------------------------------------------------------------------------


def test_non_dephased_polarisation_returns_scalar():
    result = non_dephased_polarisation(0.9, 0.5)
    assert np.ndim(result) == 0


def test_non_dephased_polarisation_identity_pulse_returns_one():
    # q0=1, phase=0 is the identity pulse; the |+⟩ initial state is unchanged
    result = non_dephased_polarisation(1.0, 0.0)
    assert abs(result - 1.0) < 1e-10


def test_non_dephased_polarisation_matches_curve_at_no_dephasing():
    q0, phase = 0.7, 0.4
    result = non_dephased_polarisation(q0, phase)
    _, z_list = dephasing_polarisation_curve(q0, phase, n_gammas=1)
    assert abs(result - z_list[0]) < 1e-10
