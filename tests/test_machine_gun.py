import numpy as np
import pytest
from qdot.machine_gun import (
    controlled_unitary,
    machine_gun_with_pauli_errors,
    operational_perfect_machine_gun,
    perfect_cluster_state_machine_gun,
    single_qubit_operation,
    state_constructor,
    state_to_density_matrix,
)

# ---------------------------------------------------------------------------
# state_constructor
# ---------------------------------------------------------------------------


def test_state_constructor_shape():
    for n in (1, 2, 3):
        state = state_constructor(n)
        assert state.shape == (2 ** (n + 1), 1)


def test_state_constructor_is_ground_state():
    state = state_constructor(3)
    assert state[0, 0] == 1.0
    assert np.all(state[1:] == 0)


def test_state_constructor_is_unit_norm():
    for n in (1, 2, 4):
        state = state_constructor(n)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# state_to_density_matrix
# ---------------------------------------------------------------------------


def test_state_to_density_matrix_shape():
    state = state_constructor(2)
    dm = state_to_density_matrix(state)
    assert dm.shape == (8, 8)


def test_state_to_density_matrix_trace_one():
    state = state_constructor(2)
    dm = state_to_density_matrix(state)
    assert abs(np.trace(dm) - 1.0) < 1e-12


def test_state_to_density_matrix_is_hermitian():
    state = state_constructor(2)
    dm = state_to_density_matrix(state)
    np.testing.assert_allclose(dm, dm.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# single_qubit_operation
# ---------------------------------------------------------------------------

_I2 = np.eye(2)
_X = np.array([[0, 1], [1, 0]])
_Z = np.array([[1, 0], [0, -1]])


def test_single_qubit_operation_shape():
    for n in (2, 3, 4):
        op = single_qubit_operation(_X, n, 1)
        assert op.shape == (2**n, 2**n)


def test_single_qubit_operation_identity_gives_identity():
    op = single_qubit_operation(_I2, 3, 2)
    np.testing.assert_allclose(op, np.eye(8), atol=1e-12)


def test_single_qubit_operation_invalid_target_raises():
    with pytest.raises(ValueError):
        single_qubit_operation(_X, 3, 0)
    with pytest.raises(ValueError):
        single_qubit_operation(_X, 3, 4)


def test_single_qubit_operation_target_qubit_1_acts_on_first():
    # Z on qubit 1 of a 2-qubit system: Z ⊗ I
    op = single_qubit_operation(_Z, 2, 1)
    expected = np.kron(_Z, _I2)
    np.testing.assert_allclose(op, expected, atol=1e-12)


def test_single_qubit_operation_target_qubit_last_acts_on_last():
    # Z on qubit 2 of a 2-qubit system: I ⊗ Z
    op = single_qubit_operation(_Z, 2, 2)
    expected = np.kron(_I2, _Z)
    np.testing.assert_allclose(op, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# controlled_unitary
# ---------------------------------------------------------------------------


def test_controlled_unitary_shape():
    for n in (1, 2, 3):
        gate = controlled_unitary(n, 1)
        assert gate.shape == (2 ** (n + 1), 2 ** (n + 1))


def test_controlled_unitary_is_unitary():
    for n in (1, 2, 3):
        gate = controlled_unitary(n, 1)
        product = gate @ gate.conj().T
        np.testing.assert_allclose(product, np.eye(2 ** (n + 1)), atol=1e-12)


def test_controlled_unitary_dot_zero_leaves_photon_unchanged():
    # When dot is |0⟩, the target photon should be unchanged.
    gate = controlled_unitary(1, 1)
    # |0⟩_dot ⊗ |1⟩_photon = [0, 1, 0, 0]^T
    state_in = np.array([[0], [1], [0], [0]], dtype=complex)
    state_out = gate @ state_in
    np.testing.assert_allclose(state_out, state_in, atol=1e-12)


def test_controlled_unitary_dot_one_flips_photon():
    # When dot is |1⟩ and photon is |0⟩, CNOT should flip photon to |1⟩.
    gate = controlled_unitary(1, 1)
    # |1⟩_dot ⊗ |0⟩_photon = [0, 0, 1, 0]^T
    state_in = np.array([[0], [0], [1], [0]], dtype=complex)
    expected = np.array([[0], [0], [0], [1]], dtype=complex)
    np.testing.assert_allclose(gate @ state_in, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# perfect_cluster_state_machine_gun
# ---------------------------------------------------------------------------


def test_perfect_machine_gun_operator_shape():
    for n in (1, 2, 3):
        op = perfect_cluster_state_machine_gun(n)
        assert op.shape == (2 ** (n + 1), 2 ** (n + 1))


def test_perfect_machine_gun_operator_is_unitary():
    for n in (1, 2, 3):
        op = perfect_cluster_state_machine_gun(n)
        product = op @ op.conj().T
        np.testing.assert_allclose(product, np.eye(2 ** (n + 1)), atol=1e-12)


# ---------------------------------------------------------------------------
# operational_perfect_machine_gun
# ---------------------------------------------------------------------------


def test_operational_perfect_machine_gun_shape():
    for n in (1, 2, 3):
        state = operational_perfect_machine_gun(n)
        assert state.shape == (2 ** (n + 1), 1)


def test_operational_perfect_machine_gun_is_unit_norm():
    for n in (1, 2, 3):
        state = operational_perfect_machine_gun(n)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-12


def test_operational_perfect_machine_gun_uniform_amplitudes():
    # All amplitudes should have equal magnitude: 1 / sqrt(2^(n+1))
    for n in (1, 2, 3):
        state = operational_perfect_machine_gun(n)
        magnitudes = np.abs(state.flatten())
        expected = 1.0 / np.sqrt(2 ** (n + 1))
        np.testing.assert_allclose(magnitudes, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# machine_gun_with_pauli_errors
# ---------------------------------------------------------------------------


def test_machine_gun_no_errors_matches_perfect():
    n = 3
    errors = np.zeros((n, 2), dtype=int)
    op_errors = machine_gun_with_pauli_errors(n, errors)
    op_perfect = perfect_cluster_state_machine_gun(n)
    np.testing.assert_allclose(op_errors, op_perfect, atol=1e-12)


def test_machine_gun_with_errors_differs_from_perfect():
    n = 2
    errors = np.array([[1, 2], [0, 0]])  # X error on first cycle
    op_errors = machine_gun_with_pauli_errors(n, errors)
    op_perfect = perfect_cluster_state_machine_gun(n)
    assert np.linalg.norm(op_errors - op_perfect) > 1e-6


def test_machine_gun_with_errors_output_shape():
    n = 2
    errors = np.zeros((n, 2), dtype=int)
    op = machine_gun_with_pauli_errors(n, errors)
    assert op.shape == (2 ** (n + 1), 2 ** (n + 1))


# ---------------------------------------------------------------------------
# Density-matrix formulation with error channels
# ---------------------------------------------------------------------------


def test_initial_density_matrix_is_ground_state_projector():
    from qdot.machine_gun import initial_density_matrix

    dm = initial_density_matrix(2)
    full = dm.full()
    assert full.shape == (8, 8)
    assert full[0, 0] == pytest.approx(1.0)
    assert np.sum(np.abs(full)) == pytest.approx(1.0)


def test_perfect_density_matrix_matches_state_vector_path():
    from qdot.machine_gun import (
        perfect_machine_gun_density_matrix,
        operational_perfect_machine_gun,
        state_to_density_matrix,
    )

    state = operational_perfect_machine_gun(2)
    expected = state_to_density_matrix(state.flatten())
    result = perfect_machine_gun_density_matrix(2).full()
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("channel", ["dephasing", "damping"])
def test_zero_error_strength_matches_perfect(channel):
    from qdot.machine_gun import noisy_machine_gun, perfect_machine_gun_density_matrix

    noisy = noisy_machine_gun(2, channel, 0.0).full()
    perfect = perfect_machine_gun_density_matrix(2).full()
    np.testing.assert_allclose(noisy, perfect, atol=1e-12)


@pytest.mark.parametrize("channel", ["dephasing", "damping"])
@pytest.mark.parametrize("strength", [0.3, 0.7, 1.0])
def test_noisy_machine_gun_preserves_trace(channel, strength):
    from qdot.machine_gun import noisy_machine_gun

    dm = noisy_machine_gun(2, channel, strength)
    assert dm.tr() == pytest.approx(1.0, abs=1e-10)


def test_invalid_channel_raises():
    from qdot.machine_gun import noisy_machine_gun

    with pytest.raises(ValueError, match="error_channel"):
        noisy_machine_gun(2, "depolarising", 0.5)


def test_fidelity_starts_at_one_and_decreases():
    from qdot.machine_gun import fidelity_vs_error

    strengths = np.array([0.0, 0.5, 1.0])
    fidelities = fidelity_vs_error(2, strengths, "dephasing")
    assert fidelities[0] == pytest.approx(1.0, abs=1e-6)
    assert fidelities[-1] < fidelities[0]


def test_trace_distance_starts_at_zero_and_increases():
    from qdot.machine_gun import trace_distance_vs_error

    strengths = np.array([0.0, 1.0])
    distances = trace_distance_vs_error(2, strengths, "damping")
    assert distances[0] == pytest.approx(0.0, abs=1e-6)
    assert distances[-1] > distances[0]


def test_error_list_to_array():
    from qdot.machine_gun import error_list_to_array

    result = error_list_to_array([(1, "Z"), (3, "X")], n_photons=3)
    expected = np.array([[1, 4], [0, 0], [1, 2]])
    np.testing.assert_array_equal(result, expected)


def test_error_list_to_array_validates_label():
    from qdot.machine_gun import error_list_to_array

    with pytest.raises(ValueError, match="label"):
        error_list_to_array([(1, "W")], n_photons=2)


def test_error_list_to_array_validates_index():
    from qdot.machine_gun import error_list_to_array

    with pytest.raises(ValueError, match="photon_index"):
        error_list_to_array([(3, "X")], n_photons=2)
