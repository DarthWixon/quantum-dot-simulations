"""
Coherent Spin Manipulation by a photon Gun (CSMG / "machine gun") protocol.

Simulates cluster-state generation via repeated optical pulses on a quantum dot
coupled to a photon stream. Uses numpy matrix operations (not QuTiP), so operators
are plain ndarrays.

Note: This module covers the working core of the machine gun simulation.
Density-matrix and non-unitary-error extensions are not yet complete.
"""

import numpy as np
from scipy.linalg import expm
import qutip


def state_constructor(n_photons: int) -> np.ndarray:
    """
    Build the initial |0...0⟩ state vector for the dot + n_photons system.

    Args:
        n_photons (int): Number of photon qubits.

    Returns:
        ndarray: Column vector of length 2^(n_photons + 1).
    """
    state = np.array([[1], [0]])
    photon = np.array([[1], [0]])
    for _ in range(n_photons):
        state = np.kron(state, photon)
    return state


# Unused — scaffolding for the incomplete density-matrix extension.
def state_to_density_matrix(state: np.ndarray) -> np.ndarray:
    """Convert a state vector to a density matrix via outer product |ψ⟩⟨ψ|."""
    return np.outer(state, state.conj())


def single_qubit_operation(
    operator: np.ndarray, n_qubits: int, target_qubit: int
) -> np.ndarray:
    """
    Embed a single-qubit operator into the full n_qubit Hilbert space.

    Args:
        operator (ndarray): 2×2 operator matrix.
        n_qubits (int): Total number of qubits (dot + photons).
        target_qubit (int): Index of the target qubit, counting from 1 (dot = 1).

    Returns:
        ndarray: (2^n_qubits × 2^n_qubits) operator matrix.
    """
    if target_qubit <= 0 or target_qubit > n_qubits:
        raise ValueError(
            f"target_qubit must be in [1, {n_qubits}], got {target_qubit}."
        )

    identity = np.eye(2)

    if target_qubit == n_qubits:
        result = operator
        for _ in range(n_qubits - 1):
            result = np.kron(identity, result)
    else:
        result = np.eye(2)
        for _ in range(n_qubits - target_qubit - 1):
            result = np.kron(identity, result)
        result = np.kron(operator, result)
        for _ in range(target_qubit - 1):
            result = np.kron(identity, result)

    return result


def controlled_unitary(
    n_photons: int, target_photon: int, unitary: np.ndarray | None = None
) -> np.ndarray:
    """
    Construct a controlled-unitary gate with the dot as control.

    By default implements a CNOT (controlled-X) gate.

    Args:
        n_photons (int): Number of photon qubits.
        target_photon (int): Target photon index (1-indexed).
        unitary (ndarray): 2×2 unitary to apply. Defaults to Pauli X.

    Returns:
        ndarray: (2^(n_photons+1) × 2^(n_photons+1)) gate matrix.
    """
    if unitary is None:
        unitary = np.array([[0, 1], [1, 0]])

    n_qubits = n_photons + 1
    dot_zero = np.array([[1, 0], [0, 0]])
    dot_one = np.array([[0, 0], [0, 1]])

    first_term = single_qubit_operation(dot_zero, n_qubits, 1)
    photon_op = single_qubit_operation(unitary, n_photons, target_photon)
    second_term = np.kron(dot_one, photon_op)

    return first_term + second_term


# Unused — returns Qobj-wrapped CNOTs intended for the density-matrix extension;
# the state-vector path calls controlled_unitary directly.
def cnot_array(n_photons: int) -> np.ndarray:
    """
    Build an array of CNOT operators, one per photon.

    Args:
        n_photons (int): Number of photon qubits.

    Returns:
        ndarray of Qobj: CNOT operators, shape (n_photons,).
    """
    return np.array(
        [qutip.Qobj(controlled_unitary(n_photons, i + 1)) for i in range(n_photons)]
    )


def perfect_cluster_state_machine_gun(n_photons: int) -> np.ndarray:
    """
    Operator for the ideal cluster-state machine gun on n_photons qubits.

    Applies the sequence: Uy · C_n · Uy · C_{n-1} · ... · Uy · C_1 · Uy,
    where Uy is a π/2 rotation of the dot about Y and C_i is a CNOT on photon i.

    Args:
        n_photons (int): Number of photon qubits.

    Returns:
        ndarray: (2^(n_photons+1) × 2^(n_photons+1)) unitary matrix.
    """
    Pauli_Y = np.array([[0, -1j], [1j, 0]])
    R_y = expm(-1j * np.pi / 4 * Pauli_Y)
    dot_rotator = single_qubit_operation(R_y, n_photons + 1, 1)

    total = dot_rotator
    for i in range(n_photons):
        cnot = controlled_unitary(n_photons, i + 1)
        total = dot_rotator @ cnot @ total

    return total


def operational_perfect_machine_gun(n_photons: int) -> np.ndarray:
    """
    Apply the perfect machine gun to the |0...0⟩ initial state.

    Args:
        n_photons (int): Number of photon qubits.

    Returns:
        ndarray: Final state vector.
    """
    initial = state_constructor(n_photons)
    operator = perfect_cluster_state_machine_gun(n_photons)
    return operator @ initial


# Unused — error-simulation variant; not called by the default state-vector path.
def machine_gun_with_pauli_errors(
    n_photons: int, errors_array: np.ndarray
) -> np.ndarray:
    """
    Machine gun operator with Pauli errors applied to the dot during each cycle.

    Args:
        n_photons (int): Number of photon qubits.
        errors_array (ndarray): Shape (n_photons, 2). Column 0: 1 if error occurs,
            0 otherwise. Column 1: error type — 2=X, 3=Y, 4=Z.

    Returns:
        ndarray: (2^(n_photons+1) × 2^(n_photons+1)) operator matrix.
    """
    Pauli_X = np.array([[0, 1], [1, 0]])
    Pauli_Y = np.array([[0, -1j], [1j, 0]])
    Pauli_Z = np.array([[1, 0], [0, -1]])
    error_ops = {2: Pauli_X, 3: Pauli_Y, 4: Pauli_Z}

    R_y = expm(-1j * np.pi / 4 * Pauli_Y)
    dot_rotator = single_qubit_operation(R_y, n_photons + 1, 1)
    total = dot_rotator

    for i in range(n_photons):
        cnot = controlled_unitary(n_photons, i + 1)
        if errors_array[i, 0] == 1:
            error_mat = single_qubit_operation(
                error_ops[int(errors_array[i, 1])], n_photons + 1, 1
            )
        else:
            error_mat = np.eye(2 ** (n_photons + 1))
        total = dot_rotator @ cnot @ error_mat @ total

    return total
