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


# ---------------------------------------------------------------------------
# Density-matrix formulation with error channels
# ---------------------------------------------------------------------------


def initial_density_matrix(n_photons: int) -> qutip.Qobj:
    """
    Density matrix |0...0⟩⟨0...0| for the dot + n_photons system.

    Args:
        n_photons (int): Number of photon qubits.

    Returns:
        Qobj: Density matrix of dimension 2^(n_photons + 1).
    """
    return qutip.Qobj(state_to_density_matrix(state_constructor(n_photons)))


def _embedded_kraus_superoperator(
    K_0: np.ndarray, K_1: np.ndarray, n_photons: int
) -> qutip.Qobj:
    """Embed a 2x2 Kraus pair on the dot qubit and build the superoperator."""
    full_K_0 = qutip.Qobj(single_qubit_operation(K_0, n_photons + 1, 1))
    full_K_1 = qutip.Qobj(single_qubit_operation(K_1, n_photons + 1, 1))
    return qutip.superop_reps.kraus_to_super([full_K_0, full_K_1])


def dot_dephasing_superoperator(n_photons: int, dephasing: float) -> qutip.Qobj:
    """
    Phase-damping channel on the dot qubit as a full-system superoperator.

    Kraus operators K_0 = diag(1, √(1−p)), K_1 = diag(0, √p) acting on the
    dot, embedded into the (n_photons + 1)-qubit space. p = 0 is the identity
    channel; p = 1 fully removes the dot's coherences.

    Args:
        n_photons (int): Number of photon qubits.
        dephasing (float): Dephasing strength p in [0, 1].

    Returns:
        Qobj: Superoperator on the full system.
    """
    K_0 = np.array([[1, 0], [0, np.sqrt(1 - dephasing)]])
    K_1 = np.array([[0, 0], [0, np.sqrt(dephasing)]])
    return _embedded_kraus_superoperator(K_0, K_1, n_photons)


def dot_damping_superoperator(n_photons: int, damping: float) -> qutip.Qobj:
    """
    Amplitude-damping channel on the dot qubit as a full-system superoperator.

    Kraus operators K_0 = diag(1, √(1−p)), K_1 = [[0, √p], [0, 0]] acting on
    the dot, embedded into the (n_photons + 1)-qubit space. p = 0 is the
    identity channel; p = 1 fully relaxes the dot to |0⟩.

    Args:
        n_photons (int): Number of photon qubits.
        damping (float): Damping strength p in [0, 1].

    Returns:
        Qobj: Superoperator on the full system.
    """
    K_0 = np.array([[1, 0], [0, np.sqrt(1 - damping)]])
    K_1 = np.array([[0, np.sqrt(damping)], [0, 0]])
    return _embedded_kraus_superoperator(K_0, K_1, n_photons)


def _run_density_matrix_protocol(
    n_photons: int, error_superop: qutip.Qobj | None = None
) -> qutip.Qobj:
    """
    Run the CSMG protocol on a vectorised density matrix.

    Applies the same sequence as perfect_cluster_state_machine_gun
    (Uy, then per photon: error channel, CNOT, Uy) plus a final error-channel
    application, so noise also acts during the last cycle's wait. The original
    research code applied the trailing channel only in the dephasing variant;
    here it is applied uniformly for both channels.
    """
    rho_vec = qutip.superoperator.operator_to_vector(initial_density_matrix(n_photons))

    Pauli_Y = np.array([[0, -1j], [1j, 0]])
    R_y = expm(-1j * np.pi / 4 * Pauli_Y)
    U_y = qutip.Qobj(single_qubit_operation(R_y, n_photons + 1, 1))
    U_y_super = qutip.superop_reps.to_super(U_y)

    rho_vec = U_y_super * rho_vec
    for i in range(n_photons):
        cnot_super = qutip.superop_reps.to_super(
            qutip.Qobj(controlled_unitary(n_photons, i + 1))
        )
        if error_superop is not None:
            rho_vec = error_superop * rho_vec
        rho_vec = U_y_super * cnot_super * rho_vec

    if error_superop is not None:
        rho_vec = error_superop * rho_vec

    return qutip.superoperator.vector_to_operator(rho_vec)


def perfect_machine_gun_density_matrix(n_photons: int) -> qutip.Qobj:
    """
    Final density matrix of the ideal (error-free) machine gun protocol.

    Equivalent to the state-vector path: equals
    |ψ⟩⟨ψ| with ψ = operational_perfect_machine_gun(n_photons).

    Args:
        n_photons (int): Number of photon qubits.

    Returns:
        Qobj: Final density matrix.
    """
    return _run_density_matrix_protocol(n_photons, None)


def noisy_machine_gun(
    n_photons: int, error_channel: str, error_strength: float
) -> qutip.Qobj:
    """
    Machine gun protocol with an error channel applied to the dot each cycle.

    Args:
        n_photons (int): Number of photon qubits.
        error_channel (str): "dephasing" (phase damping) or "damping"
            (amplitude damping).
        error_strength (float): Channel strength in [0, 1]; 0 reproduces the
            perfect protocol.

    Returns:
        Qobj: Final density matrix.
    """
    if error_channel == "dephasing":
        error_superop = dot_dephasing_superoperator(n_photons, error_strength)
    elif error_channel == "damping":
        error_superop = dot_damping_superoperator(n_photons, error_strength)
    else:
        raise ValueError(
            f"error_channel must be 'dephasing' or 'damping', got {error_channel!r}"
        )
    return _run_density_matrix_protocol(n_photons, error_superop)


def fidelity_vs_error(
    n_photons: int,
    error_strengths: np.ndarray,
    error_channel: str = "dephasing",
) -> np.ndarray:
    """
    Fidelity of the noisy machine gun output against the perfect output.

    Args:
        n_photons (int): Number of photon qubits.
        error_strengths (ndarray): Channel strengths to sweep, each in [0, 1].
        error_channel (str): "dephasing" or "damping".

    Returns:
        ndarray: Fidelity at each error strength, shape (len(error_strengths),).
    """
    perfect = perfect_machine_gun_density_matrix(n_photons)
    return np.array(
        [
            qutip.fidelity(perfect, noisy_machine_gun(n_photons, error_channel, s))
            for s in error_strengths
        ]
    )


def trace_distance_vs_error(
    n_photons: int,
    error_strengths: np.ndarray,
    error_channel: str = "dephasing",
) -> np.ndarray:
    """
    Trace distance of the noisy machine gun output from the perfect output.

    Args:
        n_photons (int): Number of photon qubits.
        error_strengths (ndarray): Channel strengths to sweep, each in [0, 1].
        error_channel (str): "dephasing" or "damping".

    Returns:
        ndarray: Trace distance at each error strength.
    """
    perfect = perfect_machine_gun_density_matrix(n_photons)
    return np.array(
        [
            qutip.tracedist(perfect, noisy_machine_gun(n_photons, error_channel, s))
            for s in error_strengths
        ]
    )


def error_list_to_array(
    error_list: list[tuple[int, str]], n_photons: int
) -> np.ndarray:
    """
    Convert a readable error list into the array machine_gun_with_pauli_errors takes.

    Args:
        error_list (list): Entries (photon_index, "X"|"Y"|"Z") with photon
            indices counted from 1, e.g. [(1, "Z"), (3, "X")].
        n_photons (int): Number of photon qubits.

    Returns:
        ndarray: Shape (n_photons, 2). Column 0: 1 if an error occurs on that
        photon's cycle, 0 otherwise. Column 1: error code 2=X, 3=Y, 4=Z.
    """
    codes = {"X": 2, "Y": 3, "Z": 4}
    errors_array = np.zeros((n_photons, 2), dtype=int)
    for photon_index, label in error_list:
        if not 1 <= photon_index <= n_photons:
            raise ValueError(
                f"photon_index must be in [1, {n_photons}], got {photon_index}"
            )
        if label not in codes:
            raise ValueError(f"error label must be 'X', 'Y' or 'Z', got {label!r}")
        errors_array[photon_index - 1] = (1, codes[label])
    return errors_array
