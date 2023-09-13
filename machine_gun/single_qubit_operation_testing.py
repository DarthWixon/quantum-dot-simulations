import numpy
import sys
from scipy.linalg import expm


def SingleQubitOperationCreator(operator, total_number_of_qubits, qubit_to_be_operated):
    """This function exists to create large operators involving a one single qubit operation and the identity on other qubits. It can
    apply the operation on any of the input qubits as specified by the qubit_to_be_operated variable. The total_number_of_qubits includes
    the dot and the photons.

    It takes as input the 2x2 matrix representation (in the |0> |1> basis) of the single qubit operator to be performed, along with the
    total_number_of_qubits and the number of the qubit to be operated on (with the dot labelled qubit 1, not 0).
    """
    if qubit_to_be_operated <= 0:
        print(
            "The function SingleQubitOperationCreator counts qubits starting at 1, not 0. You can't put zero as the value of qubit_to_be_operated, it won't give the write output."
        )
        sys.exit(
            "I've stopped the programme running so that you can find and fix your mistake."
        )

    identity = numpy.eye(2, dtype="float")

    if qubit_to_be_operated == total_number_of_qubits:
        state = operator
        for i in range(total_number_of_qubits - 1):
            state = numpy.kron(identity, state)
        output_state = state
        print("Loop 2 Used")

    if qubit_to_be_operated < total_number_of_qubits:
        state = numpy.eye(2, dtype="float")
        print(total_number_of_qubits - qubit_to_be_operated - 1)
        for i in range(total_number_of_qubits - qubit_to_be_operated - 1):
            state = numpy.kron(identity, state)
        print("state after first set of operations")
        print(state)
        print(state.shape)
        state = numpy.kron(operator, state)
        print("state after operator is added")
        print(state)
        print(state.shape)
        for i in range(qubit_to_be_operated - 1):
            state = numpy.kron(identity, state)
        output_state = state
        print(output_state.shape)
        print("Loop 3 Used")
    return output_state  # here the variable output_state is a matrix representing the operation described.


Pauli_Y = numpy.array(([0.0, -1j], [1j, 0]))
rotate_matrix = expm(-1j * numpy.pi / 4 * Pauli_Y)
print(rotate_matrix)

test_matrix = 8 * numpy.ones((2, 2))


test = SingleQubitOperationCreator(rotate_matrix, 2, 1)
# test2 = SingleQubitOperationCreator(test_matrix, 5, 4)

print(test)
