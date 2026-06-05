import numpy
import sys
from scipy.linalg import expm
import time
import matplotlib.pyplot as pyplot
import backbone_functions
from qutip import *


def SingleQubitOperationCreator(operator, total_number_of_qubits, qubit_to_be_operated):
    """This function exists to create large operators involving a one single qubit operation and the identity on other qubits. It can
    apply the operation on any of the input qubits as specified by the qubit_to_be_operated variable. The total_number_of_qubits includes
    the dot and the photons.

    It takes as input the 2x2 matrix representation (in the |0> |1> basis) of the single qubit operator to be performed, along with the
    total_number_of_qubits and the number of the qubit to be operated on (with the dot labelled qubit 1, not 0).
    """

    if qubit_to_be_operated <= 0:  # check for invalid qubit indices
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

    elif qubit_to_be_operated < total_number_of_qubits:
        state = numpy.eye(2, dtype="float")
        for i in range(total_number_of_qubits - qubit_to_be_operated - 1):
            state = numpy.kron(identity, state)
        state = numpy.kron(operator, state)
        for i in range(qubit_to_be_operated - 1):
            state = numpy.kron(identity, state)
        output_state = state

    # the lines below check for other errors, they should never really happen, and the resulting error messages aren't particularly helpful

    elif qubit_to_be_operated > total_number_of_qubits:
        print(
            "total_number_of_qubits = {}, qubit_to_be_operated = {}".format(
                total_number_of_qubits, qubit_to_be_operated
            )
        )
        print("qubit_to_be_operated cannot be bigger than total_number_of_qubits")
        sys.exit(
            "I've stopped the programme running so that you can find and fix your mistake."
        )

    return output_state  # here the variable output_state is a matrix representing the operation described.


def ControlledUnitaryCreator(
    n_photons, target_photon, unitary_to_be_applied=numpy.array(([0, 1], [1, 0]))
):
    """This function creates a matrix representing a CU gate with the dot as the control and an arbitrary photon as the target(specified by target_photon,
    which starts counting at the first photon = 1). By default the function implements a CNOT gate, as the majority of the time this is what I'll be doing.
    It does this by splitting the operation into 2 parts (* denotes tensor product of matrices):

            |0><0| * I * I * ... * I * ... * I(labelled first_term)
            &
            |1><1| * I * I * ... * U * ... * I (labelled second_term, U is required unitary operation, which defaults to Pauli X).

    These terms are then added to create the final matrix.

    We take the number of photons as input, not the number of qubits in total. Hopefully this will be a consistent convention I use.
    """
    n_qubits = n_photons + 1

    dot_zero_zero_term = numpy.array(
        ([1, 0], [0, 0])
    )  # matrix representing the |0><0| operation on the dot
    dot_one_one_term = numpy.array(
        ([0, 0], [0, 1])
    )  # matrix representing the |1><1| operation on the dot
    first_term = SingleQubitOperationCreator(
        dot_zero_zero_term, n_qubits, 1
    )  # we apply the dot operator on the dot, and leave the photons alone

    photon_X_matrix = SingleQubitOperationCreator(
        unitary_to_be_applied, n_photons, target_photon
    )  # we create a matrix acting on the photons, with the unitary applied to the target_photon
    second_term = numpy.kron(
        dot_one_one_term, photon_X_matrix
    )  # we add the effect of the dot operation to the matrix created above

    CNOT_on_target_photon = (
        first_term + second_term
    )  # create the matrix representing the total effect of the CNOT, by adding the separate parts

    return CNOT_on_target_photon


def CNOTArrayCreator(n_photons):
    """Returns an array of Qobj objects that reflect a CNOT operation on each of the
    qubits in turn.
    """
    CNOT_array = numpy.array([Qobj() for i in range(n_photons)])

    for i in range(n_photons):
        CNOT_array[i] = Qobj(ControlledUnitaryCreator(n_photons, i + 1))

    return CNOT_array


def PerfectClusterStateMachineGunCreator(n_photons):
    """This function creates the matrix which performs the perfect cluster state machine gun on a state of the dot and n_photons photons.
    The matrix created performs the following operation: U_y * C_i * U_y * C_i-1 * ... * U_y * C_1 * U_y, with C_i a CNOT between the dot
    and photon i, and U_y rotating the dot about the Y axis by pi/2.
    The output is an (n_photons + 1)**2 x (n_photons + 1)**2 array.
    """

    # create required Pauli operators, I may do this globally at some point, not sure yet
    Pauli_Y = numpy.array(([0, -1j], [1j, 0]))
    rotation_operator = expm(-1j * numpy.pi / 4 * Pauli_Y)

    Q_dot_rotator = SingleQubitOperationCreator(
        rotation_operator, n_photons + 1, 1
    )  # n_photons + 1 because we need to include the dot

    total_operator = Q_dot_rotator  # initialise the operator by rotating the dot once

    for photon_index in range(n_photons):
        CNOT_operator = ControlledUnitaryCreator(
            n_photons, photon_index + 1
        )  # remember this function defaults to CNOT

        total_operator = numpy.dot(
            numpy.dot(Q_dot_rotator, CNOT_operator), total_operator
        )  # add the operations to the total matrix

    return total_operator


def OperationalPerfectMachineGun(n_photons):
    """Create the state vector for perfect operation of the machine gun on n_photons quibts."""
    initial_state = backbone_functions.StateConstructor(n_photons)
    machine_gun_operator = PerfectClusterStateMachineGunCreator(n_photons)
    final_state = numpy.dot(machine_gun_operator, initial_state)
    return final_state


def MachineGunWithPauliErrorsOnDot(n_photons, errors_array):
    """This function outputs a matrix determing the operation of the CSMG when Pauli errors occur on the quantum dot.
    It can apply a single Pauli operation onto the quantum dot during each cycle of the machine gun.
    The errors_array passed is created by the ErrorWrapper function - this turns a human readable error list into a machine-usable one.
    """
    Pauli_X = numpy.array(([0, 1], [1, 0]))
    Pauli_Y = numpy.array(([0, -1j], [1j, 0]))
    Pauli_Z = numpy.array(([1, 0], [0, -1]))

    rotation_operator = expm(-1j * numpy.pi / 4 * Pauli_Y)

    Q_dot_rotator = SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1)
    total_operator = SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1)

    for photon_index in range(n_photons):
        CNOT_operator = ControlledUnitaryCreator(n_photons, photon_index + 1)

        # we apply errors to the quantum dot, during the cycle corresponding the creation of the photon labelled by photon_index

        if errors_array[photon_index, 0] == 1:
            if errors_array[photon_index, 1] == 2:
                error_matrix = SingleQubitOperationCreator(Pauli_X, n_photons + 1, 1)
            elif errors_array[photon_index, 1] == 3:
                error_matrix = SingleQubitOperationCreator(Pauli_Y, n_photons + 1, 1)
            elif errors_array[photon_index, 1] == 4:
                error_matrix = SingleQubitOperationCreator(Pauli_Z, n_photons + 1, 1)
        elif errors_array[photon_index, 0] == 0:
            error_matrix = numpy.eye(2 ** (n_photons + 1))

        total_operator = numpy.dot(
            numpy.dot(numpy.dot(Q_dot_rotator, CNOT_operator), error_matrix),
            total_operator,
        )

    return total_operator


def MachineGunWithNonUnitaryErrorsOnDot(n_photons, errors_array):
    """Outputs a matrix corresponding to non-unitary errors on the quantum dot. Currently I'm experimenting with using a dictionary
    to hold the arrays representing the errors, not sure how it's gonna work out so this is firmly a test case ATM.
    """

    Pauli_Y = numpy.array(
        ([0, -1j], [1j, 0])
    )  # this will likely be replaced by my global errors dictionary idea soon
    rotation_operator = expm(-1j * numpy.pi / 4 * Pauli_Y)
    Q_dot_rotator = SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1)
    total_operator = SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1)

    for photon_index in range(n_photons):
        CNOT_operator = ControlledUnitaryCreator(n_photons, photon_index + 1)

        if errors_array[photon_index, 0] == 1:
            local_error_matrix = global_errors_dict[errors_array[photon_index, 1]]
            error_matrix = SingleQubitOperationCreator(
                local_error_matrix, n_photons + 1, 1
            )
        elif errors_array[photon_index, 0] == 0:
            error_matrix = numpy.eye(2 ** (n_photons + 1))

        total_operator = numpy.dot(
            numpy.dot(numpy.dot(Q_dot_rotator, CNOT_operator), error_matrix),
            total_operator,
        )

    return total_operator


def PerfectStateStabiliserConfirmation(state, stabiliser_operator):
    """This will test the state given, using the stabiliser also given. According to the stuff Tom said in the meeting."""
