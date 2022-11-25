import numpy
import sys
from scipy.linalg import expm

# numpy.set_printoptions(threshold=numpy.nan) # force numpy to print an entire array

def SingleQubitOperationCreator(operator, total_number_of_qubits, qubit_to_be_operated):
	'''This function exists to create large operators involving a one single qubit operation and the identity on other qubits. It can 
	apply the operation on any of the input qubits as specified by the qubit_to_be_operated variable. The total_number_of_qubits includes
	the dot and the photons.

	It takes as input the 2x2 matrix representation (in the |0> |1> basis) of the single qubit operator to be performed, along with the 
	total_number_of_qubits and the number of the qubit to be operated on (with the dot labelled qubit 1, not 0).'''

	if qubit_to_be_operated <= 0: 																								# check for invalid qubit indices
		print("The function SingleQubitOperationCreator counts qubits starting at 1, not 0. You can't put zero as the value of qubit_to_be_operated, it won't give the write output.")
		sys.exit("I've stopped the programme running so that you can find and fix your mistake.")
		
	identity = numpy.eye(2, dtype = 'float')

	if qubit_to_be_operated == total_number_of_qubits:
		state = operator
		for i in range(total_number_of_qubits - 1):
			state = numpy.kron(identity, state)
		output_state = state

	if qubit_to_be_operated < total_number_of_qubits:
		state = numpy.eye(2, dtype = 'float')
		for i in range(total_number_of_qubits - qubit_to_be_operated - 1):
			state = numpy.kron(identity, state)
		state = numpy.kron(operator, state)
		for i in range(qubit_to_be_operated - 1):
			state = numpy.kron(identity, state)
		output_state = state

	# the lines below check for other errors, they should never really happen, and the resulting error messages aren't particularly helpful

	elif qubit_to_be_operated > total_number_of_qubits:
		print("total_number_of_qubits = {}, qubit_to_be_operated = {}".format(total_number_of_qubits, qubit_to_be_operated))
		print("qubit_to_be_operated cannot be bigger than total_number_of_qubits")
		sys.exit("I've stopped the programme running so that you can find and fix your mistake.")

	return output_state


def ControlledUnitaryCreator(n_photons, target_photon, unitary_to_be_applied = numpy.array(([0,1], [1,0]))):
	'''This function creates a matrix representing a CU gate with the dot as the control and an arbitrary photon as the target(specified by target_photon, 
	which starts counting at the first photon = 1). By default the function implements a CNOT gate, as the majority of the time this is what I'll be doing.
	It does this by splitting the operation into 2 parts (* denotes tensor product of matrices):

		|0><0| * I * I * ... * I * ... * I(labelled first_term) 
		&
		|1><1| * I * I * ... * U * ... * I (labelled second_term, U is required unitary operation, which defaults to Pauli X).

	These terms are then added to create the final matrix. 

	We take the state as input only to determine its size, we don't act on it in this function. I may change this to pass only the state size so it's
	less confusing in future.
	'''
	n_qubits = n_photons + 1

	# if n_qubits.is_integer():
	# 	n_qubits = int(n_qubits)
	# else:
	# 	sys.exit("The state isn't a power of 2 in size, something is broken.")


	# n_photons = n_qubits - 1 																			# because one of the qubits is the quantum dot
	dot_zero_zero_term = numpy.array(([1,0],[0,0])) 													# matrix representing the |0><0| operation on the dot
	dot_one_one_term = numpy.array(([0,0],[0,1])) 														# matrix representing the |1><1| operation on the dot
	first_term = SingleQubitOperationCreator(dot_zero_zero_term, n_qubits, 1) 							# we apply the dot operator on the dot, and leave the photons alone

	photon_X_matrix = SingleQubitOperationCreator(unitary_to_be_applied, n_photons, target_photon) 		# we create a matrix acting on the photons, with the unitary applied to the target_photon
	second_term = numpy.kron(dot_one_one_term, photon_X_matrix)											# we add the effect of the dot operation to the matrix created above

	CNOT_on_target_photon = first_term + second_term 													# create the matrix representing the total effect of the CNOT, by adding the separate parts

	return CNOT_on_target_photon 														# return the matrix representing the CNOT operation

photon = numpy.array([1,0])
state = numpy.array([1,0])
n_photons = 2

for i in range(n_photons): # add n_photons photons to the state
	state = numpy.kron(state, photon)

print(numpy.log2(state.shape[0]))

Pauli_Y = numpy.array(([0, 1j], [-1j, 0]))
Pauli_Z = numpy.array(([1,0], [0,-1]))

test = ControlledUnitaryCreator(n_photons, 1)
# test2 = CNOTCreater(state, 2)

print(test) 