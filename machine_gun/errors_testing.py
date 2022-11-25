import numpy
import sys
from scipy.linalg import expm

def StateConstructor(n_photons):
	state = numpy.array(([1],[0]))
	photon = numpy.array(([1],[0]))
	for i in range(n_photons):
		state = numpy.kron(state, photon)
	return state


def SingleQubitOperationCreator(operator, total_number_of_qubits, qubit_to_be_operated):
	'''This function exists to create large operators involving a one single qubit operation and the identity on other qubits. It can 
	apply the operation on any of the input qubits as specified by the qubit_to_be_operated variable. The total_number_of_qubits includes
	the dot and the photons.

	It takes as input the 2x2 matrix representation (in the |0> |1> basis) of the single qubit operator to be performed, along with the 
	total_number_of_qubits and the number of the qubit to be operated on (with the dot labelled qubit 1, not 0).'''

	if qubit_to_be_operated <= 0: 																		# check for invalid qubit indices
		print("The function SingleQubitOperationCreator counts qubits starting at 1, not 0. You can't put zero as the value of qubit_to_be_operated, it won't give the write output.")
		sys.exit("I've stopped the programme running so that you can find and fix your mistake.")
		
	identity = numpy.eye(2, dtype = 'float')

	if qubit_to_be_operated == total_number_of_qubits:
		state = operator
		for i in range(total_number_of_qubits - 1):
			state = numpy.kron(identity, state)
		output_state = state

	elif qubit_to_be_operated < total_number_of_qubits:
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

	return output_state 																				# here the variable output_state is a matrix representing the operation described.

def ControlledUnitaryCreator(n_photons, target_photon, unitary_to_be_applied = numpy.array(([0,1], [1,0]))):
	'''This function creates a matrix representing a CU gate with the dot as the control and an arbitrary photon as the target(specified by target_photon, 
	which starts counting at the first photon = 1). By default the function implements a CNOT gate, as the majority of the time this is what I'll be doing.
	It does this by splitting the operation into 2 parts (* denotes tensor product of matrices):

		|0><0| * I * I * ... * I * ... * I(labelled first_term) 
		&
		|1><1| * I * I * ... * U * ... * I (labelled second_term, U is required unitary operation, which defaults to Pauli X).

	These terms are then added to create the final matrix. 

	We take the number of photons as input, not the number of qubits in total. Hopefully this will be a consistent convention I use.
	'''
	n_qubits = n_photons + 1

	dot_zero_zero_term = numpy.array(([1,0],[0,0])) 													# matrix representing the |0><0| operation on the dot
	dot_one_one_term = numpy.array(([0,0],[0,1])) 														# matrix representing the |1><1| operation on the dot
	first_term = SingleQubitOperationCreator(dot_zero_zero_term, n_qubits, 1) 							# we apply the dot operator on the dot, and leave the photons alone

	photon_X_matrix = SingleQubitOperationCreator(unitary_to_be_applied, n_photons, target_photon) 		# we create a matrix acting on the photons, with the unitary applied to the target_photon
	second_term = numpy.kron(dot_one_one_term, photon_X_matrix)											# we add the effect of the dot operation to the matrix created above

	CNOT_on_target_photon = first_term + second_term 													# create the matrix representing the total effect of the CNOT, by adding the separate parts

	return CNOT_on_target_photon 	

def ErrorWrapper(error_list, n_photons):
	'''error_list is structured so that each number in it is the index of a photon with an error. That error is listed in the second
	column of error_list.
	'''
	errors_array = numpy.zeros((n_photons, 2), dtype = 'int')

	for i, j in enumerate(error_list[:,0].astype(int)): 	# i counts the loop, j tells me the value (0 or 1, FALSE or TRUE)
		errors_array[j-1, 0] = 1							# 1 <-> TRUE, 2 <-> X, 3 <-> Y, 4 <-> Z

		if error_list[i,1] == 'X':
			errors_array[j-1, 1] = 2
		elif error_list[i,1] == 'Y':
			errors_array[j-1, 1] = 3
		elif error_list[i,1] == 'Z':
			errors_array[j-1, 1] = 4
		else:
			print("Something has gone wrong in ErrorWrapper(), most likely the error_list was badly defined.")
			sys.exit('The programme has stopped.')

	return errors_array


def MachineGunWithPauliErrorsOnDot(n_photons, errors_array):
	''' This needs testing and proper documentation. So far it hasn't been tested at all :(
	'''
	Pauli_X = numpy.array(([0,1],[1,0]))
	Pauli_Y = numpy.array(([0,-1j],[1j,0]))
	Pauli_Z = numpy.array(([1,0],[0,-1]))

	rotation_operator = expm(-1j*numpy.pi/4*Pauli_Y)

	Q_dot_rotator = SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1)
	total_operator = SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1)

	print('Q_dot_rotator')
	print(Q_dot_rotator.shape)
	print(Q_dot_rotator)

	for photon_index in range(n_photons):
		CNOT_operator = ControlledUnitaryCreator(n_photons, photon_index + 1)
		
		print('CNOT_operator')
		print(CNOT_operator.shape)
		print(CNOT_operator)

		# we apply errors to the quantum dot, during the cycle corresponding to photon i

		if errors_array[photon_index,0] == 1:
			if errors_array[photon_index,1] == 2:
				error_matrix = SingleQubitOperationCreator(Pauli_X, n_photons + 1, 1)
			elif errors_array[photon_index,1] == 3:
				error_matrix = SingleQubitOperationCreator(Pauli_Y, n_photons + 1, 1)
			elif errors_array[photon_index,1] == 4:
				error_matrix = SingleQubitOperationCreator(Pauli_Z, n_photons + 1, 1)
		elif errors_array[photon_index,0] == 0:
			error_matrix = numpy.eye(2**(n_photons + 1))

		print('error_matrix')
		print(error_matrix)
		print(error_matrix.shape)

		total_operator = numpy.dot(numpy.dot(numpy.dot(Q_dot_rotator, CNOT_operator), error_matrix), total_operator)

	return total_operator



n_test_photons = 2

error_list = numpy.array([[1, 'Z'], [2, 'Y']])
# error_list = numpy.array([[]])
# print error_list

errors_array = ErrorWrapper(error_list, n_test_photons)

print(errors_array)

test = MachineGunWithPauliErrorsOnDot(n_test_photons, errors_array)

initial_state = StateConstructor(n_test_photons)
print(initial_state)

print(numpy.dot(test, initial_state))
