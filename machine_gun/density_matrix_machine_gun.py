import numpy
from scipy.linalg import expm
import standard_constructors as sc
import qutip
import machine_gun_basics as mgb
import time
import matplotlib.pyplot as pyplot
import matplotlib



matplotlib.rcParams.update({'font.size': 26})


colour_list = ['black', 'red', 'blue', 'green', 'purple', 'cyan']


# Plan:
# 1) create rho and U_y -> convert to Qobj()
# 2) create CNOT matrices -> convert to Qobj()
# 3) loop through machine gun cycles, changing rho each time
# 4) add errors using kraus_to_super etc

# all functions here should be those that act on both sides of the density matrix rho 


def PerfectMachineGunDensityMatrices(n_photons):
	
	# create a Qobj representing the density matrix and the U_y operator
	density_matrix = sc.DMConstructor(n_photons)
	# print 'Density Matrix:'
	# print density_matrix

	density_matrix_vec = qutip.superoperator.operator_to_vector(density_matrix)

	Pauli_Y = numpy.array(([0, -1j], [1j, 0]))
	rotation_operator = expm(-1j*numpy.pi/4*Pauli_Y)

	U_y = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1))
	U_y_superop = qutip.superop_reps.to_super(U_y)

	# print 'Dot Rotation Operator'
	# print U_y

	density_matrix_vec = U_y_superop * density_matrix_vec

	CNOT_array = mgb.CNOTArrayCreator(n_photons) # returns an array of Qobj objects

	for photon_index in range(n_photons):
		CNOT_superop = qutip.superop_reps.to_super(CNOT_array[photon_index])
		density_matrix_vec = U_y_superop * CNOT_superop * density_matrix_vec

	density_matrix = qutip.superoperator.vector_to_operator(density_matrix_vec)

	return density_matrix

def DotDephasingKrausKreator(n_photons, dephasing_value):
	K_0 = numpy.array([[1,0], [0, numpy.sqrt(1-dephasing_value)]])
	K_1 = numpy.array([[0,0], [0, numpy.sqrt(dephasing_value)]])

	K_0 = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(K_0, n_photons + 1, 1))
	K_1 = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(K_1, n_photons + 1, 1))

	kraus_op_list = [K_0, K_1]

	dephasing_superop = qutip.superop_reps.kraus_to_super(kraus_op_list)

	return dephasing_superop

def DotDampingKrausKreator(n_photons, damping_value):
	K_0 = numpy.array([[1,0], [0, numpy.sqrt(1-damping_value)]])
	K_1 = numpy.array([[0, numpy.sqrt(damping_value)], [0,0]])

	K_0 = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(K_0, n_photons + 1, 1))
	K_1 = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(K_1, n_photons + 1, 1))

	kraus_op_list = [K_0, K_1]
	damping_superop = qutip.superop_reps.kraus_to_super(kraus_op_list)

	return damping_superop

def DephasingMachineGun(n_photons, dephasing_value):

	dephasing_superop = DotDephasingKrausKreator(n_photons, dephasing_value)

	density_matrix = sc.DMConstructor(n_photons)

	density_matrix_vec = qutip.superoperator.operator_to_vector(density_matrix)

	Pauli_Y = numpy.array(([0, -1j], [1j, 0]))
	rotation_operator = expm(-1j*numpy.pi/4*Pauli_Y)

	U_y = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1))
	U_y_superop = qutip.superop_reps.to_super(U_y)

	density_matrix_vec = U_y_superop * density_matrix_vec # initial rotation of the dot

	CNOT_array = mgb.CNOTArrayCreator(n_photons) # returns an array of Qobj objects

	for photon_index in range(n_photons):
		CNOT_superop = qutip.superop_reps.to_super(CNOT_array[photon_index])
		density_matrix_vec = U_y_superop * CNOT_superop * dephasing_superop * density_matrix_vec
		# density_matrix_vec = CNOT_superop * dephasing_superop * density_matrix_vec


	density_matrix_vec = dephasing_superop * density_matrix_vec
	density_matrix = qutip.superoperator.vector_to_operator(density_matrix_vec)

	# print 'N qubits = {}'.format(n_photons+1)
	# print density_matrix

	return density_matrix

def AmplitudeDampingMachineGun(n_photons, damping_value):

	damping_superop = DotDampingKrausKreator(n_photons, damping_value)

	density_matrix = sc.DMConstructor(n_photons)

	density_matrix_vec = qutip.superoperator.operator_to_vector(density_matrix)

	Pauli_Y = numpy.array(([0, -1j], [1j, 0]))
	rotation_operator = expm(-1j*numpy.pi/4*Pauli_Y)

	U_y = qutip.qobj.Qobj(mgb.SingleQubitOperationCreator(rotation_operator, n_photons + 1, 1))
	U_y_superop = qutip.superop_reps.to_super(U_y)

	density_matrix_vec = U_y_superop * density_matrix_vec # initial rotation of the dot

	CNOT_array = mgb.CNOTArrayCreator(n_photons) # returns an array of Qobj objects

	for photon_index in range(n_photons):
		CNOT_superop = qutip.superop_reps.to_super(CNOT_array[photon_index])
		density_matrix_vec = U_y_superop * CNOT_superop * damping_superop * density_matrix_vec
		# density_matrix_vec = CNOT_superop * damping_superop * density_matrix_vec

	density_matrix = qutip.superoperator.vector_to_operator(density_matrix_vec)

	return density_matrix

def SimulationTimeTest(max_photons, n_trials):

	photon_count_array = numpy.arange(max_photons)
	time_array = numpy.zeros((n_trials, max_photons))


	for j in range(n_trials):
		for i in range(max_photons):
			start = time.time()
			dm = PerfectMachineGunDensityMatrices(i)
			end = time.time()
			time_array[j,i] = end - start
			print('i = {}'.format(i))
		print('j = {}'.format(j))

	mean_array = numpy.mean(time_array, axis = 0)
	std_dev_array = numpy.std(time_array, axis = 0)

	pyplot.figure()

	pyplot.errorbar(photon_count_array, mean_array, yerr = std_dev_array, linestyle = 'None', marker = '.')
	pyplot.xlabel('# of Photons')
	pyplot.ylabel('Time (s)')
	pyplot.title('Simulation Time')

	pyplot.show()

def BasicDephasingFidelityTest(n_photons, n_dephasing_values = 10, graphing = False, min_dephase = 0, max_dephase = 1):
	perfect_result = PerfectMachineGunDensityMatrices(n_photons)
	fidelity_array = numpy.zeros(n_dephasing_values)

	dephasing_values_array = numpy.linspace(min_dephase, max_dephase, n_dephasing_values)


	for i in range(n_dephasing_values):
		imperfect_result = DephasingMachineGun(n_photons, dephasing_values_array[i])
		fidelity_array[i] = qutip.metrics.fidelity(perfect_result, imperfect_result)

	if graphing:
		pyplot.plot(dephasing_values_array, fidelity_array)
		pyplot.xlabel('Dephasing Value')
		pyplot.ylabel('Fidelity')
		pyplot.title('Fidelity vs Perfect State (for {} photon MG)'.format(n_photons))

		pyplot.show()
	else:
		return fidelity_array

def BasicDephasingTraceDistTest(n_photons, n_dephasing_values = 10, graphing = False, min_dephase = 0, max_dephase = 1):
	perfect_result = PerfectMachineGunDensityMatrices(n_photons)
	tracedist_array = numpy.zeros(n_dephasing_values)

	dephasing_values_array = numpy.linspace(min_dephase, max_dephase, n_dephasing_values)


	for i in range(n_dephasing_values):
		imperfect_result = DephasingMachineGun(n_photons, dephasing_values_array[i])
		tracedist_array[i] = qutip.metrics.tracedist(perfect_result, imperfect_result)

	if graphing:
		pyplot.plot(dephasing_values_array, tracedist_array)
		pyplot.xlabel('Dephasing Value')
		pyplot.ylabel('Trace Distance')
		pyplot.title('Trace Distance vs Perfect State (for {} photon MG)'.format(n_photons))

		pyplot.show()
	else:
		return tracedist_array

def BasicDampingFidelityTest(n_photons, n_damping_values = 10, graphing = False, min_damping = 0, max_damping = 1):
	perfect_result = PerfectMachineGunDensityMatrices(n_photons)
	fidelity_array = numpy.zeros(n_damping_values)

	damping_values_array = numpy.linspace(min_damping, max_damping, n_damping_values)


	for i in range(n_damping_values):
		imperfect_result = AmplitudeDampingMachineGun(n_photons, damping_values_array[i])
		fidelity_array[i] = qutip.metrics.fidelity(perfect_result, imperfect_result)

	if graphing:
		pyplot.plot(damping_values_array, fidelity_array)
		pyplot.xlabel('Damping Value')
		pyplot.ylabel('Fidelity')
		pyplot.title('Fidelity vs Perfect State (for {} photon MG)'.format(n_photons))

		pyplot.show()
	else:
		return fidelity_array

# BasicDephasingFidelityTest(3)

def ManyDephasingFidelityTests(max_photons, n_dephasing_values = 20, min_dephase = 0, max_dephase = 1, output = False):
	big_fidelity_array = numpy.zeros((max_photons, n_dephasing_values))
	dephasing_values_array = numpy.linspace(min_dephase, max_dephase, n_dephasing_values)
	

	for j in range(max_photons):
		big_fidelity_array[j] = BasicDephasingFidelityTest(j, n_dephasing_values = n_dephasing_values, min_dephase = min_dephase, max_dephase = max_dephase)
		if not output:
			pyplot.plot(dephasing_values_array, big_fidelity_array[j], label = '{} Qubit State'.format(j+1), color = colour_list[j])
			# pyplot.axhline(1/(numpy.sqrt(j+1)), xmin = min_dephase, color = colour_list[j], xmax = max_dephase, linestyle = 'dashed', label = '1/sqrt({})'.format(j+1))
			# pyplot.axhline(big_fidelity_array[j,-1], xmin = min_dephase, color = colour_list[j], xmax = max_dephase, linestyle = 'dashed', label = '{:.3f}'.format(big_fidelity_array[j,-1]))

	if output:
		return big_fidelity_array	
	else:
		pyplot.xlim(xmin = min_dephase, xmax = max_dephase)
		pyplot.xlabel('Dephasing Value')
		pyplot.ylabel('Fidelity')
		pyplot.legend(loc = 3)
		pyplot.title('Fidelity for Various Sized MG States')

		pyplot.show()


def ManyDephasingTraceDistTests(max_photons, n_dephasing_values = 10, min_dephase = 0, max_dephase = 1, output = False):
	big_tracedist_array = numpy.zeros((max_photons, n_dephasing_values))
	dephasing_values_array = numpy.linspace(min_dephase, max_dephase, n_dephasing_values)
	

	for j in range(max_photons):
		big_tracedist_array[j] = BasicDephasingFidelityTest(j, n_dephasing_values = n_dephasing_values, min_dephase = min_dephase, max_dephase = max_dephase)
		if not output:
			pyplot.plot(dephasing_values_array, big_tracedist_array[j], label = '{} Qubit State'.format(j+1), color = colour_list[j])
			# pyplot.axhline(1/(numpy.sqrt(j+1)), xmin = min_dephase, color = colour_list[j], xmax = max_dephase, linestyle = 'dashed', label = '1/sqrt({})'.format(j+1))
			# pyplot.axhline(big_tracedist_array[j,-1], xmin = min_dephase, color = colour_list[j], xmax = max_dephase, linestyle = 'dashed', label = '{:.3f}'.format(big_fidelity_array[j,-1]))

	if output:
		return big_tracedist_array

	else:
		pyplot.xlim(xmin = min_dephase, xmax = max_dephase)
		pyplot.xlabel('Dephasing Value')
		pyplot.ylabel('Trace Distance')
		pyplot.legend(loc = 3)
		pyplot.title('Trace Distance for Various Sized MG States')

		pyplot.show()	

def FidelityAndTraceDistanceDephasing(max_photons, n_dephasing_values = 10, min_dephase = 0, max_dephase = 1):
	big_tracedist_array = ManyDephasingTraceDistTests(max_photons, n_dephasing_values, min_dephase, max_dephase, output = True)
	big_fidelity_array = ManyDephasingFidelityTests(max_photons, n_dephasing_values, min_dephase, max_dephase, output = True)
	dephasing_values_array = numpy.linspace(min_dephase, max_dephase, n_dephasing_values)

	for j in range(max_photons):
		pyplot.plot(dephasing_values_array, big_tracedist_array[j], label = '{} Qubit State'.format(j+1), color = colour_list[j], linestyle = 'dotted')
		pyplot.plot(dephasing_values_array, big_fidelity_array[j], label = '{} Qubit State'.format(j+1), color = colour_list[j], linestyle = 'dashed')

	pyplot.xlim(xmin = min_dephase, xmax = max_dephase)
	pyplot.xlabel('Dephasing Value')
	pyplot.ylabel('Trace Distance (Dotted) & Fidelity (Dashed)')
	pyplot.legend(loc = 1)
	pyplot.title('Trace Distance & Fidelity for Various Sized MG States')

	pyplot.show()	

def ManyDampingFidelityTests(max_photons, n_damping_values = 20, min_damping = 0, max_damping = 1):
	big_fidelity_array = numpy.zeros((max_photons, n_damping_values))
	damping_values_array = numpy.linspace(min_damping, max_damping, n_damping_values)
	

	for j in range(max_photons):
		big_fidelity_array[j] = BasicDampingFidelityTest(j, n_damping_values = n_damping_values, min_damping = min_damping, max_damping = max_damping)
		pyplot.plot(damping_values_array, big_fidelity_array[j], label = '{} Photon State'.format(j))

	pyplot.xlabel('Damping Value')
	pyplot.ylabel('Fidelity')
	pyplot.legend(loc = 3)
	pyplot.title('Fidelity for Various Sized MG States')

	pyplot.show()	

def MaximumDephasingFidelityTest(max_photons):
	results_array = numpy.zeros((max_photons, 5))

	for i in range(max_photons):
		perfect_result = PerfectMachineGunDensityMatrices(i)
		imperfect_result = DephasingMachineGun(i, 1)
		results_array[i] = i+1, qutip.metrics.fidelity(perfect_result, imperfect_result), 1/numpy.sqrt(i+1), 1/(2**(i/2)), 1/numpy.sqrt(2**(i+1))
		print(i+1)

	return results_array

def Ruth2DDephasingFidelity(max_photons, n_dephasing_values, min_dephase = 0, max_dephase = 1):
	results_array = numpy.zeros((n_dephasing_values, max_photons))

	for p in range(max_photons):
		results_array[:, p] = BasicDephasingFidelityTest(p, n_dephasing_values, min_dephase = min_dephase, max_dephase = max_dephase) # p+1 because we countr from 0

		print('{} photons done'.format(p))


	extent = (0, max_photons, min_dephase, max_dephase)
	pyplot.imshow(results_array, origin = 'lower')

	# print pyplot.yticks()
	pyplot.xlabel('# of Photons')
	# pyplot.xlim(1, max_photons)
	pyplot.ylabel('Dephasing Amount')
	pyplot.colorbar()
	pyplot.show()


# Ruth2DDephasingFidelity(7, 10)


# print MaximumDephasingFidelityTest(6)

# ManyDephasingTraceDistTests(6)

# FidelityAndTraceDistanceDephasing(3)

ManyDephasingFidelityTests(6)
# ManyDampingFidelityTests(6)

# tom_test = DephasingMachineGun(1, 1)
# perfect_test = PerfectMachineGunDensityMatrices(1)

# print tom_test
# print 'fidelity = {}'.format(qutip.metrics.fidelity(tom_test, perfect_test))
