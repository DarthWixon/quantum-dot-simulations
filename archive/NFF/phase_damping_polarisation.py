import numpy
import matplotlib.pyplot as pyplot
import qutip


inital_DM = qutip.qobj.Qobj(
    numpy.array([[1, 1], [0, 1]])
)  # start in maximally mixed state


def Z_pol_finder(density_matrix):
    Z_in_X_basis = qutip.qobj.Qobj(numpy.array([[0, 1], [1, 0]]))
    return (density_matrix * Z_in_X_basis).tr()


def Y_pol_finder(density_matrix):
    return (qutip.operators.sigmay() * density_matrix).tr()


def X_pol_finder(density_matrix):
    return (qutip.operators.sigmax() * density_matrix).tr()


def DephasingKrausKreator(dephasing_value):
    K_0 = qutip.qobj.Qobj(dephasing_value * numpy.array([[1, 0], [0, 1]]))
    K_1 = qutip.qobj.Qobj(
        numpy.sqrt(1 - dephasing_value) * (numpy.array([[1, 0], [0, -1]]))
    )

    kraus_op_list = [K_0, K_1]

    dephasing_superop = qutip.superop_reps.kraus_to_super(kraus_op_list)

    return dephasing_superop


def PulseKrausKreator(q0, phase):
    E_0 = qutip.qobj.Qobj(numpy.array([[1, 0], [0, q0 * numpy.exp(1j * phase)]]))
    E_1 = qutip.qobj.Qobj(numpy.array([[0, numpy.sqrt((1 - q0**2) / 2)], [0, 0]]))
    E_2 = qutip.qobj.Qobj(numpy.array([[0, 0], [0, numpy.sqrt((1 - q0**2) / 2)]]))

    kraus_op_list = [E_0, E_1, E_2]

    pulse_superop = qutip.superop_reps.kraus_to_super(kraus_op_list)

    return pulse_superop


def DephasingPolarisationCurve(q0, phase, n_gammas):
    initial_vec = qutip.superoperator.operator_to_vector(inital_DM)

    dephasing_list = numpy.linspace(1, 0.5, n_gammas)
    results_list = numpy.zeros(n_gammas, dtype="complex")

    pulse_operator = PulseKrausKreator(q0, phase)  # Pi pulse = 0,0

    for d in range(len(dephasing_list)):
        dephasing_operator = DephasingKrausKreator(dephasing_list[d])
        final_dm = qutip.superoperator.vector_to_operator(
            dephasing_operator * pulse_operator * initial_vec
        )
        # print final_dm
        results_list[d] = Z_pol_finder(final_dm)
        # print results_list[d]
    # print results_list
    results_list = numpy.real_if_close(results_list)
    # print results_list

    return dephasing_list, results_list


def NonDephasedPolarisation(q0, phase):
    initial_vec = qutip.superoperator.operator_to_vector(inital_DM)

    pulse_operator = PulseKrausKreator(q0, phase)

    final_dm = qutip.superoperator.vector_to_operator(pulse_operator * initial_vec)

    return Z_pol_finder(final_dm)


q0 = 0.99
phase = 0
n_gammas = 100

dephasing_list, z_pol_list = DephasingPolarisationCurve(q0, phase, n_gammas)
non_dephased_pol = NonDephasedPolarisation(q0, phase)

pyplot.figure()
pyplot.plot(dephasing_list, z_pol_list)
pyplot.xlabel("Gamma")
pyplot.ylabel("Z Polarisation")
pyplot.axhline(numpy.real_if_close(non_dephased_pol), linestyle="dotted")
pyplot.title("q0 = {}, phase = {}".format(q0, phase))
pyplot.gca().invert_xaxis()
pyplot.show()
