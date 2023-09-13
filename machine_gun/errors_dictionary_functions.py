import numpy

global_errors_dict = {
    "Pauli_X": numpy.array([[0, 1], [1, 0]]),
    "Pauli_Y": numpy.array([0, -1j], [1j, 0]),
    "Pauli_Z": numpy.array([1, 0], [0, -1]),
}


def AddDephasing(dephasing_rate):
    dephase = numpy.array([])


for i in list(global_errors_dict.keys()):
    print(i)
    print(global_errors_dict[i])
