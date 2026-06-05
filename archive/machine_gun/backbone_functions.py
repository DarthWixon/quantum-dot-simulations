import numpy
import matplotlib.pyplot as pyplot


def StateConstructor(n_photons):
    state = numpy.array(([1], [0]))
    photon = numpy.array(([1], [0]))
    for i in range(n_photons):
        state = numpy.kron(state, photon)
    return state


def ErrorWrapper(error_list, n_photons):
    """This function takes a list of the form [[photon_index, 'Error'], [photon_index, 'Error'], [photon_index, 'Error']...] and converts it
    to a form more easily used by the programme.
    """
    errors_array = numpy.zeros((n_photons, 2), dtype="int")

    for i, j in enumerate(
        error_list[:, 0].astype(int)
    ):  # i counts the loop, j tells me the value (0 or 1, FALSE or TRUE)
        errors_array[j - 1, 0] = 1  # 1 <-> TRUE, 2 <-> X, 3 <-> Y, 4 <-> Z

        if error_list[i, 1] == "X":
            errors_array[j - 1, 1] = 2
        elif error_list[i, 1] == "Y":
            errors_array[j - 1, 1] = 3
        elif error_list[i, 1] == "Z":
            errors_array[j - 1, 1] = 4
        else:
            print(
                "Something has gone wrong in ErrorWrapper(), most likely the error_list was badly defined."
            )
            sys.exit("The programme has stopped.")

    return errors_array


def TimeTakenTest(max_test_photons):
    time_array = numpy.zeros(max_test_photons)
    for i in range(max_test_photons):
        print(i)
        start = time.time()
        test_state = OperationalPerfectMachineGun(i)
        end = time.time()
        time_array[i] = end - start

    pyplot.plot(time_array)
    pyplot.xlabel("Number of Photons")
    pyplot.ylabel("Time Taken (s)")
    pyplot.show()


def StateToDensityMatrixConverter(state):
    density_matrix = numpy.outer(state, state)
    return density_matrix
