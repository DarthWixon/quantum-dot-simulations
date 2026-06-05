import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import matplotlib.pyplot as plt
import scipy.constants as constants
import isotope_parameters as ISOP
import qutip
import multiprocessing


def CalculateHamiltonians(nuclear_species, location, applied_field, region_bounds):
    # set up background stuff
    h = constants.h
    e = constants.e
    # nuclear_species = "In115"
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]

    # region_bounds = [100, 1200, 200, 1000]
    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

    # set up atom specific stuff

    # these are the coordinates of the point we want, as found in the
    # region bounds array (therefore NOT the same format as that array)
    (
        x_coord,
        y_coord,
    ) = location  # (0,0) is the first element of the region_bounds array, regardless of it's size
    # applied_field = 0.5 # in Tesla

    eta = eta_array[x_coord, y_coord]
    V_ZZ = V_ZZ_array[x_coord, y_coord]
    alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

    quadrupole_coupling_constant = (3 * e * Q) / (
        2 * h * spin * (2 * spin - 1)
    )  # constant to convert Vzz to fq
    quadrupolar_term = quadrupole_coupling_constant * V_ZZ

    Zeeman_term = (
        zeeman_frequency_per_tesla * applied_field
    )  # will normally be 0, but can be changed

    farady_hamiltonian = (
        bqf.structural_hamiltonian_creator_Faraday(
            Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
        )
        / 1e6
    )
    voigt_hamiltonian = (
        bqf.structural_hamiltonian_creator_Voigt(
            Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
        )
        / 1e6
    )

    return farady_hamiltonian, voigt_hamiltonian


def PlotHintonDiagrams(
    nuclear_species,
    location,
    applied_field=0.5,
    region_bounds=[100, 1200, 200, 1000],
    saving=False,
):
    farady_hamiltonian, voigt_hamiltonian = CalculateHamiltonians(
        nuclear_species, location, applied_field, region_bounds
    )

    qutip.hinton(farady_hamiltonian)
    if saving:
        plotname = f"{graph_path}hinton_diagram_for_{nuclear_species}_in_faraday_geometry_at_coordinates{location}_in_region_{region_bounds}_in_a_{applied_field}T_field.png"
        plt.savefig(plotname)
    else:
        plt.show()
    plt.close()

    # qutip.hinton(voigt_hamiltonian)
    # if saving:
    # 	plotname = f"{graph_path}hinton_diagram_for_{nuclear_species}_in_voigt_geometry_at_coordinates{location}_in_region_{region_bounds}_in_a_{applied_field}T_field.png"
    # 	plt.savefig(plotname)
    # else:
    # 	plt.show()
    # plt.close()


def CalculateEigenstructure(
    nuclear_species,
    location,
    applied_field=0.5,
    region_bounds=[100, 1200, 200, 1000],
    printing=False,
):
    farady_hamiltonian, voigt_hamiltonian = CalculateHamiltonians(
        nuclear_species, location, applied_field, region_bounds
    )

    faraday_eigenstructure = farady_hamiltonian.eigenstates()
    voigt_eigenstructure = voigt_hamiltonian.eigenstates()

    if printing:
        faraday_eigenvalues = np.real_if_close(faraday_eigenstructure[0])
        voigt_eigenvalues = np.real_if_close(voigt_eigenstructure[0])

        # qutip returns normalised eigenvectors (have tested it)
        # thus we can easily measure overlap using dot products
        faraday_eigenvectors = faraday_eigenstructure[1]
        voigt_eigenvectors = voigt_eigenstructure[1]

        print("----------------------------")
        print("Faraday Geometry")
        print(f"Eigenvalues: {faraday_eigenvalues}")
        print(f"Eigenvectors: {faraday_eigenvectors}")
        print("Norm of Eigenvectors")
        for i in range(len(faraday_eigenvectors)):
            print(faraday_eigenvectors[i].norm())
            print(
                f"Overlap method gives: {faraday_eigenvectors[i].overlap(faraday_eigenvectors[i])}"
            )

        print("----------------------------")
        print("Voigt Geometry")
        print(f"Eigenvalues: {voigt_eigenvalues}")
        print(f"Eigenvectors: {voigt_eigenvectors}")
        print("Norm of Eigenvectors")
        for i in range(len(voigt_eigenvectors)):
            print(voigt_eigenvectors[i].norm())

    return faraday_eigenstructure, voigt_eigenstructure


def FindingEigenvalueDifferences(faraday_eigenvalues, voigt_eigenvalues):
    diff_array = np.abs(faraday_eigenvalues - voigt_eigenvalues)
    print(diff_array)

    return diff_array


def ParameterPrinter(
    nuclear_species, location, applied_field=0.5, region_bounds=[100, 1200, 200, 1000]
):
    h = constants.h
    e = constants.e
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]

    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

    # set up atom specific stuff

    # these are the coordinates of the point we want, as found in the
    # region bounds array (therefore NOT the same format as that array)
    (
        x_coord,
        y_coord,
    ) = location  # (0,0) is the first element of the region_bounds array, regardless of it's size

    eta = eta_array[x_coord, y_coord]
    V_ZZ = V_ZZ_array[x_coord, y_coord]
    alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

    quadrupole_coupling_constant = (3 * e * Q) / (
        2 * h * spin * (2 * spin - 1)
    )  # constant to convert Vzz to fq
    quadrupolar_term = quadrupole_coupling_constant * V_ZZ

    Zeeman_term = zeeman_frequency_per_tesla * applied_field

    print("----------------------------")
    print(f"Data for {nuclear_species} at {location} in a {applied_field}T Field")
    print(f"Zeeman Term: {np.around(Zeeman_term/1e6, 1)}MHz")
    print(f"Quadrupolar Term: {np.around(quadrupolar_term/1e6, 1)}MHz")
    print(f"Ratio of Zeeman/Quadrupolar: {np.around(Zeeman_term/quadrupolar_term, 1)}")
    print(f"Euler angles: {np.around([alpha, beta, gamma], 1)}")
    print(f"Biaxiality: {np.around(eta, 1)}")


def FindingEigenvectorDifferences(faraday_eigenvectors, voigt_eigenvectors):
    norm_array = np.zeros(len(faraday_eigenvectors), dtype="complex")

    for i in range(len(norm_array)):
        norm_array[i] = np.real_if_close(
            faraday_eigenvectors[i].overlap(voigt_eigenvectors[i])
        )

    average_dot_product = np.mean(norm_array)

    # print(average_dot_product)

    return average_dot_product


def EigenvalueDataforDara(
    nuclear_species, location, applied_field=1, region_bounds=[100, 1200, 200, 1000]
):
    farady_eigenstructure, voigt_eigenstructure = CalculateEigenstructure(
        nuclear_species, location, applied_field, printing=False
    )
    faraday_eigenvalues, voigt_eigenvalues = (
        farady_eigenstructure[0],
        voigt_eigenstructure[0],
    )

    f_eig_sum = np.sum(faraday_eigenvalues)
    v_eig_sum = np.sum(voigt_eigenvalues)
    diff = np.around(np.abs(f_eig_sum - v_eig_sum), 1)

    if diff >= 0:
        ParameterPrinter(nuclear_species, location, applied_field, region_bounds)
        print(
            f"Faraday Eigenvalues: {np.around(np.real_if_close(faraday_eigenvalues), 1)}"
        )
        print(f"Voigt Eigenvalues: {np.around(np.real_if_close(voigt_eigenvalues), 1)}")
        print(f"Difference of Eigenvalue Sums: {diff}")
        print("----------------------------")


def EigenvectorDifferencesWithChangingField(field_range, location, saving=False):
    results_array = np.zeros((4, len(field_range)))

    plt.figure()
    for n, nuclear_species in enumerate(["Ga69", "Ga71", "As75", "In115"]):
        for i, applied_field in enumerate(field_range):
            faraday_eigenstructure, voigt_eigenstructure = CalculateEigenstructure(
                nuclear_species, location, applied_field
            )
            faraday_eigenvectors, voigt_eigenvectors = (
                faraday_eigenstructure[1],
                voigt_eigenstructure[1],
            )
            results_array[n, i] = np.abs(
                FindingEigenvectorDifferences(faraday_eigenvectors, voigt_eigenvectors)
            )

        plt.plot(field_range, results_array[n, :], label=nuclear_species)

    plt.xlabel("Applied Magnetic Field (T)")
    plt.ylabel("Crude Similarity Measure (1 = The Same, 0 = Perpendicular)")
    plt.title("How Do Eigenvectors Differ wrt B Field?")
    plt.legend()

    if saving:
        plotname = f"{graph_path}eigenvector_similarity_wrt_magnetic_field_at_location_{location}_for_{len(field_range)}_fields_between_{field_range[0]}_and_{field_range[-1]}T.png"
        plt.savefig(plotname)
    else:
        plt.show()

    plt.close()


def EigenvectorDifferenceWithChangingLocation(
    applied_field, x_range, y_range, saving=False
):
    results_array = np.zeros((4, len(x_range), len(y_range)))

    fig, ax = plt.subplots(2, 2)

    for n, nuclear_species in enumerate(["Ga69", "Ga71", "As75", "In115"]):
        for x, x_loc in enumerate(x_range):
            for y, y_loc in enumerate(y_range):
                location = (x_loc, y_loc)
                faraday_eigenstructure, voigt_eigenstructure = CalculateEigenstructure(
                    nuclear_species, location, applied_field
                )
                faraday_eigenvectors, voigt_eigenvectors = (
                    faraday_eigenstructure[1],
                    voigt_eigenstructure[1],
                )
                results_array[n, x, y] = np.abs(
                    FindingEigenvectorDifferences(
                        faraday_eigenvectors, voigt_eigenvectors
                    )
                )

        axes = ax.flatten()[n]
        im = axes.imshow(results_array[n])
        axes.set_title(nuclear_species)
        axes.yaxis.set_visible(False)
        axes.xaxis.set_visible(False)

    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical")

    if saving:
        plotname = f"{graph_path}eigenvector_similarity_map_with_a_{applied_field}T_applied_field.png"
        plt.savefig(plotname)
    else:
        plt.show()

    plt.close()


# for nuclear_species in ["Ga69", "Ga71", "As75", "In115"]:
# 	PlotHintonDiagrams(nuclear_species, (500, 450), saving = False)

# farady_eigenstructure, voigt_eigenstructure = CalculateEigenstructure("Ga69", (450, 450), 0.01, printing = False)
# faraday_eigenvalues, voigt_eigenvalues = farady_eigenstructure[0], voigt_eigenstructure[0]
# FindingEigenvalueDifferences(faraday_eigenvalues, voigt_eigenvalues)
# faraday_eigenvectors, voigt_eigenvectors = farady_eigenstructure[1], voigt_eigenstructure[1]

# distance_measure = FindingEigenvectorDifferences(faraday_eigenvectors, voigt_eigenvectors)
# print(distance_measure)

# field_range = np.linspace(0,5,1000)
# EigenvectorDifferencesWithChangingField(field_range, (500, 450), saving = True)

# num_cores = multiprocessing.cpu_count()


# x_range = np.arange(0, 1100)
# y_range = np.arange(0, 800)
# EigenvectorDifferenceWithChangingLocation(0.5, x_range, y_range)

# ParameterPrinter("Ga69", (450, 500), 1)

# for x in range(0,800,10):
# 	for y in range(0,1100,10):
# 		location = (x,y)

# 		for i in np.linspace(-3, 3, 50):
# 			EigenvalueDataforDara("Ga69", location, i)
# 			EigenvalueDataforDara("Ga71", location, i)
# 			EigenvalueDataforDara("As75", location, i)
# 			EigenvalueDataforDara("In115", location, i)

# for i in range(300, 400, 10):
# 	if i%100 == 0:
# 		print(f"i = {i}")
# 	for j in range(450, 550, 10):
# 		EigenvalueDataforDara("Ga69", (i,j))

EigenvalueDataforDara("In115", (450, 500), 0.5)
PlotHintonDiagrams("In115", (450, 500), saving=False)
