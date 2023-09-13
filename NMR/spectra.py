import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import multiprocessing
import time
import os.path
import backbone_quadrupolar_functions as bqf
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import isotope_parameters as ISOP
import random_point_searcher as rps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from qutip import *
import scipy.constants
import scipy.signal
import scipy.integrate
from itertools import permutations


hbar = scipy.constants.hbar
e = scipy.constants.e
h = scipy.constants.h

# plotting parameters
# print(mpl.rcParams["font.size"]) - default is 10
mpl.rcParams["font.size"] = 16


def RF_hamiltonian_creator(particle_spin, B_x, B_y, B_z, unknown_gamma=1):
    # I'm not entirely sure what unknown_gamma is at the moment, physically I mean (29/11/20)
    # the units of this Hamiltonian are probably not consistent either tbh, and may not matter at all
    I_x = qutip.jmat(particle_spin, "x")
    I_y = qutip.jmat(particle_spin, "y")
    I_z = qutip.jmat(particle_spin, "z")

    hbar = 1  # scipy.constants.hbar

    # this doesn't include the cos(wt) part which actually makes it oscillate IRL
    # just has the structure as that's all we need for finding transition rates
    rf_Ham = -hbar * unknown_gamma * (B_x * I_x + B_y * I_y + B_z * I_z)

    return rf_Ham


def transition_rate_calc(
    mixing_hamiltonian, init_state, final_state, E_init, E_final, omega_rf, delta=10e3
):
    # mixing_hamiltonian is normally made by the RF_hamiltonian function, but could in principle be anything
    transition_prob = np.abs(mixing_hamiltonian.matrix_element(final_state, init_state))

    frac_term = (2 * delta) / ((E_final - E_init - omega_rf) ** 2 + delta**2)

    return np.real_if_close(transition_prob * frac_term)


def calculate_absorbtion_single_slice_data(
    nuclear_species,
    applied_field,
    field_geometry,
    rf_freq_list,
    location,
    region_bounds=[100, 1200, 439, 880],
    data_only_return=False,
    rf_field=5e-3,
    use_sundfors_GET_vals=False,
):
    x_coord, y_coord = location

    transition_rate_data = np.zeros(len(rf_freq_list))

    # print(f"function is calculate_absorbtion_single_slice_data")
    # print(f"nuclear_species = {nuclear_species}")
    # print(f"field_geometry = {field_geometry}")

    # load the data for this species and region
    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = bqf.load_calculated_EFG_arrays(
        nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
    )
    # print(eta_array.shape)

    # find the species specific data
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]

    # find the location specific data
    eta = eta_array[x_coord, y_coord]
    V_ZZ = V_ZZ_array[x_coord, y_coord]
    alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

    # calculate the relevant constants
    quadrupole_coupling_constant = (3 * e * Q) / (
        2 * h * spin * (2 * spin - 1)
    )  # constant to convert Vzz to frequency units
    quadrupolar_term = quadrupole_coupling_constant * V_ZZ
    Zeeman_term = zeeman_frequency_per_tesla * applied_field

    # calculate the Hamiltonian and find the eigenstructure (remember, in frequency units)
    # Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
    # Hamiltonian = Hamiltonian.tidyup()
    # eigenenergies, eigenvectors = np.real_if_close(Hamiltonian.eigenstates())

    # index_list = np.arange(len(eigenenergies)) # we use this list to find the pairs of indices for matching

    if field_geometry == "Voigt":
        # static field along the x axis
        # want RF field to be perpendicular to static field, so we set it along z
        Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(
            Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
        )
        rf_Ham = RF_hamiltonian_creator(spin, 0, 0, rf_field)

    if field_geometry == "Faraday":
        # static field along the z axis
        # want RF field to be perpendicular to static field, so we set it along x
        Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(
            Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
        )
        rf_Ham = RF_hamiltonian_creator(spin, rf_field, 0, 0)

    Hamiltonian = Hamiltonian.tidyup()
    eigenenergies, eigenvectors = np.real_if_close(Hamiltonian.eigenstates())
    index_list = np.arange(
        len(eigenenergies)
    )  # we use this list to find the pairs of indices for matching

    for r, rf_freq in enumerate(rf_freq_list):
        perms = permutations(
            index_list, 2
        )  # because of how iterators work in python, we must recreate this each time
        for pair in perms:
            init_state = eigenvectors[pair[0]]
            final_state = eigenvectors[pair[1]]

            init_energy = eigenenergies[pair[0]]
            final_energy = eigenenergies[pair[1]]

            transition_rate_data[r] += transition_rate_calc(
                rf_Ham, init_state, final_state, init_energy, final_energy, rf_freq
            )

    if data_only_return:
        return transition_rate_data
    else:
        results_dict = {
            "nuclear_species": nuclear_species,
            "applied_field": applied_field,
            "field_geometry": field_geometry,
            "location": location,
            "region_bounds": region_bounds,
            "rf_freq_list": rf_freq_list,
            "data": transition_rate_data,
            "Sundfors?": use_sundfors_GET_vals,
        }

        return results_dict


def varied_B_field_spectra(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    location,
    region_bounds=[100, 1200, 439, 880],
):
    spectra_data_list = []

    for applied_field in applied_field_list:
        spectra_data_list.append(
            calculate_absorbtion_single_slice_data(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                location,
                region_bounds,
            )
        )

    plt.figure()

    for spectra_data_dict in spectra_data_list:
        applied_field = spectra_data_dict["applied_field"]
        x_points = (
            spectra_data_dict["rf_freq_list"] / 1e6
        )  # divide by 1e6 so that it plots in MHz
        data = spectra_data_dict["data"]
        plt.plot(x_points, data, label=f"{np.around(applied_field, 2)}T")

    base_info_dict = spectra_data_list[0]
    nuclear_species = base_info_dict["nuclear_species"]
    field_geometry = base_info_dict["field_geometry"]
    location = base_info_dict["location"]
    region_bounds = base_info_dict["region_bounds"]

    ax = plt.gca()
    plt.title(
        f"{nuclear_species}, Various Applied Fields, {field_geometry} Orientation"
    )
    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Absorbtion (arb. units)")
    plt.legend()

    plt.show()
    plt.close()


def varied_isotope_spectra(
    applied_field,
    field_geometry,
    rf_freq_list,
    location,
    region_bounds=[100, 1200, 439, 880],
):
    spectra_data_list = []

    for nuclear_species in bqf.nuclear_species_list:
        spectra_data_list.append(
            calculate_absorbtion_single_slice_data(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                location,
                region_bounds,
            )
        )

    plt.figure()

    for spectra_data_dict in spectra_data_list:
        nuclear_species = spectra_data_dict["nuclear_species"]
        x_points = (
            spectra_data_dict["rf_freq_list"] / 1e6
        )  # divide by 1e6 so that it plots in MHz
        data = spectra_data_dict["data"]
        plt.plot(x_points, data, label=f"{nuclear_species}")

    base_info_dict = spectra_data_list[0]
    applied_field = base_info_dict["applied_field"]
    field_geometry = base_info_dict["field_geometry"]
    location = base_info_dict["location"]
    region_bounds = base_info_dict["region_bounds"]

    ax = plt.gca()
    plt.title(f"All Isotopes, {applied_field}T Field, {field_geometry} Orientation")
    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Absorbtion (arb. units)")
    plt.legend()

    plt.show()
    plt.close()


def many_field_colourplot(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    location,
    region_bounds=[100, 1200, 439, 880],
    saving=False,
    graphing=True,
):
    data = np.zeros((len(applied_field_list), len(rf_freq_list)))

    for b, applied_field in enumerate(applied_field_list):
        data[b, :] = calculate_absorbtion_single_slice_data(
            nuclear_species,
            applied_field,
            field_geometry,
            rf_freq_list,
            location,
            region_bounds,
            data_only_return=True,
        )

    data = np.log(data)

    if graphing:
        img_extent = [
            rf_freq_list[0] / 1e6,
            rf_freq_list[-1] / 1e6,
            applied_field_list[0],
            applied_field_list[-1],
        ]

        plt.imshow(
            data,
            origin="lower",
            cmap=cm.GnBu,
            aspect="auto",
            extent=img_extent,
            interpolation="none",
        )
        plt.title(f"{nuclear_species}, {field_geometry} Orientation")

        plt.xlabel("RF Frequency (MHz)")
        plt.ylabel("Applied B Field (T)")

        if saving:
            plotname = f"{graph_path}2D_NMR_spectra_{nuclear_species}_location_{location}_in_{region_bounds}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
            plt.savefig(plotname)
            print(f"Saved file: {plotname}")
        else:
            plt.show()

        plt.close()


def many_location_colourplot(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_bounds=[100, 1200, 439, 880],
    saving=False,
):
    data = np.zeros((len(applied_field_list), len(rf_freq_list)))

    for location in loc_list:
        for b, applied_field in enumerate(applied_field_list):
            data[b, :] += calculate_absorbtion_single_slice_data(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                location,
                region_bounds,
                data_only_return=True,
            )

    data = np.log(data)

    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    plt.imshow(
        data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )
    plt.title(f"{nuclear_species}, {field_geometry} Orientation")

    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Applied B Field (T)")

    if saving:
        plotname = f"{graph_path}2D_NMR_spectra_{nuclear_species}_{len(loc_list)}_locs_in_{region_bounds}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname)
        print(f"Saved file: {plotname}")
    else:
        plt.show()

    plt.close()


def parameter_zipper(
    nuclear_species,
    applied_field,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_bounds,
    use_sundfors_GET_vals=False,
    rf_field=5e-3,
):
    n_items = len(loc_list)
    nuclear_species_list = [nuclear_species] * n_items
    applied_field_list_for_zip = [applied_field] * n_items
    field_geometry_list = [field_geometry] * n_items
    rf_freq_list_list = [rf_freq_list] * n_items
    # don't need to do anything to loc_list as it's the thing that changes
    region_bounds_list = [region_bounds] * n_items
    data_only_return_list = [True] * n_items
    rf_field_list = [rf_field] * n_items
    use_sundfors_GET_vals_list = [use_sundfors_GET_vals] * n_items

    zipped_params = list(
        zip(
            nuclear_species_list,
            applied_field_list_for_zip,
            field_geometry_list,
            rf_freq_list_list,
            loc_list,
            region_bounds_list,
            data_only_return_list,
            rf_field_list,
            use_sundfors_GET_vals_list,
        )
    )

    return zipped_params


def many_location_parallel_calculation(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_label,
    recalc=False,
    use_sundfors_GET_vals=False,
):
    # creates and saves data for NMR spectra, DOES NOT TAKE LOGS OF IT

    # print(f"function is many_location_parallel_calculation")
    # print(f"nuclear_species = {nuclear_species}")
    # print(f"field_geometry = {field_geometry}")

    # this is done weirdly cos I didn't want to change everything around when I did it (17/04/21)
    if use_sundfors_GET_vals:
        print("Using Sundfors values")
        filename = f"{data_path}NMR_absorbtion_data_using_old_GET_for_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_label}_for_{len(applied_field_list)}_B_fields_and_{len(rf_freq_list)}_rf_fields_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.npy"
    else:
        filename = f"{data_path}NMR_absorbtion_data_for_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_label}_for_{len(applied_field_list)}_B_fields_and_{len(rf_freq_list)}_rf_fields_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.npy"
        # print(filename)

    if os.path.isfile(filename) and recalc == False:
        print(f"{filename} already exists, not recalculating.")
        return

    region_bounds = bqf.QD_regions_dict[region_label]
    data = np.zeros((len(applied_field_list), len(rf_freq_list)))

    for b, applied_field in enumerate(applied_field_list):
        with multiprocessing.Pool() as pool:
            zipped_params = parameter_zipper(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                loc_list,
                region_bounds,
                use_sundfors_GET_vals,
            )
            slices_across_locations = pool.starmap(
                calculate_absorbtion_single_slice_data, zipped_params
            )
            pool.close()

        data[b, :] = np.sum(slices_across_locations, axis=0)

    np.save(filename, data)
    if os.path.isfile(filename):
        print(f"Successfully saved {filename}")


def load_many_location_data(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_label,
    use_sundfors_GET_vals=False,
):
    if use_sundfors_GET_vals:
        filename = f"{data_path}NMR_absorbtion_data_using_old_GET_for_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_label}_for_{len(applied_field_list)}_B_fields_and_{len(rf_freq_list)}_rf_fields_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.npy"
    else:
        filename = f"{data_path}NMR_absorbtion_data_for_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_label}_for_{len(applied_field_list)}_B_fields_and_{len(rf_freq_list)}_rf_fields_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.npy"
    spectra_data = np.load(filename)

    return spectra_data


def many_location_plot_using_loading(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_label="entire_dot",
    saving=False,
    recalc=False,
    use_sundfors_GET_vals=False,
):
    # print(f"function is many_location_plot_using_loading")
    # print(f"nuclear_species = {nuclear_species}")
    # print(f"field_geometry = {field_geometry}")

    many_location_parallel_calculation(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label,
        recalc,
        use_sundfors_GET_vals,
    )

    data = load_many_location_data(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label,
        use_sundfors_GET_vals,
    )

    data = np.log(data)

    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    plt.imshow(
        data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )
    plt.title(f"{nuclear_species}, {field_geometry} Orientation")

    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Applied B Field (T)")

    if saving:
        plotname = f"{graph_path}NMR_spectra_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname)
        print(f"Saved file: {plotname}")
    else:
        plt.show()

    plt.close()


def many_NMR_spectra_data_calculation(
    nuclear_species_list,
    applied_field_list,
    rf_freq_list,
    loc_list,
    region_label,
    recalc=False,
):
    # start = time.time()

    for nuclear_species in nuclear_species_list:
        for field_geometry in ["Voigt", "Faraday"]:
            many_location_parallel_calculation(
                nuclear_species,
                applied_field_list,
                field_geometry,
                rf_freq_list,
                loc_list,
                region_label,
                recalc,
            )
    region_end = time.time()

    # end = time.time()

    # print(f"Total time taken: {np.around((end-start)/3600, 2)} hours.")


def many_location_parallel_plot(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_label="entire_dot",
    region_bounds=[100, 1200, 439, 880],
    saving=False,
    graphing=True,
):
    region_bounds = bqf.QD_regions_dict[region_label]

    data = np.zeros((len(applied_field_list), len(rf_freq_list)))

    for b, applied_field in enumerate(applied_field_list):
        with multiprocessing.Pool() as pool:
            zipped_params = parameter_zipper(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                loc_list,
                region_bounds,
            )
            slices_across_locations = pool.starmap(
                calculate_absorbtion_single_slice_data, zipped_params
            )
            pool.close()

        # print(np.sum(slices_across_locations, axis = 0).shape)
        data[b, :] = np.sum(slices_across_locations, axis=0)

    if graphing:
        data = np.log(data)

        img_extent = [
            rf_freq_list[0] / 1e6,
            rf_freq_list[-1] / 1e6,
            applied_field_list[0],
            applied_field_list[-1],
        ]

        plt.imshow(
            data,
            origin="lower",
            cmap=cm.GnBu,
            aspect="auto",
            extent=img_extent,
            interpolation="none",
        )
        plt.title(f"{nuclear_species}, {field_geometry} Orientation")

        plt.xlabel("RF Frequency (MHz)")
        plt.ylabel("Applied B Field (T)")

        if saving:
            plotname = f"{graph_path}2D_NMR_spectra_FROM_PARALLEL_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_bounds}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
            plt.savefig(plotname)
            print(f"Saved file: {plotname}")
        else:
            plt.show()

        plt.close()


def random_locations_list_generator(n_locs, region_label):
    region_bounds = bqf.QD_regions_dict[region_label]
    rng = np.random.default_rng()

    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = bqf.load_calculated_EFG_arrays(
        "Ga69", region_bounds
    )  # use Ga69 cos we only care about the size of the arrays
    ax_0_range, ax_1_range = eta_array.shape
    ax_0_indices = rng.integers(ax_0_range, size=n_locs)
    ax_1_indices = rng.integers(ax_1_range, size=n_locs)

    loc_list = list(zip(ax_0_indices, ax_1_indices))
    return loc_list


def all_locations_list_generator(region_label):
    region_bounds = bqf.QD_regions_dict[region_label]

    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = bqf.load_calculated_EFG_arrays(
        "Ga69", region_bounds
    )  # use Ga69 cos we only care about the size of the arrays
    ax_0_range, ax_1_range = eta_array.shape

    loc_list = []
    for i in range(ax_0_range):
        for j in range(ax_1_range):
            loc_list.append((i, j))

    return loc_list


def Faraday_Voigt_spectra_comparison(
    nuclear_species,
    applied_field_list,
    rf_freq_list,
    loc_list,
    region_label="entire_dot",
    saving=False,
    recalc=False,
):
    many_location_parallel_calculation(
        nuclear_species,
        applied_field_list,
        "Voigt",
        rf_freq_list,
        loc_list,
        region_label,
        recalc,
    )
    many_location_parallel_calculation(
        nuclear_species,
        applied_field_list,
        "Faraday",
        rf_freq_list,
        loc_list,
        region_label,
        recalc,
    )

    voigt_data = load_many_location_data(
        nuclear_species,
        applied_field_list,
        "Voigt",
        rf_freq_list,
        loc_list,
        region_label,
    )
    faraday_data = load_many_location_data(
        nuclear_species,
        applied_field_list,
        "Faraday",
        rf_freq_list,
        loc_list,
        region_label,
    )

    data = faraday_data - voigt_data

    data_max = np.abs(np.max(data))
    data_min = np.abs(np.min(data))
    scale = np.max([data_max, data_min])

    # data = data + scale

    # transformed_data = np.log(np.abs(data))

    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    plt.imshow(
        data,
        origin="lower",
        cmap=cm.seismic,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
        vmin=-scale,
        vmax=scale,
    )
    plt.title(f"{nuclear_species}, Orientation Comparison (Faraday - Voigt)")
    # plt.colorbar(orientation = "vertical")

    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Applied B Field (T)")

    if saving:
        plotname = f"{graph_path}comparison_of_spectra_for_{nuclear_species}__{len(loc_list)}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname)
        print(f"Saved file: {plotname}")
    else:
        plt.show()

    plt.close()


def side_by_side_comparison_plotter(
    nuclear_species,
    applied_field_list,
    rf_freq_list,
    loc_list,
    region_label="entire_dot",
    saving=False,
    recalc=False,
    use_sundfors_GET_vals=False,
):
    for field_geometry in ["Faraday", "Voigt"]:
        many_location_parallel_calculation(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            loc_list,
            region_label,
            recalc,
            use_sundfors_GET_vals,
        )

    faraday_data = np.log(
        load_many_location_data(
            nuclear_species,
            applied_field_list,
            "Faraday",
            rf_freq_list,
            loc_list,
            region_label,
            use_sundfors_GET_vals,
        )
    )
    voigt_data = np.log(
        load_many_location_data(
            nuclear_species,
            applied_field_list,
            "Voigt",
            rf_freq_list,
            loc_list,
            region_label,
            use_sundfors_GET_vals,
        )
    )

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    ax1.imshow(
        faraday_data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )
    ax2.imshow(
        voigt_data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )

    ax1.set_title(f"{nuclear_species}, Faraday Orientation")
    ax1.set_xlabel("RF Frequency (MHz)")
    ax1.set_ylabel("Applied B Field (T)")
    ax1.text(0.05, 0.95, "a", transform=ax1.transAxes)

    ax2.set_title(f"{nuclear_species}, Voigt Orientation")
    ax2.set_xlabel("RF Frequency (MHz)")
    ax2.set_ylabel("Applied B Field (T)")
    ax2.text(0.05, 0.95, "b", transform=ax2.transAxes)

    if saving:
        plotname = f"{graph_path}comparison_NMR_spectra_{nuclear_species}_{len(loc_list)}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname, bbox_inches="tight")
        print(f"Saved file: {plotname}")
    else:
        plt.show()

    plt.close()


def side_by_side_GET_comparison_plotter(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_label="entire_dot",
    saving=False,
    recalc=False,
):
    for use_sundfors_GET_vals in [True, False]:
        many_location_parallel_calculation(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            loc_list,
            region_label,
            recalc,
            use_sundfors_GET_vals,
        )

    chek_data = np.log(
        load_many_location_data(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            loc_list,
            region_label,
            use_sundfors_GET_vals=False,
        )
    )
    sundfors_data = np.log(
        load_many_location_data(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            loc_list,
            region_label,
            use_sundfors_GET_vals=True,
        )
    )

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    ax1.imshow(
        chek_data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )
    ax2.imshow(
        sundfors_data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )

    ax1.set_title(f"{nuclear_species}, {field_geometry} Orientation, New GET")
    ax1.set_xlabel("RF Frequency (MHz)")
    ax1.set_ylabel("Applied B Field (T)")
    ax1.text(0.05, 0.95, "a", transform=ax1.transAxes)

    ax2.set_title(f"{nuclear_species}, {field_geometry} Orientation, Old GET")
    ax2.set_xlabel("RF Frequency (MHz)")
    ax2.set_ylabel("Applied B Field (T)")
    ax2.text(0.05, 0.95, "b", transform=ax2.transAxes)

    if saving:
        plotname = f"{graph_path}GET_comparison_NMR_spectra_{nuclear_species}_{len(loc_list)}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname, bbox_inches="tight")
        print(f"Saved file: {plotname}")
    else:
        plt.show()

    plt.close()


def Sundfors_Checkovich_comparison(
    nuclear_species,
    applied_field_list,
    rf_freq_list,
    loc_list,
    region_label,
    field_geometry,
    saving=False,
    recalc=False,
):
    many_location_parallel_calculation(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label,
        recalc,
        use_sundfors_GET_vals=True,
    )
    many_location_parallel_calculation(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label,
        recalc,
        use_sundfors_GET_vals=False,
    )

    old_GET_data = load_many_location_data(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label,
        use_sundfors_GET_vals=True,
    )
    new_GET_data = load_many_location_data(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label,
        use_sundfors_GET_vals=False,
    )

    data = np.log(new_GET_data) - np.log(old_GET_data)

    data_max = np.max(data)
    data_min = np.min(data)
    # scale = np.max([data_max, data_min])

    # data = data + scale

    # transformed_data = np.log(np.abs(data))

    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    cax = ax.imshow(
        data,
        origin="lower",
        cmap=cm.RdBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )  # , vmin = -scale, vmax = scale)

    cbar = fig.colorbar(
        cax,
        orientation="vertical",
        ticks=[0.9 * data_min, 0, 0.9 * data_max],
        shrink=0.95,
    )
    cbar.ax.set_yticklabels(
        ["Old GET \nDominates", "Similar \nValues", "New GET \nDominates"]
    )

    ax.set_title(f"{nuclear_species}")
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Applied B Field (T)")

    if saving:
        plotname = f"{graph_path}comparison_spectra_with_varying_GET_for_{nuclear_species}__{len(loc_list)}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname, bbox_inches="tight")
        print(f"Saved file: {plotname}")
    else:
        plt.show()

    plt.close()


def testing_parallel_speedup(
    nuclear_species,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    location,
    loc_list,
    region_bounds=[100, 1200, 439, 880],
):
    # timing if parallel is faster
    # tests done on a 100 * 100 grid, without showing or saving any graphs

    start = time.time()
    many_field_colourplot(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        location,
        region_bounds,
        saving=False,
        graphing=False,
    )
    end = time.time()
    taken = np.around((end - start) / 60, 2)
    print(f"Single calc using non parallel function took {taken} minutes.")

    start = time.time()
    many_location_parallel_plot(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_bounds,
        saving=False,
        graphing=False,
    )
    end = time.time()
    taken = np.around((end - start) / 60, 2)
    print(f"Single calc using parallel function took {taken} minutes.")

    start = time.time()
    for location in loc_list:
        many_field_colourplot(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            location,
            region_bounds,
            saving=False,
            graphing=False,
        )
    end = time.time()
    taken = np.around((end - start) / 60, 2)
    print(f"Many calcs using non parallel function took {taken} minutes.")

    start = time.time()
    many_location_parallel_plot(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_bounds,
        saving=False,
        graphing=False,
    )
    end = time.time()
    taken = np.around((end - start) / 60, 2)
    print(f"Many calcs using non parallel function took {taken} minutes.")


def large_hd_spectra(
    nuclear_species, loc_list, region_label, field_geometry, recalc, saving=True
):
    min_freq = 0
    max_freq = 100  # in MHz
    n_freqs = 1000
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 3  # in Tesla
    n_fields = 1000

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    many_location_plot_using_loading(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label=region_label,
        saving=saving,
        recalc=recalc,
    )


def region_characterisation_spectra(
    nuclear_species, region_label, field_geometry, recalc, saving=True
):
    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 600
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 600

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    # we expect 55 atoms in a region, but we only use 16 here
    # partly for speed, and partly because we don't expect every atom to be exactly the same
    search_range = (
        25  # search an area of size 0.6nm**2 around each site to find an atom
    )
    n_locs = 18  # picked 16 to allow us to show the spread across a region in 2 processor cycles
    best_loc_list = rps.find_best_locations(n_locs, region_label, search_range)

    many_location_plot_using_loading(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        best_loc_list,
        region_label=region_label,
        saving=saving,
        recalc=recalc,
    )


def atomic_region_spectra(nuclear_species, field_geometry, recalc, saving=True):
    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 100
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 100

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    region_label = "single_atom_region"
    n_locs = 40  # we want a really good characterisation of this region, so we take a lot of points
    loc_list = random_locations_list_generator(n_locs, region_label)

    many_location_plot_using_loading(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label=region_label,
        saving=saving,
        recalc=recalc,
    )


def comparison_spectra(nuclear_species, loc_list, region_label, recalc, saving=True):
    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 1000
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 1000

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    Faraday_Voigt_spectra_comparison(
        nuclear_species,
        applied_field_list,
        rf_freq_list,
        loc_list,
        region_label=region_label,
        saving=saving,
        recalc=recalc,
    )


def GET_values_comparison_spectra(
    nuclear_species, field_geometry, loc_list, region_label, recalc, saving=True
):
    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 600
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 600

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    Sundfors_Checkovich_comparison(
        nuclear_species,
        applied_field_list,
        rf_freq_list,
        loc_list,
        region_label,
        field_geometry,
        saving=saving,
        recalc=recalc,
    )
    side_by_side_GET_comparison_plotter(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label="entire_dot",
        saving=saving,
        recalc=recalc,
    )


def experimental_NMR_sim(field_geometry, region_label, n_locs, recalc, saving=True):
    sys.path.append("/home/will/Documents/work/research/simulations/concentration/")
    import conc_maps as conc

    region_bounds = bqf.QD_regions_dict[region_label]
    conc_data = conc.load_In_concentration_data(region_bounds)

    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 300
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 300

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    # just use basic average to start with
    mean_In_conc = np.mean(conc_data)
    # print(mean_In_conc)

    # only including Ga69 to begin with
    n_In_locs = int(
        np.ceil(mean_In_conc * n_locs)
    )  # find how many In atoms there are, floor to keep as an integer
    n_As_locs = int(np.ceil(0.5 * n_locs))  # 50% of the lattice is As
    n_Ga_locs = int(n_locs - (n_In_locs + n_As_locs))  # the rest is Ga

    In_loc_list = random_locations_list_generator(n_In_locs, region_label)
    As_loc_list = random_locations_list_generator(n_As_locs, region_label)
    Ga_loc_list = random_locations_list_generator(n_Ga_locs, region_label)

    # done in the order InGaAs
    numbers = [n_In_locs, n_Ga_locs, n_As_locs]
    # print(numbers)
    location_lists = [In_loc_list, Ga_loc_list, As_loc_list]

    data = np.zeros((len(applied_field_list), len(rf_freq_list)))

    for i, nuclear_species in enumerate(["In115", "Ga69", "As75"]):
        species_loc_list = location_lists[i]
        many_location_parallel_calculation(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            species_loc_list,
            region_label,
            recalc,
        )
        data += load_many_location_data(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            species_loc_list,
            region_label,
        )

    data = np.log(data)
    # print(data)
    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    plt.imshow(
        data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=img_extent,
        interpolation="none",
    )
    plt.title(f"Experimental Estimate, {field_geometry} Orientation")

    plt.xlabel("RF Frequency (MHz)")
    plt.ylabel("Applied B Field (T)")

    if saving:
        plotname = f"{graph_path}NMR_spectra_experiment_in_{field_geometry}_{n_locs}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname, bbox_inches="tight")
        print(f"Saved file: {plotname}")
    else:
        pass
        plt.show()

    plt.close()


def experimental_NMR_sim_transparent_version(
    field_geometry, region_label, n_locs, recalc, saving=False
):
    sys.path.append("/home/will/Documents/work/research/simulations/concentration/")
    import conc_maps as conc

    region_bounds = bqf.QD_regions_dict[region_label]
    conc_data = conc.load_In_concentration_data(region_bounds)

    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 500
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 500

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    # just use basic average to start with
    mean_In_conc = np.mean(conc_data)
    # print(mean_In_conc)

    # only including Ga69 to begin with
    n_In_locs = int(
        np.ceil(mean_In_conc * n_locs)
    )  # find how many In atoms there are, floor to keep as an integer
    n_As_locs = int(np.ceil(0.5 * n_locs))  # 50% of the lattice is As
    n_Ga_locs = int(n_locs - (n_In_locs + n_As_locs))  # the rest is Ga

    In_loc_list = random_locations_list_generator(n_In_locs, region_label)
    As_loc_list = random_locations_list_generator(n_As_locs, region_label)
    Ga_loc_list = random_locations_list_generator(n_Ga_locs, region_label)

    # done in the order InGaAs
    numbers = [n_In_locs, n_Ga_locs, n_As_locs]
    # print(numbers)
    location_lists = [In_loc_list, Ga_loc_list, As_loc_list]

    data = []

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # print(ax)
    img_extent = [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]

    # Colour mappings:
    # In115 in Grey
    # Ga69 in Blue
    # As75 in Red

    cmap_list = [cm.Greys, cm.Blues, cm.Reds]
    alpha_list = [
        1,
        0.5,
        0.25,
    ]  # decreasing so that we don't see only red or blue lines

    for i, nuclear_species in enumerate(["In115", "Ga69", "As75"]):
        species_loc_list = location_lists[i]
        many_location_parallel_calculation(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            species_loc_list,
            region_label,
            recalc,
        )
        data.append(
            np.log(
                load_many_location_data(
                    nuclear_species,
                    applied_field_list,
                    field_geometry,
                    rf_freq_list,
                    species_loc_list,
                    region_label,
                )
            )
        )
        ax.imshow(
            data[i],
            origin="lower",
            cmap=cmap_list[i],
            aspect="auto",
            extent=img_extent,
            interpolation="none",
            alpha=alpha_list[i],
        )

    ax.set_title(f"Experimental Estimate, {field_geometry} Orientation")

    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Applied B Field (T)")

    if saving:
        plotname = f"{graph_path}layered_NMR_spectra_experiment_in_{field_geometry}_{n_locs}_locs_in_{region_label}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz.png"
        plt.savefig(plotname, bbox_inches="tight")
        print(f"Saved file: {plotname}")
    else:
        pass
        plt.show()

    plt.close()


def gigahertz_range_spectra(
    nuclear_species, loc_list, region_label, field_geometry, recalc, saving=True
):
    if nuclear_species == "In115":
        min_freq = 0
        max_freq = 500  # in MHz
        n_freqs = 1000
        min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
        max_field = 5  # in Tesla
        n_fields = 1000
    else:
        min_freq = 0
        max_freq = 200  # in MHz
        n_freqs = 500
        min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
        max_field = 4  # in Tesla
        n_fields = 500

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    # print(f"function is gigahertz_range_spectra")
    # print(f"nuclear_species = {nuclear_species}")
    # print(f"field_geometry = {field_geometry}")

    many_location_plot_using_loading(
        nuclear_species,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        region_label=region_label,
        saving=saving,
        recalc=recalc,
    )


def single_field_line_spectra(
    nuclear_species,
    applied_field,
    loc_list,
    region_label,
    field_geometry,
    recalc,
    rf_field_strength=5e-3,
    saving=False,
    returning=False,
):
    min_freq = 0
    max_freq = 150
    n_freqs = 1000
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    region_bounds = bqf.QD_regions_dict[region_label]

    absorbtion_data = np.zeros(n_freqs)

    for location in loc_list:
        absorbtion_data += calculate_absorbtion_single_slice_data(
            nuclear_species,
            applied_field,
            field_geometry,
            rf_freq_list,
            location,
            region_bounds,
            data_only_return=True,
            rf_field=rf_field_strength,
        )

    absorbtion_data = absorbtion_data / np.amax(absorbtion_data)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(rf_freq_list / 1e6, absorbtion_data, color=bqf.will_dark_blue)
    ax.set_title(
        f"{nuclear_species}, {np.around(applied_field, 2)}T Applied Field, {field_geometry} Orientation"
    )
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Absorbtion (arb. units)")
    ax.text(
        0.75,
        0.85,
        f"{len(loc_list)} Nuclei \n{rf_field_strength}T RF Field",
        transform=ax.transAxes,
    )

    if saving:
        plotname = f"{graph_path}NMR_spectra_slice_{nuclear_species}_{applied_field}T_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_rf_strength_{rf_field_strength}.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        pass
        # plt.show()

    plt.close()

    if returning:
        return rf_freq_list, absorbtion_data


def applied_field_comparison_line_spectra(
    nuclear_species,
    applied_field_list,
    loc_list,
    region_label,
    field_geometry,
    recalc,
    rf_field_strength=5e-3,
    saving=False,
    in_parallel=False,
):
    min_freq = 0
    max_freq = 100
    n_freqs = 1000
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    n_fields = len(applied_field_list)

    region_bounds = bqf.QD_regions_dict[region_label]

    absorbtion_data = np.zeros((n_fields, n_freqs))

    filename = f"{data_path}NMR_spectra_slice_{nuclear_species}_varied_applied_fields_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_rf_strength_{rf_field_strength}.npy"
    if os.path.isfile(filename):
        absorbtion_data = np.load(filename)
    else:
        if in_parallel:
            for a, applied_field in enumerate(applied_field_list):
                with multiprocessing.Pool() as pool:
                    zipped_params = parameter_zipper(
                        nuclear_species,
                        applied_field,
                        field_geometry,
                        rf_freq_list,
                        loc_list,
                        region_bounds,
                        use_sundfors_GET_vals=False,
                        rf_field=rf_field_strength,
                    )
                    slices_across_locations = pool.starmap(
                        calculate_absorbtion_single_slice_data, zipped_params
                    )
                    pool.close()
                absorbtion_data[a, :] = np.sum(slices_across_locations, axis=0)
        else:
            for a, applied_field in enumerate(applied_field_list):
                for location in loc_list:
                    absorbtion_data[a, :] += calculate_absorbtion_single_slice_data(
                        nuclear_species,
                        applied_field,
                        field_geometry,
                        rf_freq_list,
                        location,
                        region_bounds,
                        data_only_return=True,
                        rf_field=rf_field_strength,
                    )

        absorbtion_data = absorbtion_data / np.amax(absorbtion_data)

        filename = f"{data_path}NMR_spectra_slice_{nuclear_species}_varied_applied_fields_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_rf_strength_{rf_field_strength}.npy"
        np.save(filename, absorbtion_data)

    fig, axs = plt.subplots(nrows=n_fields, ncols=1, sharex=True, figsize=(16, 10))

    # fig.suptitle(f"{nuclear_species}, Varied Appled Field in {field_geometry} Orientation \n{len(loc_list)} Nuclei \n{rf_field_strength}T RF Field")

    for a, ax in enumerate(axs):
        ax.plot(rf_freq_list / 1e6, absorbtion_data[a, :], color=bqf.will_dark_blue)
        # ax.set_ylabel("Absorbtion (arb. units)")
        # ax.set_xlabel("RF Frequency (MHz)")
        ax.text(
            0.75, 0.5, f"{applied_field_list[a]}T Applied Field", transform=ax.transAxes
        )

    if saving:
        plotname = f"{graph_path}NMR_spectra_slice_{nuclear_species}_varied_applied_fields_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_rf_strength_{rf_field_strength}.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def RF_comparison_line_spectra(
    nuclear_species,
    applied_field,
    loc_list,
    region_label,
    field_geometry,
    rf_field_list,
    recalc=False,
    saving=True,
    in_parallel=False,
):
    min_freq = 0
    max_freq = 100
    n_freqs = 1000
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    n_RF_fields = len(rf_field_list)

    region_bounds = bqf.QD_regions_dict[region_label]

    absorbtion_data = np.zeros((n_RF_fields, n_freqs))

    filename = f"{data_path}NMR_spectra_slice_{nuclear_species}_{n_RF_fields}_RF_fields_in_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_applied_field_{applied_field}T.npy"
    if os.path.isfile(filename):
        absorbtion_data = np.load(filename)
    else:
        if in_parallel:
            for r, rf_field in enumerate(rf_field_list):
                with multiprocessing.Pool() as pool:
                    zipped_params = parameter_zipper(
                        nuclear_species,
                        applied_field,
                        field_geometry,
                        rf_freq_list,
                        loc_list,
                        region_bounds,
                        use_sundfors_GET_vals=False,
                        rf_field=rf_field,
                    )
                    slices_across_locations = pool.starmap(
                        calculate_absorbtion_single_slice_data, zipped_params
                    )
                    pool.close()
                absorbtion_data[r, :] = np.sum(slices_across_locations, axis=0)

        else:
            for r, rf_field in enumerate(rf_field_list):
                for location in loc_list:
                    absorbtion_data[r, :] += calculate_absorbtion_single_slice_data(
                        nuclear_species,
                        applied_field,
                        field_geometry,
                        rf_freq_list,
                        location,
                        region_bounds,
                        data_only_return=True,
                        rf_field=rf_field,
                    )

        absorbtion_data = absorbtion_data / np.amax(absorbtion_data)

        filename = f"{data_path}NMR_spectra_slice_{nuclear_species}_{n_RF_fields}_RF_fields_in_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_applied_field_{applied_field}T.npy"
        np.save(filename, absorbtion_data)

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.set_ylabel("Absorbtion (arb. units)")
    ax.set_xlabel("RF Frequency (MHz)")

    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    idx1, x1 = array_find_nearest(rf_freq_list / 1e6, 69)
    idx2, x2 = array_find_nearest(rf_freq_list / 1e6, 91)
    y1 = -0.01
    y2 = 0.1

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    for a in range(n_RF_fields):
        ax.plot(
            rf_freq_list / 1e6, absorbtion_data[a, :], label=f"{rf_field_list[a]:.2e}"
        )
        axins.plot(rf_freq_list[idx1:idx2] / 1e6, absorbtion_data[a, idx1:idx2])

    plt.legend()

    if saving:
        plotname = f"{graph_path}NMR_spectra_slice_{nuclear_species}_varied_RF_fields_in_{field_geometry}_orientation_for_{len(loc_list)}_nuclei_with_applied_field_{applied_field}T.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def number_of_nuclei_comparison(
    nuclear_species,
    applied_field,
    n_nuclei_list,
    region_label,
    field_geometry,
    rf_field,
    recalc=False,
    saving=True,
    in_parallel=False,
):
    min_freq = 0
    max_freq = 150
    n_freqs = 1000
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    region_bounds = bqf.QD_regions_dict[region_label]

    absorbtion_data = np.zeros((len(n_nuclei_list), n_freqs))

    filename = f"{data_path}NMR_spectra_slice_{nuclear_species}_varied_number_of_nuclei_between{n_nuclei_list[0]}_and_{n_nuclei_list[-1]}_in_{field_geometry}_orientation_with_applied_field_{applied_field}T.npy"
    if os.path.isfile(filename):
        absorbtion_data = np.load(filename)
    else:
        if in_parallel:
            for n, n_nuclei in enumerate(n_nuclei_list):
                loc_list = random_locations_list_generator(n_nuclei, region_label)
                with multiprocessing.Pool() as pool:
                    zipped_params = parameter_zipper(
                        nuclear_species,
                        applied_field,
                        field_geometry,
                        rf_freq_list,
                        loc_list,
                        region_bounds,
                        use_sundfors_GET_vals=False,
                        rf_field=rf_field,
                    )
                    slices_across_locations = pool.starmap(
                        calculate_absorbtion_single_slice_data, zipped_params
                    )
                    pool.close()
                absorbtion_data[n, :] = (
                    np.sum(slices_across_locations, axis=0) / n_nuclei
                )

        else:
            for n, n_nuclei in enumerate(n_nuclei_list):
                loc_list = random_locations_list_generator(n_nuclei, region_label)
                for location in loc_list:
                    absorbtion_data[n, :] += calculate_absorbtion_single_slice_data(
                        nuclear_species,
                        applied_field,
                        field_geometry,
                        rf_freq_list,
                        location,
                        region_bounds,
                        data_only_return=True,
                        rf_field=rf_field,
                    )
                absorbtion_data = absorbtion_data / n_nuclei

        absorbtion_data = absorbtion_data / np.amax(absorbtion_data)

        filename = f"{data_path}NMR_spectra_slice_{nuclear_species}_varied_number_of_nuclei_between{n_nuclei_list[0]}_and_{n_nuclei_list[-1]}_in_{field_geometry}_orientation_with_applied_field_{applied_field}T.npy"
        np.save(filename, absorbtion_data)

    fig, ax = plt.subplots(figsize=(16, 10))

    # fig.suptitle(f"{nuclear_species}, Varied Number of Nuclei in {field_geometry} Orientation \n{applied_field}T Field")

    ax.set_ylabel("Absorbtion (arb. units)")
    ax.set_xlabel("RF Frequency (MHz)")

    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    idx1, x1 = array_find_nearest(rf_freq_list / 1e6, 69)
    idx2, x2 = array_find_nearest(rf_freq_list / 1e6, 91)
    y1 = -0.01
    y2 = 0.1

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    for a in range(len(n_nuclei_list)):
        ax.plot(rf_freq_list / 1e6, absorbtion_data[a, :], label=f"{n_nuclei_list[a]}")
        axins.plot(rf_freq_list[idx1:idx2] / 1e6, absorbtion_data[a, idx1:idx2])

    plt.legend()

    if saving:
        plotname = f"{graph_path}NMR_spectra_slice_{nuclear_species}_varied_number_of_nuclei_in_{field_geometry}_orientation_with_applied_field_{applied_field}T.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_lorentzian_pulse(rf_freq_list, location, width):
    gamma = width / 2

    dist = 1 / (np.pi * gamma * (1 + ((rf_freq_list - location) / gamma) ** 2))

    return dist


def load_combined_absorbtion_data(
    field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei
):
    filename = f"{data_path}all_species_NMR_spectra_slice_for_{n_freqs}_freqs_between_{min_freq}_{max_freq}MHz_at_{applied_field}T_in_{field_geometry}_over_{n_nuclei}_nuclei.npy"

    if os.path.isfile(filename):
        absorbtion_data = np.load(filename)
    else:
        raise FileExistsError(f"{filename} does not exist - it needs to be calculated!")

    return absorbtion_data


def overlaid_pulse_graphs(
    pulse_location,
    pulse_width,
    field_geometry,
    min_freq,
    max_freq,
    n_freqs,
    applied_field,
    n_nuclei,
    saving=False,
):
    absorbtion_data = load_combined_absorbtion_data(
        field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei
    )
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    pulse = create_lorentzian_pulse(
        rf_freq_list, pulse_location * 1e6, pulse_width * 1e6
    )
    pulse = pulse / np.amax(pulse)
    multiplied_absorption = absorbtion_data * pulse

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(
        rf_freq_list / 1e6,
        pulse,
        label="Pulse",
        linestyle="dotted",
        color=bqf.will_light_blue,
    )
    ax.plot(
        rf_freq_list / 1e6,
        multiplied_absorption,
        label="Absorbtion",
        color=bqf.will_dark_blue,
    )
    ax.text(
        0.05, 0.95, rf" {pulse_location} $\pm$ {pulse_width/2}", transform=ax.transAxes
    )
    ax.legend()

    if saving:
        plotname = f"{graph_path}pulse_overlaid_graph_for_all_species_in_{field_geometry}_field_and_{pulse_location}MHz_pulse_with_width_{pulse_width}MHz_std_dev.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def grid_overlaid_pulse_graphs(
    field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei, saving=False
):
    absorbtion_data = load_combined_absorbtion_data(
        field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei
    )
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    # these are input manually from the results of plotting the peak locations
    pulse_location_range = [80.2]  # [55.4, 58.4, 80.2]
    pulse_width_range = [0.01] + list(np.arange(1, 12, 3))
    fig, axs = plt.subplots(
        len(pulse_location_range), len(pulse_width_range), figsize=(16, 10)
    )

    for m, pulse_location in enumerate(pulse_location_range):
        for s, pulse_width in enumerate(pulse_width_range):
            if len(pulse_location_range) == 1:
                ax = axs[s]
            else:
                ax = axs[m, s]

            pulse = create_lorentzian_pulse(
                rf_freq_list, pulse_location * 1e6, pulse_width * 1e6
            )
            pulse = pulse / np.amax(pulse)
            multiplied_absorption = absorbtion_data * pulse

            # ax.plot(rf_freq_list/1e6, pulse*0.05, label = "Pulse", linestyle = "dotted", color = bqf.will_light_blue)
            ax.plot(
                rf_freq_list / 1e6,
                multiplied_absorption,
                label="Absorbtion",
                color=bqf.will_dark_blue,
            )
            ax.axis("off")
            ax.text(
                0.05,
                0.95,
                rf" {pulse_location} $\pm$ {pulse_width/2}",
                transform=ax.transAxes,
                fontsize=24,
            )

    if saving:
        plotname = f"{graph_path}grid_of_{len(pulse_location_range)}_x_{len(pulse_width_range)}_varying_location_and_width_pulses_over_{n_nuclei}_nuclei_for_all_species_in_{field_geometry}.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def combined_species_single_spectra_data(
    field_geometry,
    min_freq=0,
    max_freq=100,
    n_freqs=1000,
    applied_field=3,
    region_label="entire_dot",
    rf_field_strength=5e-3,
    n_nuclei=1200,
):
    region_bounds = bqf.QD_regions_dict[region_label]
    loc_list = random_locations_list_generator(n_nuclei, region_label)

    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6
    absorbtion_data = np.zeros(n_freqs)

    for nuclear_species in bqf.nuclear_species_list:
        with multiprocessing.Pool() as pool:
            zipped_params = parameter_zipper(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                loc_list,
                region_bounds,
                use_sundfors_GET_vals=False,
                rf_field=rf_field_strength,
            )
            slices_across_locations = pool.starmap(
                calculate_absorbtion_single_slice_data, zipped_params
            )
            pool.close()
        absorbtion_data += np.sum(slices_across_locations, axis=0)

    absorbtion_data = absorbtion_data / np.amax(absorbtion_data)

    filename = f"{data_path}all_species_NMR_spectra_slice_for_{n_freqs}_freqs_between_{min_freq}_{max_freq}MHz_at_{applied_field}T_in_{field_geometry}_over_{n_nuclei}_nuclei.npy"
    np.save(filename, absorbtion_data)


def combined_species_single_spectra_plot(
    field_geometry,
    min_freq=0,
    max_freq=100,
    n_freqs=1000,
    applied_field=3,
    region_label="entire_dot",
    rf_field_strength=5e-3,
    n_nuclei=1200,
    saving=False,
):
    data = load_combined_absorbtion_data(
        field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei
    )

    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(rf_freq_list / 1e6, data, color=bqf.will_dark_blue)

    ax.set_xlabel("NMR Frequency (MHz)")
    ax.set_ylabel("Absorbtion (arb. units)")

    peak_locs, _ = scipy.signal.find_peaks(data, prominence=0.01)
    peak_freqs = np.around(rf_freq_list[peak_locs] / 1e6, 1)

    # print(peak_freqs)

    ax.plot(rf_freq_list[peak_locs] / 1e6, data[peak_locs], "x")

    if saving:
        plotname = f"{graph_path}combined_species_single_spectra_plot_over_{n_freqs}_between_{min_freq}_{max_freq}MHz_at_{applied_field}T_in_{field_geometry}_for_{n_nuclei}_nuclei.png"
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def combined_species_single_spectra_plot_subregion(
    field_geometry,
    target_min=70,
    target_max=90,
    min_freq=0,
    max_freq=100,
    n_freqs=1000,
    applied_field=3,
    region_label="entire_dot",
    rf_field_strength=5e-3,
    n_nuclei=1200,
    saving=False,
):
    data = load_combined_absorbtion_data(
        field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei
    )

    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6

    target_min_arg, target_min_val = array_find_nearest(rf_freq_list / 1e6, target_min)
    target_max_arg, target_max_val = array_find_nearest(rf_freq_list / 1e6, target_max)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(
        rf_freq_list[target_min_arg:target_max_arg] / 1e6,
        data[target_min_arg:target_max_arg],
        color=bqf.will_dark_blue,
    )

    ax.set_xlabel("NMR Frequency (MHz)")
    ax.set_ylabel("Absorbtion (arb. units)")

    if target_min == 70 and target_max == 90:
        # these box limits are found to make the box line up on 77.5 and 82.5
        # only found by eye I'm afraid (10/12/21)
        box_min_x = 77.8
        box_max_x = 82.3
        shaded_min = (box_min_x - target_min_val) / (target_max_val - target_min_val)
        shaded_max = (box_max_x - target_min_val) / (target_max_val - target_min_val)
        print(shaded_min, shaded_max)
        ax.axhspan(
            0,
            1.1 * np.amax(data[target_min_arg:target_max_arg]),
            xmin=shaded_min,
            xmax=shaded_max,
            color="red",
            alpha=0.5,
        )

    if saving:
        plotname = f"{graph_path}combined_species_single_spectra_plot_between_{target_min}_{target_max}MHz_subset_at_{applied_field}T_in_{field_geometry}_for_{n_nuclei}_nuclei.png"
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def pulse_graph_area_calculations(
    field_geometry,
    pulse_location,
    pulse_width,
    min_freq=0,
    max_freq=100,
    n_freqs=1000,
    applied_field=3,
    region_label="entire_dot",
    rf_field_strength=5e-3,
    n_nuclei=1200,
):
    rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6
    absorbtion_data = load_combined_absorbtion_data(
        field_geometry, min_freq, max_freq, n_freqs, applied_field, n_nuclei
    )

    # overall pulse calcs - finding area under the pulse
    pulse = create_lorentzian_pulse(
        rf_freq_list, pulse_location * 1e6, pulse_width * 1e6
    )
    pulse = pulse / np.amax(pulse)
    multiplied_absorption = absorbtion_data * pulse

    area_under_original = scipy.integrate.simpson(absorbtion_data, rf_freq_list / 1e6)
    area_under_pulse = scipy.integrate.simpson(
        multiplied_absorption, rf_freq_list / 1e6
    )
    fraction_under_pulse = area_under_pulse / area_under_original * 100

    # pulse specific area - finding useful area

    print("------------------------------------------")
    print(f"Pulse Parameters: {pulse_location} +/- {pulse_width/2}")
    print(f"Original Area: {np.around(area_under_original, 2)}")
    print(f"Pulse Area: {np.around(area_under_pulse, 2)}")
    print(f"Percentage: {np.around(fraction_under_pulse, 2)}")


def array_find_nearest(data_array, value):
    idx = np.abs(data_array - value).argmin()
    return idx, data_array[idx]


def comparison_area_plots(
    field_geometry,
    pulse_location,
    pulse_width_range,
    applied_field=3,
    region_label="entire_dot",
    rf_field_strength=5e-3,
    n_nuclei=1200,
    saving=False,
):
    overall_min_freq = 0
    overall_max_freq = 150
    overall_n_freqs = 10000
    overall_n_nuclei = 10000

    target_min_freq = 77.5
    target_max_freq = 82.5

    side_min_freq = 50
    side_max_freq = 70

    large_side_min_freq = 20
    large_side_max_freq = 40

    rf_freq_list = (
        np.linspace(overall_min_freq, overall_max_freq, overall_n_freqs) * 1e6
    )

    pulse_min_index, _ = array_find_nearest(rf_freq_list / 1e6, target_min_freq)
    pulse_max_index, _ = array_find_nearest(rf_freq_list / 1e6, target_max_freq)

    side_min_index, _ = array_find_nearest(rf_freq_list / 1e6, side_min_freq)
    side_max_index, _ = array_find_nearest(rf_freq_list / 1e6, side_max_freq)

    large_side_min_index, _ = array_find_nearest(
        rf_freq_list / 1e6, large_side_min_freq
    )
    large_side_max_index, _ = array_find_nearest(
        rf_freq_list / 1e6, large_side_max_freq
    )

    absorbtion_data = load_combined_absorbtion_data(
        field_geometry,
        overall_min_freq,
        overall_max_freq,
        overall_n_freqs,
        applied_field,
        overall_n_nuclei,
    )

    spectrum_area = scipy.integrate.simpson(absorbtion_data, rf_freq_list / 1e6)
    pre_pulse_target_area = scipy.integrate.simpson(
        absorbtion_data[pulse_min_index:pulse_max_index],
        rf_freq_list[pulse_min_index:pulse_max_index] / 1e6,
    )
    pre_pulse_side_area = scipy.integrate.simpson(
        absorbtion_data[side_min_index:side_max_index],
        rf_freq_list[side_min_index:side_max_index] / 1e6,
    )
    pre_pulse_large_side_area = scipy.integrate.simpson(
        absorbtion_data[large_side_min_index:large_side_max_index],
        rf_freq_list[large_side_min_index:large_side_max_index] / 1e6,
    )

    pulse_area_data = np.zeros(len(pulse_width_range))
    target_area_data = np.zeros(len(pulse_width_range))
    target_saturation_data = np.zeros(len(pulse_width_range))
    side_area_data = np.zeros(len(pulse_width_range))
    large_side_area_data = np.zeros(len(pulse_width_range))

    for p, pulse_width in enumerate(pulse_width_range):
        # overall pulse calcs - finding area under the pulse

        # as of 10/12/21 I need to make the plots of target area/initial target area and of the rest of it - as discussed with Andrew

        pulse = create_lorentzian_pulse(
            rf_freq_list, pulse_location * 1e6, pulse_width * 1e6
        )
        pulse = pulse / np.amax(pulse)
        final_spectrum = absorbtion_data * pulse

        total_absorption = scipy.integrate.simpson(final_spectrum, rf_freq_list / 1e6)
        pulse_fractional_area = total_absorption / spectrum_area * 100

        area_in_target_region = scipy.integrate.simpson(
            final_spectrum[pulse_min_index:pulse_max_index],
            rf_freq_list[pulse_min_index:pulse_max_index] / 1e6,
        )
        area_in_side_region = scipy.integrate.simpson(
            final_spectrum[side_min_index:side_max_index],
            rf_freq_list[side_min_index:side_max_index] / 1e6,
        )
        area_in_large_side_region = scipy.integrate.simpson(
            final_spectrum[large_side_min_index:large_side_max_index],
            rf_freq_list[large_side_min_index:large_side_max_index] / 1e6,
        )

        target_fractional_area = area_in_target_region / total_absorption * 100
        target_saturation_fraction = area_in_target_region / pre_pulse_target_area * 100
        side_fractional_area = area_in_side_region / total_absorption * 100

        # pulse_area_data[p] = pulse_fractional_area
        # target_saturation_data[p] = target_saturation_fraction
        # target_area_data[p] = target_fractional_area
        # side_area_data[p] = side_fractional_area

        pulse_area_data[p] = total_absorption
        # target_saturation_data[p] = target_saturation_fraction
        target_area_data[p] = area_in_target_region
        side_area_data[p] = area_in_side_region
        large_side_area_data[p] = area_in_large_side_region

        # side_area_data[p] = total_absorption - area_in_target_region

    target_max = np.amax(target_area_data)
    target_max_arg = np.argmax(target_area_data)

    saturation_max = np.amax(target_saturation_data)
    saturation_max_arg = np.argmax(target_saturation_data)

    pulse_width_target_max = pulse_width_range[target_max_arg]
    pulse_width_saturation_max = pulse_width_range[saturation_max_arg]

    # print(f"Max target fraction of {np.around(target_max, 2)}% at pulse width of {np.around(pulse_width_target_max,2)} MHz.")
    # print(f"Max saturation of {np.around(saturation_max, 2)}% at pulse width of {np.around(pulse_width_saturation_max,2)} MHz.")

    fig, ax = plt.subplots(figsize=(16, 10))
    # plt.plot(pulse_width_range, pulse_area_data, color = bqf.will_light_blue, linestyle = "dashed", label = "Overall Absorbtion")
    plt.plot(
        pulse_width_range,
        target_area_data / pre_pulse_target_area,
        color=bqf.will_dark_blue,
        label="77.5 - 82.5",
    )
    # plt.plot(pulse_width_range, target_saturation_data, color = bqf.will_light_blue, label = "Transition Saturation")
    plt.plot(
        pulse_width_range,
        side_area_data / pre_pulse_side_area,
        color=bqf.will_light_blue,
        label="50-70 MHz",
    )
    plt.plot(
        pulse_width_range,
        large_side_area_data / pre_pulse_large_side_area,
        color="seagreen",
        label="20-40 MHz",
    )

    # ax.vlines([pulse_width_target_max, pulse_width_saturation_max], [0,0], [target_max, saturation_max], linestyle = "dotted", color = "red")

    plt.xlabel("Pulse Width (MHz)")
    plt.ylabel("Fractional Absorbtion")
    plt.legend()

    if saving:
        plotname = f"{graph_path}fractional_absorption_energies_vs_widths_for_various_regions.png"
        plt.savefig(plotname, bbox_inches="tight")
    else:
        plt.show()


# -----------------------------------------------------------------------------

# Control Panel
testing_hd = False
HD_spectra = False
small_region_spectra = False
atomic_spectra = False
creating_comparison = False
experimental_plot = False
comparing_GET = False
side_by_side_comparison_plots = False
layered_NMR_experiment = False
gigahertz_spectra = False
line_spectra = False
changing_B_field_line_spectra = False
changing_RF_field_line_spectra = False
changing_number_of_nuclei = True

all_species_slice_calculation = False
plotting_NMR_slice_subrange = False
plotting_all_species_slice = False
applying_pulse = False
pulse_area_calcs = False
grid_apply_pulses = False
comparative_area_calcs = False

recalculating = False

# ------------------------------------------------------------------------------

start = time.time()

# create large range, high res images of a single site from the dot
# for illustrative purposes
if HD_spectra:
    loc_list = [(220, 550)]  # a single point, right in the middle of the dot
    for nuclear_species in bqf.nuclear_species_list:
        for field_geometry in bqf.field_geometries:
            large_hd_spectra(
                nuclear_species, loc_list, "entire_dot", field_geometry, recalculating
            )

# create smaller range images made up of multiple sites within a smaller region
# used for comparison of different dot regions
if small_region_spectra:
    for region_label in ["entire_dot"]:
        for nuclear_species in bqf.nuclear_species_list:
            for field_geometry in bqf.field_geometries:
                region_characterisation_spectra(
                    nuclear_species, region_label, field_geometry, True
                )

# create a detailed map of a single atomic region
# used for showing the variation in our data due to oversampling
if atomic_spectra:
    for nuclear_species in bqf.nuclear_species_list:
        for field_geometry in bqf.field_geometries:
            atomic_region_spectra(nuclear_species, field_geometry, recalculating)

if creating_comparison:
    for region_label in ["entire_dot"]:
        n_locs = 8
        loc_list = random_locations_list_generator(n_locs, region_label)
        for nuclear_species in bqf.nuclear_species_list:
            for field_geometry in bqf.field_geometries:
                comparison_spectra(
                    nuclear_species, loc_list, region_label, recalculating
                )

if experimental_plot:
    n_locs = 24
    for region_label in [
        "central_high_In",
        "central_low_In",
        "below_outside_dot",
        "entire_dot",
    ]:
        for field_geometry in bqf.field_geometries:
            experimental_NMR_sim(
                field_geometry, region_label, n_locs, recalculating, True
            )

if comparing_GET:
    for region_label in ["entire_dot"]:
        n_locs = 8
        loc_list = random_locations_list_generator(n_locs, region_label)
        for nuclear_species in ["Ga69", "As75"]:
            for field_geometry in ["Faraday"]:
                GET_values_comparison_spectra(
                    nuclear_species,
                    field_geometry,
                    loc_list,
                    region_label,
                    recalculating,
                )

if side_by_side_comparison_plots:
    min_freq = 0
    max_freq = 75  # in MHz
    n_freqs = 600
    min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
    max_field = 2  # in Tesla
    n_fields = 600

    rf_freq_list = (
        np.linspace(min_freq, max_freq, n_freqs) * 1e6
    )  # the 1e6 makes it MHz
    applied_field_list = np.linspace(min_field, max_field, n_fields)
    loc_list = [(220, 550)]

    side_by_side_comparison_plotter(
        "Ga69",
        applied_field_list,
        rf_freq_list,
        loc_list,
        region_label="entire_dot",
        saving=True,
        recalc=False,
        use_sundfors_GET_vals=False,
    )
    side_by_side_comparison_plotter(
        "Ga71",
        applied_field_list,
        rf_freq_list,
        loc_list,
        region_label="entire_dot",
        saving=True,
        recalc=False,
        use_sundfors_GET_vals=False,
    )
    side_by_side_comparison_plotter(
        "As75",
        applied_field_list,
        rf_freq_list,
        loc_list,
        region_label="entire_dot",
        saving=True,
        recalc=False,
        use_sundfors_GET_vals=False,
    )

    n_locs = 8
    loc_list = random_locations_list_generator(n_locs, "entire_dot")
    side_by_side_GET_comparison_plotter(
        "Ga69",
        applied_field_list,
        "Voigt",
        rf_freq_list,
        loc_list,
        region_label="entire_dot",
        saving=True,
        recalc=False,
    )
    side_by_side_GET_comparison_plotter(
        "As75",
        applied_field_list,
        "Voigt",
        rf_freq_list,
        loc_list,
        region_label="entire_dot",
        saving=True,
        recalc=False,
    )

if layered_NMR_experiment:
    region_label = "entire_dot"
    for field_geometry in bqf.field_geometries:
        experimental_NMR_sim_transparent_version(
            field_geometry, region_label, n_locs=24, recalc=False, saving=True
        )

if gigahertz_spectra:
    region_label = "entire_dot"
    n_locs = 100
    loc_list = random_locations_list_generator(n_locs, region_label)
    for nuclear_species in bqf.nuclear_species_list:
        for field_geometry in bqf.field_geometries:
            gigahertz_range_spectra(
                nuclear_species,
                loc_list,
                region_label,
                field_geometry,
                recalc=True,
                saving=True,
            )

if line_spectra:
    region_label = "entire_dot"
    n_locs = 1
    # loc_list = random_locations_list_generator(n_locs, region_label)
    loc_list = [(250, 250)]
    # applied_field_list = np.linspace(0.5, 5, 10)
    applied_field_list = [3]

    for applied_field in applied_field_list:
        for nuclear_species in ["In115"]:  # bqf.nuclear_species_list:
            for field_geometry in bqf.field_geometries:
                single_field_line_spectra(
                    nuclear_species,
                    applied_field,
                    loc_list,
                    region_label,
                    field_geometry,
                    recalc=False,
                    rf_field_strength=5e-3,
                    saving=True,
                )

if changing_B_field_line_spectra:
    region_label = "entire_dot"
    n_locs = 1200
    loc_list = random_locations_list_generator(n_locs, region_label)
    applied_field_list = np.linspace(0.5, 5, 10)

    for nuclear_species in ["In115"]:  # bqf.nuclear_species_list:
        for field_geometry in ["Voigt"]:  # bqf.field_geometries:
            applied_field_comparison_line_spectra(
                nuclear_species,
                applied_field_list,
                loc_list,
                region_label,
                field_geometry,
                recalc=False,
                rf_field_strength=5e-3,
                saving=True,
                in_parallel=True,
            )

if changing_RF_field_line_spectra:
    region_label = "entire_dot"
    n_locs = 1200
    loc_list = random_locations_list_generator(n_locs, region_label)
    applied_field = 3

    rf_field_list = np.logspace(-3, 2, 6) * 3

    for nuclear_species in ["In115"]:  # bqf.nuclear_species_list:
        for field_geometry in ["Voigt"]:  # bqf.field_geometries:
            RF_comparison_line_spectra(
                nuclear_species,
                applied_field,
                loc_list,
                region_label,
                field_geometry,
                rf_field_list,
                recalc=False,
                saving=True,
                in_parallel=True,
            )

if changing_number_of_nuclei:
    region_label = "entire_dot"
    applied_field = 3
    n_nuclei_list = np.logspace(0, 4, 5).astype(int)
    print(n_nuclei_list)

    for nuclear_species in ["In115"]:  # bqf.nuclear_species_list:
        for field_geometry in ["Voigt"]:  # bqf.field_geometries:
            number_of_nuclei_comparison(
                nuclear_species,
                applied_field,
                n_nuclei_list,
                region_label,
                field_geometry,
                rf_field=5e-3,
                recalc=False,
                saving=True,
                in_parallel=True,
            )


# standard vals
min_freq = 0
max_freq = 150
n_freqs = 10000

field_geometry = "Voigt"
applied_field = 3
# n_nuclei = 10000

for n_nuclei in [10000]:
    if all_species_slice_calculation:
        # min_freq = 0
        # max_freq = 100
        # n_freqs = 1000

        combined_species_single_spectra_data(
            field_geometry,
            min_freq,
            max_freq,
            n_freqs,
            applied_field,
            region_label="entire_dot",
            rf_field_strength=5e-3,
            n_nuclei=n_nuclei,
        )

        # field_geometry = "Voigt"
        # applied_field = 3
        # n_nuclei = 1
        # low_freq_list = [50, 60, 70]
        # high_freq_list = [110, 100, 90]
        # n_freqs = 10000

        # for i in range(3):
        # 	low_freq = low_freq_list[i]
        # 	high_freq = high_freq_list[i]
        # 	combined_species_single_spectra_data(field_geometry, low_freq, high_freq, n_freqs, applied_field, region_label = "entire_dot", rf_field_strength = 5e-3, n_nuclei = n_nuclei)

    if plotting_all_species_slice:
        # min_freq = 0
        # max_freq = 100
        # n_freqs = 1000

        # field_geometry = "Voigt"
        # applied_field = 3
        # n_nuclei = 1

        combined_species_single_spectra_plot(
            field_geometry,
            min_freq,
            max_freq,
            n_freqs,
            applied_field,
            region_label="entire_dot",
            rf_field_strength=5e-3,
            n_nuclei=n_nuclei,
            saving=True,
        )

    if plotting_NMR_slice_subrange:
        # target_min = 50
        # target_max = 100
        # combined_species_single_spectra_plot_subregion(field_geometry, target_min, target_max, min_freq = 0, max_freq = 150, n_freqs = 10000, applied_field = 3, region_label = "entire_dot", rf_field_strength = 5e-3, n_nuclei = 10000, saving = True)

        low_freq_list = [50, 60, 70]
        high_freq_list = [110, 100, 90]
        n_freqs = 10000

        for i in range(len(low_freq_list)):
            low_freq = low_freq_list[i]
            high_freq = high_freq_list[i]
            combined_species_single_spectra_plot_subregion(
                field_geometry,
                low_freq,
                high_freq,
                min_freq=0,
                max_freq=150,
                n_freqs=10000,
                applied_field=3,
                region_label="entire_dot",
                rf_field_strength=5e-3,
                n_nuclei=10000,
                saving=True,
            )

    if applying_pulse:
        # in MHz
        # min_freq = 0
        # max_freq = 100
        # n_freqs = 1000

        pulse_location_range = [80.2]
        pulse_width_range = [0.01] + list(range(1, 12))

        # field_geometry = "Voigt"
        # applied_field = 3
        # n_nuclei = 1

        for pulse_location in pulse_location_range:
            for pulse_width in pulse_width_range:
                overlaid_pulse_graphs(
                    pulse_location,
                    pulse_width,
                    field_geometry,
                    min_freq,
                    max_freq,
                    n_freqs,
                    applied_field,
                    n_nuclei,
                    saving=True,
                )
                if pulse_area_calcs:
                    pulse_graph_area_calculations(
                        field_geometry,
                        pulse_location,
                        pulse_width,
                        min_freq,
                        max_freq,
                        n_freqs,
                        applied_field,
                        region_label="entire_dot",
                        rf_field_strength=5e-3,
                        n_nuclei=n_nuclei,
                    )

    if grid_apply_pulses:
        # in MHz
        # min_freq = 0
        # max_freq = 100
        n_freqs = 10000

        # field_geometry = "Voigt"
        # applied_field = 3
        # n_nuclei = 1
        grid_overlaid_pulse_graphs(
            field_geometry,
            min_freq,
            max_freq,
            n_freqs,
            applied_field,
            n_nuclei,
            saving=True,
        )

    if comparative_area_calcs:
        pulse_location = 80.2
        pulse_width_range = np.linspace(0.01, 20, 1000)

        comparison_area_plots(
            field_geometry,
            pulse_location,
            pulse_width_range,
            applied_field=3,
            region_label="entire_dot",
            rf_field_strength=5e-3,
            n_nuclei=n_nuclei,
            saving=True,
        )


end = time.time()

print(f"Time taken: {np.around((end-start)/60, 2)} minutes")
