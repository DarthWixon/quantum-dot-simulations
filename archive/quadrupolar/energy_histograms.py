import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import os.path
import numpy as np
import matplotlib.pyplot as plt

import backbone_quadrupolar_functions as bqf
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import isotope_parameters as ISOP

import scipy.constants as constants
from scipy.optimize import curve_fit
from scipy.stats import chisquare, maxwell, gamma


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def single_histogram_and_fit_guassian(
    nuclear_species,
    region_bounds,
    plotting=False,
    saving=False,
    use_sundfors_GET_vals=False,
):
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
        nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
    )

    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency_list = list(
        K * V_ZZ.flatten() / 1e6
    )  # divide by 1e6 to get in MHz

    hist_vals, bin_edges, patches = plt.hist(
        quadrupole_frequency_list,
        "auto",
        density=True,
        histtype="step",
        label=species["short_name"],
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    inital_fit_params = [1, 0, 0.2]  # I've guessed at these
    coeffs, var_matrix = curve_fit(gauss, bin_centres, hist_vals, p0=inital_fit_params)
    hist_fit = gauss(bin_centres, *coeffs)

    mean, std_dev = np.around([coeffs[1], np.abs(coeffs[2])], 2)

    ss_res = np.sum((hist_vals - hist_fit) ** 2)
    ss_tot = np.sum((hist_vals - np.mean(hist_vals)) ** 2)
    r2 = np.around(1 - (ss_res / ss_tot), 2)
    # print(f"R**2 value is: {r2}.")

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    if plotting:
        plt.plot(bin_centres, hist_fit)
        plt.title(
            f"Nuclear Species: {nuclear_species}, Region: {region_bounds}, Fit Parameters: $\mu$ = {mean}, $\sigma$ = {std_dev}"
        )

        if saving:
            plot_name = f"{graph_path}guassian_fitted_histogram_for_{nuclear_species}_in_region_{region_bounds}_using_{GET_data_source}_data.png"
            plt.savefig(plot_name)
        else:
            plt.show()

    plt.close()
    return mean, std_dev, r2


def single_histogram_and_fit_boltzmann(
    nuclear_species,
    region_bounds,
    plotting=False,
    saving=False,
    use_sundfors_GET_vals=False,
):
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
        nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
    )

    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency_list = list(
        K * V_ZZ.flatten() / 1e6
    )  # divide by 1e6 to get in MHz

    fig, ax = plt.subplots()
    hist_vals, bin_edges, patches = ax.hist(
        np.abs(quadrupole_frequency_list),
        "auto",
        density=True,
        histtype="step",
        label=species["short_name"],
        color=bqf.will_dark_blue,
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    # print(bin_centres)

    fit_params = maxwell.fit(np.abs(quadrupole_frequency_list))
    mean, var = maxwell.stats(*fit_params)
    # print(mean)
    # print(var)
    # print(fit_params)

    label_text = f"Fit Parameters: \nOrigin = {np.around(fit_params[0],2)} \nScale = {np.around(fit_params[1],2)} \nMean = {np.around(mean,2)} \nVar = {np.around(var,2)}"

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    if plotting:
        x = bin_centres
        ax.plot(
            x,
            maxwell.pdf(x, *fit_params),
            linestyle="dashed",
            color=bqf.will_light_blue,
        )
        ax.set_xlabel("Magnitude of Transition Frequency (MHz)")
        ax.set_ylabel("Probability Density")
        ax.text(0.7, 0.7, label_text, transform=ax.transAxes)

        if saving:
            plot_name = f"{graph_path}boltzmann_fitted_histogram_for_{nuclear_species}_in_region_{region_bounds}_using_{GET_data_source}_data.png"
            plt.savefig(plot_name)
        else:
            plt.show()

    plt.close()

    # return fit_params


def single_histogram_gamma_fit(
    nuclear_species, region_bounds, use_sundfors_GET_vals, saving=False
):
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
        nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
    )

    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency_list = K * V_ZZ.flatten() / 1e6  # divide by 1e6 to get in MHz

    quadrupole_frequency_list = list(
        quadrupole_frequency_list
    )  # has to be a list type for a reason I don't remember right now (24/08/21)

    fig, ax = plt.subplots()
    hist_vals, bin_edges, patches = ax.hist(
        np.abs(quadrupole_frequency_list),
        "auto",
        density=True,
        histtype="step",
        label=species["short_name"],
        color=bqf.will_dark_blue,
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    fit_params = gamma.fit(np.abs(quadrupole_frequency_list), floc=0)
    mean, var = gamma.stats(*fit_params)

    label_text = f"Fit Parameters: \nShape = {np.around(fit_params[0],2)} \nScale = {np.around(fit_params[2],2)} \nMean = {np.around(mean,2)} \nVar = {np.around(var,2)}"

    x = bin_centres
    ax.plot(x, gamma.pdf(x, *fit_params), linestyle="dashed", color=bqf.will_light_blue)
    ax.set_xlabel("Magnitude of Transition Frequency (MHz)")
    ax.set_ylabel("Probability Density")
    ax.text(0.7, 0.7, label_text, transform=ax.transAxes)

    if saving:
        plot_name = f"{graph_path}gamma_fitted_histogram_for_{nuclear_species}_in_region_{region_bounds}_while_use_sundfors_GET_values_is_{use_sundfors_GET_vals}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def all_species_gamma_fit(region_bounds, use_sundfors_GET_vals=False, saving=False):
    h = constants.h
    e = constants.e

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    ax_list = axs.flatten()

    for i, ax in enumerate(ax_list):
        nuclear_species = bqf.nuclear_species_list[i]

        species = ISOP.species_dict[nuclear_species]
        spin = species["particle_spin"]
        Q = species["quadrupole_moment"]
        eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
            nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
        )
        K = (3 * e * Q) / (
            2 * h * spin * (2 * spin - 1)
        )  # constant to convert Vzz to fq
        quadrupole_frequency_list = (
            K * V_ZZ.flatten() / 1e6
        )  # divide by 1e6 to get in MHz

        quadrupole_frequency_list = list(quadrupole_frequency_list)

        hist_vals, bin_edges, patches = ax.hist(
            np.abs(quadrupole_frequency_list),
            "auto",
            density=True,
            histtype="step",
            label=species["short_name"],
            color=bqf.will_dark_blue,
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        fit_params = gamma.fit(np.abs(quadrupole_frequency_list), floc=0)
        # print(fit_params)
        mean, var = gamma.stats(*fit_params)

        label_text = f"Fit Parameters: \nShape = {np.around(fit_params[0],2)} \nScale = {np.around(fit_params[2],2)} \nMean = {np.around(mean,2)} \nVar = {np.around(var,2)}"

        x = bin_centres
        ax.plot(
            x, gamma.pdf(x, *fit_params), linestyle="dashed", color=bqf.will_light_blue
        )
        ax.set_xlabel("Magnitude of Transition Frequency (MHz)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{nuclear_species}")
        ax.text(0.7, 0.7, label_text, transform=ax.transAxes)

    if saving:
        plot_name = f"{graph_path}gamma_fitted_histogram_for_all_species_in_region_{region_bounds}_while_use_sundfors_GET_values_is_{use_sundfors_GET_vals}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def generate_hist_data(nuclear_species, region_bounds, use_sundfors_GET_vals):
    if not use_sundfors_GET_vals:
        archive_name = f"{data_path}hist_data_and_fit_params_for_{nuclear_species}_in_{region_bounds}_using_chekhovich_GET.npz"
    else:
        archive_name = f"{data_path}hist_data_and_fit_params_for_{nuclear_species}_in_{region_bounds}_using_sundfors_GET.npz"

    if os.path.isfile(archive_name):
        print(
            f"Data file '{archive_name}' alredy exists. Skipping further calculation."
        )
        return

    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
        nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
    )

    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency_list = list(
        K * V_ZZ.flatten() / 1e6
    )  # divide by 1e6 to get in MHz

    hist_vals, bin_edges, patches = plt.hist(
        quadrupole_frequency_list,
        "auto",
        density=True,
        histtype="step",
        label=species["short_name"],
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    inital_fit_params = [1, 0, 0.2]  # I've guessed at these
    coeffs, var_matrix = curve_fit(gauss, bin_centres, hist_vals, p0=inital_fit_params)
    hist_fit = gauss(bin_centres, *coeffs)

    mean, std_dev = np.around([coeffs[1], np.abs(coeffs[2])], 2)

    ss_res = np.sum((hist_vals - hist_fit) ** 2)
    ss_tot = np.sum((hist_vals - np.mean(hist_vals)) ** 2)
    r2 = np.around(1 - (ss_res / ss_tot), 2)

    np.savez(
        archive_name,
        measured_prob_density=hist_vals,
        frequencies=bin_centres,
        fit_prob_density=hist_fit,
        mean=mean,
        std_dev=std_dev,
        r2=r2,
    )
    print(f"Saved archive: {archive_name}")


def load_hist_data(nuclear_species, region_bounds, use_sundfors_GET_vals):
    if not use_sundfors_GET_vals:
        archive_name = f"{data_path}hist_data_and_fit_params_for_{nuclear_species}_in_{region_bounds}_using_chekhovich_GET.npz"
    else:
        archive_name = f"{data_path}hist_data_and_fit_params_for_{nuclear_species}_in_{region_bounds}_using_sundfors_GET.npz"

    full_archive = np.load(archive_name)

    measured_prob_density = full_archive["measured_prob_density"]
    frequencies = full_archive["frequencies"]
    fit_prob_density = full_archive["fit_prob_density"]
    mean = full_archive["mean"]
    std_dev = full_archive["std_dev"]
    r2 = full_archive["r2"]

    return measured_prob_density, frequencies, fit_prob_density, mean, std_dev, r2


def find_all_fits(region_bounds, use_sundfors_GET_vals=False):
    if not use_sundfors_GET_vals:
        print(f"Using Checkovich 2018/19 GET Data in Region {region_bounds}")
    else:
        print(f"Using Sundfors 1974 GET Data in Region {region_bounds}")
    for nuclear_species in bqf.nuclear_species_list:
        mean, std_dev, r2 = single_histogram_and_fit_guassian(
            nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
        )
        print(
            f"{nuclear_species} has: mean = {mean} MHz & std_dev = {std_dev} MHz, with R2 = {r2}"
        )

    print("---------------------------------")


def quadrupolar_strength_distributions(
    region_bounds=[100, 1200, 200, 1000], saving=False, use_sundfors_GET_vals=False
):
    """
    Calculates and plots a histogram of quadrupolar frequencies.

    Args:
        nuclear_species (str): The atomic species to find the EFG for.
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]
        saving (bool): Whether or not to save the figure. If not saving, the figure is
            displayed using matplotlib.

    Returns:
        No return value. If the saving option is given, will save a file to disk.

    """

    quadrupole_frequency_arrays = []

    for nuclear_species in bqf.nuclear_species_list:
        species = ISOP.species_dict[nuclear_species]
        spin = species["particle_spin"]
        Q = species["quadrupole_moment"]

        h = constants.h
        e = constants.e

        eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
            nuclear_species, region_bounds, use_sundfors_GET_vals=use_sundfors_GET_vals
        )

        K = (3 * e * Q) / (
            2 * h * spin * (2 * spin - 1)
        )  # constant to convert Vzz to fq
        quadrupole_frequency_arrays.append(
            K * V_ZZ.flatten() / 1e6
        )  # divide by 1e6 to get in MHz

    hist_vals, bin_edges, patches = plt.hist(
        quadrupole_frequency_arrays,
        "auto",
        density=True,
        stacked=False,
        histtype="step",
        label=bqf.nuclear_species_list,
    )

    plt.legend()
    plt.xlabel("Frequency (MHz)")
    # plt.ylabel("Probability")
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    # xticks = ax.axes.get_xticklabels()
    # xticks = np.array(xticks)/1e6
    # ax.set_xticks(xticks)

    # plt.tight_layout()

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    plot_name = f"{graph_path}quadrupolar_frequency_distributions_in_region_{region_bounds}_using_{GET_data_source}_data.png"
    plt.title(f"Quadrupolar Strength Distributions for {GET_data_name} Data.")

    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()

    return


def layered_QI_distributions(
    region_bounds_list, saving=False, use_sundfors_GET_vals=False
):
    # this function is currently hard-coded to work with exactly 4 regions (23/20/10)
    from matplotlib.lines import Line2D

    colour_list = ["blue", "orange", "green", "red"]

    linestyle_list = ["solid", "dashed", "dotted", "dashdot"]

    fig = plt.figure(figsize=(12, 8))

    for r, region_bounds in enumerate(region_bounds_list):
        quadrupole_frequency_arrays = []

        for nuclear_species in bqf.nuclear_species_list:
            species = ISOP.species_dict[nuclear_species]
            spin = species["particle_spin"]
            Q = species["quadrupole_moment"]

            h = constants.h
            e = constants.e

            eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
                nuclear_species,
                region_bounds,
                use_sundfors_GET_vals=use_sundfors_GET_vals,
            )

            K = (3 * e * Q) / (
                2 * h * spin * (2 * spin - 1)
            )  # constant to convert Vzz to fq
            quadrupole_frequency_arrays.append(
                K * V_ZZ.flatten() / 1e6
            )  # divide by 1e6 to get in MHz

        hist_vals, bin_edges, patches = plt.hist(
            quadrupole_frequency_arrays,
            "auto",
            density=True,
            histtype="step",
            label=bqf.nuclear_species_list,
            color=colour_list,
            ls=linestyle_list[r],
            linewidth=2,
        )

    # use these fake lines to make a legend that is nice
    custom_lines = [
        Line2D([0], [0], color=colour_list[0], label=bqf.nuclear_species_list[0]),
        Line2D([0], [0], color=colour_list[1], label=bqf.nuclear_species_list[1]),
        Line2D([0], [0], color=colour_list[2], label=bqf.nuclear_species_list[2]),
        Line2D([0], [0], color=colour_list[3], label=bqf.nuclear_species_list[3]),
    ]  # ,
    # Line2D([0], [0], color = "black", ls = linestyle_list[0], label = "Center"),
    # Line2D([0], [0], color = "black", ls = linestyle_list[1], label = "Left"),
    # Line2D([0], [0], color = "black", ls = linestyle_list[2], label = "Right"),
    # Line2D([0], [0], color = "black", ls = linestyle_list[3], label = "Bottom")]

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Probability Density")
    plt.legend(handles=custom_lines)
    # ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    plt.tight_layout()

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    plot_name = f"{graph_path}multiple_quadrupolar_frequency_distributions_with_{GET_data_source}_data.png"
    plt.title(f"Distribution of Quadrupolar Frequencies Using {GET_data_name} GET Data")

    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()


def new_vs_old_comparison_hists(region_bounds_list, saving=False):
    from matplotlib.lines import Line2D

    colour_list = [bqf.will_dark_blue, "darkred", bqf.will_light_blue, "salmon"]

    linestyle_list = ["solid", "solid", "dashed", "dashed"]

    fig = plt.figure(figsize=(12, 8))

    changed_species_list = ["Ga69", "As75", "Ga69", "As75"]
    label_list = ["Ga69 2018/19", "As75 2018/19", "Ga69 1974", "As75 1974"]

    for r, region_bounds in enumerate(region_bounds_list):
        quadrupole_frequency_arrays = []

        # the species count number is my hacky way of getting the new and old values to behave nicely
        # we increase it through the loop, then reset per region
        # when we're halfway through, we swap from new to old GET values
        species_count = 0
        for nuclear_species in changed_species_list:
            species = ISOP.species_dict[nuclear_species]
            spin = species["particle_spin"]
            Q = species["quadrupole_moment"]

            h = constants.h
            e = constants.e

            if species_count <= 1:
                eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
                    nuclear_species, region_bounds, use_sundfors_GET_vals=False
                )
            else:
                eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
                    nuclear_species, region_bounds, use_sundfors_GET_vals=True
                )

            K = (3 * e * Q) / (
                2 * h * spin * (2 * spin - 1)
            )  # constant to convert Vzz to fq
            quadrupole_frequency_arrays.append(
                K * V_ZZ.flatten() / 1e6
            )  # divide by 1e6 to get in MHz

            species_count += 1

        hist_vals, bin_edges, patches = plt.hist(
            quadrupole_frequency_arrays,
            "auto",
            density=True,
            histtype="step",
            label=label_list,
            color=colour_list,
            ls=linestyle_list[r],
            linewidth=2,
        )

    custom_lines = [
        Line2D([0], [0], color=colour_list[0], label=label_list[0]),
        Line2D([0], [0], color=colour_list[1], label=label_list[1]),
        Line2D([0], [0], color=colour_list[2], label=label_list[2]),
        Line2D([0], [0], color=colour_list[3], label=label_list[3]),
    ]
    # Line2D([0], [0], color = "black", ls = linestyle_list[0], label = "Center"),
    # Line2D([0], [0], color = "black", ls = linestyle_list[1], label = "Left"),
    # Line2D([0], [0], color = "black", ls = linestyle_list[2], label = "Right"),
    # Line2D([0], [0], color = "black", ls = linestyle_list[3], label = "Bottom")]

    plt.xlabel("Transition Frequency (MHz)")
    plt.ylabel("Probability Density")
    plt.legend(handles=custom_lines)
    plt.tight_layout()

    plot_name = f"{graph_path}splitting_distribution_comparison.png"
    # plt.title(f"Comparison of Distributions with Changing GET Values")

    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# set font size for poster purposes
# plt.rcParams.update({'font.size': 20})


# set region corners, entire dot region is:[100, 1200, 200, 1000], in the order left, right, top, bottom (06/05/20)
# dot only region seems to be: [100,1200, 439, 880] (06/05/20)
# small testing region is: [750, 800, 400, 500]

# central region guess from 19/20/20: [450, 850, 550, 650]
# region_bounds = [100,1200, 439, 880]

# nuclear_species = "Ga71"

# single_histogram_and_fit_boltzmann(nuclear_species, region_bounds, plotting = True, saving = True, use_sundfors_GET_vals = False)
# single_histogram_and_fit_boltzmann(nuclear_species, region_bounds, plotting = True, saving = True, use_sundfors_GET_vals = True)


# step_size = 1
# step_size_list = list(range(0,100,5)) # make a list cos python3 returns a range object which doesn't support assignment
# step_size_list[0] = 1 # do this so the first entry is 1, rather than 0

# standard list of nuclear species

region_bounds_list = []
# region_bounds_list.append([450, 850, 550, 650])
# region_bounds_list.append([20, 420, 650, 750])
# region_bounds_list.append([1010, 1410, 650, 750])
# region_bounds_list.append([450, 850, 660, 760])
# region_bounds_list.append([100, 1200, 200, 1000])
# region_bounds_list.append([100,1200, 439, 880])
# region_bounds_list.append([600, 625, 585, 610])
# region_bounds_list.append([450, 850, 775, 875])
region_bounds_list.append([100, 1200, 439, 880])


# print(region_bounds_list)

# for region in region_bounds_list:
#     print(f"Creaing Histogram for {region}")
#     quadrupolar_strength_distributions(region_bounds = region, saving = True, use_sundfors_GET_vals = False)
#     quadrupolar_strength_distributions(region_bounds = region, saving = True, use_sundfors_GET_vals = True)

# layered_QI_distributions(region_bounds_list, saving = False, use_sundfors_GET_vals = False)
# layered_QI_distributions(region_bounds_list, saving = False, use_sundfors_GET_vals = True)

# test_region = region_bounds_list[0]

# quadrupolar_strength_distributions(region_bounds, saving = False, use_sundfors_GET_vals = True)

new_vs_old_comparison_hists(region_bounds_list, saving=True)

# single_histogram_and_fit_guassian("Ga69", region_bounds)

# print("---------------------------------")
# for region_bounds in region_bounds_list:
#     find_all_fits(region_bounds, use_sundfors_GET_vals = False)
#     find_all_fits(region_bounds, use_sundfors_GET_vals = True)

# for nuclear_species in bqf.nuclear_species_list:
#     for region_bounds in region_bounds_list:
#         generate_hist_data(nuclear_species, region_bounds, use_sundfors_GET_vals = False)
#         generate_hist_data(nuclear_species, region_bounds, use_sundfors_GET_vals = True)


# region_bounds = [100,1200, 439, 880]

# all_species_gamma_fit(region_bounds, use_sundfors_GET_vals = False, saving = True)
# all_species_gamma_fit(region_bounds, use_sundfors_GET_vals = True, saving = True)

# for nuclear_species in bqf.nuclear_species_list:
#     single_histogram_gamma_fit(nuclear_species, region_bounds, use_sundfors_GET_vals = False, saving = True)
#     single_histogram_gamma_fit(nuclear_species, region_bounds, use_sundfors_GET_vals = True, saving = True)
