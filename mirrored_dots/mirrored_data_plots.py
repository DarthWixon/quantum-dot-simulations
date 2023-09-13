import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import os.path
import time

from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import isotope_parameters as ISOP
import create_mirrored_data

import numpy as np

import multiprocessing
from itertools import permutations

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.constants as constants
from scipy.optimize import curve_fit
from scipy.stats import chisquare, maxwell, gamma


def load_mirrored_In_concentration_data(region_bounds, faked_strain_type="left_right"):
    """
    Loads the interpolated concentration data from Sokolov.

    Paper these data are from (DOI): 10.1103/PhysRevB.93.045301

    Args:
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Entire dot region is:[100, 1200, 200, 1000].
            Dot only region is: [100,1200, 439, 880].
            Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 1 (every site is used).
        method (str): The interpolation method used to get the data to be the
            same size as the strain data. Default is cubic (using a cubic spline).

    Returns:
        dot_conc_data (ndarray): The percentage concentration of In115 in the QD.
    """

    filename = f"{data_path}{faked_strain_type}_mirrored_In_conc_data_in_region_{region_bounds}.npy"
    dot_conc_data = np.load(filename)

    return dot_conc_data


def plot_concentration_image(region_bounds, faked_strain_type, saving=False):
    conc_data = load_mirrored_In_concentration_data(region_bounds, faked_strain_type)

    plt.imshow(conc_data, cmap=cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())
    plt.colorbar(orientation="horizontal", label="Indium Concentration", shrink=0.5)
    plt.axis("off")
    # plt.title("Indium Concentration")
    # plt.tight_layout()

    if saving:
        filename = f"{graph_path}In_concentration_map_{faked_strain_type}_flipped.png"
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved graph: {filename}")
    else:
        plt.show()

    plt.close()


def plot_strain_image(region_bounds, faked_strain_type, saving=False):
    xx_array, xz_array, zz_array = bqf.load_mirrored_data(
        region_bounds, faked_strain_type=faked_strain_type
    )
    fig, axs = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(6, 9))

    xx_min = np.min(xx_array)
    xx_max = np.max(xx_array)
    zz_min = np.min(zz_array)
    zz_max = np.max(zz_array)
    xz_min = np.min(xz_array)
    xz_max = np.max(xz_array)

    cbar_min = np.min([xx_min, zz_min, xz_min])
    cbar_max = np.max([xx_max, zz_max, xz_max])

    cmap = cm.get_cmap("RdBu")
    normalizer = Normalize(cbar_min, cbar_max)
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    axs[0].imshow(xx_array, cmap=cmap, norm=normalizer)
    axs[0].axis("off")
    axs[0].text(20, 150, "$\epsilon_{xx}$", fontsize=16, fontdict=None)

    axs[1].imshow(zz_array, cmap=cmap, norm=normalizer)
    axs[1].axis("off")
    axs[1].text(20, 150, "$\epsilon_{zz}$", fontsize=16, fontdict=None)

    axs[2].imshow(xz_array, cmap=cmap, norm=normalizer)
    axs[2].axis("off")
    axs[2].text(20, 150, "$\epsilon_{xz}$", fontsize=16, fontdict=None)

    cbar = plt.colorbar(
        im, ax=axs.ravel().tolist(), shrink=0.95, orientation="vertical"
    )

    if saving:
        plot_name = f"{graph_path}Sokolov_strain_graph_vertical_for_{faked_strain_type}_mirrored_dot_in_region_{region_bounds}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def get_mirrored_EFG_data(nuclear_species, region_bounds, faked_strain_type):
    bqf.calculate_and_save_EFG(
        nuclear_species,
        region_bounds,
        use_sundfors_GET_vals=False,
        real_strain_data=False,
        faked_strain_type=faked_strain_type,
    )
    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(
        nuclear_species,
        region_bounds,
        use_sundfors_GET_vals=False,
        real_strain_data=False,
        faked_strain_type=faked_strain_type,
    )

    return eta, V_XX, V_YY, V_ZZ, euler_angles


def plot_arrows_with_strength_background(
    nuclear_species, region_bounds, faked_strain_type, saving=False
):
    eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(
        nuclear_species, region_bounds, faked_strain_type
    )

    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e
    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency = np.abs(K * V_ZZ) / 1e6  # convert to MHz

    spacing = 50
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

    n, m = quadrupole_frequency.shape

    X, Y = np.meshgrid(np.arange(0, m, 1), np.arange(0, n, 1))

    r = (quadrupole_frequency**2 + quadrupole_frequency**2) ** 0.5
    arrow_lengths = quadrupole_frequency / r

    Z_angles = (
        euler_angles[:, :, 2] * 180 / np.pi
    )  # convert angles from radians to degrees for the quiver function

    # plot the arrows showing QI size and direction
    ax.quiver(
        X[::spacing, ::spacing],
        Y[::spacing, ::spacing],
        arrow_lengths[::spacing, ::spacing],
        arrow_lengths[::spacing, ::spacing],
        angles=Z_angles[::spacing, ::spacing],
        minshaft=5,
        pivot="tail",
        color="black",
    )

    im = ax.imshow(
        quadrupole_frequency,
        cmap=cm.GnBu,
        vmin=quadrupole_frequency.min(),
        vmax=quadrupole_frequency.max(),
    )
    plt.axis("off")

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(im, location="bottom", shrink=0.5)
    cbar.set_label("Quadrupole Frequency (MHz)")

    # plt.tight_layout()

    if saving:
        filename = f"{graph_path}QI_arrows_in_region_{region_bounds}_for_{nuclear_species}_with_QI_strength_background_in_{faked_strain_type}_mirrored_dot.png"
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved graph: {filename}")
    else:
        plt.show()

    plt.close()


def plot_biaxiality(nuclear_species, region_bounds, faked_strain_type, saving=False):
    species = ISOP.species_dict[nuclear_species]
    species_name = species["short_name"]

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(
        nuclear_species, region_bounds, faked_strain_type
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(eta, cmap=cm.GnBu, vmin=0, vmax=1)
    plt.axis("off")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(im, location="bottom", shrink=0.5)
    cbar.set_label(r"$\eta$")

    plot_name = f"{graph_path}biaxiality_for_{species_name}_in_region_{region_bounds}_in_{faked_strain_type}_mirrored_dot.png"

    if saving:
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    return


def quadrupolar_strength_distributions(
    region_bounds, faked_strain_type, saving=False, positive_only=False
):
    quadrupole_frequency_arrays = []
    pos_text = ""

    for nuclear_species in bqf.nuclear_species_list:
        species = ISOP.species_dict[nuclear_species]
        spin = species["particle_spin"]
        Q = species["quadrupole_moment"]

        h = constants.h
        e = constants.e

        eta, V_XX, V_YY, V_ZZ, euler_angles = (
            eta,
            V_XX,
            V_YY,
            V_ZZ,
            euler_angles,
        ) = get_mirrored_EFG_data(nuclear_species, region_bounds, faked_strain_type)

        K = (3 * e * Q) / (
            2 * h * spin * (2 * spin - 1)
        )  # constant to convert Vzz to fq
        quadrupole_frequency_list = K * V_ZZ.flatten() / 1e6

        if positive_only:
            if nuclear_species == "Ga69":
                # Ga 69 points the other direction inside the dot, and therefore we must adjust for that as well
                quadrupole_frequency_list = quadrupole_frequency_list[
                    quadrupole_frequency_list <= 0
                ]
            else:
                quadrupole_frequency_list = quadrupole_frequency_list[
                    quadrupole_frequency_list >= 0
                ]
            pos_text = "_using_positive_frequencies_only"

        quadrupole_frequency_list = list(quadrupole_frequency_list)

        quadrupole_frequency_arrays.append(
            quadrupole_frequency_list
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

    plot_name = f"{graph_path}quadrupolar_frequency_distributions_in_region_{region_bounds}_for_{faked_strain_type}_mirrored_dot{pos_text}.png"

    if saving:
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    return


def single_histogram_boltzmann_fit(
    nuclear_species, region_bounds, faked_strain_type, saving=False, positive_only=False
):
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    pos_text = ""

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(
        nuclear_species, region_bounds, faked_strain_type
    )

    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency_list = K * V_ZZ.flatten() / 1e6  # divide by 1e6 to get in MHz

    if positive_only:
        if nuclear_species == "Ga69":
            # Ga 69 points the other direction inside the dot, and therefore we must adjust for that as well
            quadrupole_frequency_list = quadrupole_frequency_list[
                quadrupole_frequency_list <= 0
            ]
        else:
            quadrupole_frequency_list = quadrupole_frequency_list[
                quadrupole_frequency_list >= 0
            ]
        pos_text = "_using_positive_frequencies_only"

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
    # print(bin_centres)

    fit_params = maxwell.fit(np.abs(quadrupole_frequency_list))
    mean, var = maxwell.stats(*fit_params)
    # print(mean)
    # print(var)
    # print(fit_params)

    label_text = f"Fit Parameters: \nOrigin = {np.around(fit_params[0],2)} \nScale = {np.around(fit_params[1],2)} \nMean = {np.around(mean,2)} \nVar = {np.around(var,2)}"

    x = bin_centres
    ax.plot(
        x, maxwell.pdf(x, *fit_params), linestyle="dashed", color=bqf.will_light_blue
    )
    ax.set_xlabel("Magnitude of Transition Frequency (MHz)")
    ax.set_ylabel("Probability Density")
    ax.text(0.7, 0.7, label_text, transform=ax.transAxes)

    if saving:
        plot_name = f"{graph_path}boltzmann_fitted_histogram_for_{nuclear_species}_in_region_{region_bounds}_for_{faked_strain_type}_mirrored_dot{pos_text}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def all_species_boltzmann_fit(
    region_bounds, faked_strain_type, saving=False, positive_only=False
):
    h = constants.h
    e = constants.e

    pos_text = ""

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    ax_list = axs.flatten()

    for i, ax in enumerate(ax_list):
        nuclear_species = bqf.nuclear_species_list[i]

        species = ISOP.species_dict[nuclear_species]
        spin = species["particle_spin"]
        Q = species["quadrupole_moment"]
        eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(
            nuclear_species, region_bounds, faked_strain_type
        )
        K = (3 * e * Q) / (
            2 * h * spin * (2 * spin - 1)
        )  # constant to convert Vzz to fq
        quadrupole_frequency_list = (
            K * V_ZZ.flatten() / 1e6
        )  # divide by 1e6 to get in MHz

        if positive_only:
            if nuclear_species == "Ga69":
                # Ga 69 points the other direction inside the dot, and therefore we must adjust for that as well
                quadrupole_frequency_list = quadrupole_frequency_list[
                    quadrupole_frequency_list <= 0
                ]
            else:
                quadrupole_frequency_list = quadrupole_frequency_list[
                    quadrupole_frequency_list >= 0
                ]
            pos_text = "_using_positive_frequencies_only"

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

        fit_params = maxwell.fit(np.abs(quadrupole_frequency_list))
        mean, var = maxwell.stats(*fit_params)

        label_text = f"Fit Parameters: \nOrigin = {np.around(fit_params[0],2)} \nScale = {np.around(fit_params[1],2)} \nMean = {np.around(mean,2)} \nVar = {np.around(var,2)}"

        x = bin_centres
        ax.plot(
            x,
            maxwell.pdf(x, *fit_params),
            linestyle="dashed",
            color=bqf.will_light_blue,
        )

        ax.set_xlabel("Magnitude of Transition Frequency (MHz)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{nuclear_species}")
        ax.text(0.7, 0.7, label_text, transform=ax.transAxes)

    if saving:
        plot_name = f"{graph_path}boltzmann_fitted_histogram_for_all_species_in_region_{region_bounds}_for_{faked_strain_type}_mirrored_dot{pos_text}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def single_histogram_gamma_fit(
    nuclear_species, region_bounds, faked_strain_type, saving=False, positive_only=False
):
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    pos_text = ""

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(
        nuclear_species, region_bounds, faked_strain_type
    )

    K = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))  # constant to convert Vzz to fq
    quadrupole_frequency_list = K * V_ZZ.flatten() / 1e6  # divide by 1e6 to get in MHz

    if positive_only:
        if nuclear_species == "Ga69":
            # Ga 69 points the other direction inside the dot, and therefore we must adjust for that as well
            quadrupole_frequency_list = quadrupole_frequency_list[
                quadrupole_frequency_list <= 0
            ]
        else:
            quadrupole_frequency_list = quadrupole_frequency_list[
                quadrupole_frequency_list >= 0
            ]
        pos_text = "_using_positive_frequencies_only"

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

    fit_params = gamma.fit(np.abs(quadrupole_frequency_list))
    mean, var = gamma.stats(*fit_params)

    label_text = f"Fit Parameters: \nShape = {np.around(fit_params[0],2)} \nScale = {np.around(fit_params[2],2)} \nMean = {np.around(mean,2)} \nVar = {np.around(var,2)}"

    x = bin_centres
    ax.plot(x, gamma.pdf(x, *fit_params), linestyle="dashed", color=bqf.will_light_blue)
    ax.set_xlabel("Magnitude of Transition Frequency (MHz)")
    ax.set_ylabel("Probability Density")
    ax.text(0.7, 0.7, label_text, transform=ax.transAxes)

    if saving:
        plot_name = f"{graph_path}gamma_fitted_histogram_for_{nuclear_species}_in_region_{region_bounds}_for_{faked_strain_type}_mirrored_dot{pos_text}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def all_species_gamma_fit(
    region_bounds, faked_strain_type, saving=False, positive_only=False
):
    h = constants.h
    e = constants.e

    pos_text = ""

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    ax_list = axs.flatten()

    for i, ax in enumerate(ax_list):
        nuclear_species = bqf.nuclear_species_list[i]

        species = ISOP.species_dict[nuclear_species]
        spin = species["particle_spin"]
        Q = species["quadrupole_moment"]
        eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(
            nuclear_species, region_bounds, faked_strain_type
        )
        K = (3 * e * Q) / (
            2 * h * spin * (2 * spin - 1)
        )  # constant to convert Vzz to fq
        quadrupole_frequency_list = (
            K * V_ZZ.flatten() / 1e6
        )  # divide by 1e6 to get in MHz

        if positive_only:
            if nuclear_species == "Ga69":
                # Ga 69 points the other direction inside the dot, and therefore we must adjust for that as well
                quadrupole_frequency_list = quadrupole_frequency_list[
                    quadrupole_frequency_list <= 0
                ]
            else:
                quadrupole_frequency_list = quadrupole_frequency_list[
                    quadrupole_frequency_list >= 0
                ]
            pos_text = "_using_positive_frequencies_only"

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
        plot_name = f"{graph_path}gamma_fitted_histogram_for_all_species_in_region_{region_bounds}_for_{faked_strain_type}_mirrored_dot{pos_text}.png"
        plt.savefig(plot_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def random_sites_energy_level_diagram(
    nuclear_species,
    region_bounds,
    field_geometry,
    faked_strain_type,
    n_sites,
    min_field=0,
    max_field=2,
    n_fields=100,
    saving=False,
):
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = get_mirrored_EFG_data(nuclear_species, region_bounds, faked_strain_type)

    x_dim, y_dim = eta_array.shape
    rng = np.random.default_rng()
    x_points = rng.choice(x_dim, n_sites, replace=False)
    y_points = rng.choice(y_dim, n_sites, replace=False)
    points = list(zip(x_points, y_points))

    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    quadrupole_coupling_constant = (3 * e * Q) / (
        2 * h * spin * (2 * spin - 1)
    )  # constant to convert Vzz to fq

    n_levels = int((2 * ISOP.species_dict[nuclear_species]["particle_spin"]) + 1)

    data = np.zeros((n_sites, n_levels, len(applied_field_list)))

    for n in range(n_sites):
        x_coord, y_coord = points[n]
        for b, applied_field in enumerate(applied_field_list):
            eta = eta_array[x_coord, y_coord]
            V_ZZ = V_ZZ_array[x_coord, y_coord]
            alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

            quadrupolar_term = quadrupole_coupling_constant * V_ZZ
            Zeeman_term = zeeman_frequency_per_tesla * applied_field

            if field_geometry == "Faraday":
                Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(
                    Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
                )
            elif field_geometry == "Voigt":
                Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(
                    Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
                )

            eigen_energies = Hamiltonian.eigenenergies() / 1e6  # plot them in MHz
            data[n, :, b] = np.real_if_close(eigen_energies)

    plt.figure(figsize=(12, 10))

    for n in range(n_sites):
        for i in range(n_levels):
            plt.plot(applied_field_list, data[n, i, :], color="black", linewidth=0.5)

    plt.xlabel("Applied B Field")
    plt.ylabel("Splitting Energy (MHz)")

    if saving:
        plot_title = "random_sites_energy_level_diagram_of_{}_with_changing_b_field_in_{}_orientation_over_region_{}_in_{}_mirrored_dot.png".format(
            nuclear_species, field_geometry, region_bounds, faked_strain_type
        )
        plt.savefig(f"{graph_path}{plot_title}")
    else:
        plt.show()
    plt.close()


def random_sites_geometry_comparison_EL_diagram(
    nuclear_species,
    region_bounds,
    faked_strain_type,
    n_sites,
    min_field=0,
    max_field=2,
    n_fields=100,
    saving=False,
):
    applied_field_list = np.linspace(min_field, max_field, n_fields)

    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = get_mirrored_EFG_data(nuclear_species, region_bounds, faked_strain_type)

    x_dim, y_dim = eta_array.shape
    rng = np.random.default_rng()
    x_points = rng.choice(x_dim, n_sites, replace=False)
    y_points = rng.choice(y_dim, n_sites, replace=False)
    points = list(zip(x_points, y_points))

    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    quadrupole_coupling_constant = (3 * e * Q) / (
        2 * h * spin * (2 * spin - 1)
    )  # constant to convert Vzz to fq

    n_levels = int((2 * spin) + 1)

    full_faraday_data = np.zeros((n_sites, n_levels, len(applied_field_list)))
    full_voigt_data = np.zeros((n_sites, n_levels, len(applied_field_list)))

    for n in range(n_sites):
        x_coord, y_coord = points[n]
        for b, applied_field in enumerate(applied_field_list):
            eta = eta_array[x_coord, y_coord]
            V_ZZ = V_ZZ_array[x_coord, y_coord]
            alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

            quadrupolar_term = quadrupole_coupling_constant * V_ZZ
            Zeeman_term = zeeman_frequency_per_tesla * applied_field

            Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(
                Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
            )
            eigen_energies = Hamiltonian.eigenenergies() / 1e6  # plot them in MHz
            full_faraday_data[n, :, b] = np.real_if_close(eigen_energies)

            Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(
                Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
            )
            eigen_energies = Hamiltonian.eigenenergies() / 1e6  # plot them in MHz
            full_voigt_data[n, :, b] = np.real_if_close(eigen_energies)

    min_faraday = np.amin(full_faraday_data)
    max_faraday = np.amax(full_faraday_data)
    min_voigt = np.amin(full_voigt_data)
    max_voigt = np.amax(full_voigt_data)

    min_range = np.amin([min_faraday, min_voigt])
    max_range = np.amax([max_faraday, max_voigt])

    y_max = 1.2 * max_range
    y_min = 1.2 * min_range

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

    for n in range(n_sites):
        for i in range(n_levels):
            ax1.plot(applied_field_list, full_faraday_data[n, i, :], color="black")
            ax2.plot(applied_field_list, full_voigt_data[n, i, :], color="black")

    ax1.set_xlabel("Applied Field (T)")
    ax2.set_xlabel("Applied Field (T)")

    ax1.set_ylabel("Frequency (MHz)")
    ax2.set_ylabel("Frequency (MHz)")

    ax1.set_title("Faraday Orientation")
    ax2.set_title("Voigt Orientation")

    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    ax1.text(0.1, 0.9, "a", transform=ax1.transAxes)
    ax2.text(0.1, 0.9, "b", transform=ax2.transAxes)

    if saving:
        plot_title = f"{graph_path}random_sample_of_{nuclear_species}_both_orientations_energy_level_structure_in_{faked_strain_type}_mirrored_dot.png"
        plt.savefig(plot_title, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def NMR_calculate_absorbtion_single_slice(
    nuclear_species,
    applied_field,
    field_geometry,
    region_bounds,
    faked_strain_type,
    rf_freq_list,
    location,
    data_only_return=True,
    rf_field=5e-3,
):
    x_coord, y_coord = location

    transition_rate_data = np.zeros(len(rf_freq_list))

    # load the data for this species and region
    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = get_mirrored_EFG_data(nuclear_species, region_bounds, faked_strain_type)

    # find the species specific data
    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]  #

    h = constants.h
    e = constants.e

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

    if field_geometry == "Voigt":
        # static field along the x axis
        # want RF field to be perpendicular to static field, so we set it along z
        Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(
            Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
        )
        rf_Ham = bqf.RF_hamiltonian_creator(spin, 0, 0, rf_field)

    if field_geometry == "Faraday":
        # static field along the z axis
        # want RF field to be perpendicular to static field, so we set it along x
        Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(
            Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma
        )
        rf_Ham = bqf.RF_hamiltonian_creator(spin, rf_field, 0, 0)

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

            transition_rate_data[r] += bqf.transition_rate_calc(
                rf_Ham, init_state, final_state, init_energy, final_energy, rf_freq
            )

    if data_only_return:
        return transition_rate_data


def random_locations_list_generator(n_locs, region_bounds):
    rng = np.random.default_rng()

    (
        eta_array,
        V_XX_array,
        V_YY_array,
        V_ZZ_array,
        euler_angles_array,
    ) = get_mirrored_EFG_data(
        "As75", region_bounds, "left_right"
    )  # dummy values just to size arrays lazily
    ax_0_range, ax_1_range = eta_array.shape
    ax_0_indices = rng.integers(ax_0_range, size=n_locs)
    ax_1_indices = rng.integers(ax_1_range, size=n_locs)

    loc_list = list(zip(ax_0_indices, ax_1_indices))
    return loc_list


def NMR_parameter_zipper(
    nuclear_species,
    applied_field,
    field_geometry,
    rf_freq_list,
    loc_list,
    region_bounds,
    faked_strain_type,
):
    n_items = len(loc_list)
    nuclear_species_list = [nuclear_species] * n_items
    applied_field_list_list = [applied_field] * n_items
    field_geometry_list = [field_geometry] * n_items
    rf_freq_list_list = [rf_freq_list] * n_items
    # don't need to do anything to loc_list as it's the thing that changes
    region_bounds_list = [region_bounds] * n_items
    data_only_return_list = [True] * n_items
    rf_field_list = [5e-3] * n_items

    faked_strain_type_list = [faked_strain_type] * n_items

    zipped_params = list(
        zip(
            nuclear_species_list,
            applied_field_list_list,
            field_geometry_list,
            region_bounds_list,
            faked_strain_type_list,
            rf_freq_list_list,
            loc_list,
            data_only_return_list,
            rf_field_list,
        )
    )

    return zipped_params


def NMR_parrallel_calculation(
    nuclear_species,
    region_bounds,
    faked_strain_type,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    recalc=False,
):
    # creates and saves data for NMR spectra, DOES NOT TAKE LOGS OF IT

    filename = f"{data_path}NMR_absorbtion_data_for_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_bounds}_for_{len(applied_field_list)}_B_fields_and_{len(rf_freq_list)}_rf_fields_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz_in_{faked_strain_type}_mirrored_dot.npy"

    if os.path.isfile(filename) and recalc == False:
        print(f"{filename} already exists, not recalculating.")
        return

    data = np.zeros((len(applied_field_list), len(rf_freq_list)))

    for b, applied_field in enumerate(applied_field_list):
        with multiprocessing.Pool() as pool:
            zipped_params = NMR_parameter_zipper(
                nuclear_species,
                applied_field,
                field_geometry,
                rf_freq_list,
                loc_list,
                region_bounds,
                faked_strain_type,
            )
            slices_across_locations = pool.starmap(
                NMR_calculate_absorbtion_single_slice, zipped_params
            )
            pool.close()

        data[b, :] = np.sum(slices_across_locations, axis=0)

    np.save(filename, data)
    if os.path.isfile(filename):
        print(f"Successfully saved {filename}")


def NMR_load_data(
    nuclear_species,
    region_bounds,
    faked_strain_type,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
):
    filename = f"{data_path}NMR_absorbtion_data_for_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_bounds}_for_{len(applied_field_list)}_B_fields_and_{len(rf_freq_list)}_rf_fields_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz_in_{faked_strain_type}_mirrored_dot.npy"
    spectra_data = np.load(filename)

    return spectra_data


def NMR_plot(
    nuclear_species,
    region_bounds,
    faked_strain_type,
    applied_field_list,
    field_geometry,
    rf_freq_list,
    loc_list,
    saving=False,
    recalc=False,
):
    NMR_parrallel_calculation(
        nuclear_species,
        region_bounds,
        faked_strain_type,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
        recalc=recalc,
    )

    data = NMR_load_data(
        nuclear_species,
        region_bounds,
        faked_strain_type,
        applied_field_list,
        field_geometry,
        rf_freq_list,
        loc_list,
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
        filename = f"{graph_path}NMR_spectra_{nuclear_species}_in_{field_geometry}_{len(loc_list)}_locs_in_{region_bounds}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz_for_{faked_strain_type}_mirrored_dot.png"
        plt.savefig(filename)
        print(f"Saved file: {filename}")
    else:
        plt.show()

    plt.close()


def NMR_experiment_plot(
    field_geometry, region_bounds, faked_strain_type, n_locs, recalc=False, saving=False
):
    conc_data = load_mirrored_In_concentration_data(region_bounds, faked_strain_type)

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

    mean_In_conc = np.mean(conc_data)

    n_In_locs = int(
        np.ceil(mean_In_conc * n_locs)
    )  # find how many In atoms there are, floor to keep as an integer
    n_As_locs = int(np.ceil(0.5 * n_locs))  # 50% of the lattice is As
    n_Ga_locs = int(n_locs - (n_In_locs + n_As_locs))  # the rest is Ga

    In_loc_list = random_locations_list_generator(n_In_locs, region_bounds)
    As_loc_list = random_locations_list_generator(n_As_locs, region_bounds)
    Ga_loc_list = random_locations_list_generator(n_Ga_locs, region_bounds)

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
        NMR_parrallel_calculation(
            nuclear_species,
            region_bounds,
            faked_strain_type,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            species_loc_list,
            recalc=recalc,
        )

        data.append(
            np.log(
                NMR_load_data(
                    nuclear_species,
                    region_bounds,
                    faked_strain_type,
                    applied_field_list,
                    field_geometry,
                    rf_freq_list,
                    species_loc_list,
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
        filename = f"{graph_path}layered_NMR_spectra_experiment_in_{field_geometry}_{n_locs}_locs_in_{region_bounds}_over_{applied_field_list[0]}_{applied_field_list[-1]}T_{rf_freq_list[0]/1e6}_{rf_freq_list[-1]/1e6}MHz_in_{faked_strain_type}_mirrored_dot.png"
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved file: {filename}")
    else:
        pass
        plt.show()

    plt.close()


# --------------------------------------------------------------------------------

# region_bounds = [750, 800, 400, 500]
# create_mirrored_data.create_left_right_strain_data(region_bounds)
# create_mirrored_data.create_left_right_conc_data(region_bounds)

# plot_concentration_image(region_bounds, "left_right", saving = False)
# plot_strain_image(region_bounds, "left_right", saving = False)

# plot_arrows_with_strength_background("As75", region_bounds, "left_right", saving = False)
# plot_biaxiality("As75", region_bounds, "left_right", saving = False)

# quadrupolar_strength_distributions(region_bounds, "left_right", saving = False)
# single_histogram_boltzmann_fit("As75", region_bounds, "left_right", saving = False)

# random_sites_energy_level_diagram("As75", region_bounds, "Faraday", "left_right", 10, saving = False)
# random_sites_geometry_comparison_EL_diagram("As75", region_bounds, "left_right", 10, saving = False)


# min_freq = 0
# max_freq = 75 # in MHz
# n_freqs = 300
# min_field = 0.001 # to stop divide by 0 errors when taking logs (16/12/20)
# max_field = 2 # in Tesla
# n_fields = 300

# rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6 # the 1e6 makes it MHz
# applied_field_list = np.linspace(min_field, max_field, n_fields)
# loc_list = random_locations_list_generator(10, region_bounds)

# NMR_plot("As75", region_bounds, "left_right", applied_field_list, "Faraday", rf_freq_list, loc_list, saving = False, recalc = False)

# NMR_experiment_plot("Faraday", region_bounds, "left_right", 10, recalc = False, saving = False)

# -------------------------------------------------------------------------------

start = time.time()

left_right_region_bounds = create_mirrored_data.left_right_region_bounds
right_left_region_bounds = create_mirrored_data.right_left_region_bounds

region_bounds_list = [left_right_region_bounds, right_left_region_bounds]

# create_mirrored_data.create_left_right_strain_data(left_right_region_bounds)
# create_mirrored_data.create_left_right_conc_data(left_right_region_bounds)

# create_mirrored_data.create_right_left_strain_data(right_left_region_bounds)
# create_mirrored_data.create_right_left_conc_data(right_left_region_bounds)

min_freq = 0
max_freq = 75  # in MHz
n_freqs = 1000
min_field = 0.001  # to stop divide by 0 errors when taking logs (16/12/20)
max_field = 2  # in Tesla
n_fields = 1000

n_locs = 24

rf_freq_list = np.linspace(min_freq, max_freq, n_freqs) * 1e6  # the 1e6 makes it MHz
applied_field_list = np.linspace(min_field, max_field, n_fields)

for r, faked_strain_type in enumerate(["left_right", "right_left"]):
    region_bounds = region_bounds_list[r]

    loc_list = random_locations_list_generator(n_locs, region_bounds)

    # plot_concentration_image(region_bounds, faked_strain_type, saving = True)

    # plot_strain_image(region_bounds, faked_strain_type, saving = True)

    # all_species_boltzmann_fit(region_bounds, faked_strain_type, saving = False, positive_only = False)
    all_species_gamma_fit(
        region_bounds, faked_strain_type, saving=True, positive_only=False
    )
    all_species_gamma_fit(
        region_bounds, faked_strain_type, saving=True, positive_only=True
    )

    # quadrupolar_strength_distributions(region_bounds, faked_strain_type, saving = True, positive_only = True)

    for nuclear_species in bqf.nuclear_species_list:
        # plot_arrows_with_strength_background(nuclear_species, region_bounds, faked_strain_type, saving = True)

        # plot_biaxiality(nuclear_species, region_bounds, faked_strain_type, saving = True)

        # single_histogram_boltzmann_fit(nuclear_species, region_bounds, faked_strain_type, saving = True, positive_only = True)
        single_histogram_gamma_fit(
            nuclear_species,
            region_bounds,
            faked_strain_type,
            saving=True,
            positive_only=False,
        )
        single_histogram_gamma_fit(
            nuclear_species,
            region_bounds,
            faked_strain_type,
            saving=True,
            positive_only=True,
        )

        # random_sites_geometry_comparison_EL_diagram(nuclear_species, region_bounds, faked_strain_type, n_locs, saving = True)

        # for field_geometry in bqf.field_geometries:
        #     NMR_plot(nuclear_species, region_bounds, faked_strain_type, applied_field_list, field_geometry, rf_freq_list, loc_list, saving = True, recalc = False)
        #     random_sites_energy_level_diagram(nuclear_species, region_bounds, field_geometry, faked_strain_type, n_locs, saving = True)
        #     NMR_experiment_plot(field_geometry, region_bounds, faked_strain_type, n_locs, recalc = False, saving = True)

end = time.time()
print(f"Time taken: {np.around((end-start)/60, 2)} minutes.")
