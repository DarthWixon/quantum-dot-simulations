import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import isotope_parameters as ISOP
import scipy.constants as constants
import time

def equivalent_B_field_calculation(nuclear_species, region_bounds = [100, 1200, 200, 1000]):
    """
    Calculates the magnetic field equivalent of the quadrupolar field at each site.

    For each site in the given region, this calculates the equivalent magnetic field
    required to give the same splitting in energy levels via the Zeeman interaction.
    This is useful as it shows the size of the quadrupolar interaction that a specific
    nucleus experiences.

    Args:
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]

    Returns:
        equivalent_B_field (ndarray): The magnetic field equivalent to the QI 
            at each site.
    """

    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)
    n = V_ZZ.shape[0]
    m = V_ZZ.shape[1]

    K = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq
    quadrupole_frequency = K * V_ZZ

    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]

    equivalent_B_field = quadrupole_frequency/zeeman_frequency_per_tesla

    return equivalent_B_field

def single_equivalent_B_field_grapher(nuclear_species, equivalent_B_field, saving = False, region_bounds = [100, 1200, 200, 1000]):
    """
    Plots a heatmap of equivalent magnetic field across a particular region for a single species.

    This function takes the result of the equivalent_B_field_calculation function and plots
    it as a heatmap. It can save the heatmap or simply display it.

    Args:
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        equivalent_B_field (ndarray): The magnetic field equivalent to the QI 
            at each site.
        saving (bool): A boolean that determines if the resulting figure should be saved.
            Default is False (figure is not saved).
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]

    Returns:
        Does not return anything. Depending on options will either save a figure
            or display it to the screen.
    """

    species = ISOP.species_dict[nuclear_species] # one of ["Ga69", "Ga71", "As75", "In115"] (as strings)
    fig, ax = plt.subplots()
    im = ax.imshow(equivalent_B_field)
    plt.axis("off")
    plt.colorbar(im)
    plot_name = f"{graph_path}equivalent_B_field_for_" + species["short_name"] + "_in_region_{}.png".format(str(region_bounds))
    plt.title("Equivalent B Field for {}".format(species["name"]))
    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()

def all_equivalent_B_field_grapher(B_field_arrays, saving = True, region_bounds = [100, 1200, 200, 1000]):
    """
    Plots the equivalent magnetic field across a region for all 4 nuclear species.

    This creates a more complex graph than the single_equivalent_B_field_grapher function,
    it shows all 4 species in the same figure. If you want to create individual plots, 
    use the single_equivalent_B_field_grapher function instead.

    Args:
        B_field_arrays (ndarray): An set of arrays, each containing the equivalent magnetic 
            field data created by equivalent_B_field_calculation. Arrays should be presented
            in the order of: Ga69, Ga71, As75, In115.
        saving (bool): A boolean that determines if the resulting figure should be saved.
            Default is True (Figure is saved).
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]

    Returns:
        Does not return anything. Depending on options will either save a figure
            or display it to the screen.
    """

    nuclear_speciesList = ["Ga69", "Ga71", "As75", "In115"]

    fig, axes = plt.subplots(2,2)
    print(axes.shape)

    for i, ax in enumerate(axes.flatten()):
        
        nuclear_species = nuclear_speciesList[i]
        species = ISOP.species_dict[nuclear_species]
        equivalent_B_field = B_field_arrays[i]

        im = ax.imshow(equivalent_B_field)
        ax.set_title("Equivalent B Field for {}".format(species["name"]), fontsize = 20)
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax3, with 20% width of ax3
        cax = divider.append_axes("right", size="20%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        cbar = plt.colorbar(im, cax=cax)#, ticks=MultipleLocator(0.2), format="%.2f")
        cbar.ax.tick_params(labelsize = 14)
        ax.axis("off")

    # fig.tight_layout()
    plot_name = f"{graph_path}equivalent_B_field_for_all_elements_in_region_{region_bounds}.png"
    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()

def one_time_one_field_all_sites_correlator(t, applied_field, correlator_axis, nuclear_species, region_bounds = [100, 1200, 200, 1000]):
    """
    Calculates the spin correlator across all sites, for one time and one applied field.

    Args:
        t (float): The time at which to calculate the correlator.
        applied_field (float): The magnetic field applied to the quantum dot.
        correlator_axis (str): The spin axis along which to calculate the correlator.
            Options are: x, y, z.
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]

    Returns:
        dor_correlator (float): The spin correlator for the specified species, 
            calculated over the entire dot at a specific time and magnetic field.
    """

    data_dict = {}
    data_dict["eta"], data_dict["V_XX"], data_dict["V_YY"], data_dict["V_ZZ"], data_dict["euler_angles"] = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

    n_sites = data_dict["eta"].size
    total_correlator = 0
    h = constants.h
    e = constants.e

    species = ISOP.species_dict[nuclear_species]

    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]
    quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq

    for s in range(n_sites):
        biaxiality = data_dict["eta"].flatten()[s] # needs to be flattened here to make sure it's 1D
        euler_angles = data_dict["euler_angles"].reshape(n_sites, 3)[s] # has already been reshaped when I load it (as of 24/08/20 this may no longer be true)
        V_ZZ = data_dict["V_ZZ"].flatten()[s]

        total_correlator += bqf.site_correlator_calculator_not_parallel(t, zeeman_frequency_per_tesla, quadrupole_coupling_constant, spin, biaxiality, euler_angles, V_ZZ, applied_field, correlator_axis)
    
    dot_correlator = total_correlator/n_sites #

    return dot_correlator

# ControlPanel
calculating_EFG_data_1_step_size = True
calculating_EFG_many_step_sizes = False
creating_equivalent_B_field_graphs = False
creating_4_way_B_field_graphs = False

# set region corners, entire dot region is:[100, 1200, 200, 1000], in the order left, right, top, bottom (06/05/20)
# dot only region seems to be: [100,1200, 439, 880] (06/05/20)
# small testing region is: [750, 800, 400, 500]
region_bounds = [450, 850, 550, 650]


step_size = 1
step_size_list = list(range(0,100,5)) # make a list cos python3 returns a range object which doesn't support assignment
step_size_list[0] = 1 # do this so the first entry is 1, rather than 0

# standard list of nuclear species
nuclear_speciesList = ["Ga69", "Ga71", "As75", "In115"]

region_list = []
region_list.append([450, 850, 550, 650])
region_list.append([20, 420, 650, 750])
region_list.append([1010, 1410, 650, 750])
region_list.append([450, 850, 660, 760])

if calculating_EFG_data_1_step_size:
    print("Calculating Data")
    for nuclear_species in nuclear_speciesList:
        for region in region_list:
                bqf.calculate_and_save_EFG(nuclear_species, region, step_size = step_size)

if calculating_EFG_many_step_sizes:
    print("Calculating Data For Many Step Sizes")
    for nuclear_species in nuclear_speciesList:
        for step_size in step_size_list:
            bqf.calculate_and_save_EFG(nuclear_species, region_bounds, step_size = step_size)

if creating_equivalent_B_field_graphs:
    print("Creating Graphs")
    for nuclear_species in nuclear_speciesList:
        equivalent_B_field = equivalent_B_field_calculation(nuclear_species, region_bounds)
        single_equivalent_B_field_grapher(nuclear_species, equivalent_B_field, saving = True, region_bounds = region_bounds)

if creating_4_way_B_field_graphs:
    print("Creating 4 Way B Field Graph")
    B_field_arrays = []
    for i, nuclear_species in enumerate(nuclear_speciesList):
        B_field_arrays.append(equivalent_B_field_calculation(nuclear_species, region_bounds))
    all_equivalent_B_field_grapher(B_field_arrays, region_bounds = region_bounds)

