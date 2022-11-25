import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import isotope_parameters as ISOP
import scipy.constants as constants

def quadrupolar_interaction_strength_plotter(nuclear_species, region_bounds = [100, 1200, 200, 1000], saving = False):
    """
    Calculates and plots the quadrupolar frequency across a region.

    For each site in the given region, this calculates the quadrupolar 
    transition frequency of the particular species and plots it.

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
        
    """

    species = ISOP.species_dict[nuclear_species]
    species_name = species["short_name"]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)
    n = V_ZZ.shape[0]
    m = V_ZZ.shape[1]

    K = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq
    quadrupole_frequency = K * V_ZZ

    fig, ax = plt.subplots()
    im = ax.imshow(quadrupole_frequency)
    plt.axis("off")
    plt.colorbar(im)

    plot_name = f"{graph_path}quadrupolar_field_for_{species_name}_in_region_{region_bounds}.png"
    plt.title(f"Quadrupolar Frequency for {species_name}")
    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()

    return

def biaxiality_plotter(nuclear_species, region_bounds = [100, 1200, 200, 1000], saving = False, use_sundfors_GET_vals = False):
    """
    Calculates and plots the quadrupolar frequency across a region.

    For each site in the given region, this calculates the quadrupolar 
    transition frequency of the particular species and plots it.

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
        
    """

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    species = ISOP.species_dict[nuclear_species]
    species_name = species["short_name"]

    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds, use_sundfors_GET_vals = use_sundfors_GET_vals)
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 8), constrained_layout = True)
    im = ax.imshow(eta, cmap = cm.GnBu, vmin = 0, vmax = 1)
    plt.axis("off")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(im, location = "bottom", shrink = 0.5)
    cbar.set_label(r"$\eta$")

    plot_name = f"{graph_path}biaxiality_for_{species_name}_in_region_{region_bounds}_using_{GET_data_source}.png"
    # plt.title(f"Biaxiality of {species_name} Using {GET_data_name}")
    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name, bbox_inches = "tight")
    else:
        plt.show()
    plt.close()

    return

# set region corners, entire dot region is:[100, 1200, 200, 1000], in the order left, right, top, bottom (06/05/20)
# dot only region seems to be: [100,1200, 439, 880] (06/05/20)
# small testing region is: [750, 800, 400, 500]
region_bounds = [100, 1200, 200, 1000]


step_size = 1
step_size_list = list(range(0,100,5)) # make a list cos python3 returns a range object which doesn't support assignment
step_size_list[0] = 1 # do this so the first entry is 1, rather than 0

# standard list of nuclear species
nuclear_speciesList = ["Ga69", "Ga71", "As75", "In115"]

print("Calculating Data")
for nuclear_species in nuclear_speciesList:
    bqf.calculate_and_save_EFG(nuclear_species, region_bounds, step_size = step_size)

print("Creating Colourplots")
for nuclear_species in nuclear_speciesList:
    # quadrupolar_interaction_strength_plotter(nuclear_species, region_bounds = region_bounds, saving = False)
    biaxiality_plotter(nuclear_species, region_bounds = region_bounds, saving = True, use_sundfors_GET_vals = True)
    biaxiality_plotter(nuclear_species, region_bounds = region_bounds, saving = True, use_sundfors_GET_vals = False)
