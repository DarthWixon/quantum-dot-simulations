import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import isotope_parameters as ISOP
import scipy.constants as constants

def load_mirrored_In_concentration_data(region_bounds, faked_strain_type = "left_right"):
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

    filename = f"{data_path}{faked_strain_type}_mirrored_In_conc_data.npy"
    full_conc_data = np.load(filename)

    H_1, H_2, L_1, L_2 = region_bounds
    step_size = 1
    # slice out range of data we want, slicing syntax is [start:stop:step]
    dot_conc_data = full_conc_data[L_1:L_2:step_size,H_1:H_2:step_size]

    return dot_conc_data

def plot_concentration_image(region_bounds, faked_strain_type, saving = False):

    conc_data = load_mirrored_In_concentration_data(region_bounds, faked_strain_type)

    plt.imshow(conc_data, cmap = cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())
    plt.colorbar(orientation = "horizontal", label = "Indium Concentration", shrink = 0.5)
    plt.axis("off")
    # plt.title("Indium Concentration")
    # plt.tight_layout()

    if saving:
        filename = f"{graph_path}In_concentration_map{faked_strain_type}_flipped.png"
        plt.savefig(filename, bbox_inches = "tight")
        print(f"Saved graph: {filename}")
    else:
        plt.show()

    plt.close()


def get_mirrored_EFG_data(nuclear_species, region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False, real_strain_data = True, faked_strain_type = "left_right"):

    bqf.calculate_and_save_EFG(nuclear_species, region_bounds, use_sundfors_GET_vals = use_sundfors_GET_vals, real_strain_data = True, faked_strain_type = "left_right")
    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds, use_sundfors_GET_vals = use_sundfors_GET_vals, real_strain_data = True, faked_strain_type = "left_right")

    return eta, V_XX, V_YY, V_ZZ, euler_angles

def plot_arrows_with_strength_background(nuclear_species, region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False, saving = False, real_strain_data = True, faked_strain_type = "left_right"):
    eta, V_XX, V_YY, V_ZZ, euler_angles = get_mirrored_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals, real_strain_data, faked_strain_type)

    species = ISOP.species_dict[nuclear_species]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]

    h = constants.h
    e = constants.e
    K = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq
    quadrupole_frequency = np.abs(K * V_ZZ)/1e6 # convert to MHz

    spacing = 50
    fig, ax = plt.subplots(1, 1, figsize = (12, 8), constrained_layout = True)

    n, m = quadrupole_frequency.shape

    X,Y = np.meshgrid(np.arange(0,m,1),np.arange(0,n,1))

    r = (quadrupole_frequency**2 + quadrupole_frequency**2)**0.5
    arrow_lengths = quadrupole_frequency/r

    Z_angles = euler_angles[:,:,2]*180/np.pi # convert angles from radians to degrees for the quiver function

    # plot the arrows showing QI size and direction
    # lines are repeated to get arrowheads at both ends
    ax.quiver( X[::spacing, ::spacing], Y[::spacing, ::spacing], arrow_lengths[::spacing, ::spacing], arrow_lengths[::spacing, ::spacing], angles=Z_angles[::spacing, ::spacing], 
                minshaft=5, pivot="tail", color="black")

    im = ax.imshow(quadrupole_frequency, cmap = cm.GnBu, vmin=quadrupole_frequency.min(), vmax=quadrupole_frequency.max())
    plt.axis("off")

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"
        
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cbar = plt.colorbar(im, location = "bottom", shrink = 0.5)
    cbar.set_label("Quadrupole Frequency (MHz)")

    # plt.tight_layout()

    if saving:
        filename = f"{graph_path}QI_arrows_in_region_{region_bounds}_for_{nuclear_species}_with_QI_strength_background_using_{GET_data_source}_GET_values.png"
        plt.savefig(filename, bbox_inches = "tight")
        print(f"Saved graph: {filename}")
    else:
        plt.show()

    plt.close()

region_bounds = [100, 1500, 200, 1000]
# faked_strain_type = "right_left"
faked_strain_type = "left_right"
# plot_concentration_image(region_bounds, faked_strain_type, saving = False)

for faked_strain_type in ["left_right", "right_left"]:
    plot_concentration_image(region_bounds, faked_strain_type, saving = True)