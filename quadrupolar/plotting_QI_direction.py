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


# this file needs to be updated to take advantage of my archives of data for the EFG tensor, rather than recalculating it each time.
# also need to update graphs so they display which species, and using what data

def get_EFG_data(nuclear_species, region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False):

    bqf.calculate_and_save_EFG(nuclear_species, region_bounds, use_sundfors_GET_vals = use_sundfors_GET_vals)
    eta, V_XX, V_YY, V_ZZ, euler_angles = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds, use_sundfors_GET_vals = use_sundfors_GET_vals)

    return eta, V_XX, V_YY, V_ZZ, euler_angles

def plot_arrows_with_biax_background(nuclear_species, region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False, saving = False):

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals)

    spacing = 20
    fig, ax = plt.subplots(figsize = (12, 8))

    n, m = V_ZZ.shape

    X,Y = np.meshgrid(np.arange(0,m,1),np.arange(0,n,1))

    Z_angles = euler_angles[:,:,2]*180/np.pi # convert angles from radians to degrees for the quiver function

    # plot the arrows showing QI size and direction
    # lines are repeated to get arrowheads at both ends
    ax.quiver( X[::spacing, ::spacing], Y[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], angles=Z_angles[::spacing, ::spacing], 
                minshaft=5, pivot="middle", color="black")
    ax.quiver( X[::spacing, ::spacing], Y[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], angles=Z_angles[::spacing, ::spacing]+180,
                minshaft=5, pivot="middle", color="black")

    im = ax.imshow(eta, cmap = cm.GnBu, vmin=eta.min(), vmax=eta.max())
    plt.axis("off")

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    plt.title(f"Size and Direction of QI for {nuclear_species} with $\eta$ Shown. Calculated Using {GET_data_name} GET Values")

    cbar=plt.colorbar(im, orientation = "horizontal")
    cbar.ax.set_xlabel("$\eta$", fontsize=16)

    plt.tight_layout()

    if saving:
        filename = f"{graph_path}QI_arrows_in_region_{region_bounds}_for_{nuclear_species}_with_biaxiality_using_{GET_data_source}_GET_values.png"
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()

def comparison_plot_with_biax_background(nuclear_species, region_bounds):
    fig = plt.figure()

    grid = AxesGrid(fig, 111, nrows_ncols = (2,1), cbar_mode = "single", cbar_location = "right", axes_pad = 0.25, cbar_pad = 0.1)
    arrow_spacing = 40

    ax = grid[0]

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals = False)
    n, m = V_ZZ.shape
    X,Y = np.meshgrid(np.arange(0,m,1),np.arange(0,n,1))
    Z_angles = euler_angles[:,:,2]*180/np.pi # convert angles from radians to degrees for the quiver function

    # plot the arrows showing QI size and direction
    # lines are repeated to get arrowheads at both ends
    ax.quiver( X[::arrow_spacing, ::arrow_spacing], Y[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], angles=Z_angles[::arrow_spacing, ::arrow_spacing], 
                minshaft=5, pivot="middle", color="black")
    ax.quiver( X[::arrow_spacing, ::arrow_spacing], Y[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], angles=Z_angles[::arrow_spacing, ::arrow_spacing]+180,
                minshaft=5, pivot="middle", color="black")

    im_sundfors = ax.imshow(eta, cmap = cm.GnBu, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("2018/19 Data")

    ax = grid[1]

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals = True)
    n, m = V_ZZ.shape
    X,Y = np.meshgrid(np.arange(0,m,1),np.arange(0,n,1))
    Z_angles = euler_angles[:,:,2]*180/np.pi # convert angles from radians to degrees for the quiver function

    # plot the arrows showing QI size and direction
    # lines are repeated to get arrowheads at both ends
    ax.quiver( X[::arrow_spacing, ::arrow_spacing], Y[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], angles=Z_angles[::arrow_spacing, ::arrow_spacing], 
                minshaft=5, pivot="middle", color="black")
    ax.quiver( X[::arrow_spacing, ::arrow_spacing], Y[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], V_ZZ[::arrow_spacing, ::arrow_spacing], angles=Z_angles[::arrow_spacing, ::arrow_spacing]+180,
                minshaft=5, pivot="middle", color="black")

    im_check = ax.imshow(eta, cmap = cm.GnBu, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("1974 Data")

    cbar = ax.cax.colorbar(im_sundfors)
    cbar = grid.cbar_axes[0].colorbar(im_sundfors)
    cbar.ax.set_ylabel("$\eta$", fontsize=16, rotation = 0)

    fig.tight_layout()
    plt.show()



def plot_arrows_with_conc_background(nuclear_species, region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False, saving = False):
    sys.path.append("/home/will/Documents/work/research/simulations/concentration/")
    import conc_maps as conc

    conc_data = conc.load_In_concentration_data(region_bounds)

    eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals)

    spacing = 20
    fig, ax = plt.subplots(figsize = (12, 8))

    n, m = V_ZZ.shape

    X,Y = np.meshgrid(np.arange(0,m,1),np.arange(0,n,1))

    Z_angles = euler_angles[:,:,2]*180/np.pi # convert angles from radians to degrees for the quiver function

    # plot the arrows showing QI size and direction
    # lines are repeated to get arrowheads at both ends
    ax.quiver( X[::spacing, ::spacing], Y[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], angles=Z_angles[::spacing, ::spacing], 
                minshaft=5, pivot="middle", color="black")
    ax.quiver( X[::spacing, ::spacing], Y[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], V_ZZ[::spacing, ::spacing], angles=Z_angles[::spacing, ::spacing]+180,
                minshaft=5, pivot="middle", color="black")

    im = ax.imshow(conc_data, cmap = cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())
    plt.axis("off")

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"
        
    # plt.title(f"Size and Direction of EFG for {nuclear_species} with In Concentration Shown. Calculated Using {GET_data_name} GET Values")

    cbar=plt.colorbar(im, orientation = "horizontal")
    cbar.ax.set_xlabel("In Conc.", fontsize=16)

    plt.tight_layout()

    if saving:
        filename = f"{graph_path}QI_arrows_in_region_{region_bounds}_for_{nuclear_species}_with_In_concentration_using_{GET_data_source}_GET_values.png"
        plt.savefig(filename)
        print(f"Saved graph: {filename}")
    else:
        plt.show()

    plt.close()

def plot_arrows_with_strength_background(nuclear_species, region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False, saving = False):
    eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals)

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

def plot_arrows_with_strength_background_all_species(region_bounds = [100, 1200, 200, 1000], use_sundfors_GET_vals = False, saving = False):

    fig, axs = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)

    freq_max_list = []
    quad_array_list = []

    for i in range(4):
        nuclear_species = bqf.nuclear_species_list[i]
        eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals)

        species = ISOP.species_dict[nuclear_species]
        spin = species["particle_spin"]
        Q = species["quadrupole_moment"]

        h = constants.h
        e = constants.e
        K = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq
        quadrupole_frequency = np.abs(K * V_ZZ)/1e6 # convert to MHz

        freq_max_list.append(np.max(quadrupole_frequency))
        quad_array_list.append(quadrupole_frequency)

    # COLOR STUFF HERE
    cmap = cm.get_cmap("GnBu")
    normalizer = Normalize(0, np.max(freq_max_list))
    im = cm.ScalarMappable(norm = normalizer, cmap = cmap)

    for i, ax in enumerate(axs.flat):
        nuclear_species = bqf.nuclear_species_list[i]
        species = ISOP.species_dict[nuclear_species]
        species_name = species["graph_name"]

        spacing = 50
        quadrupole_frequency = quad_array_list[i]
        eta, V_XX, V_YY, V_ZZ, euler_angles = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals)

        n, m = quadrupole_frequency.shape

        X,Y = np.meshgrid(np.arange(0,m,1),np.arange(0,n,1))
        r = (quadrupole_frequency**2 + quadrupole_frequency**2)**0.5
        arrow_lengths = quadrupole_frequency/r

        Z_angles = euler_angles[:,:,2]*180/np.pi # convert angles from radians to degrees for the quiver function

        # plot the arrows showing QI size and direction
        ax.quiver( X[::spacing, ::spacing], Y[::spacing, ::spacing], arrow_lengths[::spacing, ::spacing], arrow_lengths[::spacing, ::spacing], angles=Z_angles[::spacing, ::spacing], 
                    minshaft=5, pivot="tail", color="black")

        ax.imshow(quadrupole_frequency, cmap = cm.GnBu, norm = normalizer)
        ax.set_title(species_name)
        ax.axis("off")

    cbar = fig.colorbar(im, ax = axs.ravel().tolist(), shrink = 0.9)
    cbar.set_label("Quadrupole Frequency (MHz)")

    if use_sundfors_GET_vals:
        GET_data_name = "Sundfors 1974"
        GET_data_source = "sundfors_1974"
    else:
        GET_data_name = "Checkovich 2018/19"
        GET_data_source = "checkovich_2018_19"

    if saving:
        filename = f"{graph_path}QI_arrows_in_region_{region_bounds}_for_all_species_with_QI_strength_background_using_{GET_data_source}_GET_values.png"
        plt.savefig(filename, bbox_inches = "tight")
        print(f"Saved graph: {filename}")
    else:
        plt.show()

    plt.close()


def data_comparison(nuclear_species, region_bounds = [100, 1200, 200, 1000]):
    output = ["eta", "V_XX", "V_YY", "V_ZZ", "euler_angles"]
    check_data = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals = False) # [4] because euler angles is the 5th element 
    sund_data = get_EFG_data(nuclear_species, region_bounds, use_sundfors_GET_vals = True)  # in the result of these functions, and is all I want  

    for i in range(len(check_data)):
        print(f"Same for {output[i]}: {np.allclose(check_data[i], sund_data[i])}")

# region_bounds = [500, 550, 900, 1000]
# region_bounds = [10, 1500, 400, 900]
# region_bounds = [100, 1200, 200, 1000]
# region_bounds = [100,1200, 439, 880]

# region_bounds = [450, 850, 550, 650]
# for nuclear_species in bqf.nuclear_species_list:
#     plot_arrows_with_biax_background(nuclear_species, region_bounds = region_bounds, use_sundfors_GET_vals = True, saving = True)
#     plot_arrows_with_biax_background(nuclear_species, region_bounds = region_bounds, use_sundfors_GET_vals = False, saving = True)
#     plot_arrows_with_conc_background(nuclear_species, region_bounds = region_bounds, use_sundfors_GET_vals = True, saving = True)
#     plot_arrows_with_conc_background(nuclear_species, region_bounds = region_bounds, use_sundfors_GET_vals = False, saving = True)


# plot_arrows_with_biax_background("Ga69", region_bounds = region_bounds, use_sundfors_GET_vals = False, saving = False)
# plot_arrows_with_conc_background("Ga69", region_bounds = region_bounds, use_sundfors_GET_vals = False, saving = False)
# plot_arrows_with_conc_background("Ga69", region_bounds = region_bounds, use_sundfors_GET_vals = True, saving = False)

# for nuclear_species in bqf.nuclear_species_list:
#     print("----------")
#     print(nuclear_species)
#     data_comparison(nuclear_species, [750, 800, 400, 500])

# plot_arrows_with_biax_background("Ga69", use_sundfors_GET_vals = False)

# comparison_plot_with_biax_background("Ga69", region_bounds)

# plot_arrows_with_strength_background("Ga69", region_bounds = region_bounds, use_sundfors_GET_vals = False, saving = False)

for nuclear_species in bqf.nuclear_species_list:
    plot_arrows_with_strength_background(nuclear_species, region_bounds = region_bounds, use_sundfors_GET_vals = False , saving = True)
    plot_arrows_with_strength_background(nuclear_species, region_bounds = region_bounds, use_sundfors_GET_vals = True , saving = True)


# plot_arrows_with_strength_background_all_species(use_sundfors_GET_vals = False , saving = False)

# plot_arrows_with_strength_background_all_species(region_bounds = region_bounds, use_sundfors_GET_vals = False , saving = True)
# plot_arrows_with_strength_background_all_species(region_bounds = region_bounds, use_sundfors_GET_vals = True , saving = True)