import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import isotope_parameters as ISOP
import scipy.constants as constants

def sokolov_strain_grapher(region_bounds = [100, 1200, 200, 1000], saving = False):
    """
    Function to recreate the graphs of strain found in the Sokolov paper.

    Args:
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]
        saving (bool): A boolean that determines if the resulting figure should be saved.
            Default is False (figure is not saved).

    Returns:
        Does not return anything. Depending on options will either save a figure
            or display it to the screen.    
    """

    xx_array, xz_array, zz_array = bqf.load_sokolov_data(region_bounds)
    print(xx_array.shape)
    # Strain tensor components
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    im=axs[0].imshow(xx_array, vmin=-0.02, vmax=0.02)
    axs[0].axis("off")
    axs[0].text(20, 150,"$\epsilon_{xx}$",fontsize=16, fontdict=None)

    axs[1].imshow(zz_array, vmin=-0.02, vmax=0.02)
    axs[1].axis("off")
    axs[1].text(20, 150,"$\epsilon_{zz}$",fontsize=16, fontdict=None)


    axs[2].imshow(xz_array, vmin=-0.02, vmax=0.02)
    axs[2].axis("off")
    axs[2].text(20, 150,"$\epsilon_{xz}$",fontsize=16, fontdict=None)

    cbar_ax = fig.add_axes([0.1, 0.1, 0.5, 0.05])
    cbar=plt.colorbar(im, cax=cbar_ax, orientation="horizontal")

    if saving:
        plot_name = f"{graph_path}Sokolov_strain_graph_recreation_region_{region_bounds}.png"
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()

def sokolov_strain_vertical_grapher(region_bounds = [100, 1200, 200, 1000], saving = False):
    """
    Function to recreate the graphs of strain found in the Sokolov paper.

    Args:
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]
        saving (bool): A boolean that determines if the resulting figure should be saved.
            Default is False (figure is not saved).

    Returns:
        Does not return anything. Depending on options will either save a figure
            or display it to the screen.    
    """

    xx_array, xz_array, zz_array = bqf.load_sokolov_data(region_bounds)
    fig, axs = plt.subplots(ncols = 1, nrows = 3, constrained_layout = True, figsize = (6, 9))

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
    im = cm.ScalarMappable(norm = normalizer, cmap = cmap)

    axs[0].imshow(xx_array, cmap = cmap, norm = normalizer)
    axs[0].axis("off")
    axs[0].text(20, 150,"$\epsilon_{xx}$",fontsize=16, fontdict=None)

    axs[1].imshow(zz_array, cmap = cmap, norm = normalizer)
    axs[1].axis("off")
    axs[1].text(20, 150,"$\epsilon_{zz}$",fontsize=16, fontdict=None)


    axs[2].imshow(xz_array, cmap = cmap, norm = normalizer)
    axs[2].axis("off")
    axs[2].text(20, 150,"$\epsilon_{xz}$",fontsize=16, fontdict=None)

    cbar=plt.colorbar(im, ax=axs.ravel().tolist(), shrink = 0.95, orientation = "vertical")

    if saving:
        plot_name = f"{graph_path}Sokolov_strain_graph_vertical_region_{region_bounds}.png"
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name, bbox_inches = "tight")
    else:
        plt.show()
    plt.close()

def strain_histograms(region_bounds = [100, 1200, 200, 1000], saving = False):
    """
    Function to create a histogram of shear strains across a region.

    Args:
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]
        saving (bool): A boolean that determines if the resulting figure should be saved.
            Default is False (figure is not saved).

    Returns:


    # Paper these data are from (DOI): 10.1103/PhysRevB.93.045301

    """

    # we have to bear in mind the assumptions the paper makes about strain
    # quoting directly from the paper:
    # "We take ε_yy = ε_xx and ε_yz = ε_xz keeping
    #  ε_xy = 0. While ε_xy = 0 is a reasonable assumption since it is
    #  large at the heteroboundary, which is unsharp in the annealed
    #  QDs, other assumptions require consideration."

    # See the following papers for more detail on these assumptions:
    # - https://doi.org/10.1103/PhysRevB.88.075430
    # - https://doi.org/10.1103/PhysRevLett.104.196803
    # - https://doi.org/10.1103/PhysRevB.69.161301

    xx_array, xz_array, zz_array = bqf.load_sokolov_data(region_bounds)

    calculated_shear_strain = (2*np.abs(xz_array)).flatten() # based on the normal definition
    measured_shear_strain = xz_array.flatten() # what Sokolov calls shear strain

    plt.figure(figsize = (12, 8))
    vals, bins, patches = plt.hist([calculated_shear_strain, measured_shear_strain], "auto", density = False, histtype = "step", label = ["$\epsilon_S = |\epsilon_{xy}| + |\epsilon_{yz}| + |\epsilon_{xz}|$", "$\epsilon_{xz}$ as measured"])
    plt.legend()
    plt.xlabel("Shear Strain")
    plt.ylabel("Some Measure of Amount")
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)

    plot_name = f"{graph_path}strain_hists_in_region_{region_bounds}.png"
    plt.title("Shear Strain Distributions")
    if saving:
        print("Saving plot: {}".format(plot_name))
        plt.savefig(plot_name)
    else:
        plt.show()

    plt.close()

    return

def single_row_strain_grapher(row, region_bounds = [100, 1200, 200, 1000], saving = False):
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    # row is here given as a number within the region bounds, so 0 is the first always
    xx_array, xz_array, zz_array = bqf.load_sokolov_data(region_bounds)

    num_y, num_x = xx_array.shape
    x_range = list(range(num_x))
    y_range = list(range(num_y))
    
    plt.figure(figsize = (12, 8))

    plt.plot(xx_array[row, :], label = "XX Strain")
    plt.plot(xz_array[row, :], label = "XZ Strain")
    plt.plot(zz_array[row, :], label = "ZZ Strain")
    plt.legend()
    plt.show()
    plt.close()

def many_rows_strain_grapher_3d(strain_axis = "xz", region_bounds = [100, 1200, 200, 1000], saving = False):
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    xx_array, xz_array, zz_array = bqf.load_sokolov_data(region_bounds)

    if strain_axis == "xz":
        strain_array = xz_array
    elif strain_axis == "xx":
        strain_array = xx_array
    elif strain_axis == "zz":
        strain_array = zz_array

    num_y, num_x = strain_array.shape
    x_range = list(range(num_x))
    y_range = list(range(num_y))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x_range, y_range)

    surf = ax.plot_surface(X,Y, strain_array, cmap = cm.YlGn)
    ax.set_xlabel("X Co-ord")
    ax.set_ylabel("Y Co-ord")
    ax.set_zlabel(f"{strain_axis} Strain")
    plt.show()

def finding_atomic_sites(strain_axis = "xz", region_bounds = [100, 1200, 200, 1000], saving = False):
    from skimage.feature import peak_local_max
    xx_array, xz_array, zz_array = bqf.load_sokolov_data(region_bounds)

    if strain_axis == "xz":
        strain_array = xz_array
    elif strain_axis == "xx":
        strain_array = xx_array
    elif strain_axis == "zz":
        strain_array = zz_array
    
    locs = peak_local_max(strain_array, min_distance = 10, indices = False)
    # print(np.sum(locs))
    
    plt.imshow(locs, origin = "lower")
    # plt.grid(which = "both")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Peak Locations")

    plt.show()



# Control panel
recreating_Sokolov_strain = False
neater_solokov_strain = True
plotting_histograms = False
graphing_a_row = False
graphing_many_rows = False
finding_sites = False

# region_bounds = [100, 1100, 800, 1000]
region_bounds = [100, 1200, 200, 1000]
row = 0

if recreating_Sokolov_strain:
    print("Graphing Strain")
    # makes a weird graph as of 03/09/20
    sokolov_strain_grapher(region_bounds, saving = False)

if neater_solokov_strain:
    sokolov_strain_vertical_grapher(region_bounds, saving = True)

if plotting_histograms:
    strain_histograms(region_bounds, saving = True)

if graphing_a_row:
    single_row_strain_grapher(row, region_bounds, saving = False)

if graphing_many_rows:
    many_rows_strain_grapher_3d()

if finding_sites:
    finding_atomic_sites()