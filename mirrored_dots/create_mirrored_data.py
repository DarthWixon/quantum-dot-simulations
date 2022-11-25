import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")

from backbone_quadrupolar_functions import data_path

import numpy as np

def load_region_of_sokolov_data(region_bounds = [100, 1200, 200, 1000], step_size = 1):
    """
    Loads the full set of epsilon data from Sokolov, and packages it in a neat format.

    These datasets have different names according to the naming conventions
    that Sokolov puts in their paper. It's mainly a code error carried forward that 
    meant that the saved data has a weird file name. 
    The correct output labels are xx, xz and zz. These are the axes along which
    strain is measured in the paper.
    Paper these data are from (DOI): 10.1103/PhysRevB.93.045301

    Args:
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Entire dot region is:[100, 1200, 200, 1000].
            Dot only region is: [100,1200, 439, 880].
            Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 1 (every site is used).

    Returns:
        dot_epsilon_xx (ndarray): The xx components of strain across the sample area.
            Each point in the array represents a specific site.
        dot_epsilon_xz (ndarray): The xz components of strain across the sample area.
            Each point in the array represents a specific site.
        dot_epsilon_zz (ndarray): The zz components of strain across the sample area.
            Each point in the array represents a specific site.
    """

    # these full_epsilon_AA datasets are 1600x1600 arrays
    full_xx_data = np.loadtxt(f"{data_path}full_epsilon_xx.txt") 
    full_xy_data = np.loadtxt(f"{data_path}full_epsilon_xy.txt") 
    full_yy_data = np.loadtxt(f"{data_path}full_epsilon_yy.txt")

    H_1, H_2, L_1, L_2 = region_bounds

    # slice out range of data we want, slicing syntax is [start:stop:step]
    region_epsilon_xx = full_xx_data[L_1:L_2:step_size,H_1:H_2:step_size]
    region_epsilon_xz = -full_xy_data[L_1:L_2:step_size,H_1:H_2:step_size]
    region_epsilon_zz = -full_yy_data[L_1:L_2:step_size,H_1:H_2:step_size]

    return region_epsilon_xx, region_epsilon_xz, region_epsilon_zz

def load_In_concentration_data(region_bounds = [100, 1200, 200, 1000], step_size = 1, method = "cubic"):
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

    filename = f"{data_path}conc_data_to_scale_{method}_interpolation.npy"
    full_conc_data = np.load(filename)

    H_1, H_2, L_1, L_2 = region_bounds

    # slice out range of data we want, slicing syntax is [start:stop:step]
    region_conc_data = full_conc_data[L_1:L_2:step_size,H_1:H_2:step_size]

    return region_conc_data

def mirror_array_left_to_right(data_array):
	
	left_half = data_array[:, 0:int(data_array.shape[1]/2)]

	right_half = np.fliplr(left_half)

	left_right_symmetric_array = np.hstack([left_half, right_half])

	return left_right_symmetric_array

def mirror_array_right_to_left(data_array):

	right_half = data_array[:, int(data_array.shape[1]/2):int(data_array.shape[1])]

	left_half = np.fliplr(right_half)

	right_left_symmetric_array = np.hstack([left_half, right_half])

	return right_left_symmetric_array

def create_left_right_strain_data(region_bounds = [100, 1200, 200, 1000]):
	xx, xy, yy = load_region_of_sokolov_data(region_bounds)

	flipped_xx = mirror_array_left_to_right(xx)
	flipped_xy = mirror_array_left_to_right(xy)
	flipped_yy = mirror_array_left_to_right(yy)

	np.savez(f"{data_path}left_right_mirrored_strain_data_in_region_{region_bounds}.npz", full_xx_data = flipped_xx, full_xy_data = flipped_xy, full_yy_data = flipped_yy)

def create_right_left_strain_data(region_bounds = [100, 1200, 200, 1000]):
	xx, xy, yy = load_region_of_sokolov_data(region_bounds)

	flipped_xx = mirror_array_right_to_left(xx)
	flipped_xy = mirror_array_right_to_left(xy)
	flipped_yy = mirror_array_right_to_left(yy)

	np.savez(f"{data_path}right_left_mirrored_strain_data_in_region_{region_bounds}.npz", full_xx_data = flipped_xx, full_xy_data = flipped_xy, full_yy_data = flipped_yy)

def create_left_right_conc_data(region_bounds = [100, 1200, 200, 1000]):
    conc_data = load_In_concentration_data(region_bounds)

    flipped_conc = mirror_array_left_to_right(conc_data)
    max_conc = np.amax(flipped_conc)
    # print(f"Max In Concentration in Left-Right array: {max_conc}")

    np.save(f"{data_path}left_right_mirrored_In_conc_data_in_region_{region_bounds}.npy", flipped_conc)

def create_right_left_conc_data(region_bounds = [100, 1200, 200, 1000]):
    conc_data = load_In_concentration_data(region_bounds)

    flipped_conc = mirror_array_right_to_left(conc_data)
    max_conc = np.amax(flipped_conc)
    # print(f"Max In Concentration in Right-Left array: {max_conc}")

    np.save(f"{data_path}right_left_mirrored_In_conc_data_in_region_{region_bounds}.npy", flipped_conc)


left_right_region_bounds = [100, 1300, 440, 880]
# left_right_region_bounds = [100, 1300, 200, 1000]
create_left_right_strain_data(left_right_region_bounds)
create_left_right_conc_data(left_right_region_bounds)

right_left_region_bounds = [100, 1300, 440, 880]
# right_left_region_bounds = [100, 1300, 200, 1000]
create_right_left_strain_data(right_left_region_bounds)
create_right_left_conc_data(right_left_region_bounds)
