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


def EFG_data_creator(nuclear_species_list, region_bounds_list, step_size_list, use_sundfors_GET_vals = False):
    for nuclear_species in nuclear_species_list:
        for region_bounds in region_bounds_list:
            for step_size in step_size_list:
                bqf.calculate_and_save_EFG(nuclear_species, region_bounds, step_size, use_sundfors_GET_vals)
    print("All calculations complete. Good job Will.")

def EFG_data_creator_fake_strain(nuclear_species_list, region_bounds_list, step_size_list, use_sundfors_GET_vals = False, faked_strain_type = "left_right"):
    for nuclear_species in nuclear_species_list:
        for region_bounds in region_bounds_list:
            for step_size in step_size_list:
                bqf.calculate_and_save_EFG(nuclear_species, region_bounds, step_size, use_sundfors_GET_vals, real_strain_data = False, faked_strain_type = faked_strain_type)
    print("All calculations complete. Good job Will.")

# set region corners, entire dot region is:[100, 1200, 200, 1000], in the order left, right, top, bottom (06/05/20)
# dot only region seems to be: [100,1200, 439, 880] (06/05/20)
# small testing region is: [750, 800, 400, 500]
# central box region is: [450, 850, 550, 650] (19/20/20)


step_size = 1
step_size_list = list(range(0,25,5)) # make a list cos python3 returns a range object which doesn't support assignment
step_size_list[0] = 1 # do this so the first entry is 1, rather than 0

step_size_list = [1]

# standard list of nuclear species is bqf.nuclear_species_list

region_bounds_list = []
# region_bounds_list.append([450, 850, 550, 650])
# region_bounds_list.append([20, 420, 650, 750])
# region_bounds_list.append([1010, 1410, 650, 750])
# region_bounds_list.append([450, 850, 660, 760])
# region_bounds_list.append([100, 1200, 200, 1000])
region_bounds_list.append([100,1200, 439, 880])
# region_bounds_list.append([750, 800, 400, 500])
# region_bounds_list.append([1010, 1410, 450, 550])
# region_bounds_list.append([600, 625, 585, 610])
# region_bounds_list.append([450, 850, 775, 875])
# region_bounds_list.append([560, 565, 600, 605])
# region_bounds_list.append([560, 575, 600, 625])


EFG_data_creator(bqf.nuclear_species_list, region_bounds_list, step_size_list, use_sundfors_GET_vals = True)
EFG_data_creator(bqf.nuclear_species_list, region_bounds_list, step_size_list, use_sundfors_GET_vals = False)

# EFG_data_creator_fake_strain(bqf.nuclear_species_list, region_bounds_list, step_size_list, use_sundfors_GET_vals = False, faked_strain_type = "left_right")