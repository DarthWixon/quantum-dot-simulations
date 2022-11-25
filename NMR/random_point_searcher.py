import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import time
import os.path
import backbone_quadrupolar_functions as bqf
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import numpy as np
import matplotlib.pyplot as plt
import isotope_parameters as ISOP

def find_best_locations(n_locs, region_bounds_label, search_range, testing = False):
	region_bounds = bqf.QD_regions_dict[region_bounds_label]
	rng = np.random.default_rng()

	xx_data, xz_data, zz_data = bqf.load_sokolov_data(region_bounds)

	if testing:
		data = 100 * rng.random(xz_data.shape)
	else:
		data = xx_data + xz_data + zz_data

	ax_0_range, ax_1_range = data.shape
	ax_0_indices = rng.integers(ax_0_range, size = n_locs)
	ax_1_indices = rng.integers(ax_1_range, size = n_locs)

	loc_list = list(zip(ax_0_indices, ax_1_indices))
	
	# print(loc_list)

	highest_point_list = []

	for loc in loc_list:

		x_coord, y_coord = loc
		x_min = x_coord - search_range
		x_max = x_coord + search_range + 1
		y_min = y_coord - search_range
		y_max = y_coord + search_range + 1


		if x_min < 0:
			x_min = 0
		if x_max >= data.shape[0]:
			x_max = data.shape[0] - 1
		if y_min < 0:
			y_min = 0
		if y_max >= data.shape[1]:
			y_max = data.shape[1] - 1

		search_area = data[x_min:x_max, y_min:y_max]

		relative_area_max_coords = np.unravel_index(np.argmax(search_area, axis = None), search_area.shape)

		area_max_coords = tuple(np.add(np.add(relative_area_max_coords, np.array(loc)), -1*search_range))

		highest_point_list.append(area_max_coords)

	if testing:
		return loc_list, highest_point_list
	else:
		return highest_point_list


def checking_it_works():
	n_locs = 1000
	region_bounds_label = "entire_dot"
	search_range = 10

	x_higher, x_lower, x_equal, y_higher, y_lower, y_equal = 0,0,0,0,0,0

	for i in range(10):
		random_locs, high_points = find_best_locations(n_locs, region_bounds_label, search_range, testing = True)

		for i in range(n_locs):
			if random_locs[i][0] > high_points[i][0]:
				x_higher += 1
			elif random_locs[i][0] < high_points[i][0]:
				x_lower += 1
			elif random_locs[i][0] == high_points[i][0]:
				x_equal +=1

			if random_locs[i][1] > high_points[i][1]:
				y_higher += 1
			elif random_locs[i][1] < high_points[i][1]:
				y_lower += 1
			elif random_locs[i][1] == high_points[i][1]:
				y_equal +=1

		x_tot = x_equal + x_lower + x_higher
		y_tot = y_equal + y_lower + y_higher

	print(f"Random point X coord higher {(x_higher/x_tot)*100}% of the time. It's lower {(x_lower/x_tot)*100}% of the time. They are equal {(x_equal/x_tot)*100}% of the time. {x_tot} points tested.")
	print(f"Random point Y coord higher {(y_higher/y_tot)*100}% of the time. It's lower {(y_lower/y_tot)*100}% of the time. They are equal {(x_equal/y_tot)*100}% of the time. {y_tot} points tested.")


if __name__ == "__main__":
	checking_it_works()



