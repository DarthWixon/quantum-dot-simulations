import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import time
from scipy.spatial import distance_matrix
import math
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path

def ManyParticlePositionToLengthConverter(coords_array, box_width, box_height, n_rows, n_cols):
	# pass in an array of atomic positions, returns an array of spring lengths, labelled as described in my 2nd lab book
	# coords_array is of shape m x n x 2 (n_rows x n_cols x (x,y))
	coords_array = np.reshape(coords_array, (n_rows, n_cols, 2))
	n_rows = coords_array.shape[0]
	n_cols = coords_array.shape[1]
	n_springs = 2*n_rows*n_cols + n_rows + n_cols

	length_list = np.zeros(n_springs) # derivation of this number of springs uses grid graphs and is in my 2nd lab book
	length_list = length_list + 1000 # check to see if I've missed any, they should have a ridiculous value if so

	# I define the origin of the box to be the top left hand corner, hence the max height is reached at the "bottom" of a drawing of the box
	# This is only true for the guts of the program, the Indium placing code defined later works from the bottom left, as does the plotting (12/11/19)

	for row in range(n_rows):
		for col in range(n_cols):
			# all derived in my 2nd lab book, have fun future me
			above_spring_index = 2*(n_rows+1)*col + 2*row
			below_spring_index = above_spring_index + 2
			left_spring_index = 2*(n_cols + 1)*row + 2*col + 1
			right_spring_index = left_spring_index + 2

			x_pos = coords_array[row, col, 0]
			y_pos = coords_array[row, col, 1]

			if row == 0: #check if we're on the first row of atoms
				top_y_start = 0
				if col == 0:
					left_x_start = 0
				else:
					left_x_start = coords_array[row, col-1, 0]
				top_x_start = (col+1) * box_width/(n_cols+1)
				left_y_start = (row+1) * box_height/(n_rows+1)

			else:
				top_y_start = coords_array[row - 1, col, 1]
				if col == 0:
					left_x_start = 0
				else:
					left_x_start = coords_array[row, col-1, 0]
				top_x_start = coords_array[row - 1, col, 0]
				left_y_start = coords_array[row, col - 1, 1]

			length_list[above_spring_index] = np.sqrt((x_pos - top_x_start)**2 + (y_pos - top_y_start)**2)
			length_list[left_spring_index] = np.sqrt((x_pos - left_x_start)**2 + (y_pos - left_y_start)**2)

			if row == n_rows - 1: #check if we're on the last row of atoms
				bottom_y_start = box_height
				bottom_x_start = (col+1) * box_width/(n_cols+1)

				length_list[below_spring_index] = np.sqrt((x_pos - bottom_x_start)**2 + (y_pos - bottom_y_start)**2)

			if col == n_cols - 1: #check if we're on the last column of atoms
				right_y_start = (row+1) * box_height/(n_rows+1)
				right_x_start = box_width

				length_list[right_spring_index] = np.sqrt((x_pos - right_x_start)**2 + (y_pos - right_y_start)**2)

	return length_list

def SpringEnergyFunction(length_list, constant_dict, natural_length_dict):
	potential = 0
	for i in range(len(length_list)):
		k_name = 'k{}'.format(i)
		n_name = 'n{}'.format(i)

		k = constant_dict[k_name]
		n = natural_length_dict[n_name]
		l = length_list[i]

		potential += k*(l-n)**2
	return potential

def ManyParticlePositionEnergyFunction(coords_array, spring_constant_list, natural_length_list, box_width, box_height, n_rows, n_cols):
	potential = 0
	length_list = ManyParticlePositionToLengthConverter(coords_array, box_width, box_height, n_rows, n_cols)

	for i in range(len(length_list)):
		potential += spring_constant_list[i] * (length_list[i] - natural_length_list[i])**2

	return potential

def ConnectionsPlotter(coords_array, colour, linestyle, alpha, simulated_species_array):
	n_rows = coords_array.shape[0]
	n_cols = coords_array.shape[1]

	box_width = n_rows + 2
	box_height = n_rows + 2

	for row in range(n_rows):
		for col in range(n_cols):
			x_base, y_base = coords_array[row,col,0], coords_array[row,col,1]
			if row + 1 == n_rows:
				y_above = box_height
				x_above = (col+1) * box_width/(n_cols+1)
			else:
				x_above = coords_array[row+1, col, 0]
				y_above = coords_array[row+1, col, 1]
			if col + 1 == n_cols:
				y_right = (row+1) * box_height/(n_rows+1)
				x_right = box_width
			else:
				x_right = coords_array[row,col+1, 0]
				y_right = coords_array[row,col+1, 1]
			if row == 0:
				y_below = 0
				x_below = (col+1) * box_width/(n_cols+1)
				plt.plot([x_base, x_below], [y_base, y_below], linestyle = linestyle, c = colour, alpha = alpha)
			if col == 0:
				y_left = (row+1) * box_height/(n_rows+1)
				x_left = 0
				plt.plot([x_base, x_left], [y_base, y_left], linestyle = linestyle, c = colour, alpha = alpha)

			plt.plot([x_base, x_right], [y_base, y_right], linestyle = linestyle, c = colour, alpha = alpha)
			plt.plot([x_base, x_above], [y_base, y_above], linestyle = linestyle, c = colour, alpha = alpha)

def SpringConstantFromAtomicSpecies(large_species_array, real_atoms = True):
	# species array has 0 = Ga, 1 = As, 2 = In
	# species array is also 2 rows and 2 columns bigger than the base grid
	# this allows me to populate the outer edge with unseen atoms
	# normally they will all be of the same type
	bond_strength_dict, natural_length_dict = RealOrFakeAtoms(real_atoms)

	n_rows = large_species_array.shape[0]
	n_cols = large_species_array.shape[1]
	
	small_n_rows = n_rows - 2
	small_n_cols = n_cols - 2

	spring_constant_list = np.zeros(2*small_n_rows*small_n_cols + small_n_rows + small_n_cols)

	for row in range(1, n_rows-1):
		for col in range(1, n_cols-1):
			small_row = row - 1
			small_col = col - 1
			own_species = large_species_array[row, col]

			above_species = large_species_array[row-1, col]
			below_species = large_species_array[row+1 , col]
			left_species = large_species_array[row, col-1 ]
			right_species = large_species_array[row, col+1]

			above_bond_index = '{}{}'.format(own_species, above_species)
			below_bond_index = '{}{}'.format(own_species, below_species)
			left_bond_index = '{}{}'.format(own_species, left_species)
			right_bond_index = '{}{}'.format(own_species, right_species)

			above_spring_index = 2*(small_n_rows+1)*small_col + 2*small_row
			below_spring_index = above_spring_index + 2
			left_spring_index = 2*(small_n_cols + 1)*small_row + 2*small_col + 1
			right_spring_index = left_spring_index + 2

			spring_constant_list[above_spring_index] = bond_strength_dict[above_bond_index]
			spring_constant_list[below_spring_index] = bond_strength_dict[below_bond_index]
			spring_constant_list[left_spring_index] = bond_strength_dict[left_bond_index]
			spring_constant_list[right_spring_index] = bond_strength_dict[right_bond_index]

	return spring_constant_list

def NaturalLengthFromAtomicSpecies(large_species_array, real_atoms = True):
	# species array has 0 = Ga, 1 = As, 2 = In
	# species array is also 2 rows and 2 columns bigger than the base grid
	# this allows me to populate the outer edge with unseen atoms
	# normally they will all be of the same type
	bond_strength_dict, natural_length_dict = RealOrFakeAtoms(real_atoms)

	n_rows = large_species_array.shape[0]
	n_cols = large_species_array.shape[1]

	small_n_rows = n_rows - 2
	small_n_cols = n_cols - 2

	natural_length_list = np.zeros(2*small_n_rows*small_n_cols + small_n_rows + small_n_cols)

	for row in range(1, n_rows-1):
		for col in range(1, n_cols-1):
			small_row = row - 1
			small_col = col - 1
			own_species = large_species_array[row, col]

			above_species = large_species_array[row-1, col]
			below_species = large_species_array[row+1, col]
			left_species = large_species_array[row, col - 1]
			right_species = large_species_array[row, col + 1]

			above_bond_index = '{}{}'.format(own_species, above_species)
			below_bond_index = '{}{}'.format(own_species, below_species)
			left_bond_index = '{}{}'.format(own_species, left_species)
			right_bond_index = '{}{}'.format(own_species, right_species)

			above_spring_index = 2*(small_n_rows+1)*small_col + 2*small_row
			below_spring_index = above_spring_index + 2
			left_spring_index = 2*(small_n_cols + 1)*small_row + 2*small_col + 1
			right_spring_index = left_spring_index + 2

			natural_length_list[above_spring_index] = natural_length_dict[above_bond_index]
			natural_length_list[below_spring_index] = natural_length_dict[below_bond_index]
			natural_length_list[left_spring_index] = natural_length_dict[left_bond_index]
			natural_length_list[right_spring_index] = natural_length_dict[right_bond_index]

	return natural_length_list

def ColourFromAtomicSpecies(simulated_species_array):
	# returns a flattened array as that's what plt.scatter needs to colour things
	# Ga is red
	# As is blue
	# In is green
	n_rows = simulated_species_array.shape[0]
	n_cols = simulated_species_array.shape[1]
	colour_array = np.empty((n_rows, n_cols), dtype = str)

	for row in range(n_rows):
		for col in range(n_cols):
			if simulated_species_array[row,col] == 0:
				colour_array[row,col] = 'r'
			elif simulated_species_array[row,col] == 1:
				colour_array[row,col] = 'b'
			elif simulated_species_array[row,col] == 2:
				colour_array[row,col] = 'g'

	return colour_array.flatten()

def StrainTensorCalculator(strained_coords_array, unstrained_coords_array):
	n_rows = strained_coords_array.shape[0]
	n_cols = strained_coords_array.shape[1]

	strain_tensor_array = np.zeros((n_rows, n_cols, 2, 2)) # 2x2 because the strain tensor is a 2x2 matrix

	loop_count = 0
	for row in range(n_rows):
		for col in range(n_cols):
			loop_count += 1
			W_matrix = np.zeros((2,2))
			V_matrix = np.zeros((2,2))

			base_x_unstrained = unstrained_coords_array[row, col, 0]
			base_y_unstrained = unstrained_coords_array[row, col, 1]
			base_x_strained = strained_coords_array[row, col, 0]
			base_y_strained = strained_coords_array[row, col, 1]

			for target_row in range(n_rows):
				for target_col in range(n_cols):

					target_x_unstrained = unstrained_coords_array[target_row, target_col, 0]
					target_y_unstrained = unstrained_coords_array[target_row, target_col, 1]

					target_x_strained = strained_coords_array[target_row, target_col, 0]
					target_y_strained = strained_coords_array[target_row, target_col, 1]

					unstrained_difference = np.array([base_x_unstrained - target_x_unstrained, base_y_unstrained - target_y_unstrained])
					strained_difference = np.array([base_x_strained - target_x_strained, base_y_strained - target_y_strained])

					W_matrix += np.tensordot(strained_difference, unstrained_difference, axes = 0)
					V_matrix += np.tensordot(unstrained_difference, unstrained_difference, axes = 0)
			
			F_matrix = np.dot(W_matrix, np.linalg.inv(V_matrix))
			grad_U = F_matrix - np.identity(2)
			strain_tensor_array[row, col] = 0.5*(grad_U + np.transpose(grad_U))
	return strain_tensor_array

def StrainPlotter(strain_tensor_array, absolute_vales = False, saving = True):
	if absolute_vales:
		strain_tensor_array = np.absolute(strain_tensor_array)

	n_rows = strain_tensor_array.shape[0]
	n_cols = strain_tensor_array.shape[1]

	fig, axs = plt.subplots(1, 3, constrained_layout = True)

	vmin = np.amin(strain_tensor_array)
	vmax = np.amax(strain_tensor_array)

	normalizer = Normalize(vmin, vmax)

	im = cm.ScalarMappable(cmap = cm.GnBu, norm = normalizer)

	# plot the Exx terms
	ax = axs.flatten()[0]
	ax.imshow(strain_tensor_array[:,:,0,0], origin = 'lower', vmin = vmin, vmax = vmax, cmap = cm.GnBu)
	ax.set_title("$\epsilon_{xx}$")
	ax.axis("off")

	# plot the Exy terms
	ax = axs.flatten()[1]
	ax.imshow(strain_tensor_array[:,:,0,1], origin = 'lower', vmin = vmin, vmax = vmax, cmap = cm.GnBu)
	ax.set_title("$\epsilon_{xy}$")
	ax.axis("off")

	# plot the Eyy terms
	ax = axs.flatten()[2]
	ax.imshow(strain_tensor_array[:,:,1,1], origin = 'lower', vmin = vmin, vmax = vmax, cmap = cm.GnBu)
	ax.set_title("$\epsilon_{yy}$")
	ax.axis("off")

	plt.colorbar(im, ax = axs.ravel().tolist(), orientation = 'horizontal', shrink = 0.95)

	if saving:
		plot_name = f"{graph_path}change_the_name_of_this_strain_plot.png"
		plt.savefig(plot_name, bbox_inches = "tight")

	return

def RowAndColLengthFinder(length_list, n_rows, n_cols):
	row_lengths = np.zeros(n_rows)
	col_lengths = np.zeros(n_cols)

	for row in range(n_rows):
		start_index = 1 + 2*row*n_cols # +1 because rows start with odd numbers
		end_index = start_index + 2*n_cols + 1 # plus 1 to make sure it includes the last bond
		row_lengths[row] = np.sum(length_list[start_index: end_index: 2])

	for col in range(n_cols):
		start_index = 2*col*n_rows
		end_index = start_index + 2*n_rows + 1 # plus 1 to make sure it includes the last bond
		col_lengths[col] = np.sum(length_list[start_index: end_index: 2])

	return row_lengths, col_lengths

def GaAsEqualLatticeGenerator(n_rows, n_cols, used_by_other_func = False):
	large_species_array = np.zeros((n_rows + 2, n_cols + 2), dtype = int)

	for row in range(n_rows+2):
		for col in range(n_cols+2):
			if row % 2 == 0 and col % 2 == 0:
				large_species_array[row, col] = 1
			elif row %2 != 0 and col %2 != 0:
				large_species_array[row, col] = 1


			# quick and dirty way of swapping the Ga and As around, as I had them the wrong way
			if large_species_array[row, col] == 0:
				large_species_array[row, col] = 1
			elif large_species_array[row, col] == 1:
				large_species_array[row, col] = 0

	if used_by_other_func:
		return large_species_array
	else:
		simulated_species_array = large_species_array[1:n_rows+1, 1:n_cols+1]
		return large_species_array, simulated_species_array

def GaAsLatticeWithIndiumCentre(n_rows, n_cols):
	large_species_array = GaAsEqualLatticeGenerator(n_rows, n_cols, True)

	for row in range(n_rows+2):
		for col in range(n_cols+2):
			if row == int(math.floor(n_rows/2)) + 1 and col == int(math.floor(n_cols/2)) + 1 and large_species_array[row, col] == 0:
				large_species_array[row, col] = 2

	simulated_species_array = large_species_array[1:n_rows+1, 1:n_cols+1]
	return large_species_array, simulated_species_array

def GaAsLatticeWithScatteredIndium(n_rows, n_cols, In_positions):
	large_species_array = GaAsEqualLatticeGenerator(n_rows, n_cols, True)

	for lattice_point in In_positions:
		x_coord = lattice_point[0] + 1
		y_coord = lattice_point[1] + 1

		# Ok future Will, this looks weird on the face of it I admit.
		# Throughout this code, we always loop over n_rows and then n_cols.
		# However, I just realised that when we look at the output,
		# which row we're in is the Y coordinate, not the X coordinate.
		# So instead of rewriting all of the loops to take that into account,
		# I've just changed where we put the In atoms in instead.
		# So we count from bottom left to top right of the grid to place In atoms,
		# Even though simulated_species_array is layed out differently, that's because of
		# how the plotting functions work. Ignore simulated_species_array unless you're 
		# changing the guts, which hopefully you won't have to do. Fingers crossed.
		if large_species_array[y_coord, x_coord] == 0:
			large_species_array[y_coord, x_coord] = 2

	simulated_species_array = large_species_array[1:n_rows+1, 1:n_cols+1]
	return large_species_array, simulated_species_array

def BlockInPlacer(bottom_left_coords, top_right_coords):
	In_positions = []
	for i in range(bottom_left_coords[0], top_right_coords[0]):
		for j in range(bottom_left_coords[1], top_right_coords[1]):
			In_positions.append([i,j])
	return In_positions

def InPositionGenerator(lattice_type):
	if lattice_type == 0 or lattice_type == 1:
		return [[]]
	elif lattice_type == 2:
		return BlockInPlacer(bottom_left_coords, top_right_coords)

def LatticeGenerator(n_rows, n_cols, lattice_type = 0, In_positions = [[]]):
	if lattice_type == 0:
		large_species_array, simulated_species_array = GaAsEqualLatticeGenerator(n_rows, n_cols)
	elif lattice_type == 1:
		large_species_array, simulated_species_array = GaAsLatticeWithIndiumCentre(n_rows, n_cols)
	elif lattice_type == 2:
		large_species_array, simulated_species_array = GaAsLatticeWithScatteredIndium(n_rows, n_cols, In_positions)

	return large_species_array, simulated_species_array

def UnstrainedPositionsGenerator(n_rows, n_cols):
	box_height = n_rows + 2
	box_width = n_cols + 2
	coords_array = np.zeros((n_rows, n_cols, 2))
	for row in range(n_rows):
		for col in range(n_cols):
			x_pos = (col+1) * box_width/(n_cols+1)
			y_pos = (row+1) * box_height/(n_rows+1)
			coords_array[row,col,0] = x_pos
			coords_array[row,col,1] = y_pos
	return coords_array

def RealOrFakeAtoms(real_atoms):
	# band parameters taken from Vurgaftman 2001: https://doi-org.bris.idm.oclc.org/10.1063/1.1368156
	if real_atoms:
		bond_strength_dict = {
					'00': 1,
					'01': 2.56, 
					'02': 1.22,
					'10': 2.56,
					'11': 2.99,
					'12': 2.30,
					'20': 1.22,
					'21': 2.30,
					'22': 1.58
					}

		natural_length_dict = {
					'00': 1,
					'01': 1, 
					'02': 1,
					'10': 1,
					'11': 1,
					'12': 1.06,
					'20': 1,
					'21': 1.06,
					'22': 1
					}
	else:
		bond_strength_dict = {
					'00': 1,
					'01': 5, 
					'02': 50,
					'10': 5,
					'11': 10,
					'12': 1000,
					'20': 50,
					'21': 1000,
					'22': 200
					}

		natural_length_dict = {
					'00': 1,
					'01': 1, 
					'02': 1,
					'10': 1,
					'11': 1,
					'12': 0.1,
					'20': 1,
					'21': 0.1,
					'22': 1
					}
	return bond_strength_dict, natural_length_dict

def Plotter(simulated_species_array, unstrained_positions, strained_positions, time_taken, average_row_difference, average_col_difference, showing_plots = True, real_atoms = False):
	box_width = simulated_species_array.shape[0] + 2
	box_height = simulated_species_array.shape[1] + 2

	colour_array = ColourFromAtomicSpecies(simulated_species_array).flatten()

	ConnectionsPlotter(unstrained_positions, 'black', 'dashed', 0.3, simulated_species_array)
	plt.scatter(unstrained_positions[:,:,0], unstrained_positions[:,:,1], c = colour_array, alpha = 0.5)
	ConnectionsPlotter(strained_positions, 'black', 'dashed', 0.8, simulated_species_array)
	plt.scatter(strained_positions[:,:,0], strained_positions[:,:,1], c = colour_array, alpha = 1)

	# plt.title('Time Taken: {}s, Average Row Difference: {}%, Average Col Difference: {}%'.format(time_taken, average_row_difference, average_col_difference))
	if not real_atoms:
		legend_elements = [Line2D([0],[0], color = 'black', linestyle = 'dashed', label = 'Unstrained Lattice', alpha = 0.3),
						   Line2D([0],[0], color = 'black', linestyle = 'dashed', label = 'Strained Lattice', alpha = 0.8),
						   ]
	else:	
		legend_elements = [Line2D([0],[0], color = 'black', linestyle = 'dashed', label = 'Unstrained Lattice', alpha = 0.3),
						   Line2D([0],[0], color = 'black', linestyle = 'dashed', label = 'Strained Lattice', alpha = 0.8),
						   Line2D([0],[0], color = 'w', markerfacecolor = 'r', marker = 'o', label = 'Ga Atom'),
						   Line2D([0],[0], color = 'w', markerfacecolor = 'b', marker = 'o', label = 'As Atom'),
						   Line2D([0],[0], color = 'w', markerfacecolor = 'g', marker = 'o', label = 'In Atom')]

	plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

	plt.xlim(0, box_width)
	plt.ylim(0, box_height)
	# plt.xlabel('X Position')
	# plt.ylabel('Y Position')
	plt.legend(handles = legend_elements)

	# set xticks and yticks to an empty list to make them vanish
	plt.xticks([]) 
	plt.yticks([])

	plt.tight_layout()

	strain_tensor_array = StrainTensorCalculator(strained_positions, unstrained_positions)
	StrainPlotter(strain_tensor_array, False)

	if showing_plots:
		plt.show()

def Simulation(n_rows, n_cols, lattice_type, In_positions = [[]], real_atoms = True):
	box_height = n_rows + 2
	box_width = n_cols + 2

	n_particles = n_rows*n_cols
	n_springs = 2*n_rows*n_cols + n_rows + n_cols

	large_species_array, simulated_species_array = LatticeGenerator(n_rows, n_cols, lattice_type, In_positions)

	spring_constant_list = SpringConstantFromAtomicSpecies(large_species_array, real_atoms)
	natural_length_list = NaturalLengthFromAtomicSpecies(large_species_array, real_atoms)

	unstrained_positions = UnstrainedPositionsGenerator(n_rows, n_cols)

	start = time.time()
	result = opt.minimize(ManyParticlePositionEnergyFunction, unstrained_positions, args = (spring_constant_list, natural_length_list, box_width, box_height, n_rows, n_cols))
	end = time.time()
	time_taken = np.around(end - start, decimal_places)

	strained_positions = np.reshape(result.x, (n_rows, n_cols, 2))

	length_list = ManyParticlePositionToLengthConverter(strained_positions, box_width, box_height, n_rows, n_cols)
	row_lengths, col_lengths = RowAndColLengthFinder(length_list, n_rows, n_cols)
	average_row_difference = np.around((np.sqrt(np.sum(np.square(row_lengths - box_width)))/n_rows)*100/box_width, decimal_places)
	average_col_difference = np.around((np.sqrt(np.sum(np.square(col_lengths - box_height)))/n_cols)*100/box_height, decimal_places)

	return simulated_species_array, unstrained_positions, strained_positions, time_taken, average_row_difference, average_col_difference

# -------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	# paremeters for the simulation
	# lattice types: 0 = GaAs base lattice, 1 = central In, 2 = Indium in a square block somewhere
	# n_rows and n_cols are the number of visible rows and columns
	# bottom_left_coords and top_right_coords are paremeters for sub-blocks of pure InAs, counting from the bottom left of the simulated lattice

	decimal_places = 3
	real_atoms = True
	showing_plots = False
	n_rows = 29
	n_cols = 29
	lattice_type = 2
	if lattice_type == 2:
		bottom_left_coords = [9,9]
		top_right_coords = [19,19]

	# -------------------------------------------------------------------------------------------------------------------------------------------------------

	In_positions = InPositionGenerator(lattice_type)
	simulated_species_array, unstrained_positions, strained_positions, time_taken, average_row_difference, average_col_difference = Simulation(n_rows, n_cols, lattice_type, In_positions, real_atoms)
	Plotter(simulated_species_array, unstrained_positions, strained_positions, time_taken, average_row_difference, average_col_difference, showing_plots, real_atoms)
	print('Time taken: {}s'.format(time_taken))

else:
	decimal_places = 3