import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import neat_strain_toy_model as tm
import numpy as np
import matplotlib.pyplot as plt

max_size = 25
real_atoms = True
lattice_type = 1
In_positions = [[]]
timing_array = np.zeros(max_size-1)

for i in range(2, max_size+1):
	n_rows = i
	n_cols = i
	simulated_species_array, unstrained_positions, strained_positions, time_taken, average_row_difference, average_col_difference = tm.Simulation(n_rows, n_cols, lattice_type, In_positions, real_atoms)
	timing_array[i-2] = time_taken
	print('Time taken for simulation of size {}x{}: {}s'.format(n_rows, n_cols, time_taken))

plt.scatter(list(range(2, max_size+1)), timing_array)
plt.xlabel('Grid Size')
plt.ylabel('Time Taken (s)')
plt.title('Time Taken to Simulate GaAs Lattices With A Central In Atom')
plt.show()
