import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import matplotlib.pyplot as plt
import scipy.constants as constants
import isotope_parameters as ISOP
import time

h = constants.h
e = constants.e

def single_field_energy_levels(applied_field, location, nuclear_species, field_geometry, region_bounds = [100, 1200, 200, 1000]):
	x_coord, y_coord = location

	eta_array, V_XX_array, V_YY_array, V_ZZ_array, euler_angles_array = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

	eta = eta_array[x_coord, y_coord]
	# eta = 0
	V_ZZ = V_ZZ_array[x_coord, y_coord]
	euler_angles = euler_angles_array[x_coord, y_coord]

	species = ISOP.species_dict[nuclear_species]

	spin = species["particle_spin"]
	zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
	Q = species["quadrupole_moment"]
	quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq
	alpha, beta, gamma = euler_angles
	quadrupolar_term = quadrupole_coupling_constant * V_ZZ

	Zeeman_term = zeeman_frequency_per_tesla * applied_field

	parameter_list = [Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma]
	# print(f"Paramters are: {parameter_list}")

	if field_geometry == "Faraday":
		# print("Doing Faraday")
		Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
		# Hamiltonian = Hamiltonian.tidyup()
	elif field_geometry == "Voigt":
		# print("Doing Voigt")
		Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
		# Hamiltonian = Hamiltonian.tidyup()
	else:
		print(f"Orientation parameter must be either Faraday or Voigt, not {field_geometry}.")
		return np.nan

	# print(Hamiltonian)
	eigen_energies = Hamiltonian.eigenenergies()/1e6 #plot them in MHz

	return np.real_if_close(eigen_energies)

def many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, field_geometry):

	n_levels = int((2*ISOP.species_dict[nuclear_species]["particle_spin"]) + 1)

	# print(type(n_levels))
	# print(type(len(applied_field_list)))
	
	energy_level_data = np.zeros((n_levels, len(applied_field_list)))
	for a, applied_field in enumerate(applied_field_list):
		energy_level_data[:, a] = single_field_energy_levels(applied_field, location, nuclear_species, field_geometry)

	return energy_level_data

def single_eta_energy_levels(eta, nuclear_species, field_geometry, applied_field, alpha = 0, beta = 0, gamma = 0, V_ZZ = 5e20):
	# pick some default values for Euler angles for now, not sure if they matter yet
	# default value for V_ZZ is approximately the mean over the standard region

	species = ISOP.species_dict[nuclear_species]

	spin = species["particle_spin"]
	zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
	Q = species["quadrupole_moment"]
	quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq
	quadrupolar_term = quadrupole_coupling_constant * V_ZZ

	Zeeman_term = zeeman_frequency_per_tesla * applied_field # will normally be 0, but can be changed

	if field_geometry == "Faraday":
		Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
		Hamiltonian = Hamiltonian.tidyup()
	elif field_geometry == "Voigt":
		Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
		Hamiltonian = Hamiltonian.tidyup()
	else:
		print(f"Orientation parameter must be either Faraday or Voigt, not {field_geometry}.")
		return np.nan

	eigen_energies = Hamiltonian.eigenenergies()/1e6 #plot them in MHz

	return np.real_if_close(eigen_energies)

def many_eta_energy_levels_calc(eta_list, nuclear_species, field_geometry = "Faraday", applied_field = 0):

	n_levels = int((2*ISOP.species_dict[nuclear_species]["particle_spin"]) + 1)

	# print(type(n_levels))
	# print(type(len(eta_list)))
	
	energy_level_data = np.zeros((n_levels, len(eta_list)))
	for n, eta in enumerate(eta_list):
		energy_level_data[:, n] = single_eta_energy_levels(eta, nuclear_species, field_geometry, applied_field)

	return energy_level_data

def plot_energy_level_diagram(changing_variable_list, energy_level_data, nuclear_species, field_geometry, x_axis_label, saving = False):
	n_levels = energy_level_data.shape[0]
	plt.figure(figsize = (12, 10))

	for i in range(n_levels):
		plt.plot(changing_variable_list, energy_level_data[i, :], color = "black")
		# print(energy_level_data[i,:])


	plt.xlabel(x_axis_label)
	plt.ylabel("Splitting Energy (MHz)")
	plt.title(f"Energy Level Splitting of {nuclear_species} in a {field_geometry} geometry.")

	if saving:
		print("Saving graph of changing: {}".format(x_axis_label))
		plot_title = "energy_level_diagram__of_{}_with_changing_{}_in_{}_orientation".format(nuclear_species, x_axis_label.replace(" ", "_"), field_geometry)
		plt.savefig(f"{graph_path}{plot_title}", bbox_inches = "tight")
		plt.close()
	else:
		plt.show()

def plot_and_save_all_species_EL_diagrams(location, field_geometry, region_bounds = [100, 1200, 200, 1000]):
	applied_field_list = np.linspace(0, 3, 1000)
	eta_list = np.linspace(0, 1, 500)

	for nuclear_species in ["Ga69", "Ga71", "As75", "In115"]:

		full_b_field_data = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, field_geometry)
		plot_energy_level_diagram(applied_field_list, full_b_field_data, nuclear_species, field_geometry, x_axis_label = "Magnetic Field (T)", saving = True)
		plt.close()

		full_eta_data = many_eta_energy_levels_calc(eta_list, nuclear_species, field_geometry)
		plot_energy_level_diagram(eta_list, full_eta_data, nuclear_species, field_geometry, x_axis_label = "Eta", saving = True)
		plt.close()

def plot_single_species_both_geometries_EL_diagrams(nuclear_species, location, region_bounds = [100, 1200, 200, 1000], saving = False):
	applied_field_list = np.linspace(0, 3, 1000)

	species = ISOP.species_dict[nuclear_species]
	spin = species["particle_spin"]

	full_faraday_data = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, field_geometry = "Faraday")
	full_voigt_data = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, field_geometry = "Voigt")

	min_faraday = np.amin(full_faraday_data)
	max_faraday = np.amax(full_faraday_data)
	min_voigt = np.amin(full_voigt_data)
	max_voigt = np.amax(full_voigt_data)

	min_range = np.amin([min_faraday, min_voigt])
	max_range = np.amax([max_faraday, max_voigt])

	y_max = 1.2 * max_range
	y_min = 1.2 * min_range

	fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 8))

	n_levels = int((2 * spin) + 1)

	for i in range(n_levels):
		ax1.plot(applied_field_list, full_faraday_data[i, :], color = "black")
		ax2.plot(applied_field_list, full_voigt_data[i, :], color = "black")

	ax1.set_xlabel("Applied Field (T)")
	ax2.set_xlabel("Applied Field (T)")

	ax1.set_ylabel("Frequency (MHz)")
	ax2.set_ylabel("Frequency (MHz)")

	ax1.set_ylim(y_min, y_max)
	ax2.set_ylim(y_min, y_max)

	ax1.set_title("Faraday Orientation")
	ax2.set_title("Voigt Orientation")

	ax1.text(0.1, 0.9, "a", transform = ax1.transAxes)
	ax2.text(0.1, 0.9, "b", transform = ax2.transAxes)

	if saving:
		plot_title = f"{graph_path}single_{nuclear_species}_both_orientations_energy_level_structure.png"
		plt.savefig(plot_title, bbox_inches = "tight")
	else:
		plt.show()


def all_sites_energy_level_diagram(nuclear_species, field_geometry, min_field = 0, max_field = 2, n_fields = 100, region_bounds = [100, 1200, 200, 1000], saving = False):
	applied_field_list = np.linspace(min_field, max_field, n_fields)

	eta_array, V_XX_array, V_YY_array, V_ZZ_array, euler_angles_array = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

	species = ISOP.species_dict[nuclear_species]

	spin = species["particle_spin"]
	zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
	Q = species["quadrupole_moment"]
	quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq

	x_dim, y_dim = eta_array.shape

	n_levels = int((2*ISOP.species_dict[nuclear_species]["particle_spin"]) + 1)

	data = np.zeros((x_dim, y_dim, n_levels, len(applied_field_list)))

	for x in range(x_dim):
		for y in range(y_dim):
			for b, applied_field in enumerate(applied_field_list):
				x_coord, y_coord = x,y

				eta = eta_array[x_coord, y_coord]
				V_ZZ = V_ZZ_array[x_coord, y_coord]
				alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

				quadrupolar_term = quadrupole_coupling_constant * V_ZZ
				Zeeman_term = zeeman_frequency_per_tesla * applied_field

				parameter_list = [Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma]
				# print(f"Paramters are: {parameter_list}")

				if field_geometry == "Faraday":
					Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
				elif field_geometry == "Voigt":
					Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)

				eigen_energies = Hamiltonian.eigenenergies()/1e6 #plot them in MHz
				data[x,y,:,b] =  np.real_if_close(eigen_energies)

	plt.figure(figsize = (12, 10))

	for x in range(x_dim):
		for y in range(y_dim):
			for i in range(n_levels):
				plt.plot(applied_field_list, data[x,y,i,:], color = "black", linewidth = 0.5)
				# print(energy_level_data[i,:])


	plt.xlabel("Applied B Field")
	plt.ylabel("Splitting Energy (MHz)")
	plt.title(f"Energy Level Splitting of {nuclear_species} in a {field_geometry} geometry.")

	if saving:
		plot_title = "all_sites_energy_level_diagram__of_{}_with_changing_b_field_in_{}_orientation_over_region_{}".format(nuclear_species, field_geometry, region_bounds)
		plt.savefig(f"{graph_path}{plot_title}")
		plt.close()
	else:
		plt.show()

def random_sites_energy_level_diagram(nuclear_species, field_geometry, n_sites, min_field = 0, max_field = 2, n_fields = 100, region_bounds = [100, 1200, 200, 1000], saving = False):
	applied_field_list = np.linspace(min_field, max_field, n_fields)

	eta_array, V_XX_array, V_YY_array, V_ZZ_array, euler_angles_array = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)
	
	x_dim, y_dim = eta_array.shape
	rng = np.random.default_rng()
	x_points = rng.choice(x_dim, n_sites, replace = False)
	y_points = rng.choice(y_dim, n_sites, replace = False)
	points = list(zip(x_points, y_points))

	species = ISOP.species_dict[nuclear_species]
	spin = species["particle_spin"]
	zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
	Q = species["quadrupole_moment"]
	quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq

	n_levels = int((2*ISOP.species_dict[nuclear_species]["particle_spin"]) + 1)

	data = np.zeros((n_sites, n_levels, len(applied_field_list)))

	for n in range(n_sites):
		x_coord, y_coord = points[n]
		for b, applied_field in enumerate(applied_field_list):

			eta = eta_array[x_coord, y_coord]
			V_ZZ = V_ZZ_array[x_coord, y_coord]
			alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

			quadrupolar_term = quadrupole_coupling_constant * V_ZZ
			Zeeman_term = zeeman_frequency_per_tesla * applied_field

			if field_geometry == "Faraday":
				Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
			elif field_geometry == "Voigt":
				Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)

			eigen_energies = Hamiltonian.eigenenergies()/1e6 #plot them in MHz
			data[n,:,b] =  np.real_if_close(eigen_energies)

	plt.figure(figsize = (12, 10))

	for n in range(n_sites):
		for i in range(n_levels):
			plt.plot(applied_field_list, data[n,i,:], color = "black", linewidth = 0.5)

	plt.xlabel("Applied B Field")
	plt.ylabel("Splitting Energy (MHz)")
	plt.title(f"Energy Level Splitting of {nuclear_species} in a {field_geometry} geometry.")

	if saving:
		plot_title = "random_sites_energy_level_diagram__of_{}_with_changing_b_field_in_{}_orientation_over_region_{}".format(nuclear_species, field_geometry, region_bounds)
		plt.savefig(f"{graph_path}{plot_title}")
		plt.close()
	else:
		plt.show()

def random_sites_geometry_comparison_EL_diagram(nuclear_species, n_sites, min_field = 0, max_field = 2, n_fields = 100, region_bounds = [100, 1200, 200, 1000], saving = False):
	applied_field_list = np.linspace(min_field, max_field, n_fields)

	eta_array, V_XX_array, V_YY_array, V_ZZ_array, euler_angles_array = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)
	
	x_dim, y_dim = eta_array.shape
	rng = np.random.default_rng()
	x_points = rng.choice(x_dim, n_sites, replace = False)
	y_points = rng.choice(y_dim, n_sites, replace = False)
	points = list(zip(x_points, y_points))

	species = ISOP.species_dict[nuclear_species]
	spin = species["particle_spin"]
	zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
	Q = species["quadrupole_moment"]
	quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq

	n_levels = int((2 * spin) + 1)

	full_faraday_data = np.zeros((n_sites, n_levels, len(applied_field_list)))
	full_voigt_data = np.zeros((n_sites, n_levels, len(applied_field_list)))

	for n in range(n_sites):
		x_coord, y_coord = points[n]
		for b, applied_field in enumerate(applied_field_list):

			eta = eta_array[x_coord, y_coord]
			V_ZZ = V_ZZ_array[x_coord, y_coord]
			alpha, beta, gamma = euler_angles_array[x_coord, y_coord]

			quadrupolar_term = quadrupole_coupling_constant * V_ZZ
			Zeeman_term = zeeman_frequency_per_tesla * applied_field

			Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
			eigen_energies = Hamiltonian.eigenenergies()/1e6 #plot them in MHz
			full_faraday_data[n,:,b] = np.real_if_close(eigen_energies)


			Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
			eigen_energies = Hamiltonian.eigenenergies()/1e6 #plot them in MHz
			full_voigt_data[n,:,b] = np.real_if_close(eigen_energies)

	min_faraday = np.amin(full_faraday_data)
	max_faraday = np.amax(full_faraday_data)
	min_voigt = np.amin(full_voigt_data)
	max_voigt = np.amax(full_voigt_data)

	min_range = np.amin([min_faraday, min_voigt])
	max_range = np.amax([max_faraday, max_voigt])

	y_max = 1.2 * max_range
	y_min = 1.2 * min_range


	fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 8))

	for n in range(n_sites):
		for i in range(n_levels):
			ax1.plot(applied_field_list, full_faraday_data[n, i, :], color = "black")
			ax2.plot(applied_field_list, full_voigt_data[n, i, :], color = "black")

	ax1.set_xlabel("Applied Field (T)")
	ax2.set_xlabel("Applied Field (T)")

	ax1.set_ylabel("Frequency (MHz)")
	ax2.set_ylabel("Frequency (MHz)")

	ax1.set_title("Faraday Orientation")
	ax2.set_title("Voigt Orientation")

	ax1.set_ylim(y_min, y_max)
	ax2.set_ylim(y_min, y_max)

	ax1.text(0.1, 0.9, "a", transform = ax1.transAxes)
	ax2.text(0.1, 0.9, "b", transform = ax2.transAxes)

	if saving:
		plot_title = f"{graph_path}random_sample_of_{nuclear_species}_both_orientations_energy_level_structure.png"
		plt.savefig(plot_title, bbox_inches = "tight")
	else:
		plt.show()


# def plot_faraday_voigt_comparison_B_field(nuclear_species, saving = False):
# 	location = (450, 500)
# 	applied_field_list = np.linspace(0, 3, 1000)
# 	faraday_b_field_data = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, "Faraday")
# 	voigt_b_field_data = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, "Voigt")

# 	n_levels = faraday_b_field_data.shape[0]

# 	for i in range(n_levels):
# 		plt.plot(applied_field_list, faraday_b_field_data[i,:], color = "black")
# 		plt.plot(applied_field_list, voigt_b_field_data[i,:], color = "red")



# nuclear_species = "In115"
# region_bounds = [100, 1200, 200, 1000]
# field_geometry = "Voigt" # or "Faraday" or "Voigt"
# eta_array, V_XX_array, V_YY_array, V_ZZ_array, euler_angles_array = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

# print(np.mean(eta_array))

# these are the coordinates of the point we want, as found in the
# region bounds array (therefore NOT the same format as that array)
# location = (500, 450) # (0,0) is the first element of the region_bounds array, regardless of it's size

# applied_field_list = np.linspace(0, 3, 1000)
# eta_list = np.linspace(0, 1, 500)

# n_sites = 100

# full_b_field_data = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, field_geometry)
# plot_energy_level_diagram(applied_field_list, full_b_field_data, nuclear_species, field_geometry, x_axis_label = "Magnetic Field (T)", saving = False)
# plt.close()

# full_eta_data = many_eta_energy_levels_calc(eta_list, nuclear_species, field_geometry)
# plot_energy_level_diagram(eta_list, full_eta_data, nuclear_species, field_geometry, x_axis_label = "Eta", saving = True)

# field_geometry = "Faraday"
# full_b_field_data_faraday = many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, "Faraday")
# plot_energy_level_diagram(applied_field_list, full_b_field_data_faraday, nuclear_species, "Faraday", x_axis_label = "Magnetic Field (T)", saving = False)

# field_geometry = "Voigt"
# full_b_field_data_voigt= many_fields_energy_levels_calc(applied_field_list, location, nuclear_species, "Voigt")
# plot_energy_level_diagram(applied_field_list, full_b_field_data_voigt, nuclear_species, "Voigt", x_axis_label = "Magnetic Field (T)", saving = False)

# plot_and_save_all_species_EL_diagrams((500, 450), "Faraday")
# plot_and_save_all_species_EL_diagrams((500, 450), "Voigt")

# for nuclear_species in ["Ga69", "Ga71", "As75", "In115"]:
# 	bqf.calculate_and_save_EFG(nuclear_species, region_bounds = [100,1200, 439, 880], step_size = 1)
# 	for field_geometry in ["Faraday", "Voigt"]:
# 		# all_sites_energy_level_diagram(nuclear_species, field_geometry, region_bounds = [400, 405, 500, 505], saving = True)
# 		random_sites_energy_level_diagram(nuclear_species, field_geometry, n_sites, region_bounds = [100,1200, 439, 880], saving = True)


# applied_field = 1
# location = (500, 450)
# nuclear_species = "Ga69"
# field_geometry = "Faraday"

# single_field_energy_levels(applied_field, location, nuclear_species, field_geometry, region_bounds = [100, 1200, 200, 1000])

nuclear_species = "In115"
location = (500, 450)
field_geometry = "Faraday"
n_sites = 25
# plot_single_species_both_geometries_EL_diagrams(nuclear_species, location, region_bounds = [100, 1200, 200, 1000], saving = True)

random_sites_geometry_comparison_EL_diagram(nuclear_species, n_sites, min_field = 0, max_field = 2, n_fields = 100, region_bounds = [100, 1200, 200, 1000], saving = True)

for nuclear_species in bqf.nuclear_species_list:
	plot_single_species_both_geometries_EL_diagrams(nuclear_species, location, region_bounds = [100, 1200, 200, 1000], saving = True)
	random_sites_geometry_comparison_EL_diagram(nuclear_species, n_sites, min_field = 0, max_field = 3, n_fields = 100, region_bounds = [100, 1200, 200, 1000], saving = True)