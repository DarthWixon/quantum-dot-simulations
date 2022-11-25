import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import matplotlib.pyplot as plt
import scipy.constants as constants
import isotope_parameters as ISOP

def eigenvector_calc(applied_field, location = (550, 450), nuclear_species = "In115", field_orientation = "Faraday"):
	x_coord, y_coord = location

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

	if field_orientation == "Faraday":
		Hamiltonian = bqf.structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
		Hamiltonian = Hamiltonian.tidyup()
	elif field_orientation == "Voigt":
		Hamiltonian = bqf.structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, eta, spin, alpha, beta, gamma)
		Hamiltonian = Hamiltonian.tidyup()
	else:
		print(f"Orientation parameter must be either Faraday or Voigt, not {field_orientation}.")
		return np.nan

	eigenstructure = Hamiltonian.eigenstates()

	eigen_vectors = eigenstructure[1]

	return np.real_if_close(eigen_vectors)

nuclear_species = "In115"
region_bounds = [100, 1200, 200, 1000]
h = constants.h
e = constants.e
eta_array, V_XX_array, V_YY_array, V_ZZ_array, euler_angles_array = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds)

for B in np.linspace(0,0.1,5):
	print(eigenvector_calc(B)[1])