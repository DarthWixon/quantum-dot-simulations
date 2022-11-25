"""
Functions to do the Quantum Mechanics work for Will Dixon's PhD.

These functions heaviliy utilise the Quantum Toolbox in Python (QuTiP) library,
along with Numpy.

"""

import numpy as np
import qutip
import isotope_parameters as ISOP
import os.path

graph_path = "/home/will/Documents/phd/research/simulations/graphs/"
data_path = "/home/will/Documents/phd/research/simulations/data/"
nuclear_species_list = ["Ga69", "Ga71", "As75", "In115"]
field_geometries = ["Faraday", "Voigt"]
QD_regions_dict = {
                            "central_high_In": [450, 850, 550, 650],
                            "central_low_In": [450, 850, 660, 760],
                            "dot_LHS": [20, 420, 650, 750],
                            "dot_RHS": [1010, 1410, 650, 750],
                            "top_right_outside_dot": [1010, 1410, 450, 550],
                            "below_outside_dot": [450, 850, 775, 875],
                            "single_atom_region": [560, 575, 600, 625],
                            "entire_dot": [100, 1200, 439, 880],
                            "entire_image": [100, 1200, 200, 1000]
                            }

will_dark_blue = "#192A7C"
will_light_blue = "#1E84BA"


def load_sokolov_data(region_bounds, step_size = 1):
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
    dot_epsilon_xx = full_xx_data[L_1:L_2:step_size,H_1:H_2:step_size]
    dot_epsilon_xz = -full_xy_data[L_1:L_2:step_size,H_1:H_2:step_size]
    dot_epsilon_zz = -full_yy_data[L_1:L_2:step_size,H_1:H_2:step_size]

    return dot_epsilon_xx, dot_epsilon_xz, dot_epsilon_zz

def load_mirrored_data(region_bounds, step_size = 1, faked_strain_type = "left_right"):

    # these full_epsilon_AA datasets are 1600x1600 arrays
    
    full_archive = np.load(f"{data_path}{faked_strain_type}_mirrored_strain_data_in_region_{region_bounds}.npz")

    dot_epsilon_xx = full_archive["full_xx_data"]
    dot_epsilon_xz = full_archive["full_xy_data"]
    dot_epsilon_zz = full_archive["full_yy_data"]

    return dot_epsilon_xx, dot_epsilon_xz, dot_epsilon_zz

def load_concentration_data(region_bounds, step_size = 1, method = "cubic"):
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
            Other options eventually will be "linear" and "nearest".

    Returns:
        dot_conc_data (ndarray): The percentage concentration of In115 in the QD.
    """

    filename = f"{data_path}conc_data_to_scale_{method}_interpolation.npy"
    full_conc_data = np.load(filename)

    H_1, H_2, L_1, L_2 = region_bounds

    # slice out range of data we want, slicing syntax is [start:stop:step]
    dot_conc_data = full_conc_data[L_1:L_2:step_size,H_1:H_2:step_size]

    return dot_conc_data

def euler_angles_from_rot_mat(rot_mat):
    """
    Calculates a set of Euler angles that will recreate the given rotation matrix.

    Used Slabaugh "Computing Euler Angles from Rotation Matrix" to help define this.
    Assumes rotation happens around X, then Y, then Z, in the lab frame. 
    The Euler angles are in general not unique, and we don't care about them
    specifically, so we don't find all possible sets. Just one that works.

    Args:
        rot_mat (ndarray): A 3x3 array representing a rotation matrix.

    Returns:
        alpha (float): The angle to rotate around the X axis.
        beta (float): The angle to rotate around the Y axis.
        gamma (float): The angle to rotate around the Z axis.
    """
    
    if rot_mat[2,0] != 1 and rot_mat[2,0] != -1:
        beta = -np.arcsin(rot_mat[2,0])
        alpha = np.arctan2(rot_mat[2,1]/np.cos(beta), rot_mat[2,2]/np.cos(beta))
        gamma = np.arctan2(rot_mat[1,0]/np.cos(beta), rot_mat[0,0]/np.cos(beta))
    else:
        gamma = 0 # doesn't matter, so we set it to 0 for convenience
        if rot_mat[2,0] == -1:
            beta = np.pi/2
            alpha = gamma + np.arctan2(rot_mat[0,1], rot_mat[0,2])
        else:
            beta = -np.pi/2
            alpha = -gamma + arctan2(-rot_mat[0,1], -rot_mat[0,2])
    return alpha, beta, gamma 

def calculate_EFG(nuclear_species, xx_array, xz_array, zz_array, use_sundfors_GET_vals = False):
    """
    Calculates the electric field gradient tensor for a given species in a given area.

    Args:
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        xx_array (ndarray): The xx components of strain at each site.
        xz_array (ndarray): The xz components of strain at each site.
        zz_array (ndarray): The zz components of strain at each site.

    Returns:
        eta (ndarray): The value of eta (nuclear biaxiality) at each site.
        V_XX (ndarray): The V_XX components of the EFG tensor at each site.
        V_YY (ndarray): The V_YY components of the EFG tensor at each site.
        V_ZZ (ndarray): The V_ZZ components of the EFG tensor at each site.
        euler_angles (ndarray): The Euler angles required to translate to the principal
            axis frame of each nucleus.
    """

    species = ISOP.species_dict[nuclear_species]
    if use_sundfors_GET_vals:
        species = ISOP.old_species_dict[nuclear_species]

    # print(f"Using Old Isotope Parameters: {use_sundfors_GET_vals}")
    # print(f"Current Species is: {species}")

    S11 = species['S11']
    S12 = -S11/2
    S44 = species['S44']

    # print(f"Parameters used: S11 = {S11}, S12 = {S12}, S44 = {S44}")
   
    n=xx_array.shape[0]
    m=xx_array.shape[1]
    
    eta=np.zeros((n,m))
    V_ZZ=np.zeros((n,m))
    V_XX=np.zeros((n,m))
    V_YY=np.zeros((n,m))

    euler_angles = np.zeros((n,m,3))
       
    #Full EFG tensor
    for i in range(n):
        for j in range(m):
            
            # calculate the EFG tensor from the strain tensor
            V=np.array([[S12*(zz_array[i,j]-xx_array[i,j]),     0,                                          S44*xz_array[i,j]], 
                        [0,                                     (S12+S11)*xx_array[i,j]+S12*zz_array[i,j],  S44*xz_array[i,j]],
                        [S44*xz_array[i,j],                     S44*xz_array[i,j],                          2*S12*xx_array[i,j]+S11*zz_array[i,j]]])

            # find the eigenvalues and vectors of V
            w,v = np.linalg.eig(V)

            max1=max(enumerate(abs(w)),key=lambda x: x[1])[0]
            V_ZZ[i,j]=w[max1]
            
            # change the old maximum value to 1, I think just so we don't find it twice
            # (w has values of the order 1e13)
            w[max1]=1
            
            # repeat search to find next highest value
            max2=max(enumerate(abs(w)),key=lambda x: x[1])[0]

            V_YY[i,j]=w[max2]
            w[max2]=1
            
            # repeat again, so we have found the top 3 values (all >=1)
            max3=max(enumerate(abs(w)),key=lambda x: x[1])[0]
            V_XX[i,j]=w[max3]
            
            eta[i,j]=(V_XX[i,j]-V_YY[i,j])/V_ZZ[i,j]
            
            # calculate the matrix required to translate between frames
            rot_mat = np.array([v[max3], v[max2], v[max1]]).T
            euler_angles[i,j] = euler_angles_from_rot_mat(rot_mat)

    return eta, V_XX, V_YY, V_ZZ, euler_angles

def calculate_and_save_EFG(nuclear_species, region_bounds = [100, 1200, 200, 1000], step_size = 1, use_sundfors_GET_vals = False, real_strain_data = True, faked_strain_type = "left_right"):
    """
    Calculates the EFG for a specific species over a specific region. Saves the output.

    Args:
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default is the entire dot region: [100, 1200, 200, 1000].
            Other possible regions are:
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 1 (every site is used).

    Returns:
        Nothing. But will save an npz archive to disk in a folder called data, following
        the convention for archive names of:
        {nuclear_species}_calculation_results_for_region{region_bounds}_with_step_size_{step_size}.npz"
    """

    # print("In calculate_and_save_EFG.")
    # print(f"nuclear_species = {nuclear_species}")
      
    if real_strain_data:
        xx_array, xz_array, zz_array = load_sokolov_data(region_bounds, step_size)

        if not use_sundfors_GET_vals:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}.npz"
        else:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}_with_old_params.npz"
    else:
        xx_array, xz_array, zz_array = load_mirrored_data(region_bounds, step_size, faked_strain_type)

        if not use_sundfors_GET_vals:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}_using_{faked_strain_type}_flipped_data.npz"
        else:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}_using_{faked_strain_type}_flipped_data_with_old_params.npz"

    # print(archive_name)

    if os.path.isfile(archive_name):
        # print(f"Data file '{archive_name}' alredy exists. Skipping further calculation.")
        return

    eta, V_XX, V_YY, V_ZZ, euler_angles = calculate_EFG(nuclear_species, xx_array, xz_array, zz_array, use_sundfors_GET_vals)

    np.savez(archive_name, eta = eta, V_XX = V_XX, V_YY = V_YY, V_ZZ = V_ZZ, euler_angles = euler_angles)
    print("Saved Archive: {}".format(archive_name))


def load_calculated_EFG_arrays(nuclear_species, region_bounds, step_size = 1, use_sundfors_GET_vals = False, real_strain_data = True, faked_strain_type = "left_right"):
    """
    Helper function to load EFG arrays.

    Args:
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Entire dot region is:[100, 1200, 200, 1000].
            Dot only region is: [100,1200, 439, 880].
            Standard small testing region is: [750, 800, 400, 500].

    Returns:
        eta (ndarray): The value of eta (nuclear biaxiality) at each site.
        V_XX (ndarray): The V_XX components of the EFG tensor at each site.
        V_YY (ndarray): The V_YY components of the EFG tensor at each site.
        V_ZZ (ndarray): The V_ZZ components of the EFG tensor at each site.
        euler_angles (ndarray): The Euler angles required to translate to the principal
            axis frame of each nucleus.
    """

    if real_strain_data:
        if not use_sundfors_GET_vals:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}.npz"
        else:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}_with_old_params.npz"
    else:
        if not use_sundfors_GET_vals:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}_using_{faked_strain_type}_flipped_data.npz"
        else:
            archive_name = data_path + nuclear_species + "_calculation_results_for_region" + str(region_bounds) + f"_with_step_size_{step_size}_using_{faked_strain_type}_flipped_data_with_old_params.npz"


    full_archive = np.load(archive_name)
    eta = full_archive['eta']
    dims = eta.shape

    V_XX = full_archive['V_XX']
    V_YY = full_archive['V_YY']
    V_ZZ = full_archive['V_ZZ']
    euler_angles = full_archive['euler_angles']
    # euler_angles = full_archive['euler_angles'].reshape(dims[0]*dims[1], 3) # as of 24/08/20 I am not sure why this was here, it may be needed?
    return eta, V_XX, V_YY, V_ZZ, euler_angles

def spin_rotator(alpha, beta, gamma, initial_spin):
	"""
	Function to rotate a quantum spin using Euler angles, around the axes in the order X-Y-Z/

	Args:
		alpha (float): Angle to rotate around the X axis. Measured in radians.
		beta (float): Angle to rotate around the Y axis. Measured in radians.
		gamma (float): Angle to rotate around the Z axis. Measured in radians.
		initial_spin (Qobj): A Qobj representing the initial spin state.

	Returns: rotated_spin (Qobj): A Qobj representing the rotated spin state.
	"""

	spin = (initial_spin.dims[0][0] - 1)/2
	op1 = -1j * alpha * qutip.jmat(spin, "x")
	op2 = -1j * beta * qutip.jmat(spin, "y")
	op3 = -1j * gamma * qutip.jmat(spin, "z")

	R = op1.expm() * op2.expm() * op3.expm()

	rotated_spin = R * initial_spin * R.dag()

	return rotated_spin


def structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, biaxiality, particle_spin, alpha, beta, gamma):
	"""
	Function to quickly create a Hamiltonian that governs the behaviour of a single nuclear spin in Faraday Geometry.
	
	This function creates a Qobj which represents the Hamiltonian of a single nuclear spin
	inside a quantum dot. The final Hamiltonian has the structural form:
	H = Z*I_z + Q*(2*I_z"**2 + (eta-1)I_x"**2 - (eta+1)I_y"**2).
	The unprime coordinates are in the lab frame, while the prime coordinates are in the 
	principal axis frame of the nucleus in question.

	Args:
		Zeeman_term (float): The relative size of the Zeeman interaction term Z.
		quadrupolar_term (float): The relative size of the quadrupolar term Q.
		biaxiality (float): The biaxiality of the quadrupolar field. Greek letter eta.
		particle_spin (float): The spin of the nucleus being studied.
		alpha (float): First Euler Angle, describing the rotation of the principal axis frame around the X axis.
		beta (float): Second Euler Angle, describing the rotation of the principal axis frame around the Y axis.
		gamma (float): Third Euler Angle, describing the rotation of the principal axis frame around the Z axis.

	Returns:
		nuclear_hamiltonian (Qobj): The Hamiltonian representing the nuclear spin system, in the lab frame.
	"""
	# Create Base Operators - In Lab Frame
	I_x = qutip.jmat(particle_spin, "x")
	I_y = qutip.jmat(particle_spin, "y")
	I_z = qutip.jmat(particle_spin, "z")

	# Create Operators Rotated from Lab Frame -> Quadrupolar Frame
	I_x_quad = spin_rotator(alpha, beta, gamma, I_x)
	I_y_quad = spin_rotator(alpha, beta, gamma, I_y)
	I_z_quad = spin_rotator(alpha, beta, gamma, I_z)

	nuclear_hamiltonian = Zeeman_term*I_z + quadrupolar_term*(2* I_z_quad**2 + (biaxiality - 1)* I_x_quad**2 - (biaxiality + 1)* I_y_quad**2)

	return nuclear_hamiltonian

def structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, biaxiality, particle_spin, alpha, beta, gamma):
    """
    Function to quickly create a Hamiltonian that governs the behaviour of a single nuclear spin in Voigt Geometry.
    
    This function creates a Qobj which represents the Hamiltonian of a single nuclear spin
    inside a quantum dot. The final Hamiltonian has the structural form:
    H = Z*I_x + Q*(2*I_z"**2 + (eta-1)I_x"**2 - (eta+1)I_y"**2).
    The unprime coordinates are in the lab frame, while the prime coordinates are in the 
    principal axis frame of the nucleus in question.

    Args:
        Zeeman_term (float): The relative size of the Zeeman interaction term Z.
        quadrupolar_term (float): The relative size of the quadrupolar term Q.
        biaxiality (float): The biaxiality of the quadrupolar field. Greek letter eta.
        particle_spin (float): The spin of the nucleus being studied.
        alpha (float): First Euler Angle, describing the rotation of the principal axis frame around the X axis.
        beta (float): Second Euler Angle, describing the rotation of the principal axis frame around the Y axis.
        gamma (float): Third Euler Angle, describing the rotation of the principal axis frame around the Z axis.

    Returns:
        nuclear_hamiltonian (Qobj): The Hamiltonian representing the nuclear spin system, in the lab frame.
    """
    # Create Base Operators - In Lab Frame
    I_x = qutip.jmat(particle_spin, "z")
    I_y = qutip.jmat(particle_spin, "y")
    I_z = qutip.jmat(particle_spin, "x")

    # Create Operators Rotated from Lab Frame -> Quadrupolar Frame
    I_x_quad = spin_rotator(alpha, beta, gamma, I_x)
    I_y_quad = spin_rotator(alpha, beta, gamma, I_y)
    I_z_quad = spin_rotator(alpha, beta, gamma, I_z)

    nuclear_hamiltonian = Zeeman_term*I_z + quadrupolar_term*(2* I_z_quad**2 + (biaxiality - 1)* I_x_quad**2 - (biaxiality + 1)* I_y_quad**2)

    return nuclear_hamiltonian

def RF_hamiltonian_creator(particle_spin, B_x, B_y, B_z, unknown_gamma = 1):
    # I'm not entirely sure what unknown_gamma is at the moment, physically I mean (29/11/20)
    # the units of this Hamiltonian are probably not consistent either tbh, and may not matter at all
    I_x = qutip.jmat(particle_spin, "x")
    I_y = qutip.jmat(particle_spin, "y")
    I_z = qutip.jmat(particle_spin, "z")

    hbar = 1 #scipy.constants.hbar

    # this doesn't include the cos(wt) part which actually makes it oscillate IRL
    # just has the structure as that's all we need for finding transition rates
    rf_Ham = -hbar*unknown_gamma*(B_x*I_x + B_y*I_y + B_z*I_z)

    return rf_Ham

def transition_rate_calc(mixing_hamiltonian, init_state, final_state, E_init, E_final, omega_rf, delta = 10e3):
    # mixing_hamiltonian is normally made by the RF_hamiltonian function, but could in principle be anything
    transition_prob = np.abs(mixing_hamiltonian.matrix_element(final_state, init_state))

    frac_term = (2*delta)/((E_final - E_init - omega_rf)**2 + delta**2)

    return np.real_if_close(transition_prob * frac_term)


def spin_correlator_matrix_exponential(t, nuclear_hamiltonian, spin_axis):
	"""
	Function to calculate the value of the time correlation function of a particular nucleus along a set axis.

	Args:
		t (float): The time (in seconds) at which to calculate the spin correlator.
		nuclear_hamiltonian (Qobj): The Hamiltonian describing a particular nuclear spin.
		spin_axis (str): The axis along which to calculate the correlator. Options are: "x", "y", "z".

	Returns:
		spin_correlator (float): The value of the spin correlator along a specific axis.
	"""

	particle_spin = (nuclear_hamiltonian.dims[0][0] - 1)/2

	if spin_axis == "x":
		spin_operator = qutip.jmat(particle_spin, "x")
	elif spin_axis == "y":
		spin_operator = qutip.jmat(particle_spin, "y")
	elif spin_axis == "z":
		spin_operator = qutip.jmat(particle_spin, "z")
	else:
		print(f"bqf.spin_correlator_matrix_exponential recieved an invalid spin axis. Options are x, y, z.")
		return None

	exponent = 1j * t * nuclear_hamiltonian
	U_plus = exponent.expm()
	U_minus = (-exponent).expm()

	rho = qutip.qeye(int(nuclear_hamiltonian.dims[0][0])).unit()

	spin_correlator = numpy.real_if_close((U_plus * spin_operator * U_minus * spin_operator * rho).tr())

	return spin_correlator

def site_correlator_calculator_not_parallel(t, zeeman_frequency_per_tesla, quadrupole_coupling_constant, spin, biaxiality, euler_angles, V_ZZ, applied_field, correlator_axis):
	"""
	Calculates the spin correlator at a certain site at a certain time.
	
	Args:
		t (float): The time at which to calculate the correlator.
		zeeman_frequency_per_tesla (float): The energy splitting due to the Zeeman interaction, in units of Hz/T.
    	quadrupole_moment (float): The quadrupolar moment of that nucleus, analagous to the dipole moment, in units of m**2.
    	spin (float): The nuclear spin of the particle, eg 4.5 for In115.
    	biaxiality (float): The biaxiality of the quadrupolar field at this site.
    	euler_angles (ndarray): The Euler angles required to translate to the principal	axis frame.
		V_ZZ (ndarray): The V_ZZ components of the EFG tensor this site.
		applied_field (float): The magnetic field applied to the quantum dot.
		correlator_axis (str): The spin axis along which to calculate the correlator.
			Options are: x, y, z.

	Returns:
		correlator_at_site (float): The value of the spin correlator at a particular site at time t.
	"""

	alpha = euler_angles[0]
	beta = euler_angles[1]
	gamma = euler_angles[2]

	quadrupole_frequency = quadrupole_coupling_constant * V_ZZ
	zeeman_frequency = zeeman_frequency_per_tesla * applied_field

	Hamiltonian = structural_hamiltonian_creator_Faraday(zeeman_frequency, quadrupole_frequency, biaxiality, spin, alpha, beta, gamma)
	correlator_at_site = spin_correlator_matrix_exponential(t, Hamiltonian, correlator_axis)

	return correlator_at_site

def parallel_site_correlator_calculator(elapsed_time, zeeman_frequency_per_tesla, quadrupole_coupling_constant, spin, biaxiality, alpha, beta, gamma, V_ZZ, applied_field, correlator_axis):
	"""
	Calculates the spin correlator at a certain site at a certain time, suitable for mapping in parallel.

	Does pretty much the same job as site_correlator_calculator_not_parallel. All arguments are explicit
	making it suitable for use with the zipper functions. 

	#TODO
		Combine this and site_correlator_calculator_not_parallel into a single function that can be 
		used in parallel if required. Having 2 is just a little bit dirty...
	
	Args:
		t (float): The time at which to calculate the correlator.
		zeeman_frequency_per_tesla (float): The energy splitting due to the Zeeman interaction, in units of Hz/T.
    	quadrupole_moment (float): The quadrupolar moment of that nucleus, analagous to the dipole moment, in units of m**2.
    	spin (float): The nuclear spin of the particle, eg 4.5 for In115.
    	biaxiality (float): The biaxiality of the quadrupolar field at this site.
    	euler_angles (ndarray): The Euler angles required to translate to the principal	axis frame.
		V_ZZ (ndarray): The V_ZZ components of the EFG tensor this site.
		applied_field (float): The magnetic field applied to the quantum dot.
		correlator_axis (str): The spin axis along which to calculate the correlator.
			Options are: x, y, z.

	Returns:
		correlator_at_site (float): The value of the spin correlator at a particular site at time t.
	"""

	quadrupole_frequency = quadrupole_coupling_constant * V_ZZ
	zeeman_frequency = zeeman_frequency_per_tesla * applied_field

	Hamiltonian = structural_hamiltonian_creator_Faraday(zeeman_frequency, quadrupole_frequency, biaxiality, spin, alpha, beta, gamma)

	correlator_at_site = spin_correlator_matrix_exponential(elapsed_time, Hamiltonian, correlator_axis)

	return correlator_at_site

# if __name__ == "__main__":
    # testing out the Hamiltonian creators, as they don't appear to be working (26/08/20)
    # import matplotlib.pyplot as plt

    # # Zeeman_term = 1
    # # quadrupolar_term = 0.1
    # # biaxiality = 0.4
    # # particle_spin = 9/2
    # # alpha = np.pi
    # # beta = 2
    # # gamma = -1

    # Zeeman_term = 1
    # quadrupolar_term = 0.1
    # biaxiality = 0.4
    # particle_spin = 9/2
    # alpha = np.pi
    # beta = 2
    # gamma = -1

    # parameter_list = [Zeeman_term, quadrupolar_term, biaxiality, particle_spin, alpha, beta, gamma]

    # faraday_ham = structural_hamiltonian_creator_Faraday(Zeeman_term, quadrupolar_term, biaxiality, particle_spin, alpha, beta, gamma)
    # voigt_ham = structural_hamiltonian_creator_Voigt(Zeeman_term, quadrupolar_term, biaxiality, particle_spin, alpha, beta, gamma)

    # print(f"Faraday Eigenvalues: {np.around(faraday_ham.eigenstates()[0], 1)}")
    # print(f"Voigt Eigenvalues: {np.around(voigt_ham.eigenstates()[0], 1)}")

    # # print(faraday_ham.eigenstates()[1][0])
    # # print(voigt_ham.eigenstates()[1][0])

    # # qutip.hinton(faraday_ham)
    # # plt.savefig(f"{graph_path}faraday_hamiltonain_with_parameters_{parameter_list}.png")

    # # qutip.hinton(voigt_ham)
    # # plt.savefig(f"{graph_path}voigt_hamiltonain_with_parameters_{parameter_list}.png")

    # # faraday_eigs = faraday_ham.eigenstates()

    # # print(faraday_eigs[0])
    # # print(faraday_eigs[1].shape)


    # testing out using new and old parameter sets (20/10/20)

    # region_bounds = [500, 550, 900, 1000]
    # step_size = 1

    # load_sokolov_data(region_bounds, step_size)

    # xx_array, xz_array, zz_array = load_sokolov_data(region_bounds, step_size)
    # print("-------------------------")
    # for nuclear_species in nuclear_species_list:
    #     eta, V_XX, V_YY, V_ZZ, euler_angles = calculate_EFG(nuclear_species, xx_array, xz_array, zz_array, use_sundfors_GET_vals = False)
    #     eta, V_XX, V_YY, V_ZZ, euler_angles = calculate_EFG(nuclear_species, xx_array, xz_array, zz_array, use_sundfors_GET_vals = True)
    #     print("-------------------------")

    # print(QD_regions_dict["entire_dot"])