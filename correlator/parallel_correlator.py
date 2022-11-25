import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")

import time
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import isotope_parameters as ISOP
import scipy.constants as constants

def parameter_zipper(elapsed_time, applied_field, correlator_axis, nuclear_species, region_bounds, step_size = 100):
    """
    Helper function to zip parameters together, used when running correlator calculations in parallel.

    Args:
        elapsed_time (float): The time at which to calculate the correlator.
        applied_field (float): The magnetic field applied to the quantum dot.
        correlator_axis (str): The spin axis along which to calculate the correlator.
            Options are: x, y, z.
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Standard possible regions are:
                Entire dot region: [100, 1200, 200, 1000].
                Dot only region is: [100,1200, 439, 880]
                Standard small testing region is: [750, 800, 400, 500]
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 30.

    Returns:
        n_sites (int): The number of nuclear sites in the region, depends
            on step size.
        zipped_params (list): A list containing the parameters required
            to do the correlator calculation over each site. Some are
            likely to be repeated. 
    """

    # not sure if the dictionary is necessary in the parallel version (08/05/20)
    data_dict = {}
    data_dict["eta"], data_dict["V_XX"], data_dict["V_YY"], data_dict["V_ZZ"], data_dict["euler_angles"] = bqf.load_calculated_EFG_arrays(nuclear_species, region_bounds, step_size)

    n_sites = data_dict["eta"].size
    h = constants.h
    e = constants.e

    species = ISOP.species_dict[nuclear_species]

    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]
    quadrupole_coupling_constant = (3*e*Q)/(2*h*spin*(2*spin - 1)) # constant to convert Vzz to fq

    euler_angles = data_dict["euler_angles"]

    # create big lists to zip together for the parallel calculation

    # these lists all contain the same value, should be a way to skip this step and just pass in the same number each time?
    spin_list = np.ones(n_sites)*spin
    zeeman_frequency_per_tesla_list = np.ones(n_sites)*zeeman_frequency_per_tesla
    quadrupole_coupling_constant_list = np.ones(n_sites)*quadrupole_coupling_constant
    t_list = np.ones(n_sites)*elapsed_time
    correlator_axis_list = [correlator_axis]*n_sites
    applied_field_list = np.ones(n_sites)*applied_field

    # these lists contain different values for each site
    biaxiality_list = data_dict["eta"].flatten()
    alpha_list = euler_angles[:,0]
    beta_list = euler_angles[:,1]
    gamma_list = euler_angles[:,2]
    V_ZZ_list = data_dict["V_ZZ"].flatten()

    # zip the parameters together to make an object we can use starmap with
    # site calculator takes them in the order: 
    # [elapsed_time, zeeman_frequency_per_tesla, quadrupole_coupling_constant, spin, biaxiality, alpha, beta, gamma, V_ZZ, applied_field, correlator_axis]
    zipped_params = list(zip(t_list, zeeman_frequency_per_tesla_list, quadrupole_coupling_constant_list, spin_list, biaxiality_list, alpha_list, beta_list, gamma_list, V_ZZ_list, applied_field_list, correlator_axis_list))

    return n_sites, zipped_params

def one_species_time_series_correlator_calculator(timerange, applied_field, nuclear_species, region_bounds, step_size = 100, allocated_at_once = 25):
    """
    Calculates the spin correlators for a series of times at a specific magnetic field.

    This function does the calculation in parallel, with the splitting up
    of jobs happening at the level of site calculations. As in, for a 
    specific spin axis, it divides the sites up between different workers
    and then combines them afterwards. It then moves on to the next set 
    of sites, and eventually the next spin axis.

    Args:
        timerange (ndarray): The set of times to calculate the correlator for.
        applied_field (float): The magnetic field applied to the quantum dot.
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Entire dot region is:[100, 1200, 200, 1000].
            Dot only region is: [100,1200, 439, 880].
            Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 30.
        allocated_at_once (int): Number of tasks to assign to each member
            of the worker pool at once. Default is 25.

    Returns:
        correlator_results_array (ndarray): Array containing the x, y and z 
            spin correlators for the entire region, calculated at each timepoint.
    """

    correlator_results_array = np.zeros((3, len(timerange)))

    for c, correlator_axis in enumerate(["x", "y", "z"]):
        for t, elapsed_time in enumerate(timerange):
            n_sites, zipped_params = parameter_zipper(elapsed_time, applied_field, correlator_axis, nuclear_species, region_bounds, step_size)
            with multiprocessing.Pool() as pool:
                site_correlators = pool.starmap(bqf.parallel_site_correlator_calculator, zipped_params, chunksize = allocated_at_once)
                pool.close()

            correlator_results_array[c,t] = np.average(site_correlators)

    return correlator_results_array

def log_spaced_correlator_simulation(min_time_exp, max_time_exp, n_times, applied_field, region_bounds = [100,1200, 439, 880], step_size = 100, allocated_at_once = 25):
    """
    Creates and saves data sets of logtime spaced time correlator data
    for a single magnetic field.

    Args:
        min_time_exp (int): The exponential of the smallest time in the desired range.
            The first time in the range will be 10**min_time_exp.
        max_time_exp (int): The exponential of the largest time in the desired range.
            The last time in the range will be 10**max_time_exp.
        n_times (int): The number of times at which to calculate the correlator.
        applied_field (float): The magnetic field applied to the quantum dot.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 30.
        allocated_at_once (int): Number of tasks to assign to each member
            of the worker pool at once. Default is 25.

    Returns:
        Nothing, but will save an npz archive to the data folder, using the 
        naming convention:
            log_time_correlator_data_with_B_field_of_{applied_field}T_for_{n_times}_times_over_a_timerange_of{first_time}_{last_time}s_for_region_{region_bounds}.npz
        The archive is ordered by species name, then in an array of dimension (3 x n_times),
        in the order x,y,z along the first axis
    """

    timerange = np.logspace(min_time_exp, max_time_exp, n_times)

    Ga69_data = one_species_time_series_correlator_calculator(timerange, applied_field, "Ga69", region_bounds, step_size, allocated_at_once)
    print("Ga69 Log Data Calculated")
    Ga71_data = one_species_time_series_correlator_calculator(timerange, applied_field, "Ga71", region_bounds, step_size, allocated_at_once)
    print("Ga71 Log Data Calculated")
    As75_data = one_species_time_series_correlator_calculator(timerange, applied_field, "As75", region_bounds, step_size, allocated_at_once)
    print("As75 Log Data Calculated")
    In115_data = one_species_time_series_correlator_calculator(timerange, applied_field, "In115", region_bounds, step_size, allocated_at_once)
    print("In115 Log Data Calculated")

    # print("----------------------------------------------------------------------------")
    archive_name = f"{data_path}log_time_correlator_data_with_B_field_of_{str(applied_field)}T_for_{n_times}_times_over_a_timerange_of{timerange[0]}_{timerange[-1]}s_for_region_{region_bounds}.npz"
    # print(archive_name)
    # print("----------------------------------------------------------------------------")
    np.savez(archive_name, timerange = timerange, region_bounds = region_bounds, Ga69_data = Ga69_data, Ga71_data = Ga71_data, As75_data = As75_data, In115_data = In115_data)

def linear_spaced_correlator_simulation(min_time, max_time, timestep, applied_field, region_bounds = [100, 1200, 439, 880], step_size = 100, allocated_at_once = 25):
    """
    Creates and saves data sets of linearly time spaced time correlator data
    for a single magnetic field.

    Args:
        min_time (float): The first time at which to calculate the correlator.
        max_time (float): The end of the time interval. This may or may not be 
            included in the interval, depending on the chosen timestep.
            See the documentation on numpy.arange for details.
        timestep (float): The separation between times at which to do the calculation.
        applied_field (float): The magnetic field applied to the quantum dot.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 30.
        allocated_at_once (int): Number of tasks to assign to each member
            of the worker pool at once. Default is 25.

    Returns:
        Nothing, but will save an npz archive to the data folder, using the 
        naming convention:
            linear_time_correlator_data_with_B_field_of_{applied_field}T_over_a_timerange_of{start_time}_{end_time}s_with_a_timestep_of_{timestep}_for_region_{region_bounds}.npz"
        The archive is ordered by species name, then in an array of dimension (3 x n_times),
        in the order x,y,z along the first axis
    """

    timerange = np.arange(min_time, max_time, timestep)
    n_times = np.ceil((stop - start)/step)

    Ga69_data = one_species_time_series_correlator_calculator(timerange, applied_field, "Ga69", region_bounds, step_size, allocated_at_once)
    print("Ga69 Linear Data Calculated")
    Ga71_data = one_species_time_series_correlator_calculator(timerange, applied_field, "Ga71", region_bounds, step_size, allocated_at_once)
    print("Ga71 Linear Data Calculated")
    As75_data = one_species_time_series_correlator_calculator(timerange, applied_field, "As75", region_bounds, step_size, allocated_at_once)
    print("AS75 Linear Data Calculated")
    In115_data = one_species_time_series_correlator_calculator(timerange, applied_field, "In115", region_bounds, step_size, allocated_at_once)
    print("In115 Linear Data Calculated")

    # print("----------------------------------------------------------------------------")
    archive_name = f"{data_path}linear_time_correlator_data_with_B_field_of_{str(applied_field)}T_over_a_timerange_of{timerange[0]}_{timerange[-1]}s_with_a_timestep_of_{timestep}_for_region_{region_bounds}.npz"
    # print(archive_name)
    # print("----------------------------------------------------------------------------")
    np.savez(archive_name, timerange = timerange, region_bounds = region_bounds, Ga69_data = Ga69_data, Ga71_data = Ga71_data, As75_data = As75_data, In115_data = In115_data)

def single_species_linear_time_correlator_grapher(min_time, max_time, timestep, applied_field, nuclear_species, region_bounds = [100,1200, 439, 880]):
    """
    Graphs the evolution of a correlator over time.

    This function is used to graph linearly spaced timeranges, use the 
    logtime version for logarithmically spaced ones.

    Args:
        min_time (float): The earliest time at which the correlator was calculated, in seconds.
        max_time (float): The latest time at which the correlator was calculated, in seconds.
        timestep (float): The separation between times at which the calculation was done.
        applied_field (float): The magnetic field applied to the quantum dot.
        nuclear_species (str): The atomic species to find the EFG for. 
            Possible options are: Ga69, Ga71, As75, In115.
        logtime (bool): Boolean that describes whether the timerange used
            was generated using np.logtime. Default is False.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].

    Returns:
        Nothing, but will have saved a figure to the graphs folder, using
            the naming scheme:
                log_time_correlator_graph_for_{nuclear_species}_in_{applied_field}T_field_using_{n_times}_timepoints_over_a_timerange_of{min_time}_{max_time}s.png"
            for log spaced timeranges, or:
                linear_time_correlator_graph_for_{nuclear_species}_in_{applied_field}T_field_using_{n_times}_timepoints_over_a_timerange_of{min_time}_{max_time}s.png
    """

    archive_name = f"{data_path}linear_time_correlator_data_for_{nuclear_species}_with_B_field_of_{str(applied_field)}T_over_a_timerange_of{min_time}_{max_time}s_with_a_timestep_of_{timestep}_for_region_{region_bounds}.npz"

    # print(f"Loading the archive named: {archive_name}")
    full_archive = np.load(archive_name)
    timerange = full_archive["timerange"]
    correlator_data = full_archive["correlator_data"]

    for i, correlator_axis in enumerate(["x", "y", "z"]):
        plt.semilogx(timerange, correlator_data[i,:], label = f"{correlator_axis} Axis")
    
    plot_name = f"{graph_path}linear_time_correlator_graph_for_{nuclear_species}_in_{str(applied_field)}T_field_over_a_timerange_of{timerange[0]}_{timerange[-1]}s_with_a_timestep_of_{timestep}.png"
    # plt.ticklabel_format(style = "sci", scilimits = (-3,3))

    plt.xlabel("Tau (s)")
    plt.ylabel("Correlator")
    plt.title(f"Correlator for {nuclear_species} with Applied Field of {applied_field}T")
    plt.legend()

    plt.savefig(plot_name)
    plt.close()

# def single_species_logtime_correlator_grapher(min_time, max_time, n_times, applied_field, nuclear_species, region_bounds = [100,1200, 439, 880])

def single_species_fourier_transform_grapher(min_time, max_time, timestep, applied_field, nuclear_species, region_bounds = [100,1200, 439, 880]):
    """
    Calculates and graphs the Fourier transform of a time-series corellator.

    Args:
        min_time (float): The first time at which to calculate the correlator.
        max_time (float): The end of the time interval. This may or may not be 
            included in the interval, depending on the chosen timestep.
            See the documentation on numpy.arange for details.
        timestep (float): The separation between times at which to do the calculation.
        applied_field (float): The magnetic field applied to the quantum dot.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].
        
    Returns:
        None, but will save a graph of the Fourier transform to the graphs folder.
    """

    archive_name = f"{data_path}linear_time_correlator_data_for_{nuclear_species}_with_B_field_of_{str(applied_field)}T_over_a_timerange_of{min_time}_{max_time}s_with_a_timestep_of_{timestep}_for_region_{region_bounds}.npz"
    full_archive = np.load(archive_name)
    timerange = full_archive["timerange"]
    # region_bounds = full_archive["region_bounds"]
    correlator_data = full_archive["correlator_data"]

    n_times = len(timerange)

    freq = np.fft.rfftfreq(n_times, timestep)

    for i, correlator_axis in enumerate(["x", "y", "z"]):
        fft_data = np.fft.rfft(correlator_data[i,:])
        plt.semilogx(freq, fft_data.real, label = f"{correlator_axis} Axis")
   
    plot_name = f"{graph_path}fourier_transform_of_correlator_for_{nuclear_species}_in_{str(applied_field)}T_field_over_a_timerange_of{timerange[0]}_{timerange[-1]}s_with_a_timestep_of_{timestep}.png"

    plt.xlabel("Frequency (Not Sure of Units Yet - 22/05/20)")
    plt.title(f"Fourier Transforms for {nuclear_species} with Applied Field of {applied_field}")
    plt.legend()
    # plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    plt.savefig(plot_name)
    plt.close()

def single_species_correlator_to_fourier(min_time, max_time, timestep, applied_field, nuclear_species, region_bounds = [100, 1200, 439, 880], step_size = 100, allocated_at_once = 25):
    """
    Helper function to take a single species all the way from correlator calculation to plotting the Fourier transform.

    This function is mainly used for testing purposes, as it doesn't save any data along the way. It's a
    convenient way of checking to make sure the whole pipeline is working.

    Args:
        min_time (float): The first time at which to calculate the correlator.
        max_time (float): The end of the time interval. This may or may not be 
            included in the interval, depending on the chosen timestep.
            See the documentation on numpy.arange for details.
        timestep (float): The separation between times at which to do the calculation.
        applied_field (float): The magnetic field applied to the quantum dot.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 30.
        allocated_at_once (int): Number of tasks to assign to each member
            of the worker pool at once. Default is 25.

    Returns:
        None, but will save a graph of the Fourier transform to the graphs folder.
        It will also print out how long the entire process took, in minutes.
    """

    timerange = np.arange(min_time, max_time, timestep)
    print(len(timerange))
    print(timerange)

    start = time.time()
    # calculate the correlator data
    print(f"Calculating the Correlator Data For {nuclear_species}")
    correlator_data = one_species_time_series_correlator_calculator(timerange, applied_field, nuclear_species, region_bounds, step_size, allocated_at_once)

    # save the correlator data
    archive_name = f"{data_path}linear_time_correlator_data_for_{nuclear_species}_with_B_field_of_{str(applied_field)}T_over_a_timerange_of{min_time}_{max_time}s_with_a_timestep_of_{timestep}_for_region_{region_bounds}.npz"
    # print(f"Saving archive named: {archive_name}")
    np.savez(archive_name, correlator_data = correlator_data, timerange = timerange)


    single_species_linear_time_correlator_grapher(min_time, max_time, timestep, applied_field, nuclear_species, region_bounds = [100,1200, 439, 880])

    single_species_fourier_transform_grapher(min_time, max_time, timestep, applied_field, nuclear_species, region_bounds = [100,1200, 439, 880])

    # plot the correlator over time
    print("Plotting the Correlator")
    for i, correlator_axis in enumerate(["x", "y", "z"]):
        plt.plot(timerange, correlator_data[i,:], label = f"{correlator_axis} Axis")
    plot_name = f"{graph_path}linear_time_correlator_graph_for_{nuclear_species}_in_{str(applied_field)}T_field_over_a_timerange_of{timerange[0]}_{timerange[-1]}s_with_a_timestep_of_{timestep}.png"

    plt.xlabel("Tau (s)")
    plt.ylabel("Correlator")
    plt.title(f"Correlator for {nuclear_species} with Applied Field of {applied_field}T")
    plt.legend()
    plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    plt.savefig(plot_name)
    plt.close()

    # calculate and plot the Fourier Transform
    print("Calculating and Plotting the Fourier Transform")
    n_times = len(timerange)
    freq = np.fft.rfftfreq(n_times, timestep)

    for i, correlator_axis in enumerate(["x", "y", "z"]):
        fft_data = np.fft.rfft(correlator_data[i,:])
        plt.plot(freq, fft_data.real, label = f"{correlator_axis} Axis")
    
    plot_name = f"{graph_path}fourier_transform_of_correlator_for_{nuclear_species}_in_{str(applied_field)}T_field_over_a_timerange_of{timerange[0]}_{timerange[-1]}s_with_a_timestep_of_{timestep}.png"

    plt.xlabel("Frequency (Not Sure of Units Yet - 22/05/20)")
    plt.title(f"Fourier Transforms for {nuclear_species} with Applied Field of {applied_field}")
    plt.legend()
    # plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))
    plt.savefig(plot_name)
    plt.close()

    end = time.time()
    time_taken = np.around((end-start)/60, 2)
    print(f"Time taken: {time_taken} minutes.")

def all_species_correlator_to_fourier(min_time, max_time, timestep = 10e-9, applied_B_field_list = [0.0, 0.5, 1.0]):
    """
    Runs an entire simulation, for all species along all axes.

    This is the function to use when the parameters are set and ready to go.
    It calculates the correlators for all species in a given region, over
    a given timerange then uses those correlators to find the Fourier
    transform and graph it. It saves both the data and the resulting graphs.

    Args:
        min_time (float): The first time at which to calculate the correlator.
        max_time (float): The end of the time interval. This may or may not be 
            included in the interval, depending on the chosen timestep.
            See the documentation on numpy.arange for details.
        timestep (float): The separation between times at which to do the calculation.
        applied_field (float): The magnetic field applied to the quantum dot.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].

    Returns:
        Nothing, but will save correlator data and graphs, and Fourier transform graphs.
    """

    start = time.time()
    print("Staring All Species Correlator to Fourier Transform Calculation")
    n_times = np.ceil((stop - start)/step)
    for applied_B_field in applied_B_field_list:
        # calculate and save correlators for all species
        linear_spaced_correlator_simulation(min_time, max_time, timestep, applied_B_field)
        print(f"Finished correlator computation for B = {applied_B_field}T.")

        # plot and save the graohs for all species
        single_species_time_correlator_grapher(min_time, max_time, n_times, applied_B_field, "Ga69")
        single_species_time_correlator_grapher(min_time, max_time, n_times, applied_B_field, "Ga71")
        single_species_time_correlator_grapher(min_time, max_time, n_times, applied_B_field, "As75")
        single_species_time_correlator_grapher(min_time, max_time, n_times, applied_B_field, "In115")
        print(f"Finished Correlator Graphs for B = {applied_B_field}T.")

        # plot and save the graphs of the Fourier Transform for all species
        single_species_fourier_transform_grapher(min_time, max_time, n_times, applied_B_field, "Ga69")
        single_species_fourier_transform_grapher(min_time, max_time, n_times, applied_B_field, "Ga71")
        single_species_fourier_transform_grapher(min_time, max_time, n_times, applied_B_field, "As75")
        single_species_fourier_transform_grapher(min_time, max_time, n_times, applied_B_field, "In115")
        print(f"Finished Fourier Transforms for B = {applied_B_field}")

    end = time.time()
    time_taken = np.around((end-start)/60, 2)
    print(f"Time taken: {time_taken} minutes.")

def all_species_logtime_correlator_grapher(min_exp, max_exp, n_times = 1000, applied_B_field_list = [0.0, 0.5, 1.0]):
    """
    Runs an entire simulation, for all species along all axes.

    This is the function to use when the parameters are set and ready to go.
    It calculates the correlators for all species in a given region, over
    a given timerange then uses those correlators to find the Fourier
    transform and graph it. It saves both the data and the resulting graphs.

    Args:
        min_time_exp (int): The exponential of the smallest time in the desired range.
            The first time in the range will be 10**min_time_exp.
        max_time_exp (int): The exponential of the largest time in the desired range.
            The last time in the range will be 10**max_time_exp.
        n_times (int): The number of times at which to calculate the correlator.
        applied_field (float): The magnetic field applied to the quantum dot.
        region_bounds (list): A list of 4 integers, that define the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Default option is the dot only region: [100,1200, 439, 880].
                Entire dot region is:[100, 1200, 200, 1000].
                Standard small testing region is: [750, 800, 400, 500].

    Returns:
        Nothing, but will save correlator data and graphs.
    """

    exp_min_time = 10.0**min_exp
    exp_max_time = 10.0**max_exp

    start = time.time()
    print("Staring All Species LogTime Correlator Calculation")
    for applied_B_field in applied_B_field_list:
        # calculate and save the correlators, using a logarithm time difference
        log_spaced_correlator_simulation(min_exp, max_exp, n_times, applied_B_field)
        print(f"Finished computation for B = {applied_B_field}T.")

        # plot and save the graphs of correlator over time, using a semilogx plot.
        single_species_time_correlator_grapher(exp_min_time, exp_max_time, n_times, applied_B_field, "Ga69", logtime = True)
        single_species_time_correlator_grapher(exp_min_time, exp_max_time, n_times, applied_B_field, "Ga71", logtime = True)
        single_species_time_correlator_grapher(exp_min_time, exp_max_time, n_times, applied_B_field, "As75", logtime = True)
        single_species_time_correlator_grapher(exp_min_time, exp_max_time, n_times, applied_B_field, "In115", logtime = True)
        print(f"Finished Correlator Graphs for B = {applied_B_field}T.")


    end = time.time()
    time_taken = np.around((end-start)/60, 2)
    print(f"Time taken: {time_taken} minutes.")


if __name__ == '__main__':

    num_cores = multiprocessing.cpu_count()
    print(num_cores)

    # Timing Definitions
    start_time = 0
    end_time = 0.5e-5
    time_increment = 1e-7
    num_times = np.ceil((end_time-start_time)/time_increment)

    # Dot Region and Site Counting Definitions
    dot_only_region = [100,1200, 439, 880]
    bigger_dot_region = [100, 1200, 200, 1000]
    small_test_region = [750, 800, 400, 500]

    region = dot_only_region
    num_sites = (region[1]-region[0])*(region[3]-region[2])
    site_interval = 60

    applied_magnetic_field = 0.5
    applied_magnetic_field_list = [0.2, 0.4, 0.6, 0.8, 1.0]

    num_jobs_per_species_per_time = (num_sites/site_interval)

    print(num_jobs_per_species_per_time)

    tasks_allocated = int(num_jobs_per_species_per_time/num_cores)
    print(tasks_allocated)
    # nuclear_species = "In115"

    nuclear_species_list = ["In115"] #["Ga69", "Ga71", "As75", "In115"]
    for nuclear_species in nuclear_species_list:
        single_species_correlator_to_fourier(start_time, end_time, time_increment, applied_magnetic_field, nuclear_species, step_size = site_interval, allocated_at_once = tasks_allocated)
        single_species_linear_time_correlator_grapher(start_time, end_time, time_increment, applied_magnetic_field, nuclear_species, step_size = site_interval, allocated_at_once = tasks_allocated)

    # FFT QUESTIONS
    # What is the window, and what is the sampling rate? (as defined by np.fftfreq)
    # Do I need an envelope/windowing function?
    # Is it worth log scaling the output of the FFT, to better represent what I'm seeing


    # # starting_time_exponent = -10
    # # ending_time_exponent = 5
    # # n_times = 500

    # # all_species_logtime_correlator_grapher(starting_time_exponent, ending_time_exponent, n_times, applied_B_field_list = [0.5, 1.0])

    # starting_time_exponent = -10
    # ending_time_exponent = -4
    # number_of_times = 10

    # all_species_logtime_correlator_grapher(starting_time_exponent, ending_time_exponent, number_of_times)