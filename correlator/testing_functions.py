import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")

# SiteCorrelatorSpeedTest and SiteCorrelatorSpeedTestGraphing were copied over from correlator_calc.py on 26/06/20


def SiteCorrelatorSpeedTest(
    t,
    applied_field,
    correlator_axis,
    nuclear_species,
    n_tests,
    region_bounds=[100, 1200, 200, 1000],
):
    matexp_times = np.zeros(n_tests)
    eigen_times = np.zeros(n_tests)

    data_dict = {}
    (
        data_dict["eta"],
        data_dict["V_XX"],
        data_dict["V_YY"],
        data_dict["V_ZZ"],
        data_dict["euler_angles"],
    ) = LoadCalculatedEFGArrays(nuclear_species, region_bounds)

    n_sites = data_dict["eta"].size
    total_correlator = 0
    h = constants.h
    e = constants.e

    random_sites = np.random.randint(0, n_sites, n_tests)

    species = ISOP.species_dict[nuclear_species]

    spin = species["particle_spin"]
    zeeman_frequency_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]

    quadrupole_coupling_constant = (3 * e * Q) / (
        2 * h * spin * (2 * spin - 1)
    )  # constant to convert Vzz to fq
    for method in ["matexp", "eigen"]:
        for s in range(len(random_sites)):
            biaxiality = data_dict["eta"].flatten()[
                s
            ]  # needs to be flattened here to make sure it's 1D
            euler_angles = data_dict["euler_angles"][
                s
            ]  # has already been reshaped when I loaded it
            V_ZZ = data_dict["V_ZZ"].flatten()[s]

            start = time.time()
            c = SiteCorrelatorCalculator(
                t,
                zeeman_frequency_per_tesla,
                quadrupole_coupling_constant,
                spin,
                biaxiality,
                euler_angles,
                V_ZZ,
                applied_field,
                correlator_axis,
            )
            end = time.time()

            time_taken = end - start  # in seconds at this point

            if method == "matexp":
                matexp_times[s] = time_taken
            else:
                eigen_times[s] = time_taken

    return matexp_times, eigen_times


def SiteCorrelatorSpeedTestGraphing(
    t_test,
    n_site_tests,
    n_input_tests,
    correlator_axis="x",
    nuclear_species="Ga69",
    region_bounds=[100, 1200, 200, 1000],
):
    matexp_average_times = np.zeros(n_input_tests)
    matexp_std_dev = np.zeros(n_input_tests)
    eigen_average_times = np.zeros(n_input_tests)
    eigen_std_dev = np.zeros(n_input_tests)

    # t_test determines if we are testing the impact on time or applied field
    if t_test:
        param = "Waiting Times"
        title_param = "waiting_time"
        unit = "s"
        t_range = np.linspace(0, 1, n_input_tests) * 1e-8  # set times to change
        B_range = np.ones(n_input_tests)  # set B field always to 1
        x_axis = t_range
    else:
        param = "B Fields"
        title_param = "applied_field"
        unit = "T"
        t_range = np.zeros(n_input_tests)  # set t always to 0
        B_range = np.linspace(0, 1, n_input_tests)
        x_axis = B_range

    for i in range(n_input_tests):
        t = t_range[i]
        applied_field = B_range[i]
        temp_mat, temp_eigen = SiteCorrelatorSpeedTest(
            t,
            applied_field,
            correlator_axis,
            nuclear_species,
            n_site_tests,
            region_bounds=[100, 1200, 200, 1000],
        )

        matexp_average_times[i] = np.average(temp_mat)
        matexp_std_dev[i] = np.std(temp_mat)

        eigen_average_times[i] = np.average(temp_eigen)
        eigen_std_dev[i] = np.std(temp_eigen)

    title = "speed_test_for_{}_using_{}_along_the_{}_axis".format(
        title_param, nuclear_species, correlator_axis
    )

    plt.figure(figsize=(12, 8))
    eb1 = plt.errorbar(
        x_axis,
        matexp_average_times,
        yerr=matexp_std_dev,
        label="Matrix Exponentials",
        lolims=True,
        uplims=True,
    )
    eb2 = plt.errorbar(
        x_axis,
        eigen_average_times,
        yerr=eigen_std_dev,
        label="EigenVectors",
        lolims=True,
        uplims=True,
        linestyle="dashed",
    )

    # these 2 lines are stolen from Stack Overflow, they make the errorbars dashed, eb2[-1][0] does the upper bars, and eb2[-1][1] does the lower bars
    eb2[-1][0].set_linestyle("dashed")
    eb2[-1][1].set_linestyle("dashed")

    plt.title("Speed Comparison for Different " + param)
    plt.xlabel(param + "({})".format(unit))
    plt.ylabel("Calculation Time (s)")
    plt.legend()
    plt.savefig(title)


# If statements copied over from correlator_calc.py on 26/04/20

if CorrelatorTesting:
    print("Trying Out the Correlator Functions")
    total_start = time.time()
    for nuclear_species in ["In115"]:  # ["Ga69", "Ga71", "As75", "In115"]:
        total_timer = 0
        for b_field in [1]:  # np.linspace(0,1,5):
            species_timer = 0
            print(nuclear_species, str(b_field) + "T")
            start = time.time()
            species_correlator = OneTimeOneBFieldLatticeCorrelatorCalculator(
                t=0,
                applied_field=b_field,
                correlator_axis="x",
                nuclear_species=nuclear_species,
                region_bounds=region_bounds,
            )
            end = time.time()
            species_timer += (end - start) / 60
            # print "Time Taken = {:3f} minutes".format((end-start)/60)
            print(
                "Average Correlator for {} at a Field of {} is: {}".format(
                    nuclear_species, b_field, species_correlator
                )
            )
        print(
            "Time taken for all {} calcs: {} minutes".format(
                nuclear_species, species_timer
            )
        )
    total_end = time.time()
    print(
        "Total time taken (including prints etc): {} minutes".format(
            (end - start) / 560
        )
    )

if CorrelatorSpeedTesting:
    print("Starting Speed Testing")
    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="Ga69",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="Ga69",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="Ga69",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="Ga69",
        region_bounds=region_bounds,
    )

    print("Ga69 Batch Done")

    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="Ga71",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="Ga71",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="Ga71",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="Ga71",
        region_bounds=region_bounds,
    )

    print("Ga71 Batch Done")

    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="As75",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="As75",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="As75",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="As75",
        region_bounds=region_bounds,
    )

    print("As75 Batch Done")

    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="In115",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="x",
        nuclear_species="In115",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=True,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="In115",
        region_bounds=region_bounds,
    )
    SiteCorrelatorSpeedTestGraphing(
        t_test=False,
        n_site_tests=500,
        n_input_tests=50,
        correlator_axis="z",
        nuclear_species="In115",
        region_bounds=region_bounds,
    )

    print("All Done")


# copied from parallel_correlator.py on 26/6/20
def TestingParrallelism():
    # testing parameters at first
    elapsed_time = 0
    applied_field = 1
    correlator_axis = "z"
    nuclear_species = "In115"
    region_bounds = [100, 1200, 439, 880]
    step_size = 10

    n_sites, zipped_params = ParameterZipper(
        elapsed_time,
        applied_field,
        correlator_axis,
        nuclear_species,
        region_bounds,
        step_size,
    )

    n_cores = multiprocessing.cpu_count()

    # allocated_at_once = int(n_sites/n_cores)

    loop_start = time.time()
    allocated_at_once = int(n_sites / n_cores)
    print(f"There are {n_sites} atomic sites. We have {n_cores} cores to do the work.")
    for jobs_at_once in [1, 10, 50, 100, int(n_sites / n_cores)]:
        print(
            "----------------------------------------------------------------------------"
        )
        print(f"\nStarting Loop Here. Allocating {jobs_at_once} tasks at once.")
        start = time.time()
        with multiprocessing.Pool() as pool:
            site_correlators = pool.starmap(
                bqf.SiteCorrelatorCalculatorInParallel,
                zipped_params,
                chunksize=jobs_at_once,
            )
            pool.close()

        print("\nCorrelator calculation complete. Moving on to averaging.")
        correlator = np.average(site_correlators)
        print(f"\nAveraging Complete.")
        end = time.time()
        time_taken = np.around((end - start) / 60, 2)

        print(
            f"\nCorrelator for {nuclear_species} found to be {correlator}, in {time_taken} minutes. We used {n_cores} cores to do this, and with a chunksize of {jobs_at_once}."
        )
        print(
            "----------------------------------------------------------------------------"
        )
    loop_end = time.time()
    loop_time_taken = np.around((loop_end - loop_start) / 60, 2)
    print(f"It took {loop_time_taken} minutes to do all this.")


def StepSizeTesting():
    min_exp = -9
    max_exp = 0
    n_times = 5

    min_step_size = 10
    max_step_size = 80
    n_step_sizes = 9

    timerange = np.logspace(min_exp, max_exp, n_times)
    step_range = [1] + list(np.arange(10, 100, 10))
    print(step_range)
    big_results_array = np.zeros((2, n_times, n_step_sizes))

    for s, step_size in enumerate(step_range):
        print(step_size)
        big_results_array[:, :, s] = CorrelatorTimeCalculator(
            timerange, 1, "z", "In115", [100, 1200, 439, 880], step_size, 25
        )
        print(
            f"Time taken: {time_taken}s, using step size {step_size}, to find: {big_results_array[1,:,s]}"
        )
