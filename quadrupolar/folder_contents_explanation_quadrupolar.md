# What the code does:

#### Things done on 24/08/20

1. deleted a load of other files that basically all did splitting calculations, gonna make a single good version instead, left with:
   - backbone_quadrupolar_functions (new version, still need a central method for this)
   - isotope_parameters (new version, again need a central store for it)
   - quadrupolar_splittings
   - ratio_of_QI_strengths
   - comparing_sokolov_and_checkhovich_QI_graphs
2. Have put backbone_quadrupolar_functions and isotope_parameters in their own folder :crossed_fingers: nothing changes

#### Things done on 19/08/20

1. deleted the following files:
   - understanding_quadrupolar_term
   - test_ham_exp
   - maths_functions_testing
   - list_test
   - isotope_dicts_testing
   - isotope_correlator_graph
   - exponential_methods_speed_testing
   - contour_plotting
2. moved the file quadrupolar_field_from_strain_field here from the strain folder
3. replaced the isotope_parameters file with the updated one
   - I need to make a system that has this file in a central location

#### File list (and brief explanation) as of 18/08/20:

- 2d_spin_correlator_graph
  - I think this was meant to map the spin correlator over time
  - doesn't currently work, as the version of backbone_quadrupolar_functions it uses doesn't have a SpinCorrelator function
  - probably not needed here, but won't delete just yet as I need to check the correlator_calcs folder
- backbone_quadrupolar_functions
  - very old version of the standard set of functions, this one contains some attempts to do the integrals from Stockill that I obviously no longer need to do
  - need to find a way to keep this updated, might just copy it from the correlator_calcs folder for now
  - need to check if the other functions in this folder depend on weird functions in here, as I've only just realised they're all different
- basic_splitting_graph
  - gives diagrams of quadrupolar energy levels
  - currently takes hardcoded parameters of Euler angles, spin and biaxiality
- comparison_of_hamiltonian_methods
  - has functions for comparing the difference between 2 methods of calculating the quadrupolar Hamiltonians
  - needs further investigation
- contour_plotting
  - seems to be an unfinished script to plot out energy gaps with respect to both biaxiality and Euler angle
  - depends on a weird arcane function that's only present in this folder's version of backbone_quadrupolar_functions
  - can be deleted as I don't think it will make a good graph
- correlator_calc_v2
  - old version of the correlator_calc programme
  - has functions for: 
    - Loading Sokolov Data
    - extracting Euler angles from rotation matrices
    - calculating the EFG tensor from the Sokolov data
    - saving and loading EFG tensor data arrays
    - calculating the equivalent B field of the quadrupolar interaction
    - plotting the equivalent B field
  - probably could do with streamlining and renaming to make it specific to just quadrupolar stuff
- exponential_methods_speed_testing
  - has functions to compare different methods of exponentiating a Hamiltonian, useful as I need to do that many many times
  - compares the Qutip methods with one Brian wrote
  - pretty much irrelevant now, as I've lost access to Brian's code and can use BlueCrystal to do big calculations if I need to
  - can probably be deleted
- isotope_correlator_graph
  - finds the spin correlator for a particular isotope as a function of time
  - does it using the integrals from Stockill, but I don't think it does it right
  - don't need these integrals anymore, so I can happily delete this
- isotope_dicts_testing
  - checks to see if the isotope_parameters file can be imported properly
  - no longer needed, can be deleted
- isotope_parameters
  - the file that lists the data parameters for each isotope
  - need to keep this in a central place as well, as I don't know if this version is the latest one
- list_test
  - does some stuff with lists, obviously needed it once, but not anymore
  - can be deleted
- maths_functions_testing
  - doing some maths to do with finding factors of a number from square roots
  - can be deleted
- ratio_of_QI_strength
  - plots a graph of how the relative strength of the QI changes with spin
- scaled_B_field_splitting
  - looks to be code for plotting transition frequencies as a function of B field
  - not sure how it differs from the scaled_quadrupolar_angle_splitting file
  - doesn't currently run, I think because it's built for an old version of backbone_quadrupolar_functions
- scaled_quadrupolar_angle_splitting
  - plots out the quadrupolar splitting as a function of applied B field
  - needs to be expanded and worked on, but a good idea that I can work on
- spin_correlator_graph
  - plots out the difference between the 2 methods of calculating individual spin correlators (matrix exponentials and eigenvectors)
  - output graph needs tidying, but a good start
- test_ham_exp
  - doing some kind of test comparing Brian's method of exponentiating Hamiltonians
  - no longer works as I've lost access to that code
  - can be deleted
- understanding_quadrupolar_term
  - plots out the energy level splitting as a function of biaxiality
  - compares it to the spin-twisting Hamiltonian, which I no longer care about
