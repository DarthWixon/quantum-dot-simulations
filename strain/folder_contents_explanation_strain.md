# Strain File Explanation

#### File list (and explanation) as of 18/08/20

##### Main folder

- 1d_strain_toy_model
  - looks like a pretty basic test of the optimisation functions
  - can be safely deleted
- edge_counting_test
  - testing ground for function that counts and plots the number of edges in a 2D graph
  - works fine and makes a neat plot quite easily
  - maybe should be moved to the testing folder?
- neat_strain_toy_model
  - exactly what it says on the tin
  - needs to be documented properly, but otherwise works well and shouldn't be touched!
- quadrupolar_interaction_from_strain_field
  - this is the one that recreates the plot from Sokolov, and then compares it to the same plot as from Checkovich
  - should probably be in the quadrupolar folder, but should be easy to extend to make the colour plot I want!!
  - has a load of "to do" stuff at the bottom, not sure what they're for, but I don't think they're currently relevant
- tersoff_cutoff_func_test
  - displays the cutoff ranges for the Abell-Tersoff potentials
  - doesn't give a particularly neat graph
  - not sure why I want this, but I have the graph saved so I don't want to delete it just in case
- toy_model_timer
  - times how long it takes to find the relaxed state for grids of different sizes
  - could be fun to parallelise, but not needed right now
  - does make a pretty graph tbf

##### Testing folder

- eigenvalues_speed_testing
  - just tests how long it takes qutip to find eigenvalues and eigenvectors
  - not needed at all
  - delete it

- euler_angles
  - only contains the word "Non"
  - delete
- loading_test
  - does some weird tests on loading objects etc etc
  - tries to load a data file in a very old naming format and fails
  - I see no value in fixing this
  - delete it
- where_testing
  - me mucking about with the np.where functions
  - not needed any more

#### Actions done on 19/08/20

1. deleted the following:
   - testing folder and all it's contents
   - 1d_strain_toy_model
2. moved quadrupolar_field_from_strain_field to the quadrupolar folder