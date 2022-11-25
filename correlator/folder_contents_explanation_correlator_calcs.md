# File Explanation and Description

#### Changes on 24/08/20

1. Have put backbone_quadrupolar_functions and isotope_parameters in their own folder :crossed_fingers: nothing changes

#### List (and brief explanations) of files as of 19/08/20

- backbone_quadrupolar_functions
  - the key functions needed to do my calculations
  - well documented and tested, should be the gold standard if I make a common version
- conc_recreation
  - recreates the graphs of concentration as seen in the Sokolov paper
  - needs documenting, but works well
- correlator_calc
  - overall code that can do calculations of correlators across the dot
  - thoroughly documented
- isotope_parameters
  - the physical parameters of each isotope in the dot
  - this is the gold standard of this file, and should be used if I decide to make a common version
- parallel_correlator
  - does much the same job as the correlator_calc code, but in parallel
  - well documented
- testing_functions
  - contains functions which test things like the speed of correlator calculation
  - works
  - undocumented



#### General Thoughts

- almost everything in here is well documented and tested, so it should all work off the bat
  - update - it does!