# File explanation for machine_gun folder

#### File list (and explanations) as of 19/08/20

- backbone_functions
  - core functions that permit the rest of the code to run
- CNOT_testing
  - contains functions to make operators for CNOT gates
  - probably not needed as qutip can do all this stuff way better
  - might be good in that it's specific to the machine gun
- density_matrix_machine_gun
  - unfinished code to replicate the machine gun using density matrices
  - doesn't currently run, most likely due to a python 2 -> python 3 conversion error?
    - this is my best guess
- errors_dictionary_functions
  - doesn't work, and seems to be intended to be a small part of a much bigger thing that never got made
- errors_testing
  - individual code to test whether or not the single qubit errors code works
- machine_gun_basics
  - unfished code that is supposed to combine all the other stuff together
- NFF_no_nuclear_spin_steady_state
  - built to simulate nuclear frequency focussing, uses a lot of variable names I don't understand, but does output a reasonable looking density matrix...
- perfect_machine_gun_testing
  - test code to see if the implementation of the machine gun without qubit errors works
  - it appears to, but again I need to dig down into it again to check properly
- single_qubit_operation_testing
  - testing of the code that generates single qubit operators
  - works well it seems
- standard_constructors
  - functions to construct common states in the CSMG
  - imported by loads of other things



### Overall Thoughts

As of 19/08/20, I don't really want to change anything in this folder. It all seems to work, it just isn't finished as I must have moved on to other things. Right now it isn't worth diving deeply into what's going on, but deleting things seems needless.