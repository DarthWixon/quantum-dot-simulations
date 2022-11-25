"""
A module detailing the physical properties of the nuclear species found in InGaAs quantum dots.

Each species is represented using a dictionary, which are then stored in the species_dict dictionary.
The required properties are:
    short_name (str): The short form name of the species, eg In_115.
    name (str): The full name of the species, eg Indium 115.
    particle_spin (float): The nuclear spin of the particle, eg 4.5
    zeeman_frequency_per_tesla (float): The energy splitting due to the Zeeman interaction, in units of Hz/T
    quadrupole_moment (float): The quadrupolar moment of that nucleus, analagous to the dipole moment, in units of m**2.
    S11 (float): The value of the Sxxxx component of the gradient elastic tensor.
    S44 (float): The value of the Syzyz component of the gradient elastic tensor.

"""

In_115 = {
	"short_name": 					"In_115",
	"name":							"Indium 115",
	"particle_spin":				9.0/2,
	"zeeman_frequency_per_tesla":	9.33e6, #in units of Hz/T, using values in Stockill paper
    "quadrupole_moment":            770e-31, # measured in m**2
    # taken from Sundfors (1974), converted to SI units by multiplying by 2997924.580 (from cgs units)
    # if I can find updated values that would be great, have taken S11 to be +ve
    "S11":							5.0065340486e+22,
    "S44":							-2.997924580e+22
}

Ga_69 = {
	"short_name": 					"Ga_69",
	"name":							"Gallium 69",
	"particle_spin":				3.0/2,
	"zeeman_frequency_per_tesla":	10.22e6, #in units of Hz/T, using values in Stockill paper
    "quadrupole_moment":            172e-31, # measured in m**2
    # values for S11 and S44 taken from Checkovich Complete/Cross Calibration Papers
    "S11":							-22e21,
    "S44":							-0.4 * -22e22
}

Ga_71 = {
	"short_name": 					"Ga_71",
	"name":							"Gallium 71",
	"particle_spin":				3.0/2,
	"zeeman_frequency_per_tesla":	12.98e6, #in units of Hz/T, using values in Stockill paper
    "quadrupole_moment":            107e-31, # measured in m**2
    # taken from Sundfors (1974), converted to SI units by multiplying by 2997924.580 (from cgs units)
    # if I can find updated values that would be great, have taken S11 to be +ve
    "S11":							2.73e22,
    "S44":							-2.73e22
}

As_75 = {
	"short_name": 					"As_75",
	"name":							"Arsenic 75",
	"particle_spin":				3.0/2,
	"zeeman_frequency_per_tesla":	7.22e6, #in units of Hz/T, using values in Stockill paper
    "quadrupole_moment":            314e-31, # measured in m**2
    # values for S11 and S44 taken from Checkovich Complete/Cross Calibration Papers
    "S11":							24.2e21,
    "S44":							1.98*24.2e21
}

species_dict = {
        "Ga69" : Ga_69,
        "Ga71" : Ga_71,
        "As75" : As_75,
        "In115": In_115
}