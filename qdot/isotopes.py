"""
Physical properties of the nuclear species found in InGaAs quantum dots.

Each species dict contains:
    short_name (str): e.g. "In_115"
    name (str): e.g. "Indium 115"
    graph_name (str): label used in plot titles
    particle_spin (float): nuclear spin quantum number
    zeeman_frequency_per_tesla (float): Zeeman splitting in Hz/T
    quadrupole_moment (float): nuclear quadrupole moment in m²
    S11 (float): Sxxxx component of the gradient-elastic tensor
    S44 (float): Syzyz component of the gradient-elastic tensor

Two parameter sets are provided:
    species_dict     — Checkhovich et al. values (current best)
    old_species_dict — Sundfors (1974) values, for comparison with older literature
"""

from typing import TypedDict


class SpeciesParameters(TypedDict):
    short_name: str
    name: str
    graph_name: str
    particle_spin: float
    zeeman_frequency_per_tesla: float
    quadrupole_moment: float
    S11: float
    S44: float


In_115 = {
    "short_name": "In_115",
    "name": "Indium 115",
    "graph_name": "In115",
    "particle_spin": 9.0 / 2,
    "zeeman_frequency_per_tesla": 9.33e6,
    "quadrupole_moment": 770e-31,
    "S11": 5.01e22,
    "S44": -2.998e22,
}

Ga_69 = {
    "short_name": "Ga_69",
    "name": "Gallium 69",
    "graph_name": "Ga69",
    "particle_spin": 3.0 / 2,
    "zeeman_frequency_per_tesla": 10.22e6,
    "quadrupole_moment": 172e-31,
    "S11": -22e21,
    "S44": -0.4 * -22e21,
}

Ga_71 = {
    "short_name": "Ga_71",
    "name": "Gallium 71",
    "graph_name": "Ga71",
    "particle_spin": 3.0 / 2,
    "zeeman_frequency_per_tesla": 12.98e6,
    "quadrupole_moment": 107e-31,
    "S11": 2.73e22,
    "S44": -2.73e22,
}

As_75 = {
    "short_name": "As_75",
    "name": "Arsenic 75",
    "graph_name": "As75",
    "particle_spin": 3.0 / 2,
    "zeeman_frequency_per_tesla": 7.22e6,
    "quadrupole_moment": 314e-31,
    "S11": 24.2e21,
    "S44": 1.98 * 24.2e21,
}

In_115_Old = {
    "short_name": "In_115",
    "name": "Indium 115",
    "graph_name": "In115",
    "particle_spin": 9.0 / 2,
    "zeeman_frequency_per_tesla": 9.33e6,
    "quadrupole_moment": 770e-31,
    "S11": 5.01e22,
    "S44": -2.998e22,
}

Ga_69_Old = {
    "short_name": "Ga_69",
    "name": "Gallium 69",
    "graph_name": "Ga69",
    "particle_spin": 3.0 / 2,
    "zeeman_frequency_per_tesla": 10.22e6,
    "quadrupole_moment": 172e-31,
    "S11": 2.73e22,
    "S44": -2.76e22,
}

Ga_71_Old = {
    "short_name": "Ga_71",
    "name": "Gallium 71",
    "graph_name": "Ga71",
    "particle_spin": 3.0 / 2,
    "zeeman_frequency_per_tesla": 12.98e6,
    "quadrupole_moment": 107e-31,
    "S11": 2.73e22,
    "S44": -2.73e22,
}

As_75_Old = {
    "short_name": "As_75",
    "name": "Arsenic 75",
    "graph_name": "As75",
    "particle_spin": 3.0 / 2,
    "zeeman_frequency_per_tesla": 7.22e6,
    "quadrupole_moment": 314e-31,
    "S11": 3.96e22,
    "S44": 7.94e22,
}

species_dict: dict[str, SpeciesParameters] = {
    "Ga69": Ga_69,
    "Ga71": Ga_71,
    "As75": As_75,
    "In115": In_115,
}

old_species_dict: dict[str, SpeciesParameters] = {
    "Ga69": Ga_69_Old,
    "Ga71": Ga_71_Old,
    "As75": As_75_Old,
    "In115": In_115_Old,
}
