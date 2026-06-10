import pytest
from qdot.isotopes import species_dict, old_species_dict

REQUIRED_FIELDS = [
    "short_name",
    "name",
    "graph_name",
    "particle_spin",
    "zeeman_frequency_per_tesla",
    "quadrupole_moment",
    "S11",
    "S44",
]
ALL_SPECIES = ["Ga69", "Ga71", "As75", "In115"]


def test_all_species_present():
    for key in ALL_SPECIES:
        assert key in species_dict
        assert key in old_species_dict


def test_all_required_fields_present():
    for key in ALL_SPECIES:
        for field in REQUIRED_FIELDS:
            assert field in species_dict[key], f"{key} missing field {field}"
            assert field in old_species_dict[key], f"{key} (old) missing field {field}"


def test_particle_spins():
    assert species_dict["Ga69"]["particle_spin"] == 1.5
    assert species_dict["Ga71"]["particle_spin"] == 1.5
    assert species_dict["As75"]["particle_spin"] == 1.5
    assert species_dict["In115"]["particle_spin"] == 4.5


def test_checkhovich_ga69_s11_is_negative():
    # Checkhovich updated Ga69 S11 to negative; Sundfors had it positive
    assert species_dict["Ga69"]["S11"] < 0
    assert old_species_dict["Ga69"]["S11"] > 0


def test_zeeman_frequencies_positive():
    for key in ALL_SPECIES:
        assert species_dict[key]["zeeman_frequency_per_tesla"] > 0


def test_quadrupole_coupling_matches_formula():
    import scipy.constants as const
    from qdot.isotopes import species_dict, quadrupole_coupling

    species = species_dict["Ga69"]
    spin = species["particle_spin"]
    Q = species["quadrupole_moment"]
    expected = (3 * const.e * Q) / (2 * const.h * spin * (2 * spin - 1))
    assert quadrupole_coupling(species) == pytest.approx(expected)


def test_quadrupole_coupling_positive_for_all_species():
    from qdot.isotopes import species_dict, quadrupole_coupling

    for species in species_dict.values():
        assert quadrupole_coupling(species) > 0
