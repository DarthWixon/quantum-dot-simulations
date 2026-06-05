"""
NMR absorption spectra for nuclear spins in InGaAs quantum dots.

Computes transition rates between eigenstates of the nuclear Hamiltonian
under an RF perturbation, building up an absorption spectrum by summing
over all allowed transitions at each RF frequency.
"""

import pathlib

import numpy as np
import scipy.constants as const
from itertools import permutations

from qdot.isotopes import species_dict
from qdot.io import load_efg, SOKOLOV_DOT_REGION

_VALID_SPECIES = frozenset(species_dict)
from qdot.hamiltonians import (
    faraday_hamiltonian,
    voigt_hamiltonian,
    rf_hamiltonian,
    transition_rate,
)

h = const.h
e = const.e


def absorption_spectrum(
    nuclear_species: str,
    applied_field: float,
    field_geometry: str,
    rf_freq_list: np.ndarray,
    location: tuple[int, int],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
) -> np.ndarray:
    """
    NMR absorption spectrum at a single lattice site.

    Computes the transition rate sum at each RF frequency, using the
    full nuclear Hamiltonian (Zeeman + quadrupolar) at the given site.

    Args:
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        applied_field (float): Static magnetic field in Tesla.
        field_geometry (str): "Faraday" or "Voigt".
        rf_freq_list (ndarray): RF frequencies to evaluate (Hz).
        location (tuple): (x, y) lattice site indices into the EFG arrays.
        data_dir: Directory containing pre-computed EFG archives.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        rf_field (float): RF field amplitude. Default 5 mT.
        use_sundfors (bool): Use Sundfors parameter set if True.

    Returns:
        ndarray: Transition rate at each RF frequency, shape (len(rf_freq_list),).
    """
    if nuclear_species not in _VALID_SPECIES:
        raise ValueError(
            f"nuclear_species must be one of {sorted(_VALID_SPECIES)}, got {nuclear_species!r}"
        )
    if region_bounds is None:
        region_bounds = SOKOLOV_DOT_REGION

    x, y = location
    eta_arr, _, _, V_ZZ_arr, euler_arr = load_efg(
        data_dir,
        nuclear_species,
        region_bounds,
        use_sundfors=use_sundfors,
    )

    species = species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]
    qcc = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))

    eta = eta_arr[x, y]
    V_ZZ = V_ZZ_arr[x, y]
    site_idx = x * V_ZZ_arr.shape[1] + y
    alpha, beta, gamma = euler_arr[site_idx]

    zeeman_term = zeeman_per_tesla * applied_field
    quad_term = qcc * V_ZZ

    if field_geometry == "Faraday":
        H = faraday_hamiltonian(zeeman_term, quad_term, eta, spin, alpha, beta, gamma)
        H_rf = rf_hamiltonian(spin, rf_field, 0, 0)
    elif field_geometry == "Voigt":
        H = voigt_hamiltonian(zeeman_term, quad_term, eta, spin, alpha, beta, gamma)
        H_rf = rf_hamiltonian(spin, 0, 0, rf_field)
    else:
        raise ValueError(
            f"field_geometry must be 'Faraday' or 'Voigt', got {field_geometry!r}"
        )

    H = H.tidyup()
    eigenvalues, eigenvectors = H.eigenstates()
    eigenenergies = np.real_if_close(eigenvalues)
    index_list = np.arange(len(eigenenergies))

    rates = np.zeros(len(rf_freq_list))
    for r, rf_freq in enumerate(rf_freq_list):
        for pair in permutations(index_list, 2):
            rates[r] += transition_rate(
                H_rf,
                eigenvectors[pair[0]],
                eigenvectors[pair[1]],
                eigenenergies[pair[0]],
                eigenenergies[pair[1]],
                rf_freq,
            )

    return rates


def varied_field_spectra(
    nuclear_species: str,
    applied_field_list: list[float],
    field_geometry: str,
    rf_freq_list: np.ndarray,
    location: tuple[int, int],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
) -> list[dict]:
    """
    Compute absorption spectra at multiple applied field strengths.

    Args:
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        applied_field_list (list of float): Field values in Tesla.
        field_geometry (str): "Faraday" or "Voigt".
        rf_freq_list (ndarray): RF frequencies (Hz).
        location (tuple): (x, y) site index.
        data_dir: Directory containing pre-computed EFG archives.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        rf_field (float): RF field amplitude. Default 5 mT.
        use_sundfors (bool): Use Sundfors parameter set if True.

    Returns:
        list of dict: Each entry has keys "applied_field", "rf_freq_list", "data".
    """
    return [
        {
            "applied_field": B,
            "rf_freq_list": rf_freq_list,
            "data": absorption_spectrum(
                nuclear_species,
                B,
                field_geometry,
                rf_freq_list,
                location,
                data_dir,
                region_bounds,
                rf_field=rf_field,
                use_sundfors=use_sundfors,
            ),
        }
        for B in applied_field_list
    ]
