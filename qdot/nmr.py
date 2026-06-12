"""
NMR absorption spectra for nuclear spins in InGaAs quantum dots.

Computes transition rates between eigenstates of the nuclear Hamiltonian
under an RF perturbation, building up an absorption spectrum by summing
over all allowed transitions at each RF frequency.

Beyond the single-site spectrum, this module provides the multi-site
parallel engine (absorption_map / summed_absorption_spectrum), the
concentration-weighted experimental simulation, the combined all-species
spectrum, and the RF pulse selectivity analysis from the original research
scripts.
"""

import math
import multiprocessing
import pathlib
from itertools import permutations

import numpy as np
from scipy.integrate import simpson

from qdot.isotopes import species_dict, quadrupole_coupling
from qdot.io import load_efg, load_concentration_data, SOKOLOV_DOT_REGION
from qdot.sites import random_locations
from qdot.hamiltonians import (
    faraday_hamiltonian,
    voigt_hamiltonian,
    rf_hamiltonian,
    transition_rate,
)

_VALID_SPECIES = frozenset(species_dict)


def _single_site_spectrum(
    spin: float,
    zeeman_term: float,
    quadrupolar_term: float,
    biaxiality: float,
    alpha: float,
    beta: float,
    gamma: float,
    field_geometry: str,
    rf_freqs: np.ndarray,
    rf_field: float,
) -> np.ndarray:
    """
    Absorption spectrum of a single nucleus from explicit Hamiltonian terms.

    Module-level so multiprocessing can pickle it for the parallel engine.
    """
    if field_geometry == "Faraday":
        H = faraday_hamiltonian(
            zeeman_term, quadrupolar_term, biaxiality, spin, alpha, beta, gamma
        )
        H_rf = rf_hamiltonian(spin, rf_field, 0, 0)
    elif field_geometry == "Voigt":
        H = voigt_hamiltonian(
            zeeman_term, quadrupolar_term, biaxiality, spin, alpha, beta, gamma
        )
        H_rf = rf_hamiltonian(spin, 0, 0, rf_field)
    else:
        raise ValueError(
            f"field_geometry must be 'Faraday' or 'Voigt', got {field_geometry!r}"
        )

    H = H.tidyup()
    eigenvalues, eigenvectors = H.eigenstates()
    eigenenergies = np.real_if_close(eigenvalues)
    index_list = np.arange(len(eigenenergies))

    # Each pair's matrix element is independent of the RF frequency, so compute
    # it once per pair and let transition_rate broadcast over the whole
    # frequency array.
    rates = np.zeros(len(rf_freqs))
    for pair in permutations(index_list, 2):
        rates += transition_rate(
            H_rf,
            eigenvectors[pair[0]],
            eigenvectors[pair[1]],
            eigenenergies[pair[0]],
            eigenenergies[pair[1]],
            rf_freqs,
        )
    return rates


def _site_parameters(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    region_bounds: list[int],
    locations: list[tuple[int, int]],
    step_size: int,
    use_sundfors: bool,
    real_strain: bool,
    mirror_type: str,
) -> tuple[float, float, list[tuple[float, float, float, float, float]]]:
    """
    Load the EFG archive once and extract per-site Hamiltonian parameters.

    Returns:
        spin, zeeman_per_tesla, and a list of
        (biaxiality, quadrupolar_term, alpha, beta, gamma) per location.
    """
    eta_arr, _, _, V_ZZ_arr, euler_arr = load_efg(
        data_dir,
        nuclear_species,
        region_bounds,
        step_size=step_size,
        use_sundfors=use_sundfors,
        real_strain=real_strain,
        mirror_type=mirror_type,
    )
    species = species_dict[nuclear_species]
    qcc = quadrupole_coupling(species)
    n_cols = V_ZZ_arr.shape[1]

    site_params = []
    for x, y in locations:
        alpha, beta, gamma = euler_arr[x * n_cols + y]
        site_params.append((eta_arr[x, y], qcc * V_ZZ_arr[x, y], alpha, beta, gamma))
    return species["particle_spin"], species["zeeman_frequency_per_tesla"], site_params


def absorption_spectrum(
    nuclear_species: str,
    applied_field: float,
    field_geometry: str,
    rf_freq_list: np.ndarray,
    location: tuple[int, int],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    step_size: int = 1,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
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
        step_size (int): Subsample interval the EFG archive was saved with.
        rf_field (float): RF field amplitude. Default 5 mT.
        use_sundfors (bool): Use Sundfors parameter set if True.
        real_strain (bool): False to use a mirrored-strain EFG archive.
        mirror_type (str): Mirror variant, only used when real_strain=False.

    Returns:
        ndarray: Transition rate at each RF frequency, shape (len(rf_freq_list),).
    """
    if nuclear_species not in _VALID_SPECIES:
        raise ValueError(
            f"nuclear_species must be one of {sorted(_VALID_SPECIES)}, got {nuclear_species!r}"
        )
    if region_bounds is None:
        region_bounds = SOKOLOV_DOT_REGION

    spin, zeeman_per_tesla, site_params = _site_parameters(
        data_dir,
        nuclear_species,
        region_bounds,
        [location],
        step_size,
        use_sundfors,
        real_strain,
        mirror_type,
    )
    eta, quad_term, alpha, beta, gamma = site_params[0]
    return _single_site_spectrum(
        spin,
        zeeman_per_tesla * applied_field,
        quad_term,
        eta,
        alpha,
        beta,
        gamma,
        field_geometry,
        np.asarray(rf_freq_list),
        rf_field,
    )


def varied_field_spectra(
    nuclear_species: str,
    applied_field_list: list[float],
    field_geometry: str,
    rf_freq_list: np.ndarray,
    location: tuple[int, int],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    step_size: int = 1,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
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
        step_size (int): Subsample interval the EFG archive was saved with.
        rf_field (float): RF field amplitude. Default 5 mT.
        use_sundfors (bool): Use Sundfors parameter set if True.
        real_strain (bool): False to use a mirrored-strain EFG archive.
        mirror_type (str): Mirror variant, only used when real_strain=False.

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
                step_size=step_size,
                rf_field=rf_field,
                use_sundfors=use_sundfors,
                real_strain=real_strain,
                mirror_type=mirror_type,
            ),
        }
        for B in applied_field_list
    ]


def absorption_map(
    nuclear_species: str,
    applied_field_list: np.ndarray,
    field_geometry: str,
    rf_freq_list: np.ndarray,
    locations: list[tuple[int, int]],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    step_size: int = 1,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
    processes: int | None = None,
) -> np.ndarray:
    """
    2D NMR absorption map over applied field and RF frequency.

    For each applied field, computes the single-site spectrum at every
    location in parallel (multiprocessing.Pool) and sums over sites. This is
    the multi-site engine behind the field-frequency heatmaps; the EFG
    archive is loaded once for the whole map.

    Args:
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        applied_field_list (ndarray): Applied fields to sweep, in Tesla.
        field_geometry (str): "Faraday" or "Voigt".
        rf_freq_list (ndarray): RF frequencies to evaluate (Hz).
        locations (list of (int, int)): Lattice sites to sum over.
        data_dir: Directory containing pre-computed EFG archives.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        step_size (int): Subsample interval the EFG archive was saved with.
        rf_field (float): RF field amplitude. Default 5 mT.
        use_sundfors (bool): Use Sundfors parameter set if True.
        real_strain (bool): False to use a mirrored-strain EFG archive.
        mirror_type (str): Mirror variant, only used when real_strain=False.
        processes (int): Worker process count. Default: all available cores.

    Returns:
        ndarray: Summed absorption, shape (len(applied_field_list), len(rf_freq_list)).
    """
    if nuclear_species not in _VALID_SPECIES:
        raise ValueError(
            f"nuclear_species must be one of {sorted(_VALID_SPECIES)}, got {nuclear_species!r}"
        )
    if region_bounds is None:
        region_bounds = SOKOLOV_DOT_REGION

    rf_freqs = np.asarray(rf_freq_list)
    data = np.zeros((len(applied_field_list), len(rf_freqs)))
    if not locations:
        return data

    spin, zeeman_per_tesla, site_params = _site_parameters(
        data_dir,
        nuclear_species,
        region_bounds,
        locations,
        step_size,
        use_sundfors,
        real_strain,
        mirror_type,
    )

    with multiprocessing.Pool(processes) as pool:
        for b, applied_field in enumerate(applied_field_list):
            zeeman_term = zeeman_per_tesla * applied_field
            args = [
                (
                    spin,
                    zeeman_term,
                    quad_term,
                    eta,
                    alpha,
                    beta,
                    gamma,
                    field_geometry,
                    rf_freqs,
                    rf_field,
                )
                for eta, quad_term, alpha, beta, gamma in site_params
            ]
            site_spectra = pool.starmap(_single_site_spectrum, args)
            data[b] = np.sum(site_spectra, axis=0)

    return data


def summed_absorption_spectrum(
    nuclear_species: str,
    applied_field: float,
    field_geometry: str,
    rf_freq_list: np.ndarray,
    locations: list[tuple[int, int]],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    step_size: int = 1,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
    processes: int | None = None,
) -> np.ndarray:
    """
    Absorption spectrum at one applied field, summed over many lattice sites.

    A single row of absorption_map; see there for argument details.

    Returns:
        ndarray: Summed absorption at each RF frequency, shape (len(rf_freq_list),).
    """
    return absorption_map(
        nuclear_species,
        np.array([applied_field]),
        field_geometry,
        rf_freq_list,
        locations,
        data_dir,
        region_bounds,
        step_size=step_size,
        rf_field=rf_field,
        use_sundfors=use_sundfors,
        real_strain=real_strain,
        mirror_type=mirror_type,
        processes=processes,
    )[0]


def experimental_nmr_simulation(
    field_geometry: str,
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    n_locations: int,
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    step_size: int = 1,
    rf_field: float = 5e-3,
    arsenic_fraction: float = 0.5,
    rng: np.random.Generator | int | None = None,
    processes: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Concentration-weighted multi-species NMR simulation of the dot.

    Splits n_locations lattice sites into species populations from the In
    concentration map — In115 sites in proportion to the mean In
    concentration, an arsenic_fraction share of As75, the remainder Ga69 —
    samples random sites for each species, and computes each species'
    absorption map. Following the original research code, Ga71 is excluded;
    the population heuristic is recorded in human-todo. for physics review.

    Returns the per-species maps so the caller can choose between a summed
    rendering (plot_nmr_map of the total) and a layered one
    (plot_nmr_map_layered).

    Args:
        field_geometry (str): "Faraday" or "Voigt".
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        n_locations (int): Total number of nuclei to simulate.
        data_dir: Directory containing EFG archives and concentration data.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        step_size (int): Subsample interval of the archives.
        rf_field (float): RF field amplitude. Default 5 mT.
        arsenic_fraction (float): Fraction of sites assigned to As75. Default 0.5.
        rng: numpy Generator, integer seed, or None for fresh entropy.
        processes (int): Worker process count. Default: all available cores.

    Returns:
        dict: Species name → absorption map, shape (n_fields, n_freqs), for
        "In115", "Ga69" and "As75".
    """
    if region_bounds is None:
        region_bounds = SOKOLOV_DOT_REGION

    conc_data = load_concentration_data(data_dir, region_bounds, step_size)
    mean_in_conc = float(np.mean(conc_data))

    n_in = int(math.ceil(mean_in_conc * n_locations))
    n_as = int(math.ceil(arsenic_fraction * n_locations))
    n_ga = n_locations - n_in - n_as
    if n_ga < 0:
        raise ValueError(
            f"n_locations={n_locations} is too small for the population split "
            f"(In: {n_in}, As: {n_as})"
        )

    rng = np.random.default_rng(rng)
    maps = {}
    for nuclear_species, n_sites in (("In115", n_in), ("Ga69", n_ga), ("As75", n_as)):
        locations = random_locations(n_sites, conc_data.shape, rng)
        maps[nuclear_species] = absorption_map(
            nuclear_species,
            applied_field_list,
            field_geometry,
            rf_freq_list,
            locations,
            data_dir,
            region_bounds,
            step_size=step_size,
            rf_field=rf_field,
            processes=processes,
        )
    return maps


def combined_spectrum(
    applied_field: float,
    field_geometry: str,
    rf_freq_list: np.ndarray,
    locations: list[tuple[int, int]],
    data_dir: pathlib.Path | str,
    region_bounds: list[int] | None = None,
    step_size: int = 1,
    rf_field: float = 5e-3,
    use_sundfors: bool = False,
    processes: int | None = None,
) -> np.ndarray:
    """
    All-species absorption spectrum at one applied field, normalised to 1.

    Sums the spectra of all four species over the same site list — the input
    to peak identification and RF pulse selectivity analysis.

    Args:
        applied_field (float): Static magnetic field in Tesla.
        field_geometry (str): "Faraday" or "Voigt".
        rf_freq_list (ndarray): RF frequencies in Hz.
        locations (list of (int, int)): Lattice sites shared by all species.
        data_dir: Directory containing pre-computed EFG archives.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        step_size (int): Subsample interval of the archives.
        rf_field (float): RF field amplitude. Default 5 mT.
        use_sundfors (bool): Use Sundfors parameter set if True.
        processes (int): Worker process count. Default: all available cores.

    Returns:
        ndarray: Normalised total absorption, shape (len(rf_freq_list),).
    """
    total = np.zeros(len(rf_freq_list))
    for nuclear_species in sorted(_VALID_SPECIES):
        total += summed_absorption_spectrum(
            nuclear_species,
            applied_field,
            field_geometry,
            rf_freq_list,
            locations,
            data_dir,
            region_bounds,
            step_size=step_size,
            rf_field=rf_field,
            use_sundfors=use_sundfors,
            processes=processes,
        )
    return total / np.amax(total)


def lorentzian_pulse(
    rf_freq_list: np.ndarray, centre: float, width: float
) -> np.ndarray:
    """
    Lorentzian (Cauchy) frequency profile of an RF pulse.

    Args:
        rf_freq_list (ndarray): Frequencies at which to evaluate the profile.
        centre (float): Pulse centre frequency, same units as rf_freq_list.
        width (float): Full width of the pulse; the Lorentzian half-width is
            width / 2.

    Returns:
        ndarray: Pulse profile (unit-area Cauchy distribution).
    """
    gamma = width / 2
    freqs = np.asarray(rf_freq_list)
    return 1 / (np.pi * gamma * (1 + ((freqs - centre) / gamma) ** 2))


def pulse_capture_fraction(
    spectrum: np.ndarray,
    rf_freq_list: np.ndarray,
    pulse_centre: float,
    pulse_width: float,
    window: tuple[float, float] | None = None,
) -> float:
    """
    Fraction of spectral weight a Lorentzian pulse captures.

    Multiplies the spectrum by the pulse profile (normalised to unit peak)
    and Simpson-integrates. With window=None the fraction is over the whole
    spectrum; with a (min_freq, max_freq) window both integrals are restricted
    to that band, giving the in-band capture fraction used to assess whether a
    pulse can address one species without touching others.

    Args:
        spectrum (ndarray): Absorption spectrum, e.g. from combined_spectrum.
        rf_freq_list (ndarray): Frequencies of the spectrum samples.
        pulse_centre (float): Pulse centre, same units as rf_freq_list.
        pulse_width (float): Pulse full width, same units as rf_freq_list.
        window (tuple): Optional (min_freq, max_freq) integration band.

    Returns:
        float: Captured fraction in [0, 1].
    """
    freqs = np.asarray(rf_freq_list)
    spectrum = np.asarray(spectrum)

    pulse = lorentzian_pulse(freqs, pulse_centre, pulse_width)
    pulse = pulse / np.amax(pulse)
    filtered = spectrum * pulse

    if window is not None:
        low, high = window
        low_idx = int(np.abs(freqs - low).argmin())
        high_idx = int(np.abs(freqs - high).argmin())
        selection = slice(low_idx, high_idx + 1)
    else:
        selection = slice(None)

    return float(
        simpson(filtered[selection], x=freqs[selection])
        / simpson(spectrum[selection], x=freqs[selection])
    )
