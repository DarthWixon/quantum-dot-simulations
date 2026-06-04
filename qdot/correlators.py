"""
Spin correlator calculations for nuclear spins in InGaAs quantum dots.

The spin-spin time correlation function <I_α(t) I_α(0)> is computed via
matrix exponentiation of the nuclear Hamiltonian at each lattice site and
then averaged over the dot region.

Serial and parallel implementations are both provided. The parallel version
uses multiprocessing.Pool.starmap and scales to available CPU cores.
"""

import logging
import multiprocessing
import pathlib
import numpy as np
import qutip
import scipy.constants as const

from qdot.isotopes import species_dict
from qdot.hamiltonians import faraday_hamiltonian
from qdot.io import load_efg, SOKOLOV_DOT_REGION

_VALID_SPECIES = frozenset(species_dict)

logger = logging.getLogger(__name__)


def spin_correlator(
    t: float, nuclear_hamiltonian: qutip.Qobj, spin_axis: str
) -> float | None:
    """
    Time correlation function <I_α(t) I_α(0)> for a single nuclear spin.

    Computed via Tr(ρ · e^{iHt} I_α e^{-iHt} I_α) with ρ = I/d (maximally mixed).

    Args:
        t (float): Time in seconds.
        nuclear_hamiltonian (Qobj): Single-site nuclear Hamiltonian.
        spin_axis (str): "x", "y", or "z".

    Returns:
        float: Correlator value, or None if spin_axis is invalid.
    """
    if spin_axis not in ("x", "y", "z"):
        return None

    particle_spin = (nuclear_hamiltonian.dims[0][0] - 1) / 2
    I_alpha = qutip.jmat(particle_spin, spin_axis)
    dim = int(nuclear_hamiltonian.dims[0][0])

    exponent = 1j * t * nuclear_hamiltonian
    U_plus = exponent.expm()
    U_minus = (-exponent).expm()
    rho = qutip.qeye(dim) / dim

    return np.real_if_close((U_plus * I_alpha * U_minus * I_alpha * rho).tr())


def site_correlator(
    t: float,
    zeeman_per_tesla: float,
    quadrupole_coupling: float,
    spin: float,
    biaxiality: float,
    euler_angles: np.ndarray,
    V_ZZ: float,
    applied_field: float,
    spin_axis: str,
) -> float:
    """
    Spin correlator at a single lattice site.

    Args:
        t (float): Time in seconds.
        zeeman_per_tesla (float): Zeeman splitting per Tesla (Hz/T).
        quadrupole_coupling (float): Constant converting V_ZZ to frequency (Hz/V·m⁻²).
        spin (float): Nuclear spin quantum number.
        biaxiality (float): EFG biaxiality η at this site.
        euler_angles (array-like): (alpha, beta, gamma) to the PAF.
        V_ZZ (float): Principal EFG component at this site (V·m⁻²).
        applied_field (float): Applied magnetic field in Tesla.
        spin_axis (str): "x", "y", or "z".

    Returns:
        float: Correlator value at this site and time.
    """
    alpha, beta, gamma = euler_angles
    H = faraday_hamiltonian(
        zeeman_per_tesla * applied_field,
        quadrupole_coupling * V_ZZ,
        biaxiality,
        spin,
        alpha,
        beta,
        gamma,
    )
    return spin_correlator(t, H, spin_axis)


def _parallel_site_correlator(
    t: float,
    zeeman_per_tesla: float,
    quadrupole_coupling: float,
    spin: float,
    biaxiality: float,
    alpha: float,
    beta: float,
    gamma: float,
    V_ZZ: float,
    applied_field: float,
    spin_axis: str,
) -> float:
    """Single-site correlator with unpacked Euler angles, suitable for Pool.starmap."""
    H = faraday_hamiltonian(
        zeeman_per_tesla * applied_field,
        quadrupole_coupling * V_ZZ,
        biaxiality,
        spin,
        alpha,
        beta,
        gamma,
    )
    return spin_correlator(t, H, spin_axis)


def _build_starmap_args(
    t: float,
    applied_field: float,
    spin_axis: str,
    nuclear_species: str,
    region_bounds: list[int],
    data_dir: pathlib.Path | str,
    step_size: int,
) -> tuple[int, list[tuple]]:
    """Package per-site parameters into a list suitable for Pool.starmap."""
    eta, _, _, V_ZZ_arr, euler_angles = load_efg(
        data_dir, nuclear_species, region_bounds, step_size
    )
    # euler_angles is already (n_sites, 3) from load_efg

    species = species_dict[nuclear_species]
    spin = species["particle_spin"]
    zeeman_per_tesla = species["zeeman_frequency_per_tesla"]
    Q = species["quadrupole_moment"]
    h, e = const.h, const.e
    qcc = (3 * e * Q) / (2 * h * spin * (2 * spin - 1))

    n_sites = eta.size
    biaxiality_list = eta.flatten()
    V_ZZ_list = V_ZZ_arr.flatten()
    alpha_list = euler_angles[:, 0]
    beta_list = euler_angles[:, 1]
    gamma_list = euler_angles[:, 2]

    return n_sites, list(
        zip(
            np.full(n_sites, t),
            np.full(n_sites, zeeman_per_tesla),
            np.full(n_sites, qcc),
            np.full(n_sites, spin),
            biaxiality_list,
            alpha_list,
            beta_list,
            gamma_list,
            V_ZZ_list,
            np.full(n_sites, applied_field),
            [spin_axis] * n_sites,
        )
    )


def run_correlator_series(
    data_dir: pathlib.Path | str,
    timerange: np.ndarray,
    applied_field: float,
    nuclear_species: str,
    region_bounds: list[int],
    step_size: int = 100,
    chunksize: int = 25,
) -> np.ndarray:
    """
    Compute the spin correlator time series for one species in parallel.

    Averages the correlator over all lattice sites in the region for each
    time in timerange, along all three spin axes.

    Args:
        data_dir: Directory containing pre-computed EFG archives.
        timerange (ndarray): Times at which to evaluate the correlator (seconds).
        applied_field (float): Applied magnetic field in Tesla.
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        region_bounds (list): [left, right, top, bottom].
        step_size (int): Subsampling interval for lattice sites. Default 100.
        chunksize (int): Tasks per worker per batch. Default 25.

    Returns:
        ndarray: Shape (3, len(timerange)), axes ordered x, y, z.
    """
    if nuclear_species not in _VALID_SPECIES:
        raise ValueError(
            f"nuclear_species must be one of {sorted(_VALID_SPECIES)}, got {nuclear_species!r}"
        )
    results = np.zeros((3, len(timerange)))

    with multiprocessing.Pool() as pool:
        for c, axis in enumerate(["x", "y", "z"]):
            for t_idx, t in enumerate(timerange):
                n_sites, args = _build_starmap_args(
                    t,
                    applied_field,
                    axis,
                    nuclear_species,
                    region_bounds,
                    data_dir,
                    step_size,
                )
                site_values = pool.starmap(
                    _parallel_site_correlator, args, chunksize=chunksize
                )
                results[c, t_idx] = np.mean(site_values)

    return results


def run_log_correlator_simulation(
    data_dir: pathlib.Path | str,
    save_dir: pathlib.Path | str,
    min_time_exp: int,
    max_time_exp: int,
    n_times: int,
    applied_field: float,
    region_bounds: list[int] | None = None,
    step_size: int = 100,
    chunksize: int = 25,
) -> None:
    """
    Compute and save log-spaced correlator data for all four nuclear species.

    Args:
        data_dir: Directory containing pre-computed EFG archives.
        save_dir: Directory to write the output .npz archive.
        min_time_exp (int): Exponent of earliest time (10**min_time_exp seconds).
        max_time_exp (int): Exponent of latest time (10**max_time_exp seconds).
        n_times (int): Number of time points.
        applied_field (float): Applied magnetic field in Tesla.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        step_size (int): Subsampling interval. Default 100.
        chunksize (int): Pool chunksize. Default 25.
    """
    save_dir = pathlib.Path(save_dir)

    if region_bounds is None:
        region_bounds = SOKOLOV_DOT_REGION

    timerange = np.logspace(min_time_exp, max_time_exp, n_times)
    species_list = ["Ga69", "Ga71", "As75", "In115"]
    data = {}

    for species in species_list:
        data[species] = run_correlator_series(
            data_dir,
            timerange,
            applied_field,
            species,
            region_bounds,
            step_size,
            chunksize,
        )
        logger.info("%s done.", species)

    archive_name = (
        f"log_time_correlator_data_B{applied_field}T"
        f"_{n_times}pts_{10**min_time_exp:.0e}_{10**max_time_exp:.0e}s"
        f"_region{region_bounds}.npz"
    )
    np.savez(
        save_dir / archive_name,
        timerange=timerange,
        region_bounds=region_bounds,
        **{f"{s}_data": data[s] for s in species_list},
    )


def run_linear_correlator_simulation(
    data_dir: pathlib.Path | str,
    save_dir: pathlib.Path | str,
    min_time: float,
    max_time: float,
    timestep: float,
    applied_field: float,
    region_bounds: list[int] | None = None,
    step_size: int = 100,
    chunksize: int = 25,
) -> None:
    """
    Compute and save linearly-spaced correlator data for all four nuclear species.

    Args:
        data_dir: Directory containing pre-computed EFG archives.
        save_dir: Directory to write the output .npz archive.
        min_time (float): Start time in seconds.
        max_time (float): End time in seconds.
        timestep (float): Time step in seconds.
        applied_field (float): Applied magnetic field in Tesla.
        region_bounds (list): [left, right, top, bottom]. Defaults to dot region.
        step_size (int): Subsampling interval. Default 100.
        chunksize (int): Pool chunksize. Default 25.
    """
    save_dir = pathlib.Path(save_dir)

    if region_bounds is None:
        region_bounds = SOKOLOV_DOT_REGION

    timerange = np.arange(min_time, max_time, timestep)
    species_list = ["Ga69", "Ga71", "As75", "In115"]
    data = {}

    for species in species_list:
        data[species] = run_correlator_series(
            data_dir,
            timerange,
            applied_field,
            species,
            region_bounds,
            step_size,
            chunksize,
        )
        logger.info("%s done.", species)

    archive_name = (
        f"linear_time_correlator_data_B{applied_field}T"
        f"_{min_time}_{max_time}s_dt{timestep}"
        f"_region{region_bounds}.npz"
    )
    np.savez(
        save_dir / archive_name,
        timerange=timerange,
        region_bounds=region_bounds,
        **{f"{s}_data": data[s] for s in species_list},
    )
