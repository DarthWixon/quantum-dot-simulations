"""
Data loading and saving for quantum dot simulations.

All functions take an explicit data_dir (pathlib.Path or str) rather than
relying on a global path. Pass the directory containing your data files.

Sokolov strain data files expected in data_dir:
    full_epsilon_xx.txt, full_epsilon_xy.txt, full_epsilon_yy.txt

Concentration data file expected in data_dir:
    conc_data_to_scale_{method}_interpolation.npy

EFG archives are saved to / loaded from data_dir using a naming convention
derived from the species, region bounds, and step size.
"""

import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)

# Pixel bounds of the quantum dot region in the Sokolov dataset
# (Sokolov et al. DOI: 10.1103/PhysRevB.93.045301).
SOKOLOV_DOT_REGION = [100, 1200, 439, 880]


def load_strain_data(
    data_dir: pathlib.Path | str,
    region_bounds: list[int],
    step_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load strain tensor components from whitespace-delimited text files.

    Expects three files in data_dir: full_epsilon_xx.txt, full_epsilon_xy.txt,
    full_epsilon_yy.txt. Each is a 2D array of strain values. The xz and zz
    components are sign-flipped on load to match the simulation coordinate
    convention.

    Args:
        data_dir: Directory containing full_epsilon_*.txt files.
        region_bounds (list): [left, right, top, bottom] pixel bounds.
        step_size (int): Subsample interval. 1 = every site.

    Returns:
        dot_epsilon_xx, dot_epsilon_xz, dot_epsilon_zz (ndarray)
    """
    data_dir = pathlib.Path(data_dir)
    full_xx = np.loadtxt(data_dir / "full_epsilon_xx.txt")
    full_xy = np.loadtxt(data_dir / "full_epsilon_xy.txt")
    full_yy = np.loadtxt(data_dir / "full_epsilon_yy.txt")

    H_1, H_2, L_1, L_2 = region_bounds
    xx = full_xx[L_1:L_2:step_size, H_1:H_2:step_size]
    xz = -full_xy[L_1:L_2:step_size, H_1:H_2:step_size]
    zz = -full_yy[L_1:L_2:step_size, H_1:H_2:step_size]
    return xx, xz, zz


def load_mirrored_data(
    data_dir: pathlib.Path | str,
    region_bounds: list[int],
    step_size: int = 1,
    mirror_type: str = "left_right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a mirrored (synthetic) strain dataset.

    Args:
        data_dir: Directory containing the mirrored .npz archive.
        region_bounds (list): [left, right, top, bottom].
        step_size (int): Subsample interval.
        mirror_type (str): "left_right" or other mirror variant name.

    Returns:
        epsilon_xx, epsilon_xz, epsilon_zz (ndarray)
    """
    data_dir = pathlib.Path(data_dir)
    archive_path = (
        data_dir / f"{mirror_type}_mirrored_strain_data_in_region_{region_bounds}.npz"
    )
    archive = np.load(archive_path)
    return archive["full_xx_data"], archive["full_xy_data"], archive["full_yy_data"]


def load_concentration_data(
    data_dir: pathlib.Path | str,
    region_bounds: list[int],
    step_size: int = 1,
    method: str = "cubic",
) -> np.ndarray:
    """
    Load interpolated In115 concentration data from the Sokolov dataset.

    Data from Sokolov et al. DOI: 10.1103/PhysRevB.93.045301.

    Args:
        data_dir: Directory containing conc_data_to_scale_{method}_interpolation.npy.
        region_bounds (list): [left, right, top, bottom].
        step_size (int): Subsample interval.
        method (str): Interpolation method used when the file was created ("cubic",
            "linear", "nearest").

    Returns:
        dot_conc_data (ndarray): In115 concentration fraction at each site.
    """
    data_dir = pathlib.Path(data_dir)
    full_conc = np.load(data_dir / f"conc_data_to_scale_{method}_interpolation.npy")
    H_1, H_2, L_1, L_2 = region_bounds
    return full_conc[L_1:L_2:step_size, H_1:H_2:step_size]


def _efg_archive_path(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    region_bounds: list[int],
    step_size: int,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
) -> pathlib.Path:
    """Build the canonical archive filename for a pre-computed EFG dataset."""
    data_dir = pathlib.Path(data_dir)
    # Normalise so tuples and lists produce the same filename.
    region_bounds = list(region_bounds)
    base = f"{nuclear_species}_calculation_results_for_region{region_bounds}_with_step_size_{step_size}"
    if not real_strain:
        base += f"_using_{mirror_type}_flipped_data"
    if use_sundfors:
        base += "_with_old_params"
    return data_dir / f"{base}.npz"


def save_efg(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    region_bounds: list[int],
    step_size: int,
    eta: np.ndarray,
    V_XX: np.ndarray,
    V_YY: np.ndarray,
    V_ZZ: np.ndarray,
    euler_angles: np.ndarray,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
    overwrite: bool = False,
) -> None:
    """
    Save pre-computed EFG arrays to a .npz archive.

    If the archive already exists it is left untouched (a warning is logged),
    matching the behaviour of the original calculate_and_save_EFG. Pass
    overwrite=True to replace it.

    Args:
        data_dir: Directory to write the archive into.
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        region_bounds (list): [left, right, top, bottom].
        step_size (int): Subsample interval used when the EFG was calculated.
        eta, V_XX, V_YY, V_ZZ, euler_angles (ndarray): EFG arrays from calculate_efg.
        use_sundfors (bool): True if the Sundfors parameter set was used.
        real_strain (bool): False if mirrored strain data was used.
        mirror_type (str): Mirror variant, only used when real_strain=False.
        overwrite (bool): Replace an existing archive instead of skipping it.
    """
    path = _efg_archive_path(
        data_dir,
        nuclear_species,
        region_bounds,
        step_size,
        use_sundfors,
        real_strain,
        mirror_type,
    )
    if path.exists() and not overwrite:
        logger.warning("EFG archive already exists, not overwriting: %s", path)
        return
    np.savez(path, eta=eta, V_XX=V_XX, V_YY=V_YY, V_ZZ=V_ZZ, euler_angles=euler_angles)
    logger.info("Saved EFG archive: %s", path)


def load_efg(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    region_bounds: list[int],
    step_size: int = 1,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed EFG arrays from a .npz archive.

    Args:
        data_dir: Directory containing the archive.
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        region_bounds (list): [left, right, top, bottom].
        step_size (int): Subsample interval.
        use_sundfors (bool): True to load the Sundfors-parameter variant.
        real_strain (bool): False to load the mirrored-strain variant.
        mirror_type (str): Mirror variant name, only used when real_strain=False.

    Returns:
        eta, V_XX, V_YY, V_ZZ (ndarray): EFG components, shape (n, m).
        euler_angles (ndarray): Euler angles, shape (n*m, 3).
    """
    path = _efg_archive_path(
        data_dir,
        nuclear_species,
        region_bounds,
        step_size,
        use_sundfors,
        real_strain,
        mirror_type,
    )
    archive = np.load(path)
    eta = archive["eta"]
    n_sites = eta.size
    return (
        eta,
        archive["V_XX"],
        archive["V_YY"],
        archive["V_ZZ"],
        archive["euler_angles"].reshape(n_sites, 3),
    )
