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

# Named pixel regions of the Sokolov dataset used throughout the original
# research scripts. "entire_dot" is the same as SOKOLOV_DOT_REGION.
SOKOLOV_REGIONS = {
    "central_high_In": [450, 850, 550, 650],
    "central_low_In": [450, 850, 660, 760],
    "dot_LHS": [20, 420, 650, 750],
    "dot_RHS": [1010, 1410, 650, 750],
    "top_right_outside_dot": [1010, 1410, 450, 550],
    "below_outside_dot": [450, 850, 775, 875],
    "single_atom_region": [560, 575, 600, 625],
    "entire_dot": SOKOLOV_DOT_REGION,
    "entire_image": [100, 1200, 200, 1000],
}


def region_from_rectangle(
    base_region_bounds: list[int], rectangle: list[int]
) -> list[int]:
    """
    Convert a rectangle drawn on a plotted region into absolute region bounds.

    Rectangles use the [left, bottom, width, height] convention of
    plot_concentration_with_regions; imshow plots from the top-left, so
    "bottom = 0" is at the top of the image.

    Args:
        base_region_bounds (list): [left, right, top, bottom] of the plotted region.
        rectangle (list): [left, bottom, width, height] in region pixel coordinates.

    Returns:
        list: [left, right, top, bottom] in absolute dataset coordinates.
    """
    left = base_region_bounds[0] + rectangle[0]
    right = left + rectangle[2]
    top = base_region_bounds[2] + rectangle[1]
    bottom = top + rectangle[3]
    return [left, right, top, bottom]


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


def mirror_array(data_array: np.ndarray, direction: str = "left_right") -> np.ndarray:
    """
    Make a 2D array symmetric by reflecting one half about the centre column.

    Args:
        data_array (ndarray): 2D array to mirror.
        direction (str): "left_right" keeps the left half and reflects it onto
            the right; "right_left" keeps the right half.

    Returns:
        ndarray: Symmetric array. For odd column counts the centre column is
        dropped (the output has one column fewer than the input), matching the
        original create_mirrored_data behaviour.
    """
    half = data_array.shape[1] // 2
    if direction == "left_right":
        kept = data_array[:, :half]
        return np.hstack([kept, np.fliplr(kept)])
    if direction == "right_left":
        kept = data_array[:, data_array.shape[1] - half :]
        return np.hstack([np.fliplr(kept), kept])
    raise ValueError(
        f"direction must be 'left_right' or 'right_left', got {direction!r}"
    )


def create_mirrored_strain_data(
    data_dir: pathlib.Path | str,
    region_bounds: list[int],
    mirror_type: str = "left_right",
    overwrite: bool = False,
) -> None:
    """
    Create and save a mirrored (synthetic, symmetric) strain dataset.

    Loads the real strain data for the region (already sign-flipped to the
    simulation convention by load_strain_data), mirrors each component about
    the centre column, and saves the archive that load_mirrored_data reads.
    The archive keys keep the raw-file names (full_xx_data, full_xy_data,
    full_yy_data) for compatibility, but hold the sign-flipped xx/xz/zz values.

    Args:
        data_dir: Directory containing the strain text files; the archive is
            written here too.
        region_bounds (list): [left, right, top, bottom].
        mirror_type (str): "left_right" or "right_left".
        overwrite (bool): Replace an existing archive instead of skipping it.
    """
    data_dir = pathlib.Path(data_dir)
    region_bounds = list(region_bounds)
    path = (
        data_dir / f"{mirror_type}_mirrored_strain_data_in_region_{region_bounds}.npz"
    )
    if path.exists() and not overwrite:
        logger.warning(
            "Mirrored strain archive already exists, not overwriting: %s", path
        )
        return

    xx, xz, zz = load_strain_data(data_dir, region_bounds)
    np.savez(
        path,
        full_xx_data=mirror_array(xx, mirror_type),
        full_xy_data=mirror_array(xz, mirror_type),
        full_yy_data=mirror_array(zz, mirror_type),
    )
    logger.info("Saved mirrored strain archive: %s", path)


def create_mirrored_concentration_data(
    data_dir: pathlib.Path | str,
    region_bounds: list[int],
    mirror_type: str = "left_right",
    method: str = "cubic",
    overwrite: bool = False,
) -> None:
    """
    Create and save a mirrored In115 concentration dataset.

    Args:
        data_dir: Directory containing the concentration .npy file; the
            mirrored file is written here too.
        region_bounds (list): [left, right, top, bottom].
        mirror_type (str): "left_right" or "right_left".
        method (str): Interpolation method of the source file.
        overwrite (bool): Replace an existing file instead of skipping it.
    """
    data_dir = pathlib.Path(data_dir)
    region_bounds = list(region_bounds)
    path = (
        data_dir / f"{mirror_type}_mirrored_In_conc_data_in_region_{region_bounds}.npy"
    )
    if path.exists() and not overwrite:
        logger.warning(
            "Mirrored concentration file already exists, not overwriting: %s", path
        )
        return

    conc = load_concentration_data(data_dir, region_bounds, method=method)
    np.save(path, mirror_array(conc, mirror_type))
    logger.info("Saved mirrored concentration data: %s", path)


def load_mirrored_concentration_data(
    data_dir: pathlib.Path | str,
    region_bounds: list[int],
    mirror_type: str = "left_right",
) -> np.ndarray:
    """
    Load a mirrored In115 concentration dataset created by
    create_mirrored_concentration_data.

    Args:
        data_dir: Directory containing the mirrored .npy file.
        region_bounds (list): [left, right, top, bottom] used at creation time.
        mirror_type (str): "left_right" or "right_left".

    Returns:
        ndarray: Mirrored concentration fraction at each site.
    """
    data_dir = pathlib.Path(data_dir)
    region_bounds = list(region_bounds)
    return np.load(
        data_dir / f"{mirror_type}_mirrored_In_conc_data_in_region_{region_bounds}.npy"
    )


def _nmr_map_path(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    field_geometry: str,
    n_locations: int,
    region_bounds: list[int],
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
) -> pathlib.Path:
    """Build the canonical archive filename for a 2D NMR absorption map."""
    data_dir = pathlib.Path(data_dir)
    region_bounds = list(region_bounds)
    base = (
        f"nmr_map_{nuclear_species}_{field_geometry}_{n_locations}locs"
        f"_region{region_bounds}"
        f"_{len(applied_field_list)}fields_{applied_field_list[0]:g}-{applied_field_list[-1]:g}T"
        f"_{len(rf_freq_list)}freqs_{rf_freq_list[0] / 1e6:g}-{rf_freq_list[-1] / 1e6:g}MHz"
    )
    if not real_strain:
        base += f"_using_{mirror_type}_flipped_data"
    if use_sundfors:
        base += "_with_old_params"
    return data_dir / f"{base}.npz"


def save_nmr_map(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    field_geometry: str,
    region_bounds: list[int],
    absorption_data: np.ndarray,
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    locations: list[tuple[int, int]],
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
    overwrite: bool = False,
) -> None:
    """
    Save a pre-computed 2D NMR absorption map to a .npz archive.

    If the archive already exists it is left untouched (a warning is logged).
    Pass overwrite=True to replace it.

    Args:
        data_dir: Directory to write the archive into.
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        field_geometry (str): "Faraday" or "Voigt".
        region_bounds (list): [left, right, top, bottom].
        absorption_data (ndarray): Map from qdot.nmr.absorption_map,
            shape (n_fields, n_freqs).
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        locations (list of (int, int)): Lattice sites the map was summed over.
        use_sundfors (bool): True if the Sundfors parameter set was used.
        real_strain (bool): False if mirrored strain data was used.
        mirror_type (str): Mirror variant, only used when real_strain=False.
        overwrite (bool): Replace an existing archive instead of skipping it.
    """
    path = _nmr_map_path(
        data_dir,
        nuclear_species,
        field_geometry,
        len(locations),
        region_bounds,
        applied_field_list,
        rf_freq_list,
        use_sundfors,
        real_strain,
        mirror_type,
    )
    if path.exists() and not overwrite:
        logger.warning("NMR map archive already exists, not overwriting: %s", path)
        return
    np.savez(
        path,
        absorption_data=absorption_data,
        applied_field_list=np.asarray(applied_field_list),
        rf_freq_list=np.asarray(rf_freq_list),
        locations=np.asarray(locations),
    )
    logger.info("Saved NMR map archive: %s", path)


def load_nmr_map(
    data_dir: pathlib.Path | str,
    nuclear_species: str,
    field_geometry: str,
    n_locations: int,
    region_bounds: list[int],
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    use_sundfors: bool = False,
    real_strain: bool = True,
    mirror_type: str = "left_right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a pre-computed 2D NMR absorption map saved by save_nmr_map.

    The filename is reconstructed from the parameters, so they must match the
    values used when the map was saved.

    Returns:
        absorption_data (ndarray): Shape (n_fields, n_freqs).
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        locations (ndarray): Lattice sites, shape (n_locations, 2).
    """
    path = _nmr_map_path(
        data_dir,
        nuclear_species,
        field_geometry,
        n_locations,
        region_bounds,
        applied_field_list,
        rf_freq_list,
        use_sundfors,
        real_strain,
        mirror_type,
    )
    archive = np.load(path)
    return (
        archive["absorption_data"],
        archive["applied_field_list"],
        archive["rf_freq_list"],
        archive["locations"],
    )
