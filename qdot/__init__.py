from qdot.isotopes import species_dict, old_species_dict, quadrupole_coupling
from qdot.efg import (
    calculate_efg,
    calculate_efg_vectorised,
    euler_angles_from_rot_mat,
    quadrupole_frequency,
    equivalent_b_field,
)
from qdot.hamiltonians import (
    spin_rotator,
    faraday_hamiltonian,
    voigt_hamiltonian,
    rf_hamiltonian,
    transition_rate,
    energy_levels_vs_field,
    energy_levels_vs_eta,
)
from qdot.correlators import (
    spin_correlator,
    site_correlator,
    run_correlator_series,
    run_log_correlator_simulation,
    run_linear_correlator_simulation,
    load_correlator_archive,
)
from qdot.io import (
    load_strain_data,
    load_mirrored_data,
    load_efg,
    save_efg,
    load_concentration_data,
    load_mirrored_concentration_data,
    create_mirrored_strain_data,
    create_mirrored_concentration_data,
    mirror_array,
    save_nmr_map,
    load_nmr_map,
    region_from_rectangle,
    SOKOLOV_DOT_REGION,
    SOKOLOV_REGIONS,
)
from qdot.sites import all_locations, random_locations, find_best_locations
from qdot.nmr import (
    absorption_spectrum,
    varied_field_spectra,
    absorption_map,
    summed_absorption_spectrum,
    experimental_nmr_simulation,
    combined_spectrum,
    lorentzian_pulse,
    pulse_capture_fraction,
)
from qdot.strain import run_strain_simulation, strain_tensor, strain_tensor_vectorised
from qdot.nff import (
    dephasing_polarisation_curve,
    non_dephased_polarisation,
    x_polarisation,
    y_polarisation,
    dephasing_from_scattering,
)
from qdot.machine_gun import (
    initial_density_matrix,
    perfect_machine_gun_density_matrix,
    noisy_machine_gun,
    dot_dephasing_superoperator,
    dot_damping_superoperator,
    fidelity_vs_error,
    trace_distance_vs_error,
    error_list_to_array,
)

__all__ = [
    # isotopes
    "species_dict",
    "old_species_dict",
    "quadrupole_coupling",
    # efg
    "calculate_efg",
    "calculate_efg_vectorised",
    "euler_angles_from_rot_mat",
    "quadrupole_frequency",
    "equivalent_b_field",
    # hamiltonians
    "spin_rotator",
    "faraday_hamiltonian",
    "voigt_hamiltonian",
    "rf_hamiltonian",
    "transition_rate",
    "energy_levels_vs_field",
    "energy_levels_vs_eta",
    # correlators
    "spin_correlator",
    "site_correlator",
    "run_correlator_series",
    "run_log_correlator_simulation",
    "run_linear_correlator_simulation",
    "load_correlator_archive",
    # io
    "load_strain_data",
    "load_mirrored_data",
    "load_efg",
    "save_efg",
    "load_concentration_data",
    "load_mirrored_concentration_data",
    "create_mirrored_strain_data",
    "create_mirrored_concentration_data",
    "mirror_array",
    "save_nmr_map",
    "load_nmr_map",
    "region_from_rectangle",
    "SOKOLOV_DOT_REGION",
    "SOKOLOV_REGIONS",
    # sites
    "all_locations",
    "random_locations",
    "find_best_locations",
    # nmr
    "absorption_spectrum",
    "varied_field_spectra",
    "absorption_map",
    "summed_absorption_spectrum",
    "experimental_nmr_simulation",
    "combined_spectrum",
    "lorentzian_pulse",
    "pulse_capture_fraction",
    # strain
    "run_strain_simulation",
    "strain_tensor",
    "strain_tensor_vectorised",
    # nff
    "dephasing_polarisation_curve",
    "non_dephased_polarisation",
    "x_polarisation",
    "y_polarisation",
    "dephasing_from_scattering",
    # machine gun
    "initial_density_matrix",
    "perfect_machine_gun_density_matrix",
    "noisy_machine_gun",
    "dot_dephasing_superoperator",
    "dot_damping_superoperator",
    "fidelity_vs_error",
    "trace_distance_vs_error",
    "error_list_to_array",
]
