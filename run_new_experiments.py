"""
Run the newly ported experiments against the real Sokolov dataset.

Reads strain/concentration data from real_data/ and the pre-computed EFG
archives from outputs/full_pipeline/efg/, and writes every figure to
outputs/new_experiments/. EFG archives and the concentration file are
symlinked into outputs/new_experiments/work/ so the NMR functions see one
data directory.

Sizes are chosen to finish in a few minutes on a laptop; raise N_FIELDS /
N_FREQS / site counts for publication-quality versions.
"""

import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import qdot
from qdot import plot

REPO = pathlib.Path(__file__).parent
REAL_DATA = REPO / "real_data"
EFG_DIR = REPO / "outputs" / "full_pipeline" / "efg"
OUT = REPO / "outputs" / "new_experiments"
WORK = OUT / "work"

REGION = qdot.SOKOLOV_DOT_REGION  # [100, 1200, 439, 880] -> arrays (441, 1100)

# NMR map resolution
N_FIELDS = 80
N_FREQS = 250
FIELDS = np.linspace(0.001, 2.0, N_FIELDS)  # 0.001 avoids log(0), as in archive
FREQS = np.linspace(0.0, 75.0, N_FREQS) * 1e6
N_MAP_SITES = 30
N_EXPERIMENT_SITES = 80
N_COMBINED_SITES = 40


def setup_work_dir() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for src in list(EFG_DIR.glob("*.npz")) + [
        REAL_DATA / "conc_data_to_scale_cubic_interpolation.npy",
        REAL_DATA / "full_epsilon_xx.txt",
        REAL_DATA / "full_epsilon_xy.txt",
        REAL_DATA / "full_epsilon_yy.txt",
    ]:
        link = WORK / src.name
        if not link.exists():
            link.symlink_to(src.resolve())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strain_figures(xx, xz, zz):
    plot.plot_measured_strain(
        xx, xz, zz, layout="horizontal", save_path=OUT / "strain_maps_horizontal.png"
    )
    plot.plot_measured_strain(
        xx, xz, zz, layout="vertical", save_path=OUT / "strain_maps_vertical.png"
    )
    plot.plot_strain_row_cut(
        xx, xz, zz, row=xx.shape[0] // 2, save_path=OUT / "strain_row_cut.png"
    )
    log("strain figures done")


def quadrupolar_figures(efg_data):
    b_field_maps = []
    freq_sets = {}
    for species in ("Ga69", "Ga71", "As75", "In115"):
        eta, _, _, V_ZZ, euler = efg_data[species]
        freq = qdot.quadrupole_frequency(species, V_ZZ)
        freq_sets[species] = freq
        b_field_maps.append(qdot.equivalent_b_field(species, V_ZZ))

        plot.plot_quadrupole_frequency(
            freq,
            title=f"Quadrupole Frequency — {species}",
            save_path=OUT / f"quadrupole_frequency_{species}.png",
        )
        plot.plot_biaxiality(
            eta,
            title=f"Biaxiality — {species}",
            save_path=OUT / f"biaxiality_{species}.png",
        )

    plot.plot_all_equivalent_b_fields(
        b_field_maps,
        region_bounds=REGION,
        save_path=OUT / "equivalent_b_fields_all_species.png",
    )

    # EFG direction quivers for In115: eta background and strength background.
    eta, _, _, V_ZZ, euler = efg_data["In115"]
    n, m = eta.shape
    euler_grid = euler.reshape(n, m, 3)
    plot.plot_efg_directions(
        euler_grid,
        eta,
        background_label=r"$\eta$",
        background_vmin=0.0,
        background_vmax=1.0,
        spacing=25,
        title="EFG Direction over Biaxiality — In115",
        save_path=OUT / "efg_directions_biaxiality_In115.png",
    )
    plot.plot_efg_directions(
        euler_grid,
        np.abs(qdot.quadrupole_frequency("In115", V_ZZ)) / 1e6,
        background_label="|Quadrupole Frequency| (MHz)",
        arrow_lengths=np.abs(V_ZZ),
        spacing=25,
        title="EFG Direction over Interaction Strength — In115",
        save_path=OUT / "efg_directions_strength_In115.png",
    )

    plot.plot_frequency_histograms(
        freq_sets,
        title="Quadrupole Frequency Distributions",
        save_path=OUT / "frequency_histograms_overlay.png",
    )
    plot.plot_frequency_histograms_grid(
        freq_sets, fit="gamma", save_path=OUT / "frequency_histograms_gamma_grid.png"
    )
    log("quadrupolar figures done")


def energy_level_figures(efg_data):
    sweep = np.linspace(0.0, 3.0, 120)
    site = (220, 550)  # central dot site

    for species in ("Ga69", "Ga71", "As75", "In115"):
        eta, _, _, V_ZZ, euler = efg_data[species]
        n_cols = eta.shape[1]
        angles = euler[site[0] * n_cols + site[1]]
        faraday = qdot.energy_levels_vs_field(
            species, sweep, eta[site], V_ZZ[site], angles, "Faraday"
        )
        voigt = qdot.energy_levels_vs_field(
            species, sweep, eta[site], V_ZZ[site], angles, "Voigt"
        )
        plot.plot_energy_levels_comparison(
            sweep,
            faraday,
            voigt,
            save_path=OUT / f"energy_levels_comparison_{species}.png",
        )

    # Random-site ensemble for In115: inhomogeneous level broadening.
    eta, _, _, V_ZZ, euler = efg_data["In115"]
    n, m = eta.shape
    rng = np.random.default_rng(0)
    sites = qdot.random_locations(10, (n, m), rng=rng)
    far_stack, voi_stack = [], []
    for x, y in sites:
        angles = euler[x * m + y]
        far_stack.append(
            qdot.energy_levels_vs_field(
                "In115", sweep, eta[x, y], V_ZZ[x, y], angles, "Faraday"
            )
        )
        voi_stack.append(
            qdot.energy_levels_vs_field(
                "In115", sweep, eta[x, y], V_ZZ[x, y], angles, "Voigt"
            )
        )
    plot.plot_energy_levels_comparison(
        sweep,
        np.vstack(far_stack),
        np.vstack(voi_stack),
        save_path=OUT / "energy_levels_random_sites_In115.png",
    )

    # Quadrupolar-only eta sweep at the regional mean V_ZZ.
    etas = np.linspace(0.0, 1.0, 100)
    levels = qdot.energy_levels_vs_eta("In115", etas, V_ZZ=float(np.mean(V_ZZ)))
    plot.plot_energy_levels(
        etas,
        levels,
        x_label=r"Biaxiality $\eta$",
        title="In115 Levels vs Biaxiality (B = 0)",
        save_path=OUT / "energy_levels_vs_eta_In115.png",
    )
    log("energy level figures done")


def nmr_map_figures(map_sites):
    faraday = qdot.absorption_map(
        "Ga69", FIELDS, "Faraday", FREQS, map_sites, WORK, REGION
    )
    log("Ga69 Faraday map done")
    voigt = qdot.absorption_map("Ga69", FIELDS, "Voigt", FREQS, map_sites, WORK, REGION)
    log("Ga69 Voigt map done")
    qdot.save_nmr_map(
        WORK, "Ga69", "Faraday", REGION, faraday, FIELDS, FREQS, map_sites
    )
    qdot.save_nmr_map(WORK, "Ga69", "Voigt", REGION, voigt, FIELDS, FREQS, map_sites)

    plot.plot_nmr_map(
        faraday,
        FIELDS,
        FREQS,
        title=f"Ga69 NMR — Faraday, {len(map_sites)} sites",
        save_path=OUT / "nmr_map_Ga69_Faraday.png",
    )
    plot.plot_nmr_map_pair(
        faraday,
        voigt,
        ("Faraday", "Voigt"),
        FIELDS,
        FREQS,
        save_path=OUT / "nmr_map_pair_Ga69_geometries.png",
    )
    plot.plot_nmr_map_difference(
        faraday,
        voigt,
        FIELDS,
        FREQS,
        title="Ga69: log(Faraday) − log(Voigt)",
        save_path=OUT / "nmr_map_difference_Ga69_geometries.png",
    )

    in115 = qdot.absorption_map(
        "In115", FIELDS, "Faraday", FREQS, map_sites, WORK, REGION
    )
    plot.plot_nmr_map(
        in115,
        FIELDS,
        FREQS,
        title=f"In115 NMR — Faraday, {len(map_sites)} sites",
        save_path=OUT / "nmr_map_In115_Faraday.png",
    )
    log("NMR map figures done")


def experimental_simulation_figures():
    maps = qdot.experimental_nmr_simulation(
        "Faraday", FIELDS, FREQS, N_EXPERIMENT_SITES, WORK, REGION, rng=1
    )
    total = sum(maps.values())
    plot.plot_nmr_map(
        total,
        FIELDS,
        FREQS,
        title=f"Experimental Estimate — Faraday, {N_EXPERIMENT_SITES} nuclei",
        save_path=OUT / "experimental_nmr_summed.png",
    )
    plot.plot_nmr_map_layered(
        maps,
        FIELDS,
        FREQS,
        title="Experimental Estimate — per-species layers",
        save_path=OUT / "experimental_nmr_layered.png",
    )
    log("experimental simulation figures done")


def pulse_analysis_figures():
    freqs = np.linspace(0.0, 100.0, 2000) * 1e6
    sites = qdot.random_locations(
        N_COMBINED_SITES, (441, 1100), rng=2
    )  # EFG array shape for the dot region
    spectrum = qdot.combined_spectrum(3.0, "Faraday", freqs, sites, WORK, REGION)

    peak_idx, _ = scipy.signal.find_peaks(spectrum, prominence=0.01)
    main_peak = freqs[peak_idx[np.argmax(spectrum[peak_idx])]]

    # Combined spectrum with detected peaks marked.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs / 1e6, spectrum)
    ax.plot(freqs[peak_idx] / 1e6, spectrum[peak_idx], "x", color="red")
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Absorption (arb. units)")
    ax.set_title(
        f"Combined Species Spectrum — 3 T, {N_COMBINED_SITES} nuclei per species"
    )
    fig.tight_layout()
    fig.savefig(OUT / "combined_spectrum_peaks.png")
    plt.close(fig)

    # Pulse overlaid on the spectrum at the main peak.
    width = 5e6
    pulse = qdot.lorentzian_pulse(freqs, main_peak, width)
    pulse = pulse / np.max(pulse)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs / 1e6, pulse, linestyle="dotted", label="Pulse")
    ax.plot(freqs / 1e6, spectrum * pulse, label="Pulse × Spectrum")
    ax.plot(freqs / 1e6, spectrum, alpha=0.3, label="Spectrum")
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_title(f"Lorentzian Pulse at {main_peak / 1e6:.1f} ± {width / 2e6:.1f} MHz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pulse_overlaid_spectrum.png")
    plt.close(fig)

    # Capture fraction vs pulse width: in the target window around the main
    # peak vs the whole spectrum.
    widths = np.linspace(0.5, 20.0, 40) * 1e6
    window = (main_peak - 2.5e6, main_peak + 2.5e6)
    in_band = [
        qdot.pulse_capture_fraction(spectrum, freqs, main_peak, w, window=window)
        for w in widths
    ]
    overall = [
        qdot.pulse_capture_fraction(spectrum, freqs, main_peak, w) for w in widths
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(widths / 1e6, in_band, label="Target window (peak ± 2.5 MHz)")
    ax.plot(widths / 1e6, overall, label="Whole spectrum")
    ax.set_xlabel("Pulse Width (MHz)")
    ax.set_ylabel("Captured Fraction")
    ax.set_title(f"Pulse Selectivity at {main_peak / 1e6:.1f} MHz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pulse_capture_fraction_vs_width.png")
    plt.close(fig)
    log("pulse analysis figures done")


def mirrored_data_figures():
    qdot.create_mirrored_strain_data(WORK, REGION, "left_right")
    qdot.create_mirrored_concentration_data(WORK, REGION, "left_right")
    xx_m, xz_m, zz_m = qdot.load_mirrored_data(WORK, REGION, mirror_type="left_right")
    plot.plot_measured_strain(
        xx_m,
        xz_m,
        zz_m,
        layout="vertical",
        save_path=OUT / "mirrored_strain_maps_left_right.png",
    )
    conc_m = qdot.load_mirrored_concentration_data(WORK, REGION, "left_right")
    plot.plot_concentration(
        conc_m, save_path=OUT / "mirrored_concentration_left_right.png"
    )
    log("mirrored data figures done")


def machine_gun_and_nff_figures():
    strengths = np.linspace(0.0, 1.0, 25)
    dephasing_curves = {
        f"{n} photon{'s' if n > 1 else ''}": qdot.fidelity_vs_error(
            n, strengths, "dephasing"
        )
        for n in (1, 2, 3)
    }
    plot.plot_fidelity_curves(
        strengths,
        dephasing_curves,
        x_label="Dephasing Strength",
        title="Machine Gun Fidelity under Dot Dephasing",
        save_path=OUT / "machine_gun_fidelity_dephasing.png",
    )
    damping_curves = {
        f"{n} photon{'s' if n > 1 else ''}": qdot.fidelity_vs_error(
            n, strengths, "damping"
        )
        for n in (1, 2, 3)
    }
    plot.plot_fidelity_curves(
        strengths,
        damping_curves,
        x_label="Damping Strength",
        title="Machine Gun Fidelity under Amplitude Damping",
        save_path=OUT / "machine_gun_fidelity_damping.png",
    )

    dephasing_list, z_pol = qdot.dephasing_polarisation_curve(0.9, 0.3)
    reference = qdot.non_dephased_polarisation(0.9, 0.3)
    plot.plot_polarisation_curve(
        dephasing_list,
        z_pol,
        reference=reference,
        title="NFF Polarisation, q0 = 0.9, phase = 0.3",
        save_path=OUT / "nff_polarisation_curve.png",
    )
    log("machine gun and NFF figures done")


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    setup_work_dir()

    log("loading strain data")
    xx, xz, zz = qdot.load_strain_data(REAL_DATA, REGION)
    strain_figures(xx, xz, zz)

    log("loading EFG archives")
    efg_data = {
        s: qdot.load_efg(WORK, s, REGION) for s in ("Ga69", "Ga71", "As75", "In115")
    }
    quadrupolar_figures(efg_data)
    energy_level_figures(efg_data)

    log("selecting NMR sites via find_best_locations")
    map_sites = qdot.find_best_locations(
        N_MAP_SITES, xx, xz, zz, search_range=25, rng=0
    )
    nmr_map_figures(map_sites)
    experimental_simulation_figures()
    pulse_analysis_figures()
    mirrored_data_figures()
    machine_gun_and_nff_figures()

    n_figures = len(list(OUT.glob("*.png")))
    log(f"ALL DONE: {n_figures} figures in {OUT} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
