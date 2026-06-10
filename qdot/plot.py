"""
Visualisation functions for quantum dot simulation outputs.

All functions return the matplotlib Figure. The figure is detached from the
pyplot registry before returning, so use the object-oriented API
(fig.savefig, fig.axes, ...) for further work with it.
Pass save_path (str or Path) to save instead of displaying.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit
from scipy.stats import maxwell, gamma as gamma_dist

from qdot.isotopes import species_dict

# ---------------------------------------------------------------------------
# Equivalent B-field maps
# ---------------------------------------------------------------------------


def plot_equivalent_b_field(
    nuclear_species: str,
    b_field_array: np.ndarray,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Heatmap of the equivalent magnetic field for one nuclear species.

    Args:
        nuclear_species (str): One of "Ga69", "Ga71", "As75", "In115".
        b_field_array (ndarray): Equivalent B-field values, shape (n, m).
        save_path: File path to save the figure. If None, display instead.

    Returns:
        Figure
    """
    species = species_dict[nuclear_species]
    fig, ax = plt.subplots()
    im = ax.imshow(b_field_array)
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Equivalent B Field — {species['name']}")
    _save_or_show(fig, save_path)
    return fig


def plot_all_equivalent_b_fields(
    b_field_arrays: list[np.ndarray],
    region_bounds: list[int] | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    2×2 grid of equivalent B-field heatmaps, one per nuclear species.

    Args:
        b_field_arrays (list of ndarray): Four arrays in order Ga69, Ga71, As75, In115.
        region_bounds (list): Used in the figure title if provided.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    species_order = ["Ga69", "Ga71", "As75", "In115"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, species_key, b_field in zip(axes.flatten(), species_order, b_field_arrays):
        species = species_dict[species_key]
        im = ax.imshow(b_field)
        ax.set_title(f"Equivalent B — {species['name']}", fontsize=14)
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(im, cax=cax).ax.tick_params(labelsize=12)

    if region_bounds:
        fig.suptitle(f"Region: {region_bounds}")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Strain toy model
# ---------------------------------------------------------------------------


def plot_strain_lattice(
    simulated_species: np.ndarray,
    unstrained_positions: np.ndarray,
    strained_positions: np.ndarray,
    real_atoms: bool = True,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Overlay plot of unstrained and strained lattice positions and bonds.

    Species colours: Ga = red, As = blue, In = green.

    Args:
        simulated_species (ndarray): Integer species array, shape (n_rows, n_cols).
        unstrained_positions (ndarray): Shape (n_rows, n_cols, 2).
        strained_positions (ndarray): Shape (n_rows, n_cols, 2).
        real_atoms (bool): Include species legend if True.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    colour_map = {0: "r", 1: "b", 2: "g"}
    colours = np.array([colour_map[s] for s in simulated_species.flatten()])

    n_rows, n_cols = simulated_species.shape
    # Box dimensions must match qdot.strain: width follows columns, height rows.
    box_w = n_cols + 2
    box_h = n_rows + 2

    fig, ax = plt.subplots()

    for positions, alpha in [(unstrained_positions, 0.3), (strained_positions, 0.8)]:
        for r in range(n_rows):
            for c in range(n_cols):
                x, y = positions[r, c]
                if c + 1 < n_cols:
                    x_r, y_r = positions[r, c + 1]
                else:
                    x_r = box_w
                    y_r = (r + 1) * box_h / (n_rows + 1)
                if r + 1 < n_rows:
                    x_u, y_u = positions[r + 1, c]
                else:
                    x_u = (c + 1) * box_w / (n_cols + 1)
                    y_u = box_h
                ax.plot([x, x_r], [y, y_r], "k--", alpha=alpha)
                ax.plot([x, x_u], [y, y_u], "k--", alpha=alpha)

    ax.scatter(
        unstrained_positions[:, :, 0],
        unstrained_positions[:, :, 1],
        c=colours,
        alpha=0.5,
    )
    ax.scatter(
        strained_positions[:, :, 0], strained_positions[:, :, 1], c=colours, alpha=1.0
    )

    legend = [
        Line2D([0], [0], color="k", linestyle="--", alpha=0.3, label="Unstrained"),
        Line2D([0], [0], color="k", linestyle="--", alpha=0.8, label="Strained"),
    ]
    if real_atoms:
        legend += [
            Line2D([0], [0], color="w", markerfacecolor="r", marker="o", label="Ga"),
            Line2D([0], [0], color="w", markerfacecolor="b", marker="o", label="As"),
            Line2D([0], [0], color="w", markerfacecolor="g", marker="o", label="In"),
        ]

    ax.legend(handles=legend)
    ax.set_xlim(0, box_w)
    ax.set_ylim(0, box_h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_strain_tensors(
    strain_tensor_array: np.ndarray,
    absolute_values: bool = False,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Three-panel plot of ε_xx, ε_xy, and ε_yy strain components.

    Args:
        strain_tensor_array (ndarray): Shape (n, m, 2, 2).
        absolute_values (bool): Plot |ε| if True.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    data = np.absolute(strain_tensor_array) if absolute_values else strain_tensor_array
    vmin, vmax = data.min(), data.max()
    normalizer = Normalize(vmin, vmax)

    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    components = [
        (0, 0, r"$\epsilon_{xx}$"),
        (0, 1, r"$\epsilon_{xy}$"),
        (1, 1, r"$\epsilon_{yy}$"),
    ]

    for ax, (i, j, label) in zip(axs, components):
        ax.imshow(data[:, :, i, j], origin="lower", vmin=vmin, vmax=vmax, cmap=cm.GnBu)
        ax.set_title(label)
        ax.axis("off")

    plt.colorbar(
        plt.cm.ScalarMappable(norm=normalizer, cmap=cm.GnBu),
        ax=axs.ravel().tolist(),
        orientation="horizontal",
        shrink=0.95,
    )
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Concentration maps
# ---------------------------------------------------------------------------


def plot_concentration(
    conc_data: np.ndarray,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Heatmap of In115 concentration across the dot.

    Args:
        conc_data (ndarray): Concentration fraction at each site.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    im = ax.imshow(conc_data, cmap=cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())
    plt.colorbar(
        im, ax=ax, orientation="horizontal", label="Indium Concentration", shrink=0.5
    )
    ax.axis("off")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_concentration_with_regions(
    conc_data: np.ndarray,
    rect_specs: list[tuple[float, float, float, float]],
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Concentration heatmap with highlighted rectangular regions overlaid.

    Args:
        conc_data (ndarray): Concentration fraction at each site.
        rect_specs (list): List of [left, bottom, width, height] rectangles.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(conc_data, cmap=cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())
    plt.colorbar(im, ax=ax, orientation="horizontal")
    ax.set_title("Indium Concentration")

    for left, bottom, width, height in rect_specs:
        rect = plt.Rectangle(
            (left, bottom), width, height, edgecolor="black", linewidth=1, fill=False
        )
        ax.add_patch(rect)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Correlator graphs
# ---------------------------------------------------------------------------


def plot_correlator(
    timerange: np.ndarray,
    correlator_data: np.ndarray,
    nuclear_species: str,
    applied_field: float,
    log_time: bool = False,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Plot spin correlator time series for all three axes.

    Args:
        timerange (ndarray): Time values in seconds.
        correlator_data (ndarray): Shape (3, n_times), axes x/y/z.
        nuclear_species (str): Used in title.
        applied_field (float): Used in title.
        log_time (bool): Use semilogx if True.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    plot_fn = ax.semilogx if log_time else ax.plot

    for i, axis in enumerate(["x", "y", "z"]):
        plot_fn(timerange, correlator_data[i], label=f"{axis} axis")

    ax.set_xlabel("τ (s)")
    ax.set_ylabel("Correlator")
    ax.set_title(f"Correlator — {nuclear_species}, B = {applied_field} T")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_fourier_transform(
    timerange: np.ndarray,
    correlator_data: np.ndarray,
    timestep: float,
    nuclear_species: str,
    applied_field: float,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Plot the Fourier transform of a linearly-spaced correlator time series.

    Args:
        timerange (ndarray): Time values in seconds (must be linearly spaced).
        correlator_data (ndarray): Shape (3, n_times), axes x/y/z.
        timestep (float): Uniform time spacing in seconds.
        nuclear_species (str): Used in title.
        applied_field (float): Used in title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    n_times = len(timerange)
    freq = np.fft.rfftfreq(n_times, timestep)

    fig, ax = plt.subplots()
    for i, axis in enumerate(["x", "y", "z"]):
        fft_data = np.fft.rfft(correlator_data[i])
        ax.plot(freq, np.abs(fft_data), label=f"{axis} axis")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"Fourier Transform — {nuclear_species}, B = {applied_field} T")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, save_path: pathlib.Path | str | None) -> None:
    if save_path is not None:
        fig.savefig(pathlib.Path(save_path))
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# NMR field-frequency maps
# ---------------------------------------------------------------------------


def _nmr_extent(applied_field_list, rf_freq_list) -> list[float]:
    """Image extent [f_min(MHz), f_max(MHz), B_min(T), B_max(T)] for NMR maps."""
    return [
        rf_freq_list[0] / 1e6,
        rf_freq_list[-1] / 1e6,
        applied_field_list[0],
        applied_field_list[-1],
    ]


def plot_nmr_map(
    absorption_data: np.ndarray,
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    log_scale: bool = True,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Heatmap of a 2D NMR absorption map (applied field × RF frequency).

    Args:
        absorption_data (ndarray): Map from qdot.nmr.absorption_map,
            shape (n_fields, n_freqs).
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        log_scale (bool): Plot log(absorption). Default True.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    data = np.log(absorption_data) if log_scale else absorption_data

    fig, ax = plt.subplots()
    ax.imshow(
        data,
        origin="lower",
        cmap=cm.GnBu,
        aspect="auto",
        extent=_nmr_extent(applied_field_list, rf_freq_list),
        interpolation="none",
    )
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Applied B Field (T)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_nmr_map_pair(
    absorption_data_a: np.ndarray,
    absorption_data_b: np.ndarray,
    labels: tuple[str, str],
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    log_scale: bool = True,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Side-by-side comparison of two NMR maps (e.g. Faraday vs Voigt, or
    Checkhovich vs Sundfors parameter sets).

    Args:
        absorption_data_a, absorption_data_b (ndarray): Maps, shape (n_fields, n_freqs).
        labels (tuple): Panel titles, e.g. ("Faraday", "Voigt").
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        log_scale (bool): Plot log(absorption). Default True.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    extent = _nmr_extent(applied_field_list, rf_freq_list)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, label, letter in zip(
        axes, (absorption_data_a, absorption_data_b), labels, "ab"
    ):
        plot_data = np.log(data) if log_scale else data
        ax.imshow(
            plot_data,
            origin="lower",
            cmap=cm.GnBu,
            aspect="auto",
            extent=extent,
            interpolation="none",
        )
        ax.set_xlabel("RF Frequency (MHz)")
        ax.set_title(label)
        ax.text(0.02, 0.95, letter, transform=ax.transAxes, fontsize=14)

    axes[0].set_ylabel("Applied B Field (T)")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_nmr_map_difference(
    absorption_data_a: np.ndarray,
    absorption_data_b: np.ndarray,
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    log_scale: bool = True,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Difference map of two NMR maps on a symmetric diverging colour scale.

    Plots (a − b), or log(a) − log(b) when log_scale is True.

    Args:
        absorption_data_a, absorption_data_b (ndarray): Maps, shape (n_fields, n_freqs).
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        log_scale (bool): Difference of logs. Default True.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    if log_scale:
        difference = np.log(absorption_data_a) - np.log(absorption_data_b)
    else:
        difference = absorption_data_a - absorption_data_b
    scale = np.max(np.abs(difference))

    fig, ax = plt.subplots()
    im = ax.imshow(
        difference,
        origin="lower",
        cmap=cm.seismic,
        vmin=-scale,
        vmax=scale,
        aspect="auto",
        extent=_nmr_extent(applied_field_list, rf_freq_list),
        interpolation="none",
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Applied B Field (T)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_nmr_map_layered(
    species_maps: dict[str, np.ndarray],
    applied_field_list: np.ndarray,
    rf_freq_list: np.ndarray,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Per-species transparent layers of an experimental NMR simulation.

    Each species' log map is drawn as its own colour layer with decreasing
    alpha (Greys, Blues, Reds, Greens in dict order), matching the original
    experimental-simulation figure.

    Args:
        species_maps (dict): Species name → absorption map, e.g. from
            qdot.nmr.experimental_nmr_simulation.
        applied_field_list (ndarray): Applied fields in Tesla.
        rf_freq_list (ndarray): RF frequencies in Hz.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    cmaps = [cm.Greys, cm.Blues, cm.Reds, cm.Greens]
    alphas = [1.0, 0.5, 0.25, 0.25]
    extent = _nmr_extent(applied_field_list, rf_freq_list)

    fig, ax = plt.subplots(figsize=(12, 12))
    for (species, data), cmap, alpha in zip(species_maps.items(), cmaps, alphas):
        ax.imshow(
            np.log(data),
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            aspect="auto",
            extent=extent,
            interpolation="none",
        )
    ax.set_xlabel("RF Frequency (MHz)")
    ax.set_ylabel("Applied B Field (T)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Energy level diagrams
# ---------------------------------------------------------------------------


def plot_energy_levels(
    sweep_values: np.ndarray,
    level_data: np.ndarray,
    x_label: str,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Energy level (anti-crossing) diagram over a swept parameter.

    Args:
        sweep_values (ndarray): Swept variable (e.g. B in Tesla, or η).
        level_data (ndarray): Eigenenergies in Hz, shape (n_levels, n_sweep),
            e.g. from qdot.hamiltonians.energy_levels_vs_field.
        x_label (str): Label of the swept variable.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    for level in level_data:
        ax.plot(sweep_values, level / 1e6, color="black", linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy (MHz)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_energy_levels_comparison(
    sweep_values: np.ndarray,
    faraday_levels: np.ndarray,
    voigt_levels: np.ndarray,
    x_label: str = "Applied B Field (T)",
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Two-panel Faraday | Voigt energy level diagram with shared y-limits.

    Args:
        sweep_values (ndarray): Swept variable.
        faraday_levels, voigt_levels (ndarray): Eigenenergies in Hz,
            shape (n_levels, n_sweep). Overlaying several sites is possible by
            stacking their level arrays along axis 0.
        x_label (str): Label of the swept variable.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, levels, geometry, letter in zip(
        axes, (faraday_levels, voigt_levels), ("Faraday", "Voigt"), "ab"
    ):
        for level in levels:
            ax.plot(sweep_values, level / 1e6, color="black", linewidth=0.5)
        ax.set_xlabel(x_label)
        ax.set_title(f"{geometry} Orientation")
        ax.text(0.02, 0.95, letter, transform=ax.transAxes, fontsize=14)

    axes[0].set_ylabel("Energy (MHz)")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Quadrupolar site maps and EFG direction quivers
# ---------------------------------------------------------------------------


def plot_site_map(
    site_data: np.ndarray,
    colorbar_label: str = "",
    cmap=cm.GnBu,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Generic per-site heatmap with a horizontal colorbar (axes hidden).

    Args:
        site_data (ndarray): Per-site values, shape (n, m).
        colorbar_label (str): Label under the colorbar.
        cmap: Matplotlib colormap. Default GnBu.
        vmin, vmax (float): Optional fixed colour scale.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    im = ax.imshow(site_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
    if colorbar_label:
        cbar.ax.set_xlabel(colorbar_label, fontsize=14)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_biaxiality(
    eta: np.ndarray,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """Heatmap of biaxiality η on the standard fixed 0-1 scale."""
    return plot_site_map(
        eta,
        colorbar_label=r"$\eta$",
        vmin=0.0,
        vmax=1.0,
        title=title,
        save_path=save_path,
    )


def plot_quadrupole_frequency(
    frequency: np.ndarray,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Heatmap of the quadrupolar frequency across the dot.

    Args:
        frequency (ndarray): Quadrupolar frequency in Hz, e.g. from
            qdot.efg.quadrupole_frequency.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    return plot_site_map(
        frequency / 1e6,
        colorbar_label="Quadrupole Frequency (MHz)",
        title=title,
        save_path=save_path,
    )


def plot_efg_directions(
    euler_angles: np.ndarray,
    background: np.ndarray,
    background_label: str = "",
    arrow_lengths: np.ndarray | None = None,
    spacing: int = 20,
    double_headed: bool = True,
    background_vmin: float | None = None,
    background_vmax: float | None = None,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Quiver map of the EFG principal-axis direction over a background heatmap.

    Arrow angles come from the Euler γ angle at each site; arrows are drawn
    every `spacing` sites. Pass V_ZZ as arrow_lengths to scale arrows by the
    interaction size, or leave None for unit arrows. Backgrounds used in the
    original figures: biaxiality η, In concentration, quadrupole frequency.

    Args:
        euler_angles (ndarray): Euler angles per site, shape (n, m, 3).
            (Reshape the (n·m, 3) array from load_efg before passing.)
        background (ndarray): Background values, shape (n, m).
        background_label (str): Colorbar label.
        arrow_lengths (ndarray): Optional arrow length per site, shape (n, m).
        spacing (int): Arrow subsampling interval in sites.
        double_headed (bool): Draw arrowheads at both ends (the EFG axis has
            no sign). Default True.
        background_vmin, background_vmax (float): Optional fixed colour scale.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    n, m = background.shape
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    angles_deg = euler_angles[:, :, 2] * 180 / np.pi
    lengths = arrow_lengths if arrow_lengths is not None else np.ones((n, m))

    fig, ax = plt.subplots(figsize=(12, 8))
    s = spacing
    quiver_kwargs = dict(minshaft=5, pivot="middle", color="black")
    ax.quiver(
        X[::s, ::s],
        Y[::s, ::s],
        lengths[::s, ::s],
        lengths[::s, ::s],
        angles=angles_deg[::s, ::s],
        **quiver_kwargs,
    )
    if double_headed:
        ax.quiver(
            X[::s, ::s],
            Y[::s, ::s],
            lengths[::s, ::s],
            lengths[::s, ::s],
            angles=angles_deg[::s, ::s] + 180,
            **quiver_kwargs,
        )

    im = ax.imshow(background, cmap=cm.GnBu, vmin=background_vmin, vmax=background_vmax)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
    if background_label:
        cbar.ax.set_xlabel(background_label, fontsize=14)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Quadrupolar frequency histograms
# ---------------------------------------------------------------------------


def _gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _fit_histogram(values_mhz, bin_centres, hist_vals, fit):
    """Fit a named distribution to histogram data; return (curve, annotation)."""
    if fit == "gauss":
        coeffs, _ = curve_fit(_gauss, bin_centres, hist_vals, p0=[1.0, 0.0, 0.2])
        fitted = _gauss(bin_centres, *coeffs)
        ss_res = np.sum((hist_vals - fitted) ** 2)
        ss_tot = np.sum((hist_vals - np.mean(hist_vals)) ** 2)
        r2 = 1 - ss_res / ss_tot
        text = (
            rf"$\mu$ = {coeffs[1]:.2f}, $\sigma$ = {abs(coeffs[2]):.2f}, "
            rf"$R^2$ = {r2:.2f}"
        )
    elif fit == "maxwell":
        params = maxwell.fit(values_mhz)
        fitted = maxwell.pdf(bin_centres, *params)
        text = f"Maxwell: loc = {params[0]:.2f}, scale = {params[1]:.2f}"
    elif fit == "gamma":
        params = gamma_dist.fit(values_mhz, floc=0)
        fitted = gamma_dist.pdf(bin_centres, *params)
        text = f"Gamma: a = {params[0]:.2f}, scale = {params[2]:.2f}"
    else:
        raise ValueError(
            f"fit must be 'gauss', 'maxwell', 'gamma' or None, got {fit!r}"
        )
    return fitted, text


def plot_frequency_histogram(
    frequencies: np.ndarray,
    fit: str | None = None,
    label: str | None = None,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Histogram of per-site quadrupolar frequencies with an optional fitted pdf.

    The Maxwell and gamma fits follow the original analysis and act on the
    magnitude of the frequencies; the Gaussian fit uses the signed values.
    Which distributions are physically meaningful is a physics question
    recorded in human-todo.

    Args:
        frequencies (ndarray): Per-site frequencies in Hz, e.g.
            qdot.efg.quadrupole_frequency of a V_ZZ array.
        fit (str): "gauss", "maxwell", "gamma", or None for no fit.
        label (str): Optional legend label for the histogram.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    values_mhz = np.asarray(frequencies).flatten() / 1e6
    if fit in ("maxwell", "gamma"):
        values_mhz = np.abs(values_mhz)

    fig, ax = plt.subplots()
    hist_vals, bin_edges, _ = ax.hist(
        values_mhz, bins="auto", density=True, histtype="step", label=label
    )
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    if fit is not None:
        fitted, text = _fit_histogram(values_mhz, bin_centres, hist_vals, fit)
        ax.plot(bin_centres, fitted, linestyle="dashed")
        ax.text(
            0.95,
            0.95,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel("Quadrupole Frequency (MHz)")
    ax.set_ylabel("Probability Density")
    if label:
        ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_frequency_histograms(
    frequency_sets: dict[str, np.ndarray],
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Overlaid step histograms of several frequency sets (species or regions).

    Args:
        frequency_sets (dict): Legend label → per-site frequencies in Hz.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    for label, frequencies in frequency_sets.items():
        ax.hist(
            np.asarray(frequencies).flatten() / 1e6,
            bins="auto",
            density=True,
            histtype="step",
            label=label,
        )
    ax.set_xlabel("Quadrupole Frequency (MHz)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_frequency_histograms_grid(
    frequency_sets: dict[str, np.ndarray],
    fit: str | None = "gamma",
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    2×2 grid of fitted frequency histograms, one panel per species.

    Args:
        frequency_sets (dict): Panel title → per-site frequencies in Hz
            (up to four entries).
        fit (str): "gauss", "maxwell", "gamma", or None. Default "gamma".
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (label, frequencies) in zip(axes.flatten(), frequency_sets.items()):
        values_mhz = np.asarray(frequencies).flatten() / 1e6
        if fit in ("maxwell", "gamma"):
            values_mhz = np.abs(values_mhz)
        hist_vals, bin_edges, _ = ax.hist(
            values_mhz, bins="auto", density=True, histtype="step"
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        if fit is not None:
            fitted, text = _fit_histogram(values_mhz, bin_centres, hist_vals, fit)
            ax.plot(bin_centres, fitted, linestyle="dashed")
            ax.text(
                0.95,
                0.95,
                text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
        ax.set_title(label)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Measured (Sokolov) strain maps
# ---------------------------------------------------------------------------


def plot_measured_strain(
    xx_array: np.ndarray,
    xz_array: np.ndarray,
    zz_array: np.ndarray,
    layout: str = "horizontal",
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Three-panel map of measured strain components (the Sokolov paper figure).

    "horizontal" reproduces the paper-recreation layout: panels ordered
    ε_xx, ε_zz, ε_xz on a fixed ±0.02 scale. "vertical" stacks the panels
    with a shared data-driven RdBu scale.

    Args:
        xx_array, xz_array, zz_array (ndarray): Strain components from
            qdot.io.load_strain_data, shape (n, m).
        layout (str): "horizontal" or "vertical".
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    panels = [
        (xx_array, r"$\epsilon_{xx}$"),
        (zz_array, r"$\epsilon_{zz}$"),
        (xz_array, r"$\epsilon_{xz}$"),
    ]

    if layout == "horizontal":
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        for ax, (data, label) in zip(axes, panels):
            im = ax.imshow(data, vmin=-0.02, vmax=0.02)
            ax.axis("off")
            ax.text(20, 150, label, fontsize=16)
        cbar_ax = fig.add_axes([0.1, 0.1, 0.5, 0.05])
        plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    elif layout == "vertical":
        fig, axes = plt.subplots(3, 1, figsize=(6, 9))
        all_data = np.concatenate([d.ravel() for d, _ in panels])
        normalizer = Normalize(all_data.min(), all_data.max())
        for ax, (data, label) in zip(axes, panels):
            ax.imshow(data, cmap=cm.RdBu, norm=normalizer)
            ax.axis("off")
            ax.text(20, 150, label, fontsize=16)
        plt.colorbar(
            cm.ScalarMappable(norm=normalizer, cmap=cm.RdBu),
            ax=axes.ravel().tolist(),
            shrink=0.95,
            orientation="vertical",
        )
    else:
        raise ValueError(f"layout must be 'horizontal' or 'vertical', got {layout!r}")

    _save_or_show(fig, save_path)
    return fig


def plot_strain_row_cut(
    xx_array: np.ndarray,
    xz_array: np.ndarray,
    zz_array: np.ndarray,
    row: int,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Line cut of all three strain components along one pixel row.

    Args:
        xx_array, xz_array, zz_array (ndarray): Strain components, shape (n, m).
        row (int): Row index of the cut.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    for data, label in [
        (xx_array, r"$\epsilon_{xx}$"),
        (xz_array, r"$\epsilon_{xz}$"),
        (zz_array, r"$\epsilon_{zz}$"),
    ]:
        ax.plot(data[row], label=label)
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Strain")
    ax.set_title(f"Strain Along Row {row}")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# NFF and machine gun curves
# ---------------------------------------------------------------------------


def plot_polarisation_curve(
    dephasing_list: np.ndarray,
    z_polarisations: np.ndarray,
    reference: float | None = None,
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Z polarisation against dephasing strength, with the x-axis inverted so
    dephasing increases left to right (1 = none, 0.5 = maximal).

    Args:
        dephasing_list (ndarray): Dephasing values, e.g. from
            qdot.nff.dephasing_polarisation_curve.
        z_polarisations (ndarray): Z polarisation at each value.
        reference (float): Optional undephased reference, drawn as a dotted
            horizontal line (qdot.nff.non_dephased_polarisation).
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    ax.plot(dephasing_list, np.real(z_polarisations))
    if reference is not None:
        ax.axhline(reference, linestyle="dotted", color="grey")
    ax.invert_xaxis()
    ax.set_xlabel(r"Dephasing Parameter $\gamma$")
    ax.set_ylabel("Z Polarisation")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_fidelity_curves(
    error_strengths: np.ndarray,
    fidelity_curves: dict[str, np.ndarray],
    x_label: str = "Error Strength",
    y_label: str = "Fidelity",
    title: str | None = None,
    save_path: pathlib.Path | str | None = None,
) -> plt.Figure:
    """
    Machine gun fidelity (or trace distance) against error strength.

    Args:
        error_strengths (ndarray): Channel strengths swept.
        fidelity_curves (dict): Legend label → metric values, e.g. one entry
            per photon count from qdot.machine_gun.fidelity_vs_error.
        x_label, y_label (str): Axis labels.
        title (str): Optional figure title.
        save_path: File path to save. If None, display instead.

    Returns:
        Figure
    """
    fig, ax = plt.subplots()
    for label, curve in fidelity_curves.items():
        ax.plot(error_strengths, curve, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig
