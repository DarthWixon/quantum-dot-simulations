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
