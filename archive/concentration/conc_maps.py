import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import backbone_quadrupolar_functions as bqf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_In_concentration_data(region_bounds, step_size=1, method="cubic"):
    """
    Loads the interpolated concentration data from Sokolov.

    Paper these data are from (DOI): 10.1103/PhysRevB.93.045301

    Args:
        region_bounds (list): A list of 4 integers, that definte the edges of the area to
            be looked at. In order of [left, right, top, bottom].
            Entire dot region is:[100, 1200, 200, 1000].
            Dot only region is: [100,1200, 439, 880].
            Standard small testing region is: [750, 800, 400, 500].
        step_size (int): The interval over which to select sites in the range. Higher
            intervals correspond to fewer sites. Default is 1 (every site is used).
        method (str): The interpolation method used to get the data to be the
                same size as the strain data. Default is cubic (using a cubic spline).

    Returns:
        dot_conc_data (ndarray): The percentage concentration of In115 in the QD.
    """

    filename = f"{data_path}conc_data_to_scale_{method}_interpolation.npy"
    full_conc_data = np.load(filename)

    H_1, H_2, L_1, L_2 = region_bounds

    # slice out range of data we want, slicing syntax is [start:stop:step]
    dot_conc_data = full_conc_data[L_1:L_2:step_size, H_1:H_2:step_size]

    return dot_conc_data


def plot_concentration_image(region_bounds, saving=False):
    conc_data = load_In_concentration_data(region_bounds)

    plt.imshow(conc_data, cmap=cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())
    plt.colorbar(orientation="horizontal", label="Indium Concentration", shrink=0.5)
    plt.axis("off")
    # plt.title("Indium Concentration")
    # plt.tight_layout()

    if saving:
        filename = f"{graph_path}In_concentration_map_unflipped.png"
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved graph: {filename}")
    else:
        plt.show()


def rectangle_highlights(dot_region_bounds, rect_specs):
    conc_data = load_In_concentration_data(dot_region_bounds)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.imshow(conc_data, cmap=cm.GnBu, vmin=conc_data.min(), vmax=conc_data.max())

    plt.colorbar(orientation="horizontal")
    # plt.axis("off")
    plt.title("Indium Concentration")

    for rect in rect_specs:
        left, bottom, width, height = rect
        highlight = plt.Rectangle(
            (left, bottom), width, height, edgecolor="black", linewidth=1, fill=False
        )
        ax.add_patch(highlight)

    plt.tight_layout()
    plt.show()

    filename = f"{graph_path}highlight_regions_draft.png"
    # plt.savefig(filename)
    plt.close()


def region_bound_finder(base_region_bounds, rectangle_description):
    # this function should return an array in the format of region_bounds for use with other functions

    # base region defined as: [left, right, top, bottom]
    # rectangle specs are in the order: [left, bottom, width, height], but imshow plots from the top-left, so "bottom = 0" is at the top of the image

    left = base_region_bounds[0] + rectangle_description[0]
    right = left + rectangle_description[2]
    top = base_region_bounds[2] + rectangle_description[1]
    bottom = top + rectangle_description[3]

    new_region_bounds = [left, right, top, bottom]

    return new_region_bounds


if __name__ == "__main__":
    standard_region_bounds = [100, 1200, 200, 1000]
    big_region_bounds = [10, 1500, 400, 900]
    entire_region = [0, 1600, 0, 1600]
    plot_concentration_image(entire_region, saving=False)

    # # rectangle specs are in the order: [left, bottom, width, height]
    # # imshow plots from the top-left, so "bottom = 0" is at the top of the image

    # # for the old standard region_bounds ([100, 1200, 200, 1000])
    # # central region is:        [400, 360, 250, 80]
    # # left bulbous edge is:     [220, 300, 80, 250]

    # # for the potential new ones ([10, 1500, 400, 900])
    # # central region with high In:  [440, 150, 400, 100]
    # # left region:                  [10, 250, 400, 100]
    # # right region:                 [1000, 250, 400, 100]
    # # cental region with low In:    [440, 260, 400, 100]

    # standard_rect_specs = []

    # standard_rect_specs.append([400, 360, 250, 80])
    # standard_rect_specs.append([220, 300, 80, 250])

    # big_rect_specs = []

    # big_rect_specs.append([440, 150, 400, 100])
    # big_rect_specs.append([10, 250, 400, 100])
    # big_rect_specs.append([1000, 250, 400, 100])
    # big_rect_specs.append([1000, 50, 400, 100])
    # big_rect_specs.append([440, 260, 400, 100])
    # big_rect_specs.append([440, 375, 400, 100])
    # big_rect_specs.append([550, 200, 5, 5])

    # rectangle_highlights(big_region_bounds, big_rect_specs)

    # for rect in big_rect_specs:
    #     print(region_bound_finder(big_region_bounds, rect))

    # region_bounds = [100,1500, 200, 1000]
    # plot_concentration_image(region_bounds, saving = True)
