import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from backbone_quadrupolar_functions import data_path
from backbone_quadrupolar_functions import nuclear_species_list

# AS OF 16/10/20 this is an outdated file, and should not be used. Keeping it for posterity and comparison.

# what is the format of my strain tensor?

# strain_tensor is an array of size n_rows x n_cols x 2 x 2 for 2D structures
# strain_tensor is an array of size n_rows x n_cols x n_layers x 3 x3 for 3D structures
# do 2D case first, then should expland easily enough to 3D

epsilon_xx_1 = np.loadtxt(f"{data_path}full_epsilon_xx.txt")
epsilon_xy_1 = np.loadtxt(f"{data_path}full_epsilon_xy.txt")
epsilon_yy_1 = np.loadtxt(f"{data_path}full_epsilon_yy.txt")

# Region with quantum dot
# H_1=100
# H_2=1200
# L_1=200
# L_2=1000

H_1, H_2, L_1, L_2 = [750, 800, 400, 500]

epsilon_xx = epsilon_xx_1[L_1:L_2:1, H_1:H_2:1]
epsilon_xz = -epsilon_xy_1[L_1:L_2:1, H_1:H_2:1]
epsilon_zz = -epsilon_yy_1[L_1:L_2:1, H_1:H_2:1]


def SokolovEFG(nuclearSpecies):
    # only the 69Ga has been updated to SI units as of 26/11/19
    if nuclearSpecies == "In115":
        S11 = 5.01e22
        S12 = -S11 / 2
        S44 = -2.998e22
    if nuclearSpecies == "Ga69":
        S11 = 2.73e22
        S12 = -S11 / 2
        S44 = -2.76e22
    if nuclearSpecies == "Ga71":
        S11 = 2.73e22
        S12 = -S11 / 2
        S44 = -2.73e22
    if nuclearSpecies == "As75":
        S11 = 3.96e22
        S12 = -S11 / 2
        S44 = 7.94e22

    n = epsilon_xx.shape[0]
    m = epsilon_xx.shape[1]

    eta = np.zeros((n, m))
    vec = np.zeros((n, m, 3))
    V_ZZ = np.zeros((n, m))
    V_XX = np.zeros((n, m))
    V_YY = np.zeros((n, m))

    # Full EFG tensor
    for i in range(n):
        for j in range(m):
            V = np.array(
                [
                    [
                        S12 * (epsilon_zz[i, j] - epsilon_xx[i, j]),
                        0,
                        S44 * epsilon_xz[i, j],
                    ],
                    [
                        0,
                        (S12 + S11) * epsilon_xx[i, j] + S12 * epsilon_zz[i, j],
                        S44 * epsilon_xz[i, j],
                    ],
                    [
                        S44 * epsilon_xz[i, j],
                        S44 * epsilon_xz[i, j],
                        2 * S12 * epsilon_xx[i, j] + S11 * epsilon_zz[i, j],
                    ],
                ]
            )

            w, v = np.linalg.eig(V)

            # this max function finds the index of the maximum value in w
            max1 = max(enumerate(abs(w)), key=lambda x: x[1])[0]
            V_ZZ[i, j] = w[max1]

            # change the old maximum value to 1, I think just so we don't find it twice (w has values of the order 1e13)
            w[max1] = 1

            # repeat search to find next highest value?
            max2 = max(enumerate(abs(w)), key=lambda x: x[1])[0]

            V_YY[i, j] = w[max2]
            w[max2] = 1

            # repeat again, so we have found the top 3 values (all >=1)
            max3 = max(enumerate(abs(w)), key=lambda x: x[1])[0]
            V_XX[i, j] = w[max3]

            eta[i, j] = (V_XX[i, j] - V_YY[i, j]) / V_ZZ[i, j]

            # save the V_ZZ eigenvalue as a specific vector, not sure why only this one is kept
            vec[i, j] = v[max1]

    return V_ZZ, V_YY, V_XX, n, m, eta, vec


def ChekovichEFG(nuclearSpecies):
    # not all of these have been updated by Checkovich as of 20/19/20
    if nuclearSpecies == "In115":
        S11 = 5.01e22
        S12 = -S11 / 2
        S44 = -2.998e22
    if nuclearSpecies == "Ga69":
        S11 = -22e21
        S12 = -S11 / 2
        S44 = -0.4 * -22e21
    if nuclearSpecies == "Ga71":
        S11 = 2.73e22
        S12 = -S11 / 2
        S44 = -2.73e22
    if nuclearSpecies == "As75":
        S11 = 2.42e22
        S12 = -S11 / 2
        S44 = 1.98 * 24.2e21

    n = epsilon_xx.shape[0]
    m = epsilon_xx.shape[1]

    eta = np.zeros((n, m))
    vec = np.zeros((n, m, 3))
    V_ZZ = np.zeros((n, m))
    V_XX = np.zeros((n, m))
    V_YY = np.zeros((n, m))

    # Full EFG tensor
    for i in range(n):
        for j in range(m):
            V = np.array(
                [
                    [
                        S12 * (epsilon_zz[i, j] - epsilon_xx[i, j]),
                        0,
                        S44 * epsilon_xz[i, j],
                    ],
                    [
                        0,
                        (S12 + S11) * epsilon_xx[i, j] + S12 * epsilon_zz[i, j],
                        S44 * epsilon_xz[i, j],
                    ],
                    [
                        S44 * epsilon_xz[i, j],
                        S44 * epsilon_xz[i, j],
                        2 * S12 * epsilon_xx[i, j] + S11 * epsilon_zz[i, j],
                    ],
                ]
            )

            w, v = np.linalg.eig(V)

            # this max function finds the index of the maximum value in w
            max1 = max(enumerate(abs(w)), key=lambda x: x[1])[0]
            V_ZZ[i, j] = w[max1]

            # change the old maximum value to 1, I think just so we don't find it twice (w has values of the order 1e13)
            w[max1] = 1

            # repeat search to find next highest value?
            max2 = max(enumerate(abs(w)), key=lambda x: x[1])[0]

            V_YY[i, j] = w[max2]
            w[max2] = 1

            # repeat again, so we have found the top 3 values (all >=1)
            max3 = max(enumerate(abs(w)), key=lambda x: x[1])[0]
            V_XX[i, j] = w[max3]

            eta[i, j] = (V_XX[i, j] - V_YY[i, j]) / V_ZZ[i, j]

            # save the V_ZZ eigenvalue as a specific vector, not sure why only this one is kept
            vec[i, j] = v[max1]

    return V_ZZ, V_YY, V_XX, n, m, eta, vec


def Plotter(V_ZZ, V_YY, V_XX, n, m, eta, vec, data_2018=True):
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(
        np.arange(0, np.size(epsilon_xx[0, :]), 1),
        np.arange(0, np.size(epsilon_xx[:, 0]), 1),
    )

    angles = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            angles[i, j] = np.atan(vec[i, j, 2] / vec[i, j, 0]) * 180 / np.pi

    step = 50

    # the first 2 sets plot the black arrows
    # lines are repeated to get arrowheads at both ends

    # ax.quiver( X[::step, ::step], Y[::step, ::step], V_ZZ[::step, ::step], V_ZZ[::step, ::step],
    #             angles=angles[::step, ::step]+0.2*angles[::step, ::step], minshaft=10, pivot='middle', color='k', units='dots',alpha=0.4)
    # ax.quiver( X[::step, ::step], Y[::step, ::step], V_ZZ[::step, ::step], V_ZZ[::step, ::step],
    #             angles=angles[::step, ::step]+180+0.2*angles[::step, ::step], minshaft=10, pivot='middle', color='k', units='dots',alpha=0.4)

    # ax.quiver( X[::step, ::step], Y[::step, ::step], V_ZZ[::step, ::step], V_ZZ[::step, ::step],
    #             angles=angles[::step, ::step]-0.2*angles[::step, ::step], minshaft=10, pivot='middle', color='k', units='dots',alpha=0.4)
    # ax.quiver( X[::step, ::step], Y[::step, ::step], V_ZZ[::step, ::step], V_ZZ[::step, ::step],
    #            angles=angles[::step, ::step]+180-0.2*angles[::step, ::step], minshaft=10, pivot='middle', color='k', units='dots',alpha=0.4)

    # these plot the red arrows showing the actual direction
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        V_ZZ[::step, ::step],
        V_ZZ[::step, ::step],
        angles=angles[::step, ::step],
        minshaft=5,
        pivot="middle",
        color="r",
        units="dots",
    )
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        V_ZZ[::step, ::step],
        V_ZZ[::step, ::step],
        angles=angles[::step, ::step] + 180,
        minshaft=5,
        pivot="middle",
        color="r",
        units="dots",
    )

    im = ax.imshow(eta, cmap=cm.Blues, vmin=eta.min(), vmax=eta.max())
    plt.axis("off")

    if data_2018:
        plt.title("Checkovich Data")
    else:
        plt.title("Sundfors Data")

    cbar_ax = fig.add_axes([0.9, 0.10, 0.03, 0.8])

    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.set_xlabel("$\eta$", fontsize=20)

    # plt.savefig('eta.eps')
    # savetxt('Vzz.txt', V_ZZ)
    plt.show()


output = ["V_ZZ", "V_YY", "V_XX", "n", "m", "eta", "vec"]

for nuclearSpecies in nuclear_species_list:
    print("------------")
    print(nuclearSpecies)
    Sok_results = SokolovEFG(nuclearSpecies)
    Chek_results = ChekovichEFG(nuclearSpecies)

    for i in range(len(Chek_results)):
        print(f"Same for {output[i]}: {np.allclose(Chek_results[i], Sok_results[i])}")

# Plotter(*Sok_results, data_2018 = False)
# Plotter(*Chek_results, data_2018 = True)


# ----------------------------------------------------------------------------------

# to do
# make a graph that compares the 2 values of S11 and S44 for 69Ga
# just make another function that does the same maths with different values
# then plot both and whack it on the same set of axes - bosh
