import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt


def CutoffFunc(r, R, D):
    if r <= R - D:
        return 1
    elif abs(r - R) <= D:
        return 0.5 * (1 - np.sin(np.pi * (r - R) / (2 * D)))
    elif r >= R + D:
        return 0
    else:
        return 100  # this should never happen, it's an edge case that should
        # be obvious in a graph


n_samples = 1000

# separations are all given in nm
min_separation = 0
max_separation = 1


# look for all bond types using a dictionary
bond_dict = {}
bond_dict["Ga-Ga"] = [0.35, 0.01]
bond_dict["In-In"] = [0.35, 0.01]
bond_dict["As-As"] = [0.35, 0.01]
bond_dict["Ga-As"] = [0.35, 0.01]
bond_dict["In-As"] = [0.37, 0.01]
bond_dict["Ga-In"] = [0.35, 0.01]

fig, axs = plt.subplots(nrows=2, ncols=3)

index = 0
data = np.zeros(n_samples)
r_data = np.linspace(min_separation, max_separation, n_samples)

for bond in bond_dict:
    ax = axs.flatten()[index]
    for i in range(n_samples):
        data[i] = CutoffFunc(r_data[i], bond_dict[bond][0], bond_dict[bond][1])
    ax.plot(r_data, data)
    ax.set_title("{} Bond".format(bond))
    ax.set_xlabel("Atomic Distance (nm)")
    ax.set_ylabel("Fc (r)")
    index += 1
fig.suptitle("Cutoff Functions for Different Bonds")

plt.show()
