import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
from backbone_quadrupolar_functions import graph_path
from backbone_quadrupolar_functions import data_path
import matplotlib.pyplot as plt

conc_data = sio.loadmat(f"{data_path}conc_data")
x = conc_data["x"]
y = conc_data["y"]
c = conc_data["conc"]

coords = np.dstack((x, y)).reshape(
    15011, 2
)  # 15011 because that's how many records there are
c = c.reshape(15011)  # 15011 because that's how many records there are
below_0 = c < 0
c[below_0] = 0

n = np.linspace(int(np.amin(coords)), int(np.amax(coords)), 1600)
[xi, yi] = np.meshgrid(n, n)

for method in ["nearest", "linear", "cubic"]:
    filename = f"{graph_path}concentration_interpolation_method_{method}.png"
    interpolated_result = griddata(coords, c, (xi, yi), method) / 100.0

    plt.imshow(interpolated_result)
    # plt.show()
    plt.colorbar()
    plt.savefig(filename)
    plt.close()
