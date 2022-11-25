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

coords = np.dstack((x,y)).reshape(15011, 2) # 15011 because that's how many records there are
c = c.reshape(15011) # 15011 because that's how many records there are
below_0 = c < 0
c[below_0] = 0

n = np.linspace(int(np.amin(coords)), int(np.amax(coords)), 1600) # 1600 here because the full sokolov data is 1600x1600 pixels wide
[xi, yi] = np.meshgrid(n,n)

full_xx_data = np.loadtxt(f"{data_path}full_epsilon_xx.txt") 
full_xy_data = np.loadtxt(f"{data_path}full_epsilon_xy.txt") 
full_yy_data = np.loadtxt(f"{data_path}full_epsilon_yy.txt")

print(f"Shape of full_xx_data = {full_xx_data.shape}")
print(f"Shape of full_xy_data = {full_xy_data.shape}")
print(f"Shape of full_yy_data = {full_yy_data.shape}")

for method in ["nearest", "linear", "cubic"]:
	interpolated_result = griddata(coords, c, (xi, yi), method)/100.0
	print(f"Shape of {method} interpolated date = {interpolated_result.shape}")

# best method appears to be the cubic spline, but will keep this code easy to modify anyway

method = "cubic"

conc_to_save = griddata(coords, c, (xi, yi), method)/100.0

filename = f"{data_path}conc_data_to_scale_{method}_interpolation.npy"

np.save(filename, conc_to_save)