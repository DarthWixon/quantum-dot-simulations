import sys

sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt


def EdgeCounter(n_rows, n_cols):
    return 2 * n_rows * n_cols + n_rows + n_cols


max_rows = 100
max_cols = 100

row_range = np.arange(max_rows)
col_range = np.arange(max_cols)

data = np.zeros((max_rows, max_cols))

for r in row_range:
    for c in col_range:
        data[r, c] = EdgeCounter(r, c)

plt.imshow(data, origin="lower")
plt.xlabel("Number of Columns of Atoms")
plt.ylabel("Number of Rows of Atoms")
plt.colorbar()

plt.show()
