import argparse
import pathlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

mpl.use('Qt5Agg')

# LOAD WITH FILE
# parser = argparse.ArgumentParser(description='Plot depth vs width vs MAPE.')
# parser.add_argument('results_dict', help='path to the pickle with results dict', type=pathlib.Path)
# args = parser.parse_args()
# with open(args.results_dict, 'rb') as f:
#     results = pickle.load(f)


# LOAD WITH PATH
results_path = '../reports/dim2_width_depth_nsamples_grid_search_results'
with open(results_path, 'rb') as f:
    results = pickle.load(f)

depths, widths = [], []
val_losses, val_mapes = [], []

for (d, w, n), (loss, metrics) in results.items():
    depths.append(d)
    widths.append(w)
    val_losses.append(loss.item())
    val_mapes.append(metrics['MAPE'].item())

depths = np.array(depths).reshape(9, 9)
widths = np.array(widths).reshape(9, 9)
val_mapes = np.array(val_mapes).reshape(9, 9)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(depths, widths, val_mapes, 50, cmap='binary')
ax.set_xlabel('depth')
ax.set_ylabel('width')
ax.set_zlabel('val_loss')