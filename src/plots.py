import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap

# Colormaps
viridis = get_cmap('viridis', 6)
colors = np.append(
    np.append(viridis(range(6)), [[1, 1, 1, 1]], axis=0),
    viridis([5-i for i in range(6)]),
    axis=0)
viridis_residuals = LinearSegmentedColormap.from_list("mycmap", colors)


def plot_map(c, d=None, p=None, vmin=5, vmax=15, figsize=(7,6), title=None,
             cmap='viridis'):
    """Plot the map.

    Args:
        c (matrix): The model to plot.
        d (matrix, optional): The datapath to overlay on the model.
        p (array or iterator): The points to plot allong the path.
        vmin (float): The minimum value for the scale.
        vmax (float): The maximum value for the scale.
        title (str): The title of the figure.
    """
    if cmap == 'viridis_residuals': cmap = viridis_residuals

    plt.figure(figsize=figsize)
    plt.imshow(c, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar()

    if d is not None:
        for data in d:
            plt.plot(data[:, 1], data[:, 0], c='white')

    if p is not None:
        for point in p:
            plt.scatter(point[1], point[0], c='r', s=0.5, zorder=10)

    if title:
        plt.title(title)
    plt.show()


