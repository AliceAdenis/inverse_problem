import numpy as np

from numba.decorators import jit


@jit(nopython=True)
def euclidean_distance(data):
    """Return the euclidean distance between two points, i.e. the length of the
    path between the two points.

    Args:
        data (array): Array that contain the coordinates of the end of the path.
    """
    u, v = data[0], data[1]
    return np.sqrt(np.power(u[0] - v[0], 2) + np.power(u[1] - v[1], 2))

