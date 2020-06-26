import numpy as np

from numba.decorators import jit

from src.distances import euclidean_distance

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SEISMIC INVERSION

distance = euclidean_distance

@jit(nopython=True)
def get_points(data, step=0.1):
    d = distance(data[0], data[1])
    u, v = data[0], data[1]
    vect0, vect1 = (v[0] - u[0])/d, (v[1] - u[1])/d
    for i in range(int(d/step)+1):
        yield [u[0]+i*step*vect0, u[1]+i*step*vect1]


@jit(nopython=True)
def get_cell(p0, p1):
    return [int(p0+0.5), int(p1+0.5)]


@jit(nopython=True)
def get_dist_matrix(data, step=0.1, size=(10, 10)):
    dist_matrix = np.zeros(size)
    for point in get_points(data, step=step):
        cell = get_cell(point[0], point[1])
        dist_matrix[cell[0], cell[1]] += 1*step
    return dist_matrix


def get_observations(dataset, m, noise=None):
    """Calculate the observation obtained with the given model on the selected
    dataset.

    Args:
        dataset (array): The dataset.
        m (array): The model from wich we calculate the observations.
        noise (float): If noise is needed, the standard deviation of the normal
        distribution.
    """
    if noise:
        np.random.seed(901)
        return [np.multiply(
            get_dist_matrix(data, size=np.array(m).shape),
            m).sum() + np.random.normal(scale=noise) for data in dataset]
    else:
        return [np.multiply(
            get_dist_matrix(data, size=np.array(m).shape),
            m).sum() for data in dataset]

