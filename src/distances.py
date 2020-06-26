import numpy as np

from numba.decorators import jit


@jit(nopython=True)
def euclidean_distance(c1, c2):
    """Return the euclidean distance between two points, i.e. the length of the
    path between the two points.

    Args:
        c1 (array): Array that contain the coordinates of the first point.
        c2 (array): Array that contain the coordinates of the first point.
    """
    return np.sqrt(np.power(c1[0] - c2[0], 2) + np.power(c1[1] - c2[1], 2))


@jit(nopython=True)
def build_corr_mat(coordinates, sigma, L, size_model):
    C = np.zeros((size_model*size_model, size_model*size_model))
    for i in range(size_model*size_model):
        for j in range(size_model*size_model):
            C[i, j] = sigma * sigma * np.exp( np.divide(
                - euclidean_distance(coordinates[i], coordinates[j])**2,
                2 * L**2))
    return C



def get_gaussian_correlation_matrix(sigma, L, size_model):
    """Return a gaussiab correlation matrix for a square map of size
    `size_model`.

    Args:
        sigma (float): The standard deviation of each data.
        L (float): The correlation length of the matrix.
        size_model (int): The size of the model.

    Return:
        array: The correlation matrix of size `(size_model*size_model,
        size_model*size_model)`.
    """

    coordinates = np.concatenate(
        np.array(
            [[[i, j] for j in range(size_model)] for i in range(size_model)]
        ),
        axis=0)
    return build_corr_mat(coordinates, sigma, L, size_model)

