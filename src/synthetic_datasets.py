import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from itertools import product

from src.distances import euclidean_distance

def generate_model(size_model, m_mean, m_std, seed=None):
    """Generate a random model of size `size_model`x`size_model`.

    Args:
        size_model (int): The size of the model.
        m_mean (float): Average value of the model.
        m_std (float): Standard deviation of the model.
        seed (int, optional): The random seed.

    Returns:
        (array): The model generated.
    """
    if seed: np.random.seed(seed)
    return [[
            np.random.normal(m_mean, m_std) for j in range(size_model)
        ] for i in range(size_model)]


def generate_gaussian_model(size_model, m_mean, m_std, seed=None):
    """Generate a random model using gaussian processes.

    Args:
        size_model (int): The size of the model.
        m_mean (float): Average value of the model.
        m_std (float): Standard deviation of the model.
        seed (int, optional): The random seed.

    Returns:
        (array): The model generated.
    """
    if seed: np.random.seed(seed)
    x = np.random.randint(size_model, size=10)
    y = np.random.randint(size_model, size=10)
    z = [np.random.normal(0, m_std) for i in range(10)]

    gp = GaussianProcessRegressor(kernel=RBF())
    gp.fit(np.transpose([x, y]), z)

    c_gp = gp.predict([
        (i, j) for i, j in product(
            range(size_model),
            range(size_model))
    ]).reshape(size_model, size_model)

    return c_gp + generate_model(size_model, m_mean=m_mean - np.mean(c_gp),
                             m_std=m_std/7, seed=seed)


def generate_dataset(size_data,
                     size_model,
                     min_lenght=None,
                     distance=euclidean_distance,
                     seed=None):
    """Generate a random observational dataset, i.e. random paths on the map.

    Args:
        size_data (int): The number of data to generate.
        size_model (int): The size of the model.
        min_lenght (float, optional): The minimal length for a path.
        distance (callable): The distance to use for `min_lenght`.
        seed (int, optional): The random seed

    Returns:
        (array): The dataset generated.
    """
    if seed: np.random.seed(seed)

    n_data = 0
    dataset = []
    while n_data < size_data:
        data = [np.random.rand((2))*size_model-0.5,
                np.random.rand((2))*size_model-0.5]
        if (min_lenght and distance(data[0], data[1]) > min_lenght) or min_lenght is None:
            dataset.append(data)
            n_data += 1

    return np.array(dataset)
