import numpy as np

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


def generate_dataset(size_data, size_model, min_lenght=None, distance=euclidean_distance, seed=None):
    """Generate a random observational dataset, i.e. random paths on the map.

    Args:
        size_data (int): The number of data to generate.
        size_model (int): The size of the model.
        min_lenght (float, optional): The minimal length for a path.
        distance (callable): The distance to use for `min_lenght`.
        seed (int, optional): The random seed

    Returns:
        (array): The data generated.
    """
    if seed: np.random.seed(seed)

    n_data = 0
    dataset = []
    while n_data < size_data:
        data = [np.random.rand((2))*size_model-0.5,
                np.random.rand((2))*size_model-0.5]
        if (min_lenght and distance(data) > min_lenght) or min_lenght is None:
            dataset.append(data)
            n_data += 1

    return np.array(dataset)

