import matplotlib.pyplot as plt

def plot_map(c, d=None, vmin=5, vmax=15, title=None):
    """Plot the map.

    Args:
        c (matrix): The model to plot.
        d (matrix, optional): The datapath to overlay on the model.
        vmin (float): The minimum value for the scale.
        vmax (float): The maximum value for the scale.
        title (str): The title of the figure.
    """
    plt.figure(figsize=(7,6))
    plt.imshow(c, vmin=vmin, vmax=vmax)

    if d is not None:
        for data in d:
            plt.plot(data[:, 1], data[:, 0], c='white')

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


