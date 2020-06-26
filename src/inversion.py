import numpy as np

from numba.decorators import jit


@jit(nopython=True)
def get_m_linear(m0, C_m0, d0, C_d0, G):
    """Return the model resulting of a linear inversion (following d=G.m).

    Args
        m0 (numpy array): The initial model (size N_m).
        C_m0 (numpy array): The correlation matrix of the model (size N_mxN_m).
        d0 (numpy array): The data (size N_d).
        C_d0 (numpy array): The correlation matrix of the data (size N_dxN_d).
        G (numpy array): The linear model that links the data and the model (size N_dxN_m).
    """
    residuals = np.subtract(d0, np.dot(G, m0))
    M = C_d0 + np.dot(np.dot(G, C_m0), np.transpose(G))
    return m0 + np.dot(np.dot(C_m0, np.transpose(G)), np.dot(np.linalg.inv(M), residuals))
