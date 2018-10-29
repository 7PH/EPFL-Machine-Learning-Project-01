import numpy as np


def sigmoid(tx):
    """
    Sigmoid function

    :param tx: Numpy array
    :return: 1 / (1 + e(- t))
    """
    return 1 / (1 + np.exp(- tx))
