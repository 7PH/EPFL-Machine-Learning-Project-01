import numpy as np


def sigmoid(tx):
    return 1 / (1 + np.exp(- tx))
