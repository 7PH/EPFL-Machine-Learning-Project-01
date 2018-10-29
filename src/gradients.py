import numpy as np
from src.maths_helpers import sigmoid


def compute_logistic_gradient(y, tx, w):
    """
    Compute gradient for logistic gradient descent

    :param y:
    :param tx:
    :param w:
    :return:
    """
    return tx.T @ (sigmoid(tx @ w) - y)


def compute_gradient_mse(y, tx, w):
    """
    this function computes the gradient with a mse lost function

    :param y:
    :param tx:
    :param w:
    :return:
    """

    e = y - np.dot(tx, w)
    grad = - np.dot(tx.T, e) / tx.shape[0]
    return grad, e
