import numpy as np
from src.maths_helpers import sigmoid


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return e.dot(e) / (2 * len(e))


def compute_mse(y, tx, w):
    """
    Compute the mean square error
    @TODO document
    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_sig_loss(y, tx, w):
    """
    @TODO refactor (important)

    :param y:
    :param tx:
    :param w:
    :return:
    """
    pred = sigmoid(tx @ w)
    return - np.asscalar((y.T @ np.log(pred)) + ((1 - y).T @ np.log(1 - pred)))


def compute_reg_sig_loss(y, tx, w, lambda_):
    """
    @TODO refactor (important)

    :param y:
    :param tx:
    :param w:
    :param lambda_:
    :return:
    """
    return compute_sig_loss(y, tx, w) + .5 * lambda_ * (w.T @ w)