import numpy as np
from src.maths_helpers import sigmoid


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return e.T.dot(e) / (2 * len(e))


def compute_mse(y, tx, w):
    """
    Compute the mean square error

    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_sig_loss(y, tx, w):
    """
    Compute loss for logistic regression

    :param y:
    :param tx:
    :param w:
    :return:
    """
    # Compute loss
    pred = sigmoid(- tx @ w)
    return np.ones(tx.shape[0]).dot(pred) - (y.T @ tx) @ w


def compute_reg_sig_loss(y, tx, w, lambda_):
    """
    Compute loss for regularized logistic regression

    :param y:
    :param tx:
    :param w:
    :param lambda_:
    :return:
    """
    return compute_sig_loss(y, tx, w) + .5 * lambda_ * (w.T @ w)
