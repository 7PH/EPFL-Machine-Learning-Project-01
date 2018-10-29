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
    # Prevents passing 0 to log function
    log_precision = 0.00001
    # Compute loss
    pred = sigmoid(tx @ w)
    pred[np.where(pred > 1 - log_precision)] = 1 - log_precision
    pred[np.where(pred < log_precision)] = log_precision
    return - (y.T @ np.log(pred)) + ((1 - y).T @ np.log(1. - pred))


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
