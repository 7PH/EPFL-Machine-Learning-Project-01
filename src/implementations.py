import numpy as np
from src.utils import logistic_gradient_step, reg_logistic_gradient_step


def least_squares(y, tx):
    """
    @TODO document
    :param y:
    :param tx:
    :return:
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def logistic_gradient(y, tx, w, gamma, max_iter=100):
    """
    @TODO return also loss
    :param y:
    :param tx:
    :param w:
    :param gamma:
    :param max_iter:
    :return:
    """
    for i in range(max_iter):
        w = logistic_gradient_step(y, tx, w, gamma)
    return w


def reg_logistic_gradient(y, tx, w, l, gamma, max_iter=100):
    """
    @TODO return also loss
    :param y:
    :param tx:
    :param w:
    :param l:
    :param gamma:
    :param max_iter:
    :return:
    """
    for i in range(max_iter):
        w = reg_logistic_gradient_step(y, tx, w, l, gamma)
    return w


def ridge_regression(y, tx, lamb):
    """
    @TODO document
    :param y:
    :param tx:
    :param lamb:
    :return:
    """
    ai = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + ai
    b = tx.T.dot(y)

    try:
        c = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("singular", lamb)
            c = np.full((a.shape[0],), -np.inf)
        else:
            raise
    return c
