import numpy as np
from src.gradients import compute_gradient_mse
from src.helpers import batch_iter
from src.losses import calculate_mse, compute_mse, compute_sig_loss, compute_reg_sig_loss
from src.utils import logistic_gradient_step, reg_logistic_gradient_step


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Given a gamma parameter, we iterate to find the weight vector :"max_iters" given a parameter.

    :param y: Labels vector
    :param tx: Features matrix
    :param initial_w: initial weight vector
    :param max_iters: number of iterations
    :param gamma: learning rate
    :return: weight prediction and mse loss of that prediction
    """

    # Set an initial weight's vector and compute its loss
    w = initial_w

    for iter in range(max_iters):
        # compute loss and gradient
        grad, e = compute_gradient_mse(y, tx, w)

        # update the weight's vector
        w -= gamma * grad

    return w, compute_mse(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Given a gamma parameter, we iterate to find the weight's vector

    :param y: Labels vector
    :param tx: Features matrix
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations
    :param gamma: Learning rate
    :return: Weight prediction and mse loss of that prediction
    """

    # Set an initial weight's vector and compute its loss
    w = initial_w
    loss = compute_mse(y, tx, w)
    for iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, 1):
            # compute a stochastic gradient and loss
            grad, e = compute_gradient_mse(y_batch, tx_batch, w)
            loss = calculate_mse(e)

            # stochastic gradient update
            w -= gamma * grad / (iter + 1)

    return w, loss


def least_squares(y, tx):
    """
    Computes the solution using least_square solution w* = (X'X)^(-1) * x'Y

    :param  y: Labels vector
    :param tx: Feature matrix
    :return: Weight prediction and mse loss of that prediction
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    :param y: Labels vector
    :param tx: Feature matrix
    :param lambda_: Penalisation factor
    :return: Weight and mse losss of the prediction
    """
    ai = lambda_ * np.identity(tx.shape[1])
    a = tx.T @ tx + ai
    b = tx.T @ y

    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("Singular", lambda_)
            w = np.full((a.shape[0],), -np.inf)
        else:
            raise
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    :param y: Labels vector
    :param tx: Feature matrix
    :param initial_w: Initital weight vector
    :param max_iters: Number of iterations
    :param gamma: Learning rate
    :return:
    """
    for i in range(max_iters):
        initial_w = logistic_gradient_step(y, tx, initial_w, gamma)
    return initial_w, compute_sig_loss(y, tx, initial_w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    :param y: Labels vector
    :param tx: Feature matrix
    :param lambda_: Regularition coefficient
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations
    :param gamma: Learning rate
    :return:
    """
    for i in range(max_iters):
        initial_w = reg_logistic_gradient_step(y, tx, initial_w, lambda_, gamma)
    return initial_w, compute_reg_sig_loss(y, tx, initial_w, lambda_)