import numpy as np
from src.gradients import compute_gradient_mse
from src.helpers import batch_iter
from src.losses import calculate_mse, compute_mse, compute_sig_loss, compute_reg_sig_loss
from src.utils import logistic_gradient_step, reg_logistic_gradient_step


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Given a gamma parameter, we iterate to find the weight's vector :"max_iters" given a parameter.

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """

    # Set an initial weight's vector and compute its loss
    w = initial_w
    loss = compute_mse(y, tx, w)

    print("first loss: " + str(loss))

    for iter in range(max_iters):
        # compute loss and gradient
        grad, e = compute_gradient_mse(y, tx, w)
        loss = calculate_mse(e)

        # update the weight's vector 
        w -= gamma * grad

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Given a gamma parameter, we iterate to find the weight's vector :"max_iters" given a parameter.

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """

    # Set an initial weight's vector and compute its loss
    w = initial_w
    loss = compute_mse(y, tx, w)
    for iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, e = compute_gradient_mse(y_batch, tx_batch, w)
            loss = calculate_mse(e)

            # stochastic gradient update
            w -= gamma * grad / (iter + 1)

    return w, loss


def least_squares(y, tx):
    """
    @TODO document
    :param y:
    :param tx:
    :return:
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    ai = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + ai
    b = tx.T.dot(y)

    try:
        c = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("singular", lambda_)
            c = np.full((a.shape[0],), -np.inf)
        else:
            raise
    return c


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    for i in range(max_iters):
        initial_w = logistic_gradient_step(y, tx, initial_w, gamma)
    return initial_w, compute_sig_loss(y, tx, initial_w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    for i in range(max_iters):
        initial_w = reg_logistic_gradient_step(y, tx, initial_w, lambda_, gamma)
    return initial_w, compute_reg_sig_loss(y, tx, initial_w, lambda_)
