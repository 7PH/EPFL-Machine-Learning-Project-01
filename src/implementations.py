import numpy as np

from src.gradients import compute_gradient_mse
from src.losses import calculate_mse
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

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Given a gamma parameter, we iterate to find the weight's vector :"max_iters" given a parameter.
    
    # Set an initial weight's vector and compute its loss
    w = initial_w
    loss = compute_loss(y, tx, w)
    
    print("first loss: "+str(loss))

    for iter in range(max_iters):
        
        # compute loss and gradient
        grad, e = compute_gradient_mse(y, tx, w)
        loss = calculate_mse(e)

        # update the weight's vector 
        w -= gamma * grad 

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Given a gamma parameter, we iterate to find the weight's vector :"max_iters" given a parameter.
    
    # Set an initial weight's vector and compute its loss
    w = initial_w
    loss = compute_loss(y, tx, w)
    gamma_t = gamma
    for iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            # compute a stochastic gradient and loss
            grad, e = compute_gradient_mse(y_batch, tx_batch, w)
            loss = calculate_mse(e)

            # stochastic gradient update
            w -= gamma * grad / (iter+1)
            
    return w, loss
  

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
