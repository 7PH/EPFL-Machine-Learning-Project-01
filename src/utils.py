from src.gradients import compute_logistic_gradient


def logistic_gradient_step(y, tx, w, gamma):
    """
    Step for regularized logistic gradient

    :param y: Values
    :param tx: Data
    :param w: Old weight
    :param gamma: Gamma parameter
    :return: New weight
    """
    grad = compute_logistic_gradient(y, tx, w)
    return w - gamma * grad


def reg_logistic_gradient_step(y, tx, w, l, gamma):
    """
    Step for regularized logistic gradient

    :param y: Values
    :param tx: Data
    :param w: Old weight
    :param l: Lambda parameter
    :param gamma: Gamma parameter
    :return: New weight
    """
    grad = compute_logistic_gradient(y, tx, w) + 2 * l * w
    return w - gamma * grad
