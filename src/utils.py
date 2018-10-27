from src.gradients import compute_logistic_gradient


def logistic_gradient_step(y, tx, w, gamma):
    """
    @TODO move this (logistic helper)
    :param y:
    :param tx:
    :param w:
    :param gamma:
    :return:
    """
    grad = compute_logistic_gradient(y, tx, w)
    return w - gamma * grad


def reg_logistic_gradient_step(y, tx, w, l, gamma):
    """
    @TODO move this (logistic helper)
    :param y:
    :param tx:
    :param w:
    :param l:
    :param gamma:
    :return:
    """
    grad = compute_logistic_gradient(y, tx, w) + 2 * l * w
    return w - gamma * grad
