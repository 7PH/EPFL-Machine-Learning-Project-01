from src.gradients import compute_logistic_gradient


def logistic_gradient_step(y, tx, w, gamma):
    grad = compute_logistic_gradient(y, tx, w)
    return w - gamma * grad


def reg_logistic_gradient_step(y, tx, w, l, gamma):
    grad = compute_logistic_gradient(y, tx, w) + 2 * l * w
    return w - gamma * grad
