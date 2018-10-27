from src.maths_helpers import sigmoid


def compute_logistic_gradient(y, tx, w):
    """Compute gradient for logistic gradient descent"""
    return tx.T @ (sigmoid(tx @ w) - y)