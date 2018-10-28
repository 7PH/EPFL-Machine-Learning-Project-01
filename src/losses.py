def calculate_mse(e):
    """Calculate the mse for vector e."""
    return e.dot(e) / (2 * len(e))


def compute_mse(y, tx, w):
    """
    Compute the mean square error
    @TODO document
    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
