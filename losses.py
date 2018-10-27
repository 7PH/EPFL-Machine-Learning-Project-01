

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
    return e.dot(e) / (2 * len(e))
