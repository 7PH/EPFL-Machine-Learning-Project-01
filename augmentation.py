import numpy as np


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    @TODO document
    :param x:
    :param degree:
    :return:
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_poly_minus(x,degree):
    """
    @TODO document
    :param x:
    :param degree:
    :return:
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg),x*-1]
    return poly


def poly_plus_cos(x, degree):
    """
    @TODO remove? document?
    :param x:
    :param degree:
    :return:
    """
    poly = np.ones((len(x), 1))
    for deg in range(1,degree+1):
        poly = np.c_[poly,np.power(np.cos(x),deg)]
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def mean_replacement(x):
    """
    @TODO document
    :param x:
    :return:
    """
    inds = np.where(np.equal(-999, x))
    x_clean = x.copy()
    x_clean[inds] = np.nan
    means = np.nanmean(x_clean, axis=0)
    means_tab = np.broadcast_to(means, x_clean.shape)
    complete = np.where(np.isnan(x_clean), means_tab, x_clean)

    return complete
