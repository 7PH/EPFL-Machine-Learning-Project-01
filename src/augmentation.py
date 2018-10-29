import numpy as np

from src.helpers import get_column_names


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x:
    :param degree:
    :return:
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_poly_minus(x, degree):
    """
    :param x:
    :param degree:
    :return:
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg), x * -1]
    return poly


def poly_plus_cos(x, degree):
    """
    @TODO remove? document?
    :param x:
    :param degree:
    :return:
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(np.cos(x), deg)]
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def mean_replacement(x):
    """
    this fucntion replace the '-999' values with the mean of the column
    :param x: dataset
    :return: new dataset
    """
    inds = np.where(np.equal(-999, x))
    x_clean = x.copy()
    x_clean[inds] = np.nan
    means = np.nanmean(x_clean, axis=0)
    means_tab = np.broadcast_to(means, x_clean.shape)
    complete = np.where(np.isnan(x_clean), means_tab, x_clean)

    return complete


def gaussian_distance_22(x, cols, old_new):
    """
    :param x:
    :param cols:
    :param old_new:
    :return:
    """
    types = ["mass", "centrality", "eta"]

    ar = np.array(get_column_names())

    names_cols = list(ar[cols])

    for typ in types:
        for i, coli in enumerate(names_cols):
            if typ not in coli:
                continue
            for j, colj in enumerate(names_cols):
                if j <= i:
                    continue
                if typ not in colj:
                    continue
                x = np.c_[x, np.exp(-1.0 * (np.power(x[:, old_new[i]] - x[:, old_new[j]], 2) / 2.0))]
    return x


def angles_extension_22(x, remaining_columns, old_new, nc):
    """
    @TODO document? move? => yes please move :) 
    Adds cosine difference of pairwise angle to the data.
    """
    full = x

    angle_features = [i for i in range(len(nc)) if nc[i].endswith('phi')]

    # Add pairwise modulo differences
    angle_features_remaning = np.intersect1d(angle_features, remaining_columns)

    for i in angle_features_remaning:
        for j in angle_features_remaning:
            if j <= i:
                continue

            full = np.c_[full, np.cos(x[:, old_new[i]] - x[:, old_new[j]])]

    # Add cosines
    for i in angle_features_remaning:
        full = np.c_[full, np.cos(x[:, old_new[i]])]

    return full


def drop_999(x):
    """
    this function drops a column if more than 50% of its elements are -999
    :param x:
    :return:
    """
    drop = {}
    count = {}
    column_to_delete = []
    column_remaining = []
    old_new = {}
    for i in range(30):
        drop[i] = x[:, i] == -999
        count[i] = np.sum(drop[i])
        if count[i] > (x.shape[0] * 0.5):
            column_to_delete.append(i)
        else:
            column_remaining.append(i)

        for new, old in enumerate(column_remaining):
            old_new[old] = new

    x_dropped = np.delete(x, column_to_delete, 1)
    return x_dropped, column_remaining, old_new

def features_expansion(x, degree=9):
    """
    this function does the preprocessing and then  the augmentation using angles_extansion, build_poly
    and gaussiance distances.
    :param x:
    :param degree:

    :return: augmented dataset
    """
    x_dropped, cols, old_new = drop_999(x)
    x_stuff = build_poly(angles_extension_22(x_dropped, cols, old_new, get_column_names()), degree)
    x_plus = gaussian_distance_22(x_stuff, cols, old_new)
    return x_plus
