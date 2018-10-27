import numpy as np
from src.augmentation import build_poly
from src.helpers import predict_labels, label_accuracy
from src.implementations import ridge_regression

# @TODO get this outta here
names_columns = 'DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,' \
                'DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,' \
                'DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,' \
                'PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,' \
                'PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,' \
                'PRI_jet_subleading_phi,PRI_jet_all_pt'.split(',')


def gaussian_distance_22(x, cols, old_new):
    """
    @TODO document
    :param x:
    :param cols:
    :param old_new:
    :return:
    """
    types = ["mass", "centrality", "eta"]

    ar = np.array(names_columns)

    names_cols = list(ar[cols])
    # print(names_cols)

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


def angles_extension_22(x, remaining_columns, old_new, nc=names_columns):
    """
    @TODO document? move?
    Adds absolute values of pairwise angle differences to the data.
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


def make_22_weights(y_tr, x, lambda_, degree):
    """
    @TODO document
    :param y_tr:
    :param x:
    :param lambda_:
    :param degree:
    :return:
    """
    weights = {}
    jet_masks = [
        x[:, 22] == 0,
        x[:, 22] == 1,
        x[:, 22] > 1]

    ys_train = [y_tr[mask] for mask in jet_masks]
    xs = [x[mask] for mask in jet_masks]
    x_p_tr = {}

    for i, mask in enumerate(jet_masks):
        if i == 0:
            x_p_tr[i] = features_expansion(xs[i], degree + 1)
        if i == 1:
            x_p_tr[i] = features_expansion(xs[i], degree + 2)
        if i > 1:
            x_p_tr[i] = features_expansion(xs[i], degree + 3)

        weights[i] = ridge_regression(ys_train[i], x_p_tr[i], lambda_)

    return weights


def predict_22_stuff(weights, x, degree):
    """
    @TODO document
    :param weights:
    :param x:
    :param degree:
    :return:
    """
    # weights = {}
    jet_masks = [
        x[:, 22] == 0,
        x[:, 22] == 1,
        x[:, 22] > 1]

    xs = [x[mask] for mask in jet_masks]

    y_sub = np.zeros(x.shape[0])
    x_p_te = {}
    for i, mask in enumerate(jet_masks):
        if i == 0:
            x_p_te[i] = features_expansion(xs[i], degree + 1)
        if i == 1:
            x_p_te[i] = features_expansion(xs[i], degree + 2)
        if i > 1:  # two  in our case :)
            x_p_te[i] = features_expansion(xs[i], degree + 3)

        # print(i)
        # print(x_p_te[i].shape)
        # print(weights[i].shape)

        y_sub[mask] = predict_labels(weights[i], x_p_te[i])  # should be y, should be x_p_te[i]

    # print(y_sub.shape)

    return y_sub


def drop_999(x):
    """
    @TODO document
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
    @TODO move to augmentation?
    :param x:
    :param degree:
    :return:
    """
    x_dropped, cols, old_new = drop_999(x)
    x_stuff = build_poly(angles_extension_22(x_dropped, cols, old_new), degree)
    x_plus = gaussian_distance_22(x_stuff, cols, old_new)
    return x_plus


def cross_validation_22(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    x_te = x[te_indice]
    x_tr = x[tr_indice]
    y_te = y[te_indice]
    y_tr = y[tr_indice]

    weights = make_22_weights(y_tr, x_tr, lambda_, degree)
    x_te_pred = predict_22_stuff(weights, x_te, degree)

    # @TODO check that my modification did not fuck up everything
    acc = label_accuracy(x_te_pred, y_te)

    # print(type(y_tr))
    # print(type(x_tr))
    # loss_tr = np.sqrt(2 * compute_mse(y_tr, x_tr, weights))
    # loss_te = np.sqrt(2 * compute_mse(y_te, x_te, weights))
    return acc


def k_folder_acc_22(k_fold, y, x, k_indices, lamb, degree):
    """
    @TODO refactor & document
    :param k_fold:
    :param y:
    :param x:
    :param k_indices:
    :param lamb:
    :param degree:
    :return:
    """
    acc_tmp = []
    for k in range(k_fold):
        acc = cross_validation_22(y, x, k_indices, k, lamb, degree)  ## 22 added here !
        acc_tmp.append(acc)
    return np.mean(acc_tmp)


def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
    @TODO document
    :param y:
    :param k_fold:
    :param seed:
    :return:
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def best_degree_lambda_acc_22(degrees, lambdas, x, y, k_fold):
    """
    @TODO document
    :param degrees:
    :param lambdas:
    :param x:
    :param y:
    :param k_fold:
    :return:
    """
    indices = build_k_indices(y, k_fold, 100)

    b_lambdas = []
    b_rmses = []
    for degree in degrees:
        rmse_te = []

        x_train = x  # we can build the model elsewhere => specifial for each
        # x_train = gaussian_distance(build_poly(mean_replacement(drop_22(featurize_angles(x))),degree))

        for lamb in lambdas:
            mean_k_fold = k_folder_acc_22(k_fold, y, x_train, indices, lamb, degree)
            rmse_te.append(mean_k_fold)
            print(degree, lamb, mean_k_fold)
        print(degree)

        ind_lambda_opt = np.argmax(rmse_te)
        b_lambdas.append(lambdas[ind_lambda_opt])
        b_rmses.append(rmse_te[ind_lambda_opt])

    ind_best_deg = np.argmax(b_rmses)
    return degrees[ind_best_deg], b_lambdas[ind_best_deg]  # not sure yet for lambda


# best_degree_lambda_acc_22(np.arange(7,10),np.logspace(-5, -3, 10),x_train,y_train,4)

