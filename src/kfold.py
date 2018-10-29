import numpy as np
from src.helpers import label_accuracy
from src.run_best_model import make_22_weights, predict_22


def cross_validation_22(y, x, k_indices, k, lambda_, degree):
    """
    :param k_indices: Indices of Data on which cross validation is performed
    :param y: Label vector
    :param x: Feature Matrix
    :param k: round number of the k fold
    :param lambda_: Penalisation factor for ridge regression
    :param degree: Maximun degree for build_poly
    :return: k'th round of k-fold for given parameters
    """
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    x_te = x[te_indice]
    x_tr = x[tr_indice]
    y_te = y[te_indice]
    y_tr = y[tr_indice]

    weights = make_22_weights(y_tr, x_tr, lambda_, degree)
    x_te_pred = predict_22(weights, x_te, degree)

    acc = label_accuracy(x_te_pred, y_te)

    return acc


def k_folder_acc_22(k_fold, y, x, k_indices, lamb, degree):
    """
    :param k_fold: number of time the training operation is done
    :param y: Label vector
    :param x: Feature Matrix
    :param k_indices: Indices of Data for cross validation
    :param lamb: Penalisation factor for ridge regression
    :param degree: Maximum degree for build_poly
    :return: Average of k-fold performed for given parameters
    """
    acc_tmp = []
    for k in range(k_fold):
        acc = cross_validation_22(y, x, k_indices, k, lamb, degree)
        acc_tmp.append(acc)
    return np.mean(acc_tmp)


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.

    :param y: Label vector
    :param k_fold: number of time the training operation is done
    :param seed: random seed
    :return: indices used for k-fold operation
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

    :param degrees: range of degree on which build_poly will be performed
    :param lambdas: range of penalization term tried for ridge regression
    :param x: Feature Matrix
    :param y: Label Vector
    :param k_fold: number of time the training operation is done
    :return: best parameters of build poly and lambda in ridge regression in given range
    """
    indices = build_k_indices(y, k_fold, 100)

    b_lambdas = []
    b_accs = []
    for degree in degrees:
        acc_te = []

        x_train = x  # model is feature expanded in the make_22_weights and predict_22
        print(degree)
        for lamb in lambdas:
            mean_k_fold = k_folder_acc_22(k_fold, y, x_train, indices, lamb, degree)
            acc_te.append(mean_k_fold)
            print(degree, lamb, mean_k_fold)

        ind_lambda_opt = np.argmax(acc_te)
        b_lambdas.append(lambdas[ind_lambda_opt])
        b_accs.append(acc_te[ind_lambda_opt])

    ind_best_deg = np.argmax(b_accs)
    return degrees[ind_best_deg], b_lambdas[ind_best_deg]

# best_degree_lambda_acc_22(np.arange(7,10),np.logspace(-5, -3, 10),x_train,y_train,4)
# Line above was used to test for best parameters, yielding our best parameter choise
