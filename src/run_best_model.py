from src.helpers import *
from src.implementations import ridge_regression
from src.augmentation import features_expansion

# Constants
DATA_FOLDER = 'data/'


def make_22_weights(y_tr, x, lambda_, degree):
    """
    This function splits the set x
    according the the values of the 22th column, and run a ridge regression
    using the expended sub-set, and return 3 weight's vectors

    :param y_tr: Lables of the training dataset
    :param x: Feature Matrix
    :param lambda_: Penalisation term for ridge regression
    :param degree: degree of polynomail augmentation

    :return: 3 weight vectors, one for each subset
    """
    weights = {}

    #Jet splitting (creating a mask)

    jet_masks = [
        x[:, 22] == 0,
        x[:, 22] == 1,
        x[:, 22] > 1]

    ys_train = [y_tr[mask] for mask in jet_masks]
    xs = [x[mask] for mask in jet_masks]
    x_p_tr = {}

    for i, mask in enumerate(jet_masks):
        if i == 0:
            x_p_tr[i] = features_expansion(xs[i], degree)
        if i == 1:
            x_p_tr[i] = features_expansion(xs[i], degree)
        if i > 1:
            x_p_tr[i] = features_expansion(xs[i], degree)

        weights[i] = ridge_regression(ys_train[i], x_p_tr[i], lambda_)

    return weights


def predict_22(weights, x, degree):
    """

    :param weights: 3 weight's vector, one for each jet
    :param x: Test dataset
    :param degree: Degree of polynomail augmentation

    :return: Labels of prediction
    """
    jet_masks = [
        x[:, 22] == 0,
        x[:, 22] == 1,
        x[:, 22] > 1]

    xs = [x[mask] for mask in jet_masks]

    y_sub = np.zeros(x.shape[0])
    x_p_te = {}
    for i, mask in enumerate(jet_masks):
        if i == 0:
            x_p_te[i] = features_expansion(xs[i], degree)
        if i == 1:
            x_p_te[i] = features_expansion(xs[i], degree)
        if i > 1:
            x_p_te[i] = features_expansion(xs[i], degree)

        y_sub[mask] = predict_labels(weights[i][0], x_p_te[i])

    return y_sub


print("Loading data")
y_train, x_train, x_ids = load_csv_data(DATA_FOLDER + "train.csv")
y_test, x_test, x_test_ids = load_csv_data(DATA_FOLDER + "test.csv")

print("Running")
# Degree and lambda are decided by a grid search for these two parameters. See kfold.py methods at the end of the file
optimal_lambda = 0.0001291549665014884
weights = make_22_weights(y_train, x_train, optimal_lambda, degree=8)
y_predicted = predict_22(weights, x_test, degree=8)

print("Storing prediction")
filename = "submission.csv"
create_csv_submission(x_test_ids, y_predicted, filename)

print("File name: " + filename)
