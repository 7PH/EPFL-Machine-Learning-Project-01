import numpy as np
from timeit import default_timer as timer

from src.gradients import compute_logistic_gradient
from src.helpers import load_csv_data, create_csv_submission, predict_labels, label_accuracy

# Least squares
from src.implementations import reg_logistic_gradient


def least_squares(tx, y):
    """w = (tx'.tx)-1 . tx'y"""
    txt = tx.T
    a = txt.dot(tx)
    b = txt.dot(y)
    return np.linalg.solve(a, b)


def logistic_gradient_step(y, tx, w, gamma):
    grad = compute_logistic_gradient(y, tx, w0)
    return w - gamma * grad


def reg_logistic_gradient_step(y, tx, w, l, gamma):
    grad = compute_logistic_gradient(y, tx, w0) + 2 * l * w
    return w - gamma * grad



print("Loading data")
y_train, x_train, x_ids = load_csv_data(DATA_FOLDER + "train.csv")
y_test, x_test, x_test_ids = load_csv_data(DATA_FOLDER + "test.csv")

print("Fixing shape of y_train data")
y_train = np.array([y_train]).T

print("Init")
w0 = np.zeros((x_train.shape[1], 1))

print("Starting")
start = timer()
w = reg_logistic_gradient(y_train, x_train, w0, 0.1, 0.1, 100)
print(timer() - start, "seconds to complete")

print("Local label accuracy")
print(label_accuracy(predict_labels(w, x_train), y_train))

print("Making CSV file")
create_csv_submission(x_test_ids, predict_labels(w, x_test), "predictions.csv")

print(w)
