import unittest

import numpy as np

from src.helpers import load_csv_data, predict_labels, label_accuracy
from src.implementations import least_squares_GD, least_squares_SGD, least_squares, logistic_regression, \
    reg_logistic_regression, ridge_regression

DATA_FOLDER = './data/'

y, tx, _ = load_csv_data(DATA_FOLDER + "train.csv")
y = y.reshape((len(y), 1))


class TestImplementations(unittest.TestCase):

    def setUp(self):
        self.initial_w = np.zeros((tx.shape[1], 1))
        self.max_iters = 100
        self.gamma = 0.
        self.lambda_ = 0.

    def assert_result(self, name, w, loss):
        pred = predict_labels(w, tx)
        acc = label_accuracy(pred, y)
        self.assertEqual(w.shape[0], self.initial_w.shape[0])
        self.assertEqual(w.shape[1], self.initial_w.shape[1])
        self.assertGreater(acc, 60, msg="Accuracy suspicious for " + name)
        self.assertEqual(loss.shape, (1, 1))
        self.assertGreaterEqual(loss[0, 0], 0., msg="Loss should be positive (" + str(loss) + ")")

    def test_least_squares_gd(self):
        w, loss = least_squares_GD(y, tx, self.initial_w, self.max_iters, self.gamma)
        self.assert_result('least_squares_gd', w, loss)

    def test_least_squares_sgd(self):
        w, loss = least_squares_SGD(y, tx, self.initial_w, self.max_iters, self.gamma)
        self.assert_result('least_squares_sgc', w, loss)

    def test_least_squares(self):
        w, loss = least_squares(y, tx)
        self.assert_result('least_squares', w, loss)

    def test_ridge_regression(self):
        w, loss = ridge_regression(y, tx, self.lambda_)
        self.assert_result('ridge', w, loss)

    def test_logistic_regression(self):
        w, loss = logistic_regression(y, tx, self.initial_w, self.max_iters, self.gamma)
        self.assert_result('logistic', w, loss)

    def test_reg_logistic_regression(self):
        w, loss = reg_logistic_regression(y, tx, self.lambda_, self.initial_w, self.max_iters, self.gamma)
        self.assert_result('reg_logistic', w, loss)
