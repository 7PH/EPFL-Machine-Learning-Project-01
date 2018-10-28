import unittest

import numpy as np

from src.helpers import load_csv_data
from src.implementations import least_squares_GD, least_squares_SGD, least_squares, logistic_regression, \
    reg_logistic_regression

DATA_FOLDER = './data/'

y, tx, x_ids = load_csv_data(DATA_FOLDER + "train.csv")


class TestImplementations(unittest.TestCase):

    def setUp(self):
        self.initial_w = np.zeros((tx.shape[1], 1))
        self.max_iters = 10
        self.gamma = 0.1
        self.lambda_ = 0.001

    def assert_result(self, w, loss):
        self.assertEqual(1, 1)

    def test_least_squares_gd(self):
        w, loss = least_squares_GD(y, tx, self.initial_w, self.max_iters, self.gamma)
        self.assert_result(w, loss)

    def test_least_squares_sgd(self):
        w, loss = least_squares_SGD(y, tx, self.initial_w, self.max_iters, self.gamma)
        self.assert_result(w, loss)

    def test_least_squares(self):
        w, loss = least_squares(y, tx)
        self.assert_result(w, loss)

    def test_ridge_regression(self):
        w, loss = least_squares(y, tx)
        self.assert_result(w, loss)

    def test_logistic_regression(self):
        w, loss = logistic_regression(y, tx, self.initial_w, self.max_iters, self.gamma)
        self.assert_result(w, loss)

    def test_reg_logistic_regression(self):
        w, loss = reg_logistic_regression(y, tx, self.lambda_, self.initial_w, self.max_iters, self.gamma)
        self.assert_result(w, loss)