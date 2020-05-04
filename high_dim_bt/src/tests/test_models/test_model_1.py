import pytest
import numpy as np
import pandas as pd

from high_dim_bt.models.model_1 import Model1


class TestModel1:

    @pytest.fixture(scope='class')
    def data(self):
        # (p, )
        abilities = np.array([0.5, 0.5, 1, 0])
        # games = 2
        # n x 2 x p
        X = np.array([
            [[1, -1, 0, 0], [0, 0, 1, -1]],
            [[0, 1, -1, 0], [-1, 0, 0, 1]]
        ])
        # n x 2
        y = np.array([
            [1, 1],
            [0, 1]
        ])
        return abilities, X, y

    @pytest.fixture(scope='class')
    def pandas_data(self, data):
        abilities, X, y = data

        # reformatting data to pandas objects
        date_col = 'Date'

        X = pd.DataFrame(X.reshape(-1, 4))
        # creating two different dates
        X.loc[:2, date_col] = '1'
        X.loc[2:, date_col] = '2'

        y = pd.Series(y.flatten())
        return abilities, X, y

    def test_alpha(self, pandas_data):
        abilities, X, y = pandas_data

        mod = Model1()

        with pytest.raises(Exception):
            mod.alpha

        mod.fit(X, y, 'Date', abilities)

        assert isinstance(mod.alpha, float)

    def test_tau(self, pandas_data):
        abilities, X, y = pandas_data

        mod = Model1()

        with pytest.raises(Exception):
            mod.tau

        mod.fit(X, y, 'Date', abilities)

        assert isinstance(mod.tau, float)

    def test_ln_likelihood(self, pandas_data):
        abilities, X, y = pandas_data

        mod = Model1()

        with pytest.raises(Exception):
            mod.ln_likelihood

        mod.fit(X, y, 'Date', abilities)

        assert isinstance(mod.ln_likelihood, float)

    def test_neg_log_likelihood(self, pandas_data):
        abilities, X, y = pandas_data

        negative_likelihood = Model1._neg_log_likelihood(
            [1, 0.5], X, y, 'Date', abilities)

        # TODO double check
        assert negative_likelihood == pytest.approx(2.68678)

    def test_calculate_probs(self, data):
        abilities, X, y = data

        expected_probs = np.array([
            [0.5, 0.73105857863],
            [0.37754066879, 0.37754066879]
        ])

        for t in range(len(X)):
            probs = Model1._calculate_probs(X[t], abilities)

            np.testing.assert_array_almost_equal(
                probs, expected_probs[t], decimal=5)

    def test_log_prediction_error(self, data):
        abilities, X, y = data

        # ln(probs) when win else ln(1 - probs)
        expected_log_errors = np.array([
            [np.log(0.5), np.log(0.73105857863)],
            [np.log(1 - 0.37754066879), np.log(0.37754066879)]
        ])

        for t in range(len(X)):
            probs = Model1._calculate_probs(X[t], abilities)
            log_errors = Model1._log_prediction_error(y[t], probs)

            np.testing.assert_array_almost_equal(
                log_errors, expected_log_errors[t], decimal=5)

    def test_calculate_score(self, data):
        abilities, X, y = data

        expected_p1_scores = np.array([
            [1 * (1 - .5), 1 * (1 - 0.73105857863)],
            [-1 * 0.37754066879, 1 * (1 - 0.37754066879)]
        ])

        for t in range(len(X)):
            probs = Model1._calculate_probs(X[t], abilities)
            p1_scores, p2_scores = Model1._calculate_score(
                y[t], probs)

            np.testing.assert_array_almost_equal(
                p1_scores, expected_p1_scores[t], decimal=5)

            np.testing.assert_array_almost_equal(
                p2_scores, -expected_p1_scores[t], decimal=5)

    def test_update_abilities(self, data):
        abilities, X, y = data
        tau = 0.5

        # current ability + score*tau
        expected_abilities = np.array([
            [0.5 + .5*tau,
             0.5 - .5*tau,
             1 + (1 - 0.73105857863)*tau,
             0 - (1 - 0.73105857863)*tau],
            [0.5 - (1 - 0.37754066879)*tau,
             0.5 - 0.37754066879*tau,
             1+0.37754066879*tau,
             0 + (1 - 0.37754066879)*tau]
        ])

        for t in range(len(X)):
            probs = Model1._calculate_probs(X[t], abilities)
            p1_scores, p2_scores = Model1._calculate_score(
                y[t], probs)

            new_abilities = Model1._update_abilities(
                X[t], abilities, p1_scores, p2_scores, tau)

            # ensure function does not overwrite passed array
            np.testing.assert_raises(
                AssertionError, np.testing.assert_array_almost_equal, new_abilities, abilities)

            np.testing.assert_array_almost_equal(
                new_abilities, expected_abilities[t], decimal=5)
