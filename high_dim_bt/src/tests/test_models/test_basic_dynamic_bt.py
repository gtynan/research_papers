import pytest
import numpy as np

from high_dim_bt.models.basic_dynamic_bt import BasicDynamicModel


class TestBasicDynamicModel:

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

    def test_calculate_probs(self, data):
        abilities, X, y = data

        expected_probs = np.array([
            [0.5, 0.73105857863],
            [0.37754066879, 0.37754066879]
        ])

        for t in range(len(X)):
            probs = BasicDynamicModel._calculate_probs(X[t], abilities)

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
            probs = BasicDynamicModel._calculate_probs(X[t], abilities)
            log_errors = BasicDynamicModel._log_prediction_error(y[t], probs)

            np.testing.assert_array_almost_equal(
                log_errors, expected_log_errors[t], decimal=5)

    def test_calculate_score(self, data):
        abilities, X, y = data

        expected_p1_scores = np.array(
            [1 * (1 - .5), 1 * (1 - 0.73105857863)],
            [-1 * 0.37754066879, 1 * (1 - 0.37754066879)]
        ])

        for t in range(len(X)):
            probs = BasicDynamicModel._calculate_probs(X[t], abilities)
            p1_scores, p2_scores = BasicDynamicModel._calculate_score(
                y[t], probs)

            np.testing.assert_array_almost_equal(
                p1_scores, expected_p1_scores[t], decimal = 5)

            np.testing.assert_array_almost_equal(
                p2_scores, -expected_p1_scores[t], decimal = 5)
