import pytest
import numpy as np

from high_dim_bt.models.abstract_high_dm import AbstractHighDimensionalModel
from high_dim_bt.nodes.data_engineering import get_model_input, get_starting_abilities


class TestAbstractHighDimensionalModel:

    @pytest.fixture(scope='class')
    def model_data(self, context, c_data):
        X, y, players = get_model_input(
            data=c_data,
            winner_col=context.params['winner_col'],
            loser_col=context.params['loser_col'])

        s_abilities = get_starting_abilities(
            players=players,
            data=c_data,
            winner_col=context.params['winner_col'],
            winner_pts=context.params['winner_pts'],
            loser_col=context.params['loser_col'],
            loser_pts=context.params['loser_pts'])

        return X, y, s_abilities

    @pytest.fixture(scope='class')
    def probs(self, model_data):
        '''
        Probability of p1 beating p2 in each matchup, 
        used by all 3 tests so fixture
        '''
        X, y, s_abilities = model_data

        p1_ability = s_abilities[np.argmax(X.values, axis=1)]
        p2_ability = s_abilities[np.argmin(X.values, axis=1)]

        probs = AbstractHighDimensionalModel._calculate_probs(
            p1_ability, p2_ability)

        return probs

    def test_calculate_probs(self, probs):
        # starting values for each players Pts is the values in the exponentials
        # (P1 start pts - P2 start pts)
        expected_probs = np.array([np.exp(1-.1) / (1 + np.exp(1-.1)),
                                   np.exp(1-.1) / (1 + np.exp(1-.1)),
                                   np.exp(.1-1) / (1 + np.exp(.1-1))])

        np.testing.assert_array_equal(probs, expected_probs)

    def test_log_prediction_error(self, model_data, probs):
        X, y, s_abilities = model_data

        # ln(probs) when win and because y always = 1 always ln(probs)
        expected_log_errors = np.log(probs)

        log_errors = AbstractHighDimensionalModel._log_prediction_error(
            y, probs)

        np.testing.assert_array_equal(
            log_errors, expected_log_errors)

    def test_calculate_score(self, context, model_data, probs):
        X, y, s_abilities = model_data

        # because all y = 1 (due to our get_model_input fucntion), 1 - probs = score
        # as when y = 1, score = y*(1-probs) for player_1
        expected_p1_score = 1 - probs

        p1_score, p2_score = AbstractHighDimensionalModel._calculate_score(
            y, probs)

        np.testing.assert_array_equal(expected_p1_score, p1_score)
        # p2 score is just the negative of p1
        np.testing.assert_array_equal(-expected_p1_score, p2_score)
