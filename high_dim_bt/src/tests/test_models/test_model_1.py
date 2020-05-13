import pytest
import numpy as np
import pandas as pd

from high_dim_bt.models.model_1 import Model1
from high_dim_bt.nodes.data_engineering import get_model_input, get_starting_abilities


class TestModel1:

    @pytest.fixture(scope='class')
    def model_data(self, context, c_data):
        X, y, players = get_model_input(
            data=c_data,
            winner_col=context.params['winner_col'],
            loser_col=context.params['loser_col'],
            keep_cols=context.params['model_1_keep_cols'])

        s_abilities = get_starting_abilities(
            players=players,
            data=c_data,
            winner_col=context.params['winner_col'],
            winner_pts=context.params['winner_pts'],
            loser_col=context.params['loser_col'],
            loser_pts=context.params['loser_pts'])

        return X, y, s_abilities

    def test_alpha(self, context, model_data):
        X, y, s_abilities = model_data

        mod = Model1()

        with pytest.raises(Exception):
            mod.alpha

        mod.fit(X, y, context.params['date_col'], s_abilities)

        assert isinstance(mod.alpha, tuple)
        assert isinstance(mod.alpha[0], float)
        assert isinstance(mod.alpha[1], float)

    def test_tau_b(self, context, model_data):
        X, y, s_abilities = model_data

        mod = Model1()

        with pytest.raises(Exception):
            mod.tau_b

        mod.fit(X, y, context.params['date_col'], s_abilities)

        assert isinstance(mod.tau_b, tuple)
        assert isinstance(mod.tau_b[0], float)
        assert isinstance(mod.tau_b[1], float)

    def test_tau_b(self, context, model_data):
        X, y, s_abilities = model_data

        mod = Model1()

        with pytest.raises(Exception):
            mod.ln_likelihood

        mod.fit(X, y, context.params['date_col'], s_abilities)

        assert isinstance(mod.ln_likelihood, float)

    def test_calculate_abilities(self, context, model_data):
        X, y, s_abilities = model_data

        X = X.drop(columns=context.params['date_col']).values

        expected_p1_ability = s_abilities[np.argmax(X, axis=1)]
        expected_p2_ability = s_abilities[np.argmin(X, axis=1)]

        p1_ability, p2_ability = Model1._calculate_abilities(X, s_abilities)

        np.testing.assert_array_equal(expected_p1_ability, p1_ability)
        np.testing.assert_array_equal(expected_p2_ability, p2_ability)

    def test_neg_log_likelihood(self, context, model_data):
        X, y, s_abilities = model_data

        # with alpha = 1 and tau = 0 neg_likelihood will just equal
        # -sum(log_prediction_error) as scores are not changed when tau 0.
        # neg likelihood does however transform abilities -> log1p(abilities)*alpha
        # thus account for this when calculating abilities

        p1_ability, p2_ability = Model1._calculate_abilities(
            X=X.drop(columns=context.params['date_col']).values,
            abilities=np.log1p(s_abilities))

        probs = Model1._calculate_probs(p1_ability, p2_ability)
        expected_neg_likelihood = -np.sum(
            Model1._log_prediction_error(y, probs)
        )

        neg_likelihood = Model1._neg_log_likelihood(
            [1, 0], X, y, context.params['date_col'], s_abilities)

        assert neg_likelihood == pytest.approx(
            expected_neg_likelihood)

    def test_update_abilities(self, context, model_data):
        X, y, s_abilities = model_data
        X = X.drop(columns=context.params['date_col']).values

        tau = 0.5

        # cant have player play same game on same day so need to loop through each row as if
        # seperate day
        for i, x in enumerate(X):
            # x is never flat
            x = x.reshape(1, -1)

            p1_ability, p2_ability = Model1._calculate_abilities(
                X=x, abilities=s_abilities)

            probs = Model1._calculate_probs(p1_ability, p2_ability)

            # y an array
            p1_scores, p2_scores = Model1._calculate_score(
                np.array([y[i]]), probs)

            # expected new abilities
            s_abilities[np.argmax(x, axis=1)] += p1_scores*tau
            s_abilities[np.argmin(x, axis=1)] += p2_scores*tau

            abilities = Model1._update_abilities(
                x, s_abilities, p1_scores, p2_scores, tau)

            np.testing.assert_array_equal(s_abilities, abilities)
