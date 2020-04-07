import pytest
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit, BinaryResultsWrapper


from dynamic_bt.nodes.dynamic_bt import append_weighted_ma, get_X_y, fit_model, get_parameters


@pytest.fixture(scope='module')
def x_y_data(context):
    """
    Loads NBA X and y data to fit model
    """
    return context.io.load('nba_season_X'), context.io.load('nba_season_y')


def test_append_weighted_ma(context):

    bt_params = context.params['nba_dynamic_bt']
    nba_params = context.params['nba_season_df']

    df = append_weighted_ma(context.io.load('nba_season_data'),
                            bt_params,
                            nba_params)

    # relevant columns added
    assert all(
        np.isin(
            [nba_params['home_ability'],
             nba_params['away_ability']],
            df.columns))

    # starts with starting abilities
    assert df.loc[0, nba_params['home_ability']
                  ] == bt_params['starting_home_ability']
    assert df.loc[0, nba_params['away_ability']
                  ] == bt_params['starting_away_ability']

    # no nulls
    assert not df[[nba_params['home_ability'],
                   nba_params['away_ability']]].isnull().values.any()


def test_get_X_y(context):

    nba_params = context.params['nba_season_df']

    X, y = get_X_y(context.io.load('nba_season_with_ma'), nba_params)

    # must be same length
    assert len(X) == len(y)

    # no nulls
    assert not (X.isnull().values.any()) or (y.isnull().values.any())

    # assert X and y relate to the correct columns
    assert y.name == nba_params['home_win']
    assert all(
        np.isin(
            [nba_params['home_ability'],
             nba_params['away_ability']],
            X.columns))


def test_fit_model(x_y_data):
    model_results = fit_model(*x_y_data)

    assert isinstance(model_results, BinaryResultsWrapper)


def test_get_parameters(context, x_y_data):
    model = fit_model(*x_y_data)

    nba_params = context.params['nba_season_df']

    model_params = get_parameters(model)

    assert len(model_params == 2)
    assert all(
        np.isin(
            [nba_params['home_ability'],
             nba_params['away_ability']],
            model_params.index))

    # check params ~match to paper results
    assert model_params[nba_params['home_ability']
                        ] == pytest.approx(5.503, 0.01)
    assert model_params[nba_params['away_ability']
                        ] == pytest.approx(-7.379, 0.01)
