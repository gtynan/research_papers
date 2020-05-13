import pytest
import pandas as pd
import numpy as np

from high_dim_bt.nodes.data_engineering import get_tennis_data, clean_data, get_model_input, get_starting_abilities


def test_get_tennis_data(context):
    data = get_tennis_data(
        url=context.params['tennis_data_url'],
        start_year=2005, end_year=2006,
        year_const=context.params['tennis_data_year_const'])

    assert data.loc[0, context.params['date_col']].year == 2005
    assert data.iloc[-1][context.params['date_col']].year == 2006


def test_clean_data(context, dummy_data):
    # columns where row dropped if nan
    nan_cols = [context.params['winner_pts'], context.params['loser_pts']]

    # cleaned data
    c_data = clean_data(
        data=dummy_data,
        winner_col=context.params['winner_col'],
        loser_col=context.params['loser_col'],
        drop_nan_cols=nan_cols,
        min_matches=2)

    players = np.unique(
        c_data
        [[context.params['winner_col'],
          context.params['loser_col']]].values.ravel())

    # Dick should be dropped as only played once, unique automatically sorts players
    np.testing.assert_array_equal(players, ["Harry", "Tom"])

    # row 5 has nan in loser points shoul be removed
    assert c_data[nan_cols].isnull().values.any() == False

    # index should be sequential 0 -> len(data)-1
    assert c_data.index[0] == 0
    assert c_data.index[-1] == len(c_data)-1


def test_get_model_input(context, dummy_data):
    X, y, _ = get_model_input(
        data=dummy_data,
        winner_col=context.params['winner_col'],
        loser_col=context.params['loser_col'],
        keep_cols=[context.params['date_col']])

    # output index should be same as input index
    assert list(X.index) == list(y.index) == list(dummy_data.index)

    # X and y same length
    assert len(X) == len(y) == len(dummy_data)

    # 1 for player_1 should be offset by -1 for player_2, ignoring date col
    assert all(X.drop(columns=context.params['date_col']).sum(axis=1)) == 0

    # player names should be column names
    assert all(dummy_data[context.params['winner_col']].isin(X.columns))
    assert all(dummy_data[context.params['loser_col']].isin(X.columns))

    # y should be all 1's
    assert y.sum() == len(y)


def test_get_starting_abilities(context, dummy_data):
    abilites = get_starting_abilities(
        players=["Tom", "Harry"],
        data=dummy_data,
        winner_col=context.params['winner_col'],
        winner_pts=context.params['winner_pts'],
        loser_col=context.params['loser_col'],
        loser_pts=context.params['loser_pts'])

    np.testing.assert_array_equal(abilites, np.array([100, 10]))
