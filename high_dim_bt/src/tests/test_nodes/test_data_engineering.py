import pytest
import pandas as pd
import numpy as np

from high_dim_bt.nodes.data_engineering import get_tennis_data, clean_data, get_model_input


@pytest.fixture(scope='module')
def dummy_data(context):
    data = {context.params['winner_col']: ["Tom", "Dick", "Tom", "Harry", "Tom"],
            context.params['loser_col']: ['Harry', 'Tom', 'Harry', 'Tom', 'Harry'],
            context.params['date_col']: ['01/01/2020', '01/01/2020', '02/01/2020', '02/01/2020', '03/01/2020'],
            context.params['winner_pts']: [100, 50, 200, 200, pd.NaT],
            context.params['loser_pts']: [10, 20, 30, 10, 400]}

    return pd.DataFrame.from_dict(data)


def test_get_tennis_data(context):
    data = get_tennis_data(
        url=context.params['tennis_data_url'],
        start_year=2005, end_year=2006,
        year_const=context.params['tennis_data_year_const'])

    assert data.loc[0, context.params['date_col']].year == 2005
    # seems to struggle loc[-1, "Date"] for some reason
    assert data.iloc[-1][context.params['date_col']].year == 2006


def test_clean_data(context, dummy_data):
    # columns where row dropped if nan
    nan_cols = [context.params['winner_pts'], context.params['loser_pts']]

    cleaned_data = clean_data(
        data=dummy_data,
        winner_col=context.params['winner_col'],
        loser_col=context.params['loser_col'],
        drop_nan_cols=nan_cols,
        min_matches=2)

    players = np.unique(
        cleaned_data
        [[context.params['winner_col'],
          context.params['loser_col']]].values.ravel())

    # Dick should be dropped as only played once
    assert all(np.isin(players, ["Tom", "Harry"])) == True
    # row 5 has nan in loser points shoul be removed
    assert cleaned_data[nan_cols].isnull().values.any() == False


def test_get_model_input(context, dummy_data):

    # messy index to ensure able to handle
    df = dummy_data
    df.index = [2, 6, 10, 11, 12]

    # no base case
    X, y = get_model_input(
        df, context.params['winner_col'],
        context.params['loser_col'],
        keep_cols=[context.params['date_col']])

    # output index should be same as input index
    assert list(X.index) == list(y.index) == list(df.index)

    # X and y same length
    assert len(X) == len(y) == 5

    # 1 for player_1 should be offset by -1 for player_2
    assert all(X.sum(axis=1)) == 0

    # player names should be column names
    assert all(df[context.params['winner_col']].isin(X.columns))

    # y should not be all 1's
    assert y.sum() == len(y) - len(y)//2

    # checking X and y correctly correspond to original data
    for i, winner in df[context.params['winner_col']].items():
        # either 1 or -1
        winner_value = X.loc[i, winner]
        # either 0 or 1
        outcome_value = y[i]

        if winner_value < 0:
            assert outcome_value == 0
        else:
            assert outcome_value == 1
