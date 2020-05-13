import pytest
import pandas as pd
from kedro.context import load_context
from pathlib import Path

from high_dim_bt.nodes.data_engineering import clean_data


# Kedro context to access params
@pytest.fixture(scope='session')
def context():
    return load_context(Path.cwd())


@pytest.fixture(scope='session')
def dummy_data(context):
    '''
    Made up data to test against
    '''
    data = {context.params['winner_col']: ["Tom", "Dick", "Tom", "Harry", "Tom"],
            context.params['loser_col']: ['Harry', 'Tom', 'Harry', 'Tom', 'Harry'],
            context.params['date_col']: ['01/01/2020', '01/01/2020', '02/01/2020', '02/01/2020', '02/01/2020'],
            context.params['winner_pts']: [1, 5, 2, 1, pd.NaT],
            context.params['loser_pts']: [.1, 2, 1, 3, 4]}

    return pd.DataFrame.from_dict(data)


@pytest.fixture(scope='session')
def c_data(context, dummy_data):
    '''
    Clean data is the same for all 3 models

    Returns:
        3 games:
            W-Tom, L-Harry, Date-01
            W-Tom, L-Harry, Date-02
            W-Harry, L-Tom, Date-02
    '''
    c_data = clean_data(
        data=dummy_data,
        winner_col=context.params['winner_col'],
        loser_col=context.params['loser_col'],
        drop_nan_cols=context.params['drop_na_cols'],
        min_matches=2)

    return c_data
