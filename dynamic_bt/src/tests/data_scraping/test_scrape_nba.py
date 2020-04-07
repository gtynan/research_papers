import pandas as pd
import numpy as np

from dynamic_bt.data_scraping.scrape_nba import get_season_data


def test_get_season_data(context):
    parameters = context.params['nba_season_df']

    season_df = get_season_data(parameters)

    # no null values
    assert not season_df.isnull().values.any()
    assert len(season_df) == parameters['n_matches']

    expected_columns = [
        parameters['nba_home_col'],
        parameters['nba_away_col'],
        parameters['home_win'],
        parameters['away_win']
    ]

    assert all(np.isin(expected_columns, season_df.columns))
