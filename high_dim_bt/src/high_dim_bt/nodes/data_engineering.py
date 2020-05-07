from typing import List
import pandas as pd
import numpy as np


def get_tennis_data(url: str, start_year: int, end_year: int,
                    year_const: str) -> pd.DataFrame:
    '''
    Creates master dataframe of all games within years specified

    Args:
        url: url to excel spreadsheet
        start_year: first year to be included
        end_year: last year to be included
        year_const: string constant to be replaced by actual year ie YEAR (www.xyz.com/YEAR/YEAR.xls)

    Returns:
        Dataframe of all matches played between date range
    '''
    df: pd.DataFrame = None

    for i, year in enumerate(range(start_year, end_year + 1)):

        data_url = url.replace(year_const, str(year))
        data = pd.read_excel(data_url)

        if i == 0:
            df = data
        else:
            df = pd.concat([df, data], sort=False, ignore_index=True)

    return df


def clean_data(
        data: pd.DataFrame, winner_col: str, loser_col: str, min_matches: int,
        drop_nan_cols: List[str]) -> pd.DataFrame:
    '''
    Args:
        data: unclean dataframe downloaded from tennis data
        winner_col: column containing winner 
        loser_col: column containing loser
        min_matches: min matches competed in otherwise dropped
        drop_nan_cols: all rows with nans in these columns will be dropped

    Returns:
        Cleaned data
    '''

    players, counts = np.unique(
        data[[winner_col, loser_col]].values.ravel(),
        return_counts=True)

    # only those above min matches
    keep_players = players[counts >= min_matches]

    data = data[(data[winner_col].isin(keep_players)) &
                (data[loser_col].isin(keep_players))]

    data = data.dropna(subset=drop_nan_cols)
    return data
