from typing import Optional, Tuple, List
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


def get_model_input(data: pd.DataFrame, winner_col: str, loser_col: str,
                    keep_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame,
                                                                    pd.Series, np.array]:
    """
    Given row by row match data with winner and loser sepcified returns X and y data for Bardley Terry model

    Args:
        data: dataframe of matches
        winner_col: column containing winner names
        loser_col: column containing lsoer names
        keep_cols: optional list of additional columns to keep in X data

    Reurns:
        X, y and players data input for bardley terry model, y always relates to player_1
    """
    # dont want all 1's for y values so swapping some winners out of player_1 to player_2
    index = range(len(data))
    n_to_swap = len(index)//2

    # we get position as when we deep copy the index is reset, we do this incase dataframe passed is badly indexed.
    # we pass the original index back later
    pos_to_swap = np.random.choice(index,
                                   n_to_swap,
                                   replace=False)

    player_1 = data[winner_col].copy(deep=True)
    player_1.iloc[pos_to_swap] = data[loser_col]

    player_2 = data[loser_col].copy(deep=True)
    player_2.iloc[pos_to_swap] = data[winner_col]

    # index is passed back
    y = pd.Series(player_1 == data[winner_col].values,
                  index=data.index).astype(int)

    # dict of players and their position
    participants = {participant: i for i, participant in enumerate(
        np.unique(
            data[[winner_col, loser_col]].values.ravel()
        )
    )}

    # Matrix of data
    X = np.zeros((len(data), len(participants.keys())))
    X[index, player_1.map(participants)] = 1
    X[index, player_2.map(participants)] = -1

    # index is passed back
    X = pd.DataFrame(data=X, columns=participants.keys(), index=data.index)

    if keep_cols:
        X[keep_cols] = data[keep_cols]

    print(X.shape)
    return X, y, np.array(list(participants.keys()))


def get_starting_abilities(
    players: np.array, data: pd.DataFrame, winner_col: str,
        winner_pts: str, loser_col: str, loser_pts: str) -> np.array:
    '''
    Given list of players return their first occurance points in dataset (used as starting abilites)

    Args:
        players: list of players
        data: dataset containing point information
        winner_col: column containing winner
        winner_pts: column containg winners points
        loser_col: column containing loser
        loser_pts: column conating loser points

    Returns:
        Array of points corresponding to list of players given
    '''
    # winners and losers flattened into long array
    flatten_players = data[[winner_col, loser_col]].values.ravel()

    # only unique players in dataset and their first occurance index
    unique_players, unique_index = np.unique(
        flatten_players, return_index=True)

    # player: 1st index in flatten array
    index_series = pd.Series(index=unique_players, data=unique_index)

    # get 1st occurance for all players we are interested in
    pos = index_series[players]

    # get abilities using their 1st occurance index
    abilities = data[[winner_pts, loser_pts]].values.ravel()[pos]

    # any nan's relpaced with 0
    return np.nan_to_num(abilities, nan=0)
