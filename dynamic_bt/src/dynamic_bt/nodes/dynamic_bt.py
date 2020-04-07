from typing import Tuple, Dict, Any
import pandas as pd
from statsmodels.discrete.discrete_model import Logit, BinaryResultsWrapper

from ..feature_engineering.weighted_ma import weighted_moving_average

"""
All nodes needed to run the dynamic_bt pipeline
"""


def append_weighted_ma(df: pd.DataFrame,
                       dynamic_bt_params: Dict[str, float],
                       nba_df_params: Dict[str, Any]) -> pd.DataFrame:
    """Appends weighted moving average to dataframe 

    Args:
        df: Dataframe to add weighted ma
        dynamic_bt_params: parameters related to dynamic bradley terry (located in conf/base/parameters)
        nba_df_params: parameters related to the NBA dataframe (located in conf/base/parameters)

    Returns:
        dataframe with weight MA appended to the end under `nba_df_params["home_ability"]`, `nba_df_params["away_ability"]`
    """

    df[nba_df_params["home_ability"]] = df.groupby(
        nba_df_params["nba_home_col"])[nba_df_params["home_win"]].transform(
        lambda
        results:
        weighted_moving_average(
            dynamic_bt_params["home_smoother"],
            dynamic_bt_params["starting_home_ability"],
            results,))

    df[nba_df_params["away_ability"]] = df.groupby(
        nba_df_params["nba_away_col"])[nba_df_params["away_win"]].transform(
        lambda
        results:
        weighted_moving_average(
            dynamic_bt_params["away_smoother"],
            dynamic_bt_params["starting_away_ability"],
            results,))
    return df


def get_X_y(df: pd.DataFrame,
            nba_df_params: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Creates X and y needed to fit dynamicBT

    Args:
        df: NBA season df with weight ma appended
        nba_df_params: parameters related to the NBA dataframe (located in conf/base/parameters)

    Returns:
        X and y data
    """
    X = df[[nba_df_params["home_ability"], nba_df_params["away_ability"]]]
    y = df[nba_df_params["home_win"]]

    return X, y


def fit_model(X: pd.DataFrame,
              y: pd.Series) -> BinaryResultsWrapper:
    """Fits and returns dynamicBt model

    Args:
        X: predictor variables
        y: response variable

    Reurns:
        Results wrapper
    """
    model = Logit(y, X).fit(method="newton")
    return model


def get_parameters(model: Logit) -> pd.Series:
    """
    Returns coeficients for fitted Bradley Terry (statsmodels Logit)
    """
    return model.params
