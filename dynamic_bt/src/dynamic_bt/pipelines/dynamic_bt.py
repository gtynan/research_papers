from ..data_scraping.scrape_nba import get_season_data

from kedro.pipeline import Pipeline, node
from ..nodes.dynamic_bt import (
    append_weighted_ma,
    get_X_y,
    fit_model,
    get_parameters,
)


"""
Pipeline implementation of: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9876.2012.01046.x
"""

nba_season_df_params = "params:nba_season_df"
nba_dynamic_bt_params = "params:nba_dynamic_bt"


def create_pipeline(**kwargs):

    scrape_season_data = node(func=get_season_data,
                              inputs=[nba_season_df_params],
                              outputs="nba_season_data")

    add_moving_average = node(
        func=append_weighted_ma,
        inputs=["nba_season_data", nba_dynamic_bt_params,
                nba_season_df_params, ],
        outputs="nba_season_with_ma")

    get_model_input = node(func=get_X_y,
                           inputs=["nba_season_with_ma", nba_season_df_params],
                           outputs=["nba_season_X", "nba_season_y"])

    train_model = node(func=fit_model,
                       inputs=["nba_season_X", "nba_season_y"],
                       outputs="dynamic_model")

    get_coefficients = node(func=get_parameters,
                            inputs="dynamic_model",
                            outputs="nba_season_results")

    return Pipeline(
        [
            # scrape_season_data,  # uncomment if `nba_season_data` not in data/03_primary
            add_moving_average,
            get_model_input,
            train_model,
            get_coefficients,
        ]
    )
