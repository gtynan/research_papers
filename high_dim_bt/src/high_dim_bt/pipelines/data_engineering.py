from typing import Dict

from kedro.pipeline import Pipeline, node

from ..nodes.data_engineering import get_tennis_data, clean_data, get_model_input, get_starting_abilities

from pathlib import Path


def create_pipeline(**kwargs) -> Dict[str, Pipeline]:

    data_ingestion = node(
        func=get_tennis_data,
        inputs=dict(
            url='params:tennis_data_url',
            start_year='params:tennis_data_start_year',
            end_year='params:tennis_data_end_year',
            year_const='params:tennis_data_year_const'),
        outputs='master_data')

    cleaning_data = node(
        func=clean_data,
        inputs=dict(
            data='master_data',
            winner_col='params:winner_col',
            loser_col='params:loser_col',
            min_matches='params:min_matches',
            drop_nan_cols='params:drop_na_cols'),
        outputs='cleaned_data')

    model_input = node(
        func=get_model_input,
        inputs=dict(
            data='cleaned_data',
            winner_col='params:winner_col',
            loser_col='params:loser_col',
            keep_cols='params:date_col'),
        outputs=['X', 'y', 'players'])

    starting_abilities = node(
        func=get_starting_abilities,
        inputs=dict(
            players='players',
            data='cleaned_data',
            winner_col='params:winner_col',
            winner_pts='params:winner_pts',
            loser_col='params:loser_col',
            loser_pts='params:loser_pts'),
        outputs='starting_abilities')

    return Pipeline(
        [
            data_ingestion,
            cleaning_data,
            model_input,
            starting_abilities
        ]
    )
