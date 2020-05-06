from kedro.pipeline import Pipeline, node

from ..nodes.data_engineering import get_tennis_data

from pathlib import Path


def create_pipeline(**kwargs):
    data_ingestion = node(
        func=get_tennis_data,
        inputs=dict(
            url='params:tennis_data_url',
            start_year='params:tennis_data_start_year',
            end_year='params:tennis_data_end_year',
            year_const='params:tennis_data_year_const'),
        outputs='master_data')
    return Pipeline(
        [
            data_ingestion
        ]
    )
