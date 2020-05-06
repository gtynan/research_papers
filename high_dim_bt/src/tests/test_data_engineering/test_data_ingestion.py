import pandas as pd

import pandas.api.types as ptypes

from high_dim_bt.data_engineering.data_ingestion import get_tennis_data


def test_get_tennis_data(context):
    data = get_tennis_data(
        url=context.params['tennis_data_url'],
        start_year=2005, end_year=2006,
        year_const=context.params['tennis_data_year_const'])

    assert data.loc[0, context.params['date_col']].year == 2005
    # seems to struggle loc[-1, "Date"] for some reason
    assert data.iloc[-1][context.params['date_col']].year == 2006
