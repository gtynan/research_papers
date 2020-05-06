import pandas as pd

import pandas.api.types as ptypes

from high_dim_bt.data_engineering.data_ingestion import get_tennis_data


def test_get_tennis_data(context):
    data = get_tennis_data(
        url=context.params['tennis_data_url'],
        start_year=2005, end_year=2006,
        date_col=context.params['tennis_data_date_col'],
        year_const=context.params['tennis_data_year_const'])

    # Date format needed as relied upon by other nodes
    assert ptypes.is_datetime64_any_dtype(
        data[context.params['tennis_data_date_col']])

    # Must be sorted by date earlier -> later
    pd.testing.assert_series_equal(
        left=data[context.params['tennis_data_date_col']].sort_values(
            ascending=True).reset_index(drop=True),
        right=data[context.params['tennis_data_date_col']])

    assert data.loc[0, context.params['tennis_data_date_col']].year == 2005
    # seems to struggle loc[-1, "Date"] for some reason
    assert data.iloc[-1][context.params['tennis_data_date_col']].year == 2006
