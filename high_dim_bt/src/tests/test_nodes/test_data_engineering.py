import pytest
import pandas as pd

from high_dim_bt.nodes.data_engineering import get_tennis_data

start_year = 2005
end_year = 2006


@pytest.fixture(scope='module')
def data(context):
    data = get_tennis_data(
        url=context.params['tennis_data_url'],
        start_year=start_year, end_year=end_year,
        year_const=context.params['tennis_data_year_const'])
    return data


def test_get_tennis_data(context, data):
    assert data.loc[0, context.params['date_col']].year == start_year
    # seems to struggle loc[-1, "Date"] for some reason
    assert data.iloc[-1][context.params['date_col']].year == end_year
