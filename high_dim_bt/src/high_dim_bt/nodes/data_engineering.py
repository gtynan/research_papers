import pandas as pd


def get_tennis_data(url: str, start_year: float, end_year: float,
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


def clean_data(data: pd.DataFrame, min_matches: int):
    pass
