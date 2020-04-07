from typing import Optional, Dict, Any
import pandas as pd
import lxml.html as lh
import requests
import calendar
import datetime


def get_season_data(nba_df_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Retrieves a dataframe with NBA season results for the specified year
    Args:
        year: year of interest
        home_team_col: column in table relating to home team
        away_team_col: column in table relating to away team
        n_matches: number of regular season matches in said season
    Returns:
        resuts: sorted dataframe with n_matches rows with each row relating to a game
    """
    season_df: pd.DataFrame = None

    # October -> April
    for month in [m.lower()
                  for m in calendar.month_name[10:] + calendar.month_name
                  [: 5]]:

        results_page = requests.get(
            f"https://www.basketball-reference.com/leagues/NBA_{nba_df_params['year_of_interest']}_games-{month}.html"
        )

        # if no result
        if results_page.status_code != 200:
            continue

        # website contents
        doc = lh.fromstring(results_page.content)

        table_rows = doc.xpath("//tr")

        # headers from table
        columns = [element.text_content()
                   for element in table_rows[0].iterchildren()]
        # point columns were blank so renaming them
        columns[columns.index(nba_df_params["nba_home_col"]) + 1] = "HomePTS"
        columns[columns.index(nba_df_params["nba_away_col"]) + 1] = "AwayPTS"

        # all row data
        data = []

        # getting all results from table
        for row in table_rows[1:]:
            row_data = [data.text_content() for data in row.iterchildren()]
            data.append(row_data)

        # all results relating to said month
        month_data = pd.DataFrame(columns=columns, data=data)

        if season_df is not None:
            season_df = season_df.append(month_data, ignore_index=True)
        else:
            season_df = month_data

    # Playoff is not a date
    season_df = season_df[season_df["Date"] != "Playoffs"]

    # dtypes
    season_df["Date"] = pd.to_datetime(season_df["Date"])
    season_df[["HomePTS", "AwayPTS"]] = season_df[[
        "HomePTS", "AwayPTS"]].astype(int)

    # converting to time so can sort later
    season_df["Start (ET)"] = season_df["Start (ET)"].apply(
        lambda row: datetime.timedelta(
            hours=int(row[: -1].split(":")[0]),
            minutes=int(row[: -1].split(":")[1])))

    # adding outcome
    season_df[nba_df_params["home_win"]] = (
        season_df["HomePTS"] > season_df["AwayPTS"]).astype(int)
    season_df[nba_df_params["away_win"]
              ] = 1 - season_df[nba_df_params["home_win"]]

    # remove null columns
    season_df = season_df.dropna(axis=1)

    return season_df.sort_values(
        by=["Date", "Start (ET)",
            nba_df_params["nba_home_col"]]).iloc[:
                                                 nba_df_params["n_matches"]].reset_index(drop=True)
