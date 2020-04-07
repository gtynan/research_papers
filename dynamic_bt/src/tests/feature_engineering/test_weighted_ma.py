import pandas as pd
import numpy as np

from dynamic_bt.feature_engineering.weighted_ma import weighted_moving_average


def test_weighted_moving_average(context):
    results = pd.Series([1, 0, 1, 0])
    weight = .5
    starting_param = .75

    # only goes as far as the second last as WMA returns up to final cell
    # (ie doesnt include outcome of final cell)

    res_1 = starting_param * (1 - weight) ** 0

    res_2 = weight * (results[0] * (1 - weight) **
                      0) + starting_param * (1 - weight) ** 1

    res_3 = weight * (results[1] * (1 - weight) ** 0 + results[0]
                      * (1 - weight) ** 1) + starting_param * (1 - weight) ** 2

    res_4 = weight * (results[2] * (1 - weight) ** 0 + results[1] * (1 - weight) **
                      1 + results[0] * (1 - weight) ** 2) + starting_param * (1 - weight) ** 3

    expected_results = [res_1,
                        res_2,
                        res_3,
                        res_4]

    assert expected_results == weighted_moving_average(
        weight, starting_param, results)
