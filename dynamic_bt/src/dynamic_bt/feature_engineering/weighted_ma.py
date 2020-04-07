from typing import List
import pandas as pd


def weighted_moving_average(smoothing_param: float,
                            initial_condition: float,
                            results: pd.Series) -> List[float]:
    """
    Weighted MA function based on the dynamic bradley terry paper

    Args:
        smoothing_param: weight of moving average
        initial_condition: mean pervious season results
        results: season results

    Returns:
        moving average with each value corresponding to an ability related to previous games
    """

    # current cell refers to result of previous cell
    shifted_results = results.shift(1).fillna(initial_condition)

    ability = []

    for K, index in enumerate(results.index):

        # newest to oldest
        previous_results = list(
            shifted_results
            [shifted_results.index <= index].sort_index(
                ascending=False))

        #  weight of mean of previous season
        initial_weight = (1 - smoothing_param) ** K * initial_condition

        # weight based on results
        dynamic_weight = smoothing_param * sum(
            [(1 - smoothing_param) ** i * value for i,
             value in enumerate(previous_results[: -1])])

        ability.append(initial_weight + dynamic_weight)

    return ability
