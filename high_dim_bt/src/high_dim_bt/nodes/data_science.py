from typing import Tuple
import pandas as pd
import numpy as np

from ..models.model_1 import Model1


def fit_model(X: pd.DataFrame, y: pd.Series, date_col: str, abilities: np.array) -> Model1:
    '''
    Fits and returns model
    '''
    model = Model1()
    model.fit(X, y, date_col, abilities)
    return model


def get_outputs(model: Model1) -> Tuple[pd.Series, pd.Series]:
    '''
    Returns the fitted abilities and the model params in a series
    '''
    fitted_abilities = model.get_ranking()
    model_params = pd.Series(
        {"alpha": model.alpha, "tau_b": model.tau_b,
         "log_likelihood": model.ln_likelihood})
    return fitted_abilities, model_params
