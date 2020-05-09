import pandas as pd
import numpy as np

from ..models.model_1 import Model1


def fit_model(X: pd.DataFrame, y: pd.Series, date_col: str, abilities: np.array) -> Model1:
    model = Model1()
    model.fit(X, y, date_col, abilities)
    return model
