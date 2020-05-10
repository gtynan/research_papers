from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class AbstractHighDimensionalModel(ABC):

    @property  # type: ignore
    @abstractmethod
    def alpha(self) -> Tuple[float, float]:
        '''
        Alpha, weight assigned to starting abilities

        Returns:    
            (estimate, se)
        '''
        pass

    @property  # type: ignore
    @abstractmethod
    def tau_b(self) -> Tuple[float, float]:
        '''
        Tau baseline.

        Returns:    
            (estimate, se)
        '''
        pass

    @property  # type: ignore
    @abstractmethod
    def ln_likelihood(self) -> float:
        '''
        Log likelihood of fitted model
        '''
        pass

    @abstractmethod
    def fit(self) -> None:
        '''
        Minimise the negative log likelihood in order to find best model params
        '''
        pass

    @abstractmethod
    def get_fitted_ranking(self) -> pd.Series:
        '''
        Using fitted model params rank players based on data
        '''
        pass

    @staticmethod
    def _log_prediction_error(y: np.array, probs: np.array) -> np.array:
        '''
        Log of prediction error between predicted probabilities and outcomes. 
        Sum of this becomes our log likelihood

        Args:
            y: array (n, ) of outcomes for each matchup
            probs: array (n, ) of predicted probabilites for each matchup

        Returns:
            array (n, ) of log of errors
        '''
        # both should be flat same size arrays
        assert y.ndim == 1
        assert y.shape == probs.shape

        return np.log(y * probs + (1 - y) * (1 - probs))

    @staticmethod
    @abstractmethod
    def _neg_log_likelihood() -> float:
        '''
        Needed to minimise the -log likelihood
        '''
        pass

    @staticmethod
    @abstractmethod
    def _calculate_probs() -> np.array:
        '''
        Probability of p1 beating p2
        '''
        pass

    @staticmethod
    @abstractmethod
    def _calculate_score() -> Tuple[np.array, np.array]:
        '''
        Used to adjust level of strength between both players, hence returns scores of p1 and p2
        '''
        pass

    @staticmethod
    @abstractmethod
    def _update_abilities() -> np.array:
        '''
        Players new abilities based on previous performance
        '''
        pass
