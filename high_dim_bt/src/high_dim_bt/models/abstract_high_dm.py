from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Raised when certain properties or functions called prior to fit()
fit_error: str = 'Cannot call prior to fit()'


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
    def get_ranking(self) -> pd.Series:
        '''
        Using fitted model params rank players based on data
        '''
        pass

    @staticmethod
    @abstractmethod
    def _neg_log_likelihood() -> float:
        '''
        Needed to minimise the -log likelihood
        '''
        pass

    @staticmethod
    @abstractmethod
    def _calculate_abilities() -> Tuple[np.array, np.array]:
        '''
        Player 1 and 2 current abilities
        '''
        pass

    @staticmethod
    def _calculate_probs(p1_abilities: np.array,
                         p2_abilities: np.array) -> np.array:
        '''
        Formula 1. 

        Logit probability of p1 beating p2 in a given mathcup using abilities at that time

        Args:
            p1_abilites: player_1 abilities
            p2_abilites: player_2 abilities

        Returns:
            array (n, ) of probability of p1 beating p2 in a given matchup
        '''
        assert p1_abilities.shape == p2_abilities.shape

        ability_diff = p1_abilities - p2_abilities

        print(ability_diff.shape)

        return np.exp(ability_diff) / (1 + np.exp(ability_diff))

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
    def _calculate_score(y: np.array, probs: np.array) -> Tuple[np.array, np.array]:
        '''
        Formula 3. 

        Score function to add to abilites 

        Args:
            y: array (n, ) of outcomes for each matchup
            probs: array (n, ) of predicted probabilites for each matchup

        Returns:
            p1_score, p2_score both (n, )
        '''
        # both should be flat arrays
        assert y.shape == probs.shape
        assert y.ndim == 1

        s1 = y * (1 - probs) - (1 - y) * probs
        return s1, -s1

    @staticmethod
    @abstractmethod
    def _update_abilities() -> np.array:
        '''
        Players new abilities based on previous performance
        '''
        pass
