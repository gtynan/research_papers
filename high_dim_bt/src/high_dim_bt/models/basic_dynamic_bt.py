from typing import Tuple
import numpy as np


class BasicDynamicModel:

    def __init__(self):
        pass

    def fit(self):
        pass

    @staticmethod
    def _calculate_probs(X: np.array, abilities: np.array) -> np.array:
        '''
        Calcualtes probability of p1 beating p2 in a given mathcup using abilities at that time

        Args:
            X: array (n x p) of matchups where each row corresponds to a match.
            abilites: array (p, ) of player abilites

        Returns:
            array (n, ) of probability of p1 beating p2 in a given matchup  
        '''
        # abilities to be flat array
        assert abilities.ndim == 1
        # X columns = abilites length,
        assert X.shape[1] == abilities.shape[0]

    @staticmethod
    def _log_prediction_error(y: np.array, probs: np.array) -> np.array:
        '''
        Log of prediction error between predicted probabilities and outcomes

        Args:
            y: array (n, ) of outcomes for each matchup
            probs: array (n, ) of predicted probabilites for each matchup

        Returns:
            array (n, ) of log of errors
        '''
        # both should be flat arrays
        assert y.shape == probs.shape
        assert y.ndim == 1

    @staticmethod
    def _calculate_score(y: np.array, probs: np.array) -> Tuple[np.array, np.array]:
        '''
        Score function to add to abilites weighted by tau

        Args:
            y: array (n, ) of outcomes for each matchup
            probs: array (n, ) of predicted probabilites for each matchup

        Returns:
            p1_score, p2_score both (n, )
        '''
        # both should be flat arrays
        assert y.shape == probs.shape
        assert y.ndim == 1

    @staticmethod
    def _update_abilities(
            X: np.array, abilities: np.array, p1_score: np.array,
            p2_score: np.array, tau: float):
        '''
        Generates players new abilities by adding score * tau to previous abilities in a random walk process

        Args:
            X: array (n x p) of matchups where each row corresponds to a match.
            abilites: array (p, ) of player abilites
            p1_score: array (n, ) to be weighted by tau and added to abilites 
            p2_score: array (n, ) to be weighted by tau and added to abilites 
            tau: weight of new scores

        Returns:
            Array of updated player abilities (p, )
        '''
        # p length
        assert X.shape[1] == abilities.shape[0]
        # n length
        assert X.shape[0] == p1_score.shape[0] == p2_score.shape[0]
        # flat arrays
        assert abilities.ndim == p1_score.ndim == p2_score.ndim == 1
