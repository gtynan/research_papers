from typing import Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import copy

from .abstract_high_dm import AbstractHighDimensionalModel

# Raised when certain properties or functions called prior to fit()
fit_error: str = 'Cannot call prior to fit()'


class Model1(AbstractHighDimensionalModel):

    @property
    def alpha(self) -> Tuple[float, float]:
        '''
        Weight of ln(WTA points) assigned as inital abilities.

        Returns:
            (estimate, standard error)
        '''
        try:
            return self._alpha
        except AttributeError:
            raise Exception(fit_error)

    @property
    def tau_b(self) -> Tuple[float, float]:
        '''
        Weight assigned to score function when updating abilites

        Returns:
            (estimate, standard error)
        '''
        try:
            return self._tau_b
        except AttributeError:
            raise Exception(fit_error)

    @property
    def ln_likelihood(self) -> float:
        '''
        Maximised Log likelihood of data given tau_b and alpha
        '''
        try:
            return self._ln_likelihood
        except AttributeError:
            raise Exception(fit_error)

    def get_fitted_ranking(self) -> pd.Series:
        '''
        Runs the fitting once more using the fitted alpha and tau_b to get end abilities
        '''
        try:
            abilities = Model1._neg_log_likelihood(
                params=(self.alpha[0], self.tau_b[0]),
                X=self.X, y=self.y, date_col=self.date_col,
                abilities=self.starting_abilities, return_abilities=True)

            players = self.X.drop(columns=self.date_col).columns

            return pd.Series(
                data=abilities, index=players).sort_values(
                ascending=False)

        except AttributeError:
            raise Exception(fit_error)

    def fit(
            self, X: pd.DataFrame, y: pd.Series, date_col: str,
            abilites: np.array) -> None:
        '''
        Finds tau_b and alpha parameters to maximise log likelihood.

        Args:
            X: rows of matchups with data column also (assumes sorted)
            y: results for each matchup
            date_col: column conating dates (to groupby)

        Returns:
            None
        '''
        # used to get abilities after fit
        self.X = X
        self.y = y
        self.date_col = date_col
        self.starting_abilities = abilites

        res = minimize(Model1._neg_log_likelihood,
                       x0=[0, 0],
                       bounds=((0, 1), (0, 1)),
                       args=(X, y, date_col, abilites))

        # diag of hessian inverse is variance thus sqrt of variance se
        se = np.sqrt(np.diag(res['hess_inv'].todense()))

        self._alpha = (res['x'][0], se[0])
        self._tau_b = (res['x'][1], se[1])
        self._ln_likelihood = -res['fun']

    @staticmethod
    def _neg_log_likelihood(
            params: Tuple[float, float],
            X: pd.DataFrame, y: pd.Series, date_col: str, abilities: np.array,
            return_abilities: bool = False) -> float:
        '''
        Negative of the log likelihood

        Args:
            params: {alpha, tau_b}
            X: data
            y: outcomes
            date_col: column conating dates (to groupby)
            abilities: starting abilities of players

        Reurns:
            Negative of the log likelihood
        '''
        likelihood = 0

        alpha, tau_b = params

        # 1p to avoid case of 0 abilitiy
        abilities = np.log1p(abilities) * alpha

        for t, (_, x) in enumerate(X.groupby(by=date_col)):
            index = x.index

            x = x.drop(columns=date_col).to_numpy()
            results = y[index].to_numpy()

            probs = Model1._calculate_probs(x, abilities)

            likelihood += np.sum(Model1._log_prediction_error(results, probs))

            p1_score, p2_score = Model1._calculate_score(
                results, probs)

            abilities = Model1._update_abilities(
                x, abilities, p1_score, p2_score, tau_b)

        # used by get fitted abilities
        if return_abilities:
            return abilities

        return -likelihood

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
        # X can't be flat otherwise argwhere will fail
        assert X.ndim > 1
        # X columns = abilites length,
        assert X.shape[1] == abilities.shape[0]

        # removed to see if argwhere faster
        '''
        # elementwise multiplication gives p1 and -p2 values
        # sum calculates the difference
        # ability_diff = np.sum(X * abilities, axis=1)
        '''
        # column position for each row
        p1_index = np.argmax(X, axis=1)
        p2_index = np.argmin(X, axis=1)

        ability_diff = abilities[p1_index] - abilities[p2_index]

        return np.exp(ability_diff) / (1 + np.exp(ability_diff))

    @staticmethod
    def _calculate_score(y: np.array, probs: np.array) -> Tuple[np.array, np.array]:
        '''
        Score function to add to abilites weighted by tau_b

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
    def _update_abilities(
            X: np.array, abilities: np.array, p1_score: np.array,
            p2_score: np.array, tau_b: float) -> np.array:
        '''
        Generates players new abilities by adding score * tau_b to previous abilities in a random walk process

        Args:
            X: array (n x p) of matchups where each row corresponds to a match.
            abilites: array (p, ) of player abilites
            p1_score: array (n, ) to be weighted by tau_b and added to abilites
            p2_score: array (n, ) to be weighted by tau_b and added to abilites
            tau_b: weight of new scores

        Returns:
            Array of updated player abilities (p, )
        '''
        # p length
        assert X.shape[1] == abilities.shape[0]
        # n length
        assert X.shape[0] == p1_score.shape[0] == p2_score.shape[0]
        # flat arrays
        assert abilities.ndim == p1_score.ndim == p2_score.ndim == 1

        new_abilites = copy.deepcopy(abilities)

        # only 1 entry > 0 for each matchup (row) thus max returns positon
        p1_index = np.argmax(X, axis=1)
        new_abilites[p1_index] += (p1_score * tau_b)

        # only 1 entry < 0 for each matchup (row) thus max returns positon
        p2_index = np.argmin(X, axis=1)
        new_abilites[p2_index] += (p2_score * tau_b)

        return new_abilites
