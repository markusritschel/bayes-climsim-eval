# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-28
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
from my_code_base.linalg import inv, empirical_covariance
import numpy as np
from scipy.stats import multivariate_normal

from src.bayes import calculate_axis_extent

log = logging.getLogger(__name__)


class BayesEval:
    """A class to perform statistical evaluation of different climate scenarios based on Bayesian methods."""
    def __init__(self):
        self._original_data = {}
        self._projected_data = {}
        self._distributions = {}
        self._obs_cov = np.array(0)
        self._projected_obs_cov = np.array(0)
        self.grid = None
        self._pos = None

    def add(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            self._original_data.update(args[0])
        else:
            self._original_data.update(kwargs)
        self._projected_data = {}

    def project_onto(self, guess_matrix):
        """Project all datasets onto a matrix of guess vectors. This is the step of data reduction."""
        G = guess_matrix
        P = inv(G.T.dot(G)).dot(G.T)
        self._projected_data = {k:P.dot(x) for (k,x) in self._original_data.items()}
        if np.any(self.obs_uncertainty):
            self._projected_obs_cov = P.dot(self.obs_uncertainty).dot(P.T)

    def gdf(self):
        """Fit a Gaussian normal distribution to the projected datasets."""
        parameters = {
            id: {
                'µ': np.mean(x, axis=1),
                'Σ': empirical_covariance(x) + self._projected_obs_cov
            }
            for (id,x) in self._projected_data.items()
        }
        self._distributions = {
            id: multivariate_normal(mean=params['µ'], cov=params['Σ'])
            for (id, params) in parameters.items()
        }

    @property
    def obs_uncertainty(self):
        return self._obs_cov
    
    @obs_uncertainty.setter
    def obs_uncertainty(self, Σ_obs):
        if np.any(self._obs_cov):
            print("Warning: Observation uncertainty already exists.")
        self._obs_cov = Σ_obs

    def _calculate_grid_extent(self, **kwargs):
        rv_dict = {}
        for k, x in self._projected_data.items():
            µ = np.mean(x, axis=1)
            Σ = empirical_covariance(x)
            if µ.size > 2:
                raise ValueError("Cannot calculate grid extent for dimensions higher than 2.")
            rv_dict[k] = multivariate_normal(mean=µ, cov=Σ + self._projected_obs_cov)

        minmax_df = calculate_axis_extent(rv_dict, **kwargs)
        self.grid = np.mgrid[[slice(min_,max_,500j) for (min_,max_) in minmax_df.values]]
        self.x = self.grid[0][:, 0]
        self.y = self.grid[1][0, :]
        self._pos = np.dstack(self.grid)

    def get_likelihood(self, scenario_id):
        """Evaluated on a grid"""
        self._calculate_grid_extent()
        mrv = self._distributions[scenario_id]
        likelihood = mrv.pdf
        return likelihood(self._pos)

    def get_decision_probability(self, scenario_id, **kwargs):
        """Return the decision probability (posteriori) for a given scenario.
        Evaluated on a grid.

        Parameters
        ----------
        likelihoods : dict
        scenario_id : str
        x : np.ndarray
        """
        self._calculate_grid_extent()
        if self._pos is None:
            self._calculate_grid_extent(**kwargs)
        N = len(self._distributions)
        r = self._calculate_norm_factor()
        likelihood = self._distributions[scenario_id].pdf
        return likelihood(self._pos) / (N * r)
