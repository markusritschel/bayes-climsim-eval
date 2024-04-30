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


log = logging.getLogger(__name__)


class BayesEval:
    """A class to perform statistical evaluation of different climate scenarios based on Bayesian methods."""
    def __init__(self):
        self._original_data = {}
        self._projected_data = {}
        self._distributions = {}

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

    def gdf(self):
        """Fit a Gaussian normal distribution to the projected datasets."""
        parameters = {
            id: {
                'µ': np.mean(x, axis=1),
                'Σ': empirical_covariance(x)
            }
            for (id,x) in self._projected_data.items()
        }
        self._distributions = {
            id: multivariate_normal(mean=params['µ'], cov=params['Σ'])
            for (id, params) in parameters.items()
        }

