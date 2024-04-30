# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-28
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
from my_code_base.linalg import inv
import numpy as np


log = logging.getLogger(__name__)


class BayesEval:
    """A class to perform statistical evaluation of different climate scenarios based on Bayesian methods."""
    def __init__(self):
        self._original_data = {}
        self._projected_data = {}

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
