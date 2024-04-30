# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-28
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging


log = logging.getLogger(__name__)


class BayesEval:
    """A class to perform statistical evaluation of different climate scenarios based on Bayesian methods."""
    def __init__(self):
        self._original_data = {}

    def add(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            self._original_data.update(args[0])
        else:
            self._original_data.update(kwargs)
        pass
