# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-21
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal
from src.bayes import calculate_axis_extent
from src.bayes_eval import BayesEval


def test_calculate_axis_extent_2d():
    rv_dict = {
        'scenarioA': multivariate_normal(mean=np.array((0,0)), cov=np.array([[2,0],[0,2]])),
        'scenarioB': multivariate_normal(mean=np.array((.5, 0)), cov=np.array([[9,0],[0,4]]))
    }
    xy_extent = calculate_axis_extent(rv_dict)
    expected = pd.DataFrame({'min': (-7, -5), 
                             'max': (+8, +5)})
    assert np.all(xy_extent == expected)


def test_calculate_extent_1d():
    rv_dict = {
        'scenarioA': multivariate_normal(mean=0, cov=4),
        'scenarioB': multivariate_normal(mean=.5, cov=9)
    }
    x_extent = calculate_axis_extent(rv_dict)
    expected = pd.DataFrame({'min': (-7.,), 
                             'max': (+8.,)})
    assert np.all(x_extent == expected)


class TestBayesEval:
    datasets = {
        'scenarioA': pd.DataFrame(np.random.rand(100,10), columns=[f"mem{x}" for x in np.arange(1, 11)]),
        'scenarioB': pd.DataFrame(np.random.rand(100,10), columns=[f"mem{x}" for x in np.arange(1, 11)])
    }

    @pytest.fixture
    def bayes_eval(self):
        return BayesEval()

