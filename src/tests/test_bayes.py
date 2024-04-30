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

    def test_add(self, bayes_eval):
        bayes_eval.add(self.datasets)
        assert bayes_eval._original_data.keys() == self.datasets.keys(), "Dictionary keys don't match"
        assert bayes_eval._original_data == self.datasets, "Dictionaries don't match"

    @pytest.mark.parametrize("degrees", [2,3,4,5])
    def test_projection(self, bayes_eval, degrees):
        G = legendre_polynomials(degrees=degrees, length=100)
        bayes_eval.add(self.datasets)
        bayes_eval.project_onto(G)
        assert bayes_eval._projected_data.keys() == self.datasets.keys()
        for id in self.datasets.keys():
            projected_dataset = bayes_eval._projected_data[id]
            assert projected_dataset.shape == (G.shape[1], self.datasets[id].shape[1])

    @pytest.mark.parametrize("degrees", [(0,1), 1, 2, 3, 4, np.arange(5)])
    def test_gdf(self, bayes_eval, degrees):
        G = legendre_polynomials(degrees=degrees, length=100)
        bayes_eval.add(self.datasets)
        bayes_eval.project_onto(G)
        bayes_eval.gdf()
        for id in self.datasets.keys():
            dist = bayes_eval._distributions[id]
            assert isinstance(dist, scipy.stats._multivariate.multivariate_normal_frozen)
            assert dist.mean.shape == (G.shape[1],)
            assert dist.cov.shape == (G.shape[1], G.shape[1])

    @pytest.mark.parametrize("degrees", [(0,1), 2, 3, 4, np.arange(5)])
    def test_gdf_with_obs_cov(self, bayes_eval, degrees):
        G = legendre_polynomials(degrees=degrees, length=100)
        bayes_eval.add(self.datasets)
        bayes_eval.project_onto(G)
        # create an invertible matrix
        n = G.shape[1]
        Σ_obs = np.random.rand(n, n)
        diag = np.sum(np.abs(Σ_obs), axis=1)
        np.fill_diagonal(Σ_obs, diag)
        bayes_eval.obs_uncertainty = Σ_obs
        bayes_eval.gdf()
        for id in self.datasets.keys():
            dist = bayes_eval._distributions[id]
            assert isinstance(dist, scipy.stats._multivariate.multivariate_normal_frozen)
            assert dist.mean.shape == (G.shape[1],)
            assert dist.cov.shape == (G.shape[1], G.shape[1])

    @pytest.mark.parametrize(
        "degrees,expected_behavior",
        [
            ((0, 1),       does_not_raise()),
            (2,            does_not_raise()),
            (3,            pytest.raises(ValueError)),
            (4,            pytest.raises(ValueError)),
            (np.arange(5), pytest.raises(ValueError)),
        ],
    )
    def test_calculate_grid_extent(self, bayes_eval, degrees, expected_behavior):
        G = legendre_polynomials(degrees=degrees, length=100)
        bayes_eval.add(self.datasets)
        bayes_eval.project_onto(G)
        with expected_behavior:
            bayes_eval._calculate_grid_extent()
