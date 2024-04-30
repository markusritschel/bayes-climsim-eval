# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-30
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging

import fire
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import my_code_base
import xarray as xr
from src import *
from src.bayes_eval import BayesEval
from src.data_projection import legendre_polynomials

log = setup_logger()

COLORS = {
    'historical': 'dodgerblue',
    'historicalGHG': 'crimson',
    'historicalNat': 'g',
    'piControl': 'grey'
}

EXPERIMENT_IDs = ['historical', 'historicalGHG', 'historicalNat', 'piControl']


def plot_decision_probabilities(degrees: tuple, gamma=None):
    obs = xr.open_dataset(DATA_DIR/f"processed/observations_1880-2005.nc")
    o = obs.to_dataframe()['tas_mean']

    G = legendre_polynomials(degrees=degrees, index=o.index)
    P = my_code_base.linalg.inv((G.T.dot(G))).dot(G.T)
    µ_obs = P.dot(o)

    datasets = {}
    for experiment_id in EXPERIMENT_IDs:
        ds = xr.open_dataset(DATA_DIR/f"processed/{experiment_id}_ensemble_1880-2005.nc")

        X = ds.to_dataframe().unstack()['tas'].dropna(axis=1)
        datasets[experiment_id] = X

    bayes_eval = BayesEval()
    bayes_eval.add(datasets)

    if gamma:
        Σ_ctrl = my_code_base.linalg.empirical_covariance(datasets['piControl'])
        bayes_eval.obs_uncertainty = gamma**2 * Σ_ctrl

    bayes_eval.project_onto(G)
    bayes_eval.gdf()

    fig, axes = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True, layout="constrained")
    for experiment_id, ax in zip(EXPERIMENT_IDs, axes.flatten()):
        dec_prob = bayes_eval.get_decision_probability(experiment_id)
        ax.contourf(*bayes_eval.grid, dec_prob, 
                    cmap=mcolors.LinearSegmentedColormap.from_list("mycmap", ['w', COLORS[experiment_id]])
                    )
        cs = ax.contour(*bayes_eval.grid, dec_prob, colors='k', linewidths=.5)
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize='xx-small')
        box_props = dict(boxstyle='round', facecolor='w', alpha=1, ec='none', pad=.5)
        ax.text(.95, .95, experiment_id, c=COLORS[experiment_id], transform=ax.transAxes, ha='right', va='top', bbox=box_props)

        # Add marker for the observations
        ax.scatter(*µ_obs, marker='P', s=200, color='k', ec='w', lw=2, zorder=10)
        ax.axvline(µ_obs.iloc[0], color='k', ls=':', lw=1, zorder=10)
        ax.axhline(µ_obs.iloc[1], color='k', ls=':', lw=1, zorder=10)

    fig.supxlabel(G.columns[0])
    fig.supylabel(G.columns[1])

    gamma_info = '' if gamma is None else f"_gamma-{gamma}"
    plt.savefig(PLOT_DIR/f"decision-probabilities_LP{degrees[0]}-LP{degrees[1]}{gamma_info}.png")


if __name__ == '__main__':
    fire.Fire(plot_decision_probabilities)
