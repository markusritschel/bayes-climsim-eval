# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-18
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import fire
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import multivariate_normal
import numpy as np
import xarray as xr
import my_code_base.linalg
from src.bayes import find_decision_bnds, find_decision_surfaces
from src.bayes_eval import BayesEval
from src.data_projection import legendre_polynomials
from src import *

log = setup_logger()


COLORS = {
    'historical': 'dodgerblue',
    'historicalGHG': 'crimson',
    'historicalNat': 'g',
    'piControl': 'grey'
}

EXPERIMENT_IDs = ['historical', 'historicalGHG', 'historicalNat', 'piControl']



def plot_pdfs(degrees: tuple, gamma=None):
    obs = xr.open_dataset(DATA_DIR/f"processed/observations_1880-2005.nc")
    o = obs.to_dataframe()['tas_mean']

    G = legendre_polynomials(degrees=degrees, index=o.index)
    P = my_code_base.linalg.inv((G.T.dot(G))).dot(G.T)

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

    fig = plt.figure(figsize=(8,8), constrained_layout=False)

    gs = fig.add_gridspec(2, 2, width_ratios=(5, 2), height_ratios=(2, 5), wspace=.0, hspace=.0)
    ax_joint = fig.add_subplot(gs[1,0])
    ax_margx = fig.add_subplot(gs[0,0], sharex=ax_joint)
    ax_margy = fig.add_subplot(gs[1,1], sharey=ax_joint)
    ax_legend = fig.add_subplot(gs[0,1])

    all_joint_pdfs = {}
    all_margx_pdfs = {}
    all_margy_pdfs = {}

    for i, experiment_id in enumerate(EXPERIMENT_IDs):
        color = COLORS[experiment_id]
        xlabel = G.columns[0]
        ylabel = G.columns[1]

        bayes_eval.gdf()
        likelihood = bayes_eval.get_likelihood(experiment_id)
        cs = ax_joint.contour(*bayes_eval.grid, likelihood, linewidths=1, colors=color, zorder=10)
        ax_joint.clabel(cs, cs.levels[::2], inline=True, fontsize=10)
        ax_joint.set_xlabel(xlabel)
        ax_joint.set_ylabel(G.columns[1])
        all_joint_pdfs[experiment_id] = likelihood

        µ = bayes_eval._distributions[experiment_id].mean
        Σ = bayes_eval._distributions[experiment_id].cov
        
        # Plot marginal PDF for first index
        rv = multivariate_normal(mean=µ[0], 
                                 cov=Σ[0,0])
        pdf = rv.pdf(bayes_eval.x)
        ax_margx.plot(bayes_eval.x, pdf, c=color)
        all_margx_pdfs[experiment_id] = pdf

        # Plot marginal PDF for second index
        rv = multivariate_normal(mean=µ[1], 
                                 cov=Σ[1,1])
        pdf = rv.pdf(bayes_eval.y)
        ax_margy.plot(pdf, bayes_eval.y, c=color)
        all_margy_pdfs[experiment_id] = pdf

        # Plot the projected info of each individual member as a scatterpoint
        bayes_eval._projected_data[experiment_id].T.plot.scatter(ax=ax_joint, x=xlabel, y=ylabel, 
                                                    marker='.', s=15, alpha=.3, c=color)

        legend_y_pos = np.linspace(.7, .3, len(EXPERIMENT_IDs), endpoint=True)[i]
        ax_legend.text(.2, legend_y_pos, experiment_id,
                       c=color, transform=ax_legend.transAxes)
        ax_legend.axis('off')

    for ax in fig.axes:
        ax.label_outer()
    ax_joint.set_xlim(bayes_eval.x.min(), bayes_eval.x.max())
    ax_joint.set_ylim(bayes_eval.y.min(), bayes_eval.y.max())

    # Plot decision boundaries
    # ...for the joint PDFs
    cmap = mcolors.ListedColormap(list(COLORS.values()))
    dec_bnds = find_decision_surfaces(all_joint_pdfs.values())
    ax_joint.pcolormesh(*bayes_eval.grid, dec_bnds, cmap=cmap, alpha=.25, zorder=0)
    if np.any(dec_bnds.mask):
        ax_joint.contourf(*bayes_eval.grid, dec_bnds.mask, hatches=[None, '//'], colors='none')

    # ...for the marginal PDFs
    groups = find_decision_bnds(all_margx_pdfs.values())
    for _, g in groups.iterrows():
        idx_range = g[['start','end']]
        color = COLORS[EXPERIMENT_IDs[g['id']]]
        ax_margx.fill_between(bayes_eval.x[idx_range], 0, np.max(list(all_margx_pdfs.values())), 
                              fc=color, alpha=.2)

    groups = find_decision_bnds(all_margy_pdfs.values())
    for _, g in groups.iterrows():
        idx_range = g[['start','end']]
        color = COLORS[EXPERIMENT_IDs[g['id']]]
        ax_margy.fill_betweenx(bayes_eval.y[idx_range], 0, np.max(list(all_margy_pdfs.values())), 
                               fc=color, alpha=.2)


    # Add marker for the observations
    µ_obs = P.dot(o)
    ax_joint.scatter(*µ_obs, marker='P', s=200, color='k', ec='w', lw=2, zorder=10)
    ax_joint.axvline(µ_obs.iloc[0], color='k', ls=':', lw=1, zorder=10)
    ax_joint.axhline(µ_obs.iloc[1], color='k', ls=':', lw=1, zorder=10)

    ax_margx.axvline(µ_obs.iloc[0], color='k', lw=2, zorder=10)
    ax_margy.axhline(µ_obs.iloc[1], color='k', lw=2, zorder=10)

    gamma_info = '' if gamma is None else f"_gamma-{gamma}"
    plt.savefig(PLOT_DIR/f"multivariate-pdfs_LP{degrees[0]}-LP{degrees[1]}{gamma_info}.png")


if __name__ == '__main__':
    fire.Fire(plot_pdfs)
