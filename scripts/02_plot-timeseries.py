# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-18
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
"""Call this script like
$ python scripts/02_plot-timeseries.py <TYPE>
with <TYPE> being one out of the following options:
- member_overview <EXPERIMENT_ID>
- raw
"""
import fire
from datetime import datetime
import seaborn as sns
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from my_code_base.stats.timeseries import weighted_annual_mean
from src import *

log = setup_logger()


def plot_all_member(experiment_id):
    da = xr.open_dataset(DATA_DIR/f"processed/{experiment_id}_ensemble_1880-2005.nc").tas

    da_w = weighted_annual_mean(da)
    g = da.plot(col='member', col_wrap=8, add_legend=False, sharex=True, sharey=True);
    for m, ax in zip(da.member.values, g.axs.flatten()):
        da_w_ = da_w.sel(member=m)
        da_w_.plot(lw=2, ax=ax)
        ax.set_title(m)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.savefig(PLOT_DIR/f"timeseries_overview.png")


def plot_raw_timeseries():
    colors = {
        'historical': 'dodgerblue',
        'historicalGHG': 'crimson',
        'historicalNat': 'green',
        'piControl': 'grey'
    }

    fig, axes = plt.subplots(2, 2, figsize=(12,7), sharey=True)
    volcanoes = pd.read_csv(DATA_DIR/"volcanoes.dat", comment='#', sep=r',\s', 
                            names=['year', 'volcano', 'country'], index_col='year')

    experiment_ids = ['historical', 'historicalNat', 'historicalGHG', 'piControl']
    for experiment_id, ax in zip(experiment_ids, axes.flatten()):
        ds = xr.open_dataset(DATA_DIR/f"processed/{experiment_id}_ensemble_1880-2005.nc")
        ds_weighted = weighted_annual_mean(ds).tas
        mean = ds_weighted.mean('member')
        std = ds_weighted.std('member')
        color = colors[experiment_id]

        ax.axhline(0, c='k', lw=.5, zorder=0)
        ds_weighted.plot(ax=ax, hue='member', c='.4', alpha=.1, lw=.5, add_legend=False, zorder=0)
        ax.fill_between(mean.time.values, (mean + 2*std), (mean - 2*std), alpha=.3, fc=color)
        mean.plot(ax=ax, lw=3, c=color)

        ax.text(0.03, .05, experiment_id, va='bottom', transform=ax.transAxes)
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

        if experiment_id in ['historical', 'historicalNat']:
            for year, (volcano, country) in volcanoes.iterrows():
                year = datetime.strptime(str(year), '%Y')
                ax.axvline(year, ls='--', lw=.5, c='k')
                ax.text(year, .55, volcano, 
                        fontsize='x-small', ha='center', 
                        transform=ax.transData, rotation=90, 
                        bbox=dict(facecolor='w', alpha=.7, edgecolor='none', boxstyle='round,pad=.5')
                        )
        
        if experiment_id.startswith('hist'):
            ax.axvspan(datetime.strptime("1961", '%Y'), datetime.strptime("1990", '%Y'), 
                    fc='orange', alpha=.1, zorder=1)

        sns.despine()

    fig.supylabel('Temperature anomalies [K]')
    fig.supxlabel('Time (year)')
    plt.tight_layout()

    plt.savefig(PLOT_DIR/f"timeseries_scenarios.png")




if __name__ == '__main__':
    fire.Fire({
        'member_overview': plot_all_member,
        'raw': plot_raw_timeseries, 
    })
