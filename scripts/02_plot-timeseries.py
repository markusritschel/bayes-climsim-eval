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
"""
import fire
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



if __name__ == '__main__':
    fire.Fire({
        'member_overview': plot_all_member,
    })
