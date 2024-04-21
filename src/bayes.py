# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-19
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
import numpy as np


log = logging.getLogger(__name__)


def calculate_axis_extent(rv_dict, factor=2.5):
    """Calculate the axis extents based on the mean and covariance parameters of a set of :class:`scipy.stats.multivariate_normal` objects.

    Parameters
    ----------
    rv_dict: dictionary
        A dictionary with EXPERIMENT_ID as keys and :class:`scipy.stats.multivariate_normal` with ``mean`` and ``cov`` as parameters to be evaluated.

    Returns
    -------
    DataFrame: A :class:`pandas.DataFrame` containing the calculated axis extents.
    """
    extents = {
        'min': {},
        'max': {}
    }
    for id, entry in rv_dict.items():
        µ = np.atleast_1d(entry.mean)
        Σ = np.atleast_1d(entry.cov)
        
        for dim_idx in np.arange(µ.size):
            µ_ = µ[dim_idx]
            σ_ = np.sqrt(Σ[dim_idx, dim_idx])
            extents['max'].setdefault(dim_idx, []).append(µ_ + factor*σ_)
            extents['min'].setdefault(dim_idx, []).append(µ_ - factor*σ_)
    
    extents['max'] = {k: np.max(v) for k, v in extents['max'].items()}
    extents['min'] = {k: np.min(v) for k, v in extents['min'].items()}

    return pd.DataFrame.from_dict(extents)
