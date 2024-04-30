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


def find_decision_surfaces(pdf_list):
    """Find the decision surfaces based on a list of probability density functions.
    Mask values that are smaller than the difference between 1.0 and the next smallest 
    representable float larger than 1.0 (numpy.finfo(np.float64).eps).

    Parameters
    ----------
    pdf_list : list
        A list of probability density functions.

    Returns
    -------
    decision_surfaces : numpy.ndarray
        An array containing the decision surfaces, identified by integer.
    """
    pdfs_stacked = np.dstack(list(pdf_list)).squeeze()
    decision_surfaces = np.nanargmax(pdfs_stacked, axis=-1)

    eps_mask = np.all(pdfs_stacked < np.finfo(np.float64).eps, axis=-1)
    decision_surfaces = np.ma.masked_where(eps_mask, decision_surfaces)
    return decision_surfaces


def find_decision_bnds(pdf_list) -> pd.DataFrame:
    """Find the decision boundaries based on a list of probability density functions (PDFs).

    Parameters
    ----------
    pdf_list: list
        A list of probability density functions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the start and end indices of each group of decision boundaries, 
        along with an identifier for each group (column names: [start, end, id]).
    """
    masked_array = find_decision_surfaces(pdf_list)
    # Find the indices where the array changes
    indices = np.where(np.diff(masked_array) != 0)[0] + 1
    # Add the start and end indices
    indices = np.r_[0, indices, masked_array.size - 1]
    # Create pairs of indices representing the start and end of each group
    groups = np.array(
        [
            (start, end, masked_array[start])
            for start, end in zip(indices[:-1], indices[1:])
        ]
    )
    return pd.DataFrame(groups, columns=["start", "end", "id"])
