# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-18
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
import numpy as np
import collections

import pandas as pd
from scipy import special


log = logging.getLogger(__name__)


def legendre_polynomials(degrees, length=None, index=None, norm_factor='standard', phys_units=False):
    """Generate an array of Legendre Polynomials up to a certain degree.

    Parameters
    ----------
    phys_units : boolean
    norm_factor : str / numerical
    index : pd.DatetimeIndex
    length : int
    degrees : int/tuple

    """
    if length is None and index is None:
        raise ValueError("Either length or index must be given.")

    if index is not None:
        length = len(index)

    if isinstance(degrees, int):
        degrees = np.arange(degrees)
    elif isinstance(degrees, collections.abc.Iterable):
        degrees = np.array(degrees)

    x = np.linspace(-1, 1, length)

    lp_array = np.zeros([len(degrees), length])

    for i, n in enumerate(degrees):
        if norm_factor == 'standard':
            norm_factor = np.sqrt((2*n + 1) / 2)

        lp = special.legendre(n)(x) * norm_factor

        if phys_units:
            if n == 0:
                lp = lp*0 + 1
            elif n == 1:
                lp = lp*length/100 / (max(lp) - min(lp))

        lp_array[i] = lp

    col_names = [f'LP{i}' for i in degrees]

    lp_array = pd.DataFrame(lp_array.T, columns=col_names)

    if index is not None:
        lp_array.index = index

    return lp_array
