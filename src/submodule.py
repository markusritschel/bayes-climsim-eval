# !/usr/bin/env python3
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-17
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
"""This submodule exists only for demonstration purposes, to show the concept of the logger setup and running a pytest.
"""
import logging

logger = logging.getLogger(__name__)

def generate_int_list(x: int = 3) -> list[int]:
    """This is a doctest in a docstring. You can link to other functions, e.g. :func:`src.setup_logger`.
    Function names should be self-explanatory. 

    Example
    -------
    >>> generate_int_list(3)
    [1, 2, 3]
    """
    logger.info("From sub-module")
    return list(range(1,x+1))
