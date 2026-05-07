# !/usr/bin/env python3
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-17
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import pytest
from bayes_climsim_eval.submodule import generate_int_list

def test_subfunc(global_fixture):
    l = generate_int_list()
    assert isinstance(l, list)
    assert isinstance(global_fixture, str)
