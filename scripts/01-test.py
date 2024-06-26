#!/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-04-17
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
"""The functions below only serve for demonstrating the logger setup. Feel free to delete them 
as well as this docstring once the concept is understood :-)"""
from __future__ import absolute_import, division, print_function, with_statement
import logging

from src import *
from src import submodule

# This is only necessary if this file gets imported by another one so that logs get piped
logger = logging.getLogger(__name__)


def main():
    logger.info("Test")


if __name__ == '__main__':
    logger = setup_logger()
    main()
    submodule.sub_fun()
