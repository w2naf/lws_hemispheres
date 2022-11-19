#!/usr/bin/env python
"""
This script pre-processes the MERRA2 outputs provided by Lynn Harvey for use with climo_plot.py.
"""

import os

import numpy as np
import pandas as pd

import netCDF4 as nc

data_dir = os.path.join('data','merra2')
fpath    = os.path.join(data_dir,'MERRA2_U_SuperDARN_NH_2010010100-2022073118.nc')

ds = nc.Dataset(fpath)

breakpoint()
