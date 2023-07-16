#!/usr/bin/env python

import os
import shutil
import glob
import string
letters = string.ascii_lowercase

import datetime

import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

import AIRS3D
import merra2AirsMaps

pd.set_option('display.max_rows', None)

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

    
if __name__ == "__main__":
    output_dir = os.path.join('output','Fig2_Vortex_GW_Maps')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mca = merra2AirsMaps.Merra2AirsMaps()

    dates = []
    dates.append(datetime.datetime(2018,12,9))
    dates.append(datetime.datetime(2019,1,5))
    dates.append(datetime.datetime(2019,1,16))
    dates.append(datetime.datetime(2019,2,1))

    png_name    = 'Fig2_Vortex_GW_Maps.png'
    png_fpath   = os.path.join(output_dir,png_name)

    figsize = (30,20)
    fig     = plt.figure(figsize=figsize)

    ncols           = len(dates)
    col_fwidth      = 1./ncols
    col_padded      = 0.90*col_fwidth

    row_heights     = [0.4,0.3,0.3]
    row_pad         = 0.01
    nrows           = len(row_heights)

    for col_inx,date in enumerate(dates):
        print(date)
        # fig.add_axes([left,bottom,width,height])
        left    = col_inx*col_fwidth
        width   = col_padded

        print(' --> Plotting Vortex Map')
        row_inx = 0
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(0,90))
        result  = mca.plot_ax(ax,date)

        print(' --> Plotting AIRS Global Temperature Perturbation Map')
        row_inx += 1
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.PlateCarree())
        a3dw    = AIRS3D.AIRS3DWorld(date)
        result  = a3dw.plot_ax(ax=ax)

        print(' --> Plotting AIRS Temperature Perturbation Profile')
        row_inx += 1
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height])
        a3dlp   = AIRS3D.AIRS3DLatProfile(date)
        result  = a3dlp.plot_ax(ax=ax)

#        cbar_pcoll = result.get('cbar_pcoll')
#        if cbar_pcoll is not None:
#            fig.colorbar(cbar_pcoll,label=result['cbar_label'])

    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close(fig)
    print(png_fpath)

