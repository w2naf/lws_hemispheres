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
mpl.rcParams['axes.titleweight']   = 'bold'
mpl.rcParams['axes.labelweight']   = 'bold'
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

    row_heights     = [0.35,0.15,0.2]
    row_pad         = 0.03 
    nrows           = len(row_heights)

    cbar_left   = 1.
    cbar_width  = 0.015
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
        result  = mca.plot_ax(ax,date,vmin=0.,vmax=0.8)
        title   = date.strftime('%d %b %Y')
        ax.set_title(title,pad=18,fontdict={'weight':'bold','size':36})

        if col_inx == nrows-1:
            cbar_bottom = bottom
            cbar_height = height

            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax)
            cbar.set_label(result['cbar_label'])

        profile_lat     = 55
        profile_lons    = [-85,-40]

        print(' --> Plotting AIRS Global Temperature Perturbation Map')
        row_inx += 1
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.PlateCarree())
        ax.set_aspect('auto')
        a3dw    = AIRS3D.AIRS3DWorld(date)
        result  = a3dw.plot_ax(ax=ax,vmin=-0.5,vmax=0.5)
        ax.set_title(result['title'])
        ax.axhline(profile_lat,color='red')
        ax.axvline(np.mean(profile_lons),color='red')
        if col_inx == nrows-1:
            cbar_bottom = bottom
            cbar_height = height

            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both')
            cbar.set_label(result['cbar_label'])


        print(' --> Plotting AIRS Temperature Perturbation Profile')
        row_inx += 1
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height])
        a3dlp   = AIRS3D.AIRS3DLatProfile(date,lat=profile_lat)
        result  = a3dlp.plot_ax(ax=ax,vmin=-3,vmax=3,xlim=profile_lons)
        ax.set_title(result['title'])
        if col_inx == nrows-1:
            cbar_bottom = bottom
            cbar_height = height

            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both')
            cbar.set_label(result['cbar_label'])


#        cbar_pcoll = result.get('cbar_pcoll')
#        if cbar_pcoll is not None:
#            fig.colorbar(cbar_pcoll,label=result['cbar_label'])

    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close(fig)
    print(png_fpath)

