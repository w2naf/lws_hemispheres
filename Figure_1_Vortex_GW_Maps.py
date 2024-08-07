#!/usr/bin/env python
"""
Figure_1_Vortex_GW_Maps.py
Nathaniel A. Frissell
February 2024

This script is used to generate Figure 1 of the Frissell et al. (2024)
GRL manuscript on multi-instrument measurements of AGWs, MSTIDs, and LSTIDs.
"""

import os
import shutil
import glob
import string

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

mpl.rcParams['font.size']      = 18
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.titleweight']   = 'bold'
mpl.rcParams['axes.labelweight']   = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 9])
mpl.rcParams['axes.xmargin']   = 0
    
if __name__ == "__main__":
    output_dir = os.path.join('output','Fig1_Vortex_GW_Maps')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mca = merra2AirsMaps.Merra2AirsMaps()

    # Create a dictionary with information about each column, especially
    # the base filename of the AIRS3D profile data file we want to use.
    cols_dct = {}
    cld = cols_dct[datetime.datetime(2018,12,10)] = {}
    cld['a3d_prof_bname']   = 'NEW_2018 12 10_AIRS_3D_alt_data_45_deg_lat'
    cld['a3d_prof_lat']     = 45

    cld = cols_dct[datetime.datetime(2019,1,5)] = {}
    cld['a3d_prof_bname']   = 'NEW_2019 01 05_AIRS_3D_alt_data_50_deg_lat'
    cld['a3d_prof_lat']     = 50

    cld = cols_dct[datetime.datetime(2019,2,1)] = {}
    cld['a3d_prof_bname']   = 'NEW_2019 02 01_AIRS_3D_alt_data_48_deg_lat'
    cld['a3d_prof_lat']     = 48

    png_name    = 'Fig1_Vortex_GW_Maps.png'
    png_fpath   = os.path.join(output_dir,png_name)

    figsize = (22.5,20)
    fig     = plt.figure(figsize=figsize)

    ncols           = len(cols_dct)
    col_fwidth      = 1./ncols
    col_padded      = 0.90*col_fwidth

#    row_heights     = [0.35,0.15,0.2]
    row_heights     = [0.35,0.35]
    row_pad         = 0.05 
    nrows           = len(row_heights)

    # Create array of letters for labeling panels.
    letters = np.array(list(string.ascii_lowercase[:nrows*ncols]))
    letters.shape = (nrows,ncols)

    cbar_left   = 1.
    cbar_width  = 0.015

    # Longitude at bottom of map.
    lon_down = -90

    for col_inx,(date,cld) in enumerate(cols_dct.items()):
        print(date)
        # fig.add_axes([left,bottom,width,height])
        left    = col_inx*col_fwidth
        width   = col_padded

        print(' --> Plotting Vortex Map')
        row_inx = 0
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(lon_down,90))
        AIRS_GWv_vmin       = 0.
        AIRS_GWv_vmax       = 0.8
        # Loosely dashed negative linestyle
        # See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        merra2_vortex_kw    = {'linewidths':3,'negative_linestyles':(0, (1, 3))}
#        merra2_vortex_kw    = {'linewidths':3,'vmin':0}
        result  = mca.plot_ax(ax,date,vmin=AIRS_GWv_vmin,vmax=AIRS_GWv_vmax,cmap='RdPu',
                merra2_vortex_kw=merra2_vortex_kw)
        title   = date.strftime('%d %b %Y')
        ax.set_title(title,pad=18,fontdict={'weight':'bold','size':36})

        letter = '({!s})'.format(letters[row_inx,col_inx])
        ax.text(0.025,0.95,letter,transform=ax.transAxes,
                    fontdict={'weight':'bold','size':24})

        if col_inx == nrows-1:
            cbar_bottom = bottom
            cbar_height = height

            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cax.grid(False)
            AIRS_cbar_ticks = np.arange(AIRS_GWv_vmin,AIRS_GWv_vmax+0.1,0.1)
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,ticks=AIRS_cbar_ticks)
            cbar.set_label(result['cbar_label'])

        profile_lat     = cld['a3d_prof_lat']

        print(' --> Loading AIRS3D Data')
        a3dw    = AIRS3D.AIRS3DWorld(date)
        a3dlp   = AIRS3D.AIRS3DLatProfile(cld['a3d_prof_bname'],date,lat=profile_lat)

        # Determine Longitude Boundaries for region where there is valid data.
        tf              = np.isfinite(a3dlp.Tpert)
        LONS            = a3dlp.lons.copy()
        LONS.shape      = (len(LONS),1)
        LONS            = LONS*np.ones_like(tf,dtype=float)
        profile_lons    = [np.min(LONS[tf]),np.max(LONS[tf])]

        print(' --> Plotting AIRS Global Temperature Perturbation Map')
        row_inx += 1
        bottom  = 1. - np.sum(row_heights[:row_inx+1])
        height  = row_heights[row_inx] - row_pad
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(lon_down,90))
        result  = a3dw.plot_ax(ax=ax,vmin=-0.5,vmax=0.5)
        mca.overlay_windspeed(ax,date)
        mca.overlay_vortex(ax,date,merra2_vortex_kw=merra2_vortex_kw)

        letter = '({!s})'.format(letters[row_inx,col_inx])
        ax.text(0.025,0.95,letter,transform=ax.transAxes,
                    fontdict={'weight':'bold','size':24})

#        ax.hlines(profile_lat,profile_lons[0],profile_lons[1],color='#FE6100',lw=4)
        if col_inx == nrows-1:
            cbar_bottom = bottom
            cbar_height = height

            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cax.grid(False)
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both')
            cbar.set_label(result['cbar_label'])

#        # We were originally going to have altitude profiles of the AIRS data,
#        # but later decided those plots were not necessary for this paper.
#        print(' --> Plotting AIRS Temperature Perturbation Profile')
#        row_inx += 1
#        bottom  = 1. - np.sum(row_heights[:row_inx+1])
#        height  = row_heights[row_inx] - row_pad
#        ax      = fig.add_axes([left,bottom,width,height])
#        result  = a3dlp.plot_ax(ax=ax,vmin=-3,vmax=3,xlim=profile_lons)
#        ax.set_title(result['title'])
#        if col_inx == nrows-1:
#            cbar_bottom = bottom
#            cbar_height = height
#
#            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
#            cax.grid(False)
#            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both')
#            cbar.set_label(result['cbar_label'])


#        cbar_pcoll = result.get('cbar_pcoll')
#        if cbar_pcoll is not None:
#            fig.colorbar(cbar_pcoll,label=result['cbar_label'])

    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close(fig)
    print(png_fpath)

