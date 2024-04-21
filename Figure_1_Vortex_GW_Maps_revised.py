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
import hovmoller

pd.set_option('display.max_rows', None)

mpl.rcParams['font.size']      = 22
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

    # Create a dictionary with information about each row, especially
    # the base filename of the AIRS3D profile data file we want to use.
    rows_dct = {}
    rowd = rows_dct[datetime.datetime(2018,12,10)] = {}
    rowd['a3d_prof_bname']   = 'NEW_2018 12 10_AIRS_3D_alt_data_45_deg_lat'
    rowd['a3d_prof_lat']     = 45

    rowd = rows_dct[datetime.datetime(2019,1,5)] = {}
    rowd['a3d_prof_bname']   = 'NEW_2019 01 05_AIRS_3D_alt_data_50_deg_lat'
    rowd['a3d_prof_lat']     = 50

    rowd = rows_dct[datetime.datetime(2019,2,1)] = {}
    rowd['a3d_prof_bname']   = 'NEW_2019 02 01_AIRS_3D_alt_data_48_deg_lat'
    rowd['a3d_prof_lat']     = 48

    png_name    = 'Fig1_Vortex_GW_Maps_revised.png'
    png_fpath   = os.path.join(output_dir,png_name)

    figsize = (22.5,24)
    fig     = plt.figure(figsize=figsize)

    col_fwidths     = [0.35,0.35,0.50]
    ncols           = len(col_fwidths)

    nrows           = len(rows_dct)
    row_heights     = 1./nrows
    row_pad         = 0.175*row_heights


    # Create array of letters for labeling panels.
    letters = np.array(list(string.ascii_lowercase[:nrows*(ncols-1)]))
    letters.shape = (ncols-1,nrows)
    letters = letters.T

    cbar_left   = 1.
    cbar_width  = 0.015

    # Longitude at bottom of map.
    lon_down = -90

    cbar_bottom = -0.05
    cbar_height =  0.025

    title_fontdict  = {'weight':'bold','size':32}
    letter_fontdict = {'weight':'bold','size':32}
    for row_inx,(date,rowd) in enumerate(rows_dct.items()):
        print(date)
        # fig.add_axes([left,bottom,width,height])
        bottom  = 1. - row_heights*(row_inx+1)
        height  = row_heights - row_pad

        print(' --> Plotting Vortex Map')
        col_inx = 0
        left    = np.sum(col_fwidths[:col_inx])
        width   = col_fwidths[col_inx]

        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(lon_down,90))
        AIRS_GWv_vmin       = 0.
        AIRS_GWv_vmax       = 0.8
        # Loosely dashed negative linestyle
        # See https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        merra2_vortex_kw    = {'linewidths':3,'negative_linestyles':(0, (1, 3))}
        result  = mca.plot_ax(ax,date,vmin=AIRS_GWv_vmin,vmax=AIRS_GWv_vmax,cmap='RdPu',
                merra2_vortex_kw=merra2_vortex_kw)
        title   = date.strftime('%d %b %Y')
        ax.set_title(title,pad=18,fontdict=title_fontdict)

        letter = '({!s})'.format(letters[row_inx,col_inx])
        ax.text(0.025,0.95,letter,transform=ax.transAxes,fontdict=letter_fontdict)

        if row_inx == nrows-1:
            cbar_width  = 0.8*width
            cbar_left   = left + (width-cbar_width)/2.
            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cax.grid(False)
            AIRS_cbar_ticks = np.arange(AIRS_GWv_vmin,AIRS_GWv_vmax+0.1,0.1)
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,ticks=AIRS_cbar_ticks,orientation='horizontal')
            cbar.set_label(result['cbar_label'])

        profile_lat     = rowd['a3d_prof_lat']

        print(' --> Loading AIRS3D Data')
        a3dw    = AIRS3D.AIRS3DWorld(date)
        a3dlp   = AIRS3D.AIRS3DLatProfile(rowd['a3d_prof_bname'],date,lat=profile_lat)

        # Determine Longitude Boundaries for region where there is valid data.
        tf              = np.isfinite(a3dlp.Tpert)
        LONS            = a3dlp.lons.copy()
        LONS.shape      = (len(LONS),1)
        LONS            = LONS*np.ones_like(tf,dtype=float)
        profile_lons    = [np.min(LONS[tf]),np.max(LONS[tf])]

        print(' --> Plotting AIRS Global Temperature Perturbation Map')
        col_inx += 1
        left    = np.sum(col_fwidths[:col_inx])
        width   = col_fwidths[col_inx]
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(lon_down,90))
        result  = a3dw.plot_ax(ax=ax,vmin=-0.5,vmax=0.5)
        mca.overlay_windspeed(ax,date)
        mca.overlay_vortex(ax,date,merra2_vortex_kw=merra2_vortex_kw)
        ax.set_title(title,pad=18,fontdict=title_fontdict)

        letter = '({!s})'.format(letters[row_inx,col_inx])
        ax.text(0.025,0.95,letter,transform=ax.transAxes,fontdict=letter_fontdict)

        if row_inx == nrows-1:
            cbar_width  = 0.8*width
            cbar_left   = left + (width-cbar_width)/2.
            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cax.grid(False)
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both',orientation='horizontal')
            cbar.set_label(result['cbar_label'])

    ################################################################################ 
    # Hovmoller Diagram
    col_inx += 1
    left    = np.sum(col_fwidths[:col_inx]) + 0.075
    width   = col_fwidths[col_inx]
    bottom  = 0.020
    height  = 0.940 - bottom
    ax = fig.add_axes([left,bottom,width,height])

    hov     = hovmoller.Hovmoller()
    vmin    = 0.
    vmax    = 1.
    ylim    = (datetime.datetime(2019,3,1),datetime.datetime(2018,12,1))
    result  = hov.plot_ax(ax,vmin=vmin,vmax=vmax,ylim=ylim)

    yticks  = ax.get_yticks()
    ytls    = []
    for ytick in yticks:
        date    = mpl.dates.num2date(ytick)
        ytl     = date.strftime('%b %d')
        ytls.append(ytl)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytls)

    title_pad   = 15.
    title       = '{!s} - {!s}'.format(min(ylim).strftime('%Y %b %d'),max(ylim).strftime('%Y %b %d'))
    ax.set_title(title,fontdict=title_fontdict,pad=title_pad)
    ax.set_title('(g)',loc='left',fontdict=letter_fontdict,pad=title_pad)

    cbar_width  = 0.8*width
    cbar_left   = left + (width-cbar_width)/2.
    cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
    cax.grid(False)
    cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both',orientation='horizontal')
    cbar.set_label(result['cbar_label'])
    dtick   = 0.2
    cax.set_xticks(np.arange(vmin,vmax+dtick,dtick))

    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close(fig)
    print(png_fpath)

