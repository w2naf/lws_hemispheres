#!/usr/bin/env python
"""
Figure_1_Vortex_GW_Maps.py
Nathaniel A. Frissell
August 2024

This script is used to generate Figure 1 of the Frissell et al. (2024)
GRL manuscript on multi-instrument measurements of AGWs, MSTIDs, and LSTIDs.
"""

import os
import shutil
import glob
import string

import datetime
import pytz

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

    rows_dct = {}
    rowd = rows_dct[datetime.datetime(2018,12,15)] = {}
    rowd = rows_dct[datetime.datetime(2019,1,5)] = {}
    rowd = rows_dct[datetime.datetime(2019,2,1)] = {}

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
    lon_down = 0

    cbar_bottom = -0.05
    cbar_height =  0.025

    title_fontdict  = {'weight':'bold','size':32}
    letter_fontdict = {'weight':'bold','size':32}
    for row_inx,(date,rowd) in enumerate(rows_dct.items()):
        print(date)
        bottom  = 1. - row_heights*(row_inx+1)
        height  = row_heights - row_pad

        print(' --> Plotting Vortex Map')
        col_inx = 0
        left    = np.sum(col_fwidths[:col_inx])
        width   = col_fwidths[col_inx]

        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(lon_down,90))

        AIRS_GWv_vmin       = 0.
        AIRS_GWv_vmax       = 1.
        AIRS_GWv_cmap       = 'IDL'
        AIRS_GWv_levels     = np.arange(AIRS_GWv_vmin,AIRS_GWv_vmax+0.005,0.005)
        m2vx = {}
        m2vx['colors']      = '0.7'

        m2sf = {}
        m2sf['colors']      = '0.5'

        m2ws = {}
        m2ws['levels']      = [-10000,30,50]
        m2ws['colors']      = ['white','orange','red']

        result  = mca.plot_ax(ax,date,
                vmin=AIRS_GWv_vmin,vmax=AIRS_GWv_vmax,cmap=AIRS_GWv_cmap,
                merra2_vortex_kw=m2vx,merra2_windspeed_kw=m2ws,merra2_streamfunction_kw=m2sf)
        title   = date.strftime('%d %b %Y')
        ax.set_title(title,pad=18,fontdict=title_fontdict)

        letter = '({!s})'.format(letters[row_inx,col_inx])
        ax.text(0.025,0.95,letter,transform=ax.transAxes,fontdict=letter_fontdict)

        if row_inx == nrows-1:
            cbar_width  = 0.8*width
            cbar_left   = left + (width-cbar_width)/2.
            cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
            cax.grid(False)
            AIRS_cbar_ticks = np.arange(AIRS_GWv_vmin,AIRS_GWv_vmax+0.2,0.2)
            cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,ticks=AIRS_cbar_ticks,orientation='horizontal')
            cbar.set_label(result['cbar_label'])

        print(' --> Loading AIRS3D Data')
        a3dw    = AIRS3D.AIRS3DWorld(date)

        print(' --> Plotting AIRS Global Temperature Perturbation Map')
        m2vx = {}
        m2vx['colors']      = 'white'
        col_inx += 1
        left    = np.sum(col_fwidths[:col_inx])
        width   = col_fwidths[col_inx]
        ax      = fig.add_axes([left,bottom,width,height],projection=ccrs.Orthographic(lon_down,90))
        result  = a3dw.plot_ax(ax=ax,vmin=-0.5,vmax=0.5)
        mca.overlay_windspeed(ax,date, merra2_windspeed_kw=m2ws)
        mca.overlay_vortex(ax,date,merra2_vortex_kw=m2vx)
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
    ylim    = (datetime.datetime(2019,3,1),datetime.datetime(2018,12,1))

    m2ws = {}
    m2ws['levels']      = [-20,0,30,50]
    m2ws['colors']      = ['0.5','0.5','orange','red']

    result  = hov.plot_ax(ax,
            vmin=AIRS_GWv_vmin,vmax=AIRS_GWv_vmax,cmap=AIRS_GWv_cmap,levels=AIRS_GWv_levels,
            merra2_windspeed_kw=m2ws,ylim=ylim)

    ########################################
    # Add lines showing times of maps in left columns.
    hlines  = list(rows_dct.keys())
    for hline in hlines:
        ax.axhline(hline,lw=5,color='fuchsia',zorder=10000)

    ########################################
    # Add horizontal lines for the start and end of the SSW.
    ssw_hlines  = []
    ssw_hlines.append(datetime.datetime(2018,12,25))
    ssw_hlines.append(datetime.datetime(2019,1,10))
#    for ssw_hline in ssw_hlines:
#        ax.axhline(ssw_hline,lw=5,color='orange',zorder=10000)

    # Right-hand bracket for SSW.
    transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ssw_x0  = 0
    ssw_x1  = 1.025
    ssw_ls  = {'lw':8,'clip_on':False,'color':'DarkGoldenRod','zorder':10000}

    xx  = [ssw_x0,ssw_x1]
    yy  = [ssw_hlines[0],ssw_hlines[0]]
    ax.plot(xx,yy,transform=transform,**ssw_ls)

    xx  = [ssw_x0,ssw_x1]
    yy  = [ssw_hlines[1],ssw_hlines[1]]
    ax.plot(xx,yy,transform=transform,**ssw_ls)

    xx  = [ssw_x1,ssw_x1]
    yy  = [ssw_hlines[0],ssw_hlines[1]]
    ax.plot(xx,yy,transform=transform,**ssw_ls)

    xx  = [ssw_x1,ssw_x1+0.025]
    yy  = [ssw_hlines[0]+(ssw_hlines[1]-ssw_hlines[0])/2]*2
    ax.plot(xx,yy,transform=transform,**ssw_ls)
    xx_txt  = xx[1]+0.01
    yy_txt  = yy[1]
    ax.text(xx_txt,yy_txt,'SSW',transform=transform,va='center',
            fontdict={'weight':'bold','size':'x-large'},color=ssw_ls.get('color'))

    ########################################
    # Format yticks correctly.
    extra_dates = []
    extra_dates += hlines
    extra_dates += ssw_hlines

    yticks      = ax.get_yticks()
    ytl_dates   = []
    for ytick in yticks:
        date    = mpl.dates.num2date(ytick)
        ytl_dates.append(date)

    for extra_date in extra_dates:
        extra_date = extra_date.replace(tzinfo=pytz.UTC)
        if extra_date not in ytl_dates:
            ytl_dates.append(extra_date)

    ytl_dates.sort()

    ytls    = []
    for date in ytl_dates:
        ytl     = date.strftime('%b %d')
        ytls.append(ytl)
    ax.set_yticks(ytl_dates)
    ax.set_yticklabels(ytls)
    title_pad   = 15.
    title       = '{!s} - {!s}'.format(min(ylim).strftime('%d %b %Y'),max(ylim).strftime('%d %b %Y'))
    ax.set_title(title,fontdict=title_fontdict,pad=title_pad)
    ax.set_title('(g)',loc='left',fontdict=letter_fontdict,pad=title_pad)

    cbar_width  = 0.8*width
    cbar_left   = left + (width-cbar_width)/2.
    cax     = fig.add_axes([cbar_left,cbar_bottom,cbar_width,cbar_height])
    cax.grid(False)
    cbar    = fig.colorbar(result['cbar_pcoll'],cax=cax,extend='both',orientation='horizontal')
    cbar.set_label(result['cbar_label'])
    dtick   = 0.2
    cax.set_xticks(np.arange(AIRS_GWv_vmin,AIRS_GWv_vmax+dtick,dtick))

    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close(fig)
    print(png_fpath)

