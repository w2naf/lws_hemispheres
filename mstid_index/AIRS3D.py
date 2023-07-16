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

pd.set_option('display.max_rows', None)

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

cbar_title_fontdict     = {'weight':'bold','size':42}
cbar_ytick_fontdict     = {'size':36}
xtick_fontdict          = {'weight': 'bold', 'size':24}
ytick_major_fontdict    = {'weight': 'bold', 'size':24}
ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
title_fontdict          = {'weight': 'bold', 'size':36}
ylabel_fontdict         = {'weight': 'bold', 'size':24}
reduced_legend_fontdict = {'weight': 'bold', 'size':20}

AIRS3D_base_dir = os.path.join('data','AIRS3D')

class AIRS3DWorld(object):
    def __init__(self,date):
        #data/AIRS3D/12_09_2018_data
        #2018_12_09_AIRS_World_map_data_lat.nc
        #2018_12_09_AIRS_World_map_data_long.nc
        #2018_12_09_AIRS_World_map_data_temp_pert.nc
        data_dir        = os.path.join(AIRS3D_base_dir,date.strftime('%m_%d_%Y_data'))
        lat_nc          = date.strftime('%Y_%m_%d_AIRS_World_map_data_lat.nc')
        lon_nc          = date.strftime('%Y_%m_%d_AIRS_World_map_data_long.nc')
        Tpert_nc        = date.strftime('%Y_%m_%d_AIRS_World_map_data_temp_pert.nc')

        lat_nc_path     = os.path.join(data_dir,lat_nc)
        lon_nc_path     = os.path.join(data_dir,lon_nc)
        Tpert_nc_path   = os.path.join(data_dir,Tpert_nc)

        paths   = {}
        paths['lat_nc_path']    = lat_nc_path
        paths['lon_nc_path']    = lon_nc_path
        paths['Tpert_nc_path']  = Tpert_nc_path
        self.paths      = paths

        self.date       = date
        self.lats       = xr.load_dataset(lat_nc_path)['lat'].values
        self.lons       = xr.load_dataset(lon_nc_path)['lon'].values
        self.Tpert      = xr.load_dataset(Tpert_nc_path)['temp_pert'].values

    def plot_figure(self,png_fpath='output.png',figsize=(12,4.75),**kwargs):
        fig     = plt.figure(figsize=figsize)

        result  = self.plot_ax(**kwargs)
        ax      = result.get('ax')

        ax.set_title(result['title'])

        cbar_pcoll = result.get('cbar_pcoll')
        if cbar_pcoll is not None:
            fig.colorbar(cbar_pcoll,label=result['cbar_label'])

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,fig=None,ax=None,projection=None,
                xlim=(-180,180), ylim=(0,90),
                vmin=-0.5, vmax=0.5,
                cmap='viridis',ylabel_fontdict={},**kwargs):

        if fig is None:
            fig = plt.gcf()

        if projection is None:
            projection = ccrs.PlateCarree()

        if ax is None:
            ax      = fig.add_subplot(1,1,1, projection=projection)
            ax.set_aspect('auto')

        xx = self.lons
        yy = self.lats
        zz = self.Tpert

        cbar_pcoll = ax.pcolormesh(xx,yy,zz,vmin=vmin,vmax=vmax,**kwargs) 

        ax.coastlines()
#        ax.add_feature(cfeature.LAND, color='lightgrey')
#        ax.add_feature(cfeature.OCEAN, color = 'white')
#        ax.add_feature(cfeature.LAKES, color='white')

        ax.gridlines(draw_labels=['left','bottom'])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        title = '{!s} AIRS Global Temperatures'.format(self.date.strftime('%d %b %Y'))

        result  = {}
        result['ax']            = ax
        result['cbar_pcoll'] = cbar_pcoll
        result['cbar_label']    = '4.3 micron Temperature'
        result['title']         = title
        return result

if __name__ == "__main__":
    output_dir = os.path.join('output','AIRS3D')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dates = []
    dates.append(datetime.datetime(2018,12,9))
    dates.append(datetime.datetime(2019,1,5))
    dates.append(datetime.datetime(2019,1,16))
    dates.append(datetime.datetime(2019,2,1))

    for date in dates:
        #2018_12_10_AIRS_3D_alt_data_at_55_deg_lat_alt.nc
        #2018_12_10_AIRS_3D_alt_data_at_55_deg_lat_long.nc
        #2018_12_10_AIRS_3D_alt_data_at_55_deg_lat_temp_pert.nc
        a3dw = AIRS3DWorld(date)

        png_name    = 'AIRS3DWorld_{!s}.png'.format(date.strftime('%Y%m%d'))
        png_fpath   = os.path.join(output_dir,png_name)
        a3dw.plot_figure(png_fpath=png_fpath)
        print(png_fpath)
