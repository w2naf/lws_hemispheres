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
    
    def plot_ax(self,fig=None,ax=None,
            gridlines=True,coastlines=True,
                xlim=(-180,180), ylim=(0,90),
                vmin=-0.5, vmax=0.5,
                cmap='viridis',ylabel_fontdict={},**kwargs):

        if fig is None:
            fig = plt.gcf()

        if ax is None:
            ax      = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
            ax.set_aspect('auto')

        xx = self.lons
        yy = self.lats
        zz = self.Tpert

        cbar_pcoll = ax.pcolormesh(xx,yy,zz,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree(),**kwargs) 

        if coastlines is True:
            ax.coastlines()
#        ax.add_feature(cfeature.LAND, color='lightgrey')
#        ax.add_feature(cfeature.OCEAN, color = 'white')
#        ax.add_feature(cfeature.LAKES, color='white')

        if gridlines is True:
            gl = ax.gridlines(draw_labels='x',color='k',lw=1.5)

#        ax.set_xlim(xlim)
#        ax.set_ylim(ylim)

        title = '{!s} AIRS Global Temperatures'.format(self.date.strftime('%d %b %Y'))

        result  = {}
        result['ax']            = ax
        result['cbar_pcoll'] = cbar_pcoll
        result['cbar_label']    = 'AIRS 4.3 micron Brightness\nTemperature Perturbation [K]'
        result['title']         = title
        return result

class AIRS3DLatProfile(object):
    def __init__(self,bname,date,lat=55):
        #data/AIRS3D/12_09_2018_data
        #2018_12_10_AIRS_3D_alt_data_at_55_deg_lat_alt.nc
        #2018_12_10_AIRS_3D_alt_data_at_55_deg_lat_long.nc
        #2018_12_10_AIRS_3D_alt_data_at_55_deg_lat_temp_pert.nc
        data_dir        = os.path.join(AIRS3D_base_dir,date.strftime('%m_%d_%Y_data'))

        date_str        = date.strftime('%Y_%m_%d')

#        alt_nc          = '{!s}_AIRS_3D_alt_data_at_{:0.0f}_deg_lat_alt.nc'.format(date_str,lat)
#        lon_nc          = '{!s}_AIRS_3D_alt_data_at_{:0.0f}_deg_lat_long.nc'.format(date_str,lat)
#        Tpert_nc        = '{!s}_AIRS_3D_alt_data_at_{:0.0f}_deg_lat_temp_pert.nc'.format(date_str,lat)

        alt_nc          = '{!s}_alt.nc'.format(bname)
        lon_nc          = '{!s}_long.nc'.format(bname)
        Tpert_nc        = '{!s}_temp_pert.nc'.format(bname)

        alt_nc_path     = os.path.join(data_dir,alt_nc)
        lon_nc_path     = os.path.join(data_dir,lon_nc)
        Tpert_nc_path   = os.path.join(data_dir,Tpert_nc)

        paths   = {}
        paths['alt_nc_path']    = alt_nc_path
        paths['lon_nc_path']    = lon_nc_path
        paths['Tpert_nc_path']  = Tpert_nc_path
        self.paths      = paths

        self.date       = date
        self.lat        = lat
        self.alts       = xr.load_dataset(alt_nc_path)['alt'].values
        self.lons       = xr.load_dataset(lon_nc_path)['lon'].values
        self.Tpert      = xr.load_dataset(Tpert_nc_path)['temp_pert'].values

    def plot_figure(self,png_fpath='output.png',figsize=(12,4.75),**kwargs):
        fig     = plt.figure(figsize=figsize)

        ax      = fig.add_subplot(111)
        result  = self.plot_ax(ax,**kwargs)

        ax.set_title(result['title'])

        cbar_pcoll = result.get('cbar_pcoll')
        if cbar_pcoll is not None:
            fig.colorbar(cbar_pcoll,label=result['cbar_label'])

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,
                xlim=(-85,-40), ylim=(25,55),
                vmin=-3, vmax=3,
                cmap='viridis',ylabel_fontdict={},**kwargs):

        fig = ax.get_figure()

        xx = self.lons
        yy = self.alts
        zz = self.Tpert

        cbar_pcoll = ax.pcolormesh(xx,yy,zz.T,vmin=vmin,vmax=vmax,**kwargs) 

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Altitude [km]')

        title = 'AIRS T\' on {!s} at {:0.0f}\N{DEGREE SIGN} Latitude'.format(self.date.strftime('%d %b %Y'),self.lat)

        result  = {}
        result['ax']            = ax
        result['cbar_pcoll']    = cbar_pcoll
        result['cbar_label']    = "T' [K]"
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
        a3dw = AIRS3DWorld(date)
        png_name    = 'AIRS3DWorld_{!s}.png'.format(date.strftime('%Y%m%d'))
        png_fpath   = os.path.join(output_dir,png_name)
        a3dw.plot_figure(png_fpath=png_fpath)

        a3dlp = AIRS3DLatProfile(date)
        png_name    = 'AIRS3DLatProfile_{!s}.png'.format(date.strftime('%Y%m%d'))
        png_fpath   = os.path.join(output_dir,png_name)
        a3dlp.plot_figure(png_fpath=png_fpath)

        print(png_fpath)
