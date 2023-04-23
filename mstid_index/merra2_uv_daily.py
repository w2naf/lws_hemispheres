#!/usr/bin/env python
import os
import shutil
import datetime

import numpy as np
import pandas as pd

import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt

import cartopy.crs as ccrs

import netCDF4 as nc

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

uv      = {}
uv['u'] = uvd   = {}
uvd['title']    = 'Zonal Winds U'
uvd['label']    = 'U [m/s]'
uv['v'] = uvd   = {}
uvd['title']    = 'Meridional Winds V'
uvd['label']    = 'V [m/s]'

levels	= {}
levels['1P5HPA'] 	= lvl = {}
lvl['label'] 		= '1.5 hPa (45 km alt)'

levels['3HPA'] 		= lvl = {}
lvl['label'] 		= '3 hPa (40 km alt)'

levels['5HPA'] 		= lvl = {}
lvl['label'] 		= '5 hPa (35 km alt)'

levels['10HPA'] 	= lvl = {}
lvl['label'] 		= '10 hPa (30 km alt)'

levels['20HPA'] 	= lvl = {}
lvl['label'] 		= '20 hPa (25 km alt)'

def prep_dir(output_dir,clear=False):
    if os.path.exists(output_dir) and (clear is True):
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def list_seasons(yr_0=2010,yr_1=2022):
    """
    Give a list of the string codes for the default seasons to be analyzed.

    Season codes are in the form of '20101101_20110501'
    """
    yr = yr_0
    seasons = []
    while yr < yr_1:
        dt_0 = datetime.datetime(yr,11,1)
        dt_1 = datetime.datetime(yr+1,5,1)

        dt_0_str    = dt_0.strftime('%Y%m%d')
        dt_1_str    = dt_1.strftime('%Y%m%d')
        season      = '{!s}_{!s}'.format(dt_0_str,dt_1_str)
        seasons.append(season)
        yr += 1

    return seasons

def season_to_datetime(season):
    str_0, str_1 = season.split('_')
    sDate   = datetime.datetime.strptime(str_0,'%Y%m%d')
    eDate   = datetime.datetime.strptime(str_1,'%Y%m%d')
    return (sDate,eDate)

def load_uv(nc_fl):
    """
    #Below are links to two netcdf files. One contains zonal winds, the other contains meridional winds. Each file contains winds at 5 pressure levels (20, 10, 5, 3, 1.5 hPa) that roughly correspond to (25, 30, 35, 40, and 45 km).
    #Each file has winds twice daily since 2010. Please let me know if you have any issues accessing or working with these data files.
    #
    #https://www.dropbox.com/s/vujlzxf82icm2cl/save_u_5levs_merra2_2010010100-2022073112.nc?dl=0
    #https://www.dropbox.com/s/giaon2zpvhvvkn3/save_v_5levs_merra2_2010010100-2022073112.nc?dl=0

    #ipdb> ds.variables.keys()
    #dict_keys(['DATE', 'LONGITUDE', 'LATITUDE', 'U_20HPA', 'U_10HPA', 'U_5HPA', 'U_3HPA', 'U_1P5HPA'])
    """
    #ds  = nc.Dataset(nc_fl)
    ds  = xr.open_dataset(nc_fl)

    dates   = np.array(ds['DATE'])
    datetimes = []
    for date in dates:
        date_str = '{:d}'.format(date)
        dt       = datetime.datetime.strptime(date_str,'%Y%m%d%H')
        datetimes.append(dt)

    coords  = {}
    coords['noutput']   = datetimes
    coords['nlat']      = ds['LATITUDE']
    coords['nlon']      = ds['LONGITUDE']

    ds = ds.assign_coords(coords)

    del ds['DATE']
    del ds['LATITUDE']
    del ds['LONGITUDE']

    return ds

def plot_dailies(dss,sDate,eDate,output_dir='.'):
    hours   = [0, 12]

    nrows   = 5
    ncols   = 4

    figsize = 10*np.array([nrows,ncols])

    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1] + datetime.timedelta(days=1))

    for date in dates:
        png_name    = '{!s}_uv.png'.format(date.strftime('%Y%m%d'))
        png_path    = os.path.join(output_dir,png_name)

        fig = plt.figure(figsize=figsize)

        for hr_inx,hour in enumerate(hours):

            this_time   = date + datetime.timedelta(hours = hour)

            for uv_inx,(uv_key, uvd) in enumerate(uv.items()):
                ds  = dss[uv_key]

                col = 2*hr_inx + uv_inx

                for lvl_inx,(level,lvld) in enumerate(levels.items()):

                    param   = '{!s}_{!s}'.format(uv_key.upper(),level)

                    dt_inx  = np.where(ds['noutput'] == np.datetime64(this_time))[0][0]
                    dst     = ds[param]
                    frame   = dst.values[dt_inx,:,:]

                    lats    = dst.coords['nlat'].values
                    lons    = dst.coords['nlon'].values

#                    if level == '3HPA':
#                        import ipdb; ipdb.set_trace()



                    plt_inx = lvl_inx*ncols + col + 1

                    ax      = fig.add_subplot(nrows,ncols,plt_inx, projection=ccrs.Orthographic(270,90))

                    mpbl    = ax.pcolormesh(lons,lats,frame,transform=ccrs.PlateCarree(),cmap='bwr')
                    fig.colorbar(mpbl,label=uvd.get('label',uv_key.upper()))


                    ax.coastlines(zorder=100)
                    ax.gridlines()

                    title   = []
                    title.append(this_time.strftime('%Y %b %d %H%M UT'))
                    title.append(uvd.get('title',uv_key.upper()))
                    title.append(lvld.get('label',level))
                    ax.set_title('\n'.join(title))

        fig.tight_layout()
        fig.savefig(png_path,bbox_inches='tight')
        print(png_path)
        plt.close(fig)
                    
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    output_dir  = prep_dir(os.path.join('output','merra2_uv'))

    dss = {}

    for key in uv.keys():
        # Load Zonal and Meridional Winds
        nc_fl       = os.path.join('data','merra2','save_{!s}_5levs_merra2_2010010100-2022073112.nc'.format(key))
        dss[key]    = load_uv(nc_fl)

#    sDate   = datetime.datetime(2017,1,1)
#    eDate   = datetime.datetime(2017,1,5)

    sDate   = datetime.datetime(2016,11,1)
    eDate   = datetime.datetime(2017,5,1)

    dailies_dir = prep_dir(os.path.join(output_dir,'dailies'),clear=True)
    plot_dailies(dss,sDate,eDate,output_dir=dailies_dir)

    import ipdb; ipdb.set_trace()
