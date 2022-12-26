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

import multiprocessing

mpl.rcParams['font.size']      = 18
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

vscl    = 125
uv      = {}
uv['u'] = uvd   = {}
uvd['title']    = 'Zonal Winds $U$'
uvd['label']    = 'U [m/s]'
uvd['vmin']     = -vscl
uvd['vmax']     =  vscl

uv['v'] = uvd   = {}
uvd['title']    = 'Meridional Winds $V$'
uvd['label']    = 'V [m/s]'
uvd['vmin']     = -vscl
uvd['vmax']     =  vscl

uv['u_h'] = uvd   = {}
uvd['title']    = 'Horizontal Winds $U_H$'
uvd['label']    = '$U_H$ [m/s]'
uvd['vmin']     = -vscl
uvd['vmax']     =  vscl

uv['dUdz'] = uvd   = {}
uvd['title']    = '$dU/dz$'
uvd['label']    = 'dU/dz [$s^{-1}$]'
uvd['vmin']     = -0.005
uvd['vmax']     =  0.005

uv['dVdz'] = uvd   = {}
uvd['title']    = '$dV/dz$'
uvd['label']    = 'dV/dz [$s^{-1}$]'
uvd['vmin']     = -0.0025
uvd['vmax']     =  0.0025

uv['dUVpdz']    = uvd   = {}
uvd['title']    = '$[(dU/dz)^2 + (dV/dz)^2]^p$'
uvd['label']    = '[$s^{-1}$]'
uvd['vmin']     = -0.0075
uvd['vmax']     =  0.0075

levels	= {}
levels[1.5] 	    = lvl = {}
lvl['km']           = 45
lvl['label'] 		= '1.5 hPa (45 km alt)'
lvl['2l_label'] 	= '1.5 hPa\n(45 km)'

levels[3] 		    = lvl = {}
lvl['km']           = 40
lvl['label'] 		= '3 hPa (40 km alt)'
lvl['2l_label']     = '3 hPa\n(40 km)'

levels[5]           = lvl = {}
lvl['km']           = 35
lvl['label'] 		= '5 hPa (35 km alt)'
lvl['2l_label']     = '5 hPa\n(35 km)'

levels[10] 	        = lvl = {}
lvl['km']           = 30
lvl['label'] 		= '10 hPa (30 km alt)'
lvl['2l_label']     = '10 hPa\n(30 km)'

levels[20] 	        = lvl = {}
lvl['km']           = 25
lvl['label'] 		= '20 hPa (25 km alt)'
lvl['2l_label']     = '20 hPa\n(25 km)'

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
#    ds  = nc.Dataset(nc_fl)
    ds  = xr.load_dataset(nc_fl)

    dates   = np.array(ds['DATE'])
    datetimes = []
    for date in dates:
        date_str = '{:d}'.format(date)
        dt       = datetime.datetime.strptime(date_str,'%Y%m%d%H')
        datetimes.append(dt)

    coords  = {}
    coords['ut']        = datetimes
    coords['lats']      = ds['LATITUDE'].values
    coords['lons']      = ds['LONGITUDE'].values

    ds = ds.assign_coords(coords)

    del ds['DATE']
    del ds['LATITUDE']
    del ds['LONGITUDE']

    # Change level strings into hPa floats.
    levels  = list(ds.data_vars)
    hPas    = {}
    for level in levels:
        hPa = float(level.split('_')[1].rstrip('HPA').replace('P','.'))
        hPas[level] = hPa
    ds      = ds.rename(hPas)
    ds      = ds.to_array()
    
    # Create a fresh XArray DataArray in order to eliminate uneeded/confusing exta data.
    coords  = {}
    coords['ut']    = ds['ut'].values
    coords['lats']  = ds['lats'].values
    coords['lons']  = ds['lons'].values
    coords['hPa']   = list(hPas.values())
    ds              = xr.DataArray(ds.values,coords=coords,dims=('hPa','ut','lats','lons'),attrs=ds.attrs)

    return ds

def ddz(ds,dz=5000):
    """
    Calculate vertical wind shear and return as a XArray DataArray.
    
    ds:     Horizontal wind DataArray
    dz:     Distance between horizontal wind levels [m]
    """
    result  = np.diff(ds,axis=0)/dz

    coords  = {}
    coords['ut']    = ds['ut'].values
    coords['lats']  = ds['lats'].values
    coords['lons']  = ds['lons'].values
    coords['hPa']   = ds['hPa'].values[1:]
    dXdz            = xr.DataArray(result,coords=coords,dims=('hPa','ut','lats','lons'))
    return dXdz

def dUVpdz(dss,p=0.5):
    """
    Calculate ( (dU/dz)**2 + (dV/dz)**2 ) ** p and 
    return as a XArray DataArray.
    
    dss:    Dictionary containing dUdz and dVdz DataArrays.
    p:      Power paramter
    """

    dUdz    = dss['dUdz']
    dVdz    = dss['dVdz']
    dUVpdz  = ( (dUdz)**2 + (dVdz)**2 ) ** p
    dUVpdz.attrs['p']   = p

    return dUVpdz

def plot_dailies_dct(rd):
    return plot_dailies(**rd)

def plot_dailies(dss,date,params=['u','v','u_h','dUdz','dVdz','dUVpdz'],
        output_dir='.'):

#    params=['dUdz','dVdz']

    nrows   = 5
    ncols   = len(params)

    figscl  = 5.
    figsize = (1.2*figscl*ncols+5,figscl*nrows)

    png_name    = '{!s}_uv.png'.format(date.strftime('%Y%m%d_%H%M'))
    png_path    = os.path.join(output_dir,png_name)

    fig = plt.figure(figsize=figsize)
    for uv_inx,uv_key in enumerate(params):
        col = uv_inx

        uvd = uv.get(uv_key,{})
        ds  = dss[uv_key]
        
        vmin    = uvd.get('vmin')
        vmax    = uvd.get('vmax')

        if (vmin is None) and (vmax is None):
            vmean   = ds.mean()
            vstd    = ds.std()
            
            std_scl = 3
            vmn     = np.abs(vmean - std_scl*vstd)
            vmx     = np.abs(vmean + std_scl*vstd)

            vmax    = np.max([vmn,vmx])
            vmin    = -vmax

        for lvl_inx,(level,lvld) in enumerate(levels.items()):
            row     = lvl_inx

            if level not in ds['hPa']:
                continue

            dt_inx  = np.where(ds['ut'] == np.datetime64(date))[0][0]
            frame   = ds.loc[{'hPa':level,'ut':np.datetime64(date)}]

            lats    = frame.coords['lats'].values
            lons    = frame.coords['lons'].values

            plt_inx = row*ncols + col + 1

            ax      = fig.add_subplot(nrows,ncols,plt_inx, projection=ccrs.Orthographic(270,90))
            mpbl    = ax.pcolormesh(lons,lats,frame,transform=ccrs.PlateCarree(),cmap='bwr',
                        vmin=vmin,vmax=vmax)

            cbar    = fig.colorbar(mpbl,aspect=15,shrink=0.8)
            cbar.set_label(uvd.get('label',uv_key.upper()),fontdict={'weight':'bold','size':20})

            ax.coastlines(zorder=100)
            ax.gridlines()

            label_fontdict  = {'weight':'bold','size':28}
            if col == 0:
                txt = lvld.get('2l_label',level)
                ax.text(-0.25,0.5,txt,ha='center',va='center',fontdict=label_fontdict,transform=ax.transAxes)

            if row == 0:
                txt = uvd.get('title',uv_key.upper())
                ax.text(0.5,1.1,txt,ha='center',va='center',fontdict=label_fontdict,transform=ax.transAxes)

    fig.text(0.5,1.01,date.strftime('%Y %b %d %H%M UT'),ha='center',fontdict={'weight':'bold','size':36})

    fig.tight_layout()
    fig.savefig(png_path,bbox_inches='tight')
    print(png_path)
    plt.close(fig)

if __name__ == '__main__':
    multiproc   = False
    ncpus       = multiprocessing.cpu_count()

    output_dir  = prep_dir(os.path.join('output','merra2_uv'))

    sDate   = datetime.datetime(2017,1,1)
    eDate   = datetime.datetime(2017,1,2)

#    sDate   = datetime.datetime(2016,11,1)
#    eDate   = datetime.datetime(2017,5,1)

    dt_hr   = 12

    dss = {}
    for key in ['u','v']:
        # Load Zonal and Meridional Winds
        nc_fl       = os.path.join('data','merra2','save_{!s}_5levs_merra2_2010010100-2022073112.nc'.format(key))
        print('LOADING: {!s}'.format(nc_fl))
        dss[key]    = load_uv(nc_fl)

    print('Computing u_h...')
    dss['u_h']      = np.sqrt(dss['u']**2 + dss['v']**2)

    for key in ['u','v']:
        print('Computing d{!s}/dz...'.format(key.upper()))
        dz_key      = 'd{!s}dz'.format(key.upper())
        dss[dz_key] = ddz(dss[key])

    print('Computing [(dU/dz)**2 + (dV/dz)**2]**p...')
    dss['dUVpdz']   = dUVpdz(dss,p=0.5)

    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1] + datetime.timedelta(hours=dt_hr))

    dailies_dir = prep_dir(os.path.join(output_dir,'dailies'),clear=True)
    run_dcts    = []
    for date in dates:
        rd  = {}
        rd['dss']           = dss
        rd['date']          = date
        rd['output_dir']    = dailies_dir
        run_dcts.append(rd)

    if multiproc:
        print('Plotting using multiprocessing...')
        with multiprocessing.Pool(ncpus) as pool:
            pool.map(plot_dailies_dct,run_dcts)
    else:
        print('Plotting using for loops...')
        for rd in run_dcts:
            plot_dailies_dct(rd)

    import ipdb; ipdb.set_trace()
