#!/usr/bin/env python
import os
import shutil
import datetime

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

import netCDF4 as nc

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

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

nc_fl       = os.path.join('data','merra2_geopotential','save_gph_10_1hpa_merra2_2010010100-2022073112.nc')
output_dir  = os.path.join('output','merra2_geopotential')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

ds = nc.Dataset(nc_fl)

#dict_keys(['DATE', 'LONGITUDE', 'LATITUDE', 'GPH_10HPA', 'GPH_1HPA'])

dates   = np.array(ds['DATE'])
datetimes = []
for date in dates:
    date_str = '{:d}'.format(date)
    dt       = datetime.datetime.strptime(date_str,'%Y%m%d%H')
    datetimes.append(dt)

#dates       = ds.variables['DATE']
lats        = np.array(ds.variables['LATITUDE'])
lons        = np.array(ds.variables['LONGITUDE'])
gph_10hpa   = np.array(ds.variables['GPH_10HPA'])
gph_1hpa    = np.array(ds.variables['GPH_1HPA'])

lat_min     = 65.
zscore      = True

lat_tf     = lats >= lat_min
lat_scale   = np.sin( np.deg2rad(90.-np.abs(lats)) )[lat_tf]

vals_0      = gph_10hpa[:,lat_tf,:]
vals_1      = gph_1hpa[:,lat_tf,:]

pv_index    = -np.sum( lat_scale[np.newaxis,:,np.newaxis] * (vals_1 - vals_0)**2 , (1,2))

df_0        = pd.DataFrame({'merra2_pvIndex':pv_index},index=datetimes)

if zscore:
    df_0 = (df_0 - df_0.mean())/df_0.std()

df_daily    = df_0.resample('1D').mean()

csv_fname   = 'merra2_pvIndex.csv'
csv_fpath   = os.path.join(output_dir,csv_fname)
with open(csv_fpath,'w') as fl:
    hdr = []
    hdr.append('# Frissell et al. (2016) Polar Vortex Index Calculated from MERRA2 Outputs')
    hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
    hdr.append('# See Section 2.4 in https://doi.org/10.1002/2015JA022168')
    hdr.append('# Generated on: {!s} UTC'.format(datetime.datetime.utcnow()))
    hdr.append('#')
    hdr.append('# MERRA2 Outputs Provided by V. Lynn Harvey')
    hdr.append('#')

    fl.write('\n'.join(hdr))
    fl.write('\n')

df_daily.to_csv(csv_fpath,mode='a')
print('Saved MERRA2 PV Index CSV File to: {!s}'.format(csv_fpath))

# Plot MSTID Index #############################################################
seasons = list_seasons()
for season in seasons:
    dt_0, dt_1  = season_to_datetime(season)

    tf          = np.logical_and(df_0.index >= dt_0, df_0.index < dt_1)
    df_season   = df_0[tf].copy()

    tf_daily    = np.logical_and(df_daily.index >= dt_0, df_daily.index < dt_1)
    df_season_daily = df_daily[tf_daily].copy()

    fig         = plt.figure(figsize=(15,4))
    ax          = fig.add_subplot(111)
    xx          = df_season.index
    yy          = df_season['merra2_pvIndex']
    ax.plot(xx,yy,lw=2,label='Raw')

    xx          = df_season_daily.index
    yy          = df_season_daily['merra2_pvIndex']
    ax.plot(xx,yy,marker='o',ls='',label='1 Day Resampled')

    ax.set_xlim(dt_0,dt_1)
    ax.set_xlabel('Time [UT]')
    ax.set_ylabel('$\zeta$')
    
    title       = 'MERRA2 PV Index\n{!s}'.format(season) 
    ax.set_title(title)

    ax.legend(loc='upper right')

    fig.tight_layout()
    fname       = 'merra2_pvIndex_{!s}.png'.format(season)
    fpath       = os.path.join(output_dir,fname)

    fig.savefig(fpath,bbox_inches='tight')

