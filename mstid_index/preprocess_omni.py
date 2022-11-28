#!/usr/bin/env python
"""
This script pre-processes the OMNI data downloaded from https://omniweb.gsfc.nasa.gov/form/dx1.html
"""

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

def generate_radar_dict():
    rad_list = []
    rad_list.append(('bks', 39.6, -81.1))
    rad_list.append(('wal', 41.8, -72.2))
    rad_list.append(('fhe', 42.5, -95.0))
    rad_list.append(('fhw', 43.3, -102.7))
    rad_list.append(('cve', 46.4, -114.6))
    rad_list.append(('cvw', 47.9, -123.4))
    rad_list.append(('gbr', 58.4, -59.9))
    rad_list.append(('kap', 55.5, -85.0))
    rad_list.append(('sas', 56.1, -103.8))
    rad_list.append(('pgr', 58.0, -123.5))

#    rad_list.append(('sto', 63.86, -21.031))
#    rad_list.append(('pyk', 63.77, -20.54))
#    rad_list.append(('han', 62.32,  26.61))

    radar_dict = {}
    for radar,lat,lon in rad_list:
        tmp                 = {}
        tmp['lat']          = lat
        tmp['lon']          = lon
        radar_dict[radar]   = tmp

    return radar_dict

data_dir    = os.path.join('data','omni')
output_dir  = os.path.join(data_dir,'preprocessed')
fpath       = os.path.join(data_dir,'omni2_xkfmubCupo.lst')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

cols = []
cols.append('year')
cols.append('doy')
cols.append('hour')
cols.append('OMNI_R_Sunspot_Number')
cols.append('OMNI_Dst')
cols.append('OMNI_F10.7')
cols.append('OMNI_AE')


def dateparse(row):
    year = int(row['year'])
    doy  = int(row['doy'])
    hour = int(row['hour'])

    dt = datetime.datetime(year,1,1) + datetime.timedelta(days=doy-1) + datetime.timedelta(hours=hour)
    return dt

df_0        = pd.read_table(fpath,names=cols,sep='\s+')
df_0.index  = df_0.apply(dateparse,axis=1).values
del df_0['year']
del df_0['doy']
del df_0['hour']

radars  = generate_radar_dict()
params  = []
params.append('OMNI_R_Sunspot_Number')
params.append('OMNI_Dst')
params.append('OMNI_F10.7')
params.append('OMNI_AE')

# Set bad values to NaN.
bad = {}
bad['OMNI_R_Sunspot_Number']    = 999
bad['OMNI_AE']                  = 9999
bad['OMNI_F10.7']               = 999.9
for col,val in bad.items():
    tf = df_0[col] == val
    df_0.loc[tf,col] = np.nan

df_1 = df_0.resample('2H').mean()

sYear   = 2010
eYear   = 2022
thisYr  = sYear
seasons = []
while thisYr < eYear:
    season  = '{:d}1101_{:d}0501'.format(thisYr,thisYr+1)
    seasons.append(season)
    thisYr  += 1

for season in seasons:
    for radar in radars.keys():
        fname = 'OMNI_{!s}_{!s}'.format(season,radar)


        dt_0    = datetime.datetime.strptime(season[:8],'%Y%m%d')
        dt_1    = datetime.datetime.strptime(season[9:],'%Y%m%d')

        tf      = np.logical_and(df_0.index >= dt_0, df_0.index < dt_1) 
        dft_0   = df_0[tf]

        tf      = np.logical_and(df_1.index >= dt_0, df_1.index < dt_1) 
        dft_1   = df_1[tf]

        attrs   = {}
        attrs['radar']  = radar
        attrs['lat']    = radars[radar]['lat']
        attrs['lon']    = radars[radar]['lon']
        attrs['season'] = season

        csv_path = os.path.join(output_dir,fname+'.csv')
        with open(csv_path,'w') as fl:
            hdr = []
            hdr.append('# NASA OMNI Data for Comparison with SuperDARN MSTID Index')
            hdr.append('# Data re-processed by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
            hdr.append('#')
            for attr_key,attr in attrs.items():
                hdr.append('# {!s}: {!s}'.format(attr_key,attr))
            hdr.append('#')

            fl.write('\n'.join(hdr))
            fl.write('\n')
            
            csv_cols = ['datetime_ut'] + list(dft_1.keys())
            fl.write(','.join(csv_cols))
            fl.write('\n')
        dft_1.to_csv(csv_path,mode='a',header=False)

        nc_path         = os.path.join(output_dir,fname+'.nc')
        dsr             = dft_1.to_xarray()
        dsr             = dsr.assign_coords({'radar':radar})
        dsr.attrs       = attrs
        dsr.to_netcdf(nc_path)

        print(csv_path)
        print(nc_path)

        nrows   = len(params)
        fig     = plt.figure(figsize=(15,8))

        for axn,col in enumerate(params):
            ax  = fig.add_subplot(nrows,1,axn+1)

            xx  = dft_0.index
            yy  = dft_0[col]
            ax.plot(xx,yy,label='Original',lw=2)
            
            xx  = dft_1.index
            yy  = dft_1[col]
            ax.plot(xx,yy,label='Resampled')

            ax.set_xlabel('Date [UT]')
            ax.set_ylabel(col)

            ax.legend(loc='upper right',fontsize='small')

        fig.tight_layout()
        png_path = os.path.join(output_dir,fname+'.png')
        fig.savefig(png_path,bbox_inches='tight')
        print(png_path)
        plt.close(fig)

breakpoint()
