#!/usr/bin/env python
"""
This class will generate polar maps with MERRA2 streamlines and AIRS gravity wave variances.
"""
import os
import shutil
import datetime

import numpy as np
import pandas as pd

import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt

import cartopy.crs as ccrs

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

class Merra2AirsMaps(object):
    def __init__(self,data_in='data/merra2_cips_airs_timeSeries/AIRS_GW_VARIANCE+MERRA2_20181101-20190430.nc'):
        self.load_data(data_in)
    
    def load_data(self,data_in):
        """
        ipdb> mca.ds
        <xarray.Dataset>
        Dimensions:                (ndays: 181, nlon_airs: 90, nlat_airs: 90,
                                    nlon_merra2: 144, nlat_merra2: 96)
        Dimensions without coordinates: ndays, nlon_airs, nlat_airs, nlon_merra2,
                                        nlat_merra2
        Data variables:
            DATE                   (ndays) int32 20181101 20181102 ... 20190429 20190430
            AIRS_LONS              (nlon_airs) float32 0.0 4.0 8.0 ... 348.0 352.0 356.0
            AIRS_LATS              (nlat_airs) float32 -89.0 -87.0 -85.0 ... 87.0 89.0
            MERRA2_LONS            (nlon_merra2) float32 0.0 2.5 5.0 ... 355.0 357.5
            MERRA2_LATS            (nlat_merra2) float32 -90.0 -88.11 ... 88.11 90.0
            AIRS_GW_VARIANCE       (ndays, nlat_airs, nlon_airs) float32 0.00125 ... ...
            MERRA2_STREAMFUNCTION  (ndays, nlat_merra2, nlon_merra2) float32 1.511e+0...
            MERRA2_WINDSPEED       (ndays, nlat_merra2, nlon_merra2) float32 20.45 .....
            MERRA2_VORTEX          (ndays, nlat_merra2, nlon_merra2) float32 1.0 ... 0.0
        Attributes:
            Description:  MERRA-2 wind speed and polar vortex edge valid at 800 K pot...
            Author:       V. Lynn Harvey, file was created using polar_nh_airs_varian...
            nlon_merra2:  Number of MERRA-2 Longitudes
            nlat_merra2:  Number of MERRA-2 Latitudes
            nlon_airs:    Number of AIRS Longitudes
            nlat_airs:    Number of AIRS Latitudes
            ndays:        Number of Days
        """
        ds              = xr.load_dataset(data_in)
        self.data_in    = data_in
        self.ds         = ds
        return ds

    def plot_figure(self,png_fpath='output.png',figsize=(16,8),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1, projection=ccrs.Orthographic(270,90))

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
#    def plot_ax(self,ax,vmin=-20,vmax=100,levels=11,cmap='jet',plot_cbar=True,ylabel_fontdict={},**kwargs):
    def plot_ax(self,ax,date=None,vmin=None,vmax=None,cmap='jet',**kwargs):
        fig     = ax.get_figure()

        ds      = self.ds

        date = datetime.datetime(2019,1,12)
        if date is None:
            date_inx = 0
        else:
            date_int = int(date.strftime('%Y%m%d'))
            date_inx = np.argmin(np.abs(ds['DATE'].values-date_int))


        # Plot AIRS Variances
        airs_lons       = ds['AIRS_LONS']
        airs_lats       = ds['AIRS_LATS']
        airs_gw_var     = ds['AIRS_GW_VARIANCE'].values[date_inx,:,:]

        
        mpbl    = ax.pcolormesh(airs_lons,airs_lats,airs_gw_var,transform=ccrs.PlateCarree(),cmap=cmap,
                    vmin=vmin,vmax=vmax)
        cbar    = fig.colorbar(mpbl,aspect=15,shrink=0.8)
        cbar.set_label('AIRS GW Variance [K^2]',fontdict={'weight':'bold','size':20})

        ax.coastlines(zorder=100)
        ax.gridlines()

#        dates   = [datetime.datetime.strptime(str(int(x)),'%Y%m%d') for x in ds['DATE']]
#        sDate   = min(dates)
#        eDate   = max(dates)
#
#
#        # Plot MERRA-2 Zonal Winds
#        zz  = np.array(ds['ZONAL_WIND'])
#
#        xx  = dates
#        yy  = np.nanmean(np.array(ds['GEOPOTENTIAL_HEIGHT']),axis=1)
#
#        # Keep only finite values of height.
#        tf  = np.isfinite(yy)
#        yy  = yy[tf]
#        zz  = zz[tf,:]
#
#        cbar_pcoll  = ax.contourf(xx,yy,zz,levels=levels,vmin=vmin,vmax=vmax,cmap=cmap)
#        cntr        = ax.contour(xx,yy,zz,levels=levels,colors='0.3')
#        ax.set_xlabel('UTC Date')
#        ax.set_ylabel('Geopot. Height [km]',fontdict=ylabel_fontdict)
#        ax.grid(False)
#
#        if plot_cbar:
#            lbl     = 'MERRA-2 Zonal Wind (m/s) (50\N{DEGREE SIGN} N)'
#            cbar    = fig.colorbar(cbar_pcoll,label=lbl)
#
#        # Plot CIPS GW Variance
#        ax1     = ax.twinx()
#
#        airs_cips_lw = 4
#        xx      = dates
#        yy      = np.array(ds['AIRS_GW_VARIANCE'])
#        lbl     = 'AIRS (30 km)'
#        ax1.plot(xx,yy,color='black',lw=airs_cips_lw,zorder=100,label=lbl)
#
#        xx      = dates
#        yy      = np.array(ds['CIPS_GW_VARIANCE'])
#        lbl     = 'CIPS (50 km)'
#        ax1.plot(xx,yy,color='fuchsia',lw=airs_cips_lw,zorder=100,label=lbl)
#        
#        lbl     = 'CIPS (%$^{2}$) and AIRS (K$^{2}$)\nGW Variance'
#        ax1.set_ylabel(lbl,fontdict=ylabel_fontdict)
#        ax1.grid(False)
#        ax1.set_ylim(0,0.25)
#
#        ax1.legend(loc='upper right',ncols=2,fontsize='large')
#
        result  = {}
#        result['cbar_pcoll']    = cbar_pcoll
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','merra2AirsMaps')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_fname   = 'merra2AirsMap.png'
    png_fpath   = os.path.join(output_dir,png_fname)

    mca = Merra2AirsMaps()
    mca.plot_figure(png_fpath=png_fpath)
