#!/usr/bin/env python
"""
This class will generate polar maps with MERRA2 streamlines and AIRS gravity wave variances.
"""
import os
import shutil
import datetime

import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage

import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt

from PIL import Image

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

def to_cmap(my_cdict,name='CustomCMAP',vmin=-2.,vmax=1.):
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    
    red   = []
    green = []
    blue  = []
    
    keys = list(my_cdict.keys())
    keys.sort()
    
    for x in keys:
        r,g,b, = my_cdict[x]
        x = norm(x)
        r = r/255.
        g = g/255.
        b = b/255.
        red.append(   (x, r, r))
        green.append( (x, g, g))
        blue.append(  (x, b, b))
    cdict = {'red'   : tuple(red),
             'green' : tuple(green),
             'blue'  : tuple(blue)}
    cmap  = mpl.colors.LinearSegmentedColormap(name, cdict)
    return cmap

def rainbowCmap():
    #Color Map
    img = Image.open('rainbow.png')
    data_2=np.array(img)

    vmin, vmax = (-2, 1)
    inxs = (np.linspace(vmin,vmax,data_2.shape[1])).tolist()
    vals = [tuple(x) for x in (data_2[0,:,:]).tolist()]

    rgb_dict = {}
    for inx,val in zip(inxs,vals):
        rgb_dict[inx] = val

    cmap = to_cmap(rgb_dict,vmin=vmin,vmax=vmax)
    return cmap

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

    def _date2inx(self,date=None):
        """
        Check to see if the date is in the data set.
        Return the closest date in the dataset as a datetime object and the index.
        """
        ds = self.ds

        if date is None:
            date_inx = 0
        else:
            date_int = int(date.strftime('%Y%m%d'))
            date_inx = np.argmin(np.abs(ds['DATE'].values-date_int))

        date_0 = datetime.datetime.strptime(str(ds['DATE'].values[date_inx]),'%Y%m%d')

        return (date_0, date_inx)


    def get_dates(self,sDate=None,eDate=None):
        """
        Return a list of all of the dates available in the dataset.
        """
        ds      = self.ds

        dates   = []
        for date_nr in ds['DATE'].values:
            date = datetime.datetime.strptime(str(date_nr),'%Y%m%d')

            if sDate is not None:
                if date < sDate:
                    continue

            if eDate is not None:
                if date >= eDate:
                    continue

            dates.append(date)

        return dates

    def plot_figure(self,date=None,png_fpath='output.png',figsize=(16,8),**kwargs):

        date_0, date_inx = self._date2inx(date)

        fig     = plt.figure(figsize=figsize)
#        ax      = fig.add_subplot(1,1,1, projection=ccrs.Orthographic(270,90))
        ax      = fig.add_subplot(1,1,1, projection=ccrs.Orthographic(0,90))

        result  = self.plot_ax(ax,date=date,**kwargs)

        date_str    = date_0.strftime('%Y %b %d')
        ax.set_title(date_str)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,date=None,vmin=0.,vmax=0.8,cmap='jet',**kwargs):
        date_0, date_inx = self._date2inx(date)
        ds      = self.ds

        fig     = ax.get_figure()

        # Plot AIRS Variances
        airs_lons       = ds['AIRS_LONS']
        airs_lats       = ds['AIRS_LATS']
        airs_gw_var     = ds['AIRS_GW_VARIANCE'].values[date_inx,:,:]
        
        cmap    = rainbowCmap()
        zz      = airs_gw_var.copy()
        tf      = ~np.isfinite(airs_gw_var)
        zz[tf]  = 0.
        zz[zz < 0] = 0.

        cyc_zz, cyc_lons = add_cyclic_point(zz,coord=airs_lons)
        mpbl    = ax.contourf(cyc_lons,airs_lats,cyc_zz,transform=ccrs.PlateCarree(),
                cmap=cmap,levels=np.arange(vmin,vmax+0.005,0.005),extend='max')
        cbar    = fig.colorbar(mpbl,aspect=15,shrink=0.8)
        cbar.set_label('AIRS GW Variance [K^2]',fontdict={'weight':'bold','size':20})

        merra2_lons             = ds['MERRA2_LONS']
        merra2_lats             = ds['MERRA2_LATS']
        merra2_streamfunction   = ds['MERRA2_STREAMFUNCTION'].values[date_inx,:,:]
        merra2_windspeed        = ds['MERRA2_WINDSPEED'].values[date_inx,:,:]
        merra2_vortex           = ds['MERRA2_VORTEX'].values[date_inx,:,:]


        ax.contour(merra2_lons,merra2_lats,merra2_streamfunction,colors='white',transform=ccrs.PlateCarree())

        cyc_zz, cyc_lons = add_cyclic_point(merra2_vortex,coord=merra2_lons)
        ax.contour(cyc_lons,merra2_lats,cyc_zz,colors='white',linewidths=2,transform=ccrs.PlateCarree())

        ax.contour(merra2_lons,merra2_lats,merra2_windspeed,levels=[50,70,90],colors=['yellow','orange','red'],transform=ccrs.PlateCarree())
        
        ax.coastlines(zorder=100,color='0.45')
        ax.gridlines()

        result  = {}
#        result['cbar_pcoll']    = cbar_pcoll
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','merra2AirsMaps')
    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sDate       = datetime.datetime(2018,11,1)
    eDate       = datetime.datetime(2019,5,1)

    mca = Merra2AirsMaps()

    dates       = mca.get_dates(sDate,eDate)
    for date in dates:
        date_str    = date.strftime('%Y%m%d')
        png_fname   = 'merra2AirsMap_{!s}.png'.format(date_str)
        png_fpath   = os.path.join(output_dir,png_fname)

        mca.plot_figure(date=date,png_fpath=png_fpath)
        print(png_fpath)
