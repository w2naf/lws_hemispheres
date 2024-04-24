#!/usr/bin/env python
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

class Hovmoller(object):
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        data_dir    = os.path.join('data','hovmoller_airs_merra2')
        sav_name    = 'xt_airs_variance+merra2_vortex_speed_20181101-20190501_50.sav'
        sav_path    = os.path.join(data_dir,sav_name)
        sav_data    = sp.io.readsav(sav_path)

        self.sav_data   = sav_data

        return sav_data

    def plot_figure(self,png_fpath='hovmoller.png',figsize=(8,12),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax          = fig.add_subplot(1,1,1)
        result      = self.plot_ax(ax,**kwargs)

#        cbar        = fig.colorbar(result['cbar_pcoll'],aspect=15,shrink=0.8,location='bottom')
        cbar        = fig.colorbar(result['cbar_pcoll'],location='bottom',pad=0.05)
        cbar.set_label(result['cbar_label'],fontdict={'weight':'bold','size':20})

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,vmin=0.,vmax=1,cmap='IDL',levels=None,
            merra2_windspeed_kw={},
            ylim = (datetime.datetime(2019,3,1),datetime.datetime(2018,12,1)),
            **kwargs):
        """
        merra2_windspeed_kw: Keywords passed to ax.contour() for merra2_windspeed.
        """

        fig     = ax.get_figure()

        if cmap == 'IDL':
            cmap    = rainbowCmap()

        sav_data = self.sav_data

        airs_lons   = sav_data['xnew_wrap']
        dates       = [datetime.datetime.strptime(x.decode(), '%Y%m%d') for x in sav_data['sdate_all']]

        zz      = sav_data['variance_lineplot_avg'].copy()
#        tf      = ~np.isfinite(zz)
#        zz[tf]  = 0.
#        zz[zz < 0] = 0.
        zz[zz < 0.0130] = np.nan

        if levels is None:
            levels = np.arange(vmin,vmax+0.005,0.005)

        cbar_pcoll  = ax.contourf(airs_lons,dates,zz,
                cmap=cmap,levels=levels,extend='max')
        cbar_label  = 'AIRS GW Variance [K$^{2}$]'

        # MERRA2 Wind Speed ############################################################ 
        # Default contour and coastline style parameters.
        # Tuned for an IDL-colormap style plot.

        m2ws = {} # MERRA2 Wind Stream Parameters
#        m2ws['levels']      = [-10000,40,60]
#        m2ws['colors']      = ['white','orange','red']

        m2ws['levels']      = [-20,0,40,60]
        m2ws['colors']      = ['0.5','0.5','orange','red']

#        m2ws['levels']              = [-20,0,20,50]
#        m2ws['colors']              = ['0.5','0.5','orange','red']
        m2ws['linewidths']          = 6
        m2ws['zorder']              = 1000
        m2ws.update(merra2_windspeed_kw)

        merra2_lons             = sav_data['alon_save']
        merra2_windspeed        = sav_data['u_lineplot_avg']

        WS = ax.contour(merra2_lons,dates,merra2_windspeed,**m2ws)

		# Override the linestyles based on the levels.
        for line, lvl in zip(WS.collections, WS.levels):
            if lvl < 0:
                line.set_linestyle('--')
            elif lvl == 0:
                line.set_linestyle(':')
            else:
                # Optional; this is the default.
                line.set_linestyle('-')

        try:
            ax.clabel(WS,inline=True)
        except:
            pass

        ax.set_ylim(ylim)

        ax.set_xlabel('Longitude')

        result  = {}
        result['cbar_pcoll']    = cbar_pcoll
        result['cbar_label']    = cbar_label
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','hovmoller')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hov = Hovmoller()

    png_fname   = 'hovmoller.png'
    png_fpath   = os.path.join(output_dir,png_fname)
    hov.plot_figure(png_fpath=png_fpath)
    print(png_fpath)
