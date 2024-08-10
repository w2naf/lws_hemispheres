#!/usr/bin/env python
"""
This class will plot a day of amateur radio RBN/PSKReporter/WSPRNet data time series with edge
detection and spot maps.
"""
import os
import datetime
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

# Region Dictionary
regions = {}
tmp     = {}
tmp['lon_lim']  = (-180.,180.)
tmp['lat_lim']  = ( -90., 90.)
regions['World']    = tmp

tmp     = {}
tmp['lon_lim']  = (-55.,-10)
tmp['lat_lim']  = (  40., 65.)
regions['Atlantic_Ocean']   = tmp

tmp     = {}
tmp['lon_lim']  = (-130.,-60.)
tmp['lat_lim']  = (  20., 55.)
regions['US']   = tmp

def get_bins(lim, bin_size):
    """ Helper function to split a limit into bins of the proper size """
    bins    = np.arange(lim[0], lim[1]+1*bin_size, bin_size)
    return bins

def fmt_xaxis(ax,xlim=None,label=True):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)

class HamSpotPlot(object):
    def __init__(self,date=datetime.datetime(2018,12,15),data_dir=None):
        if data_dir is None:
            data_dir    = os.path.join('data','lstid_ham')
        self.data_dir   = data_dir
        self.date       = date
        self.load_geo_data()
        self.load_edge_data()
    
    def load_edge_data(self):
        date    = self.date

        # EDGE DETECT DATA #####################
        # 20181215_edgeDetect.pkl
        date_str        = date.strftime('%Y%m%d')
        fname           = f'{date_str}_edgeDetect.pkl'
        fpath           = os.path.join(self.data_dir,fname)
        self.edge_fpath = fpath
        with open(fpath,'rb') as pkl:
            edge_data   = pickle.load(pkl)

        self.edge_data  = edge_data
        return edge_data

    def load_geo_data(self,dlat=1,dlon=1):
        # hamSpot_geo_2018_12_15.csv.bz2
        date_str        = date.strftime('%Y_%m_%d')
        fname           = f'hamSpot_geo_{date_str}.csv.bz2'
        fpath           = os.path.join(self.data_dir,fname)
        self.geo_fpath  = fpath
        geo_df          = pd.read_csv(fpath)

        xkey    = 'lon'
        ykey    = 'lat'

        xlim    = (-180,180)
        ylim    = ( -90, 90)

        xbins   = get_bins(xlim,dlon)
        ybins   = get_bins(ylim,dlat)

        xvals   = geo_df[xkey].values
        yvals   = geo_df[ykey].values
        hist, xb, yb = np.histogram2d(xvals,yvals, bins=[xbins, ybins])

        crds    = {}
        crds[xkey]          = xb[:-1]
        crds[ykey]          = yb[:-1]

        attrs   = {}
        attrs['xkey']   = xkey
        attrs['ykey']   = ykey
        
        geo_hist    = xr.DataArray(hist,crds,attrs=attrs,dims=[xkey,ykey])
        self.geo_df     = geo_df
        self.geo_hist   = geo_hist

    def plot_figure(self,png_fpath='output.png',figsize=(16,10),**kwargs):

        fig     = plt.figure(figsize=figsize)

        self.plot_map_ax(fig,subplot=(2,1,1))

        ax      = fig.add_subplot(2,1,2)
        result  = self.plot_timeSeries_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)

    def plot_map_ax(self,fig,subplot=None,
            plot_region='US'):
        ax      = fig.add_subplot(*subplot,projection=ccrs.PlateCarree())

        # Plot Map #############################
        ax.coastlines(zorder=10,color='w')
        map_data        = self.geo_hist
        map_data.name   = '14 MHz Midpoints'

        tf          = map_data < 1
        map_n       = int(np.sum(map_data))
        map_data    = np.log10(map_data)
        map_data.values[tf] = 0
        map_data.name   = 'log({})'.format(map_data.name)
        map_data.plot.contourf(x=map_data.attrs['xkey'],y=map_data.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
        ax.set_title('')
        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}
        ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(map_n),
                ha='center',transform=ax.transAxes,fontdict=fdict)

        if plot_region is not None:
            rgn         = regions.get(plot_region)
            lat_lim     = rgn.get('lat_lim')
            lon_lim     = rgn.get('lon_lim')

            ax.set_xlim(lon_lim)
            ax.set_ylim(lat_lim)
    
    def plot_timeSeries_ax(self,ax,cb_pad=0.125,plot_fit=True):
        fig             = ax.get_figure()

        result_dct      = self.edge_data
        md              = result_dct.get('metaData')
        date            = md.get('date')
        xlim            = md.get('xlim')
        winlim          = md.get('winlim')
        fitWinLim       = md.get('fitWinLim')
        lstid_criteria  = md.get('lstid_criteria')

        arr             = result_dct.get('spotArr')
        med_lines       = result_dct.get('med_lines')
        edge_0          = result_dct.get('000_detectedEdge')
        edge_1          = result_dct.get('001_windowLimits')
        sg_edge         = result_dct.get('003_sgEdge')
        sin_fit         = result_dct.get('sin_fit')
        poly_fit        = result_dct.get('poly_fit')
        p0_sin_fit      = result_dct.get('p0_sin_fit')
        p0_poly_fit     = result_dct.get('p0_poly_fit')
        stability       = result_dct.get('stability')
        data_detrend    = result_dct.get('data_detrend')

        ranges_km       = arr.coords['ranges_km']
        arr_times       = [pd.Timestamp(x) for x in arr.coords['datetimes'].values]
        Ts              = np.mean(np.diff(arr_times)) # Sampling Period

        mpbl = ax.pcolormesh(arr_times,ranges_km,arr,cmap='plasma')
        plt.colorbar(mpbl,aspect=10,pad=cb_pad,label='14 MHz Ham Radio Data')
        if not plot_fit:
            ax.set_title(f'| {date} |')
        else:
            ed0_line    = ax.plot(arr_times,edge_0,lw=2,label='Detected Edge')

            if p0_sin_fit != {}:
                ax.plot(sin_fit.index,sin_fit+poly_fit,label='Sin Fit',color='white',lw=3,ls='--')

            ax2 = ax.twinx()
            ax2.plot(stability.index,stability,lw=2,color='0.5')
            ax2.grid(False)
            ax2.set_ylabel('Edge Coef. of Variation\n(Grey Line)')

            for wl in winlim:
                ax.axvline(wl,color='0.8',ls='--',lw=2)

            for wl in fitWinLim:
                ax.axvline(wl,color='lime',ls='--',lw=2)

            ax.legend(loc='upper center',fontsize='x-small',ncols=4)

        fmt_xaxis(ax,xlim)
        ax.set_ylabel('Range [km]')
        ax.set_ylim(1000,2000)

        return

if __name__ == '__main__':
    output_dir = os.path.join('output','hamSpotPlot')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    date    = datetime.datetime(2018,12,15)
    hsp     = HamSpotPlot(date)

    png_fname   = date.strftime('%Y%m%d')+'_hamSpotPlot.png'
    png_fpath   = os.path.join(output_dir,png_fname)
    hsp.plot_figure(png_fpath=png_fpath)
    print(png_fpath)
