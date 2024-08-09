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

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

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
        self.load_data()
    
    def load_data(self):
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
        # hamSpot_geo_2018_12_15.csv.bz2

    def plot_timeSeries_figure(self,png_fpath='output.png',figsize=(16,5),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_timeSeries_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
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
    hsp.plot_timeSeries_figure(png_fpath=png_fpath)
    print(png_fpath)
