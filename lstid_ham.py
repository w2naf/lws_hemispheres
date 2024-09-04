#!/usr/bin/env python
"""
This class will generate a time series plot of Mary Lou West's LSTID Amateur Radio Statistics.
"""
import os
import shutil
import datetime
import itertools

import numpy as np
import pandas as pd
# <stdin>:1: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
pd.set_option('future.no_silent_downcasting', True)

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

def load_supermag():
#    # Load Raw SuperMAG data, remove out of range and bad data, and 
#    # compute SME.
#    data_dir    = os.path.join('data','supermag_sme')
#    fpath       = os.path.join(data_dir,'20230808-02-16-supermag.csv.bz2')
#
#    df_0        = pd.read_csv(fpath,parse_dates=[0])
#    df_0        = df_0.set_index('Date_UTC')
#
#    sDate       = datetime.datetime(2010,1,1)
#    eDate       = datetime.datetime(2023,1,1)
#    tf          = np.logical_and(df_0.index >= sDate, df_0.index < eDate)
#    df_0        = df_0[tf]
#    df_0        = df_0.replace(999999,np.nan)
#
#    df_0['SME'] = df_0['SMU'] - df_0['SML']
#
#    sDate_str   = sDate.strftime('%Y%m%d')
#    eDate_str   = eDate.strftime('%Y%m%d')
#    out_fname   = '{!s}_{!s}_SME.csv.bz2'.format(sDate_str,eDate_str)
#    out_path    = os.path.join(data_dir,out_fname)
#
#    df_0.to_csv(out_path)

    data_dir    = os.path.join('data','supermag_sme')
    fpath       = os.path.join(data_dir,'20100101_20230101_SME.csv.bz2')

    df_0        = pd.read_csv(fpath,parse_dates=[0])
    df_0        = df_0.set_index('Date_UTC')

    return df_0

class LSTID_HAM(object):
    def __init__(self,dataSet='sinFit'):
        """
        dataSet: ['sinFit']
        """
        self.dataSet = dataSet
        self.load_data()
    
    def load_data(self):
        data_in         = 'data/lstid_ham/20181101-20190430_allSinFits.csv'
#        df              = pd.read_csv(data_in,usecols=[0, 1, 2,3],names=['date','period_hr','amplitude_km','is_lstid'],parse_dates=[0],header=1)
        cols             = {}
        cols[0]          = 'date'
        cols[1]          = 'period_hr'
        cols[2]          = 'amplitude_km'
        #        cols[3] = 'phase_hr'
        #        cols[4] = 'offset_km'
        #        cols[5] = 'slope_kmph'
        cols[6]          = 'r2'
        #        cols[7] = 't_hr_guess'
        cols[8]          = 'selected' # this tells us the fit that was actually used!
        cols[11]         = 'duration_hr'
        
        usecols = list(cols.keys())
        names   = list(cols.values())
        df      = pd.read_csv(data_in,usecols = usecols,names = names,parse_dates = [0],header = 1)
        df      = df[df['selected']].copy()

        # Convert data columns to floats.
        cols_numeric    = []
        cols_numeric.append('period_hr')
        cols_numeric.append('amplitude_km')
        cols_numeric.append('phase_hr')
        cols_numeric.append('offset_km')
        cols_numeric.append('slope_kmph')
        cols_numeric.append('r2')
        cols_numeric.append('T_hr_guess')
        cols_numeric.append('duration_hr')

        for col in cols_numeric:
            if col not in df.keys():
               continue 
            df.loc[:,col] = pd.to_numeric(df[col],errors='coerce')

        self.data_in    = data_in
        self.df         = df

    def plot_figure(self,png_fpath='output.png',figsize=(16,5),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,xlim=None,ylabel_fontdict={},legend_fontsize='large',
            legend_ncols=2,plot_sme=False,**kwargs):
        
#        plot_sme=True

        fig      = ax.get_figure()
        
        df       = self.df
        hndls    = []
        
        # Set Criteria for Classifying as LSTID
        lims    = {}
        lims['amplitude_km']    = (0, 80)
        lims['r2']              = (0.35, 1.1)
        lims['period_hr']       = (1, 4.5)
        lims['duration_hr']     = (4, 24)

        # Apply LSTID Classification Criteria.
        tfs = []
        for key,limits in lims.items():
            tf = (np.logical_and(df[key] >= limits[0], df[key] < limits[1])).values
            tfs.append(tf)
        is_lstid    = np.logical_and.reduce(tfs)

#         Set days that do not meet the criteria to NaN so they do not plot.
        df.loc[~is_lstid,'period_hr']      = np.nan
        df.loc[~is_lstid,'amplitude_km']   = np.nan

#        df['amplitude_km'] = df['amplitude_km'].interpolate(method='linear')
#        df    = df.dropna()
        
        xx      = df['date']
        
        if xlim is None:
            xlim = (min(xx), max(xx))

        yy           = df['amplitude_km']
        rolling_days = 3 
        min_periods  = 2
        yy_roll      = df['amplitude_km'].rolling(rolling_days,min_periods=min_periods,center=True).mean()
        ylabel       = 'LSTID Amplitude [km]'
        hndl         = ax.plot(xx,yy,label='Raw Data',color='grey',lw=2)
        hndls.append(hndl)

        nans                = np.isnan(yy)
        yy_roll_nans        = yy_roll.copy()
        yy_roll_nans[nans]  = np.nan
        hndl         = ax.plot(xx,yy_roll_nans,label=f'{rolling_days} Day Rolling Mean',color='blue',lw=3)
        hndls.append(hndl)
        ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)
        ax.set_xlabel('UTC Date')

        vmin            = np.nanmin(yy_roll)
        vmax            = 45
        T_hr_cmap       = 'rainbow'
        cmap            = mpl.colormaps.get_cmap(T_hr_cmap)
        cmap.set_bad(color='white')
        norm            = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        mpbl            = mpl.cm.ScalarMappable(norm,cmap)
        color           = mpbl.to_rgba(yy_roll)
        trans           = mpl.transforms.blended_transform_factory( ax.transData, ax.transAxes)
        cbar_pcoll      = ax.bar(xx,1,width=1,color=color,align='edge',zorder=-1,transform=trans,alpha=0.5)
        cbar_label      = 'Amplitude [km]'

        text = 'Automated SinFit'
        
        ax.text(0.01,0.95,text,transform=ax.transAxes)
        hndls = list(itertools.chain.from_iterable(hndls))
        legend_ncols = 1
        ax.legend(handles=hndls,loc='upper right',fontsize=legend_fontsize,ncols=legend_ncols)

        if plot_sme:
            supermag = load_supermag()
            tf = np.logical_and(supermag.index >= xlim[0], supermag.index < xlim[1])
            supermag = supermag[tf].copy()

            ax2 = ax.twinx()
            ax2_xx = supermag.index
            ax2_yy = supermag['SME']
            ax2.plot(ax2_xx,ax2_yy,color='k',alpha=0.5)
            ax2.set_ylabel('SME [nT]')
            ax2.grid(False)
            ax2.set_ylim(0,2000)

        title   = 'Amateur Radio 14 MHz LSTID Observations'
        ax.set_title(title)

        ax.set_xlim(xlim)

        result  = {}
        result['cbar_pcoll'] = mpbl
        result['cbar_label'] = cbar_label
        result['title'] = title
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','lstid_ham')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for plot_sme in [True,False]:
        lstid = LSTID_HAM()
        if plot_sme:
            png_fname   = 'lstid_ham_sme.png'
        else:
            png_fname   = 'lstid_ham.png'

        png_fpath   = os.path.join(output_dir,png_fname)

        lstid.plot_figure(png_fpath=png_fpath,plot_sme=plot_sme)
        print(png_fpath)
