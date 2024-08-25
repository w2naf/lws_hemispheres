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
        data_in         = 'data/lstid_ham/20181101-20181101_sinFit.csv'
        df              = pd.read_csv(data_in,usecols=[0, 1, 2,3],names=['date','period_hr','amplitude_km','is_lstid'],parse_dates=[0],header=1)

        # Convert data columns to floats.
        cols_numeric    = ['period_hr','amplitude_km']
        for col in cols_numeric:
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
        
        fig      = ax.get_figure()
        
        df       = self.df
        hndls    = []
        
        # Eliminate waves with amplitudes > 80 km.
        tf = df['amplitude_km'] > 80
        df.loc[tf,'period_hr']      = np.nan
        df.loc[tf,'amplitude_km']   = np.nan
        df.loc[tf,'is_lstid']       = np.nan

        
        df['amplitude_km'] = df['amplitude_km'].interpolate(method='linear')
#        df    = df.dropna()
        
        xx      = df['date']
        
        if xlim is None:
            xlim = (min(xx), max(xx))

        if self.dataSet == 'MLW':
            yy      = df['tid_hours']
            label  = 'TID Occurrence [hr]'
            hndl    = ax.bar(xx,yy,width=1,label=label,color='green',align='edge')
            hndls.append(hndl)
            ylabel  = 'LSTID Occurrence [hr]\nLSTID Period [hr]'
            ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)
            ax.set_xlabel('UTC Date')

            yy      = df['period_hr']
            ylabel  = 'LSTID Period [hr]'
            hndl    = ax.bar(xx,yy,label=ylabel,color='orange',align='edge')
            hndls.append(hndl)
            text = 'MLW Manual'
        elif self.dataSet == 'sinFit':
            yy           = df['amplitude_km']
            rolling_days = 3
            yy_roll      = df['amplitude_km'].rolling(rolling_days,center=True).mean()
            ylabel       = 'LSTID Amplitude [km]'
#            hndl         = ax.plot(xx,yy,width=1,label=ylabel,color='green',align='edge')
            hndl         = ax.plot(xx,yy,label='Raw Data',color='grey')
            hndls.append(hndl)
            hndl         = ax.plot(xx,yy_roll,label='3 Day Rolling Mean',color='blue',lw=3)
            hndls.append(hndl)
            ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)
            ax.set_xlabel('UTC Date')

            vmin            = np.nanmin(yy_roll)
            vmax            = 50
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
