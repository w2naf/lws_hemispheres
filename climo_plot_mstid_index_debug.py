#!/usr/bin/env python

import os
import shutil
import glob

import datetime

import tqdm

import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

pd.set_option('display.max_rows', None)

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

#cbar_title_fontdict     = {'weight':'bold','size':30}
#cbar_ytick_fontdict     = {'size':30}
#xtick_fontdict          = {'weight': 'bold', 'size':30}
#ytick_major_fontdict    = {'weight': 'bold', 'size':28}
#ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
#corr_legend_fontdict    = {'weight': 'bold', 'size':24}
#keo_legend_fontdict     = {'weight': 'normal', 'size':30}
#driver_xlabel_fontdict  = ytick_major_fontdict
#driver_ylabel_fontdict  = ytick_major_fontdict
#title_fontdict          = {'weight': 'bold', 'size':36}

cbar_title_fontdict     = {'weight':'bold','size':42}
cbar_ytick_fontdict     = {'size':36}
xtick_fontdict          = {'weight': 'bold', 'size':24}
ytick_major_fontdict    = {'weight': 'bold', 'size':24}
ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
title_fontdict          = {'weight': 'bold', 'size':36}
ylabel_fontdict         = {'weight': 'bold', 'size':24}
reduced_legend_fontdict = {'weight': 'bold', 'size':20}

prm_dct = {}
prmd = prm_dct['meanSubIntSpect_by_rtiCnt'] = {}
prmd['scale_0']         = -0.025
prmd['scale_1']         =  0.025
prmd['cmap']            = mpl.cm.jet
prmd['cbar_label']      = 'MSTID Index'
prmd['cbar_tick_fmt']   = '%0.3f'
prmd['title']           = 'SuperDARN MSTID Index'

prmd = prm_dct['meanSubIntSpect_by_rtiCnt_reducedIndex'] = {}
prmd['title']           = 'Reduced SuperDARN MSTID Index'
prmd['ylabel']          = 'Reduced SuperDARN\nMSTID Index'
prmd['ylim']            = (-5,5)

prmd = prm_dct['U_10HPA'] = {}
prmd['scale_0']         = -100.
prmd['scale_1']         =  100.
prmd['cmap']            = mpl.cm.bwr
prmd['cbar_label']      = 'U 10 hPa [m/s]'
prmd['title']           = 'MERRA2 Zonal Winds 10 hPa [m/s]'
prmd['data_dir']        = os.path.join('data','merra2','preprocessed')

prmd = prm_dct['U_1HPA'] = {}
prmd['scale_0']         = -100.
prmd['scale_1']         =  100.
prmd['cmap']            = mpl.cm.bwr
prmd['cbar_label']      = 'U 1 hPa [m/s]'
prmd['title']           = 'MERRA2 Zonal Winds 1 hPa [m/s]'
prmd['data_dir']        = os.path.join('data','merra2','preprocessed')

# ['DAILY_SUNSPOT_NO_', 'DAILY_F10.7_', '1-H_DST_nT', '1-H_AE_nT']
# DAILY_SUNSPOT_NO_  DAILY_F10.7_    1-H_DST_nT     1-H_AE_nT
# count       40488.000000  40488.000000  40488.000000  40488.000000
# mean           58.125963    103.032365    -10.984427    162.772167
# std            46.528777     29.990254     16.764279    175.810863
# min             0.000000     64.600000   -229.500000      3.500000
# 25%            17.000000     76.400000    -18.000000     46.000000
# 50%            50.000000     97.600000     -8.000000     92.000000
# 75%            90.000000    122.900000     -0.500000    215.000000
# max           220.000000    255.000000     64.000000   1637.000000

prmd = prm_dct['DAILY_SUNSPOT_NO_'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 175.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily SN'
prmd['title']           = 'Daily Sunspot Number'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['DAILY_F10.7_'] = {}
prmd['scale_0']         = 50.
prmd['scale_1']         = 200.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily F10.7'
prmd['title']           = 'Daily F10.7 Solar Flux'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['1-H_DST_nT'] = {}
prmd['scale_0']         =  -75
prmd['scale_1']         =   25
prmd['cmap']            = mpl.cm.inferno_r
prmd['cbar_label']      = 'Dst [nT]'
prmd['title']           = 'Disturbance Storm Time Dst Index [nT]'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['1-H_AE_nT'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 400
prmd['cmap']            = mpl.cm.viridis
prmd['cbar_label']      = 'AE [nT]'
prmd['title']           = 'Auroral Electrojet AE Index [nT]'
prmd['data_dir']        = os.path.join('data','cdaweb_omni','preprocessed')

prmd = prm_dct['OMNI_R_Sunspot_Number'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 175.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily SN'
prmd['title']           = 'Daily Sunspot Number'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['OMNI_F10.7'] = {}
prmd['scale_0']         = 50.
prmd['scale_1']         = 200.
prmd['cmap']            = mpl.cm.cividis
prmd['cbar_label']      = 'Daily F10.7'
prmd['title']           = 'Daily F10.7 Solar Flux'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['OMNI_Dst'] = {}
prmd['scale_0']         =  -75
prmd['scale_1']         =   25
prmd['cmap']            = mpl.cm.inferno_r
prmd['cbar_label']      = 'Dst [nT]'
prmd['title']           = 'Disturbance Storm Time Dst Index [nT]'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['OMNI_AE'] = {}
prmd['scale_0']         = 0
prmd['scale_1']         = 400
prmd['cmap']            = mpl.cm.viridis
prmd['cbar_label']      = 'AE [nT]'
prmd['title']           = 'Auroral Electrojet AE Index [nT]'
prmd['data_dir']        = os.path.join('data','omni','preprocessed')

prmd = prm_dct['reject_code'] = {}
prmd['title']           = 'MSTID Index Data Quality Flag'

# Reject code colors.
reject_codes = {}
# 0: Good Period (Not Rejected)
reject_codes[0] = {'color': mpl.colors.to_rgba('green'),  'label': 'Good Period'}
# 1: High Terminator Fraction (Dawn/Dusk in Observational Window)
reject_codes[1] = {'color': mpl.colors.to_rgba('blue'),   'label': 'Dawn / Dusk'}
# 2: No Data
reject_codes[2] = {'color': mpl.colors.to_rgba('red'),    'label': 'No Data'}
# 3: Poor Data Quality (including "Low RTI Fraction" and "Failed Quality Check")
reject_codes[3] = {'color': mpl.colors.to_rgba('gold'),   'label': 'Poor Data Quality'}
# 4: Other (including "No RTI Fraction" and "No Terminator Fraction")
reject_codes[4] = {'color': mpl.colors.to_rgba('purple'), 'label': 'Other'}
# 5: Not Requested (Outside of requested daylight times)
reject_codes[5] = {'color': mpl.colors.to_rgba('0.9'),   'label': 'Not Requested'}

def season_to_datetime(season):
    str_0, str_1 = season.split('_')
    sDate   = datetime.datetime.strptime(str_0,'%Y%m%d')
    eDate   = datetime.datetime.strptime(str_1,'%Y%m%d')
    return (sDate,eDate)

def plot_cbar(ax_info):
    cbar_pcoll = ax_info.get('cbar_pcoll')

    cbar_label      = ax_info.get('cbar_label')
    cbar_ticks      = ax_info.get('cbar_ticks')
    cbar_tick_fmt   = ax_info.get('cbar_tick_fmt','%0.3f')
    cbar_tb_vis     = ax_info.get('cbar_tb_vis',False)
    ax              = ax_info.get('ax')

    fig = ax.get_figure()

    box         = ax.get_position()

    x0  = 1.01
    wdt = 0.015
    y0  = 0.250
    hgt = (1-2.*y0)
    axColor = fig.add_axes([x0, y0, wdt, hgt])

    axColor.grid(False)
    cbar        = fig.colorbar(cbar_pcoll,orientation='vertical',cax=axColor,format=cbar_tick_fmt)

    cbar.set_label(cbar_label,fontdict=cbar_title_fontdict)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    axColor.set_ylim( *(cbar_pcoll.get_clim()) )

    labels = cbar.ax.get_yticklabels()
    fontweight  = cbar_ytick_fontdict.get('weight')
    fontsize    = cbar_ytick_fontdict.get('size')
    for label in labels:
        if fontweight:
            label.set_fontweight(fontweight)
        if fontsize:
            label.set_fontsize(fontsize)

#    if not cbar_tb_vis:
#        for inx in [0,-1]:
#            labels[inx].set_visible(False)

def reject_legend(fig):
    x0  = 1.01
    wdt = 0.015
    y0  = 0.250
    hgt = (1-2.*y0)

    axl= fig.add_axes([x0, y0, wdt, hgt])
    axl.axis('off')

    legend_elements = []
    for rej_code, rej_dct in reject_codes.items():
        color = rej_dct['color']
        label = rej_dct['label']
        # legend_elements.append(mpl.lines.Line2D([0], [0], ls='',marker='s', color=color, label=label,markersize=15))
        legend_elements.append(mpl.patches.Patch(facecolor=color,edgecolor=color,label=label))

    axl.legend(handles=legend_elements, loc='center left', fontsize = 42)

def my_xticks(sDate,eDate,ax,radar_ax=False,labels=True,
        fontdict=None):
    if fontdict is None:
        fontdict = xtick_fontdict
    xticks      = []
    xticklabels = []
    curr_date   = sDate
    while curr_date < eDate:
        if radar_ax:
            xpos    = get_x_coords(curr_date,sDate,eDate)
        else:
            xpos    = curr_date
        xticks.append(xpos)
        xticklabels.append('')
        curr_date += datetime.timedelta(days=1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Define xtick label positions here.
    # Days of month to produce a xtick label.
    doms    = [1,15]

    curr_date   = sDate
    ytransaxes = mpl.transforms.blended_transform_factory(ax.transData,ax.transAxes)
    while curr_date < eDate:
        if curr_date.day in doms:
            if radar_ax:
                xpos    = get_x_coords(curr_date,sDate,eDate)
            else:
                xpos    = curr_date

            axvline = ax.axvline(xpos,-0.015,color='k')
            axvline.set_clip_on(False)

            if labels:
                ypos    = -0.025
                txt     = curr_date.strftime('%d %b\n%Y')
                ax.text(xpos,ypos,txt,transform=ytransaxes,
                        ha='left', va='top',rotation=0,
                        fontdict=fontdict)
        curr_date += datetime.timedelta(days=1)

    xmax    = (eDate - sDate).total_seconds() / (86400.)
    if radar_ax:
        ax.set_xlim(0,xmax)
    else:
        ax.set_xlim(sDate,sDate+datetime.timedelta(days=xmax))

#    ax.grid(zorder=500)

def get_x_coords(win_sDate,sDate,eDate,full=False):
    x1  = (win_sDate - sDate).total_seconds()/86400.
    if not full:
        x1  = np.floor(x1)
    return x1

def get_y_coords(ut_time,st_uts,radar,radars):
    # Find start time index.
    st_uts      = np.array(st_uts)
    st_ut_inx   = np.digitize([ut_time],st_uts)-1
    
    # Find radar index.
    radar_inx   = np.where(radar == np.array(radars))[0]
    y1          = st_ut_inx*len(radars) + radar_inx
    return y1


def get_coords(radar,win_sDate,radars,sDate,eDate,st_uts,verts=True):
    # Y-coordinate.
    x1  = float(get_x_coords(win_sDate,sDate,eDate))
    y1  = float(get_y_coords(win_sDate.hour,st_uts,radar,radars))

    if verts:
#        x1,y1   = x1+0,y1+0
        x2,y2   = x1+1,y1+0
        x3,y3   = x1+1,y1+1
        x4,y4   = x1+0,y1+1
        return ((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1))
    else:
        x0      = x1 + 0.5
        y0      = y1 + 0.5
        return (x0, y0)

def plot_mstid_values(data_df,ax,sDate=None,eDate=None,
        st_uts=[14, 16, 18, 20],
        xlabels=True, group_name=None,classification_colors=False,
        rasterized=False,radars=None,param=None,**kwargs):

    prmd        = prm_dct.get(param,{})
    scale_0     = prmd.get('scale_0',-0.025)
    scale_1     = prmd.get('scale_1', 0.025)
    scale       = (scale_0, scale_1)

    cmap        = prmd.get('cmap',mpl.cm.jet)
    cbar_label  = prmd.get('cbar_label',param)

    if sDate is None:
        sDate = data_df.index.min()
        sDate = datetime.datetime(sDate.year,sDate.month,sDate.day)

        eDate = data_df.index.max()
        eDate = datetime.datetime(eDate.year,eDate.month,eDate.day) + datetime.timedelta(days=1)

    if radars is None:
        radars  = list(data_df.keys())

    # Reverse radars list order so that the supplied list is plotted top-down.
    radars  = radars[::-1]

    ymax    = len(st_uts) * len(radars)

    cbar_info   = {}
    bounds      = np.linspace(scale[0],scale[1],256)
    cbar_info['cbar_ticks'] = np.linspace(scale[0],scale[1],11)
    cbar_info['cbar_label'] = cbar_label

    norm    = mpl.colors.BoundaryNorm(bounds,cmap.N)

    if classification_colors:
        # Use colorscheme that matches MSTID Index in classification plots.
        from mstid.classify import MyColors
        scale_0             = -0.025
        scale_1             =  0.025
        my_cmap             = 'seismic'
        truncate_cmap       = (0.1, 0.9)
        my_colors           = MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)
        cmap                = my_colors.cmap
        norm                = my_colors.norm
                

    ################################################################################    
    current_date = sDate
    verts       = []
    vals        = []
    while current_date < eDate:
        for st_ut in st_uts:
            for radar in radars:
                win_sDate   = current_date + datetime.timedelta(hours=st_ut)

                val = data_df[radar].loc[win_sDate]

                if not np.isfinite(val):
                    continue

                if param == 'reject_code':
                    val = reject_codes.get(val,reject_codes[4])['color']

                vals.append(val)
                verts.append(get_coords(radar,win_sDate,radars,sDate,eDate,st_uts))

        current_date += datetime.timedelta(days=1)

    if param == 'reject_code':
        pcoll = PolyCollection(np.array(verts),edgecolors='0.75',linewidths=0.25,
                cmap=cmap,norm=norm,zorder=99,rasterized=rasterized)
        pcoll.set_facecolors(np.array(vals))
        ax.add_collection(pcoll,autolim=False)
    else:
        pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,
                cmap=cmap,norm=norm,zorder=99,rasterized=rasterized)
        pcoll.set_array(np.array(vals))
        ax.add_collection(pcoll,autolim=False)

    # Make gray missing data.
    ax.set_facecolor('0.90')

    # Add radar labels.
    trans = mpl.transforms.blended_transform_factory(ax.transAxes,ax.transData)
    for rdr_inx,radar in enumerate(radars):
        for st_inx,st_ut in enumerate(st_uts):
            ypos = len(radars)*st_inx + rdr_inx + 0.5
            ax.text(-0.002,ypos,radar,transform=trans,ha='right',va='center')

    # Add UT Time Labels
    for st_inx,st_ut in enumerate(st_uts):
        ypos    = st_inx*len(radars)
        xpos    = -0.035
        line    = ax.hlines(ypos,xpos,1.0,transform=trans,lw=3,zorder=100)
        line.set_clip_on(False)
        
        txt     = '{:02d}-{:02d}\nUT'.format(int(st_ut),int(st_ut+2))
        ypos   += len(radars)/2.
        xpos    = -0.025
#        ax.text(xpos,ypos,txt,transform=trans,
#                ha='center',va='center',rotation=90,fontdict=ytick_major_fontdict)
#        xpos    = -0.015
#        xpos    = -0.05
        xpos    = -0.025
        ax.text(xpos,ypos,txt,transform=trans,
                ha='right',va='center',rotation=0,fontdict=ytick_major_fontdict)

    xpos    = -0.035
    line    = ax.hlines(1.,xpos,1.0,transform=ax.transAxes,lw=3,zorder=100)
    line.set_clip_on(False)

    ax.set_ylim(0,ymax)

    # Set xticks and yticks to every unit to make a nice grid.
    # However, do not use this for actual labeling.
    yticks = list(range(len(radars)*len(st_uts)))
    ax.set_yticks(yticks)
    ytls = ax.get_yticklabels()
    for ytl in ytls:
        ytl.set_visible(False)

    my_xticks(sDate,eDate,ax,radar_ax=True,labels=xlabels)
    
    txt = ' '.join([x.upper() for x in radars[::-1]])
    if group_name is not None:
        txt = '{} ({})'.format(group_name,txt)
    ax.set_title(txt,fontdict=title_fontdict)

    ax_info         = {}
    ax_info['ax']   = ax
    ax_info['cbar_pcoll']   = pcoll
    ax_info.update(cbar_info)
    
    return ax_info

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

class ParameterObject(object):
    def __init__(self,param,radars,seasons=None,
            output_dir='output',write_csvs=True):

        # Create parameter dictionary.
        prmd        = prm_dct.get(param,{})
        prmd['param'] = param
        if prmd.get('data_dir') is None:
            prmd['data_dir'] = os.path.join('data','mstid_index')
        self.prmd   = prmd

        # Store radar list.
        self.radars = radars

        # Get list of seasons.
        if seasons is None:
            seasons = list_seasons()

        # Load data into dictionary of dataframes.
        self.data = {season:{} for season in seasons}
        print('Loading data...')
        self._load_data()

        self.output_dir = output_dir
        if write_csvs:
            print('Generating Season CSV Files...')
            for season in seasons:
                self.write_csv(season,output_dir=self.output_dir)

            csv_fpath   = os.path.join(self.output_dir,'radars.csv')
            self.lat_lons.to_csv(csv_fpath,index=False)

        for season in seasons:
            self.calculate_reduced_index(season,write_csvs=write_csvs)

    def calculate_reduced_index(self,season,
            reduction_type='mean',daily_vals=True,zscore=True,
            smoothing_window   = '4D', smoothing_type='mean', write_csvs=True):
        """
        Reduce the MSTID index from all radars into a single number as a function of time.

        This function will work on any paramter, not just the MSTID index.
        """
        print("Calulating reduced MSTID index.")

        mstid_inx_dict  = {} # Create a place to store the data.

        df = self.data[season]['df']

        # Put everything into a dataframe.
        
        if daily_vals:
            date_list   = np.unique([datetime.datetime(x.year,x.month,x.day) for x in df.index])

            tmp_list        = []
            n_good_radars   = []    # Set up a system for parameterizing data quality.
            for tmp_sd in date_list:
                tmp_ed      = tmp_sd + datetime.timedelta(days=1)

                tf          = np.logical_and(df.index >= tmp_sd, df.index < tmp_ed)
                tmp_df      = df[tf]
                if reduction_type == 'median':
                    tmp_vals    = tmp_df.median().to_dict()
                elif reduction_type == 'mean':
                    tmp_vals    = tmp_df.mean().to_dict()

                tmp_list.append(tmp_vals)

                n_good  = np.count_nonzero(np.isfinite(tmp_df))
                n_good_radars.append(n_good)

            df = pd.DataFrame(tmp_list,index=date_list)
            n_good_df   = pd.Series(n_good_radars,df.index)
        else:
            n_good_df   = np.sum(np.isfinite(df),axis=1)

        data_arr    = np.array(df)
        if reduction_type == 'median':
            red_vals    = sp.nanmedian(data_arr,axis=1)
        elif reduction_type == 'mean':
            red_vals    = np.nanmean(data_arr,axis=1)

        ts  = pd.Series(red_vals,df.index)
        if zscore:
            ts  = (ts - ts.mean())/ts.std()

        reducedIndex = pd.DataFrame({'reduced_index':ts,'n_good_df':n_good_df},index=df.index)
#        reducedIndex['smoothed']    = reducedIndex['reduced_index'].rolling(smoothing_window,center=True).mean()
        reducedIndex['smoothed']    = getattr( reducedIndex['reduced_index'].rolling(smoothing_window,center=True), smoothing_type )()

        self.data[season]['reducedIndex']    = reducedIndex

        reducedIndex_attrs       = {}
        reducedIndex_attrs['reduction_type']     = reduction_type
        reducedIndex_attrs['zscore']             = zscore
        reducedIndex_attrs['daily_vals']         = daily_vals
        reducedIndex_attrs['smoothing_window']   = smoothing_window
        reducedIndex_attrs['smoothing_type']     = smoothing_type
        self.data[season]['reducedIndex_attrs']  = reducedIndex_attrs

        param   = '{!s}_reducedIndex'.format(self.prmd.get('param'))
        attrs   = self.data[season]['attrs']

        if write_csvs:
            output_dir = self.output_dir

            csv_fname       = '{!s}_{!s}.csv'.format(season,param)
            csv_fpath       = os.path.join(output_dir,csv_fname)
            with open(csv_fpath,'w') as fl:
                hdr = []
                hdr.append('# SuperDARN MSTID Index Datafile - Reduced Index')
                hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
                hdr.append('# Generated on: {!s} UTC'.format(datetime.datetime.utcnow()))
                hdr.append('#')
                hdr.append('# Parameter: {!s}'.format(param))
                hdr.append('#')
                hdr.append('# Original Attributes:')
                for attr in attrs:
                    hdr.append('# {!s}'.format(attr))
                hdr.append('#')
                hdr.append('# Reduction Attributes:')
                for attr in reducedIndex_attrs.items():
                    hdr.append('# {!s}'.format(attr))
                hdr.append('#')

                fl.write('\n'.join(hdr))
                fl.write('\n')
                
            reducedIndex.to_csv(csv_fpath,mode='a')

    def _load_data(self):
        """
        Load data into data frames and store in self.data dictionary.
        """

        data_dir    = self.prmd.get('data_dir')
        param       = self.prmd.get('param')

        lat_lons    = []
        for season in tqdm.tqdm(self.data.keys(),desc='Seasons',dynamic_ncols=True,position=0):
            # Load all data from a season into a single xarray dataset.
            ds      = []
            attrs   = []
            for radar in self.radars:
    #            fl  = os.path.join(data_dir,'sdMSTIDindex_{!s}_{!s}.nc'.format(season,radar))
                fl  = glob.glob(os.path.join(data_dir,'*{!s}_{!s}.nc'.format(season,radar)))[0]
                dsr = xr.open_dataset(fl)
                ds.append(dsr)
                attrs.append(dsr.attrs)

                # Store radar lat / lons to creat a radar location file.
                lat_lons.append({'radar':radar,'lat':dsr.attrs['lat'],'lon':dsr.attrs['lon']})
            dss = xr.concat(ds,dim='index')

            # Convert parameter of interest to a datafame.
            df      = dss[param].to_dataframe()
            dfrs = {}
            for radar in tqdm.tqdm(radars,desc='Radars',dynamic_ncols=True,position=1,leave=False):
                tf      = df['radar'] == radar
                dft     = df[tf]
                dates   = dft.index
                vals    = dft[param]

                for date,val in zip(dates,vals):
                    if date not in dfrs:
                        dfrs[date] = {}
                    dfrs[date][radar] = val

            df  = pd.DataFrame(dfrs.values(),dfrs.keys())
            df  = df.sort_index()
            df.index.name               = 'datetime'
            self.data[season]['df']     = df
            self.data[season]['attrs']  = attrs

        # Clean up lat_lon data table
        self.lat_lons    = pd.DataFrame(lat_lons).drop_duplicates()

    def write_csv(self,season,output_dir=None):
        """
        Save data to CSV files.
        """

        param   = self.prmd.get('param')
        df      = self.data[season]['df']
        attrs   = self.data[season]['attrs']

        if output_dir is None:
            output_dir = self.output_dir

        csv_fname       = '{!s}_{!s}.csv'.format(season,param)
        csv_fpath       = os.path.join(output_dir,csv_fname)
        with open(csv_fpath,'w') as fl:
            hdr = []
            hdr.append('# SuperDARN MSTID Index Datafile')
            hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
            hdr.append('# Generated on: {!s} UTC'.format(datetime.datetime.utcnow()))
            hdr.append('#')
            hdr.append('# Parameter: {!s}'.format(param))
            hdr.append('#')
            for attr in attrs:
                hdr.append('# {!s}'.format(attr))
            hdr.append('#')

            fl.write('\n'.join(hdr))
            fl.write('\n')
            
        df.to_csv(csv_fpath,mode='a')

    def plot_climatology(self,output_dir=None):

        if output_dir is None:
            output_dir = self.output_dir

        seasons = self.data.keys()
        radars  = self.radars
        param   = self.prmd['param']

        nrows   = 6
        ncols   = 2
        fig = plt.figure(figsize=(50,30))

        ax_list = []
        for inx,season in enumerate(seasons):
            print(' -->',season)
            ax      = fig.add_subplot(nrows,ncols,inx+1)

            data_df = self.data[season]['df']
            
            sDate, eDate = season_to_datetime(season)
            ax_info = plot_mstid_values(data_df,ax,radars=radars,param=param,sDate=sDate,eDate=eDate)
            ax_list.append(ax_info)

            season_yr0 = season[:4]
            season_yr1 = season[9:13]
            txt = '{!s} - {!s} Northern Hemisphere Winter'.format(season_yr0,season_yr1)
            ax.set_title(txt,fontdict=title_fontdict)

        fig.tight_layout(w_pad=2.25)

        if param == 'reject_code':
            reject_legend(fig)
        else:
            plot_cbar(ax_list[1])

        fpath = os.path.join(output_dir,'{!s}.png'.format(param))
        print('SAVING: ',fpath)
    #    fig.savefig(fpath)
        fig.savefig(fpath,bbox_inches='tight')

def stackplot(po_dct,params,season,radars=None,fpath='stackplot.png'):
    print(' Plotting Stackplot: {!s}'.format(fpath))
    nrows   = len(params)
    ncols   = 1
    fig = plt.figure(figsize=(25,nrows*5))

    ax_list = []
    for inx,param in enumerate(params):
        ax      = fig.add_subplot(nrows,ncols,inx+1)

        if param.endswith('_reducedIndex'):
            base_param      = param.rstrip('_reducedIndex')
            plotType        = 'reducedIndex'
        else:
            base_param  = param
            plotType        = 'climo'

        # Get Parameter Object
        po      = po_dct.get(base_param)
        if plotType == 'reducedIndex':
            data_df = po.data[season]['reducedIndex']
            prmd    = prm_dct.get(param,{})
        else:
            data_df = po.data[season]['df']
            prmd    = po.prmd

        if radars is None:
            _radars = po.radars
        else:
            _radars = radars

        if inx == nrows-1:
            xlabels = True
        else:
            xlabels = False

        if plotType == 'reducedIndex':
            handles = []

            xx      = data_df.index
            yy      = data_df['reduced_index']
            label   = 'Raw'
            hndl    = ax.plot(xx,yy,label=label)
            handles.append(hndl[0])

            xx      = data_df.index
            yy      = data_df['smoothed']
            ri_attrs    = po.data[season]['reducedIndex_attrs']
            label   = '{!s} Rolling {!s}'.format(ri_attrs['smoothing_window'],ri_attrs['smoothing_type'].capitalize())
            hndl    = ax.plot(xx,yy,lw=3,label=label)
            handles.append(hndl[0])

            ax1     = ax.twinx()
            xx      = data_df.index
            yy      = data_df['n_good_df']
            label   = 'n Data Points'
            hndl    = ax1.plot(xx,yy,color='0.8',ls='--',label=label)
            ax1.set_ylabel('n Data Points\n(Dashed Line)',fontdict=ylabel_fontdict)
            handles.append(hndl[0])

            ax.legend(handles=handles,loc='lower left',ncols=3,prop=reduced_legend_fontdict)

            ax_info = {}
            ax_info['ax']           = ax
        else: 
            ax_info = plot_mstid_values(data_df,ax,radars=_radars,param=param,xlabels=xlabels)
        ax_list.append(ax_info)

        ylim    = prmd.get('ylim')
        if ylim is not None:
            ax.set_ylim(ylim)

        ylabel  = prmd.get('ylabel')
        if ylabel is not None:
            ax.set_ylabel(ylabel,fontdict=ylabel_fontdict)

        txt = prmd.get('title',param)
        left_title_fontdict  = {'weight': 'bold', 'size': 24}
        ax.set_title('')
        ax.set_title(txt,fontdict=left_title_fontdict,loc='left')

        season_yr0 = season[:4]
        season_yr1 = season[9:13]
        txt = '{!s} - {!s} Northern Hemisphere Winter'.format(season_yr0,season_yr1)
        fig.text(0.5,1.01,txt,ha='center',fontdict=title_fontdict)

    fig.tight_layout()

    for param,ax_info in zip(params,ax_list):
        # Plot Colorbar ################################################################
        ax  = ax_info.get('ax')
        if param == 'reject_code':
            ax_pos  = ax.get_position()
            x0  = 1.005
            wdt = 0.015
            y0  = ax_pos.extents[1]
            hgt = ax_pos.height

            axl= fig.add_axes([x0, y0, wdt, hgt])
            axl.axis('off')

            legend_elements = []
            for rej_code, rej_dct in reject_codes.items():
                color = rej_dct['color']
                label = rej_dct['label']
                # legend_elements.append(mpl.lines.Line2D([0], [0], ls='',marker='s', color=color, label=label,markersize=15))
                legend_elements.append(mpl.patches.Patch(facecolor=color,edgecolor=color,label=label))

            axl.legend(handles=legend_elements, loc='center left', fontsize = 18)
        elif ax_info.get('cbar_pcoll') is not None:
            ax_pos  = ax.get_position()
            x0  = 1.01
            wdt = 0.015
            y0  = ax_pos.extents[1]
            hgt = ax_pos.height
            axColor = fig.add_axes([x0, y0, wdt, hgt])
            axColor.grid(False)

            cbar_pcoll      = ax_info.get('cbar_pcoll')
            cbar_label      = ax_info.get('cbar_label')
            cbar_ticks      = ax_info.get('cbar_ticks')
            cbar_tick_fmt   = prmd.get('cbar_tick_fmt')
            cbar_tb_vis     = ax_info.get('cbar_tb_vis',False)

			# fraction : float, default: 0.15
			#     Fraction of original axes to use for colorbar.
			# 
			# shrink : float, default: 1.0
			#     Fraction by which to multiply the size of the colorbar.
			# 
			# aspect : float, default: 20
			#     Ratio of long to short dimensions.
			# 
			# pad : float, default: 0.05 if vertical, 0.15 if horizontal
			#     Fraction of original axes between colorbar and new image axes.
            cbar  = fig.colorbar(cbar_pcoll,orientation='vertical',
                    cax=axColor,format=cbar_tick_fmt)

            cbar_label_fontdict = {'weight': 'bold', 'size': 24}
            cbar.set_label(cbar_label,fontdict=cbar_label_fontdict)
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)

            cbar.ax.set_ylim( *(cbar_pcoll.get_clim()) )

            labels = cbar.ax.get_yticklabels()
            fontweight  = cbar_ytick_fontdict.get('weight')
            fontsize    = 18
            for label in labels:
                if fontweight:
                    label.set_fontweight(fontweight)
                if fontsize:
                    label.set_fontsize(fontsize)

    fig.savefig(fpath,bbox_inches='tight')

def prep_dir(path,clear=False):
    if clear:
        if os.path.exists(path):
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    output_base_dir     = 'output'
    plot_climatologies  = False
    plot_stackplots     = True

    radars          = []
    # 'High Latitude Radars'
    radars.append('pgr')
    radars.append('sas')
    radars.append('kap')
    radars.append('gbr')
    # 'Mid Latitude Radars'
    radars.append('cvw')
    radars.append('cve')
    radars.append('fhw')
    radars.append('fhe')
    radars.append('bks')
    radars.append('wal')

#    # Ordered by Longitude
#    radars          = []
#    radars.append('cvw')
#    radars.append('pgr')
#    radars.append('cve')
#    radars.append('sas')
#    radars.append('fhw')
#    radars.append('fhe')
#    radars.append('kap')
#    radars.append('bks')
#    radars.append('wal')
#    radars.append('gbr')

    params = []
    params.append('meanSubIntSpect_by_rtiCnt')
    params.append('reject_code')
#    params.append('U_10HPA')
#    params.append('U_1HPA')

#    params.append('OMNI_R_Sunspot_Number')
#    params.append('OMNI_Dst')
#    params.append('OMNI_F10.7')
#    params.append('OMNI_AE')

#    params.append('1-H_AE_nT')
#    params.append('1-H_DST_nT')
#    params.append('DAILY_F10.7_')
#    params.append('DAILY_SUNSPOT_NO_')

    seasons = list_seasons()
    seasons = ['20121101_20130501']

    po_dct  = {}
    for param in params:
        # Generate Output Directory
        output_dir  = os.path.join(output_base_dir,param)
        prep_dir(output_dir,clear=True)

        po = ParameterObject(param,radars=radars,seasons=seasons,output_dir=output_dir)
        po_dct[param]   = po

    if plot_climatologies:
        for param,po in po_dct.items():
            print('Plotting Climatology: {!s}'.format(param))
            po.plot_climatology()

    # Generate Stackplots
    stack_sets  = {}
##    ss = stack_sets['cdaweb_omni'] = []
##    ss.append('meanSubIntSpect_by_rtiCnt')
##    ss.append('1-H_AE_nT')
##    ss.append('1-H_DST_nT')
##    ss.append('DAILY_F10.7_')
###    ss.append('DAILY_SUNSPOT_NO_')

#    ss = stack_sets['omni'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('OMNI_AE')
#    ss.append('OMNI_Dst')
##    ss.append('OMNI_F10.7')
##    ss.append('OMNI_R_Sunspot_Number')
#
#    ss = stack_sets['mstid_merra2'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('U_1HPA')
#    ss.append('U_10HPA')
#
#    ss = stack_sets['data_quality'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')
#    ss.append('reject_code')

    ss = stack_sets['mstid_index_reduced'] = []
    ss.append('meanSubIntSpect_by_rtiCnt')
    ss.append('meanSubIntSpect_by_rtiCnt_reducedIndex')

#    ss = stack_sets['mstid_index'] = []
#    ss.append('meanSubIntSpect_by_rtiCnt')

    if plot_stackplots:
        for stack_code,stack_params in stack_sets.items():
            stack_dir  = os.path.join(output_base_dir,'stackplots',stack_code)
            prep_dir(stack_dir,clear=True)
            for season in seasons:
                png_name    = '{!s}_stack_{!s}.png'.format(season,stack_code)
                png_path    = os.path.join(stack_dir,png_name) 

                stackplot(po_dct,stack_params,season,fpath=png_path)
