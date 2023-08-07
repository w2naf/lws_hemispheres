#!/usr/bin/env python

import os
import shutil
import glob

import datetime

import tqdm

import numpy as np
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

prm_dct = {}
prmd = prm_dct['meanSubIntSpect_by_rtiCnt'] = {}
prmd['scale_0']     = -0.025
prmd['scale_1']     =  0.025
prmd['cmap']        = mpl.cm.jet
prmd['cbar_label']  = 'MSTID Index'

prmd = prm_dct['U_10HPA'] = {}
prmd['scale_0']     = -100.
prmd['scale_1']     =  100.
prmd['cmap']        = mpl.cm.RdYlGn
prmd['cbar_label']  = 'U 10 hPa [m/s]'
prmd['data_dir']    = os.path.join('data','merra2','preprocessed')

prmd = prm_dct['U_1HPA'] = {}
prmd['scale_0']     = -100.
prmd['scale_1']     =  100.
prmd['cmap']        = mpl.cm.RdYlGn
prmd['cbar_label']  = 'U 1 hPa [m/s]'
prmd['data_dir']    = os.path.join('data','merra2','preprocessed')

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

#                print(val, radar, win_sDate)
#                import ipdb; ipdb.set_trace()

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

if __name__ == '__main__':

    params = []
    params.append('meanSubIntSpect_by_rtiCnt')
    params.append('reject_code')
    params.append('U_10HPA')
    params.append('U_1HPA')

    for param in params:
        prmd        = prm_dct.get(param,{})
        data_dir    = prmd.get('data_dir',os.path.join('data','mstid_index'))

        # Generate Output Directory
        output_dir  = os.path.join('output',param)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Find all available data files.
        pattern     = '*.nc'
        data_fls    = glob.glob(os.path.join(data_dir,pattern))
        data_fls.sort()

        # Identify unique seasons available
        seasons = []
        for fl in data_fls:
            spl     = os.path.basename(fl).split('_')
            season  = '{!s}_{!s}'.format(spl[1],spl[2])
            if season == '20221101_20230501':
                continue
            seasons.append(season)
        seasons = list(set(seasons))
        seasons.sort()

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

        data_dict = {}
        lat_lons  = []

        print('Generating Season CSV Files...')
        for season in tqdm.tqdm(seasons,desc='Seasons',dynamic_ncols=True,position=0):
            # Load all data from a season into a single xarray dataset.
            ds      = []
            attrs   = []
            for radar in radars:
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
            df.index.name   = 'datetime'

            csv_fname       = '{!s}_{!s}.csv'.format(season,param)
            csv_fpath       = os.path.join(output_dir,csv_fname)
            with open(csv_fpath,'w') as fl:
                hdr = []
                hdr.append('# SuperDARN MSTID Index Datafile')
                hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
                hdr.append('#')
                hdr.append('# Parameter: {!s}'.format(param))
                hdr.append('#')
                for attr in attrs:
                    hdr.append('# {!s}'.format(attr))
                hdr.append('#')

                fl.write('\n'.join(hdr))
                fl.write('\n')
                
        #        cols = ['datetime_ut'] + list(df.keys())
        #        fl.write(','.join(cols))
        #        fl.write('\n')
            df.to_csv(csv_fpath,mode='a')

            data_dict[season] = df

        # Clean up lat_lon data table and save to disk.
        ll_df       = pd.DataFrame(lat_lons).drop_duplicates()
        csv_fpath   = os.path.join(output_dir,'radars.csv')
        ll_df.to_csv(csv_fpath,index=False)

        nrows   = 6
        ncols   = 2
        print('Plotting Climatologies...')
        fig = plt.figure(figsize=(50,30))

        ax_list = []
        for inx,season in enumerate(seasons):
            print(' -->',season)
            ax      = fig.add_subplot(nrows,ncols,inx+1)

            data_df = data_dict[season]

            ax_info = plot_mstid_values(data_df,ax,radars=radars,param=param)
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
