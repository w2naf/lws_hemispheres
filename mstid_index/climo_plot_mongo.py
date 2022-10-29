#!/usr/bin/env python

import os
import datetime
import copy
import string

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

import pymongo

from mstid import general_lib as gl

cbar_title_fontdict     = {'weight':'bold','size':30}
cbar_ytick_fontdict     = {'size':30}
xtick_fontdict          = {'weight': 'bold', 'size':30}
ytick_major_fontdict    = {'weight': 'bold', 'size':28}
ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
corr_legend_fontdict    = {'weight': 'bold', 'size':24}
keo_legend_fontdict     = {'weight': 'normal', 'size':30}
driver_xlabel_fontdict  = ytick_major_fontdict
driver_ylabel_fontdict  = ytick_major_fontdict
title_fontdict          = {'weight': 'bold', 'size':36}

def plot_cbars(ax_list):
    for ax_info in ax_list:
        cbar_pcoll = ax_info.get('cbar_pcoll')
        if cbar_pcoll is None:
            continue

        cbar_label      = ax_info.get('cbar_label')
        cbar_ticks      = ax_info.get('cbar_ticks')
        cbar_tick_fmt   = ax_info.get('cbar_tick_fmt','%0.3f')
        cbar_tb_vis     = ax_info.get('cbar_tb_vis',False)
        ax              = ax_info.get('ax')

        box         = ax.get_position()
        axColor     = plt.axes([(box.x0 + box.width) * 1.01 , box.y0, 0.01, box.height])
        cbar        = plt.colorbar(cbar_pcoll,orientation='vertical',cax=axColor,format=cbar_tick_fmt)

        cbar.set_label(cbar_label,fontdict=cbar_title_fontdict)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        labels = cbar.ax.get_yticklabels()
        fontweight  = cbar_ytick_fontdict.get('weight')
        fontsize    = cbar_ytick_fontdict.get('size')
        for label in labels:
            if fontweight:
                label.set_fontweight(fontweight)
            if fontsize:
                label.set_fontsize(fontsize)

        if not cbar_tb_vis:
            for inx in [0,-1]:
                labels[inx].set_visible(False)

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
    ytransaxes = matplotlib.transforms.blended_transform_factory(ax.transData,ax.transAxes)
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


def get_radar_ax_frac(this_key,group_dict):
    """
    Computes fraction of radars in one group versus all radars.
    
    This is used to figure out how much space an axis should take
    in a figure.
    """
    all_radars      = []
    these_radars    = []
    keys            = list(group_dict.keys())
    for key in keys:
        dct_list    = group_dict[key]['dct_list']
        radars      = [dct['radar'] for dct in dct_list]
        all_radars += radars

        if this_key == key:
            these_radars = radars

    frac = float(len(these_radars)) / float(len(all_radars))
    return frac

def get_sDate_eDate(group_dict,sDate,eDate):
    """
    Finds the ealiest sDate and latest eDate for all radars in question,
    unless an sDate and eDate has been explicitly specified.
    """
    if (sDate is None) or (eDate is None):
        dates       = []
        for key,val in list(group_dict.items()):
            dct_list = val['dct_list']
            for dct in dct_list:
                dates.append(dct['list_sDate'])
                dates.append(dct['list_eDate'])

    if sDate is None:
        sDate   = np.min(dates)

    if eDate is None:
        eDate   = np.max(dates)

    return sDate, eDate

def get_xmax(sDate,eDate):
    return (eDate - sDate).total_seconds() / (86400.)

def create_music_run_list(radars,list_sDate,list_eDate,
        db_name='mstid',mongo_port=27017,
        mstid_format='guc_{radar}_{sDate}_{eDate}',
        use_input_list=False,
        input_db_name='mstid_aggregate',input_mongo_port=27017,
        input_mstid_format='guc_{radar}_{sDate}_{eDate}',
        music = False,
        **kwargs):
    """
    Generates a list of dictionaries with run parameters used by the MSTID
    MUSIC and Classification system.
    """
    if music:
        mstid_format   = 'music_guc_{radar}_{sDate}_{eDate}'

    dct_list = []
    for radar in radars:
        dct                     = {}
        dct['list_sDate']       = list_sDate
        dct['list_eDate']       = list_eDate
        dct['radar']            = radar
        date_fmt                = '%Y%m%d'
        sd_str                  = dct['list_sDate'].strftime(date_fmt)
        ed_str                  = dct['list_eDate'].strftime(date_fmt)
        dct['mstid_list']       = mstid_format.format(radar=radar,sDate=sd_str,eDate=ed_str)
        dct['db_name']          = db_name
        dct['mongo_port']       = mongo_port
        if use_input_list:
            dct['input_mstid_list'] = input_mstid_format.format(radar=radar,sDate=sd_str,eDate=ed_str)
            dct['input_db_name']    = input_db_name
            dct['input_mongo_port'] = input_mongo_port

        dct.update(kwargs)
        dct_list.append(dct)
    return dct_list

def create_group_dict(radars,list_sDate,list_eDate,group_name,group_dict={},**kwargs):
    """
    Adds a music_run_list to a dictionary.
    Each music_run_list in this dictionary is a "group".
    This makes it easy to group radars into "high latitude" and "mid latitude" groups.
    """

    dct_list                    = create_music_run_list(radars,list_sDate,list_eDate,**kwargs)

    key                         = len(list(group_dict.keys()))
    group_dict[key]             = {}
    group_dict[key]['dct_list'] = dct_list
    group_dict[key]['name']     = group_name

    return group_dict
        

def create_default_radar_groups(list_sDate=datetime.datetime(2014,11,1),list_eDate = datetime.datetime(2015,5,1),**kwargs):
    """
    Creates a radar group_dict for default sets of high latitude ('sas','pgr','kap','gbr')
    and mid latitude radars (cvw,cve,fhw,fhe,bks,wal).
    """

    # User-Defined Run Parameters Go Here. #########################################
    radar_groups = {}

    group_name      = 'High Latitude Radars'
    radars          = []
    radars.append('pgr')
    radars.append('sas')
    radars.append('kap')
    radars.append('gbr')
    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)

    group_name      = 'Mid Latitude Radars'
    radars          = []
    radars.append('cvw')
    radars.append('cve')
    radars.append('fhw')
    radars.append('fhe')
    radars.append('bks')
    radars.append('wal')
    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)

#    group_name      = 'West Looking'
#    radars          = []
#    radars.append('cvw')
#    radars.append('fhw')
#    radars.append('bks')
#    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)
#
#    group_name      = 'East Looking'
#    radars          = []
#    radars.append('cve')
#    radars.append('fhe')
#    radars.append('wal')
#    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)

    return radar_groups

def create_default_radar_groups_all_years(**kwargs):
    """
    Creates a list of default radar groups for all of the MSTID seasons I am looking at.
    """

    seDates = []
#    seDates.append( (datetime.datetime(2010,11,1),datetime.datetime(2011,5,1)) )
#    seDates.append( (datetime.datetime(2011,11,1),datetime.datetime(2012,5,1)) )
    seDates.append( (datetime.datetime(2012,11,1),datetime.datetime(2013,5,1)) )
#    seDates.append( (datetime.datetime(2013,11,1),datetime.datetime(2014,5,1)) )
#    seDates.append( (datetime.datetime(2014,11,1),datetime.datetime(2015,5,1)) )

    radar_group_list    = []
    for sDate,eDate in seDates:
        radar_group_list.append(create_default_radar_groups(sDate,eDate,**kwargs))

    return radar_group_list

def plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
        xlabels=True,db_name='mstid',mongo_port=27017,lambda_max=750.,
        highlight_ew=False,group_name=None,classification_colors=False,
        rasterized=False,**kwargs):

    mongo       = pymongo.MongoClient(port=mongo_port)
    db          = mongo[db_name]

    radar_dict  = {}
    radars      = []
    for dct in dct_list:
        radar   = dct['radar']
        radars.append(radar)

        radar_dict[radar]   = {}
        radar_dict[radar]['mstid_list'] = dct['mstid_list']

    ymax    = len(st_uts) * len(radars)

    cbar_info   = {}
    if val_key == 'music_azm':
        cbar_info['cbar_tick_fmt']  = '%.0f' 
        cbar_info['cbar_tb_vis']    = True
        cbar_info['cbar_label']     = 'MSTID\nGeog. Azim. [deg]'

        if not highlight_ew:
            cmap        = matplotlib.cm.hsv
            bounds      = np.linspace(0,360,360)
            cbar_info['cbar_ticks']     = [0, 90, 180, 270, 360]
        else:
            # Make sure red is East!!
            cmap        = matplotlib.cm.seismic_r
#            cmap        = gl.get_custom_cmap('blue_red')
            bounds      = np.linspace(90.,270.,180)
            cbar_info['cbar_ticks']     = [90, 135, 180, 225, 270]

            bounds      = np.linspace(90.,270.,180)
            cbar_ticks  = [90, 135, 180, 225, 270]

            cbar_info['cbar_ticks']     = cbar_ticks
    elif val_key == 'mstid_azm_dev':
        cbar_info['cbar_tick_fmt']  = '%.0f' 
        cbar_info['cbar_tb_vis']    = True
        cbar_info['cbar_label']     = 'Azm - $\sigma$ [deg]'

#        cmap        = gl.get_custom_cmap('blue_red')
        cmap        = matplotlib.cm.seismic
#        bounds      = np.linspace(-25,25,50)
#        cbar_info['cbar_ticks']     = [-25,-12.5,0,12.5,25]
        bounds      = np.linspace(-50,50,100)
        cbar_info['cbar_ticks']     = [-50,-25.0,0,25.0,50]
    else:
        cmap        = matplotlib.cm.jet
        bounds      = np.linspace(scale[0],scale[1],256)
        cbar_info['cbar_ticks'] = np.linspace(scale[0],scale[1],11)
        cbar_info['cbar_label'] = 'MSTID Index'

    norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

    if classification_colors:
        # Use colorscheme that matches MSTID Index in classification plots.
        from mstid.classify import MyColors
        scale_0             = -0.025
        scale_1             =  0.025
        my_cmap             = 'seismic'
        truncate_cmap       = (0.1, 0.9)
        my_colors           = MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)
        cmap            = my_colors.cmap
        norm            = my_colors.norm

    ################################################################################    
    current_date = sDate
    verts       = []
    vals        = []
    while current_date < eDate:
        for st_ut in st_uts:
            for radar in radars:
                win_sDate   = current_date + datetime.timedelta(hours=st_ut)
                mstid_list  = radar_dict[radar]['mstid_list']

                item        = db[mstid_list].find_one({'radar':radar,'sDatetime':win_sDate})
                if item is None: continue

                # Get the value to be plotted.
                if 'music_' in val_key:
                    sig_key = val_key.lstrip('music_')
                    
                    if highlight_ew:
                        azm_lim = (90.,270.)
                    else:
                        azm_lim = None
                    val = mongo_tools.get_mstid_value(item,sig_key,lambda_max=lambda_max,azm_lim=azm_lim)
                else:
                    if val_key not in item: continue
                    val = item.get(val_key)

                if val is None:
                    continue

                vals.append(val)
                verts.append(get_coords(radar,win_sDate,radars,sDate,eDate,st_uts))

        current_date += datetime.timedelta(days=1)

    pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,
            cmap=cmap,norm=norm,zorder=99,rasterized=rasterized)
    pcoll.set_array(np.array(vals))
    ax.add_collection(pcoll,autolim=False)

    # Make gray missing data.
    ax.set_facecolor('0.90')

    # Add radar labels.
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
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
        
        txt     = '{:02d}-{:02d} UT'.format(int(st_ut),int(st_ut+2))
        ypos   += len(radars)/2.
        xpos    = -0.025
#        ax.text(xpos,ypos,txt,transform=trans,
#                ha='center',va='center',rotation=90,fontdict=ytick_major_fontdict)
        xpos    = -0.015
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
    
    mongo.close()
    return ax_info



def calendar_plot(dct_list=None,group_dict=None,sDate=None,eDate=None,val_key='meanSubIntSpect_by_rtiCnt',
        scale=[-0.03,0.03], st_uts=[14, 16, 18, 20],driver=[None],
        output_dir='mstid_data/calendar',db_name='mstid',mongo_port=27017,
        fig_scale=40.,fig_scale_x=1.,fig_scale_y=0.225,
        h_pad = 0.150,
        mstid_reduced_inx=None,
        correlate=False,super_title=None,plot_radars=True,**kwargs):
    """
    dct_list: List of dictionaries with information of MongoDB collections to be ploted in the form of:
        [{'list_sDate': datetime.datetime(2012, 11, 1, 0, 0), 'list_eDate': datetime.datetime(2013, 5, 1, 0, 0), 'radar': 'pgr', 'mstid_list': 'guc_pgr_20121101_20130501', 'db_name': 'mstid', 'mongo_port': 27017}]

        dct_list may be omitted if a group_dict is supplied

    group_dict: Dictionary of dct_lists split up into groups that will be plotted.
    """
    
    driver_list         = gl.get_iterable(driver)

    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    gl.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    radar_panel_frac    = 1.0

    if (driver is not None) and (driver != [None]):
#        ax_ny += len(driver_list)
        ax_ny += 1
        radar_panel_frac = (ax_ny-1.) / ax_ny
        if len(driver_list) == 1:
            radar_panel_frac    = 0.6
        else:
            radar_panel_frac    = 0.5

    figsize     = (fig_scale*fig_scale_x,fig_scale*fig_scale_y*ax_ny)
    fig         = plt.figure(figsize=figsize)

    driver_panel_frac   = 1.-radar_panel_frac

    if (driver is not None) and (driver != [None]):
        driver_xlabels  = True
        ax_info = plot_the_drivers(sDate,eDate,driver_list,
                driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
                output_dir,mstid_reduced_inx,correlate,xlabels=driver_xlabels)
        ax_top  = ax_info['ax_top']

    #### Plot the radar data.
    radars      = []
    if plot_radars:
        keys    = list(group_dict.keys())
        keys.sort()
        nr_groups   = len(keys)
        for key in keys:
            dct_list    = group_dict[key]['dct_list']
            radars  += [dct['radar'] for dct in dct_list]

            ax_height   = get_radar_ax_frac(key,group_dict) * (radar_panel_frac - h_pad*nr_groups)
            ax_bottom   = ax_top - ax_height - h_pad
            ax_top      = ax_bottom
            ax          = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))

            if key == max(keys):
                xlabels = True
            else:
                xlabels = False
            ax_info     = plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
                    xlabels=xlabels,db_name=db_name,mongo_port=mongo_port,**kwargs)
            ax_list.append(ax_info)

        plot_cbars(ax_list)

    if super_title is None:
        date_fmt    = '%d %b %Y'
        super_title = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
    fig.text(0.5,1.005,super_title,ha='center',fontdict={'weight':'bold','size':48})

    date_fmt    = '%Y%m%d'
    filename    = []
    filename.append('calendar')
    filename.append(val_key)
    if driver[0] is not None:
        filename.append('_'.join(driver_list))
    filename.append('{}_{}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt)))
    filename.append('_'.join(radars))

    filepath    = os.path.join(output_dir,'_'.join(filename))+'.png'
    fig.savefig(filepath,bbox_inches='tight')
    print(('Calendar plot: {}'.format(filepath)))

    return filepath

if __name__ == '__main__':
    group_dicts  = create_default_radar_groups_all_years()
    for group_dict in group_dicts:
        calendar_plot(group_dict=group_dict)
    import ipdb; ipdb.set_trace()

