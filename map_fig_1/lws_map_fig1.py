#!/usr/bin/env python

import os
import shutil
import datetime
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade

import pickle

#from hamsci_psws import geopack

import pydarn
from pyDARNmusic import load_fitacf
import pyDARNmusic

import mstid

Re = 6371 # Radius of the Earth in km

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

def prep_dir(path,clear=False):
    if clear:
        if os.path.exists(path):
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def fan_plot(dataObject,
    dataSet                 = 'active',
    time                    = None,
    axis                    = None,
    scale                   = None,
    autoScale               = False,
    plotZeros               = False,
    markCell                = None,
    markBeam                = None,
    markBeam_dict           = {'color':'white','lw':2},
    plotTerminator          = True,
    parallels_ticks         = None,
    meridians_ticks         = None,
    zoom                    = 1.,
    lat_shift               = 0.,
    lon_shift               = 0.,
    cmap                    = None,
    plot_cbar               = True,
    cbar_ticks              = None,
    cbar_shrink             = 1.0,
    cbar_fraction           = 0.15,
    cbar_gstext_offset      = -0.075,
    cbar_gstext_fontsize    = None,
    model_text_size         = 'small',
    draw_coastlines         = True,
    plot_title              = True,
    title                   = None,
    projection              = ccrs.PlateCarree(),
    plot_fov                = True,
    **kwArgs):

    from pydarn import (SuperDARNRadars, Hemisphere)

    # Make some variables easier to get to...
    currentData = pyDARNmusic.utils.musicUtils.getDataSet(dataObject,dataSet)
    metadata    = currentData.metadata
    latFull     = currentData.fov["latFull"]
    lonFull     = currentData.fov["lonFull"]
    sdate       = currentData.time[0]
    coords      = metadata['coords']
    stid        = metadata['stid']
    radar       = metadata['code'].strip()

    # Get center of FOV.
    # Determine center beam.
    ctrBeamInx  = len(currentData.fov["beams"])/2
    ctrGateInx  = len(currentData.fov["gates"])/2
    ctrLat      = currentData.fov["latCenter"][int(ctrBeamInx),int(ctrGateInx)]
    ctrLon      = currentData.fov["lonCenter"][int(ctrBeamInx),int(ctrGateInx)]

    # Translate parameter information from short to long form.
    paramDict = pyDARNmusic.utils.radUtils.getParamDict(metadata['param'])
    if 'label' in paramDict:
        param     = paramDict['param']
        cbarLabel = paramDict['label']
    else:
        param = 'width' # Set param = 'width' at this point just to not screw up the colorbar function.
        cbarLabel = metadata['param']

    # Set colorbar scale if not explicitly defined.
    if(scale is None):
        if autoScale:
            sd          = sp.nanstd(np.abs(currentData.data),axis=None)
            mean        = sp.nanmean(np.abs(currentData.data),axis=None)
            scMax       = np.ceil(mean + 1.*sd)
            
            if np.min(currentData.data) < 0:
                scale   = scMax*np.array([-1.,1.])
            else:
                scale   = scMax*np.array([0.,1.])
        else:
            if 'range' in paramDict:
                scale = paramDict['range']
            else:
                scale = [-200,200]

    if stid:
        radar_lat = SuperDARNRadars.radars[stid].hardware_info.geographic.lat
        radar_lon = SuperDARNRadars.radars[stid].hardware_info.geographic.lon

    fig   = axis.get_figure()

    # Figure out which scan we are going to plot...
    if time is None:
        timeInx = 1
    else:
        timeInx = (np.where(currentData.time >= time))[0]
        if np.size(timeInx) == 0:
            timeInx = -1
        else:
            timeInx = int(np.min(timeInx))
            

    # do some stuff in map projection coords to get necessary width and height of map
    lonFull,latFull = (np.array(lonFull)+360.)%360.,np.array(latFull)

    goodLatLon  = np.logical_and( np.logical_not(np.isnan(lonFull)), np.logical_not(np.isnan(latFull)) )
    goodInx     = np.where(goodLatLon)
    goodLatFull = latFull[goodInx]
    goodLonFull = lonFull[goodInx]

    # Plot the SuperDARN data!
    ngates = np.shape(currentData.data)[2]
    nbeams = np.shape(currentData.data)[1]
    data  = currentData.data[timeInx,:,:]
    verts = []
    scan  = []
    # data  = currentData.data[timeInx,:,:]
    goodBmRg=[]
    geo = ccrs.Geodetic()
    for bm in range(nbeams):
        for rg in range(ngates):
            if goodLatLon[bm,rg] == False: continue
            if np.isnan(data[bm,rg]): continue
            if data[bm,rg] == 0 and not plotZeros: continue
            goodBmRg.append((bm,rg))
            scan.append(data[bm,rg])
            x1,y1 = projection.transform_point(lonFull[bm+0,rg+0],latFull[bm+0,rg+0],geo)
            x2,y2 = projection.transform_point(lonFull[bm+1,rg+0],latFull[bm+1,rg+0],geo)
            x3,y3 = projection.transform_point(lonFull[bm+1,rg+1],latFull[bm+1,rg+1],geo)
            x4,y4 = projection.transform_point(lonFull[bm+0,rg+1],latFull[bm+0,rg+1],geo)
            verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))
    
    # np.savetxt('lat.csv', good_lat, delimiter=',')
    # np.savetxt('data.csv', good_data, delimiter=',')

    if cmap is None:
        cmap    = mpl.cm.jet
    bounds  = np.linspace(scale[0],scale[1],256)
    norm    = mpl.colors.BoundaryNorm(bounds,cmap.N)

#        pcoll = mpl.collections.PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,cmap=cmap,norm=norm,zorder=99)
    pcoll = mpl.collections.PolyCollection(np.array(verts),edgecolors='face',closed=False,cmap=cmap,norm=norm,zorder=99)
    pcoll.set_array(np.array(scan))
    axis.add_collection(pcoll,autolim=False)

    if plot_fov:
        # Left Edge
        xx = lonFull[0,:]
        yy = latFull[0,:]
        axis.plot(xx,yy,color='k',transform=ccrs.PlateCarree())

        # Right Edge
        xx = lonFull[-1,:]
        yy = latFull[-1,:]
        axis.plot(xx,yy,color='k',transform=ccrs.PlateCarree())

        # Bottom Edge 
        xx = lonFull[:,0]
        yy = latFull[:,0]
        axis.plot(xx,yy,color='k',transform=ccrs.PlateCarree())

        # Top Edge
        xx = lonFull[:,-1]
        yy = latFull[:,-1]
        axis.plot(xx,yy,color='k',transform=ccrs.PlateCarree())

        # Radar Location
        axis.scatter(radar_lon,radar_lat,marker='o',color='k',s=40,transform=ccrs.PlateCarree())
        fontdict = {'size':14,'weight':'bold'}
        if radar == 'cvw' or radar == 'fhw':
            ha      = 'right'
            text    = radar.upper() + ' '
        else:
            ha      = 'left'
            text    = ' ' + radar.upper()

        axis.text(radar_lon,radar_lat,text,ha=ha,
                fontdict=fontdict,transform=ccrs.PlateCarree())
    
    result  = {}
    result['pcoll']     = pcoll
    result['metadata']  = metadata
    result['cbarLabel'] = cbarLabel

    return result

def load_data(load_sTime=None,load_eTime=None,cache_base='cache',clear_cache=False,
        data_dir='/data/sd-data_despeck',use_preprocessed = False,**kwargs):
    radars_dct  = {}
    radars_dct['pgr'] = {}
    radars_dct['sas'] = {}
    radars_dct['kap'] = {}
    radars_dct['gbr'] = {}
    radars_dct['cvw'] = {}
    radars_dct['cve'] = {}
    radars_dct['fhw'] = {}
    radars_dct['fhe'] = {}
    radars_dct['bks'] = {}
    radars_dct['wal'] = {}

    if load_sTime is None:
        load_sTime = kwargs.get('sTime')

    if load_eTime is None:
        load_eTime = kwargs.get('eTime')

    cache_dir = os.path.join(cache_base,data_dir.lstrip('/'))
    prep_dir(cache_dir,clear=clear_cache)

    for radar,dct in radars_dct.items(): 
        if use_preprocessed:
            dataObj         = mstid.more_music.get_dataObj(radar,load_sTime,load_eTime,data_path='mstid_data/mstid_index')
        else:
            cache_fname = '{!s}.{!s}-{!s}.p'.format(radar,load_sTime.strftime('%Y%m%d.%H%M'),load_eTime.strftime('%Y%m%d.%H%M'))
            cache_fpath  = os.path.join(cache_dir,cache_fname)

            if not os.path.exists(cache_fpath):
                fovModel                = 'GS'
                gscat                   = 1 # Ground scatter only.
#                gscat                   = 0 # All scatter.
                beam_limits             = (None, None)
                gate_limits             = (0,60)
                interp_resolution       = 60.
                filter_numtaps          = 101.
    #            process_level           = 'music'
    #            auto_range_on           = True
                bad_range_km            = None
    #            filter_cutoff_low       = 0.0003
    #            filter_cutoff_high      = 0.0012
    #            detrend                 = True
    #            hanning_window_space    = True
    #            hanning_window_time     = True
    #            zeropad                 = True
    #            kx_max                  = 0.05
    #            ky_max                  = 0.05
    #            autodetect_threshold    = 0.35
    #            neighborhood            = (10,10)

                dataObj = mstid.more_music.create_music_obj(radar.lower(), load_sTime, load_eTime
                    ,beam_limits                = beam_limits
                    ,gate_limits                = gate_limits
                    ,interp_resolution          = interp_resolution
                    ,filterNumtaps              = filter_numtaps 
                    ,srcPath                    = None
                    ,fovModel                   = fovModel
                    ,gscat                      = gscat
                    ,data_dir                   = data_dir
                    )

                if len(dataObj.get_data_sets()) == 0:
                    dataObj = None
                else:
                    gate_limits = mstid.more_music.auto_range(radar,load_sTime,load_eTime,dataObj,bad_range_km=bad_range_km)

                    pyDARNmusic.boxcarFilter(dataObj)
#                    pyDARNmusic.defineLimits(dataObj,gateLimits=gate_limits)

                    dataObj.active.applyLimits()

                    pyDARNmusic.beamInterpolation(dataObj,dataSet='limitsApplied')
                    pyDARNmusic.determineRelativePosition(dataObj)

                    pyDARNmusic.timeInterpolation(dataObj,timeRes=interp_resolution)
                    pyDARNmusic.nan_to_num(dataObj)

                    mstid.more_music.calculate_terminator_for_dataSet(dataObj)

                with open(cache_fpath,'wb') as fl:
                    pickle.dump(dataObj,fl)

            else:
                print('Using cached file: {!s}'.format(cache_fpath))
                with open(cache_fpath,'rb') as fl:
                    dataObj = pickle.load(fl)

        if dataObj is not None:
            print('Loaded: {!s}'.format(radar))
        else:
            print('NO DATA for {!s}'.format(radar))
        dct['dataObj']  = dataObj
    return radars_dct

def plot_map(radars_dct,time,dataSet='active',output_dir='output',**kwargs):
    projection = ccrs.Orthographic(-100,60)
    fig = plt.figure(figsize=(18,14))
    ax  = fig.add_subplot(1,1,1,projection=projection)
#    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.add_feature(cfeature.OCEAN, color = 'white')
    ax.add_feature(cfeature.LAKES, color='white')
    ax.add_feature(Nightshade(time, alpha=0.2))
    
    ax.gridlines(draw_labels=['left','bottom'])

    for radar,dct in radars_dct.items():
        dataObj = dct.get('dataObj')
        if dataObj is None:
            continue

        result = fan_plot(dataObj,dataSet=dataSet,
                axis=ax,projection=projection,time=time,scale=(0,30))

    cbar = fig.colorbar(result['pcoll'],orientation='vertical',shrink=0.60,fraction=0.15)
    cbar.set_label(result['cbarLabel'])
    labels = cbar.ax.get_yticklabels()
    labels[-1].set_visible(False)

    if 'gscat' in result['metadata']:
        if result['metadata']['gscat'] == 1:
            cbar.ax.text(0.5,-0.075,'Ground\nscat\nonly',ha='center',fontsize=None,transform=cbar.ax.transAxes)

    txt = 'Coordinates: ' + result['metadata']['coords'] +', Model: ' + result['metadata']['model']
    ax.text(1.01, 0, txt,
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation='vertical',
              size='small',
              weight='bold',
              transform=ax.transAxes)

    ax.set_extent((-140,-45,20,70))

    fontdict    = {'size':'x-large','weight':'bold'}
    title       = time.strftime('%d %b %Y %H%M UTC')
    ax.set_title(title,fontdict=fontdict)

    fig.tight_layout()

    fname = 'map_{!s}.png'.format(time.strftime('%Y%m%d.%H%M'))
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')


def plot_rtp(radars_dct,sTime,eTime,dataSet='active',output_dir='output',**kwargs):

    nrows   = len(radars_dct)
    ncols   = 1
    figsize = (20,4*nrows)
    fig     = plt.figure(figsize=figsize)

    ax_inx  = 0
    for radar,dct in radars_dct.items():
        beam = dct.get('beam',7)

        ax_inx += 1
        ax  = fig.add_subplot(nrows,ncols,ax_inx)

        fontdict    = {'fontsize':'x-large','weight':'bold'}
        bbox        = {'boxstyle':'round','facecolor':'white','alpha':1.0}
        text        = '{!s} Beam {!s}'.format(radar.upper(),beam)
        ax.text(0.01,0.95,text,va='top',zorder=1000,
                transform=ax.transAxes,fontdict=fontdict,bbox=bbox)

        if ax_inx == 1:
            fmt   = '%Y %b %d %H%M UTC'
            title = '{!s} - {!s}'.format(sTime.strftime(fmt),eTime.strftime(fmt))
            fontdict={'fontsize':'x-large','weight':'bold'}
            ax.set_title(title,fontdict=fontdict)

        dataObj = dct.get('dataObj')
        if dataObj is None:
            continue
        pyDARNmusic.plotting.rtp.musicRTP(dataObj,xlim=(sTime,eTime),axis=ax,dataSet=dataSet,
                beam=beam,plot_info=False,plot_title=False)



#    fig.tight_layout()
    fname   = 'rtp_{!s}-{!s}.png'.format(sTime.strftime('%Y%m%d.%H%M'),eTime.strftime('%Y%m%d.%H%M'))
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')

if __name__ == '__main__':
    rd = {}

    rd['output_dir'] = 'output'
    prep_dir(rd['output_dir'])

#    rd['sTime']         = datetime.datetime(2018,12,9,12)
#    rd['eTime']         = datetime.datetime(2018,12,9,13)
#    rd['time']          = datetime.datetime(2018,12,9,12,30)
#    rd['eTime']         = datetime.datetime(2018,12,10)
#    rd['time']          = datetime.datetime(2018,12,9,19,0)

#    rd['sTime']         = datetime.datetime(2019,1,9,12)
#    rd['eTime']         = datetime.datetime(2019,1,10)
#    rd['time']          = datetime.datetime(2019,1,9,19,00)

    rd['sTime']         = datetime.datetime(2012,12,21,14)
    rd['eTime']         = datetime.datetime(2012,12,21,22)
    rd['time']          = datetime.datetime(2012,12,21,16,10)

    rd['data_dir']      = '/data/sd-data'
#    rd['data_dir']      = '/data/sd-data_despeck'
    rd['clear_cache']   = False
    rd['radars_dct']    = load_data(**rd)

#    rd['dataSet']       = 'originalFit'
    rd['dataSet']       = 'limitsApplied'
#    rd['dataSet']       = 'active'

    plot_map(**rd)
    plot_rtp(**rd)
    import ipdb; ipdb.set_trace()
