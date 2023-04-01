#!/usr/bin/env python

import os
import datetime
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

#from hamsci_psws import geopack

import pydarn
from pyDARNmusic import load_fitacf
import pyDARNmusic

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
        # import ipdb;ipdb.set_trace()
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
    # import ipdb;ipdb.set_trace()
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
    
#    dataName = currentData.history[max(currentData.history.keys())] # Label the plot with the current level of data processing.
#    if plot_title:
#        if title is None:
#            axis.set_title(metadata['name']+' - '+dataName+currentData.time[timeInx].strftime('\n%Y %b %d %H%M UT')) 
#        else:
#            axis.set_title(title)
#
#    if plot_cbar:
#        cbar = fig.colorbar(pcoll,orientation='vertical',shrink=cbar_shrink,fraction=cbar_fraction)
#        cbar.set_label(cbarLabel)
#        if cbar_ticks is None:
#            labels = cbar.ax.get_yticklabels()
#            labels[-1].set_visible(False)
#        else:
#            cbar.set_ticks(cbar_ticks)
#
#        if 'gscat' in currentData.metadata:
#            if currentData.metadata['gscat'] == 1:
#                cbar.ax.text(0.5,cbar_gstext_offset,'Ground\nscat\nonly',ha='center',fontsize=cbar_gstext_fontsize,transform=cbar.ax.transAxes)
#
#    txt = 'Coordinates: ' + metadata['coords'] +', Model: ' + metadata['model']
#    axis.text(1.01, 0, txt,
#              horizontalalignment='left',
#              verticalalignment='bottom',
#              rotation='vertical',
#              size=model_text_size,
#              weight='bold',
#              transform=axis.transAxes)

def plot_map(output_dir='output'):
    radar   = 'bks'
    sDate   = datetime.datetime(2018,12,27,12)
    eDate   = datetime.datetime(2018,12,27,14)
    date    = sDate

    fit_sfx = "fitacf"
    data_dir = f'/sd-data/'
    fitacf  = load_fitacf(radar,sDate,eDate,data_dir=data_dir)
    dataObj = pyDARNmusic.music.musicArray(fitacf,sTime=sDate,eTime=eDate,fovModel='GS')

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18,14))
    ax  = fig.add_subplot(1,1,1,projection=projection)
    ax.coastlines()
#    ax.add_feature(cfeature.LAND, color='lightgrey')
#    ax.add_feature(cfeature.OCEAN, color = 'white')
    
    ax.gridlines(draw_labels=True)

    fan_plot(dataObj,axis=ax,projection=projection)

    # # World Limits
    # ax.set_xlim(-180,180)
    # ax.set_ylim(-90,90)

    # US Limits
    ax.set_xlim(-130,-60)
    ax.set_ylim(20,55)

    fig.tight_layout()

    fname = 'map_{!s}.png'.format(date.strftime('%Y%m%d.%H%M'))
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')


if __name__ == '__main__':
    output_dir = 'output'
    prep_dir(output_dir)

    plot_map(output_dir=output_dir)
    import ipdb; ipdb.set_trace()
