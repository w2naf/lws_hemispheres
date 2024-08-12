#!/usr/bin/env python
"""
This class will plot a day of amateur radio RBN/PSKReporter/WSPRNet data time series with edge
detection and spot maps.
"""
import os
import datetime
import pickle
import bz2

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

def fmt_xaxis(ax,xlim=None,label=True,fontdict={}):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]',fontdict=fontdict)
    ax.set_xlim(xlim)

class HamSpotPlot(object):
    def __init__(self,
            date            = datetime.datetime(2018,12,15),
            sTime           = datetime.datetime(2018,12,15,13),
            eTime           = datetime.datetime(2018,12,15,23),
            range_lim_km    = (750,2250),
            frang_khz       = (14000,14350),
            midpoint_region = 'US',
            data_dir        = None):

        if data_dir is None:
            data_dir    = os.path.join('data','lstid_ham')

        self.data_dir        = data_dir
        self.date            = date
        self.frang_khz       = frang_khz
        self.midpoint_region = midpoint_region
        self.sTime           = sTime
        self.eTime           = eTime
        self.range_lim_km    = range_lim_km

        self.load_raw_spots()

        self.load_geo_data()
        self.load_edge_data()

    def load_raw_spots(self):
        data_dir        = os.path.join(self.data_dir,'raw_spots')
        date            = self.date
        frang_khz       = self.frang_khz
        midpoint_region = self.midpoint_region
        sTime           = self.sTime
        eTime           = self.eTime
        range_lim_km    = self.range_lim_km

        networks    = []
        networks.append('WSPR')
        networks.append('RBN')
        networks.append('PSK')

        # Build Cache Name
        clst    = [date.strftime('%Y%m%d')]
        if sTime is not None:
            clst.append(sTime.strftime('%Y%m%d.%H%M'))
        if eTime is not None:
            clst.append(eTime.strftime('%Y%m%d.%H%M'))
        for network in networks:
            clst.append(network)
        if midpoint_region is not None:
            clst.append(midpoint_region)
        if frang_khz is not None:
            clst.append('{!s}-{!s}kHz'.format(*frang_khz))
        if range_lim_km is not None:
            clst.append('{!s}-{!s}km'.format(*range_lim_km))
        
        cname   = '_'.join(clst)+'.csv.bz2'
        cpath   = os.path.join(data_dir,cname)
        
        if os.path.exists(cpath):
            print('LOADING CACHED FILE: {!s}'.format(cpath))
            df  = pd.read_csv(cpath,parse_dates=['timestamp'],comment='#')
        else:
            df          = pd.DataFrame()
            date_str    = self.date.strftime('%Y-%m-%d')
            for network in networks:
                tic     = datetime.datetime.now()
                fname   = f'{date_str}_{network}.csv.bz2'
                fpath   = os.path.join(data_dir,fname)
                if not os.path.exists(fpath):
                    print(f'FILE NOT FOUND: {fpath}')
                    continue

                print(f'LOADING RAW SPOTS: {fpath}')

                cols     = {}
                cols[0]  = 'timestamp'
                cols[1]  = 'call_0'
                cols[3]  = 'lat_0'
                cols[4]  = 'lon_0'
                cols[6]  = 'call_1'
                cols[8]  = 'lat_1'
                cols[9]  = 'lon_1'
                cols[11]  = 'f_hz'
                cols[22] = 'range_km'
                cols[23] = 'lat_mid'
                cols[24] = 'lon_mid'
                usecols  = list(cols.keys())
                names    = list(cols.values())
                dft      = pd.read_csv(fpath,header = None,usecols = usecols,names = names)
                dft_0    = pd.read_csv(fpath,header = None)

                # Filter data by frequency.
                if frang_khz is not None:
                    frang_hz = np.array(frang_khz)*1e3
                    tf  = np.logical_and(dft['f_hz'] >= min(frang_hz), dft['f_hz'] < max(frang_hz))
                    dft = dft[tf]

                # Filter data by midpoint region.
                if midpoint_region is not None:
                    rgn = regions['US']
                    lat_lim = rgn['lat_lim']
                    lon_lim = rgn['lon_lim']
                    
                    lat_tf  = np.logical_and(dft['lat_mid'] >= lat_lim[0], dft['lat_mid'] < lat_lim[1])
                    lon_tf  = np.logical_and(dft['lon_mid'] >= lon_lim[0], dft['lon_mid'] < lon_lim[1])
                    tf      = np.logical_and(lat_tf,lon_tf)
                    dft     = dft[tf]

                # Filter by range_km.
                if range_lim_km is not None:
                    tf  = np.logical_and(dft['range_km'] >= min(range_lim_km),
                                         dft['range_km'] <  max(range_lim_km))
                    dft = dft[tf]

                # Parse timestamps
                dft['timestamp']    = dft['timestamp'].apply(pd.Timestamp)

                # Filter by times.
                if sTime is not None:
                    tf  = dft['timestamp'] >= sTime
                    dft = dft[tf]

                if eTime is not None:
                    tf  = dft['timestamp'] < eTime
                    dft = dft[tf]

                # Append Network Name
                dft['network']      = network

                toc     = datetime.datetime.now()
                print('   Loading Time: {!s}'.format(toc-tic))
                    
                # Create/append to one large dataframe with all requested networks.
                df = pd.concat([df,dft],ignore_index=True)

                # Sort values by timestamp.
                df  = df.sort_values('timestamp')

                # Build Header
                hdr = []
                hdr.append('# Amateur Radio Communications Spot Data File')
                hdr.append('#')
                hdr.append('# Networks: {!s}'.format(', '.join(networks)))
                hdr.append('# Date: {!s}'.format(date.strftime('%Y %b %d')))
                hdr.append('# Start Time [UTC]: {!s}'.format(sTime))
                hdr.append('# End Time [UTC]: {!s}'.format(eTime))
                hdr.append('# Midpoint Region: {!s}'.format(midpoint_region))
                rgn = regions.get(midpoint_region)
                if rgn is not None:
                    for rkey, rval in rgn.items():
                        hdr.append(f'#   {rkey}: {rval}')
                hdr.append('# Frequency Range [kHz]: {!s}'.format(frang_khz))
                hdr.append('# Range Limits [km]: {!s}'.format(range_lim_km))
                hdr.append('#')
                hdr.append('# N Spots: {!s}'.format(len(df)))
                hdr.append('#\n')
                hdr     = '\n'.join(hdr)
                csv_str = hdr+df.to_csv(None,index=False)

                with bz2.open(cpath,'wt') as fl:
                    fl.write(csv_str)

                import ipdb; ipdb.set_trace()

        import ipdb; ipdb.set_trace()
    
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
        date_str        = self.date.strftime('%Y_%m_%d')
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

    def plot_map_ax(self,fig,
            subplot             = (1,1,1),
            panel_rect          = None,
            plot_region         = 'US',
            title_size          = None,
            ticklabel_size      = None,
            label_size          = None,
            cbar_ticklabel_size = None,
            cbar_label_size     = None):

        if panel_rect is not None:
            ax = fig.add_axes(panel_rect,projection=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(*subplot,projection=ccrs.PlateCarree())

        # Plot Map #############################
        ax.coastlines(zorder=10,color='w')
        map_data        = self.geo_hist
        map_data.name   = '14 MHz Midpoints'

        tf          = map_data < 1
        map_n       = int(np.sum(map_data))
        map_data    = np.log10(map_data)
        map_data.values[tf] = 0
        map_data.name   = 'log({})'.format(map_data.name)
        cntr    = map_data.plot.contourf(x=map_data.attrs['xkey'],y=map_data.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
        cax     = cntr.colorbar.ax
        if cbar_ticklabel_size is not None:
            for ytl in cax.get_yticklabels():
                ytl.set_size(cbar_ticklabel_size)
        if cbar_label_size is not None:
            cax.set_ylabel(map_data.name,fontdict={'size':cbar_label_size})

        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}
        if label_size is not None:
            fdict.update({'size':label_size})

        ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(map_n),
                ha='center',transform=ax.transAxes,fontdict=fdict)

        if plot_region is not None:
            rgn         = regions.get(plot_region)
            lat_lim     = rgn.get('lat_lim')
            lon_lim     = rgn.get('lon_lim')

            ax.set_xlim(lon_lim)
            ax.set_ylim(lat_lim)

        if ticklabel_size is not None:
            for ttl in ax.get_xticklabels():
                ttl.set_size(ticklabel_size)

            for ttl in ax.get_yticklabels():
                ttl.set_size(ticklabel_size)

        date_str = self.date.strftime('%Y %b %d')
        title_fd = {'weight':'bold'}
        if title_size is not None:
            title_fd.update({'size':title_size})
        ax.set_title(date_str,fontdict=title_fd)
    
    def plot_timeSeries_ax(self,ax,xlim=None,ylim=None,
            plot_fit            = True,
            plot_CV             = False,
            cb_pad              = 0.125,
            title_size          = None,
            ticklabel_size      = None,
            label_size          = None,
            cbar_ticklabel_size = None,
            cbar_label_size     = None,
            cbar_label          = '14 MHz\nHam Radio Data'):

        fig             = ax.get_figure()

        result_dct      = self.edge_data
        md              = result_dct.get('metaData')
        date            = md.get('date')
        if xlim is None:
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
        cbar = plt.colorbar(mpbl,aspect=10,pad=cb_pad,label=cbar_label)
        cax  = cbar.ax
        if cbar_ticklabel_size is not None:
            for ytl in cax.get_yticklabels():
                ytl.set_size(cbar_ticklabel_size)
        if cbar_label_size is not None:
            cax.set_ylabel(cbar_label,fontdict={'size':cbar_label_size})

        if plot_fit:
            ed0_line    = ax.plot(arr_times,edge_0,lw=2,color='Aqua',label='Detected Edge')

            if p0_sin_fit != {}:
                ax.plot(sin_fit.index,sin_fit+poly_fit,label='Sin Fit',color='white',lw=4,ls='--')

            if plot_CV:
                ax2 = ax.twinx()
                ax2.plot(stability.index,stability,lw=2,color='0.5')
                ax2.grid(False)
                ax2.set_ylabel('Edge Coef. of Variation\n(Grey Line)')

            for wl in winlim:
                ax.axvline(wl,color='0.8',ls='--',lw=2)

            for wl in fitWinLim:
                ax.axvline(wl,color='lime',ls='--',lw=2)

            ax.legend(loc='upper center',fontsize='x-small',ncols=4,
                    framealpha=0.2,labelcolor='linecolor')

        label_fd = {}
        if label_size is not None:
            label_fd.update({'size':label_size})

        fmt_xaxis(ax,xlim,fontdict=label_fd)
        ax.set_ylabel('Range [km]',fontdict=label_fd)
        ax.set_ylim(ylim)

        if ticklabel_size is not None:
            for ttl in ax.get_xticklabels():
                ttl.set_size(ticklabel_size)

            for ttl in ax.get_yticklabels():
                ttl.set_size(ticklabel_size)

        date_str = self.date.strftime('%Y %b %d')
        title_fd = {'weight':'bold'}
        if title_size is not None:
            title_fd.update({'size':title_size})
        ax.set_title(date_str,fontdict=title_fd)

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
