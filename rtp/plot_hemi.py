#!/usr/bin/env python3
import os
import bz2
import glob
import datetime
import tqdm

import numpy as np

import pydarnio
import pydarn

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import multiprocessing

mpl.rcParams['font.size']        = 16
mpl.rcParams['font.weight']      = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.grid']        = True
mpl.rcParams['grid.linestyle']   = ':'
#mpl.rcParams['figure.figsize']   = np.array([15, 8])
mpl.rcParams['axes.xmargin']     = 0

class ScriptTimer(object):
    def __init__(self):
        self.sTime  = datetime.datetime.now()
        self.script = os.path.basename(__file__)

        print('{!s} Running...'.format(self.script))
        print('   Started: {!s}'.format(self.sTime))
        print()

    def stop(self):
        self.eTime  = datetime.datetime.now()
        total       = self.eTime - self.sTime

        print()
        print('#--------------------------------------#')
        print('{!s} Finished...'.format(self.script))
        print('   Started:  {!s}'.format(self.sTime))
        print('   Finished: {!s}'.format(self.eTime))
        print('   Duration: {!s}'.format(total))
        print('#--------------------------------------#')
        print()

def date_range(date_0,date_1):
    """
    Return all of the dates in a range of dates.
    """
    dates   = [date_0]
    while dates[-1] < (date_1-datetime.timedelta(days=1)):
        nextDate    = dates[-1] + datetime.timedelta(days=1)
        dates.append(nextDate)
    return dates

def gen_radar_list():
    """
    Generate a dictionary with lists of the north and south hemisphere radars.
    """
    rads    = {}
    rads['north']   = []
    rads['south']   = []

    for rid,radar in pydarn.SuperDARNRadars.radars.items():
        abr = radar.hardware_info.abbrev
        if radar.hemisphere is pydarn.Hemisphere.North:
            rads['north'].append(abr)
        elif radar.hemisphere is pydarn.Hemisphere.South:
            rads['south'].append(abr)

    return rads

def load_fitacf(sTime,eTime,radar,sd_path='/data/sd-data',fit_sfx='fitacf'):
    """
    Load FITACF data from multiple FITACF files by specifying a date range.

    This routine assumes bz2 compression.
    """
    sDate   = datetime.datetime(sTime.year,sTime.month,sTime.day)
    eDate   = datetime.datetime(eTime.year,eTime.month,eTime.day)

    # Create a list of days we need fitacf files from.
    dates   = [sDate]
    while dates[-1] < eDate:
        next_date   = dates[-1] + datetime.timedelta(days=1)
        dates.append(next_date)

    # Find the data files that fall in that date range.
    fitacf_paths_0    = []
    for date in dates:
        date_str        = date.strftime('%Y%m%d')
#        /sd-data/2021/fitacf/bks
        data_dir    = os.path.join(sd_path,date.strftime('%Y'),fit_sfx,radar)
        fpattern        = os.path.join(data_dir,'{!s}*{!s}*.{!s}.bz2'.format(date_str,radar,fit_sfx))
        fitacf_paths_0   += glob.glob(fpattern)

    # Sort the files by name.
    fitacf_paths_0.sort()

    # Get rid of any files we don't need.
    fitacf_paths = []
    for fpath in fitacf_paths_0:
        date_str    = os.path.basename(fpath)[:13]
        this_time   = datetime.datetime.strptime(date_str,'%Y%m%d.%H%M')

        if this_time <= eTime:
            fitacf_paths.append(fpath)

    # Load and append each data file.

    print()
    fitacf = []
    for fitacf_path in tqdm.tqdm(fitacf_paths,desc='Loading {!s} Files'.format(fit_sfx),dynamic_ncols=True):
        tqdm.tqdm.write(fitacf_path)
        with bz2.open(fitacf_path) as fp:
            fitacf_stream = fp.read()

        reader  = pydarnio.SDarnRead(fitacf_stream, True)
        records = reader.read_fitacf()
        fitacf += records
    return fitacf

def plot_hemisphere(sTime,eTime=None,hemisphere='south',beam=7,ymin=200,ymax=3000,zmin=0,zmax=30,
        n_rows=5,n_cols=3,figsize=(40,25),output_dir='plots',replot=False):

    if eTime is None:
        eTime = sTime + datetime.timedelta(days=1)
    
    # Define output filename and filepath.
    sTime_str   = sTime.strftime('%Y%m%d.%H%M')
    eTime_str   = eTime.strftime('%Y%m%d.%H%M')
    fname       = '{!s}-{!s}_{!s}.rti.png'.format(sTime_str,eTime_str,hemisphere)
    fpath       = os.path.join(output_dir,fname)

    # If the plot already exists and replot is False, don't plot again.
    if not replot:
        if os.path.exists(fpath):
            return

    if hemisphere.lower() == 'north':
        # 25 Northern Hemisphere Radars
#        radars  = ['ade', 'adw', 'bks', 'cve', 'cvw', 'cly', 'fhe', 'fhw', 'gbr', 'han', 'hok', 'hkw', 'inv', 'jme', 'kap', 'ksr', 'kod', 'lyr', 'pyk', 'pgr', 'rkn', 'sas', 'sch', 'sto', 'wal']
        radars  = ['bks', 'cve', 'fhw', 'gbr', 'han', 'hok', 'inv', 'kap', 'ksr', 'kod', 'pgr', 'rkn', 'sas', 'sto', 'wal']
    else:
        # 14 Southern Hemisphere Radars
        radars  = ['bpk', 'dce', 'dcn', 'fir', 'hal', 'ker', 'mcm', 'san', 'sps', 'sye', 'sys', 'tig', 'unw', 'zho']


    fig = plt.figure(figsize=figsize)

    ax_inx  = 0
    for radar in radars:
        ax_inx += 1
        ax  = fig.add_subplot(n_rows,n_cols,ax_inx)

        fitacf      = load_fitacf(sTime,eTime,radar)
        try:
            pydarn.RTP.plot_range_time(fitacf, beam_num=beam, parameter='p_l', zmin=zmin, zmax=zmax, cmap=None,
                    date_fmt='%H%M', colorbar_label='Power (dB)', ax=ax)
        except:
            ax.axis('off')
            txt = []
            txt.append('No Data / Plotting Error')
            txt.append('len(fitacf) = {!s}'.format(len(fitacf)))

            ax.text(0.5,0.5,'\n'.join(txt),transform=ax.transAxes,
                    fontdict={'weight':'normal','size':24},ha='center',va='center')

        fmt   = '%Y %b %d %H%M UT'
#        txt = ('{!s} {!s} {!s} - {!s}'.format(radar.upper(),beam,sTime.strftime(fmt),eTime.strftime(fmt)))
        txt = '{!s} {!s}'.format(radar.upper(),beam)
        ax.text(0.01,0.88,txt,fontdict={'weight':'bold','size':36},transform=ax.transAxes)

#        ax.set_ylabel('Range Gate')
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(sTime,eTime)

        ax.set_xlabel('Time [UT]')


    fmt   = '%Y %b %d %H%M UT'
    if (eTime - sTime) == datetime.timedelta(days=1):
        txt = ('{!s}ern Hemisphere {!s}'.format(hemisphere.title(),sTime.strftime(fmt)))
    else:
        txt = ('{!s}ern Hemisphere {!s} - {!s}'.format(hemisphere.title(),sTime.strftime(fmt),eTime.strftime(fmt)))
    fig.text(0.5,1.010,txt,fontdict={'weight':'bold','size':44},ha='center')

    fig.tight_layout()

    print('Saving: {!s}'.format(fpath))
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

def plot_hemisphere_dict(run_dict):
    return plot_hemisphere(**run_dict)

if __name__ == '__main__':
    timer = ScriptTimer()

    multiproc   = True
    ncpus       = multiprocessing.cpu_count()

    date_0      = datetime.datetime(2010,1,1)
    date_1      = datetime.datetime(2017,1,1)
    dates       = date_range(date_0,date_1)

    hemisphere  = 'south'

    output_dir  = os.path.join('plots',hemisphere)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Generate list of run dictionaries.
    run_dicts = []
    for date in dates:
        rd = {}
        rd['sTime']         = date
        rd['hemisphere']    = hemisphere
        rd['output_dir']    = output_dir

        run_dicts.append(rd)

    ## Load and plot SuperDARN Data
    if multiproc:
        with multiprocessing.Pool(ncpus) as pool:
            pool.map(plot_hemisphere_dict,run_dicts)
    else:
        # Single Processor
        for run_dict in run_dicts:
            fpath = plot_hemisphere_dict(run_dict)

    timer.stop()
