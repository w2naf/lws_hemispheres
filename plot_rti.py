#!/usr/bin/env python3
import os
import bz2
import glob
import datetime
import tqdm

import pydarnio
import pydarn

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


def load_fitacf(sTime,eTime,radar,data_dir='data',fit_sfx='fitacf3'):
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


if __name__ == '__main__':
    output_dir  = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sTime       = datetime.datetime(2017,11,1)
    eTime       = datetime.datetime(2017,11,2)
    radar       = 'bks'
    fitacf      = load_fitacf(sTime,eTime,radar)

    hdw_data    = pydarn.read_hdw_file(radar,sTime)

    n_beams     = hdw_data.beams

    n_rows      = n_beams
    n_cols      = 1

    fig = plt.figure(figsize=(20,3*n_rows))

    ax_inx  = 0
    for beam in range(n_beams):
        ax_inx += 1
        ax  = fig.add_subplot(n_rows,n_cols,ax_inx)
        try:
            pydarn.RTP.plot_range_time(fitacf, beam_num=beam, parameter='p_l', zmax=50, zmin=0, date_fmt='%H%M', colorbar_label='Power (dB)', cmap='viridis',ax=ax)
        except:
            pass

        fmt   = '%Y %b %d %H%M UT'
#        ax.set_title('{!s} {!s} - {!s}'.format(radar.upper(),sTime.strftime(fmt),eTime.strftime(fmt)))
#        ax.text(0.01,0.9,'Beam {!s}'.format(beam),fontdict={'weight':'bold','size':'xx-large'},transform=ax.transAxes)

        txt = ('{!s} {!s} {!s} - {!s}'.format(radar.upper(),beam,sTime.strftime(fmt),eTime.strftime(fmt)))
        ax.text(0.01,0.9,txt,fontdict={'weight':'bold','size':'xx-large'},transform=ax.transAxes)

        ax.set_ylabel('Range Gate')
        ax.set_xlim(sTime,eTime)

    ax.set_xlabel('Time [UT]')

    fig.tight_layout()
    sTime_str   = sTime.strftime('%Y%m%d.%H%M')
    eTime_str   = eTime.strftime('%Y%m%d.%H%M')
    fname       = '{!s}-{!s}_{!s}.rti.png'.format(sTime_str,eTime_str,radar)
    fpath       = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
