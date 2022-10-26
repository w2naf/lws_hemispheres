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


def load_fitacf(sTime,eTime,radar,data_dir='sd-data',fit_sfx='fitacf'):
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
        year            = date.year
        date_str        = date.strftime('%Y%m%d')
        ddir            = os.path.join(data_dir,'{!s}'.format(year),fit_sfx,radar)
        fpattern        = os.path.join(ddir,'{!s}*{!s}*.{!s}.bz2'.format(date_str,radar,fit_sfx))
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


def load_dates(fname='study_periods.csv'):
    with open(fname,'r') as fl:
        lines   = fl.read()

    lines = lines.split('\n')
    lines.sort()

    date_list = []
    for line in lines:
        if 'sDate' in line: continue
        if line == '': continue

        sd, ed   = line.split(',')
        if ed == '':
            ed = sd

        sDate = datetime.datetime.strptime(sd,'%Y-%m-%d')
        eDate = datetime.datetime.strptime(ed,'%Y-%m-%d')

        date_list.append( (sDate,eDate) )
    return date_list

def date_range(date_0,date_1):
    """
    Return all of the dates in a range of dates.
    """
    dates   = [date_0]
    while dates[-1] < date_1:
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


def make_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if __name__ == '__main__':
    hemis   = []
#    hemis.append('north')
    hemis.append('south')

    sDate       = datetime.datetime(2017,1,1)
    eDate       = datetime.datetime(2018,1,1)
    date_list   = [(sDate,eDate)]

    base_dir  = 'plots'
    make_dirs(base_dir)

#    date_list   = load_dates() 
    radar_list  = gen_radar_list()

    for date_0, date_1 in date_list:
        event_str   = '{!s}-{!s}'.format(date_0.strftime('%Y%m%d'),date_1.strftime('%Y%m%d'))
        plot_dates = date_range(date_0,date_1)
        for hemi in hemis:
            output_dir  = os.path.join(base_dir,event_str,hemi)
            make_dirs(output_dir)
            for radar in radar_list[hemi]:
                print('--> ',event_str,hemi,radar)
                for plot_date in plot_dates:

                    sTime       = plot_date
                    eTime       = plot_date + datetime.timedelta(days=1)
                    fitacf      = load_fitacf(sTime,eTime,radar)

                    if len(fitacf) == 0:
                        continue

                    sTime_str   = sTime.strftime('%Y%m%d.%H%M')
                    eTime_str   = eTime.strftime('%Y%m%d.%H%M')
                    fname       = '{!s}-{!s}_{!s}.summary.png'.format(sTime_str,eTime_str,radar)
                    fpath       = os.path.join(output_dir,fname)

                    try:
                        result      = pydarn.RTP.plot_summary(fitacf,9,watermark=False)
                        fig         = result[0]
                    except:
                        fig         = plt.figure()
                        ax          = fig.add_subplot(111)
                        ax.text(0.5,0.5,'No Data / Plotting Error',ha='center',transform=ax.transAxes)
                        ax.set_title(fname)

                    fig.savefig(fpath,bbox_inches='tight')
                    plt.close(fig)
