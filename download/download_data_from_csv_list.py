#!/usr/bin/env python
import os
import glob
import datetime

import pydarnio
import pydarn

key_path    = 'keys/w3usr'
server      = 'w2naf@sd-work9.ece.vt.edu'

rads    = {}
rads['north']   = []
rads['south']   = []

for rid,radar in pydarn.SuperDARNRadars.radars.items():
    abr = radar.hardware_info.abbrev
    if radar.hemisphere is pydarn.Hemisphere.North:
        rads['north'].append(abr)
    elif radar.hemisphere is pydarn.Hemisphere.South:
        rads['south'].append(abr)


fname   = 'study_periods.csv'
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

for sDate, eDate in date_list:
    if sDate.year < 2009:
        continue

    # Build up a list of all the days we want to get the data.
    dates   = [sDate]
    while dates[-1] < eDate:
        nextDate    = dates[-1] + datetime.timedelta(days=1)
        dates.append(nextDate)
    for date in dates:
        print('')
        print(date)
        year_str    = '{!s}'.format(date.year)
        month_str   = '{:02d}'.format(date.month)
        day_str     = '{:02d}'.format(date.day)
        date_str    = ''.join([year_str,month_str,day_str])

        for hemi in ['north','south']:
            for radar in rads[hemi]:
#                print(sDate,hemi,radar)

                data_dir    =  'sd-data/{!s}/fitacf/{!s}'.format(year_str,radar)
                if not os.path.exists(data_dir):
                   os.makedirs(data_dir) 

                path        = '/sd-data/{!s}/fitacf/{!s}'.format(year_str,radar)
                pattern     = '{!s}*{!s}*.fitacf.bz2'.format(date_str,radar)

                cmd         = 'rsync -Pav -e "ssh -i {!s}" --include="{!s}" --exclude="*" {!s}:{!s}/ {!s}/'.format(key_path,pattern,server,path,data_dir)
                print('   ',cmd)
                os.system(cmd)

            fl_count = glob.glob(os.path.join(data_dir,'*'))
            if len(fl_count) == 0:
                os.rmdir(data_dir)
