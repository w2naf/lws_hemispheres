#!/usr/bin/env python3
"""
This code is used to see what reject messages / data quality messages are stored in the MongoDB.
"""
import os
import shutil
import tqdm

import numpy as np
import pandas as pd
import matplotlib as mpl

import xarray as xr

import pymongo

def generate_radar_dict():
    rad_list = []
    rad_list.append(('bks', 39.6, -81.1))
    rad_list.append(('wal', 41.8, -72.2))
    rad_list.append(('fhe', 42.5, -95.0))
    rad_list.append(('fhw', 43.3, -102.7))
    rad_list.append(('cve', 46.4, -114.6))
    rad_list.append(('cvw', 47.9, -123.4))
    rad_list.append(('gbr', 58.4, -59.9))
    rad_list.append(('kap', 55.5, -85.0))
    rad_list.append(('sas', 56.1, -103.8))
    rad_list.append(('pgr', 58.0, -123.5))

#    rad_list.append(('sto', 63.86, -21.031))
#    rad_list.append(('pyk', 63.77, -20.54))
#    rad_list.append(('han', 62.32,  26.61))

    radar_dict = {}
    for radar,lat,lon in rad_list:
        tmp                 = {}
        tmp['lat']          = lat
        tmp['lon']          = lon
        radar_dict[radar]   = tmp

    return radar_dict

output_dir  = 'mstid_index'
mongo_port  = 27017
db_name     = 'mstid'
prefix      = 'guc'

radar_dict  = generate_radar_dict()

# Prepare output dictionary.
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

mongo   = pymongo.MongoClient(port=mongo_port)
db      = mongo[db_name]
lists   = db.list_collection_names()


keys = []
#keys.append('_id')
#keys.append('date')
#keys.append('sDatetime')
#keys.append('fDatetime')
#keys.append('radar')
#keys.append('lat')
#keys.append('lon')
keys.append('slt')
keys.append('mlt')
#keys.append('gscat')
#keys.append('height_km')
keys.append('terminator_fraction')
keys.append('good_period')
keys.append('orig_rti_cnt')
keys.append('orig_rti_fraction')
keys.append('orig_rti_mean')
keys.append('orig_rti_median')
keys.append('orig_rti_possible')
keys.append('orig_rti_std')
keys.append('intSpect')
keys.append('meanSubIntSpect')
keys.append('intSpect_by_rtiCnt')
keys.append('meanSubIntSpect_by_rtiCnt')
keys.append('category_manu')

attr_keys = []
attr_keys.append('radar')
attr_keys.append('lat')
attr_keys.append('lon')
#attr_keys.append('gscat')
#attr_keys.append('height_km')

seasons = []
for lst in lists:
    if not lst.startswith(prefix):
        continue

    spl = lst.split('_')
    seasons.append('{!s}_{!s}'.format(spl[2],spl[3]))

rejects = []

sole_terminator = []
seasons = list(set(seasons))
for season in tqdm.tqdm(seasons,desc='Seasons',dynamic_ncols=True,position=0):
    for radar in tqdm.tqdm(radar_dict.keys(),desc='Radars',dynamic_ncols=True,position=1,leave=False):
        lst = '_'.join([prefix,radar,season])

        if lst not in lists:
            continue

        dates           = []
        data            = {x:[] for x in keys}

        crsr = db.get_collection(lst).find()
        for item in crsr:
            date        = item['sDatetime']
            dates.append(date)

            reject  = item.get('reject_message',[])
            rejects += reject

            if len(reject) == 1 and 'No Terminator Fraction' in reject:
#                print('Found sole No Terminator Fraction reject.')
                sole_terminator.append( (season,radar,date) )

rejects = list(set(rejects))
print(rejects)
# Reject Messages:
# ['No RTI Fraction', 'No Data', 'High Terminator Fraction', 'Failed Quality Check', 'No Terminator Fraction', 'Low RTI Fraction']
# Reject message is None if good period.
import ipdb; ipdb.set_trace()
