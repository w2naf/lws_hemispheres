#!/usr/bin/env python
"""
This script will compare two different MSTID Index Mongo databases.
The prupose of this script is to try to understand the differences between the
processing that was done in 2016 with the processing we are able to do in 2023
after the code was ported from DaViTPy/Python 2 to pyDARN/Python 3.

Nathaniel Frissell
April 22, 2023
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


def compare_mongo_dbs(dbs,seasons=None,output_dir='output'):
    """
    Compare MSTID Mongo Databases.
    Returns a dataframe and generates a CSV with results.
    """
    csv_path = os.path.join(output_dir,'{!s}.csv'.format('-'.join(dbs)))
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path,comment='#',index_col=0)
        print('Using cached CSV: {!s}'.format(csv_path))
        return df

    prefix      = 'guc'

    radar_dict  = generate_radar_dict()

    mongo_port  = 27017
    mongo       = pymongo.MongoClient(port=mongo_port)

    index_keys  = []
    index_keys.append('sDatetime')
    index_keys.append('fDatetime')
    index_keys.append('radar')

    col_keys    = []
    col_keys.append('category_manu')
    col_keys.append('meanSubIntSpect_by_rtiCnt')

    df  = pd.DataFrame([]*len(index_keys),columns=index_keys)

    for db_name in dbs:
        db      = mongo[db_name]
        lists   = db.list_collection_names()

        if seasons is None:
            # Determine available seasons from database if seasons is not already specified.
            seasons = []
            for lst in lists:
                if not lst.startswith(prefix):
                    continue

                spl = lst.split('_')
                seasons.append('{!s}_{!s}'.format(spl[2],spl[3]))

            seasons = list(set(seasons))

        for season in tqdm.tqdm(seasons,desc='Seasons',dynamic_ncols=True,position=0):
            for radar in tqdm.tqdm(radar_dict.keys(),desc='Radars',dynamic_ncols=True,position=1,leave=False):
                lst = '_'.join([prefix,radar,season])

                if lst not in lists:
                    continue

                crsr = db.get_collection(lst).find()
                for item in crsr:

                    # Determine reject message.
                    # Possible Reject Messages:
                    # ['No RTI Fraction', 'No Data', 'High Terminator Fraction', 'Failed Quality Check', 'No Terminator Fraction', 'Low RTI Fraction']
                    reject = item.get('reject_message')
                    if reject is None:
                        reject_code = 0 # Good period
                    elif 'High Terminator Fraction' in reject:
                        reject_code = 1 # Dawn/Dusk
                    elif 'No Data' in reject:
                        reject_code = 2 # No Data
                    elif ('Failed Quality Check' in reject) or ('Low RTI Fraction' in reject):
                        reject_code = 3 # Poor data quality
                    else:
                        reject_code = 4 # Other, including No RTI Fraction and No Terminator Fraction.

                    # Search dataframe for row that matches all index_key criteria.
                    # (sDatetime, fDatetime, radar)
                    tfs = []
                    for inx_key in index_keys:
                        tf = df[inx_key] == item.get(inx_key)
                        tfs.append(tf)

                    tfs = np.logical_and.reduce(tfs)
                    row = df.loc[tfs]

                    if len(row) == 0:
                        # If no matching entry in the df, create a new row.
                        row = {}
                        for inx_key in index_keys:
                            row[inx_key] = item.get(inx_key)

                        for col_key in col_keys:
                            row[db_name+':'+col_key] = item.get(col_key)

                        row[db_name+':reject_code'] = reject_code
                        row[db_name+':collection']  = lst
                        row = pd.DataFrame([row])
                        df  = pd.concat([df,row],ignore_index=True)
                    else:
                        # If matching entry is found in df, then add columns with the new database's info.
                        for col_key in col_keys:
                            db_ckey = db_name+':'+col_key
                            df.loc[row.index,db_ckey] = item.get(col_key)

                        db_ckey = db_name+':reject_code'
                        df.loc[row.index,db_ckey] = reject_code

                        db_ckey = db_name+':collection'
                        df.loc[row.index,db_ckey] = lst

    # Convert reject codes to integers.
    for db_name in dbs:
        db_ckey     = db_name+':reject_code'
        df[db_ckey] = df[db_ckey].astype(int)
        
    # Output to CSV
    with open(csv_path,'w') as fl:
        hdr = []
        hdr.append('# Comparison of SuperDARN MSTID Mongo Databases')
        hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
        hdr.append('#')

        hdr.append('# reject_code Explanations:')
        hdr.append('#   0: Good Period (Not Rejected)')
        hdr.append('#   1: High Terminator Fraction (Dawn/Dusk in Observational Window')
        hdr.append('#   2: No Data')
        hdr.append('#   3: Poor Data Quality (including "Low RTI Fraction" and "Failed Quality Check")')
        hdr.append('#   4: Other (including "No RTI Fraction" and "No Terminator Fraction")')
        hdr.append('#')

        fl.write('\n'.join(hdr))
        fl.write('\n')
        
    #    cols = ['datetime_ut'] + list(df.keys())
    #    fl.write(','.join(cols))
    #    fl.write('\n')
    df.to_csv(csv_path,mode='a',header=True)

    return df

if __name__ == '__main__':
    dbs = []
    dbs.append('mstid_2016')
    dbs.append('fitexfilter')

    seasons = []
    seasons.append('20121101_20130501')

    recalc  = False
    output_dir  = os.path.join('output','db_compare','-'.join(dbs))
    # Prepare output dictionary.
    if os.path.exists(output_dir) and recalc:
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = compare_mongo_dbs(dbs,seasons=seasons,output_dir=output_dir)

    import ipdb; ipdb.set_trace()
