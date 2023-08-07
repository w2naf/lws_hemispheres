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
from matplotlib import pyplot as plt

import xarray as xr

import pymongo

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

index_keys  = []
index_keys.append('sDatetime')
index_keys.append('fDatetime')
index_keys.append('radar')

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

def find_record(df,search_dict):
    """
    Search dataframe for row that matches all index_key criteria.
    """

    tfs = []
    for inx_key, val in search_dict.items():
        tf = df[inx_key] == val
        tfs.append(tf)

    tfs = np.logical_and.reduce(tfs)
    row = df.loc[tfs]
    
    return row

def compare_mongo_dbs(dbs,seasons=None,output_dir='output'):
    """
    Compare MSTID Mongo Databases.
    Returns a dataframe and generates a CSV with results.
    """
    csv_path = os.path.join(output_dir,'{!s}.csv'.format('-'.join(dbs)))
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path,comment='#',index_col=0,parse_dates=[1,2])
        print('Using cached CSV: {!s}'.format(csv_path))
        return df

    prefix      = 'guc'

    radar_dict  = generate_radar_dict()

    mongo_port  = 27017
    mongo       = pymongo.MongoClient(port=mongo_port)

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
                    search_dict = {key:item.get(key) for key in index_keys}
                    row = find_record(df,search_dict)

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

def match_mstid_db(db_0_name,db_1_name,db_new_name,df,mongo_port=27017):
    """
    Generate a new database db_new using the data from db_1 that matches the good dates
    of db_0.

    db_0: MSTID database with desired good dates.
    db_1: MSTID database with desired data.
    db_new: Name of new MSTID database.
    df: Dataframe generated with compare_mongo_dbs()
    """

    mongo       = pymongo.MongoClient(port=mongo_port)
    db_1        = mongo[db_1_name]

    # Clear out new database.
    mongo.drop_database(db_new_name)

    db_new      = mongo[db_new_name]
    
    col_name    = db_1_name+':collection'
    collections = list(df[col_name].unique())

    for collection in tqdm.tqdm(collections,desc='Creating matched db {!s}'.format(db_new_name),dynamic_ncols=True,position=0):
        crsr    = db_1[collection].find()
        count   = db_1[collection].count_documents({})
        for item in tqdm.tqdm(crsr,desc=collection,dynamic_ncols=True,position=1,leave=False,total=count):
            search_dict = {key:item.get(key) for key in index_keys}
            row         = find_record(df,search_dict)

            if int(row[db_0_name+':reject_code']) != 0:
                item['intpsd_sum']                  = 'NaN'
                item['intpsd_max']                  = 'NaN'
                item['intpsd_mean']                 = 'NaN'


                del_keys = []
                del_keys.append('intSpect')
                del_keys.append('meanSubIntSpect')
                del_keys.append('intSpect_by_rtiCnt')
                del_keys.append('meanSubIntSpect_by_rtiCnt')
                for del_key in del_keys:
                    if del_key in item:
                        del item[del_key]

                item['good_period']                 =  False
                item['category_manu']               = 'None'
                item['reject_message']              = ['Not in 2016 Processing']

            db_new[collection].insert_one(item)

def db_compare_scatter(df,db_0,db_1,param,output_dir='output'):
    """
    Make a scatter plot comparing the two databases.
    df:         Dataframe generated with compare_mongo_dbs()
    db_0:       db on the x-axis
    db_1:       db on the y-axis
    param:      Parameter to be plotted.
    output_dir: Output Directory
    """

    png_fname   = '{!s}-{!s}.{!s}.scatter.png'.format(db_0,db_1,param)
    png_fpath   = os.path.join(output_dir,png_fname)

    cols        = [db+':'+param for db in [db_0,db_1]]
    dfd         = df.dropna(subset=cols)

    radars      = list(dfd['radar'].unique())
    radars.sort()

    fig = plt.figure(figsize=(10,10))
    ax  = fig.add_subplot(1,1,1,aspect='equal')

    for radar in radars:
        dft = dfd.loc[dfd['radar'] == radar]

        xx  = dft[cols[0]]
        yy  = dft[cols[1]]

        ax.scatter(xx,yy,label=radar,marker='.')

    xx_avg  = dfd[cols[0]].mean()
    lbl = []
    lbl.append('{!s} - DB:{!s}'.format(param,db_0))
    lbl.append('Mean: {!s}'.format(xx_avg))
    ax.set_xlabel('\n'.join(lbl))

    yy_avg  = dfd[cols[1]].mean()
    lbl = []
    lbl.append('{!s} - DB:{!s}'.format(param,db_1))
    lbl.append('Mean: {!s}'.format(yy_avg))
    ax.set_ylabel('\n'.join(lbl))

    ax.legend(loc='lower right',fontsize='small')

    lim = (-0.075,0.075)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ds_0 = df['sDatetime'].min().strftime('%Y %b %d')
    ds_1 = df['sDatetime'].max().strftime('%Y %b %d')
    title = []
    title.append('MSTID Data Processing Comparison')
    title.append('{!s} - {!s}'.format(ds_0,ds_1))
    ax.set_title('\n'.join(title))

    fig.savefig(png_fpath,bbox_inches='tight')

def reject_code_diff(df,db_0,db_1,output_dir='output'):
    """
    df:         Dataframe generated with compare_mongo_dbs()
    db_0:       db on the x-axis
    db_1:       db on the y-axis
    output_dir: Output Directory
    """

    reject_codes    = list(range(5))

    cols    = [db+':reject_code' for db in [db_0,db_1]]

    df_list = []
    for rej_from in reject_codes:
        dft_0 = df[df[cols[0]] == rej_from]
        for rej_to in reject_codes:
            dft_1 = dft_0[dft_0[cols[1]] == rej_to]

            count = len(dft_1)

            dct = {}
            dct['reject_from_{!s}'.format(db_0)]  = rej_from
            dct['reject_to_{!s}'.format(db_1)]    = rej_to
            dct['count']        = count
            df_list.append(dct)

    df_rej  = pd.DataFrame(df_list)

    ds_0        = df['sDatetime'].min().strftime('%Y%m%d')
    ds_1        = df['sDatetime'].max().strftime('%Y%m%d')
    fname       = '{!s}-{!s}_reject_mapping.csv'.format(ds_0,ds_1)
    csv_path    = os.path.join(output_dir,fname)

    # Output to CSV
    with open(csv_path,'w') as fl:
        hdr = []
        hdr.append('# Reject Code Mapping')
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

    df_rej.to_csv(csv_path,mode='a',header=True,index=False)

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

    db_0    = dbs[0]
    db_1    = dbs[1]

    reject_code_diff(df,db_0,db_1,output_dir=output_dir)

    param   = 'meanSubIntSpect_by_rtiCnt'
    db_compare_scatter(df,db_0,db_1,param,output_dir=output_dir)

    db_new  = '{!s}_using_{!s}_dates'.format(db_1,db_0)
    match_mstid_db(db_0,db_1,db_new,df)

    import ipdb; ipdb.set_trace()
