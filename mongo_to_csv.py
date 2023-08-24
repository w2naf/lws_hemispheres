#!/usr/bin/env python3
"""
This script will export the results from a MongoDB created with the DARNTids libary
to CSV files that can be used for analysis.

Nathaniel A. Frissell
23 August 2023
"""
import os
import shutil
import socket
from pathlib import Path
import tqdm

import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl

import xarray as xr

import pymongo

script_run_time = datetime.datetime.now()

# Set the maximum number of signals detected.
# Set to None for all available signals.
max_sigs = 2

# Set max_lambda to 750 km.
# These are not medium scale, and the MUSIC algorithm
# has a high error value here.
max_lambda = 750.

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

#db_name     = 'mstid_2016'
#db_name     = 'mstid_GSMR_fitexfilter_using_mstid_2016_dates'
db_name     =  'mstid_GSMR_fitexfilter'
prefix      = 'guc'

output_dir  = os.path.join('data','mongo_out',db_name,prefix)
mongo_port  = 27017

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
keys.append('lat')
keys.append('lon')
keys.append('slt')
keys.append('mlt')
keys.append('gscat')
keys.append('height_km')
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
    seasons.append('{!s}_{!s}'.format(spl[-2],spl[-1]))

#with open('reject_codes.txt','w') as fl:
#    fl.write('')

seasons = list(set(seasons))
for season in tqdm.tqdm(seasons,desc='Seasons',dynamic_ncols=True,position=0):
    for radar in tqdm.tqdm(radar_dict.keys(),desc='Radars',dynamic_ncols=True,position=1,leave=False):
        lst = '_'.join([prefix,radar,season])
        if lst not in lists:
            tqdm.tqdm.write('NOT FOUND: {!s}'.format(lst))
            continue

        tqdm.tqdm.write('EXPORTING: {!s}'.format(lst))
        data    = []
        crsr    = db.get_collection(lst).find()

        # Check to see if the corresponding MUSIC list is available.
        music_lst = 'music_'+lst
        if music_lst in lists:
            music_col = db.get_collection(music_lst)
        else:
            tqdm.tqdm.write('--> MUSIC Collection Not Found: {!s}'.format(music_lst))
            music_col = None

        attrs   = {}
        attrs['MongoDB_database']               = db_name
        attrs['MongoDB_collection']             = lst
        attrs['MongoDB_MUSICcollection']        = str(music_lst)
        attrs['season']                         = season
        attrs['Max Number of Signals Reported'] = max_sigs
        attrs['MongoDB_to_CSV_Script']          = str(Path(__file__))
        attrs['System Hostname']                = socket.gethostname()
        attrs['Script Run Time']                = str(script_run_time)

        for item in crsr:
            dct = {} # Temporary dictionary to hold data from each item.
            dct['date'] = item['sDatetime']

            for key in keys:
                dct[key] = item.get(key)

            # Extract MSTID Parameters from MUSIC Algorithm
            # This will only work if the MUSIC algorithm has already been run and there is corresponding
            # collection named 'music_guc_<RAD>_<SDATE>_<EDATE>'
            if music_col is not None:
                # Find item in music collection.
                music_srch = {k: item[k] for k in ['date', 'sDatetime', 'fDatetime', 'radar']}
                music_item = music_col.find_one(music_srch)
                if music_item is not None:

                    # Check to see if music_item has 'signals' key.
                    # Only report MUSIC analysis for MSTID periods.
                    sigs = music_item.get('signals')
                    if sigs is not None and item['category_manu'] == 'mstid':
                        for sig in sigs:
                            sigOrder = sig.get('order')

                            if max_sigs is not None:
                                if sigOrder > max_sigs:
                                    # Only keep the top max_sigs strongest MSTIDs.
                                    continue

                            # Reject any signals with a wavelength > max_lambda km.
                            sig_k = sig.get('k')
                            if sig_k < ( (2*np.pi)/max_lambda):
                                continue

                            for sig_key,sig_val in sig.items():
                                if sig_key in ['order','serialNr']:
                                    continue

                                new_sigKey = 'sig_{:03d}_{!s}'.format(sigOrder,sig_key)
                                if sig_key == 'lambda':
                                    new_sigKey = new_sigKey+'_km'
                                elif sig_key == 'azm':
                                    new_sigKey = new_sigKey+'_deg'
                                elif sig_key == 'freq':
                                    new_sigKey = new_sigKey+'_Hz'
                                elif sig_key == 'period':
                                    sig_val = sig_val/60.
                                    new_sigKey = new_sigKey+'_min'
                                elif sig_key == 'vel':
                                    new_sigKey = new_sigKey+'_mps'

                                dct[new_sigKey] = sig_val

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
            dct['reject_code']  = reject_code
            
            # Store completed record to lists.
            data.append(dct)

#            txt = '{!s}: {!s} {!s} {!s}\n'.format(reject_code, radar, date, reject)
#            print(txt)
#            with open('reject_codes.txt','a') as fl:
#                fl.write(txt)

        df      = pd.DataFrame(data)
        df  = df.set_index('date')

        for attr_key in attr_keys:
            attrs[attr_key] = item.get(attr_key)

        csv_path = os.path.join(output_dir,'sdMSTIDindex_{!s}_{!s}.csv'.format(season,radar))
        with open(csv_path,'w') as fl:
            hdr = []
            hdr.append('# SuperDARN MSTID Index')
            hdr.append('# Generated by Nathaniel Frissell, nathaniel.frissell@scranton.edu')
            hdr.append('#')
            for attr_key,attr in attrs.items():
                hdr.append('# {!s}: {!s}'.format(attr_key,attr))
            hdr.append('#')

            hdr.append('# The "meanSubIntSpect_by_rtiCnt" is the parameter colloquially known as the SuperDARN MSTID Index.')
            hdr.append('#   It stands for the seasonal mean subtracted integrated MSTID power spectral density')
            hdr.append('#   divided by the number of range-beam-time cells that observed ground scatter.')
            hdr.append('#   It is explained in Section 2.2 of Frissell et al. (2016) (https://doi.org/10.1002/2015JA022168)')
            hdr.append('#')

            hdr.append('# Explanation of all parameters in data file:') 
            hdr.append('#  datetime_ut: UTC time at the start of the 2-hour observation window')
            hdr.append('#  lat: Geographic latitude of center point of detection region.')
            hdr.append('#  lon: Geographic longitude of center point of detection region.')
            hdr.append('#  slt: Solar Mean Time at lat/lon point.')
            hdr.append('#  mlt: Magnetic Local Time at lat/lon point.')
            hdr.append('#  gscat: Ground scatter flag set at analysis:')
            hdr.append('#       0: all backscatter data')
            hdr.append('#       1: ground backscatter only')
            hdr.append('#       2: ionospheric backscatter only')
            hdr.append('#       3: all backscatter data with a ground backscatter flag')
            hdr.append('#  height_km: Assumed reflection point height for analysis ')
            hdr.append('#  terminator_fraction: number of radar cells in daylight divided by number of radar cells in darkness.')
            hdr.append('#  good_period: True if period determined to be good for analysis. If False, see Reject Code for explanation.')
            hdr.append('#  orig_rti_cnt: Number of radar cells in window with a backscatter measurement.')
            hdr.append('#  orig_rti_fraction: orig_rti_cnt/orig_rti_possible')
            hdr.append('#  orig_rti_mean:   Mean of all measured values in the data window.')
            hdr.append('#  orig_rti_median: Median of all measured values in the data window.')
            hdr.append('#  orig_rti_possible: Total number of possible radar cells in the data window.')
            hdr.append('#  orig_rti_std: Standard deviation of all measured values in the data window.')
            hdr.append('#  intSpect: Integrated Spectrum')
            hdr.append('#            Integration of power spectral density across all radar cells and spectral bins after MSTID bandpass filter has been applied.')
            hdr.append('#  meanSubIntSpect: Radar-Season Mean Subtracted Integrated Spectrum')
            hdr.append('#                   Integration of (PSD curve of the data window minus the mean PSD curve of all data windows for this radar and season).')
            hdr.append('#  intSpect_by_rtiCnt: Integrated Spectrum normalized by the number of Range-Time-Intensity Backscatter Counts')
            hdr.append('#                      intSpect divided by orig_rti_cnt')
            hdr.append('#  meanSubIntSpect_by_rtiCnt: Radar-Season Mean Subtracted Integrated Spectrum normalized by the number of Range-Time-Intensity Backscatter Counts')
            hdr.append('#                             ** This is the MSTID Index in Frissell et al. (2016) **')
            hdr.append('#                             meanSubIntSpect divided by orig_rti_cnt')
            hdr.append('#  category_manu: Category assigned to data window in database: MSTID, Quiet, or None')
            hdr.append('#                   This is typically automatically assigned by the classification algorithm.')

            hdr.append('# reject_code Explanations:')
            hdr.append('#   0: Good Period (Not Rejected)')
            hdr.append('#   1: High Terminator Fraction (Dawn/Dusk in Observational Window')
            hdr.append('#   2: No Data')
            hdr.append('#   3: Poor Data Quality (including "Low RTI Fraction" and "Failed Quality Check")')
            hdr.append('#   4: Other (including "No RTI Fraction" and "No Terminator Fraction")')
            hdr.append('#')

            hdr.append('# If MUSIC Processing has been run, there will be information about the signals detected in the data window.')
            hdr.append('# Multiple signals can be detected, and each data window may have a different number of signals detected.')
            hdr.append('# Signals are presented in descending order of strength in the MUSIC wavenumber spectrum, each with a signal order ID.')
            hdr.append('#')
            hdr.append('#  kx: North-South Wavenumber [1/(2*pi*km)]')
            hdr.append('#  ky: East-West Wavenumber [1/(2*pi*km)]')
            hdr.append('#  k: Horizontal Wavenumber [1/(2*pi*km)]')
            hdr.append('#  lambda_km: Horizontal Wavelength [km]')
            hdr.append('#  azm_deg: Propagation azimumuth [degrees clockwise from geographic North]')
            hdr.append('#  freq_Hz: Frequency of maxium MSTID-band Power Spectral Density for data window [Hz]')
            hdr.append('#  period_min: Period of strongest MSTID in data window [minutes]')
            hdr.append('#  vel_mps: MSTID Phase Velocity [m/s]')
            hdr.append('#  max: Wavenumber Spectral Density Value')
            hdr.append('#  area: Number of pixels of Karr plot in detected region of this signal.')
            hdr.append('#')

            fl.write('\n'.join(hdr))
            fl.write('\n')
            
            cols = ['datetime_ut'] + list(df.keys())
            fl.write(','.join(cols))
            fl.write('\n')
        df.to_csv(csv_path,mode='a',header=False)
        tqdm.tqdm.write('--> WROTE: {!s}'.format(csv_path))

        dsr = df.to_xarray()
        dsr = dsr.assign_coords({'radar':radar})
        dsr.attrs = attrs
        nc_path = os.path.join(output_dir,'sdMSTIDindex_{!s}_{!s}.nc'.format(season,radar))
        dsr.to_netcdf(nc_path)
        tqdm.tqdm.write('--> WROTE: {!s}'.format(nc_path))
