#!/usr/bin/env python
"""
This script was written in response to comments in two e-mails from Bob Gerzoff sent on 15 Nov 2022.
    1. When I put together all of your CSV files, I wind up with about 10% of the observations being exact duplicates. 
    2. Just in case you see anything amiss.  I thought these [the meanSubIntspect_by_rtiCnt MSTID Index] were normalized?
       I would have expected the mean to be closer to zero and not consistently negative?

This script checks for duplicates and computes some statistics on the MSTID index. It was found that:
    1. All dupes are NaNs. No actual duplicated data found.
       N Rows: 18009 N Dupes: 6932 (38.5% Dupes)
    2. The radar-season MSTID index means are in fact consistently negative.
"""

import os
import glob

import pandas as pd
import numpy as np

data_dir    = os.path.join('output','climo')
files = glob.glob(os.path.join(data_dir,'20*.csv'))

seasons = []
means   = []
stds    = []
nobs    = []

df = []
for fl in files:
    dft = pd.read_csv(fl,comment='#',parse_dates=[0]).set_index('datetime')
    df.append(dft)

    bn      = os.path.basename(fl)
    season  = bn[0:17]
    seasons.append(season)
    means.append(dft.mean())
    stds.append(dft.std())
    nobs.append(dft.count())

df  = pd.concat(df,ignore_index=False)
#df  = df.set_index('datetime')
df  = df.sort_index()

dupes = df[df.duplicated(keep=False)]

if np.all(np.isnan(dupes)):
    print('All dupes are NaNs. No actual duplicated data found.')
else:
    print('Warning: Duplicate rows with data found.')

pct_dupes = (len(dupes)/len(df))*100
print('N Rows: {!s} N Dupes: {!s} ({:0.1f}% Dupes)'.format(len(df),len(dupes),pct_dupes))

fpath   = os.path.join(data_dir,'means.csv')
means   = pd.DataFrame(means,index=seasons)
means.to_csv(fpath)
print('Saved: {!s}'.format(fpath))

fpath   = os.path.join(data_dir,'stds.csv')
stds    = pd.DataFrame(stds,index=seasons)
stds.to_csv(fpath)
print('Saved: {!s}'.format(fpath))

fpath   = os.path.join(data_dir,'n_observations.csv')
nobs    = pd.DataFrame(nobs,index=seasons)
nobs.to_csv(fpath)
print('Saved: {!s}'.format(fpath))

import ipdb; ipdb.set_trace()
