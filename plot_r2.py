#!/usr/bin/env python
import os
import shutil
import datetime
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


import matplotlib as mpl
from matplotlib import pyplot as plt

from cycler import cycler

#default_cycler  =  cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
#custom_cycler   = (default_cycler +  cycler(linestyle=['-', '--', ':', '-.']))
#ls_cycler       = cycler(linestyle=['-', '--', ':', '-.'])

lss = ['-', '--', ':', '-.']

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0
mpl.rcParams['axes.titleweight']    = 'bold'
mpl.rcParams['axes.labelweight']    = 'bold'

def plot_r2_csv(csv_path):
    png_path = csv_path.replace('.csv','.png')
    bname    = os.path.basename(csv_path)
    df = pd.read_csv(csv_path,index_col=[0])

    fig     = plt.figure(figsize=(10,8))
    ax      = fig.add_subplot(111)
#    ax.set_prop_cycle(custom_cycler)

    values  = df.values

    xx = np.array(df.keys(),dtype=int)

    for rinx,season in enumerate(df.index):
        yy = values[rinx,:]

        if season == 'all':
            lw = 5
            ls = '-'
        else:
            lw = 1
            ls = lss[rinx%len(lss)]

        ax.plot(xx,yy,lw=lw,ls=ls,label=season)

        ax.legend(loc='lower right',fontsize='small')

        ax.set_xlabel('Polar Vortex Delay (Days)')
        ax.set_ylabel(bname)
        ax.set_title(csv_path)

        fig.tight_layout()
        fig.savefig(png_path,bbox_inches='tight')

if __name__ == '__main__':
    csv_paths = []
    csv_paths.append('output/correlate_mstid_pv/df_r2.csv')
    csv_paths.append('output/correlate_mstid_pv/df_m_slope.csv')
    for csv_path in csv_paths:
        plot_r2_csv(csv_path)

    import ipdb; ipdb.set_trace()
