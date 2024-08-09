#!/usr/bin/env python
"""
This class will plot a day of amateur radio RBN/PSKReporter/WSPRNet data time series with edge
detection and spot maps.
"""
import os
import datetime
import pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

class HamSpotPlot(object):
    def __init__(self,date=datetime.datetime(2018,12,15),data_dir=None):
        if data_dir is None:
            data_dir    = os.path.join('data','lstid_ham')
        self.data_dir   = data_dir
        self.date       = date
        self.load_data()
    
    def load_data(self):
        date    = self.date

        # EDGE DETECT DATA #####################
        # 20181215_edgeDetect.pkl
        date_str        = date.strftime('%Y%m%d')
        fname           = f'{date_str}_edgeDetect.pkl'
        fpath           = os.path.join(self.data_dir,fname)
        self.edge_fpath = fpath
        with open(fpath,'rb') as pkl:
            import ipdb; ipdb.set_trace()
            edge_data   = pickle.load(pkl)

        self.edge_data  = edge_data
        # hamSpot_geo_2018_12_15.csv.bz2

        import ipdb; ipdb.set_trace()

    def plot_figure(self,png_fpath='output.png',figsize=(16,5),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax):
        fig      = ax.get_figure()
        
        result  = {}
        return result

if __name__ == '__main__':
    output_dir = os.path.join('output','hamSpotPlot')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hsp = HamSpotPlot()

    png_fpath   = os.path.join(output_dir,png_fname)
    hsp.plot_figure(png_fpath=png_fpath)
    print(png_fpath)
