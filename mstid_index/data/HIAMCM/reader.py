#!/usr/bin/env python
"""
This script will load in a HIAMCM Fortran binary output file, create an XArray dataset,
and save it as a netCDF, and make a sample plot.
"""
import datetime
import numpy as np
from array import array
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

fpath = "07DEC2018-16MAR2019.mzgw.grads"

#      it = 0
#      krec  = 0
# 100  it = it + 1
#      print*
#      print*,'try to read time step, it = ',it 
#      do id = 1,idat
#        do lz = 1,lev1  
#          krec = krec+1
#          read(12,rec=krec,err=900)( dat(j,lz,id), j=1,nlat )
#        enddo
#      enddo 

nlat    = 90
lev1    = 261
idat    = 54
ndates  = 100
undef   = -9.99e33

sDate   = datetime.datetime(2018,12,7)
dates   = [sDate]
while len(dates) < ndates:
    new_day = dates[-1] + datetime.timedelta(days=1)
    dates.append(new_day)

ylat = [-89 + 2*x for x in range(nlat)]
altz = """
   0.0   0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0   5.5          
   6.0   6.5   7.0   7.5   8.0   8.5   9.0   9.5  10.0  10.5  11.0  11.5          
  12.0  12.5  13.0  13.5  14.0  14.5  15.0  15.5  16.0  16.5  17.0  17.5          
  18.0  18.5  19.0  19.5  20.0  20.5  21.0  21.5  22.0  22.5  23.0  23.5          
  24.0  24.5  25.0  25.5  26.0  26.5  27.0  27.5  28.0  28.5  29.0  29.5          
  30.0  30.5  31.0  31.5  32.0  32.5  33.0  33.5  34.0  34.5  35.0  35.5          
  36.0  36.5  37.0  37.5  38.0  38.5  39.0  39.5  40.0  40.5  41.0  41.5          
  42.0  42.5  43.0  43.5  44.0  44.5  45.0  45.5  46.0  46.5  47.0  47.5          
  48.0  48.5  49.0  49.5  50.0  50.5  51.0  51.5  52.0  52.5  53.0  53.5          
  54.0  54.5  55.0  55.5  56.0  56.5  57.0  57.5  58.0  58.5  59.0  59.5          
  60.0  60.5  61.0  61.5  62.0  62.5  63.0  63.5  64.0  64.5  65.0  65.5          
  66.0  66.5  67.0  67.5  68.0  68.5  69.0  69.5  70.0  71.0  72.0  73.0          
  74.0  75.0  76.0  77.0  78.0  79.0  80.0  81.0  82.0  83.0  84.0  85.0          
  86.0  87.0  88.0  89.0  90.0  91.0  92.0  93.0  94.0  95.0  96.0  97.0          
  98.0  99.0 100.0 101.0 102.0 103.0 104.0 105.0 106.0 107.0 108.0 109.0          
 110.0 111.0 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0 120.0 122.0          
 124.0 126.0 128.0 130.0 132.0 134.0 136.0 138.0 140.0 142.0 144.0 146.0          
 148.0 150.0 152.0 154.0 156.0 158.0 160.0 162.0 164.0 166.0 168.0 170.0          
 172.0 174.0 176.0 178.0 180.0 182.0 184.0 186.0 188.0 190.0 192.0 194.0          
 196.0 198.0 200.0 205.0 210.0 215.0 220.0 225.0 230.0 235.0 240.0 245.0          
 250.0 255.0 260.0 265.0 270.0 275.0 280.0 285.0 290.0 295.0 300.0 310.0          
 320.0 330.0 340.0 350.0 360.0 370.0 380.0 390.0 400.0"""

altz    = np.array(altz.split(),dtype=np.float32)
dat     = np.zeros((ndates,nlat,lev1,idat),dtype=np.float32)*np.nan

#Define the length of each record. Be sure to include the 2 extra integers for sequential access.
recl    = 4*nlat
fl      = open(fpath,'rb')
for date_inx in range(ndates):
    print(date_inx)
    for idat_inx in range(idat):
        for lz in range(lev1):
            tmp     = fl.read(recl)
            tmp1    = array('f',tmp)

#            # Pull out data array (leaving behind fortran control records)for fortran sequential
#            tmp2    = tmp1[1:-1]

            dat[date_inx,:,lz,idat_inx] = tmp1
#            print(tmp1)

# Set undefined values to NaN
dat[dat<=undef] = np.nan

prms = {}
prms['u']       = {'desc':'u (m/s)'}
prms['v']       = {'desc':'v (m/s)'}
prms['oma']     = {'desc':'oma (Pa/s)'}
prms['w']       = {'desc':'w (m/s)'}
prms['T']       = {'desc':'T / Ts (K)'}
prms['p']       = {'desc':'p / ps (Pa)'}
prms['rho']     = {'desc':'rho / rhos (kg/m**3)'}
prms['cp']      = {'desc':'cp (m**2/s**2/K)'}
prms['psi']     = {'desc':'horiz. streamf. (m**2/s)'}
prms['wir']     = {'desc':'relative vorticity (1/s)'}
prms['div']     = {'desc':'horiz. divergence (1/s)'}
prms['fcor']    = {'desc':'Coriolis parameter (1/s)'}
prms['fcory']   = {'desc':'y-derivative of Cor. param. (1/s/m)'}
prms['ug']      = {'desc':'ug (m/s)'}
prms['vg']      = {'desc':'vg (m/s)'}
prms['uyg']     = {'desc':'dug/dy (1/s)'}
prms['vyg']     = {'desc':'dvg/dy (1/s)'}
prms['goe']     = {'desc':'goe (m**2/s**2)'}
prms['goex']    = {'desc':'goex (m/s**2)'}
prms['goey']    = {'desc':'goey (m/s**2)'}
prms['goeL']    = {'desc':'goeL (1/s**2)'}
prms['dugdp']   = {'desc':'dugdp (m/s/Pa)'}
prms['dvgdp']   = {'desc':'dvgdp (m/s/Pa)'}
prms['dTdp']    = {'desc':'dTdp (m/s/Pa)'}
prms['uu']      = {'desc':'u*u* (m**2/s**2)'}
prms['uv'] 	    = {'desc':'u*v* (m**2/s**2)'}
prms['vv'] 		= {'desc':'v*v* (m**2/s**2)'}
prms['uo'] 		= {'desc':'u*oma* (m*Pa/s**2)'}
prms['vo'] 		= {'desc':'v*oma* (m*Pa/s**2)'}
prms['To'] 		= {'desc':'T*oma* (K*Pa/s**2)'}
prms['go'] 		= {'desc':'goe*oma* (m**2/s**2 * Pa/s)'}
prms['gdiv']    = {'desc':'goe*div* (m**2/s**3)'}
prms['TT'] 		= {'desc':'T*T* (K**2)'}
prms['ww'] 		= {'desc':'w*w* (m**2/s**2)'}
prms['ugox'] 	= {'desc':'u* dgoe*/dx (m**2/s**3)'}
prms['vgoy'] 	= {'desc':'v* dgoe*/dy (m**2/s**3)'}
prms['dragx'] 	= {'desc':'-dp(u*oma)'}
prms['dragy'] 	= {'desc':'-dp(v*oma)'}
prms['uion'] 	= {'desc':'zonal ion drag (m/s/d)'}
prms['vion'] 	= {'desc':'meridional ion drag (m/s/d)'}
prms['epsme'] 	= {'desc':'mech. diss. (K/d)'}
prms['epsth'] 	= {'desc':'therm. diss. (K/da / ...'}
prms['hion'] 	= {'desc':'ion drag frict. heating (K/d) / ...'}
prms['vndy'] 	= {'desc':'vert. dyn visco. (m**2/s)'}
prms['vnmo'] 	= {'desc':'vert. molecular. visco (m**2/s)'}
prms['hndy'] 	= {'desc':'horiz.dyn.visco.(m**2/s)'}
prms['hnmo'] 	= {'desc':'horiz.mol.visco.(m**2/s)'}
prms['QSW'] 	= {'desc':'SW rad.heat.(K/d)'}
prms['QLW'] 	= {'desc':'LW rad.heat.(K/d)'}
prms['Qlat'] 	= {'desc':'latent heat.(K/d)'}
prms['Qsens'] 	= {'desc':'sens. heat.(K/d)'}
prms['NBE'] 	= {'desc':'nonlinear balance equations (1/s**2)'}
prms['MKS'] 	= {'desc':'MKS (m**2/s**3)'}
prms['MPC'] 	= {'desc':'MPC (m**2/s**3)'}

ds = []
for idat_inx,(prm,prmd) in enumerate(prms.items()):
    arr = xr.DataArray(dat[:,:,:,idat_inx],
            coords  = {'dates':dates,'lats':ylat,'alts':altz},
            dims    = ['dates','lats','alts'])
    ds.append(arr.to_dataset(name=prm))

ds  = xr.merge(ds)

# Save to netCDF.
ncPath = '{!s}.nc'.format(fpath)
ds.to_netcdf(ncPath)


fig     = plt.figure(figsize=(10,8))
ax      = fig.add_subplot(1,1,1)
prm     = 'u'
lat     = 61.
lat_inx = np.where(ds['lats'] == lat)[0][0]
xx      = ds[prm]['dates']
yy      = ds[prm]['alts']
zz      = ds[prm][:,lat_inx,:]
mpbl    = ax.contourf(xx,yy,zz.T,cmap='jet')
ax.set_xlabel('Date')
ax.set_ylabel('Altitude [km]')
cbar    = fig.colorbar(mpbl,aspect=15,shrink=0.8)
cbar.set_label(prms[prm].get('desc'))

ax.set_title('HIAMCM {!s}\N{DEGREE SIGN} N'.format(lat))

fig.tight_layout()
png_name    = 'hiamcm_{!s}.png'.format(prm)
fig.savefig(png_name,bbox_inches='tight')
print(png_name)
import ipdb; ipdb.set_trace()
