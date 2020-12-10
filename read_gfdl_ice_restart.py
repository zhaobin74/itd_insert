#! /usr/bin/env python

from netCDF4 import Dataset
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import array
import matplotlib.cm as mcm
from mpl_toolkits.basemap import Basemap
from collections import defaultdict
import cmocean as cm
from scipy.spatial import cKDTree
import glob
import collections
import struct
import datetime
import time
import sys
sys.path.append('/home/bzhao/python_utils')
#import read_utils
#import plot_utils
#import math_utils
#import read_utils
import data_utils
#import get_info
#from pylab import *
import functools


gridfile = '/discover/nobackup/yvikhlia/coupled/Forcings/a90x540_o360x200/INPUT/grid_spec.nc'

with Dataset(gridfile) as grid:
    x_t =  grid['geolon_t'][:]
    y_t =  grid['geolat_t'][:]

icein = 'ice_model.res.nc'

with Dataset(icein) as src:
    aicen = src['part_size'][0]
    vicen = src['h_ice'][0]
    vsnon = src['h_snow'][0]
    tskin = src['t_surf'][0]

tskini = tskin[1:,:,:]

print 'ts: ', tskin.shape
print 'aice: ', aicen.shape
print 'vice: ', vicen.shape
print 'vsno: ', vsnon.shape

aice = np.sum(aicen[1:,:,:], axis=0)
area = np.sum(aicen, axis=0)
print area.min(), aice.min()
print area.max(), aice.max()

print 'ts  min,max', tskin.min(), tskin.max()
print 'tsi min,max', tskini.min(), tskini.max()



cmp = mcm.get_cmap('jet')
meridians=[1,0,1,1]

fig=plt.figure(figsize=(10,10), facecolor='w')
#fig.subplots_adjust(left=0.05, right=1.0, top=0.99, bottom=0.01,wspace=0.05,hspace=0.05)
ax=fig.add_axes([0.06, 0.0, 0.98, 0.98])

m = Basemap(projection='npstere',lon_0=0,boundinglat=45, resolution='l')
m.drawcoastlines()
m.fillcontinents()
m.drawcountries()

x, y =m(x_t,y_t)
#outside = (x <= m.xmin) | (x >= m.xmax) | (y <= m.ymin) | (y >= m.ymax)
#fbot = ma.masked_where(outside, fbot)
#m.pcolormesh(x,y,hice,cmap=cmp,vmin=0.0, vmax=4.0)
levl = 0 #m.pcolormesh(x,y,hice,cmap=cmp,vmin=-2, vmax=2)
m.pcolormesh(x,y,aice,cmap=cmp,vmin=0.0, vmax=1.0)
#m.pcolormesh(x,y,hice[levl],cmap=cmp,vmin=0.0, vmax=4.0)

#m.pcolormesh(x,y,hice,cmap=cmp,vmin=250, vmax=280)
#if POLE == 'N':
 #  m.plot(0.0,90.0,'ko',markersize=15, latlon=True)
m.drawparallels(np.arange(-90.,120.,15.),labels=[1,0,0,0]) # draw parallels
m.drawmeridians(np.arange(0.,420.,30.),labels=meridians) # draw meridians
plt.colorbar(orientation='vertical',extend='both',shrink=0.8)
#plt.tight_layout()
plt.show()

