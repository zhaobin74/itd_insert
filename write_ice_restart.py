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


MAPL_RHOWTR = 1000.0

from cice import (ncat, nilyr, nslyr)

def create_tile_rst(fin, fout, tilesize, aice_in, vice_in, 
                 vsno_in, tskin_in, eice_in, esno_in, sst=None, sss=None):


   with Dataset(fin) as src, Dataset(fout, "w") as dst:
    # copy global attributes all at once via dictionary
    #dst.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        #print name, "-->", len(dimension), dimension
        if name == 'tile':
            dst.createDimension(name, tilesize)
        else:
            dst.createDimension(
               name, (len(dimension) if not dimension.isunlimited() else None))
    # copy all file data except for the excluded
    for name, variable in src.variables.items():
        #print name, variable.datatype, variable.dimensions
        #print src[name].__dict__
        x = dst.createVariable(name, variable.datatype, variable.dimensions)
        #dst[name][:] = src[name][:]
        # copy variable attributes all at once via dictionary
        dst[name].setncatts(src[name].__dict__)

    dst['time'][:] = 0  
    dst['HSKINI'][:] = 0.5*MAPL_RHOWTR 
    dst['SSKINI'][:] = 30.0
    dst['QS'][:]     = 0.01
    dst['CH'][:]     = 1.0e-4
    dst['CM'][:]     = 1.0e-4
    dst['CQ'][:]     = 1.0e-4
    dst['Z0'][:]     = 0.00005
    dst['WW'][:]     = 0.0
    dst['SLMASK'][:] = 0.0
    dst['APONDN'][:] = 0.0
    dst['HPONDN'][:] = 0.0
    dst['TAUAGE'][:] = 0.0
    dst['VOLPOND'][:] = 0.0
    if 'TWMTS' in src.variables:
        dst['TWMTS'][:] = 0.0
    if 'HSKINW' in src.variables:
        dst['HSKINW'][:] =  2.0*MAPL_RHOWTR
    dims_in = aice_in.shape
    fr = dst['FR'][:]  
    dims_ou = fr.shape
    if dims_in[0] == dims_ou[0]:   
        dst['FR'][:] = aice_in
    else:
        aiceout = np.zeros(fr.shape)
        aiceout[1:,:] = aice_in
        aiceout[0,:]  = np.maximum(0.0, np.minimum(1.0, 1.0-np.sum(aice_in,axis=0))) 
        dst['FR'][:] = aiceout
    dst['VOLICE'][:] = vice_in
    dst['VOLSNO'][:] = vsno_in
    dst['TSKINI'][:] = tskin_in
    dst['ERGICE'][:] = eice_in
    dst['ERGSNO'][:] = esno_in
    if 'TSKINW' in src.variables:
        if sst is None:
            print 'missing SST data; required to populate TW field'
            sys.exit(1) 
        else:
            dst['TSKINW'][:] = sst
    if 'SSKINW' in src.variables:
        if sss is None:
            print 'missing SSS data; required to populate SW field'
            sys.exit(1) 
        else:
            dst['SSKINW'][:] = sss
    if 'TSKINW' in src.variables:
       print 'sst min, max :', sst.min(), sst.max()
    if 'SSKINW' in src.variables:
       print 'sss min, max :', sss.min(), sss.max()
    print 'tskin min, max :', tskin_in.min(), tskin_in.max()
    print 'aicen min, max :', aiceout.min(), aiceout.max()
    aice = np.sum(aiceout, axis=0)
    print 'sum(FR) min, max :', aice.min(), aice.max()
    print 'vice min, max :', vice_in.min(), vice_in.max()
    print 'vsno min, max :', vsno_in.min(), vsno_in.max()
 
               
if __name__ == "__main__":

   iceout = 'seaicethermo_internal_rst.copy'
   saltout = 'saltwater_internal_rst.copy'
   fout = 'seaicethermo_internal_rst.empty'
   icein = 'seaicethermo_internal_rst'
   saltin = 'saltwater_internal_rst'

   create_tile_rst(icein, iceout, 24034000)
   create_tile_rst(saltin, saltout, 24034000)

