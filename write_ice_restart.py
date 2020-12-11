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

def create_tile_rst(fout, tilesize):

   icein = 'seaicethermo_internal_rst'

   with Dataset(icein) as src, Dataset(iceout, "w") as dst:
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
    dst['HSKINI'][:] = 0.5*1000.0 
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
 
               

iceout = 'seaicethermo_internal_rst.copy'
fout = 'seaicethermo_internal_rst.empty'

create_tile_rst(iceout, 24034000)

