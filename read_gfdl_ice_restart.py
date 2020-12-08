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



with Dataset(icein) as src, Dataset(iceout, "w") as dst:
    # copy global attributes all at once via dictionary
    dst.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))
    # copy all file data except for the excluded
    for name, variable in src.variables.items():
        x = dst.createVariable(name, variable.datatype, variable.dimensions)
        dst[name][:] = src[name][:]
        # copy variable attributes all at once via dictionary
        dst[name].setncatts(src[name].__dict__)
    aicenout = dst['FR']
    vicenout = dst['VOLICE']
    vsnonout = dst['VOLSNO']
    tskinout = dst['TSKINI']
    eicenout = dst['ERGICE']
    esnonout = dst['ERGSNO']
    aicen = dst['FR'][:]
    vicen = dst['VOLICE'][:]
    vsnon = dst['VOLSNO'][:]
    tskin = dst['TSKINI'][:]
    eicen = dst['ERGICE'][:]
    esnon = dst['ERGSNO'][:]
