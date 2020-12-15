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
#import get_info
#from pylab import *
import functools


def read_mit(infile):
    # read in the MITgcm sea ice restart file (binary or netcdf)
    # return aice,hice/volice in dimension (ny, nx)
    # there is no category dimension since mitgcm has only 1 category 
    '''
    : type infile: str  
    : rtype: (NDArray, NDArray, NDArray, NDArray)
    ''' 
        
    return (aicen, vicen, vsnon, tskin)

def remap_mit(aicen_source, vicen_source, inds, indis, indjs, aicen_target, vicen_target):
    '''
    repartition ice fraction and volume to cice categories
    '''
    pass


if __name__ == "__main__":

    pass
