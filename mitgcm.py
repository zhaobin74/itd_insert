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



from cice import *

def read_mit(infile):
    # read in the MITgcm sea ice restart file (binary or netcdf)
    # return aice,hice/volice in dimension (ny, nx)
    # there is no category dimension since mitgcm has only 1 category 
    '''
    : type infile: str  
    : rtype: (NDArray, NDArray, NDArray, NDArray)
    ''' 
    nx = 5400
    ny = 15
    ro = 'A'
    with open(infile,'rb') as f:
        aicen = np.fromfile(f, dtype=np.float32, count = nx*ny)
        aicen = np.reshape(aicen,(ny, nx), order=ro)   
        vicen = np.fromfile(f, dtype=np.float32, count = nx*ny)
        vicen = np.reshape(vicen,(ny, nx), order=ro)   
        vsnon = np.fromfile(f, dtype=np.float32, count = nx*ny)
        vsnon = np.reshape(vsnon,(ny, nx), order=ro)   
        tskin = np.fromfile(f, dtype=np.float32, count = nx*ny)
        tskin = np.reshape(tskin,(ny, nx), order=ro)   
        sst   = np.fromfile(f, dtype=np.float32, count = nx*ny)
        sst   = np.reshape(sst,(ny, nx),  order=ro)   
        sss   = np.fromfile(f, dtype=np.float32, count = nx*ny)
        sss   = np.reshape(sss,(ny, nx),  order=ro)   
#        print np.reshape(data,(2,3))

    return (aicen, vicen, vsnon, tskin, sst, sss)

def remap_mit(aicen_src, vicen_src, vsnon_src, tskini_src, 
              ind, indi, indj,
              aicen_tar, vicen_tar, vsnon_tar, tskini_tar,
              *args):
    '''
    repartition ice fraction and volume to cice categories
    '''

    atmp = np.zeros(aicen_tar.shape[1], dtype='float32')
    htmp = np.zeros(aicen_tar.shape[1], dtype='float32')
    atmp[ind] = aicen_src[indj, indi]
    maska = atmp > puny
    maskb = np.logical_not(maska)
    htmp[maska] = vicen_src[indj[maska],indi[maska]]/atmp[maska]
    hb = ice_cat_bounds()  
    hb[-1] = 1.e15
    for n in range(1, ncat+1):
        maskc = np.logical_and(htmp > hb[n-1], htmp <= hb[n])   
        aicen_tar[n-1, ind[maskc]] =  aicen_src[indj[maskc], indi[maskc]]
        vicen_tar[n-1, ind[maskc]] =  vicen_src[indj[maskc], indi[maskc]]
        vsnon_tar[n-1, ind[maskc]] =  vsnon_src[indj[maskc], indi[maskc]]
        tskini_tar[n-1, ind[maskc]] =  tskini_src[indj[maskc], indi[maskc]]
    if args:
        tw_src, sw_src, tw_tar, sw_tar = args
        tw_tar[ind] =  tw_src[indj, indi]
        sw_tar[ind] =  sw_src[indj, indi]
             
    


if __name__ == "__main__":

    pass
