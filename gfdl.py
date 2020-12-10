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



def read_gfdl(infile):
    # read in the GFDL SIS sea ice restart file (netcdf)
    # return aice,hice/volice in dimension (5, ny, nx)
    '''
    : type infile: str  
    : rtype: NDArray, NDArray
    ''' 
    with Dataset(infile) as src:
       aicen = src['part_size'][0]
       vicen = src['h_ice'][0]
       vsnon = src['h_snow'][0]
       tskin = src['t_surf'][0]
    return (aicen[1:,:,:], vicen, vsnon, tskin[1:,:,:])

def remap_gfdl(aicen_src, vicen_src, vsnon_src, aicen_tar, vicen_tar, vsnon_tar):
    #sis_ic=np.array([0.0, 0.1, 0.3, 0.7, 1.1], dtype='float64')
    aicen_tar[0] = np.sum(aicen_src[:3]) 
    aicen_tar[1] = aicen_src[3]
    aicen_tar[2] = aicen_src[4]
    aicen_tar[3:] = 0.0
    vicen_tar[0] = np.dot(aicen_src[:3], vicen_src[:3]) 
    vicen_tar[1] = aicen_src[3] * vicen_src[3]
    vicen_tar[2] = aicen_src[4] * vicen_src[4]
    vicen_tar[3:] = 0.0
    vsnon_tar[0] = np.dot(aicen_src[:3], vsnon_src[:3]) 
    vsnon_tar[1] = aicen_src[3] * vsnon_src[3]
    vsnon_tar[2] = aicen_src[4] * vsnon_src[4]
    vsnon_tar[3:] = 0.0

if __name__ == "__main__":
    icein = 'ice_model.res.nc'
    ai_s, vi_s, vs_s, ts_s = read_gfdl(icein)
    ai_t = np.zeros(5) 
    vi_t = np.zeros(5) 
    vs_t = np.zeros(5) 
    i, j = 100, 199 
    remap_gfdl(ai_s[:,j,i], vi_s[:,j,i], vs_s[:,j,i],
               ai_t, vi_t, vs_t) 
    print ai_t
    print vi_t
    print vs_t
