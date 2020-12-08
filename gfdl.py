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
    pass

def remap_gfdl(aicen_source, vicen_source, inds, indis, indjs, aicen_target, vicen_target):
    sis_ic=np.array([0.0, 0.1, 0.3, 0.7, 1.1], dtype='float64')
    ''' 
    fac1=(0.6445-0.3)/(0.7-0.3)
    fac2=((1.391-1.1)/(2.035-1.085))
    fac3=((2.47-2.035)/(3.42-2.035))
    fac4=((4.567-3.42)/(5.31-3.42))
    aicenpm5[0, inds] = sum(aicenpm[:2,indjs,indis], axis=0) + \
                            aicenpm[2,indjs,indis] * fac1     
    aicenpm5[1, inds] = aicenpm[2,indjs,indis]*(1.-fac1) + \
                        aicenpm[3,indjs,indis] +           \
                        aicenpm[4,indjs,indis] 
    aicenpm5[2, inds] = aicenpm[3,indjs,indis]*(1.-fac2) + aicenpm[4,indjs,indis] * fac3
    aicenpm5[3, inds] = aicenpm[4,indjs,indis]*(1.-fac3) + aicenpm[5,indjs,indis] * fac4
    aicenpm5[4, inds] = aicenpm[5,indjs,indis]*(1.-fac4) + np.sum(aicenpm[6:,indjs,indis],axis=0)
    '''

if __name__ == "__main__":

    pass
