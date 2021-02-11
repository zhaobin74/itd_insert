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

puny = 1.e-11
Tice = 273.16

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

def remap_gfdl(aicen_src, vicen_src, vsnon_src, tskini_src, 
               aicen_tar, vicen_tar, vsnon_tar, tskini_tar):
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
    if aicen_tar[0] > puny:
       tskini_tar[0] = np.minimum(Tice, np.dot(aicen_src[:3], tskini_src[:3])/aicen_tar[0])  
    else:
        tskini_tar[0] = Tice
    tskini_tar[1] =  np.minimum(Tice, tskini_src[3])
    tskini_tar[2] =  np.minimum(Tice, tskini_src[4])
    tskini_tar[3:] = Tice

def remap_gfdl_vec(aicen_src, vicen_src, vsnon_src, tskini_src, 
                    ind, indi, indj,
                    aicen_tar, vicen_tar, vsnon_tar, tskini_tar):
    #sis_ic=np.array([0.0, 0.1, 0.3, 0.7, 1.1], dtype='float64')
    aicen_tar[0,ind] = np.sum(aicen_src[:3, indj, indi], axis=0) 
    aicen_tar[1,ind] = aicen_src[3, indj, indi]
    aicen_tar[2,ind] = aicen_src[4, indj, indi]
    aicen_tar[3:,ind] = 0.0
    #vicen_tar[0,ind] = np.dot(aicen_src[:3,indj,indi], vicen_src[:3,indj,indi]) 
    vicen_tar[0,ind] = np.sum(aicen_src[:3,indj,indi]*vicen_src[:3,indj,indi],axis=0) 
    vicen_tar[1,ind] = aicen_src[3,indj, indi] * vicen_src[3,indj, indi]
    vicen_tar[2,ind] = aicen_src[4,indj, indi] * vicen_src[4,indj, indi]
    vicen_tar[3:,ind] = 0.0
    #vsnon_tar[0,ind] = np.dot(aicen_src[:3,indj, indi], vsnon_src[:3,indj, indi]) 
    vsnon_tar[0,ind] = np.sum(aicen_src[:3,indj,indi]*vsnon_src[:3,indj,indi],axis=0) 
    vsnon_tar[1,ind] = aicen_src[3,indj,indi] * vsnon_src[3,indj,indi]
    vsnon_tar[2,ind] = aicen_src[4,indj,indi] * vsnon_src[4,indj,indi]
    vsnon_tar[3:,ind] = 0.0
    maska = aicen_tar[0,:] > puny
    maskb = aicen_tar[0,:] <= puny
    tskini_tar[0,ind[maska]] = np.minimum(Tice, np.sum(aicen_src[:3,indj[maska],indi[maska]]* 
                                tskini_src[:3,indj[maska],indi[maska]], axis=0)/aicen_tar[0,ind[maska]])  
    tskini_tar[0, ind[maskb]] = Tice
    tskini_tar[1,ind] =  np.minimum(Tice, tskini_src[3,indj,indi])
    tskini_tar[2,ind] =  np.minimum(Tice, tskini_src[4,indj,indi])
    tskini_tar[3:,ind] = Tice

if __name__ == "__main__":
    icein = 'ice_model.res.nc'
    ai_s, vi_s, vs_s, ti_s = read_gfdl(icein)
    ai_t = np.zeros(5) 
    vi_t = np.zeros(5) 
    vs_t = np.zeros(5) 
    ti_t = np.zeros(5) 
    i, j = 100, 199 
    remap_gfdl(ai_s[:,j,i], vi_s[:,j,i], vs_s[:,j,i],  ti_s[:,j,i], 
               ai_t, vi_t, vs_t, ti_t) 
    print ai_t
    print vi_t
    print vs_t
    print ti_t
    ncat = 5
    nilyr = 4
    nslyr = 1
    Tf = -1.8
    for n in range(ncat):
          # assume linear temp profile and compute enthalpy
        if ai_t[n] > puny: 
            ts_s = ti_t[n]-Tice 
            height = vi_t[n]/ai_t[n]
            hi = height
            hs = 0.0  
            nls = nilyr  
            if vs_t[n] > puny:  
               hs = vs_t[n]/ai_t[n]      
               height += hs 
               nls += nslyr
               slope = (Tf - ts_s) / height 
            print n+1, 0, ts_s, hi, hs  
            for k in range(nls):
               if k < nls - nilyr:
                   Ti =  ts_s + slope*(k+0.5)*hs/float(nslyr)
               else:        
                   kl = k if nls == nilyr else k-nslyr    
                   Ti =  ts_s + slope*(hs+(kl+0.5)*hi/float(nilyr))
               print n+1, k+1, Ti  
