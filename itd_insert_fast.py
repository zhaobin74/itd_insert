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


def plot_pole_new(LON,LAT,VAR,levels,setcolor,setnorm,titlestr,POLE,ptype,MER):

    if  (POLE=='N'):
        m = Basemap(projection='npstere',lon_0=0,boundinglat=45)
    if  (POLE=='S'):
        m = Basemap(projection='spstere',lon_0=180,boundinglat=-45)
    m.drawcoastlines()
    m.fillcontinents()
    m.drawcountries()
    plt.title(titlestr)
    x, y =m(LON,LAT)
    if ptype=='cont':
        m.contourf(x,y,VAR, levels, origin='lower',cmap=setcolor, norm=setnorm, extend='both')
        #plt.colorbar(orientation='vertical',extend='both',shrink=0.4)
    if ptype=='plot':
        plt.plot(x,y,'.')#marker='.',color='k')
    if ptype=='scatter':
        #plt.plot(x,y,'.')#marker='.',color='k')
        VAR[abs(VAR)>999]='nan'
        print 'min='+str(np.nanmin(VAR))
        print 'max='+str(np.nanmax(VAR))
        valmin=min(levels)
        valmax=max(levels)

        plt.scatter(x,y,8*VAR/VAR,VAR,marker='o',vmin=valmin,vmax=valmax,cmap='jet',linewidths=0)

    m.drawparallels(np.arange(-90.,120.,15.),labels=[1,0,0,0]) # draw parallels
    m.drawmeridians(np.arange(0.,420.,30.),labels=MER) # draw meridians

def plot_pole_2(LON,LAT,VAR, LON1,LAT1,VAR1, levels, levels1, setcolor,setnorm,titlestr,POLE,ptype,MER):

    if  (POLE=='N'):
        m = Basemap(projection='npstere',lon_0=0,boundinglat=85)
    if  (POLE=='S'):
        m = Basemap(projection='spstere',lon_0=180,boundinglat=-45)
    m.drawcoastlines()
    m.fillcontinents()
    m.drawcountries()
    plt.title(titlestr)
    x, y =m(LON1,LAT1)
    plt.contour(x,y,VAR1, levels1, origin='lower',colors='k', linewidths=2)
    x, y =m(LON,LAT)
    plt.contourf(x,y,VAR, levels, origin='lower',cmap=setcolor, norm=setnorm, extend='both')

    m.drawparallels(np.arange(-90.,120.,15.),labels=[1,0,0,0]) # draw parallels
    m.drawmeridians(np.arange(0.,420.,30.),labels=MER) # draw meridians




class saltwatertile:

    def __init__(self, file): 
         
       header = np.genfromtxt(file, dtype='i4', usecols=(0), max_rows=8)
       #print header
       self.atm = 'x'.join([str(x) for x in header[3:5]])
       self.ocn = 'x'.join([str(x) for x in header[6:]])
       self.nx, self.ny = header[6], header[7]
       print self.atm, self.ocn 
       tile=np.genfromtxt(file, dtype=[('type','i1'), ('area','f8'), ('lon','f8'),('lat','f8'), ('gi1','i4'),
                           ('gj1','i4'), ('gw1','f8'),
                           ('idum','i4'), ('gi2','i4'), ('gj2','i4'), ('gw2','f8')], skip_header=8)
       n1=0
       n2=0
       for n in range(1, tile.shape[0]+1, 1):
           if tile[n-1][0] == 0:
               n1 = n
               break
       #print n1
       for n in range(n1, tile.shape[0]+1, 1):
           if tile[n-1][0] != 0:
               n2 = n
               break
       #print n2
       icetile=tile[n1-1:]
       #print icetile.shape
       #print 'hhh: ',icetile[0][2], icetile[-1][2]
       self.size = icetile.shape[0]
       self.gi = icetile['gi2'][:]
       self.gj = icetile['gj2'][:]
       #return icetile


def get_nearest(lon, lat, LON, LAT, rad):
    lon[lon>80.0]=lon[lon>80.0]-360.0
    xs, ys, zs = lon_lat_to_cartesian(lon.flatten(), lat.flatten())
    xt, yt, zt = lon_lat_to_cartesian(LON.flatten(), LAT.flatten())
    points_in = zip(xs, ys, zs)
    print len(points_in)
    tree = cKDTree(points_in)
    #find indices of the nearest neighbors in the flattened array
    #d, inds = tree.query(zip(xt, yt, zt), k = 1)
    #get interpolated 2d field
    #zout = LON.copy().flatten()
    points = zip(xt, yt,zt)
    #print len(points)
    d, inds = tree.query(points, k = 1)
    return inds
    

def nearest_interp_new(z, LON, LAT, inds):
    zout = z.flatten()[inds].reshape(LON.shape)
    #zout.shape = LON.shape
    return zout

def lon_lat_to_cartesian(lon, lat, R = 1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z


def get_grid(atm, ocn): #reads lat lon for tripolar ocean grid 
    ##ncfile=Dataset('/gpfsm/dnb42/projects/p17/gvernier/SAND_BOXES/PLOT_ODAS/DATA/grid_spec_720x410x40.nc', "r")
    #ncfile=Dataset('/discover/nobackup/yvikhlia/coupled/Forcings/Ganymed/a90x540_o720x410/INPUT/grid_spec.nc',"r")
    #ncfile=Dataset('/gpfsm/dnb02/projects/p23/bzhao/s2s3-duoc04/scratch/INPUT/grid_spec.nc',"r")
    if ocn=='1440x1080':
       ncfile = Dataset('/discover/nobackup/yvikhlia/coupled/Forcings/a'+atm+'_o'+ocn+'.newtile/INPUT/grid_spec.nc', "r")
    else:
       ncfile = Dataset('/discover/nobackup/yvikhlia/coupled/Forcings/a'+atm+'_o'+ocn+'/INPUT/grid_spec.nc', "r")
    LON     = ncfile.variables['x_T'][:]
    LAT     = ncfile.variables['y_T'][:]
    numlev     = ncfile.variables['num_levels'][:]
    ncfile.close()

    return LON, LAT, numlev


def num_tiles(ifile):
   num = 0
   with Dataset(ifile) as src:
       num = src.dimensions['tile']
   return num



from mitgcm import (read_mit, remap_mit)
from gfdl import (read_gfdl, remap_gfdl, remap_gfdl_vec)



#CICE dimensions
from cice import *

Tmlt = melt_temp()

func_map = {'gfdl': (read_gfdl, remap_gfdl_vec), 
#func_map = {'gfdl': (read_gfdl, remap_gfdl), 
            'mit': (read_mit, remap_mit)} 

tilefile = sys.argv[1]
institute = sys.argv[2]
ice_source = sys.argv[3]
print 'tile file: ', tilefile
print 'source sea ice datafile: ', ice_source
print 'from ', institute
icein = sys.argv[4]
print 'GEOS restart template file: ', icein
#iceout = 'seaicethermo_internal_rst-'+institute+'_inserted_vectorized'
#iceout = 'saltwater_internal_rst-'+institute+'_inserted_vectorized'
#iceout = 'seaicethermo_internal_rst-'+institute+'_inserted'
iceout = icein+'-'+institute+'_inserted_vectorized'

print 'output GEOS seaice thermo restart file: ', iceout

index=[[0 for _ in range(ncat-1)] for _ in range(ncat)]
ii=range(ncat)
dist = defaultdict(list) 
for j in ii:
   for i in ii:
      if i != j:
          dist[j].append((abs(i-j),i+1)) 
for j in dist:
    dist[j].sort(key=lambda x: (x[0], x[1]*(-1)))
for j in dist:
    index[j]= [x[1]-1 for x in dist[j]]     
    #print j, dist[j]





sw = saltwatertile(tilefile)

# read in source dataset
source_data = func_map[institute][0](ice_source)

if len(source_data) == 4:
   aicen_s, vicen_s, vsnon_s, tskin_s = source_data 
elif len(source_data) == 6:
   aicen_s, vicen_s, vsnon_s, tskin_s, tw_s, sw_s = source_data 

print aicen_s.shape

def remap_energy(i, aicen, vicen, vsnon, tskin, eicen, esnon):
    for n in range(ncat):
       # assume linear temp profile and compute enthalpy
        if aicen[n,i] > puny: 
           ts_s = tskin[n,i]-Tice 
           height = vicen[n,i]/aicen[n,i]
           hi = height
           hs = 0.0  
           nls = nilyr  
           if vsnon[n,i] > puny:  
               hs = vsnon[n,i]/aicen[n,i]      
               height += hs 
               nls += nslyr
           slope = (Tf - ts_s) / height 
           for k in range(nls):
               if k < nls - nilyr:
                   Ti =  ts_s + slope*(k+0.5)*hs/float(nslyr)
                   esnon[k,n,i] =  (-rhos * (-cp_ice*Ti + Lfresh)) * \
                                 vsnon[n,i]/float(nslyr) 
               else:        
                   kl = k if nls == nilyr else k-nslyr    
                   Ti =  ts_s + slope*(hs+(kl+0.5)*hi/float(nilyr))
                   eicen[kl,n,i] =    \
                      -(rhoi * (cp_ice*(Tmlt[kl]-Ti)  \
                      + Lfresh*(1.0-Tmlt[kl]/Ti) - cp_ocn*Tmlt[kl])) \
                      * vicen[n,i]/float(nilyr) 
 
if num_tiles(icein) == sw.size:
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
       apondnout = dst['APONDN']
       hpondnout = dst['HPONDN']
       volpondout = dst['VOLPOND']
       tauageout = dst['TAUAGE']
       aicen = dst['FR'][:]
       vicen = dst['VOLICE'][:]
       vsnon = dst['VOLSNO'][:]
       tskin = dst['TSKINI'][:]
       eicen = dst['ERGICE'][:]
       esnon = dst['ERGSNO'][:]
       slmask = dst['SLMASK'][:]
       eicen = np.swapaxes(eicen,0,1)
       #print 'eice shape ', eicen.shape
       esnon = np.swapaxes(esnon,0,1)

       aicen[:,:]  = 0.0
       vicen[:,:]  = 0.0
       vsnon[:,:]  = 0.0
       eicen[:,:]  = 0.0
       esnon[:,:]  = 0.0
       tskin[:,:]  = Tice


       indi = sw.gi[:]-1 
       indj = sw.gj[:]-1   
       ind = np.array(range(sw.size))

       start = time.time()

       hs = np.zeros((ncat,sw.size), dtype='float64') 
       hin = np.zeros((ncat,sw.size), dtype='float64') 
       qin = np.zeros((nilyr,ncat,sw.size), dtype='float64')
       qsn = np.zeros((nslyr,ncat,sw.size), dtype='float64')

       #'''  
       func_map[institute][1](aicen_s, 
                              vicen_s, 
                              vsnon_s, 
                              tskin_s, 
                              ind, indi, indj, 
                              aicen, vicen, 
                              vsnon, tskin) 
       #'''

       for i in range(sw.size):
           '''
           func_map[institute][1](aicen_s[:,indj[i],indi[i]], 
                                  vicen_s[:,indj[i],indi[i]], 
                                  vsnon_s[:,indj[i],indi[i]], 
                                  tskin_s[:,indj[i],indi[i]], 
                                  aicen[:,i], vicen[:,i], 
                                  vsnon[:,i], tskin[:,i]) 
           '''
           remap_energy(i, aicen, vicen, vsnon, tskin, eicen, esnon) 


       maska = slmask > 0.5
       for n in range(ncat):
          aicen[n, maska] = 0.0
          vicen[n, maska] = 0.0
          vsnon[n, maska] = 0.0
          tskin[n, maska] = Tice
          for k in range(nilyr):
              eicen[k,n,maska] = 0.0
          for k in range(nslyr):
              esnon[k,n,maska] = 0.0


       aicenout[:] = aicen[:]
       tskinout[:] = tskin[:]
       vicenout[:] = vicen[:]
       vsnonout[:] = vsnon[:]
       eicenout[:] = np.swapaxes(eicen,0,1) 
       esnonout[:] = np.swapaxes(esnon,0,1)
       tauageout[:] = 0.0
       apondnout[:] = 0.0
       hpondnout[:] = 0.0
       volpondout[:] = 0.0

       end = time.time()
       print("Elapsed (aggregating onto CICE ITD) = %s" % (end - start))
       

else:
    from write_ice_restart import create_tile_rst

    #fin = 'saltwater_internal_rst' 

    aicen = np.zeros((ncat, sw.size), dtype='float64')
    vicen = np.zeros((ncat, sw.size), dtype='float64')
    vsnon = np.zeros((ncat, sw.size), dtype='float64')
    tskin = np.zeros((ncat, sw.size), dtype='float32')
    eicen = np.zeros((nilyr, ncat, sw.size), dtype='float64')
    esnon = np.zeros((nslyr, ncat, sw.size), dtype='float64')
    tskinw = np.zeros(sw.size, dtype='float32')
    sskinw = np.zeros(sw.size, dtype='float32')

    tskin[:]  = Tice


    indi = sw.gi[:]-1 
    indj = sw.gj[:]-1   
    ind = np.array(range(sw.size))

    start = time.time()

    if len(source_data) == 4:
        func_map[institute][1](aicen_s, 
                               vicen_s, 
                               vsnon_s, 
                               tskin_s, 
                               ind, indi, indj, 
                               aicen, vicen, 
                               vsnon, tskin) 
    elif len(source_data) == 6:
        func_map[institute][1](aicen_s, 
                               vicen_s, 
                               vsnon_s, 
                               tskin_s, 
                               ind, indi, indj, 
                               aicen, vicen, 
                               vsnon, tskin,
                               tw_s, sw_s, tskinw, sskinw) 
    

    for i in range(sw.size):
         '''
           func_map[institute][1](aicen_s[:,indj[i],indi[i]], 
                                  vicen_s[:,indj[i],indi[i]], 
                                  vsnon_s[:,indj[i],indi[i]], 
                                  tskin_s[:,indj[i],indi[i]], 
                                  aicen[:,i], vicen[:,i], 
                                  vsnon[:,i], tskin[:,i]) 
         '''
         remap_energy(i, aicen, vicen, vsnon, tskin, eicen, esnon) 

    ''' 
    for n in range(ncat):
        maska = aicen[n,:] > 1.e20
        aicen[n, maska] = 0.0
        vicen[n, maska] = 0.0
        vsnon[n, maska] = 0.0
        tskin[n, maska] = Tice
        for k in range(nilyr):
             eicen[k,n,maska] = 0.0
        for k in range(nslyr):
             esnon[k,n,maska] = 0.0
    '''

    eicen = np.swapaxes(eicen,0,1) 
    esnon = np.swapaxes(esnon,0,1)

    end = time.time()
    print("Elapsed (aggregating onto CICE ITD) = %s" % (end - start))

    start = time.time()
    if len(source_data) == 4:
        create_tile_rst(icein, iceout, sw.size, aicen, vicen,
                       vsnon, tskin, eicen, esnon)
    elif len(source_data) == 6:
        create_tile_rst(icein, iceout, sw.size, aicen, vicen,
                       vsnon, tskin, eicen, esnon, sst=tskinw, sss=sskinw)

    end = time.time()
    print("Elapsed (output final restart file) = %s" % (end - start))





