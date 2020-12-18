#! /usr/bin/env python
import numpy as np

ncat = 5
nilyr = 4
nslyr = 1

puny = 1.e-11

c3 = 3.0
c15 = 15.0
p5 = 0.5
c1 = 1.0
c2 = 2.0

Tice = 273.16
Tf = -1.8
saltmax = 3.2
nsal = 0.407
msal = 0.573
depressT = 0.054
rhoi      = 917.0
rhow      = 1026.0
rhos      = 330.0
cp_ice    = 2106.
cp_ocn    = 4218.
Lsub      = 2.835e6
Lvap      = 2.501e6
Lfresh    = Lsub-Lvap 

def ice_cat_bounds():

   hin_max = np.zeros(ncat+1, dtype='float64') 
    
   cc1 = c3/float(ncat)
   cc2 = c15*cc1
   cc3 = c3
   for n in range(1, ncat+1):
       x1 = float(n-1)/float(ncat)
       hin_max[n] = hin_max[n-1] \
                   + cc1 + cc2*(c1 + np.tanh(cc3*(x1-c1)))

   return hin_max

def melt_temp():
   Tmlt = np.zeros(nilyr+1, dtype='float64')
   salin = np.zeros(nilyr+1, dtype='float64')
   for k in range(nilyr):
       zn = (float(k+1)-p5) / float(nilyr)
       salin[k]=(saltmax/c2)*(c1-np.cos(np.pi*np.power(zn,nsal/(msal+zn))))
       Tmlt[k] = -salin[k]*depressT
   salin[nilyr] = saltmax
   Tmlt[nilyr] = -salin[nilyr]*depressT
   return Tmlt 

if __name__ == "__main__":
   hb = ice_cat_bounds()
   for h in hb:
      print h
