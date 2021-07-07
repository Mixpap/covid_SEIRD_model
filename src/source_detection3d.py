import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
from os.path import expanduser
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel,Gaussian2DKernel
home = expanduser("~")
from importlib import reload
from astropy import units as u
from scipy.interpolate import pchip
from scipy.signal import find_peaks
from photutils import detection
from scipy.optimize import minimize,differential_evolution,basinhopping,dual_annealing
from astropy.cosmology import WMAP9 as cosmo
from astropy.stats import mad_std,sigma_clipped_stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import math


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cores", help="number of cores",type=int, default=4)
parser.add_argument("--maxN", help="maximum number of 3d gaussians",type=int, default=10)
parser.add_argument("--gridR", help="grid size",type=float, default=0.2)
parser.add_argument("--test", help="small grid for testing",type=bool, default=False)
args = parser.parse_args()

z=0.014313
arc_to_kpc=(cosmo.angular_diameter_distance(z=z)/206265).to(u.kpc)/u.arcsec

import utils

reload(utils)
cubefolder='../data/NGC6328/cubes'#home+'/astro/ARC/data/cubes/NGC6328/'
cube21_file=cubefolder+'NGC6328_CO_selfcal_20km_natural_pbcor.fits'
ucube21_file=cubefolder+'NGC6328_CO_selfcal_20km_natural.fits'
z=0.014313
CO21,CO21_total=utils.load_fits_cube(cube_file=cube21_file,ucube_file=ucube21_file,pbfile=None,
                    drop={'x':2.6,'y':4.9,'v':460},z=z,sigma=5,
                     beam=[0.296,0.220,38.624],zaxis='freq',restFreq=230.538e9,
                    debug=False,nosignalregions = [{'x0':300,'y0':600,'dx':200,'dy':500},{'x0':1200,'y0':600,'dx':200,'dy':500}])

CO21['cube']=CO21['cube'].fillna(0)

def gmodel(x,y,v,N,p):
    s=np.zeros(x.shape)
    pars=np.split(np.array(p),N)
    for A,xc,yc,vc,a,b,c,sv in pars:
        s+=A*np.exp(-0.5*(a*(x-xc)**2 + c*(y-yc)**2 +b*(x-xc)*(y-yc)+ (v-vc)**2/sv**2))
    return s

def L(pars,x,y,v,data,er,N):
    return np.sum((gmodel(x,y,v,N,pars)-data)**2/er**2)
    
    
Rg=args.gridR#0.2
if args.test:
    xgrid=np.arange(-0.5,-0.3,2*Rg)
    ygrid=np.arange(-1,0,2*Rg)
else:
    xgrid0=-0.61
    xgrid1=2
    ygrid0=-3.4
    ygrid1=4.8
    xgrid=np.arange(xgrid0,xgrid1,2*Rg)
    ygrid=np.arange(-3.4,4.8,2*Rg)

print(f"Starting the detection of 3d clouds in a grid starting from {xgrid0},{ygrid0} with a width of {Rg} (total grid points {len(xgrid)*len(ygrid)}) with {args.cores} cores ({len(xgrid)*len(ygrid)/args.cores:2f} grids/core)")


rms=0.000247108285
er=3*rms
rms3=3*rms
ncrit=300
maxN=args.maxN
def loop(xy):
    xx=xy[0]
    yy=xy[1]
    name=f"{xx:.2f}_{yy:.2f}"
    tes=CO21.where((CO21.x>xx-Rg)&(CO21.x<xx+Rg)&(CO21.y>yy-Rg)&(CO21.y<yy+Rg),drop=True)
    #print(xx,yy,np.sum(tes['cube'].data>rms3))
    print(f"Trying grid point {name} ({xx-Rg:.2f} to {xx+Rg:.2f},{yy-Rg:.2f} to {yy+Rg:.2f}) (data-points (3s) {np.sum(tes['cube'].data>rms3)} ({np.sum(tes['cube'].data>rms3)/(2*Rg)**2:.2f} points/kpc2))")
    
    if (np.sum(tes['cube'].data>rms3)>ncrit):
        Xcloud=np.array([])
        Ycloud=np.array([])
        Vcloud=np.array([])
        acloud=np.array([])
        bcloud=np.array([])
        ccloud=np.array([])
        Svcloud=np.array([])
        Acloud=np.array([])
        #Vg,Yg,Xg=np.meshgrid(tes.v,tes.y,tes.x)
        Vg,Yg,Xg=np.meshgrid(tes.v,tes.y,tes.x, indexing='ij')
        minbic=1e17
        bic=minbic-11
        N=1
        
        while True:
            bounds=[[rms3,np.nanmax(tes['cube'].data)],[xx-Rg,xx+Rg],[yy-Rg,yy+Rg],[-400,400],[65,625],[-560,560],[65,625],[20,40]]*N
            
            sol=differential_evolution(L,bounds=bounds,args=(Xg,Yg,Vg,tes['cube'].data,er,N))
            params=sol.x
            logL=sol.fun
            n=tes['cube'].data[~np.isnan(tes['cube'].data)].size#xb.size
            k=len(bounds)
            bic=k*np.log(n)+2*n*logL
            
            print(f" fit {name} with {N} gaussians: bic:{bic:.2f} ({bic-minbic:.2f})")
            if (bic>minbic): 
                pars=np.split(np.array(bpars),N-1)
                print(f"Found {N-1} clouds in {name} grid point")
                for ac,xc,yc,vc,a,b,c,sv in pars:
                    print(f"Results for {name}: max {ac/rms:.2f} xc:{xc:.2f}, yc:{yc:.2f} vc:{vc:.2f} a:{a:.2f}, b:{b:.2f}, c:{c:.2f}, sv:{sv:.2f}")
                    Xcloud=np.append(Xcloud,xc)
                    Ycloud=np.append(Ycloud,yc)
                    Vcloud=np.append(Vcloud,vc)
                    acloud=np.append(acloud,a)
                    bcloud=np.append(bcloud,b)
                    ccloud=np.append(ccloud,c)
                    Svcloud=np.append(Svcloud,sv)
                    Acloud=np.append(Acloud,ac)  
                break
            elif (N>=maxN):
                pars=np.split(np.array(params),N)
                print(f"Found {N} clouds in {name} grid point")
                for ac,xc,yc,vc,a,b,c,sv in pars:
                    print(f"Results for {name}: max {ac/rms:.2f} xc:{xc:.2f}, yc:{yc:.2f} vc:{vc:.2f} a:{a:.2f}, b:{b:.2f}, c:{c:.2f}, sv:{sv:.2f}")
                    Xcloud=np.append(Xcloud,xc)
                    Ycloud=np.append(Ycloud,yc)
                    Vcloud=np.append(Vcloud,vc)
                    acloud=np.append(acloud,a)
                    bcloud=np.append(bcloud,b)
                    ccloud=np.append(ccloud,c)
                    Svcloud=np.append(Svcloud,sv)
                    Acloud=np.append(Acloud,ac)
                break
            else:
                minbic=bic
                bpars=params
                N+=1
    return [Xcloud,Ycloud,Vcloud,acloud,bcloud,ccloud,Svcloud,Acloud]



import itertools
import multiprocessing
xy=list(itertools.product(xgrid,ygrid))

pool = multiprocessing.Pool(args.cores)
res=pool.map(loop,xy)
print(res)
np.save('res3d.npy', res, allow_pickle=True)
