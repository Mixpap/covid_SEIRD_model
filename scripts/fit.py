import numpy as np
import math

import os
from os.path import expanduser
home = expanduser("~")
os.environ["OMP_NUM_THREADS"] = "1"
#import warnings
#warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, '../src')
from importlib import reload

#astropy
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel,Gaussian2DKernel
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.stats import mad_std,sigma_clipped_stats
from photutils import detection

#scipy
from scipy.interpolate import pchip
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.optimize import minimize,differential_evolution,basinhopping,dual_annealing,NonlinearConstraint

#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display

#mcmc
import emcee

z=0.014313
arc_to_kpc=(cosmo.angular_diameter_distance(z=z)/206265).to(u.kpc)/u.arcsec

import utils
cubefolder='../data/NGC6328/cubes/'
cloudsfolder='../data/NGC6328/clouds/'
datafolder='../data/NGC6328/other/'


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file to mine for data points (.out)")
parser.add_argument("--walkers", help="number of walkers",type=int, default=2)
parser.add_argument("--steps", help="number of steps",type=int, default=300)
parser.add_argument("--save", help="ssave file of the chains")
args = parser.parse_args()


file = args.file
X3d=np.array([])
Y3d=np.array([])
A3d=np.array([])
V3d=np.array([])
a3d=np.array([])
b3d=np.array([])
c3d=np.array([])
Sv3d=np.array([])
with open(file, "r") as fp:
    lines = [line for line in fp if 'Results' in line]
    for line in lines:
        A3d=np.append(A3d,float(line.split()[4]))
        X3d=np.append(X3d,float(line.split()[5][3:-1]))
        Y3d=np.append(Y3d,float(line.split()[6][3:]))
        V3d=np.append(V3d,float(line.split()[7][3:]))
        a3d=np.append(a3d,float(line.split()[8][3:-1]))
        b3d=np.append(b3d,float(line.split()[9][3:-1]))
        c3d=np.append(c3d,float(line.split()[10][3:-1]))
        Sv3d=np.append(Sv3d,float(line.split()[11][3:]))
        #print(float(line.split()[8][3:-1]))
        # if int(line.split()[11])<300:
        #     tried.append(line.split()[3])
print(A3d.shape)       
R3d= np.sqrt(X3d**2+Y3d**2)
PHI3d=np.arctan2(Y3d,X3d)

sys.path.insert(1, '../scripts/fitters')
import p_six
import model_creation

six_dic=model_creation.prepare_params(p_six.dic)
R_d = np.arange(0.01,5.2,0.05)
dR_d=0.08/2
bounds=six_dic['bounds']
params0=six_dic['params0']


def posterior(params,X,Y,V,R,PHI,disks_f,R_d,dR_d,dic,dV0=500,delta_v=30):
    prior = -0.5*np.sum((params-dic['p0'])**2/dic['dp']**2)

    R_d,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=disks_f(params,R_d,dR_d,dic)
    dV=np.array([dV0]*X.shape[0])

    for k,(x,y,v,r,phi) in enumerate(zip(X,Y,V,R,PHI)):
        dvi_min=dV0
        for m,(rd,vcd,i_d,phi_d,tani2,vsini,tani12,tani22) in enumerate(zip(R_d,Vc_d,I_d,Phi_d,tani2s,vsinis,tanI12,tanI22)):

            rell1=(rd-dR_d)/(1+tani12*np.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i1,R-dR)
            rell2=(rd+dR_d)/(1+tani22*np.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i2,R+dR)

            if (r>=rell1) and (r<=rell2):
                Vd_sky=vsini*np.sin(phi-phi_d)/(1 +tani2*np.cos(phi-phi_d)**2)**0.5
                dvi=np.abs(Vd_sky-v)
                if dvi<dvi_min:
                    dvi_min=dvi
                    dV[k]=Vd_sky-v

    like = -0.5*np.sum(dV**2)/delta_v**2
    
    return prior+like,prior,like



def mcmc_fit(posterior,walkers,nsteps,cores,bounds,args=None,backend=None):
    ndim = bounds.shape[0] # How many parameters to fit
    nwalkers = ndim*walkers#60#16#4 # Minimum of 2 walkers per free parameter
    print(ndim,nwalkers,nwalkers/cores)
    pos = [np.random.uniform(bounds[:,0],bounds[:,1]) for i in range(nwalkers)]
    if backend is not None:
        backend = emcee.backends.HDFBackend(backend)
        backend.reset(nwalkers, ndim)
    #print(kwargs)
    if cores>1:
        with Pool(cores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior,pool=pool,backend=backend,args=args,
                                            moves=[(emcee.moves.DEMove(),0.75),(emcee.moves.DESnookerMove(),0.25),(emcee.moves.StretchMove(),0.25),])# Setup the sampler
            result=sampler.run_mcmc(pos, nsteps,progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior,backend=backend,args=args,
                                        moves=[(emcee.moves.DEMove(),0.1),(emcee.moves.DESnookerMove(),0.1),(emcee.moves.StretchMove(),0.8),])# Setup the sampler
        result=sampler.run_mcmc(pos, nsteps,progress=True)
    #samples = sampler.chain[:, 0:, :].reshape((-1, ndim))
    return sampler

sample=mcmc_fit(posterior,args.walkers,args.steps,1,six_dic['bounds'],
    args=[X3d,Y3d,V3d,R3d,PHI3d,p_six.make_disks,R_d,dR_d,six_dic,400,40],backend=args.save+'.h5')