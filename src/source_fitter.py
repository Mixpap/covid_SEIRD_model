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
from scipy import ndimage
from scipy.signal import find_peaks
from photutils import detection
from scipy.optimize import minimize,differential_evolution,NonlinearConstraint
from astropy.cosmology import WMAP9 as cosmo
from astropy.stats import mad_std,sigma_clipped_stats

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
import math
import emcee
import corner
from multiprocessing import Pool

import utils
import source_detection


z=0.014313
arc_to_kpc=(cosmo.angular_diameter_distance(z=z)/206265).to(u.kpc)/u.arcsec

# cubefolder=home+'/astro/ARC/data/cubes/NGC6328/'
# cube21_file=cubefolder+'CO21/NGC6328_CO_selfcal_20km_natural_pbcor.fits'
# ucube21_file=cubefolder+'CO21/NGC6328_CO_selfcal_20km_natural.fits'
# CO21,CO21_total=utils.load_fits_cube(cube_file=cube21_file,ucube_file=ucube21_file,pbfile=None,
#                     drop={'x':2.6,'y':4.9,'v':460},z=z,sigma=5,
#                      beam=[0.296,0.220,38.624],zaxis='freq',restFreq=230.538e9,
#                     debug=False,nosignalregions = [{'x0':300,'y0':600,'dx':200,'dy':500},{'x0':1200,'y0':600,'dx':200,'dy':500}])



# IX=CO21.attrs['ixx']
# IY=CO21.attrs['iyy']
# X=CO21.x.data[IX]
# Y=CO21.y.data[IY]

# np.save('IX.npy',IX)
# np.save('IY.npy',IY)
# np.save('X.npy',X)
# np.save('Y.npy',Y)

IX=np.load('IX.npy',allow_pickle=True)
IY=np.load('IY.npy',allow_pickle=True)
X=np.load('X.npy',allow_pickle=True)
Y=np.load('Y.npy',allow_pickle=True)

PARS=np.load('PARS.npy',allow_pickle=True)
BIC=np.load('BIC.npy',allow_pickle=True)
NN=np.load('NN.npy',allow_pickle=True)


#first filter
filt= (BIC>0)#&(X>-1)&(X<1)
X=X[filt]
Y=Y[filt]
IY=IY[filt]
IX=IX[filt]
#R=np.sqrt(X**2+Y**2)
NN=NN[filt]
PARS=PARS[filt]
BIC=BIC[filt]

# second filter
#filter_noise=3.7*CO21.attrs['rms']
rms=0.000247108
filter_noise=3.5*rms
deli=[]
for i in range(PARS.shape[0]):
    dell=[]
    n=0
    for j in range(0,PARS[i].shape[0],3):
        if PARS[i][j]<filter_noise:
            dell.append(j)
            dell.append(j+1)
            dell.append(j+2)
            n+=1
    PARS[i]=np.delete(PARS[i],dell)
    NN[i]=NN[i]-n
    if NN[i]<=0:
        deli.append(i)
X=np.delete(X,deli)
Y=np.delete(Y,deli)
IX=np.delete(IX,deli)
IY=np.delete(IY,deli)

NN=np.delete(NN,deli)
BIC=np.delete(BIC,deli)
PARS=np.delete(PARS,deli)
    
R=np.sqrt(X**2+Y**2)
PHI=np.arctan2(Y,X)
#real=np.ones(X.shape,dtype=bool)


XX=[]
YY=[]
for i in range(NN.shape[0]):
    XX=np.append(XX,np.ones(int(NN[i]))*X[i])
    YY=np.append(YY,np.ones(int(NN[i]))*Y[i])   
A=[]
for i in range(NN.shape[0]):
    A.append([None]*int(NN[i]))
V=[]
for i in range(NN.shape[0]):
    V.append([None]*int(NN[i]))
    
S=[]
for i in range(NN.shape[0]):
    S.append([None]*int(NN[i]))
    
for k,(n,pp) in enumerate(zip(NN,PARS)):
    V[k]=np.array(np.split(pp,n))[:,1]
    A[k]=np.array(np.split(pp,n))[:,0]
    S[k]=np.array(np.split(pp,n))[:,2]

VV=np.concatenate(V).ravel()
AA=np.concatenate(A).ravel()#/a.rms
AAs=AA/rms
SS=np.concatenate(S).ravel()#/a.rms

from scipy import special
def calc_res(A,S,V):
    return np.sqrt(np.pi/2)*A*S*(special.erf((V+400)/(np.sqrt(2)*S))-special.erf((V-400)/(np.sqrt(2)*S)))

II=calc_res(AA,SS,VV)



macc_HI_Vrot=np.loadtxt('../../reports/NGC6328/macc_HI_Vrot.csv',delimiter=',')
vc_HI_Vrot=np.loadtxt('../../reports/NGC6328/vc_HI_Vrot.csv',delimiter=',')
V_sys=4274

min_to_kpc=(cosmo.angular_diameter_distance(0.014313)/u.radian).to(u.kpc/u.arcmin)
sec_to_kpc=(cosmo.angular_diameter_distance(0.014313)/u.radian).to(u.kpc/u.arcsec)

R_macc_HI=(macc_HI_Vrot[:,0])*u.arcmin*min_to_kpc
V_macc_HI=macc_HI_Vrot[:,1]-V_sys

R_vc_HI=(vc_HI_Vrot[:,0])*u.arcsec*sec_to_kpc
V_vc_HI=vc_HI_Vrot[:,1]-V_sys-20

Rm=np.abs(R_macc_HI.value)
Rvc=np.abs(R_vc_HI.value)
Vm_20=V_macc_HI/np.sin(np.radians(20))
Vvc_20=V_vc_HI/np.sin(np.radians(20))
HI_sigma=40
Rc=20

Rmf=Rm[(Rm>21)&(Rm<36)]
Vmf=np.abs(Vm_20[(Rm>21)&(Rm<36)])
Rvcf=Rvc[(Rvc>21)&(Rvc<36)]
Vvcf=np.abs(Vvc_20[(Rvc>21)&(Rvc<36)])

param_set={
   '7':{'pa':{0.01:[69.85,[10,80],[70.5,15],True,[-10,10]],
                0.3:[72.75,[20,80],[73,8],True,[-8,8]],
                0.7:[72.1,[20,180],[72.1,0.5],False,[-1,1]],
                  1.0:[145,[100,160],[125,20],True,[-15,15]],
                  #1.5:[145,[80,170],[120,20],True,[-15,15]],
                  2.0:[165.3,[140,185],[162,15],True,[-11,11]],
                  2.75:[165.3,[140,185],[162,10],True,[-11,11]],
                #3.5:[178.2,[165,190],[178.2,6],True],
                  3.5:[176,[160,185],[178,8],True,[-5,5]],#check this
                  4.1:[170,[150,185],[172,10],True,[-9,9]],
                  5.3:[161.4,[130,180],[162,20],False,[-20,20]],  
                 11.8:[155,[151,190],[127,10],False,[-9.4,9.4]],13.3:[149,[70,200],[118,10],False,[-9.4,9.4]],14.7:[142,[70,200],[118,10],False,[-9.4,9.4]],16.2:[135,[70,200],[118,10],False,[-9.4,9.4]],17.7:[127,[70,200],[118,10],False,[-9.4,9.4]],19.2:[120,[70,200],[118,10],False,[-9.4,9.4]],20.6:[116,[70,200],[105,10],False,[-9.4,9.4]],22.1:[112,[70,200],[105,10],False,[-9.4,9.4]],23.6:[109,[70,200],[105,10],False,[-9.4,9.4]]},             
      'i':{0.01:[123.5,[100,170],[180-57,10],True,[-10,10]],
                0.3:[124.6,[100,170],[180-56.3,10],True,[-10,10]],
                0.7:[109,[90,180],[180-71.7,5],False,[-4.5,4.5]],
                  1.0:[100,[90,140],[180-81.5,15],True,[-7,7]],
                 #1.5:[100,[90,150],[180-81.5,5],True,[-7,7]],
                  2.0:[110,[20,150],[180-70,15],True,[-12,12]],
                 2.75:[110,[20,160],[180-70,5],True,[-12,12]],
                  #3.5:[89.4,[72,90],[89.4,6],True],
                  3.5:[86,[24,130],[86,10],True,[-8,5]],
                  4.1:[60.85,[20,80],[57.8,10],True,[-8,8]],
                  5.3:[47,[10,80],[46.5,20],False,[-17,17]],  
             11.8:[35.7,[20,90],[40,10],False,[-10,10]],13.3:[30.8,[20,90],[30,10],False,[-10,10]],14.7:[26.15,[20,90],[30,10],False,[-10,10]],16.2:[24.3,[20,90],[30,10],False,[-10,10]],17.7:[23.1,[20,90],[30,10],False,[-10,10]],19.2:[22.4,[20,90],[30,10],False,[-10,10]],20.6:[22.1,[20,90],[30,10],False,[-10,10]],22.1:[22,[20,90],[30,10],False,[-10,10]],23.6:[22,[20,90],[30,10],False,[-10,10]]},
             'logM':{0:[11.37,[11.,11.7],[11.3,0.4],True,[-0.12,0.12]]},
                   'a':{0:[2.58,[2.,4],[2.6,0.4],True,[-0.28,0.28]]},
             'Mbh':{0:[4.1e8,[2e8,5e8],[4.1e8,100],False,[-5e7,5e7]]},'g_phi0':{0:[120,[0,360],[120,20],False,[110,130]]},'g_i':{0:[20,[20,90],[30,10],False,[10,30]]},
            'ephi':{0:[0.18,[0.01,0.15],[0.08,0.005],False,[0.15,0.21]]},'theta0':{0:[87,[0,90],[72,10],False,[82,95]]},'theta1':{0:[87,[0,90],[72,10],False,[80,100]]},'T':{0:[900,[10,1500],[350,100],False,[700,1000]]},
             'logMvir':{0:[12.5,[12.,13],[12.8,0.5],True,[-0.11,0.11]]},
              'c':{0:[8,[2,18],[8.47,0.1],False,[-1,1]]},
             'disp0':{0:[27,[5,60],[25,8],False,[-6.5,6.5]]}, #[-5]
             'disp1':{0:[21.7,[5,60],[22,5],False,[-4,4]]}, #[-4]
             'dispR':{0:[2.1,[0.2,20],[2,2],False,[-1,1]]}, #[-3]
                   'A':{0:[2,[0,18],[2,1],False,[-1,1]]},
                   'nbeam':{0:[0.3,[0.2,2],[0.5,0.1],False,[-0.1,0.1]]}, #[-2]
                   'dRings':{0:[0.5,[0.2,3],[1,1],False,[-0.3,0.3]]}, #[-1] 
                  'e':{0:[5,[1e-6,500],[5,30],False,[-0.3,0.3]]}
             },
  '6':{'pa':{0.01:[69.85,[10,80],[70.5,15],True,[-10,10]],
                0.3:[72.75,[20,80],[73,8],True,[-8,8]],
                0.7:[72.1,[20,180],[72.1,0.5],False,[-1,1]],
                  1.0:[145,[100,160],[125,20],True,[-15,15]],
                  #1.5:[145,[80,170],[120,20],True,[-15,15]],
                  2.0:[165.3,[140,185],[162,15],True,[-11,11]],
                  #2.5:[165.3,[140,185],[162,10],True,[-11,11]],
                #3.5:[178.2,[165,190],[178.2,6],True],
                  3.5:[176,[160,185],[178,8],True,[-5,5]],#check this
                  4.1:[170,[150,185],[172,10],True,[-9,9]],
                  5.3:[161.4,[130,180],[162,20],False,[-20,20]],  
                 11.8:[155,[151,190],[127,10],False,[-9.4,9.4]],13.3:[149,[70,200],[118,10],False,[-9.4,9.4]],14.7:[142,[70,200],[118,10],False,[-9.4,9.4]],16.2:[135,[70,200],[118,10],False,[-9.4,9.4]],17.7:[127,[70,200],[118,10],False,[-9.4,9.4]],19.2:[120,[70,200],[118,10],False,[-9.4,9.4]],20.6:[116,[70,200],[105,10],False,[-9.4,9.4]],22.1:[112,[70,200],[105,10],False,[-9.4,9.4]],23.6:[109,[70,200],[105,10],False,[-9.4,9.4]]},             
      'i':{0.01:[123.5,[100,170],[180-57,10],True,[-10,10]],
                0.3:[124.6,[100,170],[180-56.3,10],True,[-10,10]],
                0.7:[109,[90,180],[180-71.7,5],False,[-4.5,4.5]],
                  1.0:[100,[90,140],[180-81.5,15],True,[-7,7]],
                 #1.5:[100,[90,150],[180-81.5,5],True,[-7,7]],
                  2.0:[110,[20,150],[180-70,15],True,[-12,12]],
                 #2.5:[110,[20,160],[180-70,5],True,[-12,12]],
                  #3.5:[89.4,[72,90],[89.4,6],True],
                  3.5:[86,[24,130],[86,10],True,[-8,5]],
                  4.1:[60.85,[20,80],[57.8,10],True,[-8,8]],
                  5.3:[47,[10,80],[46.5,20],False,[-17,17]],  
             11.8:[35.7,[20,90],[40,10],False,[-10,10]],13.3:[30.8,[20,90],[30,10],False,[-10,10]],14.7:[26.15,[20,90],[30,10],False,[-10,10]],16.2:[24.3,[20,90],[30,10],False,[-10,10]],17.7:[23.1,[20,90],[30,10],False,[-10,10]],19.2:[22.4,[20,90],[30,10],False,[-10,10]],20.6:[22.1,[20,90],[30,10],False,[-10,10]],22.1:[22,[20,90],[30,10],False,[-10,10]],23.6:[22,[20,90],[30,10],False,[-10,10]]},
             'logM':{0:[11.37,[11.,11.7],[11.3,0.4],True,[-0.12,0.12]]},
                   'a':{0:[2.58,[2.,4],[2.6,0.4],True,[-0.28,0.28]]},
             'Mbh':{0:[4.1e8,[2e8,5e8],[4.1e8,100],False,[-5e7,5e7]]},'g_phi0':{0:[120,[0,360],[120,20],False,[110,130]]},'g_i':{0:[20,[20,90],[30,10],False,[10,30]]},
            'ephi':{0:[0.18,[0.01,0.15],[0.08,0.005],False,[0.15,0.21]]},'theta0':{0:[87,[0,90],[72,10],False,[82,95]]},'theta1':{0:[87,[0,90],[72,10],False,[80,100]]},'T':{0:[900,[10,1500],[350,100],False,[700,1000]]},
             'logMvir':{0:[12.5,[12.,13],[12.8,0.5],True,[-0.11,0.11]]},
              'c':{0:[8,[2,18],[8.47,0.1],False,[-1,1]]},
             'disp0':{0:[27,[5,60],[25,8],False,[-6.5,6.5]]}, #[-5]
             'disp1':{0:[21.7,[5,60],[22,5],False,[-4,4]]}, #[-4]
             'dispR':{0:[2.1,[0.2,20],[2,2],False,[-1,1]]}, #[-3]
                   'A':{0:[2,[0,18],[2,1],False,[-1,1]]},
                   'nbeam':{0:[0.3,[0.2,2],[0.5,0.1],False,[-0.1,0.1]]}, #[-2]
                   'dRings':{0:[0.5,[0.2,3],[1,1],False,[-0.3,0.3]]}, #[-1] 
                  'e':{0:[5,[1e-6,500],[5,30],False,[-0.3,0.3]]}
             },
      '5':{'pa':{0.01:[69.85,[10,80],[70.5,15],True,[-10,10]],
                0.3:[72.75,[20,80],[73,8],True,[-8,8]],
                0.7:[72.1,[20,180],[72.1,0.5],False,[-1,1]],
                  1.0:[145,[100,160],[125,20],True,[-15,15]],
                  #1.5:[145,[80,170],[120,20],True,[-15,15]],
                  2.0:[165.3,[140,185],[162,15],True,[-11,11]],
                  #2.5:[165.3,[140,185],[162,10],True,[-11,11]],
                #3.5:[178.2,[165,190],[178.2,6],True],
                  3.5:[176,[160,185],[178,8],True,[-5,5]],#check this
                  4.1:[170,[150,185],[172,10],False,[-9,9]],
                  5.3:[161.4,[130,180],[162,20],False,[-20,20]],  
                 11.8:[155,[151,190],[127,10],False,[-9.4,9.4]],13.3:[149,[70,200],[118,10],False,[-9.4,9.4]],14.7:[142,[70,200],[118,10],False,[-9.4,9.4]],16.2:[135,[70,200],[118,10],False,[-9.4,9.4]],17.7:[127,[70,200],[118,10],False,[-9.4,9.4]],19.2:[120,[70,200],[118,10],False,[-9.4,9.4]],20.6:[116,[70,200],[105,10],False,[-9.4,9.4]],22.1:[112,[70,200],[105,10],False,[-9.4,9.4]],23.6:[109,[70,200],[105,10],False,[-9.4,9.4]]},             
      'i':{0.01:[123.5,[100,170],[180-57,10],True,[-10,10]],
                0.3:[124.6,[100,170],[180-56.3,10],True,[-10,10]],
                0.7:[109,[90,180],[180-71.7,5],False,[-4.5,4.5]],
                  1.0:[100,[90,140],[180-81.5,15],True,[-7,7]],
                 #1.5:[100,[90,150],[180-81.5,5],True,[-7,7]],
                  2.0:[110,[20,150],[180-70,15],True,[-12,12]],
                 #2.5:[110,[20,160],[180-70,5],True,[-12,12]],
                  #3.5:[89.4,[72,90],[89.4,6],True],
                  3.5:[86,[24,130],[86,10],True,[-8,5]],
                  4.1:[60.85,[20,80],[57.8,10],False,[-8,8]],
                  5.3:[47,[10,80],[46.5,20],False,[-17,17]],  
             11.8:[35.7,[20,90],[40,10],False,[-10,10]],13.3:[30.8,[20,90],[30,10],False,[-10,10]],14.7:[26.15,[20,90],[30,10],False,[-10,10]],16.2:[24.3,[20,90],[30,10],False,[-10,10]],17.7:[23.1,[20,90],[30,10],False,[-10,10]],19.2:[22.4,[20,90],[30,10],False,[-10,10]],20.6:[22.1,[20,90],[30,10],False,[-10,10]],22.1:[22,[20,90],[30,10],False,[-10,10]],23.6:[22,[20,90],[30,10],False,[-10,10]]},
             'logM':{0:[11.37,[11.,11.7],[11.3,0.4],True,[-0.12,0.12]]},
                   'a':{0:[2.58,[2.,4],[2.6,0.4],True,[-0.28,0.28]]},
             'Mbh':{0:[4.1e8,[2e8,5e8],[4.1e8,100],False,[-5e7,5e7]]},'g_phi0':{0:[120,[0,360],[120,20],False,[110,130]]},'g_i':{0:[20,[20,90],[30,10],False,[10,30]]},
            'ephi':{0:[0.18,[0.01,0.15],[0.08,0.005],False,[0.15,0.21]]},'theta0':{0:[87,[0,90],[72,10],False,[82,95]]},'theta1':{0:[87,[0,90],[72,10],False,[80,100]]},'T':{0:[900,[10,1500],[350,100],False,[700,1000]]},
             'logMvir':{0:[12.5,[12.,13],[12.8,0.5],True,[-0.11,0.11]]},
              'c':{0:[8,[2,18],[8.47,0.1],False,[-1,1]]},
             'disp0':{0:[27,[5,60],[25,8],False,[-6.5,6.5]]}, #[-5]
             'disp1':{0:[21.7,[5,60],[22,5],False,[-4,4]]}, #[-4]
             'dispR':{0:[2.1,[0.2,20],[2,2],False,[-1,1]]}, #[-3]
                   'A':{0:[2,[0,18],[2,1],False,[-1,1]]},
                   'nbeam':{0:[0.3,[0.2,2],[0.5,0.1],False,[-0.1,0.1]]}, #[-2]
                   'dRings':{0:[0.5,[0.2,3],[1,1],False,[-0.3,0.3]]}, #[-1] 
                  'e':{0:[5,[1e-6,500],[5,30],False,[-0.3,0.3]]}
             }
    
}

def make_disks_7(params,R_disks,dR_disks,dic):
    pa=dic['pa']
    i=dic['i']#,Rpa,paf,i,Ri,iif,params_dic
    pa[dic['paf']]=params[:7]
    i[dic['iif']]=params[7:14]
    logM=params[14]
    a=params[15]
    logMvir=params[16]
    c=dic['params_dic']['c']
    
    _paRi=pchip(dic['Rpa'],pa)
    _iRi=pchip(dic['Ri'],i)
    
    i_d=np.radians(_iRi(R_disks))
    pa_d=np.radians(_paRi(R_disks))
    v_d=Vcirc(R_disks,dic['params_dic']['Mbh'],logM,a,logMvir, c)
    tanid2=np.tan(np.radians(_iRi(R_disks)))**2
    vsini=v_d*np.sin(i_d)
    
    k0=(1+np.tan(i_d)**2)**0.5#np.sqrt(1+np.tan(i0)**2)
    k1=(R_disks-dR_disks)/(R_disks/k0-dR_disks)
    i1=np.arctan((k1*k1-1)**0.5)#np.arctan(np.sqrt(k1**2-1))

    i1[np.isnan(i1)]=i_d[np.isnan(i1)]
    k2=(R_disks+dR_disks)/(R_disks/k0+dR_disks)
    i2=np.arctan((k2*k2-1)**0.5)#np.arctan(np.sqrt(k2**2-1))
    i2[np.isnan(i2)]=i_d[np.isnan(i2)]
    
    return R_disks,i_d,pa_d,v_d,tanid2,vsini,np.tan(i1)**2,np.tan(i2)**2
def HI_7(x):
    #logM,a,logMvir,c=x[14:18]
    logM=x[14]
    a=x[15]
    logMvir=x[16]
    Vm=Vcirc(Rmf,dic['params_dic']['Mbh'],logM,a,logMvir,dic['params_dic']['c'])
    Vc=Vcirc(Rvcf,dic['params_dic']['Mbh'],logM,a,logMvir,dic['params_dic']['c'])
    return 0.5*np.sum((Vm-Vmf)**2)/40**2+0.5*np.sum((Vc-Vvcf)**2)/40**2#np.max(np.abs(Vm-np.abs(Vm_20[Rm>Rc])))

def conmM_7(x):
    logM=x[14]
    logMvir=x[16]
    #m=10**logM
    #M=10**logMvir
    return logMvir-logM
#######################
def make_disks_6(params,R_disks,dR_disks,dic):
    pa=dic['pa']
    i=dic['i']#,Rpa,paf,i,Ri,iif,params_dic
    pa[dic['paf']]=params[:6]
    i[dic['iif']]=params[6:12]
    logM=params[12]
    a=params[13]
    logMvir=params[14]
    c=dic['params_dic']['c']
    
    _paRi=pchip(dic['Rpa'],pa)
    _iRi=pchip(dic['Ri'],i)
    
    i_d=np.radians(_iRi(R_disks))
    pa_d=np.radians(_paRi(R_disks))
    v_d=Vcirc(R_disks,dic['params_dic']['Mbh'],logM,a,logMvir, c)
    tanid2=np.tan(np.radians(_iRi(R_disks)))**2
    vsini=v_d*np.sin(i_d)
    
    k0=(1+np.tan(i_d)**2)**0.5#np.sqrt(1+np.tan(i0)**2)
    k1=(R_disks-dR_disks)/(R_disks/k0-dR_disks)
    i1=np.arctan((k1*k1-1)**0.5)#np.arctan(np.sqrt(k1**2-1))

    i1[np.isnan(i1)]=i_d[np.isnan(i1)]
    k2=(R_disks+dR_disks)/(R_disks/k0+dR_disks)
    i2=np.arctan((k2*k2-1)**0.5)#np.arctan(np.sqrt(k2**2-1))
    i2[np.isnan(i2)]=i_d[np.isnan(i2)]
    
    return R_disks,i_d,pa_d,v_d,tanid2,vsini,np.tan(i1)**2,np.tan(i2)**2
def HI_6(x):
    #logM,a,logMvir,c=x[14:18]
    logM=x[12]
    a=x[13]
    logMvir=x[14]
    Vm=Vcirc(Rmf,dic['params_dic']['Mbh'],logM,a,logMvir,dic['params_dic']['c'])
    Vc=Vcirc(Rvcf,dic['params_dic']['Mbh'],logM,a,logMvir,dic['params_dic']['c'])
    return 0.5*np.sum((Vm-Vmf)**2)/40**2+0.5*np.sum((Vc-Vvcf)**2)/40**2#np.max(np.abs(Vm-np.abs(Vm_20[Rm>Rc])))

def conmM_6(x):
    logM=x[12]
    logMvir=x[14]
    #m=10**logM
    #M=10**logMvir
    return logMvir-logM


##############
def make_disks_5(params,R_disks,dR_disks,dic):
    pa=dic['pa']
    i=dic['i']#,Rpa,paf,i,Ri,iif,params_dic
    pa[dic['paf']]=params[:5]
    i[dic['iif']]=params[5:10]
    logM=params[10]
    a=params[11]
    logMvir=params[12]
    c=dic['params_dic']['c']
    
    _paRi=pchip(dic['Rpa'],pa)
    _iRi=pchip(dic['Ri'],i)
    
    i_d=np.radians(_iRi(R_disks))
    pa_d=np.radians(_paRi(R_disks))
    v_d=Vcirc(R_disks,dic['params_dic']['Mbh'],logM,a,logMvir, c)
    tanid2=np.tan(np.radians(_iRi(R_disks)))**2
    vsini=v_d*np.sin(i_d)
    
    k0=(1+np.tan(i_d)**2)**0.5#np.sqrt(1+np.tan(i0)**2)
    k1=(R_disks-dR_disks)/(R_disks/k0-dR_disks)
    i1=np.arctan((k1*k1-1)**0.5)#np.arctan(np.sqrt(k1**2-1))

    i1[np.isnan(i1)]=i_d[np.isnan(i1)]
    k2=(R_disks+dR_disks)/(R_disks/k0+dR_disks)
    i2=np.arctan((k2*k2-1)**0.5)#np.arctan(np.sqrt(k2**2-1))
    i2[np.isnan(i2)]=i_d[np.isnan(i2)]
    
    return R_disks,i_d,pa_d,v_d,tanid2,vsini,np.tan(i1)**2,np.tan(i2)**2
def HI_5(x):
    #logM,a,logMvir,c=x[14:18]
    logM=x[10]
    a=x[11]
    logMvir=x[12]
    Vm=Vcirc(Rmf,dic['params_dic']['Mbh'],logM,a,logMvir,dic['params_dic']['c'])
    Vc=Vcirc(Rvcf,dic['params_dic']['Mbh'],logM,a,logMvir,dic['params_dic']['c'])
    return 0.5*np.sum((Vm-Vmf)**2)/40**2+0.5*np.sum((Vc-Vvcf)**2)/40**2#np.max(np.abs(Vm-np.abs(Vm_20[Rm>Rc])))

def conmM_5(x):
    logM=x[10]
    logMvir=x[12]
    return logMvir-logM

fun_set={'7':make_disks_7,'6':make_disks_6,'5':make_disks_5}
constraints_set={'7':[NonlinearConstraint(HI_7,0., 2),NonlinearConstraint(conmM_7,1, 2)],
            '6':[NonlinearConstraint(HI_6,0., 2),NonlinearConstraint(conmM_6,1, 2)],
             '5':[NonlinearConstraint(HI_5,0., 2),NonlinearConstraint(conmM_5,1, 2)]
            }


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--name", help="save file name")
parser.add_argument("--cores", help="number of cores",type=int, required=False,default=1)
parser.add_argument("--steps", help="number of steps",type=int, required=False,default=100)
parser.add_argument("--walkers", help="number x ndim = walkers",type=int, required=False,default=2)
parser.add_argument("--fitter", help="mcmc or diffev",choices=['mcmc','diffev'])
parser.add_argument("--pset", help="parameter set",choices=list(param_set.keys()))
#parser.add_argument("--dm", help="include dark matter (Y/N/YFM (YES and FIT mass)/YFMC (YES and FIT mass and c))",choices=['Y','N','YFM','YFMC'])
# parser.add_argument("--pweight", help="Prior weight", required=False,type=float,default=1)
# parser.add_argument("--fit", help="pvd or cube", required=True,default='pvd')
# parser.add_argument("--dif", help="beams between rings", required=False,type=float,default=0.5)
# parser.add_argument("--dring", help="Distance between rings", required=False,type=float,default=1)
parser.add_argument("--dvsearch", help="dv search", required=False,type=float,default=20)
# parser.add_argument("--sigma", help="sigma cube", required=False,type=float,default=4)
# parser.add_argument("--error2", help="error in flux squared", required=False,type=float,default=0.01)
# parser.add_argument("--HIerror", help="error in HI data points", required=False,type=float,default=40)
parser.add_argument("--fit", help="fit dV or A", choices=['A','dV','AdV'],default='A')
parser.add_argument("--dv0", help="Starting dV for dV fit", required=False,type=float,default=500)
parser.add_argument("--dv", help="dV error for dV fit", required=False,type=float,default=50)
parser.add_argument("--da", help="da error for AdV fit (sigmas)", required=False,type=float,default=50)

args = parser.parse_args()

parameters_=param_set[args.pset]
constraints_=constraints_set[args.pset]
disks_f=fun_set[args.pset]

# params0,parnames,bounds,p0,dp,params_dic,n,pa,Rpa,paf,i,Ri,iif=source_detection.prepare_params(parameters_)

#bounds=np.array(bounds)
dic=source_detection.prepare_params(parameters_)

R_d = np.arange(0.01,4.4,0.06)
dR_d=0.08/2

G=4.304574992e-06
def VNFW(R, M_vir, c):
    return (0.0013329419990568908*M_vir**0.5 *( (np.log(1.+119.73662477388707*R*c/M_vir**0.3333333)-119.73662477388707*R*c/(M_vir**0.3333333*(1.+119.73662477388707*R*c/M_vir**0.3333333)))/(-c/(c+1.) + 
        np.log(c+1.)))**0.5/R**0.5)

def Vcirc(r,Mbh,logM,a,logMvir, c):
    return (G*Mbh/r + (G*10**logM *r)/(r+a)**2 +VNFW(r,10**logMvir,c)**2)**0.5
def m_to_M(m,M,mM0,M1,b,g): return 2*mM0/((M/M1)**-b+(M/M1)**g)
def make_disks(params,R_disks,dR_disks,dic):
    pa=dic['pa']
    i=dic['i']#,Rpa,paf,i,Ri,iif,params_dic
    pa[dic['paf']]=params[:6]
    i[dic['iif']]=params[6:12]
#     logM,a,logMvir,c=params[14:18]
    logM=params[12]
    a=params[13]
    logMvir=params[14]
    c=dic['params_dic']['c']
    
    _paRi=pchip(dic['Rpa'],pa)
    _iRi=pchip(dic['Ri'],i)
    
    i_d=np.radians(_iRi(R_disks))
    pa_d=np.radians(_paRi(R_disks))
    v_d=Vcirc(R_disks,dic['params_dic']['Mbh'],logM,a,logMvir, c)
    tanid2=np.tan(np.radians(_iRi(R_disks)))**2
    vsini=v_d*np.sin(i_d)
    
    k0=(1+np.tan(i_d)**2)**0.5#np.sqrt(1+np.tan(i0)**2)
    k1=(R_disks-dR_disks)/(R_disks/k0-dR_disks)
    i1=np.arctan((k1*k1-1)**0.5)#np.arctan(np.sqrt(k1**2-1))

    i1[np.isnan(i1)]=i_d[np.isnan(i1)]
    k2=(R_disks+dR_disks)/(R_disks/k0+dR_disks)
    i2=np.arctan((k2*k2-1)**0.5)#np.arctan(np.sqrt(k2**2-1))
    i2[np.isnan(i2)]=i_d[np.isnan(i2)]
    
    return R_disks,i_d,pa_d,v_d,tanid2,vsini,np.tan(i1)**2,np.tan(i2)**2

# def conDM(x):
#     #logM,a,logMvir,c=x[14:18]
#     logM=params[12]
#     a=params[13]
#     logMvir=params[14]
#     m=10**logM
#     M=10**logMvir
#     mM=m_to_M(m,M,0.02817,10**11.899,1.068,0.611)
#     return math.log10(m/M)-math.log10(mM)
# nlcDM = NonlinearConstraint(conDM,-0.5, 0.77)

# def conc(x):
#     logM,a,logMvir,c=x[14:18]
#     M=10**logMvir
#     logc_wmap=1.025-0.097*np.log10(M*0.72/(1e12))
#     return c-10**logc_wmap
# nlc = NonlinearConstraint(conc,-2., 2)



def get_RES_A_fast(params,disks_f,R_d,dR_d,dic,epsilon_v):
    e=params[15]
    #prior=np.sum((p0-params)**2/dp**2)
    Rd,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=disks_f(params,R_d,dR_d,dic)
    flag=[]
    for j in range(NN.shape[0]):
        flag.append([False]*int(NN[j]))
#     A=[]
#     for j in range(NN.shape[0]):
#         A.append([None]*int(NN[j]))
#     dV=[]
#     for i in range(NN.shape[0]):
#         dV.append(np.ones(int(NN[i]))*dV0)
    for k,(x,y,r,phi,n,pp) in enumerate(zip(X,Y,R,PHI,NN,PARS)):
        v=np.array(np.split(pp,n))[:,1]
        #A[k]=np.array(np.split(pp,n))[:,0]
        #print(f"{k} point: ({x:.2f},{y:.2f}) | r,phi= ({r:.2f},{np.degrees(phi):.2f}) -> {[f'{vi:.1f}' for vi in v]} km/s")
        for rd,vcd,phi_d,tani2,vsini,tani12,tani22 in zip(R_d,Vc_d,Phi_d,tani2s,vsinis,tanI12,tanI22):
            #print(f"{m} disk: R={rd:.2f} pa={np.degrees(phi_d):.2f} i={np.degrees(i_d):.2f} vc={vcd:.1f} km/s")
#             Rd_sky=rd/(1+np.tan(i_d)**2*np.cos(phi-phi_d)**2)**0.5
            #Rd_sky=rd/(1+tani2*np.cos(phi-phi_d)**2)**0.5
            rell1=(rd-dR_d)/(1+tani12*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i1,R-dR)
            rell2=(rd+dR_d)/(1+tani22*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i2,R+dR)
            #if np.abs(Rd_sky-r)<epsilon_r:
            if (r>=rell1) and (r<=rell2):
                #print(f"A this point disk {m} (R={rd:.2f} pa={np.degrees(phi_d):.2f} i={np.degrees(i_d):.2f} vc={vcd:.1f}) is close: {Rd_sky:.2f} ({Rd_sky-r:.2f})")
                #Vd_sky=vcd*np.sin(i_d)*np.sin(phi-phi_d)/(1 +np.tan(i_d)**2*np.cos(phi-phi_d)**2)**0.5
                Vd_sky=vsini*math.sin(phi-phi_d)/(1 +tani2*math.cos(phi-phi_d)**2)**0.5                
                #dv=np.abs(Vd_sky-v)
                flag[k]=flag[k] | (np.abs(Vd_sky-v)<epsilon_v)
                #dV[k]=np.nanmin([dV[k],dv],axis=0)
    #return np.sum(np.concatenate(A).ravel()[np.concatenate(flag).ravel()]) 
    #like=np.nansum(np.concatenate(dV).ravel()[~np.concatenate(flag).ravel()]**2/delta_v**2)
    resA=np.nansum(II[~np.concatenate(flag).ravel()]**2)
    #print(like,prior,resA)
    return resA/e

def get_RES_A_fast_M(params,epsilon_v):
    
    outofrange=0
    for l in range(len(params)):
        outofrange+=1-(bounds[l,0] <= params[l] <=bounds[l,1])
    if outofrange:# or (param[-4]>param[-5]):
        return -np.inf
    e=params_dic['e']#[15]
    
    prior=-np.sum((p0-params)**2/dp**2)
    Rd,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=make_disks(params,pa,Rpa,paf,i,Ri,iif,R_d,dR_d,params_dic)
    flag=[]
    for j in range(NN.shape[0]):
        flag.append([False]*int(NN[j]))
#     A=[]
#     for j in range(NN.shape[0]):
#         A.append([None]*int(NN[j]))
#     dV=[]
#     for i in range(NN.shape[0]):
#         dV.append(np.ones(int(NN[i]))*dV0)
    for k,(x,y,r,phi,n,pp) in enumerate(zip(X,Y,R,PHI,NN,PARS)):
        v=np.array(np.split(pp,n))[:,1]
        #A[k]=np.array(np.split(pp,n))[:,0]
        #print(f"{k} point: ({x:.2f},{y:.2f}) | r,phi= ({r:.2f},{np.degrees(phi):.2f}) -> {[f'{vi:.1f}' for vi in v]} km/s")
        for rd,vcd,phi_d,tani2,vsini,tani12,tani22 in zip(R_d,Vc_d,Phi_d,tani2s,vsinis,tanI12,tanI22):
            #print(f"{m} disk: R={rd:.2f} pa={np.degrees(phi_d):.2f} i={np.degrees(i_d):.2f} vc={vcd:.1f} km/s")
#             Rd_sky=rd/(1+np.tan(i_d)**2*np.cos(phi-phi_d)**2)**0.5
            #Rd_sky=rd/(1+tani2*np.cos(phi-phi_d)**2)**0.5
            rell1=(rd-dR_d)/(1+tani12*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i1,R-dR)
            rell2=(rd+dR_d)/(1+tani22*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i2,R+dR)
            #if np.abs(Rd_sky-r)<epsilon_r:
            if (r>=rell1) and (r<=rell2):
                #print(f"A this point disk {m} (R={rd:.2f} pa={np.degrees(phi_d):.2f} i={np.degrees(i_d):.2f} vc={vcd:.1f}) is close: {Rd_sky:.2f} ({Rd_sky-r:.2f})")
                #Vd_sky=vcd*np.sin(i_d)*np.sin(phi-phi_d)/(1 +np.tan(i_d)**2*np.cos(phi-phi_d)**2)**0.5
                Vd_sky=vsini*math.sin(phi-phi_d)/(1 +tani2*math.cos(phi-phi_d)**2)**0.5                
                #dv=np.abs(Vd_sky-v)
                flag[k]=flag[k] | (np.abs(Vd_sky-v)<epsilon_v)
                #dV[k]=np.nanmin([dV[k],dv],axis=0)
    #return np.sum(np.concatenate(A).ravel()[np.concatenate(flag).ravel()]) 
    #like=np.nansum(np.concatenate(dV).ravel()[~np.concatenate(flag).ravel()]**2/delta_v**2)
    like=-np.nansum(II[~np.concatenate(flag).ravel()]**2)/e
    #likec=-II[~np.concatenate(flag).ravel()].shape[0]*math.log(e)
    #likec=-II.shape[0]*math.log(6.283185307179586*e)
    logM=params[12]
    a=params[13]
    logMvir=params[14]
    
    Vm=Vcirc(Rmf,params_dic['Mbh'],logM,a,logMvir,params_dic['c'])
    Vc=Vcirc(Rvcf,params_dic['Mbh'],logM,a,logMvir,params_dic['c'])
    likeHI=-np.sum((Vm-Vmf)**2)/1600-np.sum((Vc-Vvcf)**2)/1600 #40^2
    #print(like,prior,resA)
    return like+prior+likeHI

def get_RES_fast_V(params,disks_f,R_d,dR_d,dic,epsilon_v=20,dV0=500,delta_v=60):
    #se=params[18]
    #prior=np.sum((p0-params)**2/dp**2)
    Rd,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=disks_f(params,R_d,dR_d,dic)
#     flag=[]
#     for j in range(NN.shape[0]):
#         flag.append([False]*int(NN[j]))
#     A=[]
#     for i in range(NN.shape[0]):
#         A.append([None]*int(NN[i]))
    dV=[]
    for j in range(NN.shape[0]):
        dV.append(np.ones(int(NN[j]))*dV0)
    for k,(x,y,r,phi,n,pp) in enumerate(zip(X,Y,R,PHI,NN,PARS)):
        v=np.array(np.split(pp,n))[:,1]
        #A[k]=np.array(np.split(pp,n))[:,0]
        for rd,vcd,phi_d,tani2,vsini,tani12,tani22 in zip(R_d,Vc_d,Phi_d,tani2s,vsinis,tanI12,tanI22):
            rell1=(rd-dR_d)/(1+tani12*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i1,R-dR)
            rell2=(rd+dR_d)/(1+tani22*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i2,R+dR)
            if (r>=rell1) and (r<=rell2):
                #print(f"A this point disk {m} (R={rd:.2f} pa={np.degrees(phi_d):.2f} i={np.degrees(i_d):.2f} vc={vcd:.1f}) is close: {Rd_sky:.2f} ({Rd_sky-r:.2f})")
                #Vd_sky=vcd*np.sin(i_d)*np.sin(phi-phi_d)/(1 +np.tan(i_d)**2*np.cos(phi-phi_d)**2)**0.5
                Vd_sky=vsini*math.sin(phi-phi_d)/(1 +tani2*math.cos(phi-phi_d)**2)**0.5                
                dv=np.abs(Vd_sky-v)
                #flag[k]=flag[k] | (dv<epsilon_v)
                dV[k]=np.min([dV[k],dv],axis=0)
    #return np.sum(np.concatenate(A).ravel()[np.concatenate(flag).ravel()]) 
    like=np.sum(np.concatenate(dV)**2)/delta_v**2#.ravel()[~np.concatenate(flag).ravel()]**2/delta_v**2)
    #resA=np.nansum(np.concatenate(A).ravel()[~np.concatenate(flag).ravel()])
    #print(like,prior,resA)
    return like#+prior

def get_RES_fast_AV(params,disks_f,R_d,dR_d,dic,epsilon_v=20,dV0=500,delta_v=60,delta_a=3):
    #se=params[18]
    #prior=np.sum((p0-params)**2/dp**2)
    Rd,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=disks_f(params,R_d,dR_d,dic)
    flag=[]
    for j in range(NN.shape[0]):
        flag.append([False]*int(NN[j]))
#     A=[]
#     for i in range(NN.shape[0]):
#         A.append([None]*int(NN[i]))
    dV=[]
    for j in range(NN.shape[0]):
        dV.append(np.ones(int(NN[j]))*dV0)
    for k,(x,y,r,phi,n,pp) in enumerate(zip(X,Y,R,PHI,NN,PARS)):
        v=np.array(np.split(pp,n))[:,1]
        #A[k]=np.array(np.split(pp,n))[:,0]
        for rd,vcd,phi_d,tani2,vsini,tani12,tani22 in zip(R_d,Vc_d,Phi_d,tani2s,vsinis,tanI12,tanI22):
            rell1=(rd-dR_d)/(1+tani12*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i1,R-dR)
            rell2=(rd+dR_d)/(1+tani22*math.cos(phi-phi_d)**2)**0.5#self.rR(phi,phi0,i2,R+dR)
            if (r>=rell1) and (r<=rell2):
                #print(f"A this point disk {m} (R={rd:.2f} pa={np.degrees(phi_d):.2f} i={np.degrees(i_d):.2f} vc={vcd:.1f}) is close: {Rd_sky:.2f} ({Rd_sky-r:.2f})")
                #Vd_sky=vcd*np.sin(i_d)*np.sin(phi-phi_d)/(1 +np.tan(i_d)**2*np.cos(phi-phi_d)**2)**0.5
                Vd_sky=vsini*math.sin(phi-phi_d)/(1 +tani2*math.cos(phi-phi_d)**2)**0.5                
                dv=np.abs(Vd_sky-v)
                flag[k]=flag[k] | (dv<epsilon_v)
                dV[k]=np.nanmin([dV[k],dv],axis=0)
    #return np.sum(np.concatenate(A).ravel()[np.concatenate(flag).ravel()]) 
    F=~np.concatenate(flag).ravel()
    like=np.nansum(np.concatenate(dV).ravel()[F]**2*AAs[F]/(delta_a*delta_v**2))
    #resA=np.nansum(np.concatenate(A).ravel()[~np.concatenate(flag).ravel()])
    #print(like,prior,resA)
    return like#+prior

import time

start = time.process_time()
#A=differential_evolution(func=a.logposterior_julia_min,bounds=np.array(bounds),maxiter=int(args.i),tol=float(args.tol))
#disks_f=make_disks
dic=dic
R_d=R_d
dR_d=dR_d
bounds=dic['bounds']
if args.fit == 'A':
    if args.fitter == 'diffev':
        print('===Starting diffev===')
        sol=differential_evolution(get_RES_A_fast,bounds,args=(disks_f,R_d,dR_d,dic,args.dvsearch),tol=0.1,
                                   workers=int(args.cores),constraints=constraints_)
        print('==== Diffev Finished in {:.2f} m ===='.format((time.process_time() - start)/60))
        print(sol)
    else:
        walkers=n*args.walkers
        print('# ==== MCMC Posterior Sampling ====')
        print(f'## Setup starting positions for {walkers} walkers')
        pos = [np.random.normal(params0,0.01*np.array(params0)) for i in range(walkers)]
        #print(pos)
        #print(pos.shape)
        kwargs={'epsilon_v':args.dvsearch}#'bounds':self.bounds,'RR':self.RR,'cube':,'ixx':ixx,'xx_sky':xx_sky,'iyy':iyy,'yy_sky':yy_sky,'zz':zz}
        if args.name is not None:
            filename=args.name
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(walkers, n)
        else:
            backend=None
        with Pool(int(args.cores)) as pool:
            sampler = emcee.EnsembleSampler(walkers, n, get_RES_A_fast_M,backend=backend,kwargs=kwargs,pool=pool,
            moves=[(emcee.moves.DEMove(),0.05),(emcee.moves.DESnookerMove(),0.1),(emcee.moves.StretchMove(),0.85),])
            result=sampler.run_mcmc(pos, args.steps,progress=True)
        #a.samples = sampler.chain[:, 0:, :].reshape((-1, a.ndim))
elif args.fit == 'dV':
    print('===Starting diffev===')
    sol=differential_evolution(get_RES_fast_V,bounds,args=(disks_f,R_d,dR_d,dic,args.dvsearch,args.dv0,args.dv),tol=0.1,
                               workers=int(args.cores),constraints=constraints_)
    
    print('==== Diffev Finished in {:.2f} m ===='.format((time.process_time() - start)/60))
    print(sol)
    for pn,p in zip(parnames,sol.x): print(f"{pn}: {p:.3f}")
elif args.fit == 'AdV':
    print('===Starting diffev===')
    sol=differential_evolution(get_RES_fast_AV,bounds,args=(disks_f,R_d,dR_d,dic,args.dvsearch,args.dv0,args.dv,args.da),tol=0.1,
                               workers=int(args.cores),constraints=constraints_)
    
    print('==== Diffev Finished in {:.2f} m ===='.format((time.process_time() - start)/60))
    print(sol)
    for pn,p in zip(parnames,sol.x): print(f"{pn}: {p:.3f}")

# print('===Results===')
# np.save(args.name,sol.x)