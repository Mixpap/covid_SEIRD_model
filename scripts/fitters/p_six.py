import numpy as np
import math
from scipy.interpolate import pchip
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u

dic={'pa':{0.01:[69.85,[40,80],[70.5,15],True,[-10,10]],
                0.3:[72.75,[40,80],[73,8],True,[-8,8]],
                0.57:[72.1,[20,180],[72.1,0.5],False,[-1,1]],
                  1.0:[145,[100,160],[125,20],True,[-15,15]],
                  #1.5:[145,[80,170],[120,20],True,[-15,15]],
                  2.0:[165.3,[140,185],[162,15],True,[-11,11]],
                  #2.5:[165.3,[140,185],[162,10],True,[-11,11]],
                #3.5:[178.2,[165,190],[178.2,6],True],
                  3.5:[176,[170,185],[178,8],True,[-5,5]],#check this
                  4.1:[170,[160,185],[172,10],True,[-9,9]],
                  5.3:[161.4,[130,180],[162,20],False,[-20,20]],  
                 11.8:[155,[151,190],[127,10],False,[-9.4,9.4]],13.3:[149,[70,200],[118,10],False,[-9.4,9.4]],14.7:[142,[70,200],[118,10],False,[-9.4,9.4]],16.2:[135,[70,200],[118,10],False,[-9.4,9.4]],17.7:[127,[70,200],[118,10],False,[-9.4,9.4]],19.2:[120,[70,200],[118,10],False,[-9.4,9.4]],20.6:[116,[70,200],[105,10],False,[-9.4,9.4]],22.1:[112,[70,200],[105,10],False,[-9.4,9.4]],23.6:[109,[70,200],[105,10],False,[-9.4,9.4]]},             
      'i':{0.01:[123.5,[100,150],[180-57,10],True,[-10,10]],
                0.3:[124.6,[100,150],[180-56.3,10],True,[-10,10]],
                0.57:[109,[90,180],[180-71.7,5],False,[-4.5,4.5]],
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
                  'e':{0:[5,[1e-6,500],[5,1],False,[-0.3,0.3]]}
             }
G=4.304574992e-06
def VNFW(R, M_vir, c):
    return (0.0013329419990568908*M_vir**0.5 *( (np.log(1.+119.73662477388707*R*c/M_vir**0.3333333)-119.73662477388707*R*c/(M_vir**0.3333333*(1.+119.73662477388707*R*c/M_vir**0.3333333)))/(-c/(c+1.) + 
        np.log(c+1.)))**0.5/R**0.5)

def Vcirc(r,Mbh,logM,a,logMvir, c):
    return (G*Mbh/r + (G*10**logM *r)/(r+a)**2 +VNFW(r,10**logMvir,c)**2)**0.5

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


#import os
#print(os.getcwd())
datafolder='../data/NGC6328/other/'

macc_HI_Vrot=np.loadtxt(datafolder+'macc_HI_Vrot.csv',delimiter=',')
vc_HI_Vrot=np.loadtxt(datafolder+'vc_HI_Vrot.csv',delimiter=',')
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

def HI(x):
    #logM,a,logMvir,c=x[14:18]
    logM=x[12]
    a=x[13]
    logMvir=x[14]
    Vm=Vcirc(Rmf,4.1e8,logM,a,logMvir,8)
    Vc=Vcirc(Rvcf,4.1e8,logM,a,logMvir,8)
    return 0.5*np.sum((Vm-Vmf)**2)/40**2+0.5*np.sum((Vc-Vvcf)**2)/40**2#np.max(np.abs(Vm-np.abs(Vm_20[Rm>Rc])))

def conmM(x):
    logM=x[12]
    logMvir=x[14]
    #m=10**logM
    #M=10**logMvir
    return logMvir-logM
