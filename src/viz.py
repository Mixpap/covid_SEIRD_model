from lmfit import Model
import numpy as np
from scipy import ndimage
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
import astropy.units as u
from astropy.stats import mad_std,sigma_clipped_stats
from astropy import constants as const

def results(params,disks_f,R_d,dR_d,dic,galaxy,fix_arccos=0,warp_fit=None,bounds=None,params0=None):
    R_d,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=disks_f(params,R_d,dR_d,dic)#Rpa,Ri,R_d,dR_d,params_dic)
    #pa,Rpa,paf,i,Ri,iif,R_d,dR_d,params_dic=disks_f_args
    gphi0=np.radians(galaxy['pa'])
    gi=np.radians(galaxy['i'])
    gR=galaxy['R']

    Wg=np.array([np.sin(gi)*np.cos(gphi0),
               np.sin(gi)*np.sin(gphi0),
               np.cos(gi)])

    Ig=np.array([-np.sin(gphi0),np.cos(gphi0),0])#*gR
    Mg=np.array([-np.cos(gi)*np.cos(gphi0),-np.cos(gi)*np.sin(gphi0),np.sin(gi)])#*gR
    Ng=Wg/np.linalg.norm(Wg)#*gR

    TILT=np.array([])
    LON=np.array([])
    RR=np.array([])
    WW=np.array([])
    

    # def dot(a, b):
    #     return np.sum(a * b, axis=-1)

    # def mag(a):
    #     return np.sqrt(np.sum(a*a, axis=-1))

    # def angle(a, b):
    #     cosab = dot(a, b) / (mag(a) * mag(b)) # cosine of angle between vectors
    #     angle = np.arccos(cosab) # what you currently have (absolute angle)

    #     b_t = -b#[:,[1,0]] * [1, -1] # perpendicular of b

    #     is_cc = dot(a, b_t) < 0

    #     # invert the angles for counter-clockwise rotations
    #     print(is_cc)
    #     if is_cc:
    #         return  2*np.pi - angle
    #     else:
    #         return angle
        # angle[is_cc] = 2*np.pi - angle[is_cc]
        # return angle
    #testt=np.array([])
    for m,(rd,vcd,i_d,phi_d,tani2,vsini,tani12,tani22) in enumerate(zip(R_d,Vc_d,I_d,Phi_d,tani2s,vsinis,tanI12,tanI22)):

        Wd=np.array([np.sin(i_d)*np.cos(phi_d),
               np.sin(i_d)*np.sin(phi_d),
               np.cos(i_d)])

        Id=np.array([-np.sin(phi_d),np.cos(phi_d),0])#*dR
        Md=np.array([-np.cos(i_d)*np.cos(phi_d),-np.cos(i_d)*np.sin(phi_d),np.sin(i_d)])#*dR
        Nd=Wd/np.linalg.norm(Wd)#*dR

        Pdg=np.cross(Nd,Ng)#project_on_plane(Md,Ng)

        #if 
        anglePI = np.degrees(np.arccos(np.dot(Pdg,Ig)/(np.linalg.norm(Pdg)*np.linalg.norm(Ig))))
        #test=np.degrees(np.arctan2(np.dot(Pdg,Ig)/(np.linalg.norm(Pdg)*np.linalg.norm(Ig))))
        tilt=np.degrees(np.arccos(np.dot(Nd,Ng)/(np.linalg.norm(Nd)*np.linalg.norm(Ng))))

        RR=np.append(RR,rd)
        WW=np.append(WW,vcd/rd)
        TILT =np.append(TILT,tilt)
        LON =np.append(LON,anglePI)
        #testt=np.append(testt,test)


    figsize=(23,20) #figure size
    mosaic= [['i','pa'],['lon','.']]
    gridspec_kw={'width_ratios': [1,1],'height_ratios': [1,1]}
    fig = plt.figure(constrained_layout=False,figsize=figsize)
    ax = fig.subplot_mosaic(mosaic,gridspec_kw=gridspec_kw)
    
    ax['i'].plot(R_d,np.degrees(I_d),label='LON precess angle $\phi$')
    
    ax['pa'].plot(R_d,np.degrees(Phi_d),label='LON precess angle $\phi$')
    ax['lon'].plot(RR,LON,'--',label='LON precess angle $\phi$')
    
    #ax['lon'].plot(RR,50*np.sign(np.cos(np.radians(LON))),'--',label='LON precess angle $\phi$')
    
    if fix_arccos>0:
        LON=np.where(RR<fix_arccos,LON,360-LON)
    
    # if fix_arccos:
    #     d2=np.diff(LON,2)
    #     sd=np.where(d2< -1)[0]
    #     print(sd)
    #     # ax['lon'].axvline(RR[sd])
    #     d=1
    #     LON[sd[0]+d:]=2*LON[sd[0]+d]+LON[sd[0]+d:]*-1
    #     for i in range(1,len(sd)):
    #         if sd[i]-sd[i-1]>2:
    #             LON[sd[i]+d:]=2*LON[sd[i]+d]+LON[sd[i]+d:]*-1
    #     LON=LON-LON[~np.isnan(LON)][-1]
    ax['lon'].plot(RR,LON,'-',label='LON precess angle $\phi$')
    #ax['lon'].plot(RR,LON,label='LON precess angle $\phi$')
    
    Rpa=dic['Rpa']
    Ri=dic['Ri']
    paf=dic['paf']
    pa=dic['pa']
    iif=dic['iif']
    i=dic['i']
    
    if bounds is not None:
        for r,b in zip(Rpa[paf],bounds[:len(pa[paf])]):
            ax['pa'].vlines(r,b[0],b[1])
        for r,b in zip(Ri[iif],bounds[len(pa[paf]):len(pa[paf])+len(i[iif])]):
            ax['i'].vlines(r,b[0],b[1])
    if params0 is not None:
        for r,b in zip(Rpa[paf],params0[:len(pa[paf])]):
            ax['pa'].plot(r,b,'o')
        for r,b in zip(Ri[iif],params0[len(pa[paf]):len(pa[paf])+len(i[iif])]):
            ax['i'].plot(r,b,'o')
        
    if warp_fit is not None:
                        
        method= warp_fit['method'] if 'method' in warp_fit.keys() else 'nelder'
        R1= warp_fit['R1'] if 'R1' in warp_fit.keys() else 0
        R2= warp_fit['R2'] if 'R2' in warp_fit.keys() else 100

        mask = (RR>R1) & (RR<R2)
        WW_myr=(WW[mask]*u.km*u.rad/(u.kpc*u.s)).to(u.deg*u.Myr**-1)
        def _warp(R,ephi,theta0,theta1,T):
            return theta1-ephi*np.cos(np.deg2rad(theta0))*WW_myr.value*T
        wmod=Model(_warp)
        wmod.param_hints=warp_fit['param_hints']
        wmod.make_params()

        wfit=wmod.fit(R=RR[mask], data=LON[mask], nan_policy='omit', method=method,weights=5*np.ones(LON[mask].shape))
        print(wfit.fit_report())
        best_params=wfit.best_values
        ax['lon'].plot(RR[mask],_warp(RR[mask],**best_params),label='warp-fit')
    for axi in ['lon','i','pa']:
        ax[axi].set(xlabel='R [kpc]')
        ax[axi].grid('both')
    #returndic.update({'R':RR,'LON':LON})
    
def resultspvd_bay(cube_total,params,disks_f,R_d,dR_d,dic,X,Y,V,A,papvds,
                   slit=0.1,N=10,lwd=3,lwd_n=1,alpha=0.1,levels=np.arange(2,10,1),
                   xlim=[-5,5],vlim=[-380,380],figsize=(20,10),
                   dxticks=0.5,dvticks=100,save=None,sn=0.000001):
    
    ithick = int(round(slit/cube_total.attrs['dx']))
    ixc = round(cube_total.dims['x']/2)
    y=cube_total.y.data
    v=cube_total.v.data
    Npas=len(papvds)
    R_d,I_d,Phi_d,Vc_d,tani2s,vsinis,tanI12,tanI22=disks_f(params,R_d,dR_d,dic)#,Rpa,Ri,R_d2,dR_d,params_dic)
    figsize=(15*(xlim[1]-xlim[0])/10,2.*Npas*(vlim[1]-vlim[0])/350) if figsize == 'auto' else figsize
    fig,ax=plt.subplots(nrows=Npas,figsize=figsize)#(20,6*len(papvds)))
    for j,papvd in enumerate(papvds):
        rotcube=ndimage.interpolation.rotate(cube_total['cube'].fillna(0).data,papvd, axes=(1, 2), reshape=False)
        cubeslice = np.nansum(rotcube[:,:,ixc-ithick:ixc+ithick],axis=2)
        rotmadmap=ndimage.interpolation.rotate(cube_total['madcube'].fillna(0).data, papvd, axes=(1, 2), reshape=False)
        pvdmadmap=np.nansum(rotmadmap[:,:,ixc-ithick:ixc+ithick],axis=2)
        #pvdmadmap=np.sqrt(np.nansum(rotmadmap[:,:,ixc-ithick:ixc+ithick]**2,axis=2))
        pvd_cube=cubeslice/pvdmadmap;  #pvdlvs[0]
        pvd_cube=np.where(pvd_cube>=0,pvd_cube,np.nan)
        ax[j].contourf(y,v,pvd_cube,levels=levels,alpha=0.4,cmap='viridis')
        
        phi=np.radians(papvd+90)
        r_model_m=[]
        v_model_m=[]
        r_model=[]
        v_model=[]
        #lwd= 1#lw[p] if p in lw.keys() else 2.
        for rd,vcd,phi_d,tani2,vsini,tani12,tani22 in zip(R_d,Vc_d,Phi_d,tani2s,vsinis,tanI12,tanI22):
            
            Rd_sky=rd/(1+tani2*np.cos(phi-phi_d)**2)**0.5
            Vd_sky=vsini*math.sin(phi-phi_d)/(1 +tani2*math.cos(phi-phi_d)**2)**0.5      
            
#             PP=ax[j].plot(-Rd_sky,-Vd_sky,'o',alpha=1,label='Mean Posterior')#,color='black')#,label=p)
#             ax[j].plot(Rd_sky,Vd_sky,'o',alpha=1,c=PP[0].get_c())
            
            r_model_m = np.append(r_model_m,-Rd_sky)
            r_model = np.append(r_model,Rd_sky)
            v_model_m = np.append(v_model_m,-Vd_sky)
            v_model = np.append(v_model,Vd_sky)
#                 ax[j].plot(r_model[np.argsort(r_model)],v_model[np.argsort(r_model)],'o',markersize=3.5,alpha=0.7,label=pard)

        PP=ax[j].plot(r_model_m,v_model_m,'-',linewidth=lwd,alpha=1,label='Mean Posterior')#,color='black')#,label=p)
        ax[j].plot(r_model,v_model,'-',linewidth=lwd,alpha=1,c=PP[0].get_c())
        
        cospa=np.cos(phi)
        sinpa=np.sin(phi)
        tanpa=np.tan(phi)
        Yslit_l=slit/(2*cospa)+X*tanpa
        Yslit_u=-slit/(2*cospa)+X*tanpa
        
        XX_s=X[(Yslit_l<Y)&(Y<Yslit_u)]
        YY_s=Y[(Yslit_l<Y)&(Y<Yslit_u)]
        VV_s=V[(Yslit_l<Y)&(Y<Yslit_u)]
        AA_s=A[(Yslit_l<Y)&(Y<Yslit_u)]
        #F_s=F[(Yslit_l<Y)&(Y<Yslit_u)]
        
        PHI_s=np.arctan2(YY_s,XX_s)
        RR_s=np.sqrt(XX_s**2+YY_s**2)
        
        norm = mpl.colors.Normalize(vmin=-3, vmax=-2)
        smap = mpl.cm.ScalarMappable(norm=norm, cmap='magma')
        Ca_res=smap.to_rgba(np.log10(AA_s))

        ax[j].scatter(RR_s*np.sign(-np.cos(PHI_s)),VV_s,marker='o',c='black')#AA_s,norm=LogNorm(),alpha=0.2)
        #ax[j].scatter((RR_s*np.sign(-np.cos(PHI_s)))[#],VV_s[#],marker='.',c=Ca_res[#],alpha=0.2)

        ax[j].xaxis.set_ticks(np.arange(-8,8+dxticks,dxticks))
        ax[j].yaxis.set_ticks(np.arange(-600,600,dvticks))
        ax[j].set(xlim=xlim,ylim=vlim,xlabel=r'Projected Axis Radius [kpc] (PA: ${}^\circ$)'.format(papvd),ylabel=r'Projected Velocity [km s$^{-1}$]')
        ax[j].grid(True,which='both')
        ax[j].legend(loc=2)
        