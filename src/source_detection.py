from astropy.stats import mad_std,sigma_clipped_stats
import numpy as np
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
from scipy.optimize import minimize,differential_evolution,basinhopping,dual_annealing,NonlinearConstraint

def gaussians(x,N,p):
    s=np.zeros(x.shape)
    pars=np.split(np.array(p),N)
    #print(pars)
    for A,x0,dx in pars:
        s=s+A*np.exp(-(x-x0)**2/(2*dx**2))
    return s

def L(pars,N,vdata,sdata,error):
    #n=vdata.size
    #k=len(pars)
    model=gaussians(vdata,N,pars)
    logL=np.nansum((model-sdata)**2/(2*error**2))+np.log(error*np.sqrt(2*np.pi))
    #bic=k*np.log(n)+2*n*chi2
    return logL

def residual_positions_algorithm(x,y,cube,crit_sigma=3):
    
    xx=cube.x.data
    ixx=cube.attrs['ixx']
    yy=cube.y.data
    iyy=cube.attrs['iyy']
    vv=cube.v.data
    
    ix=np.argsort(np.abs(xx-x))[0]
    iy=np.argsort(np.abs(yy-y))[0]

    print(ix,xx[ix],iy,yy[iy])
    figsize=(23,7) #figure size
    mosaic= [['im','s']]
    gridspec_kw={'width_ratios': [1,1],'height_ratios': [1]}
    fig = plt.figure(constrained_layout=False,figsize=figsize)
    ax = fig.subplot_mosaic(mosaic,gridspec_kw=gridspec_kw)

    divider = make_axes_locatable(ax['im'])
    cax0 = divider.append_axes("top", size="2.5%", pad=0.05)
    cm=ax['im'].pcolormesh(xx,yy,np.nansum(cube['cube'].data,axis=0))#,origin='lower')
    plt.colorbar(cm,cax=cax0,orientation='horizontal',fraction=0.046,pad=0.04).ax.tick_params(labelsize=9)
    cax0.xaxis.set_ticks_position("top")
    ax['im'].grid(True,which='both',linewidth=0.5)
    klos=0.5
    ax['im'].set(aspect=1,xlabel='X',ylabel='Y',xlim=[x-klos,x+klos],ylim=[y-klos,y+klos])#,title=r'Image {}'.format(index))
    ax['im'].plot(x,y,'o',color='red')

    ss=cube['cube'].data[:,iy,ix]
    
    ax['s'].plot(vv,ss,'--',label='data')
    
    rms=max(mad_std(ss[(vv>350)|(vv<-350)]),cube.attrs['rms'])

    msk=ss>crit_sigma*rms
    print(np.sum(msk))
    ax['s'].plot(vv,ss,label='data')
    
    parst=[[crit_sigma*rms,0.01],[-360,360],[21,30]]
    p0=[[3*crit_sigma*rms],[-260],[30]]
    ax['s'].axhline(rms,linestyle='--')
    ax['s'].axhline(crit_sigma*rms,linestyle='--')

    N=1
    minbic=1e7
    bic=minbic-11
    
#     def anti_con(x):
#         vs = x[1::3]
#         return np.min(np.diff(np.sort(vs),append=1000))
#     nlc_ac = NonlinearConstraint(anti_con,35, 1000)
    
    while ((bic-minbic)<-10) and (N<=8):
        pars=parst*N
        p0s=p0*N

        
        bpars=sol.x if N>1 else 0
        minbic=bic
        bN=N-1
        
        sol=differential_evolution(L,pars,args=(N,vv,ss,rms),maxiter=500,
                                   tol=0.1,popsize=10)#,constraints=nlc_ac)
        #sol=minimize(L,x0=np.array(p0s).flatten(),bounds=pars,args=(N,vv,ss,rms),tol=0.1,constraints=nlc_ac)
        logL=sol.fun
        n=vv.size
        k=len(pars)
        bic=k*np.log(n)+2*n*logL

        print(f" bic ({N}) {bic:.2f} - bic ({N-1}) {minbic:.2f}  = {bic-minbic:.2f}")
        N=N+1
    
        #ax['s'].plot(aclas.data[data_name]['vv'],gaussians(aclas.data[data_name]['vv'],N,sol.x),label=f'{N} model (chi:{logL:.1f} | bic:{bic:.1f})',linewidth=0.5)
    vvv=np.linspace(-420,420,800)
    ax['s'].plot(vvv,gaussians(vvv,bN,bpars),label=f'{bN} -best- model (bic:{minbic:.1f})',linewidth=3)
    
    ax['s'].plot(vv,ss-gaussians(vv,bN,bpars),label='residuals',linewidth=2)
    ibpars=np.split(np.array(bpars),bN)
    for A,x0,dx in ibpars:
        print(f'Best params A = {A:.5f} | v0 = {x0:.2f} | dv = {dx:.2f}')
        ax['s'].plot(vvv,A*np.exp(-(vvv-x0)**2/(2*dx**2)),'--')

    ax['s'].legend()
    
    
def source_find(cube,crit_sigma=4,critn=3,maxn=30):
    NN=np.array([])
    PARS=[]
    BIC=np.array([])

    xx=cube.x
    ixx=cube.attrs['ixx']
    yy=cube.y
    iyy=cube.attrs['iyy']
    vv=cube.v

    n=vv.size

    parst=[[crit_sigma*cube.attrs['rms'],0.01],[-360,360],[21,30]]
    nn=0
    for ix,x,iy,y in zip(tqdm.tqdm(ixx),xx[ixx],iyy,yy[iyy]):
        #print(ix,iy)
        ss=cube['cube'][:,iy,ix]
        rms=np.max([mad_std(ss[(vv>350)|(vv<-350)]),cube.attrs['rms']])
        msk=ss>crit_sigma*rms
        #print(ix,iy,x,y,np.sum(msk))
        if np.sum(msk)>critn:
            
            N=1
            minbic=1e7
            bic=minbic-11

            while ((bic-minbic)<-10) and (N<=8):
                pars=parst*N
                bpars=sol.x if N>1 else None
                minbic=bic
                bN=N-1

                sol=differential_evolution(L,pars,args=(N,vv,ss,rms),maxiter=500,tol=0.1,popsize=10)
                logL=sol.fun

                k=len(pars)
                bic=k*np.log(n)+2*n*logL
                #print(f" bic ({N}) {bic:.2f} - bic ({N-1}) {minbic:.2f}  = {bic-minbic:.2f}")
                N=N+1
            NN=np.append(NN,bN)
            PARS.append(bpars)
            BIC=np.append(BIC,minbic)
        else:
            NN=np.append(NN,0)
            PARS.append([0])
            BIC=np.append(BIC,-1)
        nn+=1
        if nn>maxn: break
    return NN,PARS,BIC


def prepare_params(dicparams):
    params=[]
    params_dic={}
    p0=[]
    dp=[]
    bounds=[]
    n=0
    #slices={}
    Rpa=np.array([])
    Ri=np.array([])
    minpa=100
    maxpa=0
    mini=100
    maxi=0
    
    paf=np.array([],dtype=bool)
    pa0=np.array([])
    iif=np.array([],dtype=bool)
    i0=np.array([])
    parnames=[]
    for p in dicparams:
        k0=n
        for i in dicparams[p]:
            if i>0:
                params_dic.update({p+'_'+str(i):dicparams[p][i][0]})
            else:
                params_dic.update({p:dicparams[p][i][0]})
#             if p == 'pa':
#                 Rpa=np.append(Rpa,i)
            if p == 'pa':
                pa0=np.append(pa0,dicparams[p][i][0])
                paf=np.append(paf,dicparams[p][i][3])
                Rpa=np.append(Rpa,i)
            elif p == 'i':
                i0=np.append(i0,dicparams[p][i][0])
                iif=np.append(iif,dicparams[p][i][3])
                Ri=np.append(Ri,i)     
            if dicparams[p][i][3]:
                if p == 'pa':
                    if n<minpa: minpa=n
                    if n>maxpa: maxpa=n
                elif p == 'i':
                    if n<mini: mini=n
                    if n>maxi: maxi=n
                else:
                    print(f"Parameter {p} has index {n}")
                if i>0:
                    parnames.append(p+'_'+str(i))
                else:
                    parnames.append(p)
                params.append(dicparams[p][i][0])
                bounds.append(dicparams[p][i][1])
                p0.append(dicparams[p][i][2][0])
                dp.append(dicparams[p][i][2][1])
                n+=1
        #slices.update({p:slice(k0,n)})
    print(f"PA parameters have indexes [{minpa}:{maxpa+1}]")
    print(f"Inc parameters have indexes [{mini}:{maxi+1}]")
    print(f'Total Parameters we are going to fit {n}')
    return {'params0':params,'param_names':parnames,'bounds':np.array(bounds),'p0':np.array(p0),
          'dp':np.array(dp),'params_dic':params_dic,'n':n,'pa':pa0,'Rpa':Rpa,'i':i0,'Ri':Ri,
          'paf':paf,'iif':iif}
    #return params,parnames,bounds,np.array(p0),np.array(dp),params_dic,n,pa0,Rpa,paf,i0,Ri,iif