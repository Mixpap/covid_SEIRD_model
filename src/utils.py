import numpy as np
# import sympy as sm
import emcee
import corner
import time
from multiprocessing import Pool
import h5py
from scipy.interpolate import interp1d,pchip
import astropy.units as u
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.stats import mad_std,sigma_clipped_stats
from astropy import constants as const
from astropy.cosmology import LambdaCDM
from astropy.table import Column,Table

from scipy.integrate import quad,cumtrapz
import xarray as xr
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
from random import shuffle
from lmfit import Model
from scipy.signal import find_peaks
from photutils import EllipticalAperture, EllipticalAnnulus,aperture_photometry,detection,CircularAperture,DAOStarFinder,IRAFStarFinder
from astropy.cosmology import WMAP9 as cosmo



def load_fits_cube(cube_file=None,ucube_file=None,pbfile=None,z=0,sigma=1,area_mask=None,center='center',voffset=0.,zaxis='vel',n=2,restFreq=230.5380e+9,nosignalregions = [{'x0':0,'y0':0,'dx':100,'dy':100}],drop=None,debug=False,beam=None):
    """Load a fits cube file and stores it internally in class.
    The user must provide 2/3 cubes (pb corrected and uncorrected, pb corrected and pb correction, uncorrected and pb correction).

    Args:
        name ([str]): Internal name for the cube (ie co21)
        cube_file ([str], optional): Filename of the pb-corrected cube. Defaults to None.
        ucube_file ([str], optional): Filename of the pb-uncorrected cube. Defaults to None.
        pbfile ([str], optional): Filename of the pb-correction cube. Defaults to None.
        sigma (float, optional): Sigma masking value. The mask is (2d-area) means all channels in area are preserved.  Defaults to 1.
        area_mask ([dic], optional): Mask a 3d area with a 3dellipsoid using a dic of dic. Defaults to None. the dictionaries are in form:
            {'tag for the area(unique)':{'x0':center in x, 'Rx': radius in x,'y0':center in y, 'Ry': radius in y, 'v0':center in v, 'Rv': radius in v, 's':'del' to remove this data, 'add' to add this data}}
        center ([str or dic], optional): if 'center' cube is centered on Y/2, X/2, else moves the cube by padding
        voffset ([float], optional): Voffset NOT IMPLEMENTED Defaults to 0..
        zaxis (str, optional): Z axis, velocities ('vel') or frequencies ('freq'). Defaults to 'vel'.
        n (int, optional): Molecular level, (for CO). Defaults to 2.
        restFreq ([float], optional): Rest frequency of the line in case zaxis='freq'. Defaults to 230.5380e+9.
        nosignalregions (list, optional): list of dics, with the areas for calculating the noise. Defaults to [{'x0':0,'y0':0,'dx':100,'dy':100}].
        drop ([dic], optional): Drop data for memory savings, ie drop={'x':2,'y':5,'v':1000} to keep [[-2,2],[-5,5],[-1000,1000]]]. Defaults to None.
        debug (bool, optional): Show debugging maps. Defaults to False.
        beam ([list], optional): In case header does not provide beam, ie drop={'x':2.8,'y':5,'v':1000}. Defaults to None.
        fit (bool, optional): If the cube will be used on 3d fitting. Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
    """        
    if (pbfile is None) and (ucube_file is not None) and (cube_file is not None):
        hduCO = fits.open(cube_file)[0]
        hdr_cube, cube = hduCO.header, hduCO.data
        hduUCO = fits.open(ucube_file)[0]
        hdrUCO, ucube = hduUCO.header, hduUCO.data

        if cube.shape[0]==1:
            cube=cube[0]
        if ucube.shape[0]==1:
            ucube=ucube[0]
        print(f'Original Data Corrected Cube shape {cube.shape}')
        print(f'Original Data UnCorrected Cube shape {ucube.shape}')
        if center != 'center':
            #pivot=[center['y'],center['x']]
            padX = [cube.shape[2] - center['x'], center['x']]
            padY = [cube.shape[1] - center['y'], center['y']]
            padz = [0,0]
            cube = np.pad(cube, [padz,padY, padX], 'constant')
            ucube = np.pad(ucube, [padz,padY, padX], 'constant')

            print(f'Padded Data Corrected Cube shape {cube.shape}')

        #pbcorr=cube[int(cube.shape[0]/2),:,:]/ucube[int(cube.shape[0]/2),:,:] #TODO: load the pbcorr image, it changes with the frequency
        pbcorr=cube/ucube
        madmap3d = np.ones(shape=(ucube.shape))
        rms=np.array([])
        for nosireg in nosignalregions:
            rmsi=mad_std(ucube[:,nosireg['y0']:nosireg['y0']+nosireg['dy'],nosireg['x0']:nosireg['x0']+nosireg['dx']])
            rms=np.append(rms,rmsi)
            print(f"RMS Noise in area {nosireg} {rmsi}")
        rms=np.nanmean(rms)
        print('Mean RMS: ',rms)
        madmap3d=pbcorr*rms

    elif (pbfile is not None) and (ucube_file is None) and (cube_file is not None):
        hduUCO = fits.open(pbfile)[0]
        hdrUCO, pbcorr = hduUCO.header, hduUCO.data
        if pbcorr.shape[0]==1:
            pbcorr=pbcorr[0]
        print(f'Original pbema shape {pbcorr.shape}')

        hduCO = fits.open(cube_file)[0]
        hdr_cube, cube = hduCO.header, hduCO.data
        if cube.shape[0]==1:
            cube=cube[0]
        print(f'Original Data Corrected Cube shape {cube.shape}')

        ucube=cube/pbcorr

        print(f'Original Data UnCorrected Cube shape {ucube.shape}')
        if center != 'center':

            padX = [cube.shape[2] - center['x'], center['x']]
            padY = [cube.shape[1] - center['y'], center['y']]
            padz = [0,0]
            cube = np.pad(cube, [padz,padY, padX], 'constant')
            ucube = np.pad(ucube, [padz,padY, padX], 'constant')

            print(f'Padded Data Corrected Cube shape {cube.shape}')

        rms=np.array([])
        for nosireg in nosignalregions:
            rmsi=mad_std(ucube[:,nosireg['y0']:nosireg['y0']+nosireg['dy'],nosireg['x0']:nosireg['x0']+nosireg['dx']])
            rms=np.append(rms,rmsi)
            print(f"RMS Noise in area {nosireg} {rmsi}")
        rms=np.nanmean(rms)
        print('Mean RMS: ',rms)
        madmap3d=pbcorr*rms

    elif (pbfile is not None) and (ucube_file is not None) and (cube_file is None):
        hduUCO = fits.open(pbfile)[0]
        hdrUCO, pbcorr = hduUCO.header, hduUCO.data
        if pbcorr.shape[0]==1:
            pbcorr=pbcorr[0]
        print(f'Original pbema shape {pbcorr.shape}')

        hduUCO = fits.open(ucube_file)[0]
        hdr_cube, ucube = hduUCO.header, hduUCO.data
        if ucube.shape[0]==1:
            ucube=ucube[0]
        print(f'Original Data UnCorrected Cube shape {ucube.shape}')

        cube=pbcorr*ucube

        print(f'Original Data Corrected Cube shape {cube.shape}')
        print(f'Original Data UnCorrected Cube shape {ucube.shape}')
        if center != 'center':
            #pivot=[center['y'],center['x']]
            padX = [cube.shape[2] - center['x'], center['x']]
            padY = [cube.shape[1] - center['y'], center['y']]
            padz = [0,0]
            cube = np.pad(cube, [padz,padY, padX], 'constant')
            ucube = np.pad(ucube, [padz,padY, padX], 'constant')

            print(f'Padded Data Corrected Cube shape {cube.shape}')

        rms=np.array([])
        for nosireg in nosignalregions:
            rmsi=mad_std(ucube[:,nosireg['y0']:nosireg['y0']+nosireg['dy'],nosireg['x0']:nosireg['x0']+nosireg['dx']])
            rms=np.append(rms,rmsi)
            print(f"RMS Noise in area {nosireg} {rmsi}")
        rms=np.nanmean(rms)
        print('Mean RMS: ',rms)
        madmap3d=pbcorr*rms
    else:
        raise Exception('We need at least two cubes (pb corrected and uncorrected, pb corrected and pb correction, uncorrected and pb correction)')

    wmap = WCS(hdr_cube)
    if debug:
        fig1 = plt.figure(figsize=(11,50))
        gs=gridspec.GridSpec(6, 2, height_ratios=[1,1,1,1,1,1], width_ratios=[1,0.025])
        axc = fig1.add_subplot(gs[0,0])
        axu = fig1.add_subplot(gs[1,0])
        caximc=fig1.add_subplot(gs[0,1])
        caximu=fig1.add_subplot(gs[1,1])

        axc.set_title('Primary Beam Corrected integrated Image')
        imc = axc.imshow(np.nansum(cube,axis=0),origin='lower')
        plt.colorbar(imc, cax=caximc)

        axu.set_title('Non Corrected integrated Image')
        imu = axu.imshow(np.nansum(ucube,axis=0),origin='lower')
        if center!='center':
            axu.axvline(center['x']);axu.axhline(center['y'])
        for nosireg in nosignalregions:
            rect = patches.Rectangle((nosireg['x0'],nosireg['y0']),nosireg['dx'],nosireg['dy'],linewidth=1,edgecolor='r',facecolor='none')
            axu.add_patch(rect)
        plt.colorbar(imu, cax=caximu)#,orientation = 'horizontal', ticklocation = 'top')
        plt.show()


    dx = np.abs(hdr_cube['CDELT1'])*3600
    if zaxis=='vel':
        dv=hdr_cube['CDELT3']/1e3
        b=hdr_cube['CRVAL3']/1e3-hdr_cube['CRPIX3']*dv
        vel=np.linspace(b+dv,b+dv*hdr_cube['NAXIS3'],hdr_cube['NAXIS3'])#np.arange(b+dv,b+dv*hdrCO['NAXIS3']+dv,dv)#np.linspace(b,b+dv*hdrCO['NAXIS3'],hdrCO['NAXIS3'])
    elif zaxis=='freq':
        nu = (hdr_cube['CRVAL3'] + np.arange(hdr_cube['NAXIS3']) * hdr_cube['CDELT3']) # in Hz
        #nu0 = 230.5380e+9/(1. + z)     # Restframe CO(2-1) frequency in GHz, from splatalogue
        nu0 = restFreq/(1.+z)
        vel = ((nu0**2 - nu**2) / (nu0**2 + nu**2) * const.c.value * 1.e-3)
    else:
        print('zaxis must be vel or freq')

    dv = np.mean(vel[1:] - vel[:-1])   # average channel width in km/s
    dx = np.abs(hdr_cube['CDELT1'])*3600
    dy = np.abs(hdr_cube['CDELT2'])*3600
#         if center == 'center':
    x1=np.linspace(-cube.shape[2]*dx/2.,cube.shape[2]*dx/2.,int(cube.shape[2]))
    y1=np.linspace(-cube.shape[1]*dy/2.,cube.shape[1]*dy/2.,int(cube.shape[1]))
#         else:
#             x1=np.linspace(-center['x']*dx,cube.shape[2]*dx/2.,int(cube.shape[2]))
#             y1=np.linspace(-center['y']*dy/2.,cube.shape[1]*dy/2.,int(cube.shape[1]))
    if 'BMAJ' in hdr_cube.keys():
        Bmaj=hdr_cube['BMAJ']*3600
        Bmin=hdr_cube['BMIN']*3600
        Bpa=hdr_cube['BPA']#*3600
    elif beam is not None: 
        print('BEAM not in HEADER, using user values')
        Bmaj,Bmin,Bpa=beam
    else:
        raise Exception('BEAM not in HEADER, use the beam argument')
    arctokpc=((cosmo.angular_diameter_distance(z=z)/206265).to(u.kpc)/u.arcsec).value
    data=xr.Dataset()
    #print(arctokpc,x1,dx,cube.shape)
    data['cube']=xr.DataArray(cube, coords=[vel, y1*arctokpc,x1*arctokpc], dims=['v', 'y' ,'x'])
    data['madcube']=xr.DataArray(madmap3d, coords=[vel, y1*arctokpc,x1*arctokpc], dims=['v', 'y' ,'x'])
    
   
    if drop is not None:
        # data=data.where(((data.v>-drop['v'])&(data.v<drop['v']))&((data.x>-drop['x'])&(data.x<drop['x']))&((data.y>-drop['y'])&(data.y<drop['y'])),drop=True)
        wx=(-drop['x']<data.x)&(drop['x']>data.x)
        wy=(-drop['y']<data.y)&(drop['y']>data.y)
        wv=(-drop['v']<data.v)&(drop['v']>data.v)
        vs=slice(np.ix_(wv)[0][0],np.ix_(wv)[0][-1])
        xs=slice(np.ix_(wx)[0][0],np.ix_(wx)[0][-1])
        ys=slice(np.ix_(wy)[0][0],np.ix_(wy)[0][-1])
        data=data.where(wx&wy&wv,drop=True)
        # ws=slice(np.ix_(wv,wy,wx))
        # wmap2=wmap.slice((ws))
        wmap2=wmap.celestial[ys,xs,]#.slice((ws))
    else:
        wmap2=wmap.celestial#[ivlim[0]:ivlim[1],:,:]

    data.attrs['z']=z
    data.attrs['header_original']=hdr_cube
    #data.attrs['wcs']=wmap2
    data.attrs['n']=n
    data.attrs['x_units']=u.kpc
    data.attrs['y_units']=u.kpc
    data.attrs['v_units']=u.km/u.s
    data.attrs['cube_units']=u.Jy/u.beam
    data.attrs['arctokpc']=arctokpc
    data.attrs['xsize']=data.dims['x']*dx*arctokpc
    data.attrs['ysize']=data.dims['y']*dy*arctokpc
    data.attrs['vsize']=data.dims['v']*dv
    data.attrs['dx']=dx*arctokpc
    data.attrs['dy']=dy*arctokpc
    data.attrs['dv']=dv
    data.attrs['rms']=rms
    data.attrs['beam']=np.array([Bmaj*arctokpc,Bmin*arctokpc,Bpa])
    data.attrs['beam_area']=data.attrs['beam'][0]*data.attrs['beam'][1]*np.pi/(4*np.log(2))*u.kpc**2/u.beam#kpc^2
    data.attrs['pixel_area']=data.attrs['dx']*data.attrs['dy']/u.pixel**2

    display(data.attrs['beam_area'])
    display(data.attrs['pixel_area'])
    
    mask=(data['cube']>sigma*data['madcube'])
    
    mom0=np.nansum(mask,axis=0)
    mask=mom0>0
    if area_mask is not None:
        amask=~(data['cube'].x>-100)&(data['cube'].y>-100)&(data['cube'].v>-4000)
        for area in area_mask:
            x0=area_mask[area]['x0']
            Rx=area_mask[area]['Rx']
            y0=area_mask[area]['y0']
            Ry=area_mask[area]['Ry']
            v0=area_mask[area]['v0']
            Rv=area_mask[area]['Rv']
            if area_mask[area]['s'] == 'del':
                amask=amask | (~(((data['cube'].x-x0)/Rx)**2+((data['cube'].y-y0)/Ry)**2+((data['cube'].v-v0)/Rv)**2<1))
            elif area_mask[area]['s'] == 'add':
                amask=amask | (((data['cube'].x-x0)/Rx)**2+((data['cube'].y-y0)/Ry)**2+((data['cube'].v-v0)/Rv)**2<1)
            else:
                raise Exception('Not an option (del|add)')
        cubedata=data.where(mask).where(amask,drop=True)
        cubedata_total=data.where(amask,drop=False)
        #data.update({name:{'cube':data.where(mask).where(amask,drop=True),'cube_total':data.where(amask,drop=False)}})
    else:
        cubedata=data.where(mask).where(mask)
        cubedata_total=data
        #data.update({name:{'cube':data.where(mask),'cube_total':data}})

    mom0=np.nansum(cubedata['cube'].data,axis=0)
    iyy,ixx=np.where(np.where(mom0>0,True,False))
    cubedata.attrs['ixx']=ixx
    cubedata.attrs['iyy']=iyy
   
    if debug:
        fig1 = plt.figure(figsize=(11,20))
        axu = fig1.add_subplot(1,1,1)

        axu.set_title('Masked image')
        imu = axu.pcolormesh(cubedata.x,cubedata.y,np.nansum(cubedata['cube'],axis=0))
        axu.axvline(0);axu.axhline(0)
        if area_mask is not None:
            for area in area_mask:
                x0=area_mask[area]['x0']
                Rx=area_mask[area]['Rx']
                y0=area_mask[area]['y0']
                Ry=area_mask[area]['Ry']
                el = patches.Ellipse((x0,y0),2*Rx,2*Ry,linewidth=1,edgecolor='r',facecolor='none')
                axu.add_patch(el)
        axu.set(aspect=1)
        plt.show()

    return cubedata,cubedata_total

def Lco_solomon(Idv,z,n): 
    D=cosmo.luminosity_distance(z)
    return 3.25e7*Idv*D**2/((n*115.271)**2*(1+z))#**3)
def jytoMo(jydv,z,n,aco=4.2):
    v0=115.271
    D=cosmo.luminosity_distance(z).value
    return 3.25e7*aco*jydv*D**2/(n*v0)**2/(1+z)
def outflow_properties(regions,z,n=2,aco=[0.8,1,4.4]):
    total={a:0 for a in aco} 
    for region in regions:
        print(region)
        V=regions[region]['V']*u.km/u.s
        R=regions[region]['R']*u.kpc
        JVout=regions[region]['JdV']
        
        for a in aco:
            print('========-a={}-========'.format(a))
            M=jytoMo(JVout,z,n,a) *u.solMass
            display(M)
            Mout=(V*M/R).to(u.solMass/u.year)
            
            Pout=(0.5*V**2*Mout).to(u.erg/u.s)
            momentum=(V*Mout).to(u.dyn)
            #dMout=np.sqrt((dV*M/R)**2+(dR*M*V/R**2)**2).to(u.solMass/u.year)
            print('Mass Outflow Rate')
            display(Mout)#,dMout)
            print('Outflow Kinetic Power')
            display(Pout)#,dMout)
            print('Outflow Momentum')
            display(momentum)#,dMout)
            total.update({a:total[a]+Mout})
    display(total)