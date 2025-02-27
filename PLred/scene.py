import numpy as np
import joblib
from sklearn.decomposition import PCA, IncrementalPCA
import os, glob
import time
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt



def make_simple_star(Vrot, Rstar, incl_angle, PA, ngrid=128, fov=30, plot=True):
    '''
    make simple Lorentzian star model
    Assumes Lorentzian spectral profile for each light-emitting patch
    and solid-body rotation

    args:
        Vrot: stellar rotation velocity (km/s)
        Rstar: stellar radius (mas)
        incl_angle: inclination angle (radians)
        PA: star position angle (radians)
        ngrid: resolution of the output image
        fov: field of view (diameter) of the output image (mas)
    
    returns:
        intenmap: intensity map
        velmap: velocity map
        xg: x grid in mas
        yg: y grid in mas
    '''

    # output grid
    xa = np.linspace(-fov/2, fov/2, ngrid)
    yg, xg = np.meshgrid(xa, xa, indexing='ij') 

    # grid in the star cylindrical coordinates
    mat = np.array([[np.cos(PA) * np.cos(incl_angle), np.sin(PA) * np.cos(incl_angle)], 
                    [-np.sin(PA), np.cos(PA)]]) / np.cos(incl_angle)
    [xg2, yg2] = (mat @ (np.array([xg.flatten(), yg.flatten()]))) 
    xg2 = xg2.reshape((ngrid, ngrid))
    yg2 = yg2.reshape((ngrid, ngrid))

    # distance to the star center
    rmap = np.sqrt(xg2**2 + yg2**2)

    # intensity map of the star (assuming uniform intensity for simplicity)
    intenmap = np.ones_like(rmap)
    intenmap[xg**2+yg**2 > (Rstar)**2] = 0

    # velocity map for solid-body rotation
    velmap = -(Vrot) * np.sin(incl_angle) * (np.cos(PA) * xg + np.sin(PA) * yg) / Rstar
    velmap[xg**2+yg**2 > (Rstar)**2] = 0

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        axs[0].imshow(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2))
        # axs[0].contour(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
        #                levels=[0.5], colors='white')
        axs[1].contour(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
                       levels=[0.5], colors='white')
        p = axs[1].imshow(velmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2), cmap='RdBu')
        # axs[1].contour(velmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
        #                colors='black', linestyles='--', alpha=0.3, levels=[0])
        plt.colorbar(p, ax=axs[1])
        
        axs[0].set_title('Intensity map')
        axs[1].set_title('Velocity map')

        for ax in axs:
            ax.set_xlabel('x (mas)')
            ax.set_ylabel('y (mas)')
    return intenmap, velmap, xg, yg




def make_simple_gaussian_disk(Vrot, Rstar, disk_fwhm, incl_angle, PA, beta = -0.5,
                              ngrid = 128, fov = 30, plot = True):
    '''
    make simple Gaussian disk model
    Assumes Gaussian intensity distribution
    and simple rotation (zero inward/outward velocity)

    args:
        Vrot: stellar rotation velocity (km/s)
        Rstar: stellar radius (mas)
        disk_fwhm: gaussian FWHM of the disk intensity distribution (mas)
        incl_angle: inclination angle (radians)
        PA: disk position angle (radians)
        beta: power of the rotation velocity profile (-0.5 for Keplerian)
        ngrid: resolution of the output image
        fov: field of view (diameter) of the output image (mas)
    
    returns:
        intenmap: intensity map
        velmap: velocity map
        xg: x grid in mas
        yg: y grid in mas
    '''

    # output grid
    xa = np.linspace(-fov/2, fov/2, ngrid)
    yg, xg = np.meshgrid(xa, xa, indexing = 'ij') 

    # grid in the disk cylindrical coordinates
    mat = np.array([[np.cos(PA) * np.cos(incl_angle), np.sin(PA) * np.cos(incl_angle)], [-np.sin(PA), np.cos(PA)]]) / np.cos(incl_angle)
    [xg2,yg2] = (mat @ (np.array([xg.flatten(), yg.flatten()]))) 
    xg2 = xg2.reshape((ngrid, ngrid))
    yg2 = yg2.reshape((ngrid, ngrid))

    # distance to the disk center
    rmap = np.sqrt(xg2**2 + yg2**2)
    rmask = np.sqrt(xg**2 + yg**2) > Rstar

    # intensity map of the disk
    disk_sig = disk_fwhm / (2*np.sqrt(2*np.log(2)))
    intenmap = np.exp(-0.5*(rmap/disk_sig)**2) * rmask

    # define rotation velocity profile
    def vfun(r): return Vrot * (r / Rstar)**beta

    # velocity map
    velmap = -(vfun(rmap)/rmap)* np.sin(incl_angle) * (np.cos(PA) * xg + np.sin(PA) * yg) * rmask

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        axs[0].imshow(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2))
        # axs[0].contour(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
        #                levels = [0.5], colors='white')
        p=axs[1].imshow(velmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2), cmap='RdBu')
        axs[1].contour(velmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
                       colors='black',linestyles='--', alpha=0.3,
                       levels=[0])
        plt.colorbar(p, ax=axs[1])
        
        axs[0].set_title('Intensity map')
        axs[1].set_title('Velocity map')

        for ax in axs:
            ax.set_xlabel('x (mas)')
            ax.set_ylabel('y (mas)')
    return intenmap, velmap, xg, yg


def make_simple_powerlaw_disk(Vrot, Rstar, Rout, power_index, incl_angle, PA, beta = -0.5,
                              ngrid = 128, fov = 30, 
                              plot = True):
    '''
    make simple Gaussian disk model
    Assumes Gaussian intensity distribution
    and simple rotation (zero inward/outward velocity)

    args:
        Vrot: stellar rotation velocity (km/s)
        Rstar: stellar radius (mas)
        Rout: disk outer radius (mas)
        power_index: power of the power-law intensity distribution
        incl_angle: inclination angle (radians)
        PA: disk position angle (radians)
        beta: power of the rotation velocity profile (-0.5 for Keplerian)
        ngrid: resolution of the output image
        fov: field of view (diameter) of the output image (mas)
    
    returns:
        intenmap: intensity map
        velmap: velocity map
        xg: x grid in mas
        yg: y grid in mas
    '''

    # output grid
    xa = np.linspace(-fov/2, fov/2, ngrid)
    yg, xg = np.meshgrid(xa, xa, indexing = 'ij') 

    # grid in the disk cylindrical coordinates
    mat = np.array([[np.cos(PA) * np.cos(incl_angle), np.sin(PA) * np.cos(incl_angle)], [-np.sin(PA), np.cos(PA)]]) / np.cos(incl_angle)
    [xg2,yg2] = (mat @ (np.array([xg.flatten(), yg.flatten()]))) 
    xg2 = xg2.reshape((ngrid, ngrid))
    yg2 = yg2.reshape((ngrid, ngrid))

    # distance to the disk center
    rmap = np.sqrt(xg2**2 + yg2**2)
    rmask = (np.sqrt(xg**2 + yg**2) > Rstar) & (rmap < Rout)

    # intensity map of the disk
    
    intenmap = (rmap/Rout)**power_index * rmask

    # define rotation velocity profile
    def vfun(r): return Vrot * (r / Rstar)**beta

    # velocity map
    velmap = -(vfun(rmap)/rmap)* np.sin(incl_angle) * (np.cos(PA) * xg + np.sin(PA) * yg) * rmask

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        axs[0].imshow(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2))
        # axs[0].contour(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
        #                levels = [0.5], colors='white')
        p=axs[1].imshow(velmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2), cmap='RdBu')
        axs[1].contour(velmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
                       colors='black',linestyles='--', alpha=0.3,
                       levels=[0])
        plt.colorbar(p, ax=axs[1])
        
        axs[0].set_title('Intensity map')
        axs[1].set_title('Velocity map')

        for ax in axs:
            ax.set_xlabel('x (mas)')
            ax.set_ylabel('y (mas)')
    return intenmap, velmap, xg, yg


def get_iso_velocity_map(intenmap, velmap, vgrid, vsig = None):
    '''
    makes iso-velocity maps given intensity map, velocity map, and velocity grid
    this only works with regular velocity grid
    '''

    assert np.all(np.isclose(np.diff(vgrid), np.diff(vgrid)[0])), "only supports regular velocity grid!"
    
    if vsig is None:
        dv = np.diff(vgrid)[0]
        vsig = dv / (2*np.sqrt(2*np.log(2)))

    iso_maps = []
    for v in vgrid:
        iso_maps.append(intenmap * (1/vsig/np.sqrt(2*np.pi))*np.exp(-(velmap - v)**2 / 2 / vsig**2))

    return np.array(iso_maps)



def get_iso_velocity_map_lorentzian_absorption(intenmap, velmap, vgrid, star_fwhm, depth):
    '''
    makes iso-velocity maps given intensity map, velocity map, and velocity grid
    this only works with regular velocity grid

    args:
        intenmap: intensity map
        velmap: velocity map
        vgrid: array of velocities (km/s)
        star_fwhm: Lorentzian FWHM of the star spectral profile (km/s)
        depth: depth of the absorption (compared to unity)
    
    returns:
        iso_maps: array of iso-velocity maps
    '''
    from astropy.modeling.functional_models import Lorentz1D

    assert np.all(np.isclose(np.diff(vgrid), np.diff(vgrid)[0])), "only supports regular velocity grid!"

    iso_maps = []
    for v in vgrid:
        lorentzian = Lorentz1D(amplitude=depth, x_0=v, fwhm=star_fwhm)
        iso_map = intenmap * (1-lorentzian(velmap))
        iso_maps.append(iso_map)

    return np.array(iso_maps)

class Scene:
    '''
    Scene object
    '''

    def __init__(self, fov, n_grid, vels):
        '''
        initialize scene object

        Parameters
        ----------
        fov: float
            field of view (diameter) of the scene (mas)
        n_grid: int
            number of grid points in each dimension
        vels: array
            velocity grid
        '''

        self.fov = fov
        self.n_grid = n_grid
        self.vels = vels
        self.nvels = len(vels)
        
        assert np.all(np.isclose(np.diff(vels), np.diff(vels)[0])), "only supports regular velocity grid!"
        self.dv = np.diff(vels)[0]

        # output grid
        xa = np.linspace(-fov/2, fov/2, n_grid)
        self.yg, self.xg = np.meshgrid(xa, xa, indexing = 'ij') 

        self.iso_maps = np.zeros((self.nvels, n_grid, n_grid))
        self.intenmap = np.zeros((n_grid,n_grid))

        self.center_coord = n_grid // 2
    
    def make_kepler_disk(self, Vrot, Rstar, disk_fwhm, incl_angle, PA, beta = -0.5, plot=False,
                         initialize = True, vsig = None):

        '''
        make simple Keplerian disk model

        Parameters
        ----------
        Vrot: float
            stellar rotation velocity (km/s)
        Rstar: float
            stellar radius (mas)
        disk_fwhm: float
            gaussian FWHM of the disk intensity distribution (mas)
        incl_angle: float
            inclination angle (radians)
        PA: float
            disk position angle (radians)
        beta: float
            power of the rotation velocity profile (-0.5 for Keplerian)
        '''

        # initialize iso map
        if initialize:
            self.iso_maps *= 0
            self.intenmap *= 0

        d_intenmap, self.velmap, _, _ = make_simple_gaussian_disk(Vrot, Rstar, disk_fwhm, incl_angle, PA, beta = beta,
                              ngrid = self.n_grid, fov = self.fov, plot = plot)

        self.iso_maps += get_iso_velocity_map(d_intenmap, self.velmap, self.vels, vsig = vsig)
        self.intenmap += d_intenmap

    # def make_star(self, Vrot, Rstar, incl_angle, PA, plot=False, initialize = True):

    #     '''
    #     make simple star model

    #     Parameters
    #     ----------
    #     Vrot: float
    #         stellar rotation velocity (km/s)
    #     Rstar: float
    #         stellar radius (mas)
    #     incl_angle: float
    #         inclination angle (radians)
    #     PA: float
    #         star position angle (radians)
    #     '''

    #     # initialize iso map
    #     if initialize:
    #         self.iso_maps *= 0
    #         self.intenmap *= 0

    #     d_intenmap, self.velmap, _, _ = make_simple_star(Vrot, Rstar, incl_angle, PA,
    #                           ngrid = self.n_grid, fov = self.fov, plot = plot)

    #     self.iso_maps += get_iso_velocity_map(d_intenmap, self.velmap, self.vels)
    #     self.intenmap += d_intenmap

    def add_point_source(self, inten_ratio):

        '''
        add a point source with constant continuum emission
        note the grid needs to be odd number in order to place the star really in the center
        '''
        flux = inten_ratio * np.sum(self.intenmap)


        for vind in range(self.nvels):
            self.iso_maps[vind,self.center_coord, self.center_coord] += flux
    


        
