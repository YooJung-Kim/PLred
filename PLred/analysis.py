import numpy as np
import joblib
from sklearn.decomposition import PCA, IncrementalPCA
import os, glob
import time
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt

from .spec import polar_1, polar_2

NFIB = 38
zaber_microns = 0.047 # 1 zaber step = 0.047 microns
diameter = 8.2 # m


def flatten(cube, sum_polar = True):

    if len(np.shape(cube)) == 2:
        cube = np.array([cube])

    flattened_cube = []

    for frameind in range(len(cube)):

        if sum_polar:
            flattened_cube.append(np.array([(cube[frameind][polar_1[fibind]] + cube[frameind][polar_2[fibind]]) for fibind in range(19)]).flatten())
        else:
            flattened_cube.append(np.array([cube[frameind][fibind] for fibind in range(38)]).flatten())

    return np.array(flattened_cube)

def normalize_flatten(cube, sum_polar = True):

    if len(np.shape(cube)) == 2:
        cube = np.array([cube])
    
    sumspec = np.sum(cube, axis=1)

    flattened_cube = []

    for frameind in range(len(cube)):

        if sum_polar:
            flattened_cube.append(np.array([(cube[frameind][polar_1[fibind]] + cube[frameind][polar_2[fibind]]) / sumspec[frameind] for fibind in range(19)]).flatten())
        else:
            flattened_cube.append(np.array([cube[frameind][fibind] / sumspec[frameind] for fibind in range(38)]).flatten())

    return np.array(flattened_cube)

def compute_PCA(files, outname, sum_polar = True, n_components = 50, normalize = True):

    sizes = []
    all_data = []

    for i in tqdm(range(len(files))):

        cube = fits.getdata(files[i])
        if normalize:
            normalized_dat = normalize_flatten(cube, sum_polar = sum_polar)
        else:
            normalized_dat = flatten(cube, sum_polar = sum_polar)
        all_data.append(normalized_dat)
        sizes.append(len(normalized_dat))
    
    all_data = np.vstack(all_data)

    # perform PCA
    start = time.time()
    pca = PCA(n_components=n_components)
    pca.fit_transform(all_data)
    end = time.time()
    print(f"time elapsed: {end - start:.2f} seconds")

    # save the model
    joblib.dump(pca, outname + '.pkl')  


def compute_IPCA(files, outname, batch_size = 250, sum_polar = True, n_components = 50):
    '''

    '''
    memmap_filename = 'intermediate_data.memmap'

    sizes = []

    with open(memmap_filename, "w+b") as mmap_file:

        for i in tqdm(range(len(files))):

            cube = fits.getdata(files[i])
            normalized_dat = normalize_flatten(cube, sum_polar = sum_polar)

            mmap_file.write(normalized_dat.astype("float32").tobytes())

            sizes.append(len(normalized_dat))

            if i == 0:
                feature_size = np.shape(normalized_dat)[1]
    
    cumsize = np.cumsum([0] + sizes)
    total_rows = cumsize[-1]

    all_data = np.memmap(memmap_filename, dtype='float32', mode='r',
                         shape = (total_rows, feature_size))
    
    # perform IPCA
    start = time.time()
    pca = IncrementalPCA(n_components=n_components)
    for start_idx in tqdm(range(0, total_rows, batch_size)):
        end_idx = min(start_idx + batch_size, total_rows)

        print(f"perform IPCA for batch {start_idx}:{end_idx}")
        pca.partial_fit(all_data[start_idx:end_idx])
    
    end = time.time()
    print(f"time elapsed: {end - start:.2f} seconds")

    # save the model
    joblib.dump(pca, outname + '.pkl')

    # clean up memory-mapped file
    del all_data
    os.remove(memmap_filename)


def plot_PCA_results(pcafile, outdir, components_upto = 5, xmin = None, xmax = None,
                     vertical_marks = []):

    os.makedirs(outdir, exist_ok = True)

    pca = joblib.load(pcafile)
    nspec = pca.n_features_in_ // (NFIB//2)

    if xmin is None: xmin = 0
    if xmax is None: xmax = nspec

    specarr = np.arange(nspec)

    fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(15,12))
    axs = axs.flatten()

    for fibind in range(NFIB//2):
        axs[fibind].plot(specarr[xmin:xmax], pca.mean_[nspec*fibind:nspec*(fibind+1)][xmin:xmax])
        axs[fibind].set_title('port %d' % fibind)

        for v in vertical_marks: axs[fibind].axvline(v)
    
    fig.tight_layout()
    
    fig.savefig(outdir+'/pca_mean.png')

    for ind in range(components_upto):

        fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(15,12))
        axs = axs.flatten()

        for fibind in range(NFIB//2):
            axs[fibind].plot(specarr[xmin:xmax], pca.components_[ind,nspec*fibind:nspec*(fibind+1)][xmin:xmax])
            axs[fibind].set_title('port %d' % fibind)
        
            for v in vertical_marks: axs[fibind].axvline(v)

        fig.tight_layout()
        fig.savefig(outdir+'/pca_mode%d.png' % ind)

def plot_coupling_maps(couplingmap_file, outname, norm = False, fnumber = 8,
                       specinds = None):

    if norm:
        cube = fits.open(couplingmap_file)[2].data
    else:
        cube = fits.open(couplingmap_file)[1].data

    cubeheader = fits.open(couplingmap_file)[0].header

    window_step = cubeheader['WINDOW']
    npt = cubeheader['NPT']

    x_pos = np.linspace(0 - window_step/2, 0 + window_step/2, npt) * zaber_microns
    y_pos = np.linspace(0 - window_step/2, 0 + window_step/2, npt) * zaber_microns

    plate_scale = 206265 / (fnumber * diameter * 1e6) * 1e3 # mas per micron

    x_pos_mas = x_pos * plate_scale
    y_pos_mas = y_pos * plate_scale

    if specinds is None:
        specinds = np.arange(np.shape(cube)[1])
    
    cube_sliced = np.average(cube[:, specinds, :, :], axis=1)
    
    fig, axs = plt.subplots(ncols=5, nrows=8, figsize=(10,16), sharex=True, sharey=True)
    axs = axs.flatten()


    for fibind in range(NFIB):
        axs[fibind].imshow(cube_sliced[fibind, :, :], origin='lower',
                        extent = (min(x_pos_mas), max(x_pos_mas), min(y_pos_mas), max(y_pos_mas)))
        axs[fibind].set_title('Fiber {}'.format(fibind))
        if fibind // 5 == 7: axs[fibind].set_xlabel('x (mas)')
        if fibind % 5 == 0: axs[fibind].set_ylabel('y (mas)')
    axs[39].imshow(np.sum(cube_sliced[:, :, :], axis=0), origin='lower',
                extent = (min(x_pos_mas), max(x_pos_mas), min(y_pos_mas), max(y_pos_mas)))
    axs[39].set_title('Sum')
    axs[38].axis('off')

    axs[39].set_xlabel('x (mas)')
    axs[39].set_ylabel('y (mas)')

    fig.savefig(outname+'.png')

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

    # intensity map of the disk
    disk_sig = disk_fwhm / (2*np.sqrt(2*np.log(2)))
    intenmap = np.exp(-0.5*(rmap/disk_sig)**2)

    # define rotation velocity profile
    def vfun(r): return Vrot * (r / Rstar)**beta

    # velocity map
    velmap = -(vfun(rmap)/rmap)* np.sin(incl_angle) * (np.cos(PA) * xg + np.sin(PA) * yg)

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        axs[0].imshow(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2))
        axs[0].contour(intenmap, origin='lower', extent=(-fov/2, fov/2, -fov/2, fov/2),
                       levels = [0.5], colors='white')
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

    # if not drop_last_vgrid:
    #     print('Warning - this is not implemented')

    # dvs = np.diff(vgrid)
    # vsigs = dvs / (2*np.sqrt(2*np.log(2)))

    # iso_maps = []
    # for (v, vsig) in zip(vgrid, vsigs):

    #     iso_maps.append(intenmap * (1/vsig/np.sqrt(2*np.pi))*np.exp(-(velmap - v)**2 / 2 / vsig**2))
    
    # return np.array(iso_maps)

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

        # d_intenmap = np.zeros_like(self.intenmap)
        # d_intenmap[self.center_coord, self.center_coord] += flux

        # const_velmap = 

        # self.iso_maps += get_iso_velocity_map(d_intenmap, const_velmap, self.vels)
        


        # self.intenmap[self.center_coord, self.center_coord] += flux

        for vind in range(self.nvels):
            self.iso_maps[vind,self.center_coord, self.center_coord] += flux
    

class PLSimulator:
    '''
    Simulate PL spectrum given the coupling map and the input scene
    '''

    # interpolation scheme for wavelength
    WAV_INTERP_KIND = 'cubic'

    # spectrograph's resolving power
    RESOLVING_POWER = 3000

    def __init__(self, couplingmap_file,
                 fov, n_grid, vels, velzero,
                 wavelength_min = None, wavelength_max = None,
                 oversample_factor = 4):
        '''
        Initialize the simulator

        Parameters
        ----------
        couplingmap_file: str
            path to the coupling map file
        fov: float
            field of view (diameter) of the scene (mas)
        n_grid: int
            number of grid points in each dimension
        vels: array
            velocity grid
        velzero: float
            wavelength of zero velocity (nm)
        '''

        self.read_coupling_map(couplingmap_file, velzero, 
                               wavelength_min=wavelength_min, wavelength_max=wavelength_max,
                               oversample_factor = oversample_factor)
        self.scene = Scene(fov, n_grid, vels)
        self.scene_resolution = fov / n_grid

        # self.zoom_fac = self.cube_resolution / self.scene_resolution 

        # print("zoom factor: ", self.zoom_fac)

        self.compute_interp_models()


    def read_coupling_map(self, couplingmap_file, velzero, fnumber= 8, 
                          oversample_factor = 4,
                          wavelength_nbin = 4,
                          wavelength_min = None, wavelength_max = None):
        '''
        read coupling map file and do key operations

        Parameters
        ----------
        couplingmap_file: str
            path to the coupling map file
        velzero: float
            wavelength of zero velocity (nm)
        fnumber: float
            f number where the coupling map is taken
        oversample_factor: float
            zoom factor to oversample coupling maps
        wavelength_nbin: int
            number of wavelength channels to average coupling maps
        wavelength_min: float
            minimum of wavelength range
        wavelength_max: float
            maximum of wavelength range
        
        '''
        self.cube = fits.open(couplingmap_file)[1].data
        self.cube_header = fits.open(couplingmap_file)[0].header
        self.cube_w = self.cube_header['WINDOW']
        self.cube_n = self.cube_header['NPT']
        self.cube_model = fits.open(couplingmap_file)[1].header['MODEL']
        
        if '.npz' in self.cube_model:
            self.wavmap = np.load(self.cube_model)['wav_map']
        else:
            self.wavmap = np.load(self.cube_model+'_wavmap.npy')
        
        self.wavs = self.wavmap[0] # reference index
        self.vels = (self.wavs - velzero) / velzero * 3e5 # km/s

        try:
            # potentially add FNUM to header keyword
            self.fnumber = self.cube_header['FNUM']
        except:
            self.fnumber = fnumber
        
        # truncate wavelength range if wavelength_min and wavelength_max are given
        wav_idx = np.where((self.wavs >= wavelength_min) & (self.wavs <= wavelength_max))[0]
        self.wavs = self.wavs[wav_idx]
        self.vels = self.vels[wav_idx]
        self.cube = self.cube[:, wav_idx, :, :]

        # copy the wavelengths before binning
        self.wavs0 = self.wavs.copy()
        self.vels0 = self.vels.copy()

        # bin wavelength to get better S/N (caution! this will blur the coupling map)
        effective_num = (len(self.wavs) // wavelength_nbin) * wavelength_nbin
        self.wavs = np.average(self.wavs[:effective_num].reshape((-1, wavelength_nbin)), axis=1)
        self.vels = np.average(self.vels[:effective_num].reshape((-1, wavelength_nbin)), axis=1)
        self.cube = np.average(self.cube[:,:effective_num].reshape((NFIB, -1, wavelength_nbin, self.cube_n, self.cube_n)), axis=2)


        print("wavelength range: ", self.wavs[0], self.wavs[-1], "number of wavelengths: ", len(self.wavs))
        self.nwav = len(self.wavs)

        self.oversample_factor = oversample_factor
        self.cube_oversampled = np.zeros((NFIB, len(self.vels), oversample_factor * self.cube_n, oversample_factor * self.cube_n))
        
        if oversample_factor != 1:
            from scipy.ndimage import zoom
            for fibind in range(NFIB):
                for specind in range(len(self.vels)):
                    self.cube_oversampled[fibind, specind, :, :] = zoom(self.cube[fibind, specind, :, :], oversample_factor)
        else:
            self.cube_oversampled = self.cube
        
        x_pos = np.linspace(0 - self.cube_w/2, 0 + self.cube_w/2, oversample_factor * self.cube_n) * zaber_microns
        y_pos = np.linspace(0 - self.cube_w/2, 0 + self.cube_w/2, oversample_factor * self.cube_n) * zaber_microns

        plate_scale = 206265 / (fnumber * diameter * 1e6) * 1e3 # mas per micron

        x_pos_mas = x_pos * plate_scale
        y_pos_mas = y_pos * plate_scale

        self.x_pos_mas = x_pos_mas
        self.y_pos_mas = y_pos_mas

        self.cube_resolution = self.x_pos_mas[1] - self.x_pos_mas[0]


    def compute_interp_models(self):
        '''
        compute interpolation functions to make zoom-in coupling maps 
        '''

        from scipy.interpolate import RegularGridInterpolator

        self.interp_maps = []

        for fibind in range(NFIB):
            
            temp = []
            for wavind in range(self.nwav):
                temp.append(
                    RegularGridInterpolator((self.x_pos_mas, self.y_pos_mas),
                                            self.cube_oversampled[fibind, wavind])
                    )
            self.interp_maps.append(temp)
        
    
    def resample_coupling_map(self, fibind, wavind, xcenter, ycenter, plot=False):
        '''
        resample coupling map on the grid that matches the scene
        '''

        flat_coors = (self.scene.yg.flatten() + ycenter, self.scene.xg.flatten() + xcenter)
        resampled_map = self.interp_maps[fibind][wavind](flat_coors).reshape((self.scene.n_grid, self.scene.n_grid))
        
        if plot:
            map0 = self.cube_oversampled[fibind, wavind, :, :]
            plt.imshow(map0, origin='lower',
                    extent = (min(self.x_pos_mas), max(self.x_pos_mas), min(self.y_pos_mas), max(self.y_pos_mas)),
                    vmin=0,vmax=6000)
            plt.plot(xcenter, ycenter, 'o', color='red')
            plt.axvline(xcenter - self.scene.fov/2, color='white')
            plt.axvline(xcenter + self.scene.fov/2, color='white')
            plt.axhline(ycenter - self.scene.fov/2, color='white')
            plt.axhline(ycenter + self.scene.fov/2, color='white')
            plt.show()

            plt.imshow(map0, origin='lower',vmin=0,vmax=6000,
                    extent = (min(self.x_pos_mas), max(self.x_pos_mas), min(self.y_pos_mas), max(self.y_pos_mas)))
            plt.xlim([xcenter - self.scene.fov/2, xcenter + self.scene.fov/2])
            plt.ylim([ycenter - self.scene.fov/2, ycenter + self.scene.fov/2])
            plt.show()

        return resampled_map
    
    def interp_wavelength_func(self, maps):

        '''
        interpolate between zoom-in coupling maps to get
        finer sampling in wavelength
        '''

        flat_maps = np.reshape(maps, newshape=(len(maps), -1))
        flat_maps = np.flip(flat_maps, axis=0)
        from scipy.interpolate import interp1d
        
        interp_func = interp1d(np.flip(self.vels), flat_maps.T, kind=self.WAV_INTERP_KIND)
        return interp_func

    
    def simulate_spectra(self, xc, yc, iso_maps = None,
                         downsample = False):
        '''
        simulate spectrum, given the position of the input scene at
        (xc, yc) in mas.
        '''

        if iso_maps is None:

            iso_maps = self.scene.iso_maps
        
        # resample coupling map at the position (xc, yc)

        interp_funcs = []

        for fibind in range(NFIB):

            temp = []
            
            for wavind in range(self.nwav):

                # print(f"resampling fibind {fibind}, wavind {wavind}")

                resampled = self.resample_coupling_map(fibind, wavind, xc, yc)
                temp.append(resampled)
            
            interp_func = self.interp_wavelength_func(np.array(temp))

            interp_funcs.append(interp_func)

        # compute the overlap

        specs = np.zeros((NFIB, len(self.scene.vels)))

        for iv, vel in (enumerate(self.scene.vels)):

            # iso velocity map for current velocity
            iso_map = iso_maps[iv]

            # interpolate coupling maps to get the map of this velocity

            for fibind in range(NFIB):

                cm = reshape_square(interp_funcs[fibind](vel))
                specs[fibind, iv] = np.sum(cm * iso_map)
        
        if downsample:

            downsampled_specs = []
            for fibind in range(NFIB):
                downsampled_specs.append(self.downsample_spec(specs[fibind]))
            return np.array(downsampled_specs)
        else:
            return specs
    
    def downsample_spec(self, spec):

        from scipy.ndimage import gaussian_filter

        sigma = (3e5 / self.RESOLVING_POWER) / self.scene.dv / (2*np.sqrt(2*np.log(2)))

        convolved_spec = gaussian_filter(spec, sigma)
        downsampled_spec = np.interp(self.vels0, self.scene.vels, convolved_spec)
        return downsampled_spec
        

class PLfitter(PLSimulator):

    def __init__(self, couplingmap_file, wavelengths,
                 fov, n_grid, vels, velzero,
                 wavelength_min = None, wavelength_max = None,
                 oversample_factor = 4,
                 wavelength_nbin = 1):        

        self.read_coupling_map(couplingmap_file, wavelengths, velzero, 
                               wavelength_min=wavelength_min, wavelength_max=wavelength_max,
                               oversample_factor = oversample_factor,
                               wavelength_nbin=wavelength_nbin)
        
        self.scene = Scene(fov, n_grid, vels)
        self.scene_resolution = fov / n_grid

        # self.zoom_fac = self.cube_resolution / self.scene_resolution 

        # print("zoom factor: ", self.zoom_fac)

        self.compute_interp_models()

    def read_coupling_map(self, couplingmap_file, wavelengths, velzero, fnumber= 8, 
                          oversample_factor = 4,
                          wavelength_nbin = 4,
                          wavelength_min = None, wavelength_max = None):
        '''
        read coupling map file and do key operations

        Parameters
        ----------
        couplingmap_file: str
            path to the coupling map file
        velzero: float
            wavelength of zero velocity (nm)
        fnumber: float
            f number where the coupling map is taken
        oversample_factor: float
            zoom factor to oversample coupling maps
        wavelength_nbin: int
            number of wavelength channels to average coupling maps
        wavelength_min: float
            minimum of wavelength range
        wavelength_max: float
            maximum of wavelength range
        
        '''
        self.cube = fits.open(couplingmap_file)[1].data
        self.cube_header = fits.open(couplingmap_file)[0].header
        self.cube_w = self.cube_header['MAS_W']
        self.cube_n = self.cube_header['NPT']

        
        self.wavs = wavelengths #self.wavmap[0] # reference index
        self.vels = (self.wavs - velzero) / velzero * 3e5 # km/s

        # truncate wavelength range if wavelength_min and wavelength_max are given
        wav_idx = np.where((self.wavs >= wavelength_min) & (self.wavs <= wavelength_max))[0]
        self.wavs = self.wavs[wav_idx]
        self.vels = self.vels[wav_idx]
        self.cube = self.cube[:, wav_idx, :, :]

        # copy the wavelengths before binning
        self.wavs0 = self.wavs.copy()
        self.vels0 = self.vels.copy()

        # bin wavelength to get better S/N (caution! this will blur the coupling map)
        effective_num = (len(self.wavs) // wavelength_nbin) * wavelength_nbin
        self.wavs = np.average(self.wavs[:effective_num].reshape((-1, wavelength_nbin)), axis=1)
        self.vels = np.average(self.vels[:effective_num].reshape((-1, wavelength_nbin)), axis=1)
        self.cube = np.average(self.cube[:,:effective_num].reshape((NFIB, -1, wavelength_nbin, self.cube_n, self.cube_n)), axis=2)


        print("wavelength range: ", self.wavs[0], self.wavs[-1], "number of wavelengths: ", len(self.wavs))
        self.nwav = len(self.wavs)

        self.oversample_factor = oversample_factor
        self.cube_oversampled = np.zeros((NFIB, len(self.vels), oversample_factor * self.cube_n, oversample_factor * self.cube_n))
        
        if oversample_factor != 1:
            from scipy.ndimage import zoom
            for fibind in range(NFIB):
                for specind in range(len(self.vels)):
                    self.cube_oversampled[fibind, specind, :, :] = zoom(self.cube[fibind, specind, :, :], oversample_factor)
        else:
            self.cube_oversampled = self.cube
        
        x_pos_mas = np.linspace(0 - self.cube_w/2, 0 + self.cube_w/2, oversample_factor * self.cube_n) #* zaber_microns
        y_pos_mas = np.linspace(0 - self.cube_w/2, 0 + self.cube_w/2, oversample_factor * self.cube_n) #* zaber_microns

        self.x_pos_mas = x_pos_mas
        self.y_pos_mas = y_pos_mas

        self.cube_resolution = self.x_pos_mas[1] - self.x_pos_mas[0]


        

def reshape_square(array):
    n = int(np.sqrt(len(array)))
    return array.reshape((n,n))
        

        




    
    # def resize_input_scene(self, scene:Scene, x_offset = 0, y_offset = 0)