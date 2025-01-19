from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from .parameters import telescope_params, firstcam_params
from PLred.scene import Scene

diameter = telescope_params['diameter']
NFIB = firstcam_params['NFIB']
zaber_microns = firstcam_params['zaber_microns']
RESOLVING_POWER = firstcam_params['resolving_power']


def reshape_square(array):
    n = int(np.sqrt(len(array)))
    return array.reshape((n,n))


class PLSimulator:
    '''
    Simulate PL spectrum given the coupling map and the input scene
    '''

    # interpolation scheme for wavelength
    WAV_INTERP_KIND = 'cubic'

    # spectrograph's resolving power

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

        sigma = (3e5 / RESOLVING_POWER) / self.scene.dv / (2*np.sqrt(2*np.log(2)))

        convolved_spec = gaussian_filter(spec, sigma)
        downsampled_spec = np.interp(self.vels0, self.scene.vels, convolved_spec)
        return downsampled_spec