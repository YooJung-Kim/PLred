import numpy as np

class SimulatedDetector:

    gain_volts = 1
    gain_amplifier = 10
    dark_level = 0.01

    def __init__(self, shape, bias_level = 200, QE_level = 0.8, RN_level = 0.3):

        self.shape = shape

        self.bias_map = np.ones(shape) * bias_level # volts
        self.QE_map = np.ones(shape) * QE_level # unitless (0-1)
        self.RN_map = np.ones(shape) * RN_level # volts
    
    def add_DSNU(self, sig = 0.06):
        '''
        Add dark signal non-uniformity to the detector.
        sig: standard deviation of the noise. (volts)
        '''
        self.bias_map += np.random.normal(0, sig, self.shape)
    
    def add_PRNU(self, sig = 0.06):
        '''
        Add pixel response non-uniformity to the detector.
        sig: standard deviation of the noise. (unitless)
        '''
        self.QE_map += np.random.normal(0, sig, self.shape)
        self.QE_map[self.QE_map < 0] = 0
        self.QE_map[self.QE_map > 1] = 1

    def add_RN_non_uniformity(self, sig = 0.06, skew = 0):
        '''
        Add read noise non-uniformity to the detector.
        sig: standard deviation of the noise. (volts)
        '''
        self.RN_map += np.random.normal(0, sig, self.shape)
        # self.RN_map = 
    
    def store_nonlinearity(self, nonlinearity_func):
        '''
        Store the nonlinearity function.
        nonlinearity_func: function that takes a 2D array of volts and returns a 2D array of volts.
        '''
        self.nonlinearity_func = nonlinearity_func
    
    def illuminate(self, illumination_map):
        '''
        Illuminates the detector with the given illumination map.
        illumination_map: 2D array of the same shape as the detector. (photons)
        '''
        self.illumination_map = illumination_map

        # shot noise
        true_charges = np.random.poisson(illumination_map)

        # dark noise
        true_charges += np.random.poisson(np.ones_like(illumination_map) * self.dark_level)

        # apply quantum efficiency
        electrons = np.random.binomial(true_charges, self.QE_map).astype(float)

        # apply gain
        volts = electrons * self.gain_volts

        # apply read noise
        volts += np.random.normal(0, self.RN_map)

        # apply amplifier
        volts *= self.gain_amplifier

        # apply bias
        volts += self.bias_map

        # apply nonlinearity
        if self.nonlinearity_func is not None:
            volts = self.nonlinearity_func(volts)

        # apply ADC
        count = np.trunc(volts)

        self.count = count
        self.true_charges = true_charges

        return count
    
    def illuminate_single_pixel(self, pix_x, pix_y, illumination):
        '''
        Illuminates a single pixel with the given illumination.
        pix_x, pix_y: pixel coordinates
        illumination: number of photons
        '''

        # shot noise
        true_charges = np.random.poisson(illumination)

        # dark noise
        true_charges += np.random.poisson(self.dark_level)

        # apply quantum efficiency
        electrons = float(np.random.binomial(true_charges, self.QE_map[pix_x, pix_y]))

        # apply gain
        volts = electrons * self.gain_volts

        # apply read noise
        volts += np.random.normal(0, self.RN_map[pix_x, pix_y])

        # apply amplifier
        volts *= self.gain_amplifier

        # apply bias
        volts += self.bias_map[pix_x, pix_y]

        # apply nonlinearity
        if self.nonlinearity_func is not None:
            volts = self.nonlinearity_func(volts)

        # apply ADC
        count = np.trunc(volts)

        return count

    def get_distribution(self, pix_x, pix_y, illum, n = 1000):

        counts = []
        for i in range(n):
            count = self.illuminate_single_pixel(pix_x, pix_y, illum)
            counts.append(count)

        return np.array(counts)