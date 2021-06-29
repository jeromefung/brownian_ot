import numpy as np
from collections import namedtuple


class Beam:
    '''
    Stores information about incident trapping beam (assumed to be Gaussian)
    in optical tweezers calculations.
    
    This class interfaces with the ott beam object.
    '''

    def __init__(self, wavelen, pol, NA, n_med, power):
        '''
        Parameters
        ----------
        wavelen : float
            Incident wavelength in vacuum.
        pol : list, tuple, or ndarray (2)
            Jones vector describing incident polarization state.
            Can be complex if polarization is elliptical.
        NA : float
            Objective lens numerical aperture.
        n_med : float
            Refractive index of medium surrounding particle.
        power : float
            Incident beam power.
        '''
        self.wavelen = wavelen
        self.pol = list(pol) # matlab doesn't play nicely with ndarrays
        for idx, elt in enumerate(self.pol):
            if elt.__class__ == np.complex128:
                self.pol[idx] = complex(elt)
                # matlab doesn't like np complex type
                # This is a problem if using functions like np.exp()
        self.NA = NA
        self.n_med = n_med
        self.power = power
        # Note: ott_beam.m sets beam power to 1 upon creation of ott beam object
        # So polarization vector doesn't need to be normalized.
        

class LGBeam(Beam):
    '''
    Class to store information about an arbitrary Laguerre-Gaussian beam.
    '''
    def __init__(self, mode, wavelen, pol, NA, n_med, power):
        '''
        Parameters
        ----------

        mode : list (2)
            Mode indices [p, l]. The radial index p is the number of 
            radial nodes. The azimuthal index l indicates how many times
            the optical phase advances by 2 pi radians.

        '''
        self.type = 'LG' # future-proof if we add Hermite-Gaussian in the future
        self.mode = mode
        super().__init__(wavelen, pol, NA, n_med, power)
        
