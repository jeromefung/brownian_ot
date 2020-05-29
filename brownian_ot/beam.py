import numpy as np
from collections import namedtuple


class Beam:
    '''
    Stores information about incident trapping beam in optical tweezers 
    calculations.
    
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
        self.NA = NA
        self.n_med = n_med
        self.power = power
        # Note: ott_beam.m sets beam power to 1 upon creation of ott beam object
        # So polarization vector doesn't need to be normalized.
        


