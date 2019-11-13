'''
Utility functions for Brownian dynamics simulations
'''

import numpy as np
from numpy import sqrt, log, arctan, pi

def spheroid_D(a, c, kT, eta):
    '''
    Calculate diffusion tensor for a spheroid whose axis of rotational 
    symmetry is the z axis following Perrin, Journal de Physique 1934.

    a: radius in x/y
    c: radius in z
    kT: thermal energy
    eta: viscosity

    A prolate spheroid obtains for c>a, oblate for c<a.
    '''
    # note that comparing to perrin's paper, his b is our a, and
    # his a is our c.

    # result of elliptic integral depends on prolate/oblate
    if c >= a: # prolate, eq. 10
        S = 2./sqrt(c**2 - a**2) * log((c + sqrt(c**2 - a**2))/a)
    else: # oblate, eq. 10 bis
        S = 2./sqrt(a**2 - c**2) * arctan(sqrt(a**2 - c**2)/a)
    # corner case: code does not correctly pass to spherical limit a = c
        
    # translational friction    
    f_xy = 32*pi*eta*(c**2 - a**2)/((2*c**2 - 3*a**2)*S + 2*c)
    f_z = 16*pi*eta*(c**2 - a**2)/((2*c**2 - a**2)*S - 2*c)

    # rotational friction
    c_xy = 32*pi/3 * eta * (c**4 - a**4) / ((2*c**2-a**2)*S - 2*c)
    c_z = 32*pi/3 * eta * (c**2 - a**2)*a**2 / (2*c - a**2*S)

    # D = kT / f, but put in kT at the end
    diffusion_tensor = np.diag(np.array([1/f_xy, 1/f_xy, 1/f_z,
                                         1/c_xy, 1/c_xy, 1/c_z]))
    if kT == 0:
        return diffusion_tensor # divide out kT
    else:
        return kT * diffusion_tensor # normally, correct units
    
