'''
Utility functions for Brownian dynamics simulations
'''

import numpy as np
from numpy import sqrt, log, arctan, pi, sin, cos

def sphere_D(a, kT, eta):
    '''
    Calculate diffusion tensor for a sphere.

    Parameters
    ----------
    a : float
        Particle radius.
    kT : float
        Temperature in energy units.
    eta : float
        Solvent viscosity

    Returns
    -------
    array_like
        Diffusion tensor
    '''
    D_t = 1/(6*pi*eta*a) # put in kT later if nonzero
    D_r = 1/(8*pi*eta*a**3)
    diffusion_tensor = np.diag([D_t, D_t, D_t, D_r, D_r, D_r])

    if kT == 0:
        return diffusion_tensor # with kT factored out
    else:
        return kT * diffusion_tensor 

    
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


def dimer_D(a, kT, eta):
    '''
    Implement Nir/Acrivos analytic solution.
    '''
    # TODO: dummy implementation for now
    return np.identity(6)
    
def rot_x(theta):
    '''
    Calculate rotation matrix for rotation by theta about the x-axis.
    Rotation angle theta is positive clockwise when viewed from the origin.
    '''
    return np.array([[1, 0, 0],
                     [0, cos(theta), -sin(theta)],
                     [0, sin(theta), cos(theta)]])
    
def rot_y(theta):
    '''
    Calculate rotation matrix for rotation by theta about the y-axis.
    Rotation angle theta is positive clockwise when viewed from the origin.
    '''
    return np.array([[cos(theta), 0, sin(theta)],
                     [0, 1, 0],
                     [-sin(theta), 0, cos(theta)]])

def rot_z(theta):
    '''
    Calculate rotation matrix for rotation by theta about the z-axis.
    Rotation angle theta is positive clockwise when viewed from the origin.
    '''
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])
