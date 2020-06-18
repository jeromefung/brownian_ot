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
    return kT * diffusion_tensor 


def dimer_D(a, kT, eta):
    '''
    Calculate the diffusion tensor for a symmetric dimer using the
    Nir & Acrivos analytic solution for the friction tensor (see Appendix of
    J. Fluid Mech. 1973, pp. 209-223).
    '''

    # Friction tensor needs to be scaled by pi * eta * a, see Table 2 of paper
    # So, diffusion tensor needs to be scaled by kT/(pi*eta*a)

    # Numbers in Table 2; come from horrible integrals of bessel functions
    fxx = 8.691
    fzz = 8.691 - 0.951
    txx = 29.92
    tzz = 29.92 - 15.50

    diffusion_tensor = np.diag(np.array([1/fxx, 1/fxx, 1/fzz,
                                         1/txx, 1/txx, 1/tzz]))
    # Scale
    scale_factor = kT / (pi * eta) *np.diag(np.concatenate((1/a * np.ones(3),
                                                            1/a**3 * np.ones(3))))
    return scale_factor * diffusion_tensor


    
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
