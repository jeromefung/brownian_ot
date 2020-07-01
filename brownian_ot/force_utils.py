import numpy as np
from .utils import rot_x, rot_y
import scipy
from scipy.optimize import brentq

# TODO: too much code duplication in this module

def calc_fz(zs, force_function):
    '''
    Return z component of optical force for particle at x = 0, y = 0
    and in standard orientation at different z.
    '''
    # A loop is needed because of the matlab interface
    if isinstance(zs, (list, np.ndarray)):
        output = np.array([force_function(np.array([0, 0, z]),
                                          np.identity(3))[2] for z in zs])
    else: # a scalar
        output = force_function([0, 0, zs], np.identity(3))[2]

    return output


def calc_fx(xs, force_function, z = 0):
    '''
    Return x component of optical force for a particle at y = 0, z = z
    and in standard orientation at different x.

    To explore forces near equilibrium, specify the equilibrium z height.
    '''
    # A loop is needed because of the matlab interface
    if isinstance(xs, (list, np.ndarray)):
        output = np.array([force_function(np.array([x, 0, z]),
                                          np.identity(3))[0] for x in xs])
    else: # a scalar
        output = force_function([xs, 0, z], np.identity(3))[0]

    return output


def calc_fy(ys, force_function, z = 0):
    '''
    Return y component of optical force for a particle at x = 0, z = z
    and in standard orientation at different y.

    To explore forces near equilibrium, specify the equilibrium z height.
    '''
    # A loop is needed because of the matlab interface
    if isinstance(ys, (list, np.ndarray)):
        output = np.array([force_function(np.array([0, y, z]),
                                          np.identity(3))[1] for y in ys])
    else: # a scalar
        output = force_function([0, ys, z], np.identity(3))[1]

    return output


def calc_tx(thetas, force_function, z = 0):
    '''
    Return x component of optical torque for a particle at x = 0, y = 0,
    z = z rotated by by different angles about the x axis.

    thetas should be in radians.
    '''
    # A loop is needed because of the matlab interface
    if isinstance(thetas, (list, np.ndarray)):
        output = np.array([force_function(np.array([0, 0, z]),
                                          rot_x(theta))[3] for theta in thetas])
    else: # a scalar
        output = force_function([0, 0, z], rot_x(thetas))[3]

    return output


def calc_ty(thetas, force_function, z = 0):
    '''
    Return y component of optical torque for a particle at x = 0, y = 0,
    z = z rotated by by different angles about the y axis.

    thetas should be in radians.
    '''
    # A loop is needed because of the matlab interface
    if isinstance(thetas, (list, np.ndarray)):
        output = np.array([force_function(np.array([0, 0, z]),
                                          rot_y(theta))[4] for theta in thetas])
    else: # a scalar
        output = force_function([0, 0, z], rot_y(thetas))[4]

    return output




def find_zeq(force_function, guess = 0, window_size = 0.5e-6):
    '''
    Use a numerical root finder to find z such that F_z = 0 on a particle.

    Root finder uses Brent's algorithm, which searches for the root of a 1D
    function within an interval over which the function changes sign.
    Here, the window is centered at `guess`, extends from 
    `guess` - `window_size` to `guess` + `window_size`. 

    Parameters
    ----------
    force_function : callable
        Function returning generalized force on particle given position and
        orientation.
    guess : float, optional
        Center of bracketing interval.
    window_size : float, optional
        Half-width of bracketing interval. 

    Returns
    -------
    z_eq : float
        Equilibrium z position.
    '''
    return brentq(calc_fz, guess - window_size, guess + window_size,
                  args = (force_function))
