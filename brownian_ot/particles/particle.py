import numpy as np
import quaternion

from numpy import sin, cos
from brownian_ot.utils import sphere_D, spheroid_D

class Particle:
    '''
    Base class that stores information about particle being simulated.

    Diffusion tensor is scaled such that :math:`kT / \\eta` is factored out.
    This way, what's stored only depends on the particle geometry.
    '''

    def __init__(self, Ddim, cod, n_p = None):
        '''
        Parameters
        ----------
        Ddim : ndarray, float (6 x 6)
            Diffusion tensor in units of 1/length, with :math:`kT/\\eta` 
            factored out
        cod : ndarray, float (3)
            Particle center of diffusion
        n_p : complex
            Particle refractive index. Imaginary component denotes absorption.
        '''
        self.Ddim = Ddim
        self.cod = cod
        self.n_p = n_p

        # Particle to have position and orientation attributes, but these
        # shouldn't be public-facing

        self._pos = np.zeros(3)
        self._orient = quaternion.quaternion(1,0,0,0) # identity

    
    def _nice_output(self):
        return np.concatenate((self._pos,
                               quaternion.as_float_array(self._orient)))


class Sphere(Particle):
    '''
    A sphere.

    Attributes
    ----------
    a : float
        Particle radius

    '''
    def __init__(self, a, n_p = None):
        '''
        Parameters
        ----------
        a : float
            Particle radius.
        n_p : complex, optional
            Particle refractive index.
        '''
        self.a = a
        super().__init__(sphere_D(a, 1, 1), np.zeros(3), n_p = n_p)


class Spheroid(Particle):
    '''
    A spheroid whose default orientation has the axis of rotational
    symmetry along the z-axis.

    Attributes
    ----------
    a : float
       Radius in direction perpendicular to symmetry axis.
    ar : float
       Aspect ratio. Radius along symmetry axis is a * ar. ar > 1 for a 
       prolate spheroid, and ar < 1 for an oblate spheroid.
    '''

    def __init__(self, a, ar, n_p = None):
        '''
        Parameters
        ----------
        a : float
            Radius in direction perpendicular to symmetry axis.
        ar : float
            Aspect ratio.
        n_p : complex
            Particle refractive index.

        '''
        self.a = a
        self.ar = ar
        super().__init__(spheroid_D(a, a * ar, 1, 1),
                         np.zeros(3), n_p = n_p)

    @property
    def equivalent_sphere_radius(self):
        return (self.a**3 * self.ar)**(1/3)
        
