import numpy as np

import quaternion
from .particle import Particle
from brownian_ot.utils import dimer_D

class SphereCluster(Particle):
    '''
    Base class -- or is there a better way to do this?
    Want to have arbitrary user-defined SphereCluster objects
    where users specify positions and a diffusion tensor,
    but have subclasses for common cases.
    '''
    def __init__(self, sphere_pos, Ddim, cod, a, n_p = None, a_ratios = None):
        '''
        Must manually specify diffusion tensor and center of diffusion.

        Parameters
        ----------
        sphere_pos : array-like (n, 3)
            Array of particle positions in reference configuration,
            in units of particle radius a, one row per particle.
        Ddim : ndarray, float (6 x 6)
            Diffusion tensor in units of 1/length, with :math:`kT/\\eta`
            factored out.
        cod : ndarray, float (3)
            Particle center of diffusion
        a : float
            Dimensional particle radius. If a_ratios set, then dimensional
            particle radii are given by a * a_ratios.
        n_p : complex or array-like (n), optional
            Particle refractive index or array-like collection of
            refractive indices.
        a_ratios : array-like (n), optional
            Non-dimensionalized particle radii (divided by a). If None,
            all spheres have the same radius a (which would be equivalent to
            setting all values in a_ratios to 1).
        '''
        self.sphere_pos = np.asarray(sphere_pos)
        self.n_spheres = self.sphere_pos.shape[0]
        self.a = a

        if a_ratios is not None:
            self.a_ratios = a_ratios
            if np.isscalar(n_p) == False:
                if len(a_ratios) != len(n_p):
                    raise ValueError("a_ratios and n_p arrays must have the same length.")

        super().__init__(Ddim, cod, n_p)

    @property
    def equivalent_sphere_radius(self):
        return (self.n_spheres * self.a**3)**(1/3)


class Dimer(SphereCluster):
    def __init__(self, a, n_p = None):
        # Reference configuration in radius units
        # Oriented along z axis
        sphere_pos = np.array([[0, 0, 1],
                               [0, 0, -1]])

        super().__init__(sphere_pos, dimer_D(a, 1, 1), np.zeros(3), a,
                         n_p)
