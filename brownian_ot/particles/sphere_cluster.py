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
    def __init__(self, sphere_pos, Ddim, cod, a, n_p = None):
        '''
        Must manually specify diffusion tensor and center of diffusion.

        Parameters
        ----------
        sphere_pos : ndarray (n, 3)
            Array of particle positions in reference configuration,
            one row per particle.
        '''
        self.sphere_pos = sphere_pos
        self.n_spheres = self.sphere_pos.shape[0]
        self.a = a
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
        
        
