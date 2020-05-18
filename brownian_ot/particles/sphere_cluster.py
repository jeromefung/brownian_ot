import numpy as np
import quaternion
from .particle import Particle
from brownian_ot.utils import dimer_D

class SphereCluster():
    '''
    Base class -- or is there a better way to do this?
    Want to have arbitrary user-defined SphereCluster objects
    where users specify positions and a diffusion tensor,
    but have subclasses for common cases.
    '''
    pass


class Dimer(Particle, SphereCluster):
    def __init__(self, a, refractive_index, viscosity, kT,
                 pos = np.zeros(3),
                 orient = quaternion.quaternion(1,0,0,0),
                 seed = None):
        self.a = a
        self.n_spheres = 2
        # Reference configuration in radius units
        self.sphere_pos = np.array([[0, 0, 1],
                                      [0, 0, -1]])
        
        super().__init__(dimer_D(a, kT, viscosity),
                         np.zeros(3),
                         kT = kT,
                         refractive_index = refractive_index,
                         pos = pos, orient = orient, seed = seed)
