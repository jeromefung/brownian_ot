import numpy as np

from brownian_ot.particles.particle import Particle
from brownian_ot.utils import spheroid_D

class Spheroid(Particle):
    '''
    Spheroid whose axis of rotational symmetry is the z axis.
    '''
    def __init__(self, a, ar, refractive_index, viscosity, kT,
                 pos = np.zeros(3),
                 orient = quaternion.quaternion(1,0,0,0),
                 seed = None):
        '''
        a: radius in x/y
        ar: aspect ratio
        '''
        self.a = a
        self.ar = ar
        super().__init__(spheroid_D(a, a * ar, kT, viscosity),
                         np.zeros(3),
                         kT = kT,
                         refractive_index = refractive_index,
                         pos = pos, orient = orient, seed = seed)
    
