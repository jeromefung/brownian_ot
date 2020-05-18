import numpy as np
import quaternion

from brownian_ot.particles.particle import Particle
from brownian_ot.utils import sphere_D

class Sphere(Particle):
    def __init__(self, a, refractive_index, viscosity, kT,
                 pos = np.zeros(3),
                 orient = quaternion.quaternion(1,0,0,0),
                 seed = None):
        self.a = a
        super().__init__(sphere_D(a, kT, viscosity),
                         np.zeros(3),
                         kT = kT,
                         refractive_index = refractive_index,
                         pos = pos, orient = orient, seed = seed)
    
