import numpy as np

class SphereCluster(Particle):
    '''
    abstract base class?
    '''
    pass


class Dimer(Particle):
    def __init__(self, a, refractive_index, viscosity, kT,
                 pos = np.zeros(3),
                 orient = quaternion.quaternion(1,0,0,0),
                 seed = None):
        super().__init__(dimer_D(a, kT, viscosity),
                         np.zeros(3),
                         kT = kT,
                         refractive_index = refractive_index,
                         pos = pos, orient = orient, seed = seed)
