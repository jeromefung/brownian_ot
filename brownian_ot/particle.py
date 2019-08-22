import numpy as np

class Particle:
    '''
    Stores information about particle being simulated.
    '''

    def __init__(self, D, cod, f_ext, kT):
        self.D = D
        self.cod = cod
        self.f_ext = f_ext
        self.kT = kT


    def _q_random(self):
        '''
        Calculate random generalized displacement obeying generalized
        Stokes-Einstein relation (up to factor of \Delta t). Recall
        <q_i q_j> = 2D_{ij} \Delta t.
        
        Return array of random displacements.
        '''
        return 2. * np.random.multivariate_normal(np.zeros(6), self.D)

    
    def update(self, dt):
        # calc q^B in particle frame
        q_B = _q_random(self)
        
        # calc q^ext in lab frame
        #     calc force, torque at com
        #     correct torque to cod
        # convert q^ext to particle frame
        # convert total generalized displacement to lab frame
        # update cod position
        # update orientation
        pass

    
