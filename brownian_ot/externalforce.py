import numpy as np
import quaternion

def free_particle(*args):
    return np.zeros(6)


def make_constant_force(f):
    '''
    Factory function to return a callable constant generalized force,
    which is a 6-element vector (forces and torques) with those dimensions.
    '''

    def force(*args):
        return f

    return force


