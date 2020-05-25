import numpy as np

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

                                
