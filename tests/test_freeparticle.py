import numpy as np
from numpy import pi, sin, cos
from brownian_ot.particle import Particle, unbiased_rotation
from numpy.testing import assert_allclose

def test_setup():
    eta = 1e-3 # Pa s, water
    kT = 295
    a = 1e-6

    D_t = kT / (6*pi*eta*a)
    D_r = kT / (6*pi*eta*a**3)
    
    D_tensor = np.diag(np.array([D_t, D_t, D_t, D_r, D_r, D_r]))

    particle = Particle(D = D_tensor, cod = np.zeros(3),
                        f_ext = None, kT = kT)
    #particle.update(1e-4)

    # set up simulation
    #sim = Simulation(particle, 'my_simulation',
    #                 initial_pos = np.array([0,0,0]),
    #                 initial_orient = np.identity(3),
    #                 n_steps = 1e6, dt = 1e-5, save_int = 1000)

    #sim.run()
    return


def test_unbiased_rotation():
    '''
    Check implementation of unbiased infinitesimal rotation matrices
    (Beard & Schlick).
    '''
    # check pure small rotations about x, y, and z axes (individually)
    thet_x = 1e-3
    gold_x = np.array([[1, 0, 0],
                       [0, cos(thet_x), -sin(thet_x)],
                       [0, sin(thet_x), cos(thet_x)]])
    assert_allclose(unbiased_rotation(thet_x, 0, 0), gold_x)

    thet_y = -1.5e-3
    gold_y = np.array([[cos(thet_y), 0, sin(thet_y)],
                       [0, 1, 0],
                       [-sin(thet_y), 0, cos(thet_y)]])
    assert_allclose(unbiased_rotation(0, thet_y, 0), gold_y)

    thet_z = 5e-4
    gold_z = np.array([[cos(thet_z), -sin(thet_z), 0],
                       [sin(thet_z), cos(thet_z), 0],
                       [0, 0, 1]])
    assert_allclose(unbiased_rotation(0, 0, thet_z), gold_z)
    
    # calculate a small rotation about all 3 axes at once
    # infinitesimal rotations commute, so see if explicit rotation matrices
    # multiplied together come close to the single unbiased matrix
    assert_allclose(unbiased_rotation(thet_x, thet_y, thet_z),
                    np.matmul(gold_z, np.matmul(gold_y, gold_x)),
                    atol = 1e-6, rtol = 1e-6) # they aren't exactly equal
    


