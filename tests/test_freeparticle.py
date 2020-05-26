import numpy as np
from numpy import pi, sin, cos
from brownian_ot.particles import Sphere
from brownian_ot.simulation import FreeDiffusionSimulation, unbiased_rotation
from numpy.testing import assert_allclose

def test_setup():
    eta = 1e-3 # Pa s, water
    kT = 1.38e-23*295
    a = 1e-6

    particle = Sphere(a)
    sim = FreeDiffusionSimulation(particle, 1e-5, eta, kT, pos0 = np.zeros(3),
                                  orient0 = np.identity(3))
    sim.run(1000)



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
    


