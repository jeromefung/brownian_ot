import numpy as np
from numpy import pi, sin, cos, exp
from brownian_ot.particles import Sphere, Spheroid
from brownian_ot.simulation import FreeDiffusionSimulation, unbiased_rotation
from numpy.testing import assert_allclose
from brownian_ot.analysis import calc_msd, calc_axis_autocorr


eta = 1e-3 # Pa s, water
kT = 1.38e-23*295

def test_setup():
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
    

def test_spheroid_diffusion():
    '''
    Test angular quantities from the diffusion of a spheroid.
    '''
    a = 2e-8 # 20 nm minor radius
    ar = 5
    spheroid = Spheroid(a, ar)
    dt = 1e-5
    n_steps = 5000
    sim = FreeDiffusionSimulation(spheroid, dt, eta, kT, seed = 987654321,
                                  pos0 = np.zeros(3), orient0 = np.identity(3))
    traj = sim.run(n_steps)
    clstr_msd_x, clstr_msd_y, clstr_msd_z = calc_msd(traj, max_steps = 5,
                                                     particle_frame = True)
    axis_x, axis_y, axis_z = calc_axis_autocorr(traj, max_steps = 5)

    D = np.diag(spheroid.Ddim * kT / eta)
    tbase = (np.arange(5) + 1) * dt
    # There's going to be statistical fluctuation in the MSDs and
    # axis autocorrelations.
    # Roughly, expect at least n_steps / 5 independent measurements,
    # so a fractional uncertainty of 1/sqrt(n_steps/5)
    estimated_tol = 1/np.sqrt(n_steps/5) * 1.2 # fudge factor
    
    assert_allclose(clstr_msd_x, 2 * D[0] * tbase, rtol = estimated_tol)
    assert_allclose(clstr_msd_y, 2 * D[1] * tbase, rtol = estimated_tol)
    assert_allclose(clstr_msd_z, 2 * D[2] * tbase, rtol = estimated_tol)
    assert_allclose(axis_x, exp(-(D[4] + D[5])*tbase), rtol = estimated_tol)
    assert_allclose(axis_y, exp(-(D[3] + D[5])*tbase), rtol = estimated_tol)
    assert_allclose(axis_z, exp(-(D[3] + D[4])*tbase), rtol = estimated_tol)

