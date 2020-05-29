import numpy as np
from numpy.testing import assert_allclose

import brownian_ot
from brownian_ot.beam import Beam
from brownian_ot.particles import Sphere, Spheroid, Dimer, SphereCluster
from brownian_ot.simulation import OTSimulation
from brownian_ot.ott_wrapper import make_ott_force
from brownian_ot.utils import sphere_D
from brownian_ot.force_utils import calc_fz


spheroid = Spheroid(a = 0.2e-6, ar = 1.5, n_p = 1.5)
sphere = Sphere(a = 0.6e-6, n_p = 1.45+0.01j)
dimer = Dimer(0.5e-6, 1.4)
small_dimer = Dimer(0.015e-6, 1.4)

beam = Beam(wavelen = 1064e-9, pol = [1, 1j], NA = 1.2,
            n_med = 1.33, power = 5e-3)

def test_spheroid_ot():
    sim = OTSimulation(spheroid, beam, timestep = 1e-5,
                       viscosity = 1e-3, kT = 295 * 1.38e-23,
                       seed = 12345678,
                       pos0 = np.array([0, 0, 1e-7]),
                       orient0 = np.identity(3))
    sim.run(100)

def test_dimer():
    sim = OTSimulation(dimer, beam, timestep = 1e-5,
                       viscosity = 1e-3, kT = 295 * 1.38e-23,
                       seed = 12345678,
                       pos0 = np.array([0, 0, -1e-7]),
                       orient0 = np.identity(3))
    sim.run(100)

    
def test_dimer_force_calc():
    '''
    Check force calculations for a Rayleigh-sized dimer.
    In particular, check that F_z for the dimer is close to that obtained
    for a sphere of the same volume.
    '''
    dimer_force_func = make_ott_force(small_dimer, beam)
    zs = np.linspace(-2e-6, 2e-6, 101)
    dimer_fz = calc_fz(zs, dimer_force_func)
    equiv_sphere = Sphere(a = small_dimer.equivalent_sphere_radius,
                          n_p = small_dimer.n_p)
    equiv_sphere_force_func = make_ott_force(equiv_sphere, beam)
    sphere_fz = calc_fz(zs, equiv_sphere_force_func)
    assert_allclose(dimer_fz, sphere_fz, atol = 1e-15)

    
def test_sphere_ott_mstm():
    '''
    Compare optical force calculated using ott-generated sphere T-matrix
    to force calculated using mstm-generated T-matrix.
    '''
    ott_force_func = make_ott_force(sphere, beam)
    ott_force = ott_force_func(np.zeros(3), np.identity(3))

    # Manually create a 1-sphere "cluster"
    pos = np.array([np.zeros(3)])
    cluster = SphereCluster(pos, sphere.Ddim, np.zeros(3),
                            sphere.a, sphere.n_p)
    mstm_force_func = make_ott_force(cluster, beam)
    mstm_force = mstm_force_func(np.zeros(3), np.identity(3))

    assert_allclose(mstm_force, ott_force, atol = 1e-15)
    # Setting atol is necessary because of roundoff of floats that are nearly 0


    
