import numpy as np
from numpy import pi
from numpy.testing import assert_allclose

import brownian_ot
from brownian_ot.particles import Sphere
from brownian_ot.simulation import ConstantForceSimulation
from brownian_ot.analysis import expand_trajectory

sphere = Sphere(a = 1e-6)
eta = 1e-3
v_term = 1e-5 # m/s, or 10 microns/second
# Calculate F_0 to give a 10 micron/second terminal velocity
F_0 = 6 * pi * eta * sphere.a * v_term

def test_athermal():
    traj_len = 1000
    dt = 1e-4
    sim = ConstantForceSimulation(sphere, dt,
                                  np.array([0, 0, -F_0, 0, 0, 0]),
                                  eta, 0, pos0 = np.zeros(3),
                                  orient0 = np.identity(3))

    traj = sim.run(traj_len)
    traj = expand_trajectory(traj)
    expected_pos = np.array([0, 0, -v_term * traj_len * dt, 1, 0, 0,
                             0, 1, 0, 0, 0, 1])
    assert_allclose(traj[-1], expected_pos)
    
    
