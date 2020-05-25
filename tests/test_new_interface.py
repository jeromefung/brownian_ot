import numpy as np
import brownian_ot
from brownian_ot.beam import Beam
from brownian_ot.particles import Spheroid
from brownian_ot.simulation import OTSimulation

spheroid = Spheroid(a = 0.2e-6, ar = 1.5, n_p = 1.5)
beam = Beam(wavelen = 1064e-9, pol = [1, 1j], NA = 1.2,
            n_med = 1.33, power = 5e-3)

def test_spheroid_ot():
    sim = OTSimulation(spheroid, beam, timestep = 1e-5,
                       viscosity = 1e-3, kT = 295 * 1.38e-23,
                       seed = 12345678,
                       pos0 = np.array([0, 0, 1e-7]),
                       orient0 = np.identity(3))
    sim.run(100)
