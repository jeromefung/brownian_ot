import numpy as np
from numpy import pi
from brownian_ot.particle import Particle

def test_setup():
    eta = 1e-3 # Pa s, water
    kT = 295
    a = 1e-6

    D_t = kT / (6*pi*eta*a)
    D_r = kT / (6*pi*eta*a**3)
    
    D_tensor = np.diag(np.array([D_t, D_t, D_t, D_r, D_r, D_r]))

    particle = Particle(D = D_tensor, cod = np.zeros(3),
                        f_ext = None, kT = kT)

    # set up simulation
    #sim = Simulation(particle, 'my_simulation',
    #                 initial_pos = np.array([0,0,0]),
    #                 initial_orient = np.identity(3),
    #                 n_steps = 1e6, dt = 1e-5, save_int = 1000)

    #sim.run()
