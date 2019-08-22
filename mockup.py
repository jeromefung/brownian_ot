'''
Mock up the use case for running a Brownian dynamics simulation of particle
in OT.
'''

import numpy as np

# set up external force
# have ExternalForce be the base class
# subclasses could be OpticalForce, FreeParticle
# OpticalForce gets its own constructor, etc.
ext_force = OpticalForce(tmatrix, power, n_med)


# set up particle
D_tensor = np.load('my_diffusion_tensor.npy')
particle = Particle(D = D_tensor, cod = np.array([0,0,0]),
                    f_ext = ext_force, kT = 295.)

# set up simulation
sim = Simulation(particle, ext_force, 'my_simulation',
                 initial_pos = np.array([0,0,0]),
                 initial_orient = np.identity(3),
                 n_steps = 1e6, dt = 1e-5, save_int = 1000)

sim.run()
