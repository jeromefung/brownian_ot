'''
Mock up the use case for running a Brownian dynamics simulation of particle
in OT.
'''

particle = Spheroid(a = 0.2e-6, ar = 1.5, n_p = 1.5,
                    kT = 295*1.38e-23, viscosity = 1e-3)

# use a NamedTuple for this?
# Python 3.7 has a DataClass object, which might actually be great for this,
# but comet doesn't have 3.7 nor do I.
# I'm less keen on using a straight dictionary, but I can't see the Beam
# objects having any non-trivial methods. (It's going to get passed to ott
# to create an ott beam object, where a lot of math happens)
beam = Beam(lambda_0 = 1064e-9,
            pol = np.array([1, 1j]),
            NA = 1.2,
            medium_index = 1.33,
            power = 5e-3)

# the main reason to make Simulation a class is to allow for inheritance
# could have a base Simulation class
# and also a FreeDiffusionSim that subclasses that,
# and a OTSimulation class
# By subclassing Simulation, and attaching a particular force model to it,
# the user doesn't need to explicitly construct the force.

sim = OTSimulation(particle, beam,
                   timestep = 1e-5, seed = 12345678,
                   pos_0 = np.array([0, 0, 1.04103e-7]))

sim.run(1000000, 'my_simulation_results.npy')


# Also, mock up a force calculation
force_func = make_force(particle, beam)
zs = np.linspace(-2e-6, 2e6, 201)
f_z = calc_fz(zs, force_func)
zeq = find_zeq(force_func)
xs = np.linspace(-3e-6, 3e-6, 301)
f_x = calc_fx(xs, force_func, z = zeq)
