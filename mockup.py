'''
Mock up the use case for running a Brownian dynamics simulation of particle
in OT.
'''

# Use case #1: Brownian OT for a spheroid
particle1 = Spheroid(a = 0.2e-6, ar = 1.5, n_p = 1.5)
# Spheroids could have some methods like equiv_sphere_radius
# or, a sphere cluster might have some specific attributes or similar methods
# particle needs its own center of diffusion.
# But I'm less clear that Particle should have a position or orientation
# attribute

# TODO: make Beam a class.
# This way, init can make beam.pol a "legal" python list whether you pass
# in a list, a tuple, or an ndarray
beam = Beam(wavelen = 1064e-9,
            pol = np.array([1, 1j]), # This should be legal!
            NA = 1.2,
            n_med = 1.33,
            power = 5e-3)

sim = OTSimulation(particle1, beam, timestep = 1e-5,
                   viscosity = 1e-3,
                   kT = 295 * 1.38e-23, seed = 1234578,
                   pos0 = np.array([0, 0, 1.04103e-7]),
                   orient0 = np.identity(3))

# It's only when simulating that kT matters.
# Ditto for viscosity, or rng seed, or initial position/orientation
# Or maybe the particle should have its own position, orientation
# because they're really a property of the particle.
# But I would make them private.

sim.run(1000000, 'my_simulation_results')



# Use case #2: force calculation
force_func = make_ott_force(particle, beam) # associated w/specific particle and beam
zs = np.linspace(-2e-6, 2e6, 201)
f_z = calc_fz(zs, force_func) # TODO: implement convenience function
zeq = find_zeq(force_func) # TODO: implement convenience function
xs = np.linspace(-3e-6, 3e-6, 301)
f_x = calc_fx(xs, force_func, z = zeq) # TODO: implement convenience function


# Use case #3: a simulation with free diffusion
particle2 = Sphere(a = particle1.equivalent_radius())
# have a default refractive index of None, raise ValueError?
sim2 = FreeDiffusionSim(particle2, timestep = 5e-4, viscosity = 1e-3,
                        kT = 295 * 1.38e-23, seed = 987654321)
# default position0, orient0
traj = sim2.run(10000) # don't save this









# use a NamedTuple for this.
# Python 3.7 has a DataClass object, which might actually be great for this,
# but comet doesn't have 3.7 nor do I.
# A dictionary with arbitrary keys seems less appropriate.
# But a class seems like overkill: I don't think this will have any
# non-trivial methods. It's going to get passed to ott
# to create an ott beam object, where a lot of math happens)

# The main reason to make Simulation a class is to allow for inheritance.
# There is a base Simulation class
# and also a FreeDiffusionSim subclass,
# and a OTSimulation class.
# By subclassing Simulation, and attaching a particular force model to it,
# the user doesn't need to explicitly construct the force.





