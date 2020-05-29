.. _ot_tutorial:

Calculations With Optical Forces
================================

Here, we'll discuss calculations involving optical forces. First, we'll
build on :ref:`bd_tutorial` by considering Brownian dynamics simulations
in which particles experience an external force due to optical tweezers.
Then, we'll consider calculations of optical forces at fixed positions and
orientations. (These latter calculations could be done using the ott
package in Matlab, but brownian_ot provides a consistent interface particularly
for particles like sphere clusters that are not natively supported by ott.)


Brownian dynamics in optical tweezers
-------------------------------------

Let's simulate a 1-micrometer-diameter colloidal sphere. Once again, begin
with the necessary imports:

.. testcode::

   import numpy as np
   from brownian_ot import Beam
   from brownian_ot.particles import Sphere
   from brownian_ot.simulation import OTSimulation

We need a way to tell brownian_ot about the optical details of the system
we're simulating. This is done by defining a `Beam` object:

.. testcode::

   beam = Beam(wavelen = 1064e-9, pol = [1, 1j], NA = 1.2, n_med = 1.33,
               power = 5e-3)

The `wavelen` argument is the beam's incident vacuum wavelength. Here, we're
working in SI units (although brownian_ot can work in any consistent set of
units), so this is a 1064-nm-wavelength beam (such as can be obtained from
some diode lasers or a Nd:YAG laser). The incident beam's polarization is
specified by the `pol` argument, which can be a 2-element list or ndarray
corresponding to the beam's Jones vector. (The Jones vector, which in this case
represents right circularly polarized light, does not need to be normalized.)
If we wanted to use, for example, a beam with linear polarization along the
:math:`x` axis, we would instead specify :code:`pol = [1, 0]`.
The `NA` argument is the numerical aperture of the microscope objective lens
that focuses the incident beam. (Following the convention in microscopy, the
numerical aperture is defined as :math:`n_{im}\sin\alpha`, where :math:`n_{im}`
is the refractive index of the lens's immersion medium, if present, and
:math:`\alpha` is half of the lens's acceptance angle.)
The refractive index of the medium surrounding the particle (in this case,
water) is given by `n_med`.
Finally, the incident beam's power is given by `power`.

The focus of a `Beam` object is assumed to be at the origin.

We again need to define a `Sphere` object. This time, however, we need to
specify the sphere's refractive index using the `n_p` keyword argument:

.. testcode::

   sphere = Sphere(a = 0.5e-6, n_p = 1.59)

This refractive index is typical for polystyrene particles.

Now we can construct a `OTSimulation` object. The constructor is similar
to those we encountered previously, except that we now have to specify
the `Beam` object as well:

.. testcode::

   sim = OTSimulation(sphere, beam, timestep = 2e-5, viscosity = 1e-3,
                      kT = 295 * 1.38e-23)

The simulation can then be run as before:

.. testcode::

   traj = sim.run(500)

Note that optical tweezer simulations can take longer to run, particularly for
large particles. This is because the optical force calculations can be
intensive. The calculations are done using a T-matrix approach; see
Nieminen *et al.* [Nieminen2007]_ for details.


Calculating forces and torques
------------------------------

In addition to simulating Brownian motion of particles, you may need
to calculate the optical forces and torques on a particle at fixed locations
and orientations. This can be useful for assessing whether or not a particle
can in fact be optically trapped, as well as for choosing an appropriate
time step. This can be done as follows:

..  testcode::

    from brownian_ot.ott_wrapper import make_ott_force
    force_func = make_ott_force(sphere, beam)  
   
`force_func`, the output of `make_ott_force`, is *itself* a function.
This is the one place where brownian_ot makes assumptions about units:
the speed of light is needed to calculate properly-dimensioned forces and
torques. The default value is in SI units. This function can be called as
follows:

..  testcode::
    
    force = force_func(np.zeros(3), np.identity(3)) 
    print(force)

..  testoutput::

    [ 0.00000000e+00  0.00000000e+00  6.07406601e-13  0.00000000e+00
      0.00000000e+00 -7.46591080e-35]

Here, we're calculating the generalized force when the sphere is at the origin.
There is a force in the :math:`+z` direction, which is plausible since
radiation pressure tends to push particles to equilibrium slightly beyond
the beam focus. As expected for a sphere, the torque on the particle is nearly
0 (the tiny nonzero :math:`z` component is due to roundoff).

Frequently, we are only interested in one component of the generalized force,
and we're interested in that component as a parameter -- one of the particle's
coordinates, or perhaps a rotation angle -- is varied. While this information
can be obtained by calling the force function directly, brownian_ot implements
several convenience functions in the `brownian_ot.force_utils` module.
For example, here is how you could calculate the :math:`z` component of
the force on the sphere we're considering for different values of :math:`z`
and with :math:`x = 0` and :math`y = 0`:

..  testcode::

    from brownian_ot.force_utils import calc_fz
    calc_fz(0, force_func)

We can also pass in a range of values, which can be useful for plotting:

..  testcode::

    zs = np.linspace(-2e-6, 2e-6, 101)
    calc_fz(zs, force_func)
    
Similar convenience functions apply to the :math:`x` and :math:`y` components
of the force, as well as to the :math:`x` and :math:`y` components of the
torque.

These convenience functions are most often useful near equilibrium.
For a rotationally symmetric particle (like a sphere or dimer),
the equilibrium position lies on the :math:`z` axis, but not at :math:`z=0`.
Thus, there is also a convenience function for finding that equilibrium.
We'll demonstrate this by calculating the :math:`y` component of the
torque on a spheroid at its equilibrium height when rotated by various
angles about the :math:`y` axis.

..  testcode::

    from brownian_ot.particles import Spheroid
    from brownian_ot.force_utils import find_zeq, calc_ty
    spheroid = Spheroid(a = 150e-9, ar = 1.6, n_p = 1.45)
    spheroid_force = make_ott_force(spheroid, beam)
    z_eq = find_zeq(spheroid_force)

This spheroid has a minor radius of 150 nm and an aspect ratio of 1.6.
Now the torques can be calculated:

..  testcode::

    thetas = np.linspace(-np.pi/2, np.pi/2, 101)
    torques = calc_ty(thetas, spheroid_force, z = z_eq)




A cautionary note
-----------------
The Python interface to ott relies on Matlab global variables.
What this means is that *if* a new optical force function is created,
any preceding ones will no longer work as expected. For example,
in the examples here, after creating `spheroid_force`, `force_func` will
no longer correctly calculate the optical forces on the sphere that was
originally specified. The solution is to re-run :code:`make_ott_force`, as
shown below:

..  testcode::

    force_func = make_ott_force(sphere, beam)  
    calc_fz(0, force_func) # now correct 
    

