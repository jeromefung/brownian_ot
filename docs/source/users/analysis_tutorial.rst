.. _analysis_tutorial:

Analyzing Results
=================

brownian_ot includes functions for analyzing particle trajectories
in the :code:`brownian_ot.analysis` module.

Let's demonstrate those capabilities by first simulating a freely-diffusing
dimer:

..  testcode::
    
    import numpy as np
    from brownian_ot.particles import Dimer
    from brownian_ot.simulation import FreeDiffusionSimulation
    from brownian_ot.analysis import calc_msd, calc_axis_autocorr

Next,

..  testcode::

    dimer = Dimer(a = 0.4e-6)
    dt = 5e-5
    sim = FreeDiffusionSimulation(dimer, dt, 1e-3, 295*1.38e-23)
    traj = sim.run(5000)

To analyze the translational motion of a particle, we usually want to calculate
mean-squared displacements in the particle frame. (This means that we decompose
the particle's displacements in the simulation frame into its components
along the particle's reference axes.) This can be done as follows:

..  testcode::

    msd_x, msd_y, msd_z = calc_msd(traj, max_steps = 20, particle_frame = True)
    times = (np.arange(20) + 1) * dt

The last line yields the physical times at which the MSDs are calculated.

To analyze the rotational diffusion, we can examine axis autocorrelation
functions, which are calculated in a similar way:

..  testcode::

    axis_x, axis_y, axis_z = calc_axis_autocorr(traj, max_steps = 20)


