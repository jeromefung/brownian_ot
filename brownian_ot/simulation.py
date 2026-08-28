import numpy as np
from numpy import sin, cos
from brownian_ot.ott_wrapper import make_ott_force
import quaternion
import asdf
from pathlib import Path


class Simulation:
    """
    Base class.
    """

    def __init__(
        self,
        particle,
        timestep,
        f_ext,
        viscosity,
        kT,
        pos0=np.zeros(3),
        orient0=np.identity(3),
        seed=None,
    ):
        self.particle = particle
        self.timestep = timestep
        self.f_ext = f_ext
        self.rng_seed = seed
        self.viscosity = viscosity
        self.kT = kT
        # set particle initial position
        particle._pos = pos0
        # Be able to handle any array-like object
        particle._pos = np.asarray(pos0)
        if particle._pos.shape != (3,):
            raise ValueError("Initial position must be array-like with length 3")
        # Be able to handle both a quaternion or a rotation matrix
        if isinstance(orient0, quaternion.quaternion):
            particle._orient = orient0
        elif isinstance(orient0, np.ndarray) and orient0.shape == (3, 3):
            particle._orient = quaternion.from_rotation_matrix(orient0)
        else:
            raise TypeError(
                "orient0 must be a quaternion or a 3 x 3 ndarray representing a rotation matrix"
            )
        self.rng = np.random.RandomState(seed)

    def _q_random(self):
        """
        Calculate random generalized displacement obeying generalized
        Stokes-Einstein relation. Recall
        <q_i q_j> = 2D_{ij} Delta t.

        Return array of random displacements, prior to rescaling.
        Need to scale by sqrt(2 dt)
        """
        # calculate diffusion tensor in physical units
        D = self.particle.Ddim * self.kT / self.viscosity
        return self.rng.multivariate_normal(np.zeros(6), D)

    def _update(self):
        if self.kT == 0:
            q_B = np.zeros(6)  # no brownian motion at 0 temperature
        else:
            # calc q^B in particle frame with appropriate scaling
            q_B = self._q_random() * np.sqrt(2 * self.timestep)

        # calculate generalized force in lab frame
        force = self.f_ext(
            self.particle._pos, quaternion.as_rotation_matrix(self.particle._orient)
        )

        # find vector d from COM to COD, known in particle frame,
        # in lab frame.
        # quaternion package does this by converting quaternion
        # to rotation matrix, but could be done via conjugation
        d = quaternion.rotate_vectors(self.particle._orient, self.particle.cod)

        # correct torque (last 3 elts of generalized force) to be about cod
        # minus sign because d points TO the COD
        force[3:] = force[3:] - np.cross(d, force[0:3])

        # convert generalized force from lab to particle frame
        # need inverse of orientation quaternion
        force_pf = np.ravel(
            quaternion.rotate_vectors(
                self.particle._orient.inverse(), force.reshape((2, -1))
            )
        )

        # Calculate q^D in particle frame
        # particle.Ddim has no kT in it
        q_D = np.matmul(self.particle.Ddim / self.viscosity, force_pf) * self.timestep

        # find q_total
        q_total = q_B + q_D  # still in particle frame

        # convert spatial part of generalized displacement to lab frame
        # following Garcia de la Torre, use non-updated orientation
        delta_xyzlab = quaternion.rotate_vectors(self.particle._orient, q_total[0:3])
        # Update particle COM position
        self.particle._pos = self.particle._pos + delta_xyzlab

        # Update orientation quaternion
        infntsml_rotmat = unbiased_rotation(*q_total[3:6])
        infntsml_quat = quaternion.from_rotation_matrix(infntsml_rotmat)
        # see BROWNRIG paper eq. 19, use quaternion composition
        self.particle._orient = self.particle._orient * infntsml_quat

    def run(self, n_steps, outfname = None, downsample_interval = None,
            downsampled_fname = None):
        """
        Run the simulation.

        Parameters
        ----------
        n_steps: integer
            Number of time steps to run.
        outfname : string, optional
            If not None, name of file to optionally save the result to. 
            Trajectory and metadata are saved as an ASDF file.
            A .asdf extension is automatically appended to the name if not present.
        downsample_interval : None or integer, optional
            If not None, save a reduced trajectory keeping every downsample_interval points.

        Returns
        -------
        traj: ndarray (`n_steps + 1`, 7)
            Particle trajectory. This is an array with `n_steps + 1` rows
            since the initial position and orientation are specified in the
            first row. Each row contains the particle's x, y, and z
            coordinates followed by its orientation specified by a quaternion. 

        """
        # Validate
        if downsample_interval is not None and outfname is None:
            raise TypeError("The 'outfname' keyword argument is required if 'donwsample_interval' is provided.")
            
        if downsampled_fname is not None and downsample_interval is None:
            raise TypeError("The 'downsample_interval keyword argument is required if 'downsampled_fname' is provided.")
             
        # Preallocate
        file_len = n_steps + 1
        output = np.zeros((file_len, 7)) # com coords, quaternion
        output[0] = self.particle._nice_output() # First row is initial position
        
        # main loop
        for ctr in np.arange(1, n_steps + 1):
            # step the particle
            self._update()
            output[ctr] = self.particle._nice_output()

        if outfname is not None:
            metadata_tree = self._prepare_metadata(output)
            asdf_object = AsdfFile(metadata_tree)
            
            output_path = Path(outfname)
            if not output_path.suffix:
                output_path = output_path.with_suffix(".asdf")
                
            asdf_object.write_to(output_path)
            
            if downsample_interval is not None:
                downsampled_tree = self._prepare_downsampled_metadata(metadata_tree, 
                                                                      downsample_interval)
                downsampled_asdf_object = AsdfFile(downsampled_tree)
                if downsampled_fname is None:
                    orig_path = Path(outfname)
                    downsampled_path = orig_path.with_name(orig_path.stem + '_downsampled'
                                                           + orig_path.suffix)
                else:
                    downsampled_path = Path(downsampled_fname)
                    if not p.suffix:
                        downsampled_path = downsampled_path.with_suffix(".asdf")
                downsampled_asdf_object.write_to(downsampled_path)
                
        return output
    
    
    def _prepare_metadata(self, trajectory):
        # Every simulation has:
            # simulation parameters
            # data
            # a particle (which should write its own metadata)
        simulation_dict = {
            'timestep' : self.timestep,
            'type' : self.__class__.__name__,
            'seed' : sim.rng_seed,
            'viscosity' : sim.viscosity,
            'kT' : sim.kT,
            'n_steps' : len(traj) - 1
        }
        particle_dict = self.particle._prepare_metadata()
        tree = {
            'trajectory' : trajectory,
            'particle' : particle_dict,
            'simulation' : simulation_dict
        }
        return tree
    
    
    def _prepare_downsampled_metadata(tree, interval):
        trajectory = tree['trajectory']
        simulation_dict = tree['simulation']
        simulation_dict['downsampled_timestep'] = simulation_dict['timestep'] * interval
        
        tree2 = {
            'trajectory' : trajectory[::interval, :].copy(),
            'particle' : particle_dict,
            'beam' : beam_dict,
            'simulation' : simulation_dict
        }
        return tree2
        
            
'''
        
        if outfname is not None:
            # prepare asdf file output
            particle_meta = {
                "type": type(self.particle).__name__,
                "Ddim": np.asarray(self.particle.Ddim),
                "cod": np.asarray(self.particle.cod),
                "n_p": None if self.particle.n_p is None else np.asarray(self.particle.n_p)}

            if hasattr(self.particle, "a"):
                particle_meta["a"] = self.particle.a
            if hasattr(self.particle, "ar"):
                particle_meta["ar"] = self.particle.ar
            if hasattr(self.particle, "n_spheres"):
                particle_meta["n_spheres"] = self.particle.n_spheres
            if hasattr(self.particle, "sphere_pos"):
                particle_meta["sphere_pos"] = np.asarray(self.particle.sphere_pos)
            if hasattr(self.particle, "a_ratios"):
                particle_meta["a_ratios"] = np.asarray(self.particle.a_ratios)
            if hasattr(self.particle, "equivalent_sphere_radius"):
                particle_meta["equivalent_sphere_radius"] = (
                    self.particle.equivalent_sphere_radius
            )
        simulation_meta = {
            "class": type(self).__name__,
            "n_steps": n_steps,
            "timestep": self.timestep,
            "viscosity": self.viscosity,
            "kT": self.kT,
            "rng_seed": self.rng_seed,
            "pos0": data[0, :3],
            "orient0": data[0, 3:],
        }
        if hasattr(self, "c"):
            simulation_meta["c"] = self.c
        if hasattr(self, "force"):
            simulation_meta["force"] = np.asarray(self.force)
        asdf_data = {
            "schema_version": "1.0",
            "simulation": simulation_meta,
            "particle": particle_meta,
            "trajectory": {
                "n_rows": data.shape[0],
                "columns": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            },
            "data": data,
        }
        if hasattr(self, "beam"):
            beam = self.beam
            beam_meta = {
                "type": getattr(beam, "type", "Gaussian"),
                "wavelen": beam.wavelen,
                "pol": np.asarray(beam.pol, dtype=complex),
                "NA": beam.NA,
                "n_med": beam.n_med,
                "power": beam.power,
            }
            if hasattr(beam, "mode"):
                beam_meta["mode"] = np.asarray(beam.mode)
            asdf_data["beam"] = beam_meta

        output = asdf.AsdfFile(asdf_data)

        if outfname is not None:
            if outfname[-4:] != ".asdf":
                outfname = outfname + ".asdf"
            output.write_to(outfname)

        return output
'''

class FreeDiffusionSimulation(Simulation):
    """
    Simulates the Brownian motion of a particle with no external forces.
    """

    def __init__(
        self,
        particle,
        timestep,
        viscosity,
        kT,
        pos0=np.zeros(3),
        orient0=np.identity(3),
        seed=None,
    ):
        """
        Parameters
        ----------
        particle : Particle object
            Particle to be simulated.
        timestep : float
            Time step for simulation.
        viscosity : float
            Solvent viscosity.
        kT : float
            Thermal energy scale.
        pos0 : array-like (3), optional
            Initial position of particle. Defaults to the origin (0,0,0).
        orient0 : array-like (3x3) or quaternion, optional
            Initial orientation of particle. Defaults to particle reference
            orientation (identity rotation matrix).
        seed : integer, optional
            Seed for NumPy random number generator.
        """

        def zero_force(pos, orient):
            return np.zeros(6)

        super().__init__(
            particle, timestep, zero_force, viscosity, kT, pos0, orient0, seed
        )


class OTSimulation(Simulation):
    """
    Simulates a particle experiencing forces due to optical tweezers.
    """

    def __init__(
        self,
        particle,
        beam,
        timestep,
        viscosity,
        kT,
        pos0=np.zeros(3),
        orient0=np.identity(3),
        seed=None,
        c=3e8,
    ):
        """
        Parameters
        ----------
        particle : Particle object
            Particle to be simulated.
        beam : Beam object
            Describes incident beam.
        timestep : float
            Time step for simulation.
        viscosity : float
            Solvent viscosity.
        kT : float
            Thermal energy scale.
        pos0 : array-like (3), optional
            Initial position of particle. Defaults to the origin (0,0,0).
        orient0 : array-like (3x3) or quaternion, optional
            Initial orientation of particle. Defaults to particle reference
            orientation (identity rotation matrix).
        seed : integer, optional
            Seed for NumPy random number generator.
        c : float, optional
            The speed of light. Default value is in SI units; change to use
            any other self-consistent unit system.
        """
        # Check that particle has a refractive index specified
        if particle.n_p is None:
            raise TypeError("Particle refractive index needs to be specified.")
        super().__init__(
            particle,
            timestep,
            make_ott_force(particle, beam, c),
            viscosity,
            kT,
            pos0,
            orient0,
            seed,
        )
        self.beam = beam
        self.c = c
        
    def _prepare_metadata(self, trajectory):
        tree = super()._prepare_metadata(trajectory)
        # add beam
        beam_dict = self.beam._prepare_metadata()
        beam_dict['c'] = self.c
        tree['beam'] = beam_dict
        return(tree)


class ConstantForceSimulation(Simulation):
    """
    Simulates a particle experiencing a constant external generalized force.

    """

    def __init__(
        self,
        particle,
        timestep,
        force,
        viscosity,
        kT,
        pos0=np.zeros(3),
        orient0=np.identity(3),
        seed=None,
    ):
        """
        Parameters
        ----------
        particle : Particle object
            Particle to be simulated.
        timestep : float
            Time step for simulation.
        force: ndarray (6)
            Generalized force vector (force + torque)
        viscosity : float
            Solvent viscosity.
        kT : float
            Thermal energy scale.
        pos0 : array-like (3), optional
            Initial position of particle. Defaults to the origin (0,0,0).
        orient0 : array-like (3x3) or quaternion, optional
            Initial orientation of particle. Defaults to particle reference
            orientation (identity rotation matrix).
        seed : integer, optional
            Seed for NumPy random number generator.

        """

        def const_force(pos, orient):
            """
            Dummy input variables, but Simulation._update() expects a callable
            function.
            """
            return force

        super().__init__(
            particle, timestep, const_force, viscosity, kT, pos0, orient0, seed
        )
        self.force = force
        
    def _prepare_metadata(self, trajectory):
        tree = super()._prepare_metadata(trajectory)
        # append force to simulation parameters
        tree['simulation']['force'] = self.force
        return tree


def unbiased_rotation(a, b, c):
    """
    Calculate unbiased rotation operator given infinitesimal rotation angles
    a, b, c about x, y, and z axes.

    See Beard & Schlick, Biophys. J. (2003), eq. 5.

    Note: typo fixed in 22 element.
    """
    omsq = a**2 + b**2 + c**2

    # allow for case of no rotation
    if omsq == 0:  # no rotation, return identity matrix
        return np.identity(3)
    else:
        om = np.sqrt(omsq)
        m11 = ((b**2 + c**2) * cos(om) + a**2) / omsq
        m12 = a * b * (1 - cos(om)) / omsq - c * sin(om) / om
        m13 = a * c * (1 - cos(om)) / omsq + b * sin(om) / om

        m21 = a * b * (1 - cos(om)) / omsq + c * sin(om) / om
        m22 = ((a**2 + c**2) * cos(om) + b**2) / omsq
        m23 = b * c * (1 - cos(om)) / omsq - a * sin(om) / om

        m31 = a * c * (1 - cos(om)) / omsq - b * sin(om) / om
        m32 = b * c * (1 - cos(om)) / omsq + a * sin(om) / om
        m33 = ((a**2 + b**2) * cos(om) + c**2) / omsq

        return np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
