import numpy as np
from numpy import pi
import warnings
import yaml
import os
import tempfile
import shutil
import subprocess
from brownian_ot.particles import Sphere, Spheroid, SphereCluster

# Load config file, add path to ott, wrappers to matlab engine
dir = os.path.dirname(__file__)
config_fname = os.path.join(dir, '../paths_config.yaml')
with open(config_fname, mode = 'r') as file:
    config = yaml.load(file, Loader = yaml.SafeLoader)
    # plain load deprecated for security reasons
    # see https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation

# Allow module to be imported even if importing Matlab engine fails.
# Exception with meaningful error message raised by make_ott_force
try:
    import matlab.engine
    # Start Matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(config['ott_path'])
    eng.addpath(config['matlab_wrapper_path'])
    _MATLAB_ENGINE = True
except ImportError:
    _MATLAB_ENGINE = False


# Unit conversion from efficiencies to real forces:
# force: multiply efficiency by P * n_med / c
# torque: multiply by P / omega

def make_ott_force(particle, beam, c = 3e8):
    '''
    Factory function that returns a function to calculate the optical force
    on a particle.

    Parameters
    ----------
    particle : `Particle` object
        The particle experiencing the force.
    beam : `Beam` object
        The incident beam.
    c : float, optional
        The speed of light. The default value assumes SI units. Change
        this if you want to use any other self-consistent unit system.

    Returns
    -------
    force_func : function
        Function that calculates optical forces and torques.
    '''

    if not _MATLAB_ENGINE:
        raise RuntimeError("MATLAB engine could not be started. Check that MATLAB and the MATLAB Engine API for Python are properly installed.")

    # use hasattr here to switch between old vs new syntax
    if hasattr(beam, 'mode'):
        mode = matlab.double(beam.mode)
    else: # assume [0, 0] default
        mode = matlab.double([0,0]) # needed b/c of arithmetic in ott

    beam_nmax = eng.ott_beam(beam.wavelen, beam.pol[0], beam.pol[1],
                             beam.NA, beam.n_med, mode)

    if isinstance(particle, Sphere):
        eng.ott_tmatrix_sphere(particle.n_p, particle.a,
                               beam.wavelen, beam.n_med, nargout = 0)
    elif isinstance(particle, Spheroid):
        eng.ott_tmatrix_spheroid(particle.n_p, particle.a,
                                 particle.a * particle.ar,
                                 beam.wavelen, beam.n_med, nargout = 0)
    elif isinstance(particle, SphereCluster):
        # create temporary directory
        temp_dir = tempfile.mkdtemp()
        # write mstm input deck
        # TODO: make convergence criteria user-controllable w/defaults
        _make_mstm_input(particle, beam,
                         os.path.join(temp_dir, 'cluster.inp'))
        # run mstm
        subprocess.run([config['mstm_executable_path'], 'cluster.inp'],
                       cwd = temp_dir)
        # call eng.ott_tmatrix_from_mstm to read cluster_tmatrix.dat
        tmatrix_path = os.path.join(temp_dir, 'cluster_tmatrix.dat')
        particle_nmax = eng.ott_tmatrix_from_mstm(tmatrix_path)
        # TODO: do something beyond raising a warning
        if beam_nmax < particle_nmax:
            warnings.warn('ott beam n_max < mstm cluster n_max',
                          RuntimeWarning)

        # temp_dir should get garbage-collected
    else:
        raise NotImplementedError('Other scatterers not yet implemented.')

    omega = 2*pi*c/beam.wavelen # angular frequency

    def force(pos, rot_matrix):
        '''
        Calculates generalized optical force on a particle.

        Parameters
        ----------
        pos : list or ndarray (3)
            Coordinates of particle center of mass.
        rot_matrix : list or ndarray (3,3)
            Rotation matrix describing particle orientation.

        Returns
        -------
        force : ndarray (6)
            Generalized force (force + torque).
        '''
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()

        pos = matlab.double(pos)

        if isinstance(rot_matrix, np.ndarray):
            rot_matrix = rot_matrix.tolist()

        rot_matrix = matlab.double(rot_matrix)

        fx, fy, fz, tx, ty, tz = eng.ott_calc_force(pos, rot_matrix,
                                                    nargout = 6)
        # incident beam has unit power, so normalize correctly
        return np.array([fx, fy, fz, tx, ty, tz]) * \
            np.concatenate((beam.n_med * beam.power / c * np.ones(3),
                            beam.power / omega * np.ones(3)))


    return force


def _make_mstm_input(particle, beam, save_name):
    mstm_length_scale_factor = 2 * pi * particle.a / beam.wavelen

    # mstm-modules-v3.0.f90 specifies defaults for many parameters
    # in module spheredata.
    # So, input deck doesn't need to specify them all.
    # Also, order isn't critical. See loop in subroutine inputdata,
    # line 1907.

    deck_file = open(save_name, 'w', encoding='utf-8')
    deck_file.write('number_spheres\n')
    deck_file.write(str(particle.n_spheres) + '\n')
    deck_file.write('sphere_position_file\n')
    deck_file.write('\n')
    deck_file.write('output_file\n')
    deck_file.write('cluster_output.dat\n')
    deck_file.write('run_print_file\n')
    deck_file.write('\n') # Leave blank for now, which writes to screen
    deck_file.write('length_scale_factor\n')
    deck_file.write('{:.15f}\n'.format(mstm_length_scale_factor))
    deck_file.write('real_ref_index_scale_factor\n')
    if np.isscalar(particle.n_p):
        deck_file.write('{:.15f}\n'.format(particle.n_p.real))
    else: # For arrays, make this 1 and use 6-entries per particle
        deck_file.write('1.0\n')
    deck_file.write('imag_ref_index_scale_factor\n')
    if np.isscalar(particle.n_p):
        deck_file.write('{:.15f}\n'.format(particle.n_p.imag))
    else:
        deck_file.write('1.0\n')
    deck_file.write('medium_real_ref_index\n')
    deck_file.write('{:.15f}\n'.format(beam.n_med))
    deck_file.write('medium_imag_ref_index\n')
    deck_file.write('0.d0\n') # non-absorbing media only
    deck_file.write('mie_epsilon\n') # TODO: make epsilons user-controllable
    deck_file.write('1.0d-7\n')
    deck_file.write('translation_epsilon\n')
    deck_file.write('1.0d-8\n')
    deck_file.write('solution_epsilon\n')
    deck_file.write('1.0d-8\n')
    deck_file.write('max_number_iterations\n')
    deck_file.write('5000\n')
    deck_file.write('store_translation_matrix\n')
    deck_file.write('0\n')
    deck_file.write('sm_number_processors\n') # not using MPI
    deck_file.write('1\n')
    deck_file.write('iterations_per_correction\n')
    deck_file.write('20\n')
    deck_file.write('number_scattering_angles\n')
    deck_file.write('0\n')
    deck_file.write('fixed_or_random_orientation\n')
    deck_file.write('1\n')
    deck_file.write('calculate_t_matrix\n')
    deck_file.write('1\n')
    deck_file.write('t_matrix_file\n')
    deck_file.write('cluster_tmatrix.dat\n')
    deck_file.write('t_matrix_convergence_epsilon\n') #TODO: make settable
    deck_file.write('1.d-7\n')
    deck_file.write('sphere_sizes_and_positions\n')

    # Iterate over particles to write radii and positions.
    if hasattr(particle, 'a_ratios'):
        particle_data = np.hstack((np.array([particle.a_ratios]).transpose(),
                                   particle.sphere_pos))
    else:
        particle_data = np.hstack((np.ones(particle.n_spheres).reshape((particle.n_spheres,-1)),
                                   particle.sphere_pos))
    # Use 4-column format if all particles have the same index
    if np.isscalar(particle.n_p):
        for datum in particle_data:
            deck_file.write('{:.15f} {:.15f} {:.15f} {:.15f} \n'.format(*datum))
    else:
        for datum, rindex in zip(particle_data, particle.n_p):
            deck_file.write('{:.15f} {:.15f} {:.15f} {:.15f} {:.15f} {:.15f} \n'.format(*datum,
                                                                                        rindex.real,
                                                                                        rindex.imag))
    deck_file.close()
    return
