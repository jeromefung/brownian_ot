import numpy as np
from numpy import pi
import matlab.engine
import yaml
import os
import tempfile
import shutil

# TODO: some sort of graceful exception handling if importing matlab fails
# Code should still be usable for other functionality if matlab engine not
# present.

# Start Matlab engine
eng = matlab.engine.start_matlab()

# Load config file, add path to ott, wrappers to matlab engine
dir = os.path.dirname(__file__)
config_fname = os.path.join(dir, '../paths_config.yaml')
with open(config_fname, mode = 'r') as file:
    config = yaml.load(file, Loader = yaml.SafeLoader)
    # plain load deprecated for security reasons
    # see https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
eng.addpath(config['ott_path'])
eng.addpath(config['matlab_wrapper_path'])

# Unit conversion from efficiencies to real forces:
# force: multiply efficiency by P * n_med / c
# torque: multiply by P / omega

def make_ots_force(particle, beam, c = 3e8):
    '''
    Factory to return a function to calculate the optical force
    on a particle.

   
    '''

    eng.ott_beam(beam.lambda_0, beam.pol[0], beam.pol[1], beam.NA,
                 beam.n_med, nargout = 0)

    if isinstance(particle, Sphere):
        eng.ott_tmatrix_sphere(particle.n_p, particle.a,
                               beam.lambda_0, beam.n_med, nargout = 0)
    elif isinstance(particle, Spheroid):
        eng.ott_tmatrix_spheroid(particle.n_p, particle.a,
                                 particle.a * particle.ar,
                                 beam.lambda_0, beam.n_med, nargout = 0)
    elif isinstance(particle, SphereCluster):
        # create temporary directory
        temp_dir = tempfile.mkdtemp()
        # write mstm input deck
        # TODO: make convergence criteria user-controllable w/defaults
        make_mstm_input(particle, beam,
                        os.path.join(temp_dir, 'cluster.inp'))
        # run mstm
        # call eng.ott_tmatrix_from_mstm to read in
        # delete temp_dir
    else:
        raise NotImplementedError('Other scatterers not yet implemented.')


    omega = 2*pi*c/lambda_0 # angular frequency
    
    def force(pos, rot_matrix):
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
            np.concatenate((n_med * power / c * np.ones(3),
                            power / omega * np.ones(3)))
        
        
    return force


def make_mstm_input(particle, beam, save_name):
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
    deck_file.write('{:.15f}\n'.format(particle.a.real))
    deck_file.write('imag_ref_index_scale_factor\n')
    deck_file.write('{:.15f}\n'.format(particle.a.imag))
    deck_file.write('mie_epsilon\n') # TODO: make epsilons user-controllable
    deck_file.write('1.0d-6\n')
    deck_file.write('translation_epsilon\n')
    deck_file.write('1.0d-6\n')
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

    # Iterate over array of sphere positions
    for pos in particle.sphere_pos:
        deck_file.write('1.d0 {:.15f} {:.15f} {:.15f} \n'.format(*pos))

    
    deck_file.close()
    return
