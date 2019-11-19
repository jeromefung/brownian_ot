import numpy as np
from numpy import pi
import matlab.engine
import yaml
import os

# TODO: some sort of graceful exception handling if importing matlab fails
# Code should still be usable for other functionality if matlab engine not
# present.

# Start Matlab engine
eng = matlab.engine.start_matlab()

# Load config file, add path to ott, wrappers to matlab engine
dir = os.path.dirname(__file__)
config_fname = os.path.join(dir, '../paths_config.yaml')
with open(config_fname, mode = 'r') as file:
    config = yaml.load(file)
eng.addpath(config['ott_path'])
eng.addpath(config['matlab_wrapper_path'])

# Unit conversion from efficiencies to real forces:
# force: multiply efficiency by P * n_med / c
# torque: multiply by P / omega

def make_force(lambda_0, pol, n_med, NA, power, scatterer_type, scatterer_dict,
               c = 3e8):
    '''
    Factory to return a function to calculate the optical force
    on a sphere.

    Inputs:
    lambda_0 : vacuum wavelength
    pol : 2-element list/ndarray (x and y components)
    n_med : medium index
    NA : objective numerical aperture
    power : beam power (in self-consistent units)
    scatterer_type : 'sphere' or 'spheroid'
    scatterer_dict : see below
    c : speed of light in self-consistent units (default SI)
    '''

    eng.ott_beam(lambda_0, pol[0], pol[1], NA, n_med, nargout = 0)
    if scatterer_type == 'sphere':
        eng.ott_tmatrix_sphere(scatterer_dict['n_p'],
                               scatterer_dict['r_p'],
                               lambda_0, n_med, nargout = 0)
    elif scatterer_type == 'spheroid':
        eng.ott_matrix_spheroid(scatterer_dict['n_p'],
                                scatterer_dict['a'],
                                scatterer_dict['c'],
                                lambda_0, n_med, nargout = 0)
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

