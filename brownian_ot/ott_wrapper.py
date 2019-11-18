import numpy as np
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

# TODO: unit conversion from efficiencies to real forces

def make_sphere_force(lambda_0, pol, n_med, NA, power, n_p, r_p):
    '''
    Factory to return a function to calculate the optical force
    on a sphere.

    Inputs:
    lambda_0 : vacuum wavelength
    pol : 2-element list/ndarray (x and y components)
    n_med : medium index
    NA : objective numerical aperture
    n_p : particle index
    r_p : particle radius
    '''

    eng.ott_beam(lambda_0, pol[0], pol[1], NA, n_med, nargout = 0)
    eng.ott_tmatrix_sphere(n_p, r_p, lambda_0, n_med, nargout = 0)

    def force(pos, rot_matrix):
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()

        pos = matlab.double(pos)

        if isinstance(rot_matrix, np.ndarray):
            rot_matrix = rot_matrix.tolist()

        rot_matrix = matlab.double(rot_matrix)

        fx, fy, fz, tx, ty, tz = eng.ott_calc_force(pos, rot_matrix,
                                                    nargout = 6)
        # TODO: fix normalization
        return np.array([fx, fy, fz, tx, ty, tz])
        
        
    return force

