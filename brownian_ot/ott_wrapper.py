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

def make_optical_force():
    return

