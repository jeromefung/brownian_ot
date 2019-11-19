from brownian_ot.ott_wrapper import make_force
import numpy as np

lambda_0 = 1.064e-6
pol = [1, 1j]
n_med = 1.33
NA = 1.2
power = 1e-3

def test_sphere_force():
    sphere_dict = {'n_p' : 1.5,
                   'r_p' : 0.5e-6}
    force = make_force(lambda_0, pol, n_med, NA, power,
                       'sphere', sphere_dict)
    force(np.zeros(3), np.identity(3))
    print(force)
    
    

