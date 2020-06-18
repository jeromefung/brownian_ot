import numpy as np
from numpy import pi
from numpy.testing import assert_allclose

import brownian_ot
from brownian_ot.utils import rot_x, rot_y, rot_z, sphere_D, dimer_D
from brownian_ot.force_utils import calc_fx, calc_fy, calc_fz, calc_tx, calc_ty

def test_rotation_matrices():
    invert_yz = np.diag([1, -1, -1])
    invert_xz = np.diag([-1, 1, -1])
    invert_xy = np.diag([-1, -1, 1])
    identity = np.identity(3)
    
    assert_allclose(rot_x(pi), invert_yz, atol = 1e-15)
    assert_allclose(rot_x(2*pi), identity, atol = 1e-15)
    assert_allclose(rot_y(pi), invert_xz, atol = 1e-15)
    assert_allclose(rot_y(2*pi), identity, atol = 1e-15)
    assert_allclose(rot_z(pi), invert_xy, atol = 1e-15)
    assert_allclose(rot_z(2*pi), identity, atol = 1e-15)


def test_force_extraction():
    def dummy_force(pos, orient):
        return np.arange(6)

    func_list = [calc_fx, calc_fy, calc_fz, calc_tx, calc_ty]
    for (func, i) in zip(func_list, np.arange(5)):
        assert_allclose(func(0, dummy_force), i)


def test_dimer_D_ratios():
    '''
    Check that dimensionality in dimer diffusion tensors is correct.
    '''
    kT = 1/25 * 1e-19
    eta = 1e-3
    a = 5e-7
    D_sphere = sphere_D(a, kT, eta)
    D_dimer = dimer_D(a, kT, eta)
    assert_allclose(np.array([D_dimer[2,2]/D_sphere[2,2],
                              D_dimer[5,5]/D_sphere[5,5]]),
                    np.array([6/7.740, 8/14.42]))
    
