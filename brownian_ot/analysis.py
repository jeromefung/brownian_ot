'''
Functions for performing analysis of diffusion data.
'''

import numpy as np
import quaternion

def expand_trajectory(traj):
    '''
    Code saves trajectories as x, y, z, a, b, c, d where the
    a, b, c, d are quaternion elements. 
    Expand to provide full rotation matrix.

    Rows of output are now
    x, y, z, u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z
    '''

    # convert relevant elements of ndarrays to an array of quaternions
    quat_array = quaternion.from_float_array(traj[:, 3:])
    # now convert to array of rotation matrices
    rot_matrices = quaternion.as_rotation_matrix(quat_array)
    # transpose for convenience (so we can access unit vectors)
    # and reshape
    n_pts = traj.shape[0]
    rot_matrices = np.transpose(rot_matrices,
                                axes = (0,2,1)).reshape((n_pts, 9))
    output = np.hstack((traj[:,0:3], rot_matrices))
    return output
    
def calc_cluster_displacements(trajectory, nsteps,
                               full_output = False):
    '''
    Calculate cluster-frame squared displacements (for evaluating
    diagonal elements of the diffusion tensor) as well as axis
    autocorrelations.

    '''
    pass

    
