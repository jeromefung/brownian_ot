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


def _calc_cluster_displacements(trajectory, nsteps, cluster_frame = True,
                                full_output = False):
    '''
    Calculate cluster-frame squared displacements (for evaluating
    diagonal elements of the diffusion tensor) as well as axis
    autocorrelations.

    '''    
    n_pts = trajectory.shape[0]
    max_rows = n_pts - nsteps
    
    # calculate lab-frame displacments
    lab_disps = (np.roll(trajectory[:, 0:3], -nsteps, axis = 0)
                 - trajectory[:,0:3])[:max_rows, :]

    # calculate each particle-frame unit vector in lab coordinates
    # (look at each column of rotation matrix)
    u1_lab = trajectory[:max_rows, 3:6]
    u2_lab = trajectory[:max_rows, 6:9]
    u3_lab = trajectory[:max_rows, 9:12]

    if cluster_frame:
        # dot products
        deltax_1 = (lab_disps * u1_lab).sum(axis = 1)
        deltax_2 = (lab_disps * u2_lab).sum(axis = 1)
        deltax_3 = (lab_disps * u3_lab).sum(axis = 1)
    else:
        deltax_1 = lab_disps[:, 0]
        deltax_2 = lab_disps[:, 1]
        deltax_3 = lab_disps[:, 2]
        
    if full_output:
        # return squared displacements, but don't take mean
        # each row is a different lab frame displacement
        output =  (np.vstack((deltax_1, deltax_2, deltax_3))**2).transpose()
        return output
    else: # return MSD
        return np.array([np.mean(deltax_1**2), np.mean(deltax_2**2),
                         np.mean(deltax_3**2)])
    

def _calc_axis_dot_prods(trajectory, nsteps, full_output = False):
    n_pts = trajectory.shape[0]
    max_rows = n_pts - nsteps
        
    product = (trajectory[:, 3:] * np.roll(trajectory[:, 3:],
                                           -nsteps, axis = 0))[:max_rows ,:]

    u1dotu1 = (product[:, 0:3]).sum(axis = 1)
    u2dotu2 = (product[:, 3:6]).sum(axis = 1)
    u3dotu3 = (product[:, 6:9]).sum(axis = 1)


    if full_output:
        # each row is the dot products from a different pair of points
        # don't take the mean
        output = np.vstack((u1dotu1, u2dotu2, u3dotu3)).transpose()
        return output
    else:
        # take the mean
        return np.array([np.mean(u1dotu1), np.mean(u2dotu2),
                         np.mean(u3dotu3)])
    
        
def calc_msd(trajectory, max_steps = None, cluster_frame = True):
    # set a sensible default, half the trajectory length if not given
    if max_steps is None:
        max_steps = np.floor(trajectory.shape[0] / 2).astype('int')

    if trajectory.shape[1] == 7: # orientation still in quaternion form
        trajectory = expand_trajectory(trajectory)
 
    output = np.zeros((3, max_steps))

    for i in np.arange(1, max_steps + 1):
        output[:, i - 1] = _calc_cluster_displacements(trajectory, i,
                                                       cluster_frame =
                                                       cluster_frame)

    return output[0], output[1], output[2]


def calc_axis_autocorr(trajectory, max_steps = None):
    # set a sensible default, half the trajectory length if not given
    if max_steps is None:
        max_steps = floor(trajectory.shape[0] / 2).astype('int')

    if trajectory.shape[1] == 7: # orientation still in quaternion form
        trajectory = expand_trajectory(trajectory)
    
    output = np.zeros((3, max_steps))

    for i in np.arange(1, max_steps + 1):
        output[:, i - 1] = _calc_axis_dot_prods(trajectory, i)

    return output[0], output[1], output[2]


'''
Rotation matrices/quaternions map

PARTICLE vectors onto LAB vectors

need INVERSE to map LAB onto PARTICLE

'''

    
