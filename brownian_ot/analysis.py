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


def _calc_cluster_displacements(trajectory, nsteps, particle_frame = True,
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

    if particle_frame:
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
    
        
def calc_msd(trajectory, max_steps = None, steps = None,
             particle_frame = True):
    '''
    Calculate mean-squared displacement from a simulated trajectory.

    Parameters
    ----------
    trajectory: ndarray (n x 7) or (n x 13)
        Particle trajectory. Simulations return arrays with 7 columns
        (3 coordinates and the 4 elements of the orientation quaternion),
        but specifying all 9 elements of the rotation matrix is also 
        permitted.
    max_steps : integer, optional
        Maximum number of steps to calculate MSD for. If specified,
        calculate MSD at integer steps up to max_step.
    steps : ndarray, optional
        Calculate MSD at these steps.
    particle_frame : boolean, optional
        If True (default), calculates MSDs along particle reference axes. 
        Otherwise, calculates MSDs in simulation frame.

    Returns
    -------
    msd_x, msd_y, msd_z : ndarray(max_steps)
        Mean-squared displacements.

    Notes
    -----
    Trajectories can be converted from quaternion form to rotation matrix
    form via `expand_trajectory()`.

    `calc_msd` is agnostic about the simulation time step. To obtain
    the actual times at which the MSD is calculated, if `dt` is the timestep,
    use :code:`(np.arange(max_step) + 1) * dt`.

    '''
    if steps is None:
        if max_steps:
            steps = np.arange(1, max_steps + 1)
        else:
            raise RuntimeError('Specify either max_steps or steps.')

    if trajectory.shape[1] == 7: # orientation still in quaternion form
        trajectory = expand_trajectory(trajectory)
 
    output = np.zeros((3, len(steps)))

    for i in np.arange(len(steps)):
        output[:, i] = _calc_cluster_displacements(trajectory, steps[i],
                                                   particle_frame =
                                                   particle_frame)

    return output[0], output[1], output[2]


def calc_axis_autocorr(trajectory, max_steps = None, steps = None):
    '''
    Calculate axis autocorrelation functions from a simulated trajectory.

    Parameters
    ----------
    trajectory: ndarray (n x 7) or (n x 13)
        Particle trajectory. Simulations return arrays with 7 columns
        (3 coordinates and the 4 elements of the orientation quaternion),
        but specifying all 9 elements of the rotation matrix is also 
        permitted.
    max_steps : integer, optional
        Maximum number of steps to calculate axis autocorrelations for.

    Returns
    -------
    axis_x, axis_y, axis_z : ndarray(max_steps)


    Notes
    -----
    The axis autocorrelation for the :math:`x` axis at delay time :math:`\\tau` 
    is defined as

    .. math:: \\langle \\hat{x}(t+\\tau) \\cdot \\hat{x}(t) \\rangle 

    Trajectories can be converted from quaternion form to rotation matrix
    form via `expand_trajectory()`.

    `calc_axis_autocorr` is agnostic about the simulation time step. To obtain
    the actual times at which the autocorrelations are calculated, if 
    `dt` is the timestep, use :code:`(np.arange(max_step) + 1) * dt`.
    '''
    if steps is None:
        if max_steps:
            steps = np.arange(1, max_steps + 1)
        else:
            raise RuntimeError('Specify either max_steps or steps.')

    if trajectory.shape[1] == 7: # orientation still in quaternion form
        trajectory = expand_trajectory(trajectory)
    
    output = np.zeros((3, len(steps)))

    for i in np.arange(len(steps)):
        output[:, i] = _calc_axis_dot_prods(trajectory, steps[i])

    return output[0], output[1], output[2]


'''
Rotation matrices/quaternions map

PARTICLE vectors onto LAB vectors

need INVERSE to map LAB onto PARTICLE

'''


def quaternion_orientation_average(input):
    '''
    Perform orientation average given an input ndarray of quaternions.

    Parameters
    ----------
    input : ndarray, float (n x 4)
        Each row corresponds to one quaternion to be averaged.

    Returns
    -------
    avg_orient : ndarray (4)
        ndarray (not quaternion object!) corresponding to orientation average.

    Notes
    -----
    Implements algorithm in Markley et al. (2007):
    https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf

    This produces a quaternion minimizing the so-called L2 chordal norm.
    '''
    A = np.array([np.matmul(row.reshape((4,1)), row.reshape(1,4))
                  for row in input]).sum(axis=0)
    eigvals, eigvecs = np.linalg.eig(A)
    # choose eigenvector w/largest eigenvalue
    sstar = eigvecs[:, np.argmax(eigvals)]
    return sstar/np.linalg.norm(sstar)
    
