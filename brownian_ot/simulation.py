import numpy as np
from numpy import sin, cos
from brownian_ot.ott_wrapper import make_ott_force
import quaternion

class Simulation:
    '''
    Base class. 
    '''

    def __init__(self, particle, timestep, f_ext,
                 viscosity, kT, seed = None, pos0 = None, orient0 = None):
        self.particle = particle
        self.timestep = timestep
        self.f_ext = f_ext
        self.rng_seed = seed
        self.viscosity = viscosity
        self.kT = kT
        # if initial position and orientation specified, set particle
        if pos0 is not None:
            particle._pos = pos0
        if orient0 is not None:
            # Be able to handle both a quaternion or a rotation matrix
            if isinstance(orient0, quaternion.quaternion):
                particle._orient = orient0
            elif isinstance(orient0, np.ndarray) and orient0.shape == (3,3):
                particle._orient = quaternion.from_rotation_matrix(orient0)
            else:
                raise TypeError('orient0 must be a quaternion or a 3 x 3 ndarray representing a rotation matrix')
        self.rng = np.random.RandomState(seed)


    def _q_random(self):
        '''
        Calculate random generalized displacement obeying generalized
        Stokes-Einstein relation. Recall
        <q_i q_j> = 2D_{ij} \Delta t.
        
        Return array of random displacements, prior to rescaling.
        Need to scale by \sqrt(2 dt)
        '''
        # calculate diffusion tensor in physical units
        D = self.particle.Ddim * self.kT / self.viscosity
        return self.rng.multivariate_normal(np.zeros(6), D)
    
            
    def _update(self):
        if self.kT == 0:
            q_B = np.zeros(6) # no brownian motion at 0 temperature
        else:
            # calc q^B in particle frame with appropriate scaling
            q_B = self._q_random() * np.sqrt(2 * self.timestep)
        
        # calculate generalized force in lab frame
        force = self.f_ext(self.particle._pos,
                           quaternion.as_rotation_matrix(self.particle._orient))

        # find vector d from COM to COD, known in particle frame,
        # in lab frame.
        # quaternion package does this by converting quaternion
        # to rotation matrix, but could be done via conjugation
        d = quaternion.rotate_vectors(self.particle._orient,
                                      self.particle.cod)

        # correct torque (last 3 elts of generalized force) to be about cod
        # minus sign because d points TO the COD
        force[3:] = force[3:] - np.cross(d, force[0:3])

        # convert generalized force from lab to particle frame
        # need inverse of orientation quaternion
        force_pf = np.ravel(quaternion.rotate_vectors(self.particle._orient.inverse(),
                                                      force.reshape((2, -1))))
        
        # Calculate q^D in particle frame
        # particle.Ddim has no kT in it
        q_D = np.matmul(self.particle.Ddim / self.viscosity,
                        force_pf) * self.timestep

        # find q_total
        q_total = q_B + q_D # still in particle frame
        
        # convert spatial part of generalized displacement to lab frame
        # following Garcia de la Torre, use non-updated orientation
        delta_xyzlab = quaternion.rotate_vectors(self.particle._orient,
                                                 q_total[0:3])
        # Update particle COM position
        self.particle._pos = self.particle._pos + delta_xyzlab

        # Update orientation quaternion
        infntsml_rotmat = unbiased_rotation(*q_total[3:6])
        infntsml_quat = quaternion.from_rotation_matrix(infntsml_rotmat)
        # see BROWNRIG paper eq. 19, use quaternion composition
        self.particle._orient = self.particle._orient * infntsml_quat
       
    
    def run(self, n_steps, outfname = None):
        # Preallocate
        file_len = n_steps + 1
        output = np.zeros((file_len, 7)) # com coords, quaternion
        output[0] = self.particle._nice_output() # First row is initial position

        # main loop
        for ctr in np.arange(1, n_steps + 1):
            # step the particle
            self._update()
            output[ctr] = self.particle._nice_output()

        if outfname is not None:
            # Check if there's a .npy extension
            if outfname[-4:] != '.npy':
                outfname = outfname + '.npy'
            np.save(outfname, output)
            
        return output
    

class FreeDiffusionSimulation(Simulation):
    '''
    '''
    def __init__(self, particle, timestep,
                 viscosity, kT, seed = None, pos0 = None, orient0 = None):
        super().__init__(particle, timestep, np.zeros(6),
                         viscosity, kT, seed, pos0, orient0)

        
class OTSimulation(Simulation):
    def __init__(self, particle, beam, timestep,
                 viscosity, kT, seed = None, pos0 = None, orient0 = None):
        super().__init__(particle, timestep, make_ott_force(particle, beam),
                         viscosity, kT, seed, pos0, orient0)


class ConstantForceSimulation(Simulation):
    '''
    Performs simulations in which a particle experiences a constant
    external generalized force.
    '''
    def __init__(self, particle, timestep, force,
                 viscosity, kT, seed = None, pos0 = None, orient0 = None):
        '''
        Parameters
        ----------
        force: ndarray (6)
            Generalized force vector (force + torque)
        '''
        super().__init__(particle, timestep, force,
                        viscosity, kT, seed, pos0, orient0)
        


def unbiased_rotation(a, b, c):
    '''
    Calculate unbiased rotation operator given infinitesimal rotation angles
    a, b, c about x, y, and z axes.

    See Beard & Schlick, Biophys. J. (2003), eq. 5.

    Note: typo fixed in 22 element.
    '''
    omsq = a**2 + b**2 + c**2

    # allow for case of no rotation
    if omsq == 0: # no rotation, return identity matrix
        return np.identity(3)
    else:
        om = np.sqrt(omsq)
        m11 = ((b**2+c**2)*cos(om) + a**2) / omsq
        m12 = a*b*(1-cos(om))/omsq - c*sin(om)/om
        m13 = a*c*(1-cos(om))/omsq + b*sin(om)/om
    
        m21 = a*b*(1-cos(om))/omsq + c*sin(om)/om
        m22 = ((a**2+c**2)*cos(om) + b**2) / omsq
        m23 = b*c*(1-cos(om))/omsq - a*sin(om)/om
    
        m31 = a*c*(1-cos(om))/omsq - b*sin(om)/om
        m32 = b*c*(1-cos(om))/omsq + a*sin(om)/om
        m33 = ((a**2+b**2)*cos(om) + c**2) / omsq

        return np.array([[m11, m12, m13],
                         [m21, m22, m23],
                         [m31, m32, m33]])


