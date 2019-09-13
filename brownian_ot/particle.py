import numpy as np
import quaternion

from numpy import sin, cos

class Particle:
    '''
    Stores information about particle being simulated.
    '''

    def __init__(self, D, cod, f_ext, kT,
                 pos = np.zeros(3),
                 orient = quaternion.quaternion(1,0,0,0)):
        self.D = D
        self.cod = cod
        self.f_ext = f_ext
        self.kT = kT
        self.pos = pos
        self.orient = orient # transforms PARTICLE to LAB frames


    def _q_random(self):
        '''
        Calculate random generalized displacement obeying generalized
        Stokes-Einstein relation (up to factor of \Delta t). Recall
        <q_i q_j> = 2D_{ij} \Delta t.
        
        Return array of random displacements.
        '''
        return 2. * np.random.multivariate_normal(np.zeros(6), self.D)

    
    def _nice_output(self):
        return np.concatenate((self.pos,
                               quaternion.as_float_array(self.orient)))

    
    def update(self, dt):
        # calc q^B in particle frame
        q_B = self._q_random()
        
        # calc generalized force in lab frame
        force = self.f_ext(self.pos,
                           quaternion.as_rotation_matrix(self.orient))

        # find vector d from COM to COD, known in particle frame,
        # in lab frame.
        # quaternion package does this by converting quaternion
        # to rotation matrix, but could be done via conjugation
        d = quaternion.rotate_vectors(self.orient, self.cod)

        # correct torque (last 3 elts of generalized force) to be about cod
        # minus sign because d points TO the COD
        force[3:] = force[3:] - np.cross(d, force[0:3])
        
        # convert generalized force from lab to particle frame
        # need inverse of orientation quaternion
        force_pf = np.ravel(quaternion.rotate_vectors(self.orient.inverse(),
                                                      force.reshape((2, -1))))

        # calculate q^D in particle frame
        q_D = np.matmul(self.D, force_pf) / self.kT

        # find q_total, put time step in
        q_total = (q_B + q_D) * dt # still in particle frame
        
        # convert spatial part of generalized displacement to lab frame
        # following Garcia de la Torre, use non-updated orientation
        delta_xyzlab = quaternion.rotate_vectors(self.orient,
                                                 q_total[0:3])
        # if so, COM displaced by this amount. update!
        self.pos = self.pos + delta_xyzlab

        # update orientation quaternion
        infntsml_rotmat = unbiased_rotation(*q_total[3:6])
        infntsml_quat = quaternion.from_rotation_matrix(infntsml_rotmat)
        # see BROWNRIG paper eq. 19
        self.orient = self.orient * infntsml_quat # quaternion composition
    

    
def unbiased_rotation(a, b, c):
    '''
    Calculate unbiased rotation operator given infinitesimal rotation angles
    a, b, c about x, y, and z axes.

    See Beard & Schlick, Biophys. J. (2003), eq. 5.

    Note: typo fixed in 22 element.
    '''
    omsq = a**2 + b**2 + c**2
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
