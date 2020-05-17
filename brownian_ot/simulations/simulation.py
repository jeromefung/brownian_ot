import numpy as np


class Simulation:
    '''
    Base class. 
    '''

    def __init__(self, particle, timestep, f_ext):
        self.particle = particle
        self.timestep = timestep
        self.f_ext = f_ext
    
    def run(self, nsteps, outfname = None):
        # Preallocate
        file_len = n_steps + 1
        output = np.zeros((file_len, 7)) # com coords, quaternion
        output[0] = self.particle._nice_output() # First row is initial position

        # main loop
        for ctr in np.arange(1, n_steps + 1):
            # step the particle
            particle.update(dt, self.f_ext)
            output[ctr] = particle._nice_output()

        if outfname is not None:
            # Check if there's a .npy extension
            if outfname[-4:] != '.npy':
                outfname = outfname + '.npy'
            np.save(outfname, output)
            
        return output
    

class FreeDiffusionSimulation(Simulation):
    '''
    '''
    def __init__(self, particle, timestep):
        super().__init__(particle, timestep, None)

        
class OTSimulation(Simulation):
    def __init__(self, particle, beam, timestep):
        super().__init__(particle, timestep, make_ots_force(particle, beam))

