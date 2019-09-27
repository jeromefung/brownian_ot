import numpy as np


def run_simulation(particle, fnamebase,
                   n_steps, dt, save_int = 0):
    '''
    Inputs:
    particle: instance of Particle class
    fnamebase: base filename
    '''

    if save_int == 0:
        file_len = n_steps + 1
        outfnumber = 0 # will be made 1 later
    else:
        file_len = save_int

        
    def preallocate():
        return np.zeros((file_len, 7)) # com coords, quaternion

        
    # initialize 
    output = preallocate()
    output[0] = particle._nice_output()

    # main loop
    for ctr in np.arange(1, n_steps + 1):
        # step the particle
        particle.update(dt)

        if ctr % file_len == 0:
            outfnumber = np.floor(ctr / file_len).astype('int')
            outfname = fnamebase + str(outfnumber).zfill(4) + '.npy'
            np.save(outfname, output)
            output = preallocate()

        output[ctr % file_len] = particle._nice_output()
        

    # extract nonzero rows of output (e.g., nonzero unit quaternions)
    output = output[np.where(output.any(axis=1))[0]]
    # save for the last time
    outfname = fnamebase + str(outfnumber + 1).zfill(4) + '.npy'
    np.save(outfname, output)
    
                    
        

