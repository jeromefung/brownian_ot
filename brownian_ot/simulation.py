import numpy as np


def run_simulation(particle, n_steps, dt,
                   save = True, fnamebase = None, save_int = 0):

    '''
    Inputs:
    particle: instance of Particle class
    fnamebase: base filename
    '''

    # TODO: want to refactor to make saving to a file optional (makes
    # testing easier)

    if not save:
        file_len = n_steps + 1
    else:
        if save_int == 0:
            file_len = n_steps + 1
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

        if save:            
            if ctr % file_len == 0:
                outfnumber = np.floor(ctr / file_len).astype('int')
                outfname = fnamebase + str(outfnumber).zfill(4) + '.npy'
                np.save(outfname, output)
                output = preallocate()

        output[ctr % file_len] = particle._nice_output()
        

    # extract nonzero rows of output (e.g., nonzero unit quaternions)
    output = output[np.where(output.any(axis=1))[0]]
    # save for the last time
    if save:
        if save_int == 0:
            outfname = fnamebase + '.npy'
        else:
            outfname = fnamebase + str(outfnumber + 1).zfill(4) + '.npy'
        np.save(outfname, output)


    return output
    
                    
        

