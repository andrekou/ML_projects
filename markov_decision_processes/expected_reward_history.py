import numpy as np
from multiprocessing import Pool
import workers_a4
import os

def expected_reward_history(space_dims, pi_track, output_step, n_episodes=1000, nproc=4):
    e_track = list(range(0, len(pi_track), output_step))
    if e_track[-1] < len(pi_track)-1: e_track += [len(pi_track)-1]
    parameters = []
    output_file = 'temp'
    for e in e_track:
        parameters.append((space_dims, e, pi_track[e], n_episodes, output_file))
        
    if __name__ ==  '__main__':
        p=Pool(processes = nproc)
        p.map(workers_a4.mc_expected_reward, parameters)

    R_track = np.loadtxt(output_file, delimiter=' ')
    R_track = R_track[R_track[:, 0].argsort()]
    os.remove(output_file)

    return R_track
    
def bj_expected_reward_history(pi_track, output_step=1, n_episodes=10000, nproc=4):
    e_track = list(range(0, len(pi_track), output_step))
    if e_track[-1] < len(pi_track)-1: e_track += [len(pi_track)-1]
    parameters = []
    output_file = 'temp'
    for e in e_track:
        parameters.append((e, pi_track[e], n_episodes, output_file))
        
    if __name__ ==  '__main__':
        p=Pool(processes = nproc)
        p.map(workers_a4.mc_expected_reward, parameters)

    R_track = np.loadtxt(output_file, delimiter=' ')
    R_track = R_track[R_track[:, 0].argsort()]
    os.remove(output_file)

    return R_track