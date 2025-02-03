import gymnasium as gym
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from mountaincar_wrapper import MountainCarWrapper
import numpy as np
import matplotlib.pyplot as plt
from myutils import test_env
from rl import RL
import time

def mc_QL_grid_seach(input_tuple):


    (seed1, init_alpha, decay_eps, min_eps, n_episodes, output_file) = input_tuple
    print(seed1, init_alpha, decay_eps, min_eps, n_episodes, file=open(output_file+'_progress.txt', 'a'))
    
    base_env = gym.make('MountainCar-v0', render_mode=None)
    mountaincar = MountainCarWrapper(base_env, position_bins=50, velocity_bins=50, model=False)
    
    tic = time.perf_counter()
    _, _, pi, _, _ = RL(mountaincar).q_learning(nS=mountaincar.observation_space, nA=mountaincar.action_space.n,
                                                gamma=0.99, init_alpha=init_alpha, epsilon_decay_ratio=decay_eps, min_epsilon=min_eps,
                                                n_episodes=n_episodes, seed=seed1)
    toc = time.perf_counter()
    
    #test policy
    test_scores = test_env(env=mountaincar, n_iters=1000, pi=pi)
    print(np.mean(test_scores))
    print(np.std(test_scores))
    
    print(seed1, init_alpha, decay_eps, min_eps, n_episodes, np.mean(test_scores), toc-tic, file=open(output_file+'_results.txt', 'a'))
    
    return
    
def mc_expected_reward(input_tuple):
    (env_size, e, pi_e, n_episodes, output_file) = input_tuple
    
    base_env = gym.make('MountainCar-v0', render_mode=None)
    env = MountainCarWrapper(base_env, position_bins=env_size[0], velocity_bins=env_size[1], model=False)
    pi = {s:a for s, a in enumerate(pi_e)}
    test_scores = test_env(env=env, n_iters=n_episodes, pi=pi)
    print(e, np.mean(test_scores), np.std(test_scores), file=open(output_file, 'a'))
    return