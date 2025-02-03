"""
discretized MountainCar transition and reward (P) matrix. Adjusted from cartpole_model.py example of bettermdptools.

Example usage:
dcar = DiscretizedMountainCar(10, 10)  # Example bin sizes for each variable

"""

import numpy as np
import gymnasium as gym
from mountain_car import MountainCarEnv 

class DiscretizedMountainCar:
    def __init__(self, env,
                 position_bins,
                 velocity_bins, model):

        """
         Initializes the DiscretizedMountainCar model.

         Parameters:
         - gymnasium enviroment
         - position_bins (int): Number of discrete bins for the cart's position.
         - velocity_bins (int): Number of discrete bins for the cart's velocity.

         Attributes:
         - state_space (int): Total number of discrete states in the environment.
         - P (dict): Transition probability matrix where P[state][action] is a list of tuples (probability, next_state,
         reward, done).
         - transform_obs (lambda): Function to transform continuous observations into a discrete state index.
         """
        self.env = env
        
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.action_space = 3  # Accelerate to the left, or don't accelerate, or accelerate to the right (0, 1, 2)

        self.model = model

        # Define the range for each variable
        self.position_range = (self.env.observation_space.low[0], self.env.observation_space.high[0])
        self.velocity_range = (self.env.observation_space.low[1], self.env.observation_space.high[1])

        self.state_space = np.prod([self.position_bins, self.velocity_bins])
        if self.model:
            self.P = {state: {action: [] for action in range(self.action_space)} for state in range(self.state_space)}
            self.setup_transition_probabilities()
        self.n_states = self.velocity_bins*self.position_bins
        """
        Explanation of transform_obs lambda: 
        This lambda function will take cartpole observations, determine which bins they fall into, 
        and then convert bin coordinates into a single index.  This makes it possible 
        to use traditional reinforcement learning and planning algorithms, designed for discrete spaces, with continuous 
        state space environments. 
        
        Parameters:
        - obs (list): A list of continuous observations [position, velocity].

        Returns:
        - int: A single integer representing the discretized state index.
        """
        self.transform_obs = lambda obs: (
            np.ravel_multi_index((
                np.clip(np.digitize(obs[0], np.linspace(*self.position_range, self.position_bins)) - 1, 0,
                        self.position_bins - 1),
                np.clip(np.digitize(obs[1], np.linspace(*self.velocity_range, self.velocity_bins)) - 1, 0,
                        self.velocity_bins - 1)
            ), (self.position_bins, self.velocity_bins))
        )

    def setup_transition_probabilities(self):
        """
        Sets up the transition probabilities for the environment. This method iterates through all possible
        states and actions, simulates the next state, and records the transition probability
        (deterministic in this setup), reward, and termination status.
        """
        for state in range(self.state_space):
            position_idx, velocity_idx = self.index_to_state(state)
            position = np.linspace(*self.position_range, self.position_bins)[position_idx]
            velocity = np.linspace(*self.velocity_range, self.velocity_bins)[velocity_idx]
            #print(f'state_idx={state}: pos_idx={position_idx}, vel_idx={velocity_idx}, pos={position}, vel={velocity}') 
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(position, velocity, action)
                self.P[state][action].append((1, next_state, reward, done))

    def index_to_state(self, index):
        """
        Converts a single index into a multidimensional state representation.

        Parameters:
        - index (int): The flat index representing the state.

        Returns:
        - list: A list of indices representing the state in terms of position, velocity, angle, and angular velocity bins.
        """
        #totals = [self.position_bins, self.velocity_bins]
        #multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        #components = [int((index // multipliers[i]) % totals[i]) for i in range(2)]
        #return components
        return np.unravel_index(index, (self.position_bins, self.velocity_bins), order='C')
        

    def compute_next_state(self, position, velocity, action):
        """
        Compute the next state from Gymnasium env based on the current state indices and the action taken.
        
        Parameters:
        - env
        - action

        Returns:
        - tuple: Contains the next state index, the reward, and the done flag indicating if the episode has ended.
        """
        #env2 = self.env
        #env2.reset()
        #print (f'env.reset state: {env2.state[0]}, {env2.state[1]}')
        #env2.state = (position, velocity)
        #print (f'Set env.state: {env2.state[0]}, {env2.state[1]}')
        #new_state, reward, terminated, truncated, _ = env2.step(action)
        
        env2 = MountainCarEnv()
        env2.reset()
        env2.state = (position, velocity)
        new_state, reward, terminated, truncated, _ = env2.step(action)

        done = terminated or truncated
        new_position = new_state[0]
        new_velocity = new_state[1]
        
        new_position_idx = np.clip(np.digitize(new_position, np.linspace(*self.position_range, self.position_bins)) - 1, 0, self.position_bins-1)
        new_velocity_idx = np.clip(np.digitize(new_velocity, np.linspace(*self.velocity_range, self.velocity_bins)) - 1, 0, self.velocity_bins-1)
        
        #print(f' action={action}: pos_idx={new_position_idx}, vel_idx={new_velocity_idx}, pos={new_position}, vel={new_velocity}, {done}')
        
        new_state_idx = np.ravel_multi_index((new_position_idx, new_velocity_idx),
                                             (self.position_bins, self.velocity_bins))
        
        return new_state_idx, reward, done