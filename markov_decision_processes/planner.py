"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Class that contains functions related to planning algorithms (Value Iteration, Policy Iteration). 
Planner init expects a reward and transitions matrix P, which is nested dictionary gym style discrete environment 
where P[state][action] is a list of tuples (probability, next state, reward, terminal).

Model-based learning algorithms: Value Iteration and Policy Iteration
"""

import numpy as np
import warnings


class PlannerR:
    def __init__(self, P, env):
        self.P = P
        self.env = env

    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_iters, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        pi_track = []
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            V = np.max(Q, axis=1)
            V_track[i] = V
            pi_track.append(np.argmax(Q, axis=1))
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")

        pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
        return V, V_track, pi, pi_track

    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_iters, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        pi_track = []
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            i += 1
            old_pi = pi
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i] = V
            pi_track.append(list(pi.values()))
            #
            pi = self.policy_improvement(V, gamma)
            if old_pi == pi:
                converged = True
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        return V, V_track, pi, pi_track

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        new_pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return new_pi
        
    def test_env(self, env, n_iters, pi, convert_state_obs=lambda state: state):
        test_scores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, info = env.reset(seed=i)
            done = False
            state = convert_state_obs(state)
            total_reward = 0
            while not done:
                action = pi[state]
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = convert_state_obs(next_state)
                state = next_state
                total_reward += reward
            test_scores[i] = total_reward
            env.close()
        return test_scores