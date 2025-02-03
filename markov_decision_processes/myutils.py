import numpy as np


def test_env(env, n_iters, pi, convert_state_obs=lambda state: state):
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

#    
#    
#    
#    
#    
#    
#    
#    def expected_reward_history(env, pi_track, output_step, n_episodes=1000):
#    R_track = []
#    e_track = []
#    for e in range(len(pi_track)):
#        if (e % output_step == 0.0 or e==len(pi_track)-1):
#            # Test policy and track total reward
#            pi = {s:a for s, a in enumerate(pi_track[e])}
#            test_scores = test_env(env=env, n_iters=n_episodes, pi=pi)
#            e_track.append(e)
#            R_track.append(np.mean(test_scores))
#    return e_track, R_track