import gym
import discrete
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.zip_space = zip(itertools.repeat(observation_space), itertools.repeat(action_space))

        self.cost_exp = {0 for (s, a) in self.zip_space}
        self.normal_value = {0 for s in self.observation_space}
        # the space of pds is the same with the observation space
        self.pds_value = {0 for s in self.observation_space}

        # parameters
        # Value for delta = ? and learning rate = ? and alpha = ?
        self.delta = 0.9
        self.learning_rate = 0.9995
        self.alpha = 0.9

    def act(self, state):
        action = int(1e9)
        for a in self.action_space:
            action = min(action, self.cost_exp + self.delta * self.pds_value)
        return action

    # Batch update
    @property
    def update(self, time, actual_cost, env_state, action, d_op):
        updated_cost_exp = (1 - np.pow(self.learning_rate, time)) * self.cost_exp + np.pow(self.learning_rate, time) * actual_cost
        for (s, a) in self.zip_space:
            self.cost_exp[(s, a)] = updated_cost_exp

        updated_val_for_norm_cost = int(1e9)
        for a in self.action_space:
            updated_val_for_norm_cost = np.min(updated_val_for_norm_cost, updated_cost_exp + self.delta * self.pds_value)
        for s in self.observation_space:
            if s[3] == env_state:
                self.normal_value[s] = env_state

        for s in self.observation_space:
            if s[1] >= d_op:
                pds_s = max(s[1] - d_op - action, 0)
            self.pds_value[pds_s] = (1 - np.pow(self.alpha, time)) * self.pds_value[pds_s] + np.pow(self.alpha, time) * self.normal_value[s]

reward_list = []
avg_reward_list = []

def offload_autoscale_agent():
    env = gym.make('offload-autoscale-discrete-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    solver = Solver(observation_space, action_space)
    episode = 0
    while True:
        state = env.reset()
        episode += 1
        terminal = False # terminal condition of algo
        while True:
            action = solver.act(state)
            state, reward, done, info = env.step(action)
            reward_list.append(1 / reward)
            avg_reward_list.append(np.mean(reward_list[:]))
            solver.update()
            if done:
                break
        # Do this instead of directly check the epidsodes
        # so we can change the terminal condition
        # if need.
        if episode == 1000:
            terminal = True
        if terminal:
            exit()

offload_autoscale_agent()

df=pd.DataFrame({'x': range(1000), 'y': avg_reward_list})
plt.xlabel("Time Slot")
plt.ylabel("Time Average Cost")
plt.plot( 'x', 'y', data=df, marker='', color='skyblue', linewidth=1, label="algo")
