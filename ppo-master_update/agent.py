from socket import socket
import gym
import gym_offload_autoscale
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

start_time = time.time()

rand_seed = 1234
x = 0.5

# vì đây là miền liên tục nên các giá trị Q_values sẽ được học qua từng vòng lặp để cải thiện
# việc lựa chọn action ở các bước tiếp theo

# Q learning
class N_step_learning:

    def __init__(self, observation_space, action_space):
        self.gamma = 0.1
        self.alpha = 0.1
        self.n_step = 10
        self.max_find_q_min = 400
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=1000000)
        # the neural network
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space + action_space , ), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

    # remember, for exploitation
    def remember(self, state, q_values):
        self.memory.append((state, q_values))

    def chooseAction(self, state , max_find_q_min):  # từ một trạng thái lựa chọn một hành động
        iter = 0
        Q = []
        A = []
        while iter <= max_find_q_min:
            action = random.random() # chọn random action trong khoảng [0 , 1]
            q_values = self.model.predict(np.array([np.append(state[0] , action)]))
            Q.append(q_values)
            A.append(action)
            iter += 1

        min_q = min(Q)
        for i in range(len(Q)):
            if Q[i] == min_q:
                return A[i]

    def get_Q_values(self, state, action):
        state_temp = np.array([np.append(state[0] , action)])
        q_update = self.model.predict(state_temp)[0]
        return q_update


def agent(alpha , gamma , n_step , max_find_q_min):
    np.random.seed(3)
    # dqn
    rewards_list_dqn = []
    avg_rewards_dqn = []
    rewards_time_list_dqn = []
    avg_rewards_time_list_dqn = []
    rewards_bak_list_dqn = []
    avg_rewards_bak_list_dqn = []
    rewards_bat_list_dqn = []
    avg_rewards_bat_list_dqn = []
    avg_rewards_energy_list_dqn = []
    dqn_data = []
    t_range = 2000

    env = gym.make('offload-autoscale-v0', p_coeff=x)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    solver = N_step_learning(observation_space, action_space)
    solver.alpha = alpha
    solver.gamma = gamma
    solver.n_step = n_step
    solver.max_find_q_min = max_find_q_min

    for _ in range(96):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        states = [state]
        action = solver.chooseAction(state , max_find_q_min) # lựa chọn hành động ban đầu theo chính sách pi
        actions = [action]
        rewards = [0]
        num_hour_run = 0
        t = 0
        T = math.inf # thời gian tối đa chạy 1 episode
        while True:
            num_hour_run += 1
            if t < T:

                next_state, reward, _ , _ = env.step(action)
                next_state = np.reshape(next_state, [1, observation_space])
                states.append(state)
                rewards.append(reward)
                t_, bak, bat = env.render()
                rewards_list_dqn.append(1 / reward)
                avg_rewards_dqn.append(np.mean(rewards_list_dqn[:]))
                rewards_time_list_dqn.append(t_)
                avg_rewards_time_list_dqn.append(np.mean(rewards_time_list_dqn[:]))
                rewards_bak_list_dqn.append(bak)
                avg_rewards_bak_list_dqn.append(np.mean(rewards_bak_list_dqn[:]))
                rewards_bat_list_dqn.append(bat)
                avg_rewards_bat_list_dqn.append(np.mean(rewards_bat_list_dqn[:]))
                avg_rewards_energy_list_dqn.append(avg_rewards_bak_list_dqn[-1] + avg_rewards_bat_list_dqn[-1])
                dqn_data.append([avg_rewards_time_list_dqn[-1], avg_rewards_bak_list_dqn[-1], avg_rewards_bat_list_dqn[-1]])

                if num_hour_run >= 24:
                    T = t + 1
                else:
                    action = solver.chooseAction(next_state , max_find_q_min)
                    actions.append(action)

            tau = t - n_step + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n_step + 1, T + 1)):
                    G += np.power(gamma, i - tau - 1) * rewards[i]
                if tau + solver.n_step < T:
                    Q = solver.get_Q_values(states[tau + n_step], actions[tau + n_step])
                    G += np.power(gamma, n_step) * Q
                Q_prev = solver.get_Q_values(states[tau], actions[tau])
                Q_prev += alpha * (G - Q_prev)
                state_input = np.array([np.append(states[tau][0] , actions[tau])])
                solver.memory.append((state_input[0], Q_prev))
                solver.model.fit(state_input , [[Q_prev]])

            if tau == T - 1:
                break

            t += 1

    plotGraphCost(alpha , gamma , n_step , max_find_q_min , avg_rewards_dqn , avg_rewards_time_list_dqn , avg_rewards_bak_list_dqn , avg_rewards_bat_list_dqn , t_range)
    return avg_rewards_dqn[-1]

def tuning_parameters():
    alpha_array = [0.1]
    gamma_array = [0.1]
    n_steps = list(range(1 , 10))
    max_find_q_mins = list(range(10 , 500 , 5))

    return alpha_array , gamma_array , n_steps , max_find_q_mins

def runningTuning():
    alpha_optimize , gamma_optimize , n_step_optimize , max_find_optimize = 0 , 0 , 0 , 0
    # tuning alpha_array
    alpha_arrays , gamma_array , n_steps , max_find_q_mins , = tuning_parameters()
    min_Cost = math.inf
    for alpha in alpha_arrays:
        for gamma in gamma_array:
            for n_step in n_steps:
                for max_find in max_find_q_mins:
                    min_Cost = min(min_Cost , agent(alpha , gamma , n_step , max_find))
    for alpha in alpha_arrays:
        for gamma in gamma_array:
            for n_step in n_steps:
                for max_find_iter in max_find_q_mins:
                    if min_Cost == agent(alpha , gamma , n_step , max_find_iter):
                        alpha_optimize , gamma_optimize , n_step_optimize , max_find_optimize = alpha , gamma , n_step , max_find_iter
    return min_Cost , alpha_optimize , gamma_optimize , n_step_optimize , max_find_optimize

def plotGraphCost(alpha , gamma , n_step , max_find_q_min , avg_rewards_dqn , avg_rewards_time_list_dqn , avg_rewards_bak_list_dqn , avg_rewards_bat_list_dqn , t_range):

    print('--RESULTS--')
    print('{:15}{:<30}{:<10.5}{:<10.5}{:<10.5}'.format('dqn',avg_rewards_dqn[-1], avg_rewards_time_list_dqn[-1], avg_rewards_bak_list_dqn[-1], avg_rewards_bat_list_dqn[-1]))
    end_time = time.time()
    print('elapsed time:', end_time-start_time)
    print(len(avg_rewards_dqn[: t_range]))

    #total cost
    df=pd.DataFrame({'x': range(t_range), 'y_6': avg_rewards_dqn[ : t_range]})
    df.transpose()
    plt.xlabel("Time Slot")
    plt.ylabel("Time Average Cost")
    plt.plot('x', 'y_6', data=df, marker='x', markevery = int(t_range/10), color='green', linewidth=1, label="n-step learning")
    plt.legend()
    plt.grid()
    plt.savefig("./results/" + "alpha = " + str(alpha) + ", " + "gamma = " + str(gamma) + ", " + "n_step = " + str(n_step) + ", " + "max_find_q_min = " + str(max_find_q_min) + ".png")
    plt.close()

min_Cost , alpha_optimize , gamma_optimize , n_step_optimize , max_find = runningTuning()

with open("parameters.txt" , "a") as f:
    f.write(str(min_Cost) + " " + str(alpha_optimize) + " " + str(gamma_optimize) + " " + str(n_step_optimize) + " " + str(max_find))