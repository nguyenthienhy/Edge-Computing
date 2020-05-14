import gym
import discrete
import numpy as np
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('offload-autoscale-discrete-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

rewards_list = []
avg_rewards = []
obs = env.reset()
t = 0
for i in range(10000):
    action, _states = model.predict(obs)
    # action = np.random.uniform(0, 1, 1)
    obs, rewards, dones, info = env.step(action)
    rewards_list.append(1 / rewards)
    avg_rewards.append(np.mean(rewards_list[:]))
    if dones: env.reset()
    t += 1
    # env.render()
import matplotlib.pyplot as plt
df=pd.DataFrame({'x': range(10000), 'y1': rewards_list, 'y2': avg_rewards})

plt.plot( 'x', 'y1', data=df, marker='', color='skyblue', linewidth=1, label="reward")
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=1, label="average_rewards")

plt.legend()

plt.show()
