import gym
import gym_offload_autoscale


env = gym.make('offload-autoscale-v0')
env = DummyVecEnv([lambda: env])

# f = open("output.txt", "w+")

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

# print(env.observation_space, env.observation_space.high, env.observation_space.low)
# print(env.action_space, env.action_space.high, env.action_space.low)

observation = env.reset()
t = 0
for i in range(10000):
    action, _states = model.predict(observation)
    print("Action ", action)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
    print(t, observation, reward, done)
    t += 1
