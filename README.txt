1. agent.py:
	- Implement PPO2 agent from https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
	- Implement random, myopic, fixed 0.4kW, fixed 1kW, dqn agent
	On environment: 'offload-autoscale-v0'
2. gym_offload_autoscale/envs/offload_autoscale_env.py:
	- Implement 'offload-autoscale-v0' as a custom gym environment (https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) -- this is the continuous version
2. discrete/envs/offload_autoscale_env.py:
	- Implement 'offload-autoscale-discrete-v0' as a custom gym environment -- this is the discrete version
3. plot_compare_p.py:
	- Graph the time component vs. energy component of the optimal PPO total cost for different priority coefficients
