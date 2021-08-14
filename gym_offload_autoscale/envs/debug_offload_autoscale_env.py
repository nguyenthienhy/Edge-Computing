import gym
import math
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class OffloadAutoscaleEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.timeslot = 0.25  # hours, ~15min
        self.batery_capacity = 2000  # kWh
        self.server_service_rate = 20  # units/sec

        self.lamda_high = 100  # units/second
        self.lamda_low = 10
        self.b_high = self.batery_capacity / self.timeslot  # W
        self.b_low = 0
        self.h_high = 0.06  # ms/unit
        self.h_low = 0.02
        self.e_low = 0
        self.e_high = 2
        self.back_up_cost_coef = 0.15
        self.normalized_unit_depreciation_cost = 0.01
        self.max_number_of_server = 10

        # power model
        self.d_sta = 300
        self.coef_dyn = 10
        self.server_power_consumption = 150
        self.b_com = 10

        r_high = np.array([
            self.lamda_high,
            self.b_high,
            self.h_high,
            self.e_high])
        r_low = np.array([
            self.lamda_low,
            self.b_low,
            self.h_low,
            self.e_low])
        self.observation_space = spaces.Box(low=r_low, high=r_high)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = [0, 0, 0, 0]
        self.time = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Transition functions
    def get_lambda(self):
        return np.random.uniform(self.lamda_low, self.lamda_high)
    def get_b(self, state, g, d_op, d):
        b = state[1]
        # print('\t', end = '')
        if d_op > b:
            # print('unused batery')
            return b + g
        else:
            if g >= d:
                # print('recharge batery')
                return np.maximum(self.b_high, b + g - d)
            else:
                # print('discharge batery')
                return b + g - d
    def get_h(self):
        return np.random.uniform(self.h_low, self.h_high)
    def get_e(self):
        if self.time >= 9 and self.time < 15:
            return 2
        if self.time < 6 or self.time >= 18:
            return 0
        return 1


    def get_time(self):
        self.time += 0.25
        if self.time == 24:
            self.time = 0
    def get_g(self, e):
        if e == 0:
            return np.random.exponential(60)
        if e == 1:
            return np.random.normal(520, 130)
        return np.random.normal(800, 95)

    def check_constraints(self, m, mu, lamda):
        if mu > lamda or mu < 0: return False
        if isinstance(mu, complex): return False
        if m * self.server_service_rate <= mu: return False
        return True
    def cost_delay_local_function(self, m, mu):
        if m == 0 and mu == 0: return 0
        return mu / (m * self.server_service_rate - mu)
    def cost_delay_cloud_function(self, mu, h, lamda):
        return (lamda - mu) * h 
    def cost_function(self, m, mu, h, lamda):
        return self.cost_delay_local_function(m, mu) + self.cost_delay_cloud_function(mu, h, lamda)
    def  get_m_mu(self, de_action):
        lamd, _, h, _ = self.state
        opt_val = math.inf
        ans = [-1, -1]
        for m in range(1, self.max_number_of_server + 1):
            normalized_min_cov = self.lamda_low
            # coeff = [1, 1, (self.server_power_consumption * m - de_action) / self.server_power_consumption *  normalized_min_cov]
            # roots = np.roots(coeff)
            # for i in range(2):
            # mu = roots[i]
            mu = (de_action - self.server_power_consumption * m) * normalized_min_cov / self.server_power_consumption
            valid = self.check_constraints(m, mu, lamd)
            if valid:
                if self.cost_function(m, mu, h, lamd) < opt_val:
                    ans = [m, mu]
                    opt_val = self.cost_function(m, mu, h, lamd)
        return ans
    # power
    def get_dop(self):
        return self.d_sta + self.coef_dyn * self.state[0]
    def get_dcom(self, m, mu):
        normalized_min_cov = self.lamda_low
        return self.server_power_consumption * m + self.server_power_consumption / normalized_min_cov * mu
    
    def cal(self, action):
        lamda, b, h, _ = self.state
        d_op = self.get_dop()
        if b <= d_op + 150:
            return [0, 0]
        else:
            low_bound = 150
            high_bound = np.minimum(b - d_op, self.get_dcom(10, lamda))
            de_action = low_bound + action * (high_bound - low_bound)
            # print('deaction ', de_action)
            return self.get_m_mu(de_action)

    def reward_func(self, g, d_op, d, m, mu):
        lamda, b, h, _ = self.state
        cost_delay_wireless = 0
        cost_delay = self.cost_function(m, mu, h, lamda) + cost_delay_wireless
        if d_op > b:
            cost_batery = 0
            cost_bak = self.back_up_cost_coef * d_op
        else:
            cost_batery = self.normalized_unit_depreciation_cost * np.maximum(d - g, 0)
            cost_bak = 0
        cost = cost_delay + cost_batery + cost_bak

        cost_delay_local = self.cost_delay_local_function(m, mu)
        cost_delay_cloud = self.cost_delay_cloud_function(mu, h, lamda)
        # print('\t{:20} {:20} {:20} {:10}'.format("cost_delay_local", "cost_delay_cloud", "cost_batery", "cost_bak"))
        # print('\t{:<20.3f} {:<20.2f} {:<20.2f} {:<10.2f}'.format(cost_delay_local, cost_delay_cloud, cost_batery, cost_bak))
        return cost

    def step(self, action):
        done = False
        action = float(action)
        self.get_time()
        state = self.state
        # print('\tstate: ',state)
        # print('\ttime: ',self.time)
        g_t = self.get_g(state[3])
        # print('\tget ', g_t)
        # print('\taction: ', action)

        d_op = self.get_dop()
        number_of_server, local_workload = self.cal(action) 
        d_com = self.get_dcom(number_of_server, local_workload)
        d = d_op + d_com
        # print('\t{:20}{:20}{:20}{:20}{:10}'.format('d_op','d_com','d','number_server','local_workload'))
        # print('\t{:<20.3f}{:<20.3f}{:<20.3f}{:<20.3f}{:<10.3f}'.format(d_op, d_com, d, number_of_server, local_workload))
        reward = self.reward_func(g_t, d_op, d, number_of_server, local_workload)
        lambda_t = self.get_lambda()
        b_t = self.get_b(state, g_t, d_op, d) 
        h_t = self.get_h()
        e_t = self.get_e()
        self.state = np.array([lambda_t, b_t, h_t, e_t])
        # print('\tnew state: ', self.state)
        # print('\tcost: ', reward)
        if b_t <= 0:
            done = True
        return self.state, 1 / reward, done, {}

    def reset(self):
        self.state = np.array([self.lamda_low, self.b_low, self.h_low, self.e_low])
        self.time = 0
        return self.state

MyEnv = OffloadAutoscaleEnv()
MyEnv.reset()
for i in range(10000):
    # print('STEP: ', i)
    state, reward, done, info = MyEnv.step(MyEnv.action_space.sample())
    if done: print(i, 'done')