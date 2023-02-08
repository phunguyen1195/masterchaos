from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
#from gym.utils.renderer import Renderer
from gym.error import DependencyNotInstalled
#from gym.utils.renderer import Renderer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

class Lorenzr(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super(Lorenzr, self).__init__() 
        # self.max_speed = 8
        # self.max_torque = 2.0
        # self.dt = 0.05
        # self.g = g
        # self.m = 1.0
        # self.l = 1.0
        self.collected_states = list()
        self.goal_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection="3d")
        self.ax.scatter(0, 0, 0, s=100, color="red")

        self.sigma = 10
        self.r = 28
        self.b = 8/3
        self.x0 = np.array([10.0, 1.0, 0.0])
        self.c = [0.1,0.1,0.1]
        self.alpha = [100, 100, 100]
        self.dt = 0.01
        self.dyn = {"sigma":self.sigma , "R":self.r, "b": self.b}
        self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        action_range_high = np.array([1.0, 1.0], dtype=np.float32)
        action_range_low = np.array([-1.0, -1.0], dtype=np.float32)
        high_env_range = np.array([20.0, 30.0, 50.0], dtype=np.float32)
        low_env_range = np.array([-20.0, -30.0, -1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=action_range_low, high=action_range_high, shape = (2,),dtype=np.float32)
        self.observation_space = spaces.Box(low=low_env_range, high=high_env_range, shape = (3,),dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def lorenz (self, x0, dyn, action):
        sigma = dyn['sigma']
        R = dyn['R']
        b = dyn['b']
        x = x0[0]
        y = x0[1]
        z = x0[2]
        #print ('action in lorenz:', action)
        return np.array([sigma * (y - x) + self.alpha[0]*action[0], x * (R - z) - y+ self.alpha[1]*action[1], x * y - b * z])


    def RungeKutta (self, dyn, f, dt, x0, action):

        #try 2 options

        # k1 = f(x0, dyn, action)*dt #[x,y,z]*0.1 example
        # k2 = f(x0+0.5*k1*dt,dyn, [0,0,0])*dt
        # k3 = f(x0 + 0.5*k2*dt, dyn, [0,0,0])*dt
        # k4 = f(x0 + k3*dt, dyn, [0,0,0])*dt
        #print (action/4)
        k1 = f(x0, dyn, action/4.0)*dt #[x,y,z]*0.1 example
        k2 = f(x0+0.5*k1*dt,dyn, action/4.0)*dt
        k3 = f(x0 + 0.5*k2*dt, dyn, action/4.0)*dt
        k4 = f(x0 + k3*dt, dyn, action/4.0)*dt

        x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6


        # k1 = f(x0, dyn, np.array([0.0, 0.0, 0.0]))*dt #[x,y,z]*0.1 example
        # k2 = f(x0+0.5*k1*dt,dyn, np.array([0.0, 0.0, 0.0]))*dt
        # k3 = f(x0 + 0.5*k2*dt, dyn, np.array([0.0, 0.0, 0.0]))*dt
        # k4 = f(x0 + k3*dt, dyn, np.array([0.0, 0.0, 0.0]))*dt

        # x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6 + np.array([self.alpha[0]*action[0], self.alpha[1]*action[1],  self.alpha[0]*action[0]])*dt
        return x

    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x = self.RungeKutta(dyn, f, dt, x0, action)
        x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0]))
        return x, x_noaction

    def f_t (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x =  x0 + f(x0, dyn, action)*dt
        x_noaction = x0 + f(x0, dyn, np.array([0.0, 0.0, 0.0]))*dt
        return x, x_noaction

    def step(self, u):
        x = self.state  # th := theta
        #print ('collected states', type(self.collected_states))
        
        self.action_u = u
        dyn = self.dyn
        f = self.lorenz
        dt = self.dt
        # print (u)
        newx, self.x_noaction = self.f_x ( dyn, f, dt, x, u)
        
        #try more than the position (like 10), decrease second term.
        self.cost = np.linalg.norm(x) ** 2 + 2 * (np.linalg.norm(newx)**2) + 0.001 * (np.linalg.norm(u)**2)
        self.state = newx
        #print ('state:', type(self.state))
        #self.collected_states = self.collected_states.append(self.state) 
        #print ('dis betwen state and goal:', np.linalg.norm(np.array([-20.0, -30.0, -1.0]) - self.goal_state))
        terminated = np.linalg.norm(self.state - self.goal_state) < 0.5
        #print ('reward:', self.cost)
        #self.render()
        return self._get_obs(), -self.cost, terminated, False, {}

    def reset(self):
        #put even closer to reward
        high = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        low = np.array([-5.0, -5.0, -.5], dtype=np.float32)  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        #self.collected_states = self.collected_states.append(self.state)
        self.last_u = None
        self._render_reset()
        return self._get_obs()


    def _render_reset(self):
        self.ax = self.fig.gca(projection="3d")
        self.ax.scatter(0, 0, 0, s=100, color="red")

    def _get_obs(self):
        obsv = self.state
        #self.collected_states.append(self.state)
        
        return np.array(obsv, dtype=np.float32)

    def render(self, mode="human"):
        x = self.state
        u = self.action_u
        x_noaction_local = self.x_noaction

        print ("no action:", x_noaction_local)
        print ("with action:", x)
        print ('action:', u)
        for i, m, k in [(x, 'o', 'green'), (x_noaction_local, '^', 'blue')]:
            self.ax.scatter3D(i[0], i[1], i[2],s=10,c=k,marker=m, alpha=0.5)
        #self.ax.scatter3D(x_noaction_local[0], x_noaction_local[1], x_noaction_local[2], s=10, c='blue', alpha=0.5)
        plt.title('Lorenz attractor')
        plt.draw()
        #plt.show(block=False)
        #self.collected_states = list()
        plt.savefig('Lorenz_ppo_2d.png')

#empowerment guy