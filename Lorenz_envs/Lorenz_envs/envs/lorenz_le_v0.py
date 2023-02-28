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

class Lorenzle(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super(Lorenzle, self).__init__() 
        # self.max_speed = 8
        # self.max_torque = 2.0
        # self.dt = 0.05
        # self.g = g
        # self.m = 1.0
        # self.l = 1.0
        self.collected_states = list()
        # self.goal_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # self.goal_state = np.array([-8.485, -8.485, 27], dtype=np.float32)
        self.goal_state = np.array([8.485, 8.485, 27], dtype=np.float32)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.scatter(self.goal_state[0],self.goal_state[1],self.goal_state[2], s=100, color="red")

        self.T = 1000
        self.close_to_goal = True
        self.sphere_R = 2
        self.sigma = 10
        self.r = 28
        self.b = 8/3
        self.C = ((self.b ** 2) * ((self.sigma + self.r) ** 2)) / (4 * (self.b - 1))
        self.alpha = [100, 100, 100]
        self.dt = 0.01
        self.dyn = {"sigma":self.sigma , "R":self.r, "b": self.b}
        self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        action_range_high = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        action_range_low = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
        self.high_env_range = np.array([39.0, 39.0, 77.0], dtype=np.float32)
        self.low_env_range = np.array([-39.0, -39.0, -0.01], dtype=np.float32)
        self.action_space = spaces.Box(low=action_range_low, high=action_range_high, shape = (3,),dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_env_range, high=self.high_env_range, shape = (3,),dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def linearized_lorenz (self, x0, dyn, y_lorenz):
        sigma = dyn['sigma']
        R = dyn['R']
        b = dyn['b']
        x = x0[0]
        y = x0[1]
        z = x0[2]
        return np.array([sigma * (y - x), 
                        (R - y_lorenz[2])*x - y - y_lorenz[0]*z,
                        y_lorenz[1]*x + y_lorenz[0]*y - b*z])

    def lorenz (self, x0, dyn, action):
        sigma = dyn['sigma']
        R = dyn['R']
        b = dyn['b']
        x = x0[0]
        y = x0[1]
        z = x0[2]
        # print ('action in lorenz:', self.alpha*action)
        # print ('x0:', x0)
        return np.array([sigma * (y - x) + self.alpha[0]*action[0], 
                         x * (R - z) - y + self.alpha[1]*action[1], 
                         x * y - b * z + self.alpha[2]*action[2]])

    def RungeKutta (self, dyn, f, dt, x0, action):

        #try 2 options

        # k1 = f(x0, dyn, action)*dt #[x,y,z]*0.1 example
        # k2 = f(x0+0.5*k1*dt,dyn, [0,0,0])*dt
        # k3 = f(x0 + 0.5*k2*dt, dyn, [0,0,0])*dt
        # k4 = f(x0 + k3*dt, dyn, [0,0,0])*dt
        #print (action/4)

        # k1 = f(x0.copy(), dyn, action/4.0)*dt #[x,y,z]*0.1 example
        # k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)*dt
        # k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)*dt
        # k4 = f(x0.copy() + k3*dt, dyn, action/4.0)*dt

        # x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6

        k1 = f(x0.copy(), dyn, action/4.0) #[x,y,z]*0.1 example
        k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)
        k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)
        k4 = f(x0.copy() + k3*dt, dyn, action/4.0)

        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        # k1 = f(x0, dyn, np.array([0.0, 0.0, 0.0]))*dt #[x,y,z]*0.1 example
        # k2 = f(x0+0.5*k1*dt,dyn, np.array([0.0, 0.0, 0.0]))*dt
        # k3 = f(x0 + 0.5*k2*dt, dyn, np.array([0.0, 0.0, 0.0]))*dt
        # k4 = f(x0 + k3*dt, dyn, np.array([0.0, 0.0, 0.0]))*dt

        # x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6 + np.array([self.alpha[0]*action[0], self.alpha[1]*action[1],  self.alpha[0]*action[0]])*dt
        return x
    
    def RungeKutta_linearized (self, dyn, f, dt, x0, y):
        k1 = f(x0, dyn, y) #[x,y,z]*0.1 example
        k2 = f(x0+0.5*k1*dt,dyn, y)
        k3 = f(x0 + 0.5*k2*dt, dyn, y)
        k4 = f(x0 + k3*dt, dyn, y)
        
        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x


    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x = self.RungeKutta(dyn, f, dt, x0, action)
        #x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0, 0.0]))
        return x

    # Calculate sparse reward based on distance from equilibrium return reward
    def sphere_reward (self, state,new_state,action):
        #worked for np.linalg.norm(state - self.goal_state) ** 2 - self.sphere_R ** 2 <= 0.5
        if np.linalg.norm(state - self.goal_state) ** 2 - self.sphere_R ** 2 > 0.0:
            r = -30
        else:
            # r = -self.heuristic_reward_0 ( state, new_state, action)
            r = self.le_reward(state, action)
            # r = -1
            # r = -np.linalg.norm(state - self.goal_state)
            # penalize 
        return r
    
    def le_reward (self, s , action):
        T = self.T
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        cum = np.array([0,0,0])

        x = np.empty(shape=(len(s),T))
        v1_prime = np.empty(shape=(len(s),T))
        v2_prime = np.empty(shape=(len(s),T))
        v3_prime = np.empty(shape=(len(s),T))

        x[:, 0] = s
        v1_prime[:, 0] = v1
        v2_prime[:, 0] = v2
        v3_prime[:, 0] = v3

        for i in range(1,T):

            x[:, i] = self.RungeKutta(self.dyn, self.lorenz, self.dt, x[:, i-1], np.array([0,0,0]))
            v1_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_lorenz, self.dt, v1_prime[:, i-1], x[:, i-1])
            v2_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_lorenz, self.dt, v2_prime[:, i-1], x[:, i-1])
            v3_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_lorenz, self.dt, v3_prime[:, i-1], x[:, i-1])
            
            
            norm1 = np.linalg.norm(v1_prime[:, i])
            v1_prime[:, i] = v1_prime[:, i]/norm1
            
            GSC1 = np.dot(v1_prime[:, i], v2_prime[:, i])
            v2_prime[:, i] = v2_prime[:, i] - GSC1*v1_prime[:, i]
            
            norm2 = np.linalg.norm(v2_prime[:, i])
            v2_prime[:, i] = v2_prime[:, i]/norm2
            
            GSC2 = np.dot(v3_prime[:, i], v1_prime[:, i])
            GSC3 = np.dot(v3_prime[:, i], v2_prime[:, i])
            
            v3_prime[:, i] = v3_prime[:, i] - GSC2*v1_prime[:, i] - GSC3*v2_prime[:, i]
            norm3 = np.linalg.norm(v3_prime[:, i])
            v3_prime[:, i] = v3_prime[:, i]/norm3
            cum = cum + np.log2(np.array([norm1,norm2,norm3]))

        cum = cum/(T*self.dt)
        return cum[0]

    def heuristic_reward_0 (self, s, v, a):
        return np.linalg.norm(s - self.goal_state) ** 2 + 0.5 * (np.linalg.norm(v)**2) + 0.1 * (np.linalg.norm(a)**2)

    def heuristic_reward_8 (self, s, v, a):
        return np.linalg.norm(s - self.goal_state) ** 2 + 0.1 * (np.linalg.norm(v)**2) + 0.01 * (np.linalg.norm(a)**2)

    def step(self, u):
        x = self.state  # th := theta
        #print ('collected states', type(self.collected_states))
        
        # self.action_u = u
        dyn = self.dyn
        f = self.lorenz
        dt = self.dt
        # print ('action in lorenz:', u)
        # print ('state:', x)
        newx = self.f_x ( dyn, f, dt, x, u)
        newx = np.clip(newx, self.low_env_range, self.high_env_range)
        # print ('new state:', newx )
        # input()
        #try more than the position (like 10), decrease second term.
        self.cost = self.sphere_reward(x,newx,u)
        # self.cost = self.le_reward(x, u)
        # self.cost = self.heuristic_reward(x,newx,u)
        # print (np.linalg.norm(self.state - self.goal_state))
        # print('dist',np.linalg.norm(x - self.goal_state))
        # terminated = np.linalg.norm(x - self.goal_state) <= 0.5
        # terminated = np.linalg.norm(x - self.goal_state) <= 1
        # terminated = np.linalg.norm(x - self.goal_state) <= 0.1

        self.state = newx
        #print ('state:', type(self.state))
        #self.collected_states = self.collected_states.append(self.state) 
        #print ('dis betwen state and goal:', np.linalg.norm(np.array([-20.0, -30.0, -1.0]) - self.goal_state))
        
        #print ('reward:', self.cost)
        #self.render()
        return self._get_obs(), self.cost, False, {'velocity': newx, 'action': u}

    def reset(self):
        #put even closer to reward
        if self.close_to_goal == True:
            high = np.array([self.goal_state[0] + 3.0,self.goal_state[1]+ 3.0,self.goal_state[2]+ 3.0], dtype=np.float32)
            low = np.array([self.goal_state[0]-3.0,self.goal_state[1]-3.0, self.goal_state[2]-3.0], dtype=np.float32)  # We enforce symmetric limits.
            # print ('high:', high )
            # print ('low:', low )
            self.state = self.np_random.uniform(low=low, high=high)
            # print (self.state)
        else:
            high = self.high_env_range
            low = self.low_env_range  # We enforce symmetric limits.
            # print ('high:', high )
            # print ('low:', low )
            y3 = self.np_random.uniform(low=low[2], high=high[2])
            new_C = self.C - np.square(y3 - self.r - self.sigma)
            y2 = self.np_random.uniform(low=-np.sqrt(new_C), high=np.sqrt(new_C))
            y23 = np.square(y2) + np.square(y3 - self.r - self.sigma)
            new_C  = self.C - y23
            y1 = self.np_random.uniform(low=-np.sqrt(new_C), high=np.sqrt(new_C))
            # print (np.square(y1) + np.square(y2) + np.square(y3 - self.r - self.sigma))
            self.state = np.array([y1,y2,y3])
        # print ('state:', self.state )
        #self.collected_states = self.collected_states.append(self.state)
        self.last_u = None
        self._render_reset()
        return self._get_obs()


    def _render_reset(self):
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.scatter(self.goal_state[0],self.goal_state[1],self.goal_state[2], s=100, color="red")

    def _get_obs(self):
        obsv = self.state
        #self.collected_states.append(self.state)
        
        return np.array(obsv, dtype=np.float32)

    def render(self, mode="human"):
        x = self.state
        u = self.action_u

        for i, m, k in [(x, 'o', 'green')]:
            self.ax.scatter3D(i[0], i[1], i[2],s=10,c=k,marker=m, alpha=0.5)
        #self.ax.scatter3D(x_noaction_local[0], x_noaction_local[1], x_noaction_local[2], s=10, c='blue', alpha=0.5)
        plt.title('Lorenz attractor')
        plt.draw()
        #plt.show(block=False)
        #self.collected_states = list()
        plt.savefig('Lorenz_ppo.png')

#empowerment guy