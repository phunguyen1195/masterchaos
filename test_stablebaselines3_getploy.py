import gym
import numpy as np
import Lorenz_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EventCallback
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from stable_baselines3.common.logger import Figure


import numpy as np
from fractions import Fraction
directory = 'lorenz/'


dyn_lorenz = {"sigma":10.0, "R":28.0, "b": 8/3}
x0 = np.array([10.0, 1.0, 0.0])
# x0 = np.array([-8.485, -8.485, 27])
time_step = 500
alpha = 100

x_velocity = []
def lorenz (x0, dyn, action):
    sigma = dyn['sigma']
    R = dyn['R']
    b = dyn['b']
    x = x0[0]
    y = x0[1]
    z = x0[2]
    return np.array([sigma * (y - x), x * (R + alpha*action - z) - y, x * y - b * z])


def RungeKutta (dyn, f, dt, x0, action):
    k1 = f(x0, dyn, action/4)*dt #[x,y,z]*0.1 example
    k2 = f(x0+0.5*k1*dt,dyn, action/4)*dt
    k3 = f(x0 + 0.5*k2*dt, dyn, action/4)*dt
    k4 = f(x0 + k3*dt, dyn, action/4)*dt
    x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    return x

def f_t (dyn, f, dt, x0, T, action):
    x = np.empty(shape=(len(x0),T))
    x[:, 0] = x0     
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1], action) 
    return x

x = f_t(dyn_lorenz, lorenz, 0.01, x0, time_step, 0)

env = gym.make("lorenz_u-v0", infinite=True)
# env = gym.make('Pendulum-v1')

new_x = []
#del model # remove to demonstrate saving and loading

model = PPO.load("ppo_lorenz_0_015rad_100alpha_rreward_fixed")
env.reset()
obs = x[:,-1]
new_x.append(obs)
ac = []
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    new_x.append(obs)
    ac.append(action)
    # env.render()
    if done:
        break
    #   obs = env.reset()


xfinal = np.zeros([3,len(new_x)])
for i in range(len(new_x)):
    xfinal[:,i] = new_x[i]

# [-8.485, -8.485, 27]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection="3d")
ax.scatter3D(x0[0], x0[1], x0[2], s=50,alpha=1)
ax.plot3D(x[0], x[1], x[2], alpha=0.4)
ax.plot3D(xfinal[0], xfinal[1], xfinal[2], c='green', alpha=0.6)
ax.scatter3D(xfinal[0][0], xfinal[1][0], xfinal[2][0], s=50, c='green',alpha=1)
ax.scatter3D(  -8.485, -8.485,  27, s=50, c='red', marker='o')
ax.scatter3D(  8.485,  8.485,  27 , s=100, c='red', marker='o')
ax.scatter3D(  0,  0,  0 , s=100, c='red', marker='o')
# ax.scatter(  0,  0,  0 , s=100, c='red', marker='o')
# [ 0.0070262  -0.03513102  0.03513102]
plt.title('Lorenz attractor')
plt.draw()
plt.savefig('Lorenz_attractor.png')

xfinal_test = xfinal[0][1000:1500]
fig = plt.figure(figsize=(15,15))
ax4 = fig.add_subplot(111)
ax4.set_title('x vs. timestep')
ax4.set_xlabel('time')
ax4.set_ylabel('x')
# start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 100))
ax4.plot(np.array(list(range(len(xfinal_test)))), xfinal_test, c = 'green')
#ax4.axhline(y = 5.6929736, color = 'r', linestyle = 'dashdot')
plt.savefig('Lorenz_attractor_test.png')

a = np.zeros([3,len(ac)])
for i in range(len(ac)):
    a[:,i] = ac[i]
a_test = a[0][1000:1500]
fig = plt.figure(figsize=(15,15))
ax5 = fig.add_subplot(111)
ax5.set_title('action vs. timestep')
ax5.set_xlabel('time')
ax5.set_ylabel('action')
# start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 100))
ax5.plot(np.array(list(range(len(a_test)))), a_test, c = 'black')
#ax4.axhline(y = 5.6929736, color = 'r', linestyle = 'dashdot')
plt.savefig('Lorenz_attractor_action.png')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(221)
ax.set_title('x vs. timestep')
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.plot(np.array(list(range(len(x[2])))), x[0])
ax.plot(np.array(list(range(len(xfinal[2]))))+time_step, xfinal[0])
ax.axhline(y = 8.485, color = 'r', linestyle = 'dashdot')

ax1 = fig.add_subplot(222)
ax1.set_title('y vs. timestep')
ax1.set_xlabel('time')
ax1.set_ylabel('y')
ax1.plot(np.array(list(range(len(x[2])))), x[1])
ax1.plot(np.array(list(range(len(xfinal[2]))))+time_step, xfinal[1])
ax1.axhline(y = 8.485, color = 'r', linestyle = 'dashdot')

ax2 = fig.add_subplot(223)
ax2.set_title('z vs. timestep')
ax2.set_xlabel('time')
ax2.set_ylabel('z')
ax2.plot(np.array(list(range(len(x[2])))), x[2])
ax2.plot(np.array(list(range(len(xfinal[2]))))+time_step, xfinal[2])
ax2.axhline(y = 27, color = 'r', linestyle = 'dashdot')

fig.suptitle('Stablization at (8.485,8.485,27)', fontsize=18)
fig.tight_layout()
plt.savefig('coordinatesvstime8.png')

# env.reset()
# print(env.step(env.action_space.sample()))