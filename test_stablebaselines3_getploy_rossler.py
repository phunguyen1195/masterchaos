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
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes 
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset

dyn_lorenz = {"sigma":16.0, "R":45.92, "b": 4.0}
x0 = np.array([8, 3, 5])
dyn_rossler = {"a": 0.2, "b": 0.2, "c": 5.7 }
x0_rossler= np.array([10.0, 1.0, 0.0])
time_step = 5000


def Rossler (x0, dyn):
    a = dyn['a']
    b = dyn['b']
    c = dyn['c']
    x = x0[0]
    y = x0[1]
    z = x0[2]
    return np.array([-(y+z), x+ a*y, b + z*(x-c)])

def RungeKutta (dyn, f, dt, x0):
    k1 = f(x0, dyn)*dt #[x,y,z]*0.1 example
    k2 = f(x0+0.5*k1,dyn)*dt
    k3 = f(x0 + 0.5*k2, dyn)*dt
    k4 = f(x0 + k3, dyn)*dt
    
    x = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    return x

def f_t (dyn, f, dt, x0, T):
    x = np.empty(shape=(len(x0),T))
    x[:, 0] = x0    
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1]) 
    return x

x = f_t(dyn_rossler, Rossler, 0.01, x0_rossler, time_step)

# env = gym.make("lorenz_u-v0")
env = gym.make("rossler-v0", infinite=True)
# env = gym.make('Pendulum-v1')

new_x = []
#del model # remove to demonstrate saving and loading

model = SAC.load("sac_rossler_5_1rad_100alpha_rreward_fixed")
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

fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection="3d")
ax.scatter3D(x0_rossler[0], x0_rossler[1], x0_rossler[2], s=50,alpha=1)
ax.plot3D(x[0], x[1], x[2], alpha=0.4)

# u = np.sin(np.pi * xfinal[0]) * np.cos(np.pi * xfinal[1]) * np.cos(np.pi * xfinal[2])
# v = -np.cos(np.pi * xfinal[0]) * np.sin(np.pi * xfinal[1]) * np.cos(np.pi * xfinal[2])
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * xfinal[0]) * np.cos(np.pi * xfinal[1]) *
#      np.sin(np.pi * xfinal[2]))

# ax.quiver3D(xfinal[0], xfinal[1], xfinal[2], xfinal[0], xfinal[1], xfinal[2],length=.1, colors='green')
ax.scatter3D(xfinal[0][0], xfinal[1][0], xfinal[2][0], s=50, c='green',alpha=1)
ax.plot3D(xfinal[0], xfinal[1], xfinal[2], c='green',alpha=0.5)
ax.scatter3D(  5.6929736, -28.464869,   28.464869 , s=100, c='red', marker='o')
ax.scatter3D(  0.0070262,  -0.03513102,  0.03513102 , s=100, c='red', marker='o')
# ax.scatter(  0,  0,  0 , s=100, c='red', marker='o')
# [ 0.0070262  -0.03513102  0.03513102]
plt.title('Russle attractor')
plt.draw()
plt.savefig('Russle_attractor.png')


# xfinal_test = xfinal[0][1000:1020]
# print (xfinal_test)
# fig = plt.figure(figsize=(15,15))
# ax4 = fig.add_subplot(111)
# ax4.set_title('x vs. timestep')
# ax4.set_xlabel('time')
# ax4.set_ylabel('x')
# # start, end = ax.get_xlim()
# # ax.xaxis.set_ticks(np.arange(start, end, 100))
# ax4.plot(np.array(list(range(len(xfinal_test)))), xfinal_test, c = 'green')
# #ax4.axhline(y = 5.6929736, color = 'r', linestyle = 'dashdot')
# plt.savefig('Rossler_attractor_test.png')

a = np.zeros([3,len(ac)])
for i in range(len(ac)):
    a[:,i] = ac[i]
a_test = a[0][1000:1020]
fig = plt.figure(figsize=(15,15))
ax5 = fig.add_subplot(111)
ax5.set_title('action vs. timestep')
ax5.set_xlabel('time')
ax5.set_ylabel('action')
# start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 100))
ax5.plot(np.array(list(range(len(a_test)))), a_test, c = 'black')
#ax4.axhline(y = 5.6929736, color = 'r', linestyle = 'dashdot')
plt.savefig('Rossler_attractor_action.png')


# fig = plt.figure(figsize=(25,25))
# ax = fig.add_subplot(111)
# ax.set_title('x vs. timestep')
# ax.set_xlabel('time')
# ax.set_ylabel('x')
# # start, end = ax.get_xlim()
# # ax.xaxis.set_ticks(np.arange(start, end, 100))
# ax.plot(np.array(list(range(len(x[2])))), x[0])
# ax.plot(np.array(list(range(len(xfinal[2]))))+time_step, xfinal[0], c = 'green')
# ax.axhline(y = 5.6929736, color = 'r', linestyle = 'dashdot')
# x1 = time_step + 1000
# x2 = time_step + 1050

# # select y-range for zoomed region
# y1 = xfinal[0][1000:1050].min()
# y2 = xfinal[0][1000:1050].max()
# # print (xfinal[0][1000:1100].mean())
# # Make the zoom-in plot:
# # axins = inset_axes(ax, zoom=10, loc=1) # zoom = 2
# axins = ax.inset_axes([0.7, 0.8, 0.3, 0.2])
# axins.plot(np.array(list(range(len(xfinal[0][1000:1050])))), xfinal[0][1000:1050], c = 'green')
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# ax.indicate_inset_zoom(axins)
# plt.savefig('Rossler_attractor_idk.png')


fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(221)
ax.set_title('x vs. timestep')
ax.set_xlabel('time')
ax.set_ylabel('x')
# start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 100))
ax.plot(np.array(list(range(len(x[0])))), x[0])
ax.plot(np.array(list(range(len(xfinal[2]))))+time_step, xfinal[0], c = 'green')
ax.axhline(y = 5.6929736, color = 'r', linestyle = 'dashdot')
x1 = time_step + 1000
x2 = time_step + 1050

# select y-range for zoomed region
y1 = xfinal[0][1000:1050].min()
y2 = xfinal[0][1000:1050].max()
# print (xfinal[0][1000:1100].mean())
# Make the zoom-in plot:
# axins = inset_axes(ax, zoom=10, loc=1) # zoom = 2
axins = ax.inset_axes([0.7, 0.8, 0.3, 0.2])
axins.plot(np.array(list(range(len(xfinal[0])))), xfinal[0], c = 'green')
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.indicate_inset_zoom(axins)


ax1 = fig.add_subplot(222)
ax1.set_title('y vs. timestep')
ax1.set_xlabel('time')
ax1.set_ylabel('y')
ax1.plot(np.array(list(range(len(x[1])))), x[1])
ax1.plot(np.array(list(range(len(xfinal[1]))))+time_step, xfinal[1], c = 'green')
ax1.axhline(y = -28.464869, color = 'r', linestyle = 'dashdot')
x1 = time_step + 1000
x2 = time_step + 1050

# select y-range for zoomed region
y1 = xfinal[1][1000:1050].min()
y2 = xfinal[1][1000:1050].max()
# print (xfinal[0][1000:1100].mean())
# Make the zoom-in plot:
# axins = inset_axes(ax, zoom=10, loc=1) # zoom = 2
axins = ax1.inset_axes([0.7, 0.8, 0.3, 0.2])
axins.plot(np.array(list(range(len(xfinal[1])))), xfinal[1], c = 'green')
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax1.indicate_inset_zoom(axins)

ax2 = fig.add_subplot(223)
ax2.set_title('z vs. timestep')
ax2.set_xlabel('time')
ax2.set_ylabel('z')
ax2.plot(np.array(list(range(len(x[2])))), x[2])
ax2.plot(np.array(list(range(len(xfinal[2]))))+time_step, xfinal[2], c = 'green')
ax2.axhline(y = 28.464869, color = 'r', linestyle = 'dashdot')
x1 = time_step + 1000
x2 = time_step + 1050

# select y-range for zoomed region
y1 = xfinal[2][1000:1050].min()
y2 = xfinal[2][1000:1050].max()
# print (xfinal[0][1000:1100].mean())
# Make the zoom-in plot:
# axins = inset_axes(ax, zoom=10, loc=1) # zoom = 2
axins = ax2.inset_axes([0.7, 0.8, 0.3, 0.2])
axins.plot(np.array(list(range(len(xfinal[2])))), xfinal[2], c = 'green')
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax2.indicate_inset_zoom(axins)

fig.suptitle('Stablization at (5.6929736,-28.464869,28.464869)', fontsize=18)
fig.tight_layout()
plt.savefig('coordinatesvstime5_rossler.png')

# env.reset()
# print(env.step(env.action_space.sample()))