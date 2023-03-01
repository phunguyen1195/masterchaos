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
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization, SubprocVecEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
from fractions import Fraction
directory = 'lorenz/'


dyn_lorenz = {"sigma":10.0, "R":28.0, "b": 8/3}
x0 = np.array([10.0, 1.0, 0.0])
# x0 = np.array([-8.485, -8.485, 27])

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

x = f_t(dyn_lorenz, lorenz, 0.01, x0, 500, 0)



# env = gym.make("lorenz_le-v0")
# env = gym.make("rossler-v0")
# env = gym.make('Pendulum-v1')

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# class TensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """

#     def __init__(self, verbose=0):
#         super(TensorboardCallback, self).__init__(verbose)

#     def _on_step(self) -> bool:
#         # Log scalar value (here a random variable)
#         reward_lorenz = self.cost
#         self.logger.record('reward', reward_lorenz)
#         return True

# 

class SummaryWriterCallback(EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self
        self.counter = 0
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.acc_reward = 0.0

        # self.fig = plt.figure(figsize=(10,10))
        # self.ax = self.fig.gca(projection="3d")

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self.v = []
        self.a = []
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        continue_training = True
        # print (self.counter)
        #input()
        # self.ax.scatter3D(self.locals['new_obs'][0][0], self.locals['new_obs'][0][1], self.locals['new_obs'][0][2],s=60,c=np.array([[ 1-self.counter/self.eval_freq, self.counter/self.eval_freq, 1]]),marker='.', alpha=0.7)
        # self.counter = self.counter + 1
        # if self.counter == self.eval_freq:
        #     self.counter = 0
        # if self.locals['dones'][0] == True:
        #     self.ax.scatter(8.485, 8.485, 27, s=100, c='red')
        #     self.logger.record("trajectory/figure", Figure(self.fig, close=True), exclude=("stdout", "log", "json", "csv"))
        self.v.append(self.locals['infos'][0]['velocity'])
        self.a.append(self.locals['infos'][0]['action'][0])
        #print (self.n_calls)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # print (self.locals)
            
            #print (self.locals['infos'][0])
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            
            

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            # self.logger.record("eval/accumulated rewards", np.sum(episode_rewards))
            coordinate_list = np.array(self.v)
            self.logger.record("eval/mean velocity x", np.mean(coordinate_list[:,0]))
            self.logger.record("eval/mean velocity y", np.mean(coordinate_list[:,1]))
            self.logger.record("eval/mean velocity z", np.mean(coordinate_list[:,2]))
            self.logger.record("eval/mean action", np.mean(self.a))
            self.v = []
            self.a = []
            


            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)
            
            # self.ax.cla()


            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            #self.logger.record("eval/discount factor", self.locals["self"].gamma)
            #self.locals["self"].gamma = self.locals["self"].gamma + 0.001
            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.
        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        

      #print (self.locals)

    # def plot_action(self):
    #   #x = self.action_u[0]
    #   x = self.past_obs [0]
    #   self.ax.scatter3D(x[0], x[1], x[2], s=10, c='blue', alpha=0.5)
    #   plt.title('Lorenz attractor action')
    #   plt.draw()
    #   #plt.show(block=False)
    #   #self.collected_states = list()
    #   plt.savefig('Lorenz_ppo_action.png')

def main():
    # vec_env_cls=SubprocVecEnv
    env = make_vec_env("lorenz_le-v0", n_envs=2, seed=0, vec_env_cls=SubprocVecEnv)
    eval_callback = SummaryWriterCallback(env, best_model_save_path='./logs/',
                                log_path='./logs/', eval_freq=500,
                                deterministic=True, render=False)
    # model = PPO('MlpPolicy', env , learning_rate=0.05, verbose=1, tensorboard_log="./Lorenz_tensorboard/")
    # for (0,0,0) gamma = 0.985 lrd = linear_schedule(1e-3)
    model = PPO('MlpPolicy', env ,gamma=0.98, 
        # use_sde=True,
        # sde_sample_freq=4,
        # learning_rate=linear_schedule(1e-3),
        # learning_rate=linear_schedule(4e-3),
        learning_rate=1e-3, 
        tensorboard_log="./Lorenz_tensorboard/")
    model.learn(total_timesteps=80000, callback=eval_callback)

    model.save("ppo_lorenz_le_poseqi_vecenv")


    del model # remove to demonstrate saving and loading
    new_x = []
    model = PPO.load("ppo_lorenz_le_poseqi_vecenv")
    env.reset()
    obs = x[:,-1]
    new_x.append(obs)
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        new_x.append(obs)
        # env.render()
        if done:
            obs = env.reset()

    xfinal = np.zeros([3,1001])
    for i in range(len(new_x)):
        xfinal[:,i] = new_x[i]


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title('x vs. timestep')
    ax.set_xlabel('time')
    ax.set_ylabel('x')
    ax.plot(np.array(list(range(len(x[0])))), x[0])
    ax.plot(np.array(list(range(len(xfinal[0]))))+500, xfinal[0])
    ax.axhline(y = 0, color = 'r', linestyle = 'dashdot')
    plt.savefig('xvstime0.png')
    # env.reset()
    # print(env.step(env.action_space.sample()))

if __name__ == '__main__':
    main()