import gym
from gym import spaces
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
#import mujoco_py
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from datetime import datetime
import time
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        obs_vecs = self.training_env.get_attr('get_obs_vec')  # Returns a list of observations from all environments
        average_obs = np.mean([np.mean(obs) for obs in obs_vecs], axis=0)  # Compute the average observation
        self.logger.record("average_obs", average_obs)
        return True

def make_env(env_name, idx, seed=0):
    def _init():
        env = gym.make(f'mj_envs.robohive.envs:{env_name}')
        env.seed(seed + idx)
        return env
    return _init

if __name__ == '__main__':
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    num_cpu = 2
    env_name = "UR10eReachFixed-v3"
    envs = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])

    detect_color = 'red'
    #envs.set_attr('set_color', detect_color)
    envs.color = detect_color

    log_path = './Reach_Target_vel/policy_best_model/' + env_name + '/' + time_now + '/'
    eval_callback = EvalCallback(envs, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)

    print('Begin training')
    # Adjust here to print keys from one of the environments
    print(envs.get_attr('obs_dict')[0].keys())

    # Create a model using the vectorized environment
    model = PPO("MultiInputPolicy", envs, ent_coef=0.01, verbose=0)
    #model_num = "2024_07_02_21_36_07"
    #model = PPO.load(r"C:/Users/chery/Documents/RL-Chemist/Reach_Target_vel/policy_best_model/" + env_name + '/' + model_num + '/best_model', envs, verbose=0)


    #obs_callback = TensorboardCallback()
    callback = CallbackList([eval_callback])#, obs_callback])

    model.learn(total_timesteps=5000000, tb_log_name=env_name + "_" + time_now, callback=callback)
