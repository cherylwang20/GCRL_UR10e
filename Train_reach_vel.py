import gym
from gym import spaces
import torch 
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


class CustomDictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Adjust features_dim if needed
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),  # Adjust padding to fit your needs
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for feature concatenation
        )

        # Vector processing network
        self.mlp = nn.Linear(observation_space.spaces['vector'].shape[0], 16)
        

        # Calculate the total concatenated feature dimension
        self._features_dim = observation_space.spaces['image'].shape[0]**2 + 16  # Adjust based on actual output dimensions of cnn and mlp

    def forward(self, observations):
        images = observations['image'].permute(0, 3, 1, 2)
        image_features = self.cnn(images)
        vector_features = self.mlp(observations['vector'])
        return torch.cat([image_features, vector_features], dim=1)

class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMultiInputPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=CustomDictFeaturesExtractor,
                                                     features_extractor_kwargs={},
                                                     net_arch=[{'vf': [512, 512], 'pi': [512, 512]}])  # Adjust architecture if needed
def make_env(env_name, idx, seed=0):
    def _init():
        env = gym.make(f'mj_envs.robohive.envs:{env_name}')
        env.seed(seed + idx)
        return env
    return _init

if __name__ == '__main__':
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    num_cpu = 1
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
    #model = SAC("MultiInputPolicy", envs, buffer_size=1000, verbose=0)
    model = PPO(CustomMultiInputPolicy, envs, ent_coef=0.1, verbose=0)
    #model_num = "2024_07_12_13_53_06"
    #model = PPO.load(r"C:/Users/chery/Documents/RL-Chemist/Reach_Target_vel/policy_best_model/" + env_name + '/' + model_num + '/best_model', envs, verbose=0)


    #obs_callback = TensorboardCallback()
    callback = CallbackList([eval_callback])#, obs_callback])

    model.learn(total_timesteps=5000000, tb_log_name=env_name + "_" + time_now, callback=callback)
