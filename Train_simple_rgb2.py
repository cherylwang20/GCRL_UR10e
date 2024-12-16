import gym
import os
from gym import spaces
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch 
import random
import kornia.augmentation as KAug
import kornia.enhance as KEnhance
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
#import mujoco_py
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from datetime import datetime
import time
from wandb.integration.sb3 import WandbCallback

class CustomDictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Adjust features_dim if needed
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=8, stride=4, padding=2),  # Adjust padding to fit your needs
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for feature concatenation
        )

        # Vector processing network

        self.mlp = nn.Linear(observation_space.spaces['vector'].shape[0], 14)
        
        # Calculate the total concatenated feature dimension
        self._features_dim = 24974 # Adjust based on actual output dimensions of cnn and mlp

    def forward(self, observations):
        image = observations['image'].permute(0, 3, 1, 2)
        image_features = self.cnn(image)
        vector_features = self.mlp(observations['vector'])
        concatenated_features = torch.cat([image_features, vector_features], dim=1)
        return concatenated_features

class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMultiInputPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=CustomDictFeaturesExtractor,
                                                     features_extractor_kwargs={},
                                                     net_arch=[{'vf': [512, 512], 'pi': [512, 512]}])  # Adjust architecture if needed

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

def make_env(env_name, idx, seed=0, eval_mode=False):
    def _init():
        env = gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=eval_mode)
        env.seed(seed + idx)
        return env
    return _init

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--group", type=str, default='testing', help="environment name")

args = parser.parse_args()

time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

IS_WnB_enabled = True
env_name = 'UR10eReachFixed-v13'

training_steps = 5000000
ENTROPY = 0.01
LR = linear_schedule(0.0002)
CR = linear_schedule(0.1)

loaded_model = 'N/A'
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    config = {
        "policy_type": 'PPO',
        'name': time_now,
        "total_timesteps": training_steps,
        "env_name": env_name,
        "dense_units": 512,
        "activation": "relu",
        "max_episode_steps": 250,
        "seed": args.seed,
        "entropy": ENTROPY,
        "lr": 0.0003,
        "CR": 0.1,
        "num_envs": 8,
        "loaded_model": loaded_model,
    }
    #config = {**config, **envs.rwd_keys_wt}
    run = wandb.init(project="RL-Chemist_Reach",
                    group=args.group,
                    settings=wandb.Settings(start_method="fork"),
                    config=config,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    monitor_gym=True,  # auto-upload the videos of agents playing the game
                    save_code=True,  # optional
                    )
except ImportError as e:
    pass 


env = DummyVecEnv([make_env(env_name, i, seed=1) for i in range(8)])
env.render_mode = 'rgb_array'
envs = VecVideoRecorder(env, "videos/" + env_name + '/training_log' ,
    record_video_trigger=lambda x: x % 30000 == 0, video_length=250)
envs = VecMonitor(env)
envs = VecFrameStack(envs, n_stack = 3)

log_path = './Reach_Target_vel/policy_best_model/' + env_name + '/' + time_now + '/'
eval_callback = EvalCallback(envs, best_model_save_path=log_path, log_path=log_path, eval_freq=2000, n_eval_episodes=20, deterministic=True, render=False)

print('Begin training')
print(time_now)

#loaded_model = '2024_11_28_13_54_29'
# Create a model using the vectorized environment
#model = SAC("MultiInputPolicy", envs, buffer_size=1000, verbose=0)
model = PPO(CustomMultiInputPolicy, envs, ent_coef=ENTROPY, learning_rate=LR, clip_range=CR, n_steps = 2048, batch_size = 64, verbose=0, tensorboard_log=f"runs/{time_now}")
#model = PPO.load(r"./Reach_Target_vel/policy_best_model/" + 'UR10eReachFixed-v0' + '/' + loaded_model + '/best_model', envs, verbose=0, tensorboard_log=f"runs/{time_now}")

callback = CallbackList([eval_callback])

model.learn(total_timesteps=training_steps, callback=callback)# , tb_log_name=env_name + "_" + time_now)