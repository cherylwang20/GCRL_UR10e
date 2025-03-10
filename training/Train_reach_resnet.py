import gym
import os
import sys
sys.path.append(r'C:\Users\chery\Documents\RL-Chemist\mj_envs')
sys.path.append(r'C:\Users\chery\Documents\RL-Chemist')
sys.path.append(r'C:\Users\chery\Documents\RL-Chemist\utils')

from gym import spaces
from PIL import Image
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torch 
from utils.sac import MultiInputPolicySAC
import random
import kornia.augmentation as KAug
import kornia.enhance as KEnhance
import torch.nn as nn
import numpy as np
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


from torchvision.models.resnet import ResNet18_Weights
#model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')

import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_name", type=str, default='UR10eReach1C-v1', help="environment name")
parser.add_argument("--group", type=str, default='testing', help="environment name")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")
parser.add_argument("--algo", type=str, default='PPO', help="Algorithm to train")

parser.add_argument("--channel_num", type=int, default=4, help="channel num")
parser.add_argument("--merge", type= bool, default= False, help="merge with real world image")

args = parser.parse_args()


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
    def __init__(self, observation_space, features_dim=1024):
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)

        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.repr_dim = 1024
        self.image_channel = 3
        x = torch.randn([1] + [9, 120, 212])

        with torch.no_grad():
            out_shape = self.forward_conv(x)
        self.out_dim = out_shape.shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        self.rgb_feature = self.ln(self.fc(out_shape)).shape[1]
        
        self.binary_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # Output is 1x1x64
        )

        self.flatten = nn.Flatten()

        # Vector processing network
        self.mlp = nn.Linear(observation_space.spaces['vector'].shape[0], 512)

        # Calculate the feature dimensions separately for each network and vector features
        #self.rgb_feature_dim = self.calculate_feature_dim(self.rgb_cnn, 3, observation_space.spaces['image'].shape[0], observation_space.spaces['image'].shape[1])
        self.binary_feature_dim = self.calculate_feature_dim(self.binary_cnn, 3, observation_space.spaces['image'].shape[0], observation_space.spaces['image'].shape[1])
        # The total feature dimension is three times the sum of RGB and binary features for each frame, plus vector features
        self._features_dim = self.rgb_feature + self.binary_feature_dim + 512

    @torch.no_grad()
    def forward_conv(self, obs, flatten = True):
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.reshape(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == 'layer2':
                break
        
        conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        conv_current = conv[:, 1:, :, :, :]
        conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        if flatten:
            conv = conv.view(conv.size(0), -1)

        return conv


    def calculate_feature_dim(self, cnn, channels, height, width):
        with torch.no_grad():
            sample_input = torch.randn(1, channels, height, width)
            output = self.flatten(cnn(sample_input))
            return output.shape[1]

    def forward(self, observations):
        images = observations['image']
        rgb1 = images[:, :, :, 0:3]   # Channels 1-3
        rgb2 = images[:, :, :, 4:7]   # Channels 5-7 (0-indexed, so 4:7 slices channels 5, 6, and 7)
        rgb3 = images[:, :, :, 8:11]  # Channels 9-11

        # Concatenate the RGB frames into one tensor along a new dimension
        combined_rgb = torch.cat([rgb1, rgb2, rgb3], dim=3).permute(0, 3, 1, 2).float()
        rgb_conv = self.forward_conv(combined_rgb)
        rgb_features = self.fc(rgb_conv)
        rgb_features = self.ln(rgb_features)

        binary_channels = images[:, :, :, [3, 7, 11]].permute(0, 3, 1, 2).float()
        binary_features = self.flatten(self.binary_cnn(binary_channels))

        stacked_features_tensor = torch.cat([rgb_features, binary_features], dim=1)

        # Prepare MLP output for concatenation
        mlp_output = self.mlp(observations['vector'])
        if mlp_output.dim() == 1:
            mlp_output = mlp_output.unsqueeze(0)  # Unsqueezing if it's a flat vector without batch dimension

        # Concatenate stacked frame features with MLP output
        concatenated_features = torch.cat((stacked_features_tensor, mlp_output), dim=1)

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

def make_env(env_name, idx,  channel, MERGE, seed=0,eval_mode=False):
    def _init():
        env = gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=eval_mode, channel = channel, MERGE = MERGE)
        env.seed(seed + idx)
        return env
    return _init

def main():
    print('merge',args.merge)

    training_steps = 3500000
    env_name = args.env_name
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ENTROPY = 0.01
    LR = linear_schedule(args.learning_rate)
    CR = linear_schedule(args.clip_range)

    time_now = time_now + str(args.seed) + args.algo

    IS_WnB_enabled = True

    loaded_model = 'N/A' #'2024_09_25_13_42_113'
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
            "lr": args.learning_rate,
            "CR": args.clip_range,
            "num_envs": args.num_envs,
            "loaded_model": loaded_model,
        }
        #config = {**config, **envs.rwd_keys_wt}
        run = wandb.init(project="RL-Chemist_Reach",
                        group=args.group,
                        settings=wandb.Settings(start_method="thread"),
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        monitor_gym=True,  # auto-upload the videos of agents playing the game
                        save_code=True,  # optional
                        )
    except ImportError as e:
        pass 

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    num_cpu = args.num_envs

    env = DummyVecEnv([make_env(env_name, i, seed=args.seed, channel = args.channel_num, MERGE = args.merge ) for i in range(num_cpu)])
    env.render_mode = 'rgb_array'
    envs = VecVideoRecorder(env, "videos/" + env_name + '/training_log' ,
        record_video_trigger=lambda x: x % 30000 == 0, video_length=250)
    envs = VecMonitor(env)
    envs = VecFrameStack(envs, n_stack = 3)

    ## EVAL
    eval_env = DummyVecEnv([make_env(env_name, i, seed=args.seed,channel = args.channel_num, eval_mode=True, MERGE = args.merge) for i in range(1)])
    eval_env.render_mode = 'rgb_array'
    eval_env = VecVideoRecorder(eval_env, "videos/" + env_name + '/training_log' ,
        record_video_trigger=lambda x: x % 30000 == 0, video_length=250)
    eval_envs = VecFrameStack(eval_env, n_stack = 3)
    
    log_path = './Reach_Target_vel/policy_best_model/' + env_name + '/' + time_now + '/'
    eval_callback = EvalCallback(eval_envs, best_model_save_path=log_path, log_path=log_path, eval_freq=2000, n_eval_episodes=20, deterministic=True, render=False)
    
    print('Begin training')
    print(time_now)


    # Create a model using the vectorized environment
    #model = SAC("MultiInputPolicy", envs, buffer_size=1000, verbose=0)
    if args.algo == 'PPO':
        model = PPO(CustomMultiInputPolicy, envs, ent_coef=ENTROPY, learning_rate=LR, clip_range=CR, n_steps = 2048, batch_size = 64, verbose=0, tensorboard_log=f"runs/{time_now}")
    elif args.algo == 'SAC':
        model = SAC(MultiInputPolicySAC, envs, batch_size=32, buffer_size = 300, learning_rate=LR, verbose=0)
    #model = PPO.load(r"./Reach_Target_vel/policy_best_model/" + env_name + '/' + loaded_model + '/best_model', envs, verbose=1, tensorboard_log=f"runs/{time_now}")

    #callback = CallbackList([augment_callback, eval_callback, WandbCallback(gradient_save_freq=100)])
    callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=1000)])
    

    model.learn(total_timesteps=training_steps, callback=callback)# , tb_log_name=env_name + "_" + time_now)

    if IS_WnB_enabled:
        run.finish()

if __name__ == "__main__":
    # TRAIN
    main()