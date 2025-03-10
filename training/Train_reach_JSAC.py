import gym
import os
import sys
import multiprocessing as mp
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
from jsac.envs.rl_chemist.env import RLC_Env
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv
from jsac.helpers.eval import start_eval_process
from jsac.helpers.logger import Logger
import jax.numpy as jnp 
from jsac.algo.agent import sample_actions
from jsac.algo.initializers import init_inference_actor, get_init_data
#import mujoco_py
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from datetime import datetime
import time
from wandb.integration.sb3 import WandbCallback


from torchvision.models.resnet import ResNet18_Weights
#model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')

import numpy as np
import argparse

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 5, 2],
        [32, 32, 5, 2],
        [32, 64, 3, 2],
        [64, 64, 3, 2], 
    ],
    
    'latent_dim': 128,

    'mlp': [1024, 1024],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Main script to train an agent")

    parser.add_argument('--task_name', default='gt', type=str)
    parser.add_argument('--goal_type', default='Mask', type=str)
    parser.add_argument('--reward_mode', default="distance", type=str)   # "distance", "mask_size"
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--env_name", type=str, default='UR10eReach1C-v1', help="environment name")
    parser.add_argument("--group", type=str, default='testing', help="environment name")
    parser.add_argument("--algo", type=str, default='JSAC', help="Algorithm to train")

    parser.add_argument("--channel_num", type=int, default=4, help="channel num")
    parser.add_argument("--merge", type= bool, default= False, help="merge with real world image")

    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    parser.add_argument('--image_height', default=120, type=int)          # Mode: img, img_prop
    parser.add_argument('--image_width', default=210, type=int)          # Mode: img, img_prop     
    parser.add_argument('--image_history', default=3, type=int)          # Mode: img, img_prop
    parser.add_argument('--step_time', default=0.0, type=float) 
    parser.add_argument('--episode_steps', default=250, type=int)
    parser.add_argument('--mask_delay_type', default='none', type=str)  # "none", "n_step", "sequential"
    parser.add_argument('--mask_delay_steps', default=3, type=int) 

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=20_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--env_steps', default=300_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--global_norm', default=1.0, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=1e-4, type=float) 
    parser.add_argument('--num_critic_networks', default=5, type=int)
    parser.add_argument('--num_critic_updates', default=1, type=int)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=8, type=int)   # Sync mode: False
    
    # encoder
    parser.add_argument('--spatial_softmax', default=False, action='store_true')    # Mode: img, img_prop

    # sac
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    
    # misc
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--eval_steps', default=10_000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10_000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=500_000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--img_aug_path', default='', type=str)
    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/


    args = parser.parse_args()
    return args


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

def make_env(env_name, idx,  channel, MERGE, seed=0, eval_mode='train'):
    def _init():
        env = RLC_Env(env_name, eval_mode = eval_mode, channel = channel, MERGE = MERGE)
        #gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=eval_mode, channel = channel, MERGE = MERGE)
        env.seed(seed + idx)
        return env
    return _init

def main():
    args = parse_args()
    training_steps = 3500000
    env_name = args.env_name
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    args.sync_mode = 'sync'
    args.name = f'{args.env_name}_{args.task_name}_{args.goal_type}_{args.reward_mode}'
    args.net_params = config

    time_now = time_now + str(args.seed) + args.algo

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")


    print('Begin training')
    print(time_now)
    if args.save_wandb:
        wandb_project_name = f'{args.name}'
        wandb_run_name=f'seed_{args.seed}'
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb, wandb_project_name, 
                   wandb_run_name, args.start_step > 1)
    else:
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)
    
    env = RLC_Env(args.env_name, 
                   args.image_history, 
                   args.image_width, 
                   args.image_height, 
                   eval_mode = False,
                   MERGE = False,
                   channel = 4,
                   goal_type=args.goal_type,
                   reward_mode=args.reward_mode,
                   ofd_index=args.seed)
    
    env = WrappedEnv(env, args.episode_steps)

    set_seed_everywhere(seed=args.seed)

    args.image_shape = env.image_space.shape 
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    
    print(f'Image shape: {args.image_shape}')
    print(f'Proprioception shape: {args.proprioception_shape}')
    print(f'Action shape: {args.action_shape}')

    sync_queue = None
    agent = SACRADAgent(vars(args)) 

    if args.eval_steps > 0:
        eval_args = vars(args)
        eval_args['env_type'] = 'RLC'
        eval_args['ofd_index'] = args.seed
        # eval_args['sync'] = 'true'
        eval_queue_1 = mp.Queue()
        path1 = os.path.join(args.work_dir, 'eval_log')
        make_dir(path1)
        eval_process_1 = start_eval_process(eval_args, 
                                            path1, 
                                            eval_queue_1, 
                                            args.num_eval_episodes,
                                            False)
    
    update_paused = True
    time.sleep(5)
    state = env.reset(create_vid=False)
    
    first_step = True

    while env.total_steps < args.env_steps:
        t1 = time.time()
        if env.total_steps < args.init_steps + 100:
            action = np.random.uniform(-1, 1, args.action_shape[-1])
        else:
            action = agent.sample_actions(state)
        t2 = time.time()
        next_state, reward, done, info = env.step(action) 
        t3 = time.time()

        mask = 1.0 if not done or 'truncated' in info else 0.0
        
        agent.add(state, action, reward, next_state, mask, first_step)
        first_step = False
        state = next_state

        if done or 'truncated' in info: 
            state = env.reset(create_vid=False)
            first_step = True
            info['tag'] = 'train'
            info['elapsed_time'] = time.time() - start_time
            info['dump'] = True
            L.push(info)

        if env.total_steps >= args.init_steps and env.total_steps % args.update_every == 0:
            if not args.sync_mode and update_paused: 
                agent.resume_update()
                update_paused = False
            if sync_queue:
                sync_queue.put(1)
            update_infos = agent.update()
            if update_infos is not None and env.total_steps % args.log_every == 0:
                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False

                    L.push(update_info)

        if env.total_steps % args.xtick == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)
            
        if args.eval_steps > 0 and env.total_steps % args.eval_steps == 0:
            agent.pause_update()
            eval_queue_1.put(agent.get_actor_params())
            eval_queue_1.put(env.total_steps)
            
            if env.total_steps < args.env_steps:
                agent.resume_update()

    if args.save_model:
        agent.checkpoint(env.total_steps)
        
    if args.eval_steps > 0:    
        eval_queue_1.put('close')
        eval_process_1.join()
        
    L.plot()
    L.close()
    env.close()
    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - start_time}s')
    

    return args

if __name__ == "__main__":
    # TRAIN
    main()