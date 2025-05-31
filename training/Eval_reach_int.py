import gymnasium as gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
import skvideo
import skvideo.io
import numpy as np
import os
import cv2 as cv
import random
from tqdm.auto import tqdm
import torch
import warnings
from collections import deque
from PIL import Image
import io
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/cherylwang/Documents/GitHub/GCRL_UR10e/mj_envs')
sys.path.append('/Users/cherylwang/Documents/GitHub/GCRL_UR10e')

class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, n_stack=3):
        super().__init__(env)
        self.env = env
        self.n_stack = n_stack
        self.image_frames = deque([], maxlen=n_stack)
        self.vector_frames = deque([], maxlen=n_stack)

        # Assume observation space contains 'image' and 'vector'
        image_space = env.observation_space.spaces['image']
        vector_space = env.observation_space.spaces['vector']

        # Update the observation space for stacked images and vectors
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=np.tile(image_space.low, n_stack),
                                    high=np.tile(image_space.high, n_stack),
                                    dtype=image_space.dtype),
            'vector': gym.spaces.Box(low=np.tile(vector_space.low, n_stack),
                                     high=np.tile(vector_space.high, n_stack),
                                     dtype=vector_space.dtype)
        })

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.image_frames.append(obs['image'])
            self.vector_frames.append(obs['vector'])
        stacked_images = np.concatenate(self.image_frames, axis=-1)
        stacked_vectors = np.concatenate(self.vector_frames, axis=-1)
        obs['image'] = stacked_images
        obs['vector'] = stacked_vectors
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.image_frames.append(obs['image'])
        self.vector_frames.append(obs['vector'])
        stacked_images = np.concatenate(self.image_frames, axis=-1)
        stacked_vectors = np.concatenate(self.vector_frames, axis=-1)
        obs['image'] = stacked_images
        obs['vector'] = stacked_vectors
        return obs, reward, done, info

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--env_name", type=str, default='NA', help="environment name")
parser.add_argument("--model_num", type=str, default='testing', help="environment name")
parser.add_argument("--movie", type=str, default='False', help="environment name")
parser.add_argument("--channel_num", type=int, default=4, help="channel num")

parser.add_argument("--merge", type= bool, default= False, help="merge with real world image")
parser.add_argument("--fs", type=int, default= 20, help="frameskip")


args = parser.parse_args()

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")

model_num = args.model_num  
env_name = args.env_name 
print(env_name)
env = gym.make(f'mj_envs.robohive.envs:{env_name}', channel = args.channel_num, MERGE = args.merge, fs = args.fs)
env = CustomFrameStack(env, n_stack=3)

seed_value = 47004  # Seed value for reproducibility
#env.seed(seed_value)

movie = True
frame_width = 212
frame_height = 120

model = PPO.load(os.getcwd() + '/policy/' + '/' + model_num)
policy = model.policy

print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

env.reset(seed = seed_value)

trial = 2
success = 0

frames_rgb = []
frames_mask = []
view = 'front'
all_rewards = []
saliency_map = []
for i in tqdm(range(trial)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    step = 0
    while not done and step < 200:   
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, done, info  = env.step(action)
          solved = info['solved']
          if i < trial:
              frame_n = env.unwrapped.rgb_out
              mask = env.unwrapped.mask_out
              frames_rgb.append(frame_n)
              frames_mask.append(mask)
          step += 1
          ep_rewards += reward
    print(step)
    if solved:
        success += 1
    all_rewards.append(ep_rewards)

print(f"Average reward: {np.mean(all_rewards)}")
env.close()
print(f"Success rate: {success/trial}")


if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames_rgb), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
