import os
import gym
import sys
sys.path.append("src/") 
sys.path.append('../')
sys.path.append('.')
import torch
import skvideo
import skvideo.io
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import skvideo
import skvideo.io
from stable_baselines3 import PPO
from src.reachGrasp_env.GdinoReachGraspEnv import BasicReachGrasp
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
import numpy as np
import cv2 as cv
import warnings
from tqdm.auto import tqdm
from collections import deque

model_num = 'baseline'
model = PPO.load(os.getcwd()+'/policy/' + model_num)
policy = model.policy


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

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stack):
            self.image_frames.append(obs['image'])
            self.vector_frames.append(obs['vector'])
        stacked_images = np.concatenate(self.image_frames, axis=-1)
        stacked_vectors = np.concatenate(self.vector_frames, axis=-1)
        obs['image'] = stacked_images
        obs['vector'] = stacked_vectors
        return obs

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.image_frames.append(obs['image'])
        self.vector_frames.append(obs['vector'])
        stacked_images = np.concatenate(self.image_frames, axis=-1)
        stacked_vectors = np.concatenate(self.vector_frames, axis=-1)
        obs['image'] = stacked_images
        obs['vector'] = stacked_vectors
        return obs, reward, done, _, info

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")


env_name = 'GdinoReachGraspEnv'
env = BasicReachGrasp(render_mode='human', channel= 4)
env = CustomFrameStack(env, n_stack=3)

print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

env.reset()


frames = []
frames_mask = []
movie = True
view = 'end_effector'


trial = 10
success = 0
all_rewards = []
for i in tqdm(range(trial)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    step = 0
    while not done and step < 10:    
          action, _ = model.predict(obs, deterministic=False)
          obs, reward, done, _, info = env.step(action)
          solved = info['solved']
          if i < trial:
              mask = env.mask_out
              frames.append(env.rgb_out)
              frames_mask.append(env.mask_out)
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
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})