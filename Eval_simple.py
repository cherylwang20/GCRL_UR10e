import gym
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
import mujoco
import warnings
from collections import deque

class FrameStackingWrapper(gym.Wrapper):
    def __init__(self, env, n_frames):
        super(FrameStackingWrapper, self).__init__(env)
        self.n_frames = n_frames
        print(env.observation_space.shape)
        self.frames = np.zeros((n_frames,) + env.observation_space.shape)

        # Update observation space to accommodate the stacked frames
        low = np.repeat(env.observation_space.low, n_frames, axis=0)
        high = np.repeat(env.observation_space.high, n_frames, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.frames = np.zeros((self.n_frames,) + (17,))
        obs = self.env.reset()
        self.frames[0] = obs
        return self.frames.flatten()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = obs
        return self.frames.flatten(), reward, done, info

env_name = 'UR10eReachFixed-v12'
policy_env = 'UR10eReachFixed-v12'

model_num = '2024_12_14_16_06_1116'


env = gym.make(f'mj_envs.robohive.envs:{env_name}')
env = FrameStackingWrapper(env, n_frames=3)

seed_value = 47004  # Seed value for reproducibility
env.seed(seed_value)

movie = True
frame_width = 212
frame_height = 120
#cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_vel/policy_best_model/' + policy_env +'/' + model_num + r'/best_model')


print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

env.reset()

trial = 5
success = 0
frames = []
joint_pos = []
action_array = []

obs_array = np.loadtxt('observe.csv', delimiter=',', dtype=np.float32)

view = 'front'
all_rewards = []
for i in tqdm(range(trial)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    step = 0
    while not solved and step < 200:
          action, _ = model.predict(obs, deterministic=True)
          action_array.append(action)
          #print('action', action)
          obs, reward, done, info = env.step(action)
          solved = info['solved']
          if i < trial:
              frame_n = env.sim.renderer.render_offscreen(width = 640, height = 480, camera_id = 'front_cam')
              frames.append(frame_n)
          step += 1
          ep_rewards += reward
    print(step)
    if solved:
        success += 1
    all_rewards.append(ep_rewards)

print(f"Average reward: {np.mean(all_rewards)}")
env.close()
print(f"Success rate: {success/trial}")

np.savetxt('action.csv', action_array, fmt='%f', delimiter=',')

if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video_new.mp4', np.asarray(frames), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
