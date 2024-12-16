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
        obs, reward, done, info = self.env.step(action)
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
parser.add_argument("--policy_env", type=str, default='NA', help="environment name")
parser.add_argument("--model_num", type=str, default='testing', help="environment name")
parser.add_argument("--movie", type=str, default='False', help="environment name")
args = parser.parse_args()

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")

#obj2mjcf --obj-dir . --obj-filter beaker --save-mjcf --compile-model --decompose --overwrite --coacd-args.max-convex-hull 15

model_num = args.model_num  
env_name = args.env_name 
print(env_name)
policy_env = args.policy_env
env = gym.make(f'mj_envs.robohive.envs:{env_name}')
env = CustomFrameStack(env, n_stack=3)

seed_value = 47004  # Seed value for reproducibility
env.seed(seed_value)

movie = True
frame_width = 212
frame_height = 120
#cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_vel/policy_best_model/' + policy_env +'/' + model_num + r'/best_model')
#model = PPO.load('./models/'+ model_num + r'/model')


print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

detect_color = 'green'

env.reset()


trial = 1
success = 0

frames = []
frames_mask = []
view = 'front'
all_rewards = []
joint_angle = []
action_array = []
for i in tqdm(range(trial)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    #obs = np.stack([obs, obs, obs])
    step = 0
    joint_angle.append(env.sim.data.qpos[:7].copy())
    #ret, frame = cap.read()
    while not solved and step < 200:
          #obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = np.stack([obs, obs, obs])
          #obs = env.get_obs_dict()        
          action, _ = model.predict(obs, deterministic=False)
          action_array.append(action)
          obs, reward, done, info = env.step(action)
          joint_angle.append(env.sim.data.qpos[:7].copy())
          solved = info['solved']
          if i < trial:
              frame_n = env.rgb_out #env.sim.renderer.render_offscreen(width = 640, height = 480, camera_id = 'front_cam')
              #print(frame_n)
              frames.append(frame_n)
          step += 1
          ep_rewards += reward
    if solved:
        success += 1
    all_rewards.append(ep_rewards)

print(f"Average reward: {np.mean(all_rewards)}")
env.close()
print(f"Success rate: {success/trial}")

np.savetxt('action_rgb.csv', action_array, fmt='%f', delimiter=',')
np.savetxt('joint_rgb.csv', joint_angle, fmt='%f', delimiter=',')

if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})