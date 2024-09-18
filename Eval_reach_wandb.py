import gym
from gym import spaces
from stable_baselines3 import PPO
import skvideo
import skvideo.io
import numpy as np
import os
import cv2 as cv
import random
from tqdm.auto import tqdm
import torch
import mujoco

#obj2mjcf --obj-dir . --obj-filter beaker --save-mjcf --compile-model --decompose --overwrite --coacd-args.max-convex-hull 15

model_num = '2024_09_13_12_56_365' #'2024_08_10_19_05_524' #'2024_06_22_19_48_33'
env_name = "UR10eReach4C-v1"
env = gym.make(f'mj_envs.robohive.envs:{env_name}')

movie = True
frame_width = 224
frame_height = 224
#cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_vel/policy_best_model/' + "UR10eReach4C-v1" +'/' + model_num + r'/best_model')
#model = PPO.load('./models/'+ model_num + r'/model')


print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

detect_color = 'green'

env.reset()

trial = 3
success = 0

frames = []
frames_mask = []
view = 'front'
all_rewards = []
for i in tqdm(range(trial)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    step = 0
    #ret, frame = cap.read()
    while not done and step < 200:
          #obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()        
          action, _ = model.predict(obs, deterministic=False)
          #print(action)
          obs, reward, done, info = env.step(action)
          solved = info['solved']
          if i < 3:
              frame_n = env.rgb_out
              mask = env.mask_out
              frame_n = np.rot90(np.rot90(frame_n))
              frames.append(frame_n[::-1, :, :])
              frames_mask.append(mask)
          step += 1
          ep_rewards += reward
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
