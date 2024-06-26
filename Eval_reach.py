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
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

model_num = '2024_06_26_00_35_08' #'2024_06_22_19_48_33'
env_name = "UR10eReachFixed-v2"
movie = True
frame_width = 200
frame_height = 200
cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_CNN/policy_best_model/' + env_name +'/' + model_num + r'/best_model')
env = gym.make(f'mj_envs.robohive.envs:{env_name}')

detect_color = 'red'

env.reset()
env.set_color(detect_color)

frames = []
frames_mask = []
view = 'front'
all_rewards = []
for _ in tqdm(range(2)):
    ep_rewards = 0
    solved = False
    obs = env.reset()
    step = 0
    ret, frame = cap.read()
    while not solved and step < 150:
          #obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()        
          action, _ = model.predict(obs, deterministic=True)
          #env.sim.data.ctrl[:] = action
          obs, reward, done, info = env.step(action)
          solved = info['solved']
          if movie:
              frame = env.sim.renderer.render_offscreen(width=200, height=200, camera_id=f'end_effector_cam')
              rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
              blurred = cv.GaussianBlur(rgb, (11, 11), 0)
              hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
              # construct a mask for the color "green", then perform
              # a series of dilations and erosions to remove any small
              # blobs left in the mask
              redLower = np.array([0, 70, 50], dtype=np.uint8)
              redUpper = np.array([7, 233, 255], dtype=np.uint8)
              mask = cv.inRange(hsv, redLower, redUpper)
              mask = cv.erode(mask, None, iterations=2)
              mask = cv.dilate(mask, None, iterations=2)
              
              #define the grasping rectangle
              x1, y1 = int(53/200 * frame_width), 0
              x2, y2 = int(156/200 * frame_width), int(68/200 * frame_height)

              cv.rectangle(frame, (x1, 0), (x2, y2), (0, 0, 255), thickness=2)
              cv.rectangle(mask, (x1, 0), (x2, y2), 255, thickness=1)
              cv.imshow("rbg", rgb)
              cv.waitKey(1)
              frame = np.rot90(np.rot90(frame))
              frames.append(frame[::-1, :, :])
              frames_mask.append(mask)
          step += 1
          ep_rewards += reward
    all_rewards.append(ep_rewards)

print(f"Average reward: {np.mean(all_rewards)}")
env.close()


if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
