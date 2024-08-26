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

model_num = '2024_08_19_18_16_32' #'2024_06_22_19_48_33'
env_name = "UR10eReachFixed-v3"
movie = True
frame_width = 224
frame_height = 224
#cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_vel/policy_best_model/' + env_name +'/' + model_num + r'/best_model')
env = gym.make(f'mj_envs.robohive.envs:{"UR10eReachFixed-v2"}')

print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

detect_color = 'green'

env.reset()

frames = []
frames_mask = []
view = 'front'
all_rewards = []
for _ in tqdm(range(2)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    step = 0
    #ret, frame = cap.read()
    while not done and step < 500:
          #obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()        
          action, _ = model.predict(obs, deterministic=True)
          #print(action)
          obs, reward, done, info = env.step(action)
          #print(obs)
          solved = info['solved']
          if movie:
              frame_n = env.sim.renderer.render_offscreen(width=frame_width, height=frame_height, camera_id=f'ft_cam')
              rgb = cv.cvtColor(frame_n, cv.COLOR_BGR2RGB)
              blurred = cv.GaussianBlur(rgb, (11, 11), 0)
              hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
              # construct a mask for the color "green", then perform
              # a series of dilations and erosions to remove any small
              # blobs left in the mask
              if env.touch_success < 1:
                if env.color == 'red':
                    Lower = (0, 70, 50)
                    Upper = (7, 233, 255)
                elif env.color == 'green':
                    Lower = (29, 86, 56)
                    Upper = (64, 255, 255)
                elif env.color == 'blue':
                    Lower = (80, 50, 20)
                    Upper = (100, 255, 255)
                else:
                    raise Warning('please define a valid color (red, gree, blue)')
              else:        
                  #print(self.touch_success)
                  Lower = (0, 0, 0)
                  Upper = (0, 0, 0)
              mask = cv.inRange(hsv, Lower, Upper)
              mask = cv.erode(mask, None, iterations=2)
              mask = cv.dilate(mask, None, iterations=2)
              
              #define the grasping rectangle
              x1, y1 = int(63/200 * frame_width), frame_height - int(68/200 * frame_height)
              x2, y2 = int(136/200 * frame_width), frame_height 

              cv.rectangle(frame_n, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
              cv.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=1)
              #cv.imshow("rbg", rgb)
              #cv.waitKey(1)
              frame_n = np.rot90(np.rot90(frame_n))
              frames.append(frame_n[::-1, :, :])
              frames_mask.append(mask)
          step += 1
          ep_rewards += reward
    print(step)
    all_rewards.append(ep_rewards)

print(f"Average reward: {np.mean(all_rewards)}")
env.close()


if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
