import gym
from gym import spaces
from stable_baselines3 import PPO
import skvideo
import skvideo.io
import numpy as np
import os
import random
from tqdm.auto import tqdm

model_num = '2024_04_30_14_24_34'
env_name = "UR10eReachFixed-v0"
movie = True

model = PPO.load('./Reach_Target/policy_best_model/' + env_name +'/' + model_num + r'/best_model')
env = gym.make(f'mj_envs.robohive.envs:{env_name}')

env.reset()

frames = []
view = 'front'
for _ in tqdm(range(2)):
    ep_rewards = []
    solved = False
    obs = env.reset()
    step = 0
    while not solved and step < 300:
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()        
          action, _ = model.predict(obs, deterministic=True)
          #env.sim.data.ctrl[:] = action
          obs, reward, done, info = env.step(action)
          solved = info['solved']
          if movie:
                  #geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  #env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.sim.renderer.render_offscreen(width=640, height=480,camera_id=f'right_cam')
                  frame = np.rot90(np.rot90(frame))
            # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                  frames.append(frame[::-1,:,:])
                  #env.sim.mj_render(mode='window') # GUI
          step += 1

env.close()

if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	
