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
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')

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
parser.add_argument("--channel_num", type=int, default=4, help="channel num")
args = parser.parse_args()

# Ignore specific warning
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")

#obj2mjcf --obj-dir . --obj-filter beaker --save-mjcf --compile-model --decompose --overwrite --coacd-args.max-convex-hull 15

model_num = args.model_num  
env_name = args.env_name 
print(env_name)
policy_env = args.policy_env
env = gym.make(f'mj_envs.robohive.envs:{env_name}', channel = args.channel_num)
env = CustomFrameStack(env, n_stack=3)

seed_value = 47004  # Seed value for reproducibility
env.seed(seed_value)

movie = True
frame_width = 212
frame_height = 120
#cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_vel/policy_best_model/' + policy_env +'/' + model_num + r'/best_model')
#model = PPO.load('./models/'+ model_num + r'/model')
policy = model.policy

print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

detect_color = 'green'

env.reset()

def visualize_saliency(observation):
    device = next(policy.parameters()).device
    
    # Assuming observation['image'] is a tensor with shape [height, width, 12]
    # Convert inputs to PyTorch tensors and move them to the appropriate device
    image = observation['image'].transpose(2, 0, 1)  # Convert from HWC to CHW
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=True)

    # Assuming vector data is also present
    vector_tensor = torch.tensor(observation['vector'], dtype=torch.float32, device=device, requires_grad=True)

    image_tensor_policy = image_tensor.permute(1, 2, 0)
    # Prepare the observation dictionary for the policy
    obs_dict = {
        'image': image_tensor_policy.unsqueeze(0),  # Add batch dimension
        'vector': vector_tensor.unsqueeze(0)  # Add batch dimension
    }
    
    # Forward pass to get action and compute gradients
    with torch.enable_grad():
        action = policy.forward(obs_dict)
        action_logits = action[0].squeeze(0)  # Assuming the first tensor is action logits
        action_value = action_logits.max()  # Use the max action value for simplicity
        action_value.backward() 
    
    # Compute saliency maps from gradients for each RGB and masking channel set
    gradients = image_tensor.grad.data.abs()
    rgb_saliency = []
    mask_saliency = []

    for i in range(3):  # Assuming 3 sets of RGB and Mask
        rgb_start_idx = i * 4
        mask_idx = rgb_start_idx + 3
        if mask_idx < gradients.size(0):
            # Compute the max over the three channels for each RGB set
            rgb_saliency.append(gradients[rgb_start_idx:mask_idx, :, :].max(0)[0].cpu().numpy())
            mask_saliency.append(gradients[mask_idx, :, :].cpu().numpy())

    # Calculate average RGB and mask saliency maps
    average_rgb_saliency = np.mean(rgb_saliency, axis=0) if rgb_saliency else None
    average_mask_saliency = np.mean(mask_saliency, axis=0) if mask_saliency else None

    # Visualization setup
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Two plots: one for average RGB, one for average mask

    # Average RGB saliency visualization
    if average_rgb_saliency is not None:
        cax_rgb = axes[0].imshow(average_rgb_saliency, cmap='hot', vmax = 0.03, vmin = 0)
        fig.colorbar(cax_rgb, ax=axes[0])
        axes[0].set_title("Average RGB Saliency Map")
        axes[0].axis('off')

    # Average mask saliency visualization
    if average_mask_saliency is not None:
        cax_mask = axes[1].imshow(average_mask_saliency, cmap='hot', vmax = 0.03, vmin = 0)
        fig.colorbar(cax_mask, ax=axes[1])
        axes[1].set_title("Average Masking Channel Saliency")
        axes[1].axis('off')

    # Save and return the image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frame_image = Image.open(buf)
    plt.close(fig)

    return frame_image




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
    #obs = np.stack([obs, obs, obs])
    step = 0
    #ret, frame = cap.read()
    while not done and step < 200:   
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, done, info = env.step(action)
          #frame_saliency = visualize_saliency(obs)
          #saliency_map.append(frame_saliency)
          solved = info['solved']
          if i < trial:
              frame_n = env.rgb_out
              #mask = env.mask_out
              frames_rgb.append(frame_n)
              #frames_mask.append(mask)
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
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_saliency_video.mp4', np.asarray(saliency_map), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    #skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
