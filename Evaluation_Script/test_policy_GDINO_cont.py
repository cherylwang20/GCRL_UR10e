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
from src.reachGrasp_env.GdinoReachGraspEnv_cont import BasicReachGrasp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
import numpy as np
import cv2 as cv
import warnings
from tqdm.auto import tqdm
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

model_num = "2024_12_14_16_33_305" #"2024_11_19_15_09_082"
model = PPO.load('./' + model_num + r'/best_model')
policy = model.policy


def visualize_saliency(observation):
    device = next(policy.parameters()).device
    
    # Convert inputs to PyTorch tensors and move them to the appropriate device
    image = observation['image'].transpose(2, 0, 1)  # Convert from HWC to CHW
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=True)
    vector_tensor = torch.tensor(observation['vector'], dtype=torch.float32, device=device, requires_grad=True)

    image_tensor_policy = image_tensor.permute(1, 2, 0)
    
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
    
    # Compute saliency maps from gradients
    image_saliency = image_tensor.grad.data.abs().max(dim=0)[0].cpu().numpy()  # Ensure data is on CPU and in NumPy format
    vector_saliency = vector_tensor.grad.data.abs().cpu().numpy()  # Ensure data is on CPU and in NumPy format
    
    # Compute the average of the three repeated 17-element arrays
    average_vector_saliency = np.mean([vector_saliency[i*14:(i+1)*14] for i in range(3)], axis=0)

    # Visualization of both saliency maps
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Image saliency visualization
    cax1 = axes[0].imshow(image_saliency, cmap='hot', vmax=0.03, vmin=0)
    fig.colorbar(cax1, ax=axes[0])
    axes[0].set_title("Image Saliency Map")
    
    # Vector saliency visualization using average
    axes[1].bar(range(14), average_vector_saliency, color='purple')
    axes[1].set_title("Average Vector Saliency")
    axes[1].set_ylim([0, 0.3])

    # Save plot to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frame_image = Image.open(buf)
    plt.close(fig)

    return frame_image

env_name = 'GdinoReachGraspEnv_cont'
env = BasicReachGrasp(render_mode='human')
env = CustomFrameStack(env, n_stack=3)

print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

env.reset()

print(env.receive.getActualQ())
env.render()

frames = []
frames_mask = []
saliency_map = []
movie = True
view = 'end_effector'


trial = 2
success = 0

all_rewards = []
for i in tqdm(range(trial)):
    ep_rewards = 0
    solved, done = False, False
    obs = env.reset()
    step = 0
    while not solved and step < 40:    
          action, _ = model.predict(obs, deterministic=False)
          #print(action)
          obs, reward, done, _, info = env.step(action)
          frame_saliency = visualize_saliency(obs)
          saliency_map.append(frame_saliency)
          solved = info['solved']
          if i < trial:
              frames.append(env.rgb_out)
              frames_mask.append(env.mask_out)
          step += 1
          ep_rewards += reward
    print(step)
    if solved:
        success += 1
    all_rewards.append(ep_rewards)
env.control.speedStop()

print(f"Average reward: {np.mean(all_rewards)}")
env.close()
print(f"Success rate: {success/trial}")


if movie:
    os.makedirs('./videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_saliency_video.mp4', np.asarray(saliency_map), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})