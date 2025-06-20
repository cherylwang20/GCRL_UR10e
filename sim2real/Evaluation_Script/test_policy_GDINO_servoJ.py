import os
import gym
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append("../")
sys.path.append(".")
import torch
import skvideo
import skvideo.io
from PIL import Image
import time
import io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")
import skvideo
import skvideo.io
from src.reachGrasp_env.GdinoReachGraspEnv_servoJ import BasicReachGrasp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecFrameStack,
)
import numpy as np
import cv2 as cv
import csv
import warnings
from tqdm.auto import tqdm
from collections import deque

dt = 20


class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, n_stack=3):
        super().__init__(env)
        self.env = env
        self.n_stack = n_stack
        self.image_frames = deque([], maxlen=n_stack)
        self.vector_frames = deque([], maxlen=n_stack)

        # Assume observation space contains 'image' and 'vector'
        image_space = env.observation_space.spaces["image"]
        vector_space = env.observation_space.spaces["vector"]

        # Update the observation space for stacked images and vectors
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=np.tile(image_space.low, n_stack),
                    high=np.tile(image_space.high, n_stack),
                    dtype=image_space.dtype,
                ),
                "vector": gym.spaces.Box(
                    low=np.tile(vector_space.low, n_stack),
                    high=np.tile(vector_space.high, n_stack),
                    dtype=vector_space.dtype,
                ),
            }
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stack):
            self.image_frames.append(obs["image"])
            self.vector_frames.append(obs["vector"])
        stacked_images = np.concatenate(self.image_frames, axis=-1)
        stacked_vectors = np.concatenate(self.vector_frames, axis=-1)
        obs["image"] = stacked_images
        obs["vector"] = stacked_vectors
        return obs

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.image_frames.append(obs["image"])
        self.vector_frames.append(obs["vector"])
        stacked_images = np.concatenate(self.image_frames, axis=-1)
        stacked_vectors = np.concatenate(self.vector_frames, axis=-1)
        obs["image"] = stacked_images
        obs["vector"] = stacked_vectors
        return obs, reward, done, _, info


# Ignore specific warning
model_num = "baseline"
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")
model = PPO.load(os.getcwd() + "/policy/" + model_num)
policy = model.policy


env_name = "GdinoReachGraspEnv"
env = BasicReachGrasp(render_mode="human", channel=4)
env = CustomFrameStack(env, n_stack=3)

print("Action Space Lower Bounds:", env.action_space.low)
print("Action Space Upper Bounds:", env.action_space.high)

env.reset()

print(env.receive.getActualQ())
# env.render()

frames = []
frames_mask = []
movie = True
view = "end_effector"

reach_distance = []
reach_step = []
reach_solved = []
reach_time = []
trial = 5
pbar = tqdm(total=trial)


success = 0
saliency_map = []
all_rewards = []
total_action = []
i = 0
while i < trial:
    ep_rewards = 0
    solved, done = False, False
    step = 0
    obs = env.reset()
    episode_action = []

    while not done:  # and step < 50:
        if step == 0:
            start_time = time.time()
        action, _ = model.predict(obs, deterministic=False)
        episode_action.append(np.sum(np.abs(action[:5])))
        obs, reward, done, _, info = env.step(action)
        # frame_saliency = visualize_saliency(obs)
        # saliency_map.append(frame_saliency)
        solved = info["solved"]
        if i < trial:
            frames.append(env.rgb_out)
            frames_mask.append(env.mask_out)
        step += 1
        ep_rewards += reward
    end_time = time.time()  # End timing after the step completes
    print(step)

    if step > 5:
        pbar.update(1)
        i += 1
        if solved:
            success += 1
            reach_solved.append(True)
        else:
            reach_solved.append(False)
        reach_time.append(end_time - start_time)
        last_distance = np.linalg.norm(env.obs_dict["reach_err"], axis=-1)
        reach_distance.append(last_distance)
        print(last_distance)
        reach_step.append(step)
        total_action.append(np.sum(episode_action))
        all_rewards.append(ep_rewards)

print(f"Average reward: {np.mean(all_rewards)}")
env.reset()
env.close()
print(f"Success rate: {success/trial}")
print(
    "reach distance:", reach_distance, "average reach_distance", np.mean(reach_distance)
)

print("reach step:", reach_step, "average reach step", np.mean(reach_step))
print("total action sum:", total_action, "average action sum", np.mean(total_action))
print("averaged reach time:", reach_time, "average action sum", np.mean(reach_time))

assert (
    len(reach_distance) == len(reach_solved) == len(reach_step)
), "Arrays must be of the same length"

# CSV file name
filename = f"dt{dt}_4msSim2Real_Results.csv"

# Header labels
headers = ["Distance", "Solved", "Total Steps", "Sum of Actions", "Averaged Time"]

# Write to a CSV file
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header
    for data in zip(reach_distance, reach_solved, reach_step, total_action, reach_time):
        writer.writerow(data)  # Write the data rows

print(f"Data has been written to {filename}")


if movie:
    os.makedirs("./videos" + "/" + env_name, exist_ok=True)
    skvideo.io.vwrite(
        "./videos" + "/" + env_name + "/servoj/ID4ms" + model_num + f"{view}_video.mp4",
        np.asarray(frames),
        inputdict={"-r": "50"},
        outputdict={"-pix_fmt": "yuv420p"},
    )
    # skvideo.io.vwrite('./videos'  +'/' + env_name + '/servoj/' + model_num + f'{view}_saliency_video.mp4', np.asarray(saliency_map), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite(
        "./videos"
        + "/"
        + env_name
        + "/servoj/ID4ms"
        + model_num
        + f"{view}_mask_video.mp4",
        np.asarray(frames_mask),
        inputdict={"-r": "50"},
        outputdict={"-pix_fmt": "yuv420p"},
    )
