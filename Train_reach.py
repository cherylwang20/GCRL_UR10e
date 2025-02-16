import gym
import os
from gym import spaces
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch 
import random
import kornia.augmentation as KAug
import kornia.enhance as KEnhance
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
#import mujoco_py
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from datetime import datetime
import time
from wandb.integration.sb3 import WandbCallback

import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_name", type=str, default='N/A', help="environment name")
parser.add_argument("--group", type=str, default='testing', help="environment name")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")

parser.add_argument("--channel_num", type=int, default=4, help="channel num")
args = parser.parse_args()

class KorniaAugmentationCallback(BaseCallback):
    def __init__(self, augment_images, low, high, verbose=0):
        super().__init__(verbose)
        self.low = low
        self.high = high
        self.augment_images = augment_images
        self.display = False
        self.transform = torch.nn.Sequential(
            KAug.RandomContrast(contrast=(low, high), clip_output=True, p=0.8),
            KAug.RandomBrightness((low, high)),
            KAug.RandomSaturation((low, high)), 
            KAug.RandomGaussianBlur(kernel_size=(5, 5), sigma=(low, high), p=0.5)
        )
    
    def show_images(self, images, title='Image'):
        #print(images)
        images = images * 255
        images = images.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        for index, img in enumerate(images):
            cv2.imwrite(f'test' + title + f'_{index}.png', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _on_rollout_end(self):
        # Assume observations['image'] has the shape [batch_size, height, width, channels]
        images = self.model.rollout_buffer.observations['image']

        images = images.reshape(images.shape[0] * images.shape[1], 120, 212, 12)

        # Reshape to separate frames and channels
        # New shape: [batch_size, height, width, num_frames, channels_per_frame]
        images = np.split(images, 3, axis = 3)

        final_tensors = []

        for image in images:
            image = torch.from_numpy(image)
            image = image.permute(0, 3, 1, 2)
            rgb_channel = image[:, :3, :, :]
            mask_channel = image[:, 3:4, :, :]
            
            if self.display:
                # Display original first half
                self.show_images(rgb_channel, title='Original First Half')

            rgb_transform = self.transform(rgb_channel)

            if self.display:
                # Display original first half
                self.show_images(rgb_transform, title='After')

            enhanced_transform = torch.empty_like(rgb_transform)
            idx = np.random.choice(len(self.augment_images), size=rgb_channel.size(0), replace=True)
            alpha = np.random.uniform(self.low, 1, size=rgb_channel.size(0))
            beta, gamma = 1 - alpha, np.zeros(rgb_channel.size(0))

            self.augment_images = np.array(self.augment_images)

            #enhanced_transform = KEnhance.add_weighted(rgb_transform, alpha[:, None, None, None], torch.from_numpy(self.augment_images[idx]), beta[:, None, None, None], gamma[:, None, None, None])

            #augmented_tensor = torch.cat((enhanced_transform, mask_channel), dim=1)
            augmented_tensor = torch.cat((rgb_transform, mask_channel), dim=1)

            final_tensors.append(augmented_tensor)
        
        concatenated = torch.cat(final_tensors, dim=1)
        final_output = concatenated.unsqueeze(1) 
        final_output = final_output.permute(0, 1, 3, 4, 2)

        #print("Final output shape:", final_output.numpy().shape)

        # Update the observations in the rollout buffer
        self.model.rollout_buffer.observations['image'] = final_output.numpy()

        return True
    
    def _on_step(self):
        return True

def load_images(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    for file in image_files:
        image = Image.open(file).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((120, 212)),
            transforms.ToTensor()
        ])
        image = transform(image)
        image = image[[2, 1, 0], :, :]
        images.append(image)
    return images

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        obs_vecs = self.training_env.get_attr('get_obs_vec')  # Returns a list of observations from all environments
        average_obs = np.mean([np.mean(obs) for obs in obs_vecs], axis=0)  # Compute the average observation
        self.logger.record("average_obs", average_obs)
        return True

class CustomDictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Adjust features_dim if needed
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)

        num_input_channels = observation_space.spaces['image'].shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=8, stride=4, padding=2),  # Adjust padding to fit your needs
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for feature concatenation
        )

        # Vector processing network
        self.mlp = nn.Linear(observation_space.spaces['vector'].shape[0], 512)
        
        print(observation_space.spaces.keys())

        # Calculate the total concatenated feature dimension
        self._features_dim = 24960 + 512 # Adjust based on actual output dimensions of cnn and mlp

    def forward(self, observations):
        image = observations['image'].permute(0, 3, 1, 2)
        image_features = self.cnn(image)
        vector_features = self.mlp(observations['vector'])
        concatenated_features = torch.cat([image_features, vector_features], dim=1)
        return concatenated_features

class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMultiInputPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=CustomDictFeaturesExtractor,
                                                     features_extractor_kwargs={},
                                                     net_arch=[{'vf': [512, 512], 'pi': [512, 512]}])  # Adjust architecture if needed

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_env(env_name, idx,  channel, seed=0,eval_mode=False):
    def _init():
        env = gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=eval_mode, channel = channel)
        env.seed(seed + idx)
        return env
    return _init

def main():


    training_steps = 1500000
    env_name = args.env_name
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ENTROPY = 0.01
    LR = linear_schedule(args.learning_rate)
    CR = linear_schedule(args.clip_range)

    time_now = time_now + str(args.seed)

    IS_WnB_enabled = True

    #loaded_model = '2024_09_25_13_42_113'
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        config = {
            "policy_type": 'PPO',
            'name': time_now,
            "total_timesteps": training_steps,
            "env_name": env_name,
            "dense_units": 512,
            "activation": "relu",
            "max_episode_steps": 250,
            "seed": args.seed,
            "entropy": ENTROPY,
            "lr": args.learning_rate,
            "CR": args.clip_range,
            "num_envs": args.num_envs,
            "loaded_model": loaded_model,
        }
        #config = {**config, **envs.rwd_keys_wt}
        run = wandb.init(project="RL-Chemist_Reach",
                        group=args.group,
                        settings=wandb.Settings(start_method="fork"),
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        monitor_gym=True,  # auto-upload the videos of agents playing the game
                        save_code=True,  # optional
                        )
    except ImportError as e:
        pass 

    augment_images = load_images('background/212x120')
    augment_callback = KorniaAugmentationCallback(
                    augment_images=augment_images,
                    low = 1, 
                    high = 1
                    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    num_cpu = args.num_envs

    env = DummyVecEnv([make_env(env_name, i, seed=args.seed, channel = args.channel_num) for i in range(num_cpu)])
    env.render_mode = 'rgb_array'
    envs = VecVideoRecorder(env, "videos/" + env_name + '/training_log' ,
        record_video_trigger=lambda x: x % 30000 == 0, video_length=250)
    envs = VecMonitor(env)
    envs = VecFrameStack(envs, n_stack = 3)

    ## EVAL
    eval_env = DummyVecEnv([make_env(env_name, i, seed=args.seed,channel = args.channel_num, eval_mode=True) for i in range(1)])
    eval_env.render_mode = 'rgb_array'
    eval_env = VecVideoRecorder(eval_env, "videos/" + env_name + '/training_log' ,
        record_video_trigger=lambda x: x % 30000 == 0, video_length=250)
    eval_envs = VecFrameStack(eval_env, n_stack = 3)
    
    log_path = './Reach_Target_vel/policy_best_model/' + env_name + '/' + time_now + '/'
    eval_callback = EvalCallback(eval_envs, best_model_save_path=log_path, log_path=log_path, eval_freq=2000, n_eval_episodes=20, deterministic=True, render=False)
    
    print('Begin training')
    print(time_now)


    # Create a model using the vectorized environment
    #model = SAC("MultiInputPolicy", envs, buffer_size=1000, verbose=0)
    model = PPO(CustomMultiInputPolicy, envs, ent_coef=ENTROPY, learning_rate=LR, clip_range=CR, n_steps = 2048, batch_size = 64, verbose=0, tensorboard_log=f"runs/{time_now}")
    #model = PPO.load(r"./Reach_Target_vel/policy_best_model/" + env_name + '/' + loaded_model + '/best_model', envs, verbose=1, tensorboard_log=f"runs/{time_now}")

    #callback = CallbackList([augment_callback, eval_callback, WandbCallback(gradient_save_freq=100)])
    callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=100)])
    

    model.learn(total_timesteps=training_steps, callback=callback)# , tb_log_name=env_name + "_" + time_now)

    if IS_WnB_enabled:
        run.finish()

if __name__ == "__main__":
    # TRAIN
    main()