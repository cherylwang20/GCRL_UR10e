import gym
from gym import spaces
import torch 
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
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
parser.add_argument("--env_name", type=str, default=1, help="environment name")
parser.add_argument("--group", type=str, default='testing', help="environment name")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")

args = parser.parse_args()

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

class CustomNeptuneCallback(BaseCallback):
    def __init__(self, run):
        super(CustomNeptuneCallback, self).__init__(verbose=1)
        self.run = run
        # You might want to add more parameters here if needed

    def _on_step(self) -> bool:
        # Check if an episode has ended
        if 'episode' in self.locals["infos"][0]:
            episode_info = self.locals["infos"][0]['episode']
            # Log episodic information to Neptune
            self.run["metrics/episode_reward"].append(episode_info['r'])
            self.run["metrics/episode_length"].append(episode_info['l'])

        return True

class CustomDictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Adjust features_dim if needed
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),  # Adjust padding to fit your needs
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for feature concatenation
        )

        # Vector processing network
        self.mlp = nn.Linear(observation_space.spaces['vector'].shape[0], 16)
        
        print(observation_space.spaces.keys())

        # Calculate the total concatenated feature dimension
        self._features_dim = observation_space.spaces['image'].shape[0]**2 + 16  # Adjust based on actual output dimensions of cnn and mlp

    def forward(self, observations):
        image = observations['image'].permute(0, 3, 1, 2)
        image_features = self.cnn(image)
        vector_features = self.mlp(observations['vector'])
        return torch.cat([image_features, vector_features], dim=1)

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

def make_env(env_name, idx, seed=0, eval_mode=False):
    def _init():
        env = gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=eval_mode)
        env.seed(seed + idx)
        return env
    return _init

def main():

    training_steps = 2500000
    env_name = args.env_name
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ENTROPY = 0.01
    LR = linear_schedule(args.learning_rate)
    CR = linear_schedule(args.clip_range)

    time_now = time_now + str(args.seed)

    IS_WnB_enabled = True

    loaded_model = 'N/A'
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
            "max_episode_steps": 200,
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # num_cpu = args.num_envs
    num_envs = 4
    num_eval_envs = 1

    env = SubprocVecEnv([make_env(env_name, i, seed=args.seed) for i in range(num_envs)])
    env.render_mode = 'rgb_array'
    envs = VecVideoRecorder(env, "videos/" + env_name + '/training_log' ,
        record_video_trigger=lambda x: x % 30000 == 0.3333, video_length=300)
    envs = VecMonitor(envs)

    ## EVAL
    eval_env = SubprocVecEnv([make_env(env_name, i, seed=args.seed, eval_mode=True) for i in range(num_eval_envs)])
    eval_env.render_mode = 'rgb_array'
    eval_envs = VecVideoRecorder(eval_env, "videos/" + env_name + '/training_log' ,
        record_video_trigger=lambda x: x % 30000 == 0, video_length=300)

    log_path = './Reach_Target_vel/policy_best_model/' + env_name + '/' + time_now + '/'
    eval_callback = EvalCallback(eval_envs, best_model_save_path=log_path, log_path=log_path, eval_freq=2000, n_eval_episodes=20, deterministic=True, render=False)
    
    print('Begin training')
    print(time_now)


    # Create a model using the vectorized environment
    #model = SAC("MultiInputPolicy", envs, buffer_size=1000, verbose=0)
    model = PPO(CustomMultiInputPolicy, envs, ent_coef=ENTROPY, learning_rate=LR, clip_range=CR, verbose=0, tensorboard_log=f"runs/{time_now}")
    #model = PPO.load(r"./Reach_Target_vel/policy_best_model/" + env_name + '/' + loaded_model + '/best_model', envs, verbose=1, tensorboard_log=f"runs/{time_now}")

    obs_callback = TensorboardCallback()
    callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=100,
                model_save_freq=1000,
                model_save_path=f"models/{time_now}")])#, obs_callback])
    

    model.learn(total_timesteps=training_steps, callback=callback)# , tb_log_name=env_name + "_" + time_now)

    if IS_WnB_enabled:
        run.finish()

if __name__ == "__main__":
    # TRAIN
    main()