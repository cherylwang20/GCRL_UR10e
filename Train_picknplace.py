import gym
from gym import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
#import mujoco_py
from datetime import datetime
import time

class TensorboardCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, verbose=0):
	    super(TensorboardCallback, self).__init__(verbose)

	def _on_step(self) -> bool:
	    # Log scalar value (here a random variable)
	    value = self.training_env.get_obs_vec()
	    self.logger.record("obs", value)
	
	    return True
	


start_time = time.time()
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

env_name = "UR10ePickPlaceFixed-v0"

log_path = './Pick&Place_Target/policy_best_model/' + env_name + '/' + time_now + '/'
#env = gym.make(f'mj_envs.robohive.envs.myo:{"myoHandPoseRandom-v0"}')
env = gym.make(f'mj_envs.robohive.envs:{env_name}')
eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)
print('Begin training')
policy_kwargs = {
    'activation_fn': torch.nn.modules.activation.ReLU,
    'net_arch': {'pi': [256, 256], 'vf': [256, 256]}
    }

model = PPO('MlpPolicy', env, verbose=0, policy_kwargs =policy_kwargs)
obs_callback = TensorboardCallback()
callback = CallbackList([eval_callback])

model.learn(total_timesteps= 500000, tb_log_name=env_name+"_" + time_now, callback=callback)


