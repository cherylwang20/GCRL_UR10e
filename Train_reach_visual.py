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

env_name = "UR10eReachFixed_v2d-v0"

detect_color = 'red'

log_path = './Reach_Target-chem/policy_best_model/' + env_name + '/' + time_now + '/'
env = gym.make(f'mj_envs.robohive.envs:{env_name}')
env.set_color(detect_color)
eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)
print('Begin training')


policy_kwargs = {
    'activation_fn': torch.nn.modules.activation.ReLU,
    'net_arch': {'pi': [128, 128], 'vf': [128, 128]}
    }
#model_num = '2024_06_19_22_21_33'
model = PPO('MultiInputPolicy', env, verbose=0) #, ent_coef= 0.01, policy_kwargs =policy_kwargs)
#model = PPO.load(r"C:/Users/chery/Documents/RL-Chemist/Reach_Target-chem/policy_best_model/" + env_name + '/' + model_num + '/best_model', env, verbose=0, policy_kwargs=policy_kwargs)

obs_callback = TensorboardCallback()
callback = CallbackList([eval_callback])

model.learn(total_timesteps= 5000000, tb_log_name=env_name+"_" + time_now, callback=callback)


