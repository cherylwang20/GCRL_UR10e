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
	
model_num = '2024_07_12_13_53_06' #'2024_06_22_19_48_33'
env_name = "UR10eReachFixed-v3"
movie = True
frame_width = 200
frame_height = 200
#cap = cv.VideoCapture(0)

model = PPO.load('./Reach_Target_vel/policy_best_model/' + env_name +'/' + model_num + r'/best_model')



# Access the policy network directly (for inspection or modification)
policy_net = model.policy

print(policy_net)

# Example: Print out weights of the first layer (modify as needed)
#for name, param in policy_net.named_parameters():
      # Adjust layer name based on your model architecture
    #print(f"Layer: {name} | Size: {param.size()} | Values: \n{param.data}")

