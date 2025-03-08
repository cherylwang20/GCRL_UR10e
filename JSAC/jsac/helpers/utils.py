import os
import time
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from gymnasium import spaces
from gymnasium.core import Env
import matplotlib.pyplot as plt


class MODE:
    IMG = 'img'
    IMG_PROP = 'img_prop'
    PROP = 'prop'

def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path

def set_seed_everywhere(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        env.seed(seed)

def save_learning_curve(fname, 
                        returns, 
                        ep_lens, 
                        xtick):
    sns.set_theme(rc={'figure.figsize':(10, 7)})
    sns.set_style("whitegrid")

    df = pd.DataFrame(columns=["Step", "Return"])

    steps = 0
    rets = []
    end_step = xtick
    data = []
    for (i, epi_s) in enumerate(ep_lens):
        if steps + epi_s > end_step:
            if len(rets) > 0:
                data.append([end_step, sum(rets)/len(rets)]) 
                rets = []
            end_step += xtick
        
        steps += epi_s 
        ret = returns[i]  
        rets.append(ret) 
        
    if len(rets) > 0:  
        data.append([end_step, sum(rets)/len(rets)])
    df = pd.DataFrame(data, columns=["Step", "Return"])
    sns.lineplot(x="Step", y='Return', data=df, 
                 color=sns.color_palette('bright')[0], 
                 linewidth=1.5, errorbar=None)

    plt.title('Learning Curve')
    plt.savefig(fname)
    plt.close()
    
def save_eval_learning_curve(fname, 
                             returns, 
                             steps):
    sns.set_theme(rc={'figure.figsize':(10, 7)})
    sns.set_style("whitegrid")

    df = pd.DataFrame(columns=["Step", "Return"])

    curr_step = steps[0]
    rets = [] 
    data = []
    for (i, step) in enumerate(steps):
        if step != curr_step:
            if len(rets) > 0:
                data.append([curr_step, sum(rets)/len(rets)]) 
                rets = []
            curr_step = step
         
        ret = returns[i]  
        rets.append(ret) 
        
    if len(rets) > 0:  
        data.append([curr_step, sum(rets)/len(rets)])
        
    df = pd.DataFrame(data, columns=["Step", "Return"])
    sns.lineplot(x="Step", y='Return', data=df, 
                 color=sns.color_palette('bright')[0], 
                 linewidth=1.5, errorbar=None)

    plt.title('Eval Learning Curve')
    plt.savefig(fname)
    plt.close()


## SRC: https://github.com/kindredresearch/SenseAct/blob/master/senseact/utils.py

class EnvSpec():
    def __init__(self, 
                 env_spec, 
                 observation_space, 
                 action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self._unwrapped_spec = env_spec

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])


class WrappedEnv(Env):
    def __init__(
            self,
            env,
            episode_max_steps=-1,
            is_min_time=False,
            reward_scale=1.0,
            reward_penalty=0,
            steps_penalty=0,
            start_step=0,
            start_episode=0):
        
        if is_min_time:
            assert episode_max_steps > 0
        
        self._wrapped_env = env
        self._episode_max_steps = episode_max_steps
        self._is_min_time = is_min_time
        self._reward_scale = reward_scale
        self._reward_penalty = reward_penalty
        self._steps_penalty = steps_penalty

        self._total_steps  = start_step
        self._episode = start_episode

    def _reset_stats(self):
        self._reward_sum = 0
        self._episode_steps = 0
        if self._is_min_time:
            self._sub_episode = 0
            self._sub_episode_steps=0
        self._start_time = time.time()

    def _monitor(self, reward, done, info):
        self._reward_sum += reward
        self._episode_steps += 1
        self._total_steps += 1 

        new_info = {}
        
        if 'solved' in info:
            new_info['solved'] = info['solved']
            
        if 'x' in info:
            new_info['x'] = info['x']
            new_info['y'] = info['y']
            new_info['prompt'] = info['prompt']
            new_info['reach_err'] = np.array(info['reach_err'])  
            new_info['reach_err'] = np.linalg.norm(new_info['reach_err'])
        
        if 'gdino_step' in info:
            new_info['gdino_step'] = info['gdino_step']
            new_info['gdino_time'] =  info['gdino_time']
            new_info['gdino_accuracy'] =  info['gdino_accuracy']
            
        if 'battery_charge' in info:
            new_info['battery_charge'] = info['battery_charge']

        if not self._is_min_time and self._episode_max_steps > 0 \
            and self._episode_steps == self._episode_max_steps:
            new_info['truncated'] = True

        if 'TimeLimit.truncated' in info or 'truncated' in info:
            new_info['truncated'] = True

        if done or (not self._is_min_time and 'truncated' in new_info):
            new_info['episode'] = self._episode
            new_info['step'] = self._total_steps
            new_info['episode_steps'] = self._episode_steps
            new_info['duration'] = time.time() - self._start_time
            new_info['return'] = self._reward_sum
            self._episode += 1
            return done, new_info
        
        if self._is_min_time:
            self._sub_episode_steps += 1
            if self._sub_episode_steps == self._episode_max_steps:
                self._reward_sum += self._reward_penalty
                self._total_steps += self._steps_penalty
                self._episode_steps += self._steps_penalty
                
                new_info['truncated'] = True
                new_info['episode'] = self._episode
                new_info['sub_episode'] = self._sub_episode
                new_info['sub_episode_steps'] = self._sub_episode_steps 
                self._sub_episode_steps = 0
                self._sub_episode += 1
                return done, new_info
        
        return done, new_info

    def reset(self, create_vid=False, reset_stats=True, **kwargs):
        if create_vid:
            ret = self._wrapped_env.reset(create_vid=create_vid, **kwargs)
        else:
            ret = self._wrapped_env.reset(**kwargs)
        if reset_stats:
            self._reset_stats()
        return ret

    def step(self, action):
        # rescale the action
        if isinstance(self._wrapped_env.action_space, spaces.Box):
            lb = self._wrapped_env.action_space.low
            ub = self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
          
        done, info = self._monitor(reward, done, info)

        return Step(next_obs, reward * self._reward_scale, done, info)

    def __str__(self):
        return "RealTimeEnv: %s" % self._wrapped_env


    @property
    def total_steps(self):
        return self._total_steps

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def start(self):
        return self._wrapped_env.start()

    def close(self):
        if hasattr(super(), 'close'):
            super().close()
        return self._wrapped_env.close()


    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def __getattr__(self, attr):
        try:
            orig_attr = self._wrapped_env.__getattribute__(attr)
            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    result = orig_attr(*args, **kwargs)
                    # prevent wrapped_class from becoming unwrapped
                    if result == self._wrapped_env:
                        return self
                    return result
                return hooked
            else:
                return orig_attr
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


def render_interactive(env):
    ## IMPORTS FOR INTERACTIVE RENDERING
    import mujoco
    import cv2
    # env = Franka_Env(env_name="FrankaEnv-v0",render_interactive=True) # replace model path accordingly
    width, height = 640, 480

    ## END EFFECTOR CAMERA VIEW
    camera_name = "end_effector_cam"
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = cam_id
    # Set up an offscreen rendering context specifically for capturing images
    offscreen_context = mujoco.MjrContext(env.model, 0)
    viewport = mujoco.MjrRect(0, 0, width, height)
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    print(env.model.jnt_range)
    # print(env.model.njnt)
    while env.viewer.is_running():
        mujoco.mj_step(env.model, env.data, 20)
        env.sync_view()
         # Update the scene from the specified camera for rendering
        mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, offscreen_context)

        # Retrieve the image from the offscreen buffer
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_image, None, viewport, offscreen_context)

        # Flip the image vertically as MuJoCo renders it upside-down
        rgb_image = np.flipud(rgb_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # Display or process the captured image
        cv2.imshow("End Effector Camera View", bgr_image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()