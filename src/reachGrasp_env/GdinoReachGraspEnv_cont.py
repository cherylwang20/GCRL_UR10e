#change to 3 for colour image
N_CHANNELS = 3

import sys
import os
import math
sys.path.append('../')
sys.path.append('.')
sys.path.append('../GroundingDINO')
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from PIL import Image, ImageDraw
from torchvision.ops import box_convert
import torch

import gym
import rtde_receive
import rtde_control
import dashboard_client
import time
import numpy as np
import random
import robotiq_gripper
import csv
import collections
import pyrealsense2 as rs
import cv2 as cv
#import tkinter

#change to 3 for colour image
N_CHANNELS = 3

FIXED_START = [4.78, -2.06, 2.62, 3.05, -1.58, 0]
FIXED_START_2 = [4.779965400695801, -2.07406335348, 2.6200059095965784, 3.05422013445491455, -1.58000356355776, 1.430511474609375e-05]
FIXED_START_3 = [1.62, -0.81679, -2.60786, -0.43981, 1.57924, 0.045]

class BasicReachGrasp(gym.Env):
    
    DEFAULT_PROPRIO_KEYS = [
        'qp_robot', 'qv_robot'
    ]
    metadata = {
        "render_modes": ["None", "human", "rgb_array","red_channel"],
        "render_fps": 50,
    }
    def __init__(self, render_mode=None):
        super(BasicReachGrasp, self).__init__()
        self.render_mode = render_mode
        self.step_count = 0  # steps
        self.episode_count = 0
        self.STEPS_IN_EPISODE = 50
        self.rwd_keys_wt = {
                                "reach": -1, #the reach reward here is the distance
                                "contact": 1,
                                'sparse': 0,
                                'solved': 0,
                                "done": 10,
                            }
        #self.STEP_SIZE = 0.05


        self.LAST_ACTION = None
        self.LAST_STATE = None
        self.LAST_IMAGE_STATE = None
        self.DEFAULT_OBS_KEYS = [
        'qp_robot', 'qv_robot'
    ]

        self.HOST = "192.168.0.110"  # the IP address 127.0.0.1 is for URSim, 192.168.0.110 for UR10E

        self.HEIGHT = 848
        self.WIDTH = 480
        self.rgb_out = np.ones((self.WIDTH, self.HEIGHT))
        self.mask_out = np.ones((self.WIDTH, self.HEIGHT))
        self.CAMERA = rs.pipeline()
        config = rs.config()


        # Start streaming
        self.CAMERA.start(config)
        den = 1

        self.robot_pos_bound = np.array([[-6.28, 6.28], 
                                [-6.28, 6.28],
                                [-6.28, 6.28],
                                [-6.28, 6.28],
                                [-6.28, 6.28],
                                [-0.7, 0.7],
                                [0, 0.8]])

        self.robot_vel_bound = np.array([
                                [-2.09/ den, 2.09/ den], 
                                [-2.09/ den, 2.09/ den], 
                                [-1.57/ den, 1.57/ den],
                                [-1.57/ den, 1.57/ den],
                                [-1.57/ den, 1.57/ den],
                                [-1.57/ den, 1.57/ den],
                                [-3, 3]
                            ])


        self.control = rtde_control.RTDEControlInterface(self.HOST)
        self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
        self.dashboard = dashboard_client.DashboardClient(self.HOST)
        self.grip_status = None
        self.done = False

        print("Creating gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        self.gripper.connect(self.HOST, 63352)

        print("Activating gripper...")
        self.gripper.activate()
        #[4.549111843109131, -0.7404842537692566, 1.6890791098224085, 2.703993006343506, -1.430075470601217, -3.2127202192889612]

        self.mask_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.BOX_THRESHOLD = 0.45
        self.TEXT_THRESHOLD = 0.25
        self.TEXT_PROMPT = 'toy dinosaur'
        self.GDINO_Coord = [0, 0]
        self.GDINO_array = []


        obs_range = (-10, 10)
        self.obs_dict = {}
        #Define OpenAI Gym action and state spaces
        self.normalize_act = True
        self.current_image = None
        self.contact = 3
        self.done = False
        self.mask_image = np.ones((self.WIDTH, self.HEIGHT, ))
        self.DT = 0.1
        self.target_x, self.target_y = 0, 0
        self.step_count = 0
        self.vel_action = [0] * 6
        self.duck_pos = [-0.19879814317014585, 1.1419941786129197, 0.09736425260562352]
        self.frog_pos = [-0.45390599718515445, 1.002975621079947, 0.05415598941643378]
        self.goal_pos = [-0.035703643440070995, 0.9983709711752897, -0.19109855918934883]
        act_low = -np.ones(7) 
        act_high = np.ones(7) 
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32) # clockwise or counterclockwise, for each of the 5 moving joints
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(120, 212, 4), dtype=np.float32),  # Use np.float32 here
            'vector': gym.spaces.Box(obs_range[0]*np.ones(14), obs_range[1]*np.ones(14), dtype=np.float32)  # Ensure consistency in dtype usage
        })
    
    def get_obs_dict(self):
        self.obs_dict['qp_robot'] = self.receive.getActualQ() + [0.159 ]
        self.obs_dict['qv_robot'] = np.concatenate((self.vel_action[:6].copy(), [0]))
        self.obs_dict['reach_err'] = np.array(self.goal_pos) - np.array(self.receive.getActualTCPPose()[:3]) #use the end_effector to check
        #self.current_observation = self.get_observation(show=True)

        return self.obs_dict
    
    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        #print(reach_dist)
        self.depth = reach_dist
        rwd_dict = collections.OrderedDict((
            # Optional Keys[]
            ('reach',  np.array([reach_dist ])),
            ('contact', np.array([self.contact == 2])),
            ('penalty', np.array([-1])),
            ('sparse', np.array([0.])),
            ('solved',  np.array([reach_dist < 0.13])),
            ('done', np.array([False])), #    obj_height  - self.obj_init_z > 0.2, #reach_dist > far_th
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def vtp_step(self, ctrl_desired, last_qpos, n_frames = 1):
        """
        Apply controls and step forward in time
        INPUTS:
            ctrl_desired:       Desired control to be applied(sim_space)
            step_duration:      Step duration (seconds)
            ctrl_normalized:    is the ctrl normalized to [-1, 1]
            realTimeSim:        run simulate real world speed via sim
        """
        control_v = (self.robot_vel_bound[:7, 1]+self.robot_vel_bound[:7, 0])/2.0 + ctrl_desired*(self.robot_vel_bound[:7, 1]-self.robot_vel_bound[:7, 0])/2.0
        
        #gripper current position:
        current_grip = self.gripper.get_current_position()
        new_grip = np.clip(int(current_grip + control_v[-1]*self.DT), self.robot_pos_bound[6, 0], self.robot_pos_bound[6, 1])

        for _ in range(n_frames):
            self.gripper.move(0, 255, 255)
            protective_stop = self.speedJ(control_v[:6])
            
        return control_v, control_v
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def speedJ(self, new_vel):
        success = self.control.speedJ(new_vel, acceleration = 0.1, time = 0)
        time.sleep(self.DT)
        while not success:
            if self.receive.isProtectiveStopped(): 
                self.reconnect()
                protective_stop = True
                time.sleep(6)  # cannot unlock protective stop before it has been stopped for 5 seconds
                print(self.dashboard.safetystatus())
                self.dashboard.unlockProtectiveStop()
                time.sleep(2)
                print(self.dashboard.safetystatus())
                self.dashboard.closeSafetyPopup()
                #self.reconnect()

                if not self.receive.isProtectiveStopped():
                    self.episode_count += 1
                    self.done = True
                    print('epsiode ends with protective stop')

                return protective_stop
            else:
                self.reconnect()
                success = self.control.speedJ(new_vel, acceleration = 0.01)
                time.sleep(0.4)
    
    
    def step(self, a):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        change control method here if needed 
        """
        self.state = self.receive.getActualQ()
        self.LAST_STATE = np.append(self.state.copy(), self.gripper.get_current_position()) # save the last state before we take the action. and normalized the gripper to [-1, 1]
        self.LAST_ACTION = a
        
        frames = self.CAMERA.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        self.image_state = np.asanyarray(color_frame.get_data())
        self.image_state = cv.cvtColor(self.image_state, cv.COLOR_BGR2RGB)


        #self.LAST_IMAGE_STATE = self.image_state.copy()

        a = np.clip(a, self.action_space.low, self.action_space.high)
        self.last_ctrl, self.vel_action = self.vtp_step(ctrl_desired=a,
                                    last_qpos = self.LAST_STATE,
                                    )
        if self.done:
            self.reconnect()
            self.control.reuploadScript()
            return self.forward()
        
        info = {}
        self.step_count += 1

        self.contact = self.gripper._get_var('OBJ')

        if self.control == 2:
            old_pos = self.receive.getActualQ()
            new_pos = old_pos
            new_pos[1] = new_pos[1] - 0.3
            self.control.moveJ(new_pos)
            self.episode_count += 1
            if self.gripper._get_var('OBJ') == 2:
                self.rwd_dict['dense'] += 10
                print('episode ends with successful gripping')
            else:
                print('gripping not successful')
            self.done = True
            self.contact.moveJ(old_pos)
            self.gripper.move_and_wait_for_pos(0, 25, 25)
            return self.forward()
            
        if self.step_count == self.STEPS_IN_EPISODE:
            print("episode " + str(self.episode_count) + " terminated")
            self.step_count = 0
            self.episode_count += 1
            self.done = True
            info["TimeLimit.truncated"] = True
            return self.forward()

        return self.forward()
    
    
    def forward(self):
        """
        Forward propagate env to recover env details
        Returns current obs(t), rwd(t), done(t), info(t)
        """

        self.obs_dict = self.get_obs_dict()
        self.rwd_dict = self.get_reward_dict(self.obs_dict)

        env_info = {
            'rwd_dense': self.rwd_dict['dense'],    # MDP(t)
            'rwd_sparse': self.rwd_dict['sparse'],  # MDP(t)
            'solved': self.rwd_dict['solved'],      # MDP(t)
            'done': self.rwd_dict['done'],          # MDP(t)
            'obs_dict': self.obs_dict,                  # MDP(t)
            'rwd_dict': self.rwd_dict,                  # MDP(t)
        }

        obsvec = np.zeros(0)
        for keys in self.DEFAULT_OBS_KEYS:
            obsvec = np.concatenate([obsvec, self.obs_dict[keys]])
        
        ##
        self.mask_image = self.get_mask(self.image_state)

        self.current_image = np.concatenate((self.image_state/255, np.expand_dims(self.mask_image/255, axis=-1)), axis=2)


        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        #print(image.size)
        obs = {'image': self.current_image.reshape((120, 212, 4)), 'vector': obsvec}

        return obs, env_info['rwd_dense'], self.done, None, env_info
    
    def get_mask(self, rgb):
        mask = np.zeros(( self.WIDTH,  self.HEIGHT), dtype=np.uint8)
        
        #rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        boxes, logits, phrases = predict(
            model=self.mask_model,
            image=self.load_image2(pil_image),
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD
            )
        self.rgb_out = annotate(image_source=rgb, boxes=boxes, logits=logits, phrases=phrases)
        if logits.nelement() > 0:
            max, indices = torch.max(logits, dim = 0)
            boxes = boxes.numpy()
            boxes = boxes[indices]
        
        mask = np.zeros((self.WIDTH,  self.HEIGHT), dtype=np.uint8)

        mask = self.create_mask(mask, boxes=boxes)

        self.mask_out = mask

        mask = cv.resize(mask, dsize=(120, 212), interpolation=cv.INTER_CUBIC)
        self.image_state = cv.resize(rgb, dsize=(120, 212), interpolation=cv.INTER_CUBIC)

        return mask
    
    def load_image2(self, image_source):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_source, None)
        return image_transformed

    def create_mask(self, image_source: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
        """
        This function creates a mask with white rectangles on a black background,
        where the rectangles are defined by the bounding boxes.

        Parameters:
        image_source (np.ndarray): The source image for determining the size of the mask.
        boxes (torch.Tensor): A tensor containing bounding box coordinates in cxcywh format.

        Returns:
        np.ndarray: The mask image.
        """
        # Get the dimensions of the source image
        h, w = image_source.shape

        # Scale the boxes to the image dimensions
        boxes = torch.tensor(boxes, dtype=torch.float32) * torch.Tensor([w, h, w, h])

        # Convert boxes from cxcywh to xyxy format
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Create a black mask
        mask = np.zeros((h, w), dtype=np.uint8)


        # Draw each box as a white rectangle on the mask
        if xyxy.size != 0:
            top_left = (int(xyxy[0]), int(xyxy[1]))
            bottom_right = (int(xyxy[2]), int(xyxy[3]))
            cv.rectangle(mask, top_left, bottom_right, (255), thickness=-1)  # Fill the rectangle
            white_pixels = np.argwhere(mask == 255)
        
        # Calculate the mean of each column (x, y coordinates)
            centroid = np.mean(white_pixels, axis=0).astype(int)  # Returns (y, x)

        # Convert from (row, col) to (x, y)
            centroid = (centroid[1], centroid[0])


        return mask
    
    def reconnect(self):

        while not self.control.isConnected():
            print("Control not connected, reconnecting...")
            self.control = rtde_control.RTDEControlInterface(self.HOST)
            time.sleep(5)

        while not self.receive.isConnected():
            print("Receive not connected, reconnecting...")
            self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
            time.sleep(5)

        while not self.dashboard.isConnected():
            print("Dashboard not connected, reconnecting...")
            self.dashboard = dashboard_client.DashboardClient(self.HOST)
            self.dashboard.connect()
            time.sleep(5)


    def reset(self, seed=None, options = None):
        print("reset has been called")

        self.control.speedStop()
        start_q = FIXED_START_2

        success = self.control.moveJ(start_q, speed=0.25, acceleration=0.5)
        

        while not success:
            if self.receive.isProtectiveStopped():
                self.reconnect()
                time.sleep(6)  # cannot unlock protective stop before it has been stopped for 5 seconds
                self.dashboard.unlockProtectiveStop()
            else:
                self.reconnect()
                self.control.reuploadScript()
                print('reuploadingscript')
                success = self.control.moveJ(start_q, speed=0.25, acceleration=0.5)
                time.sleep(1)
        
        self.gripper.move_and_wait_for_pos(0, 25, 25)
        self.state = self.receive.getActualQ()  # this is a 6D array, including the position of wrists 2 & 3
        if self.state == []:
            raise ValueError("Connection issue in reset")
        
        target_names = ['toy rubber duck', 'green apple']
        self.TEXT_PROMPT = np.random.choice(target_names)
        print(self.TEXT_PROMPT)
        
        frames = self.CAMERA.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        self.image_state = np.asanyarray(color_frame.get_data())

        #obs = super().reset(reset_qpos = reset_qpos, reset_qvel = None, **kwargs)
        #self._last_robot_qpos = self.sim.model.key_qpos[0].copy()
        self.final_image = np.ones((120, 212, 4), dtype=np.uint8)
        self.step_count = 0
        self.done = False
        obs = self.receive.getActualQ()+ [0.159]+  self.receive.getActualQd() + [0]
        return {'image': self.final_image, 'vector': obs}

    def reconnect(self):

        while not self.control.isConnected():
            print("Control not connected, reconnecting...")
            self.control = rtde_control.RTDEControlInterface(self.HOST)
            time.sleep(5)

        while not self.receive.isConnected():
            print("Receive not connected, reconnecting...")
            self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
            time.sleep(5)

        while not self.dashboard.isConnected():
            print("Dashboard not connected, reconnecting...")
            self.dashboard = dashboard_client.DashboardClient(self.HOST)
            self.dashboard.connect()
            time.sleep(5)


    def render(self, mode="human"):
        #concatenate the arrays so the display is side by side
        if mode == "human":
            cv.imshow('UR10e View', self.rgb_out)
            cv.imshow('observation', self.mask_out)
            cv.waitKey(2)
        elif mode == 'rgb_array':
            cv.imshow('observation', self.image_state)
            cv.waitKey(5)
            return self.image_state

    def close(self):
        self.CAMERA.stop()
        self.control.disconnect()
        self.receive.disconnect()
        self.dashboard.disconnect()
