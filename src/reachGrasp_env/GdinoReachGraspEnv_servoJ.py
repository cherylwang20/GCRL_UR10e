# change to 3 for colour image
N_CHANNELS = 3

import sys
import os
import math

sys.path.append("../")
sys.path.append("./")
sys.path.append("../GroundingDINO")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import groundingdino
import threading
import queue
import matplotlib.pyplot as plt

plt.switch_backend("agg")
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
import reachGrasp_env.robotiq_gripper as robotiq_gripper
import csv
import collections
import pyrealsense2 as rs
import cv2 as cv

# import tkinter

# change to 3 for colour image
N_CHANNELS = 3

FIXED_START = [4.78, -2.06, 2.62, 3.05, -1.58, 0]
FIXED_START_2 = [
    4.779965400695801,
    -2.07406335348,
    2.6200059095965784,
    3.05422013445491455,
    -1.58000356355776,
    1.430511474609375e-05,
]


class BasicReachGrasp(gym.Env):

    DEFAULT_PROPRIO_KEYS = ["qp_robot", "qv_robot"]
    metadata = {
        "render_modes": ["None", "human", "rgb_array", "red_channel"],
        "render_fps": 50,
    }

    def __init__(self, channel=4, render_mode=None):
        super(BasicReachGrasp, self).__init__()
        self.render_mode = render_mode
        self.step_count = 0  # steps
        self.episode_count = 0
        self.episode_count = 0
        self.rwd_keys_wt = {
            "reach": -1,  # the reach reward here is the distance
            "contact": 1,
            "sparse": 0,
            "solved": 0,
            "done": 10,
        }
        # self.STEP_SIZE = 0.05

        self.LAST_ACTION = None
        self.LAST_STATE = None
        self.LAST_IMAGE_STATE = None
        self.DEFAULT_OBS_KEYS = ["qp_robot", "qv_robot"]
        self.GRIPPER_COUNT = 5
        self.gripper_cd = self.GRIPPER_COUNT
        self.hard_grip = False
        self.re_pos = False

        self.HOST = "192.168.0.110"  # the IP address 127.0.0.1 is for URSim, 192.168.0.110 for UR10E

        self.HEIGHT = 848
        self.WIDTH = 480
        self.SPEED = 1.25
        self.ACCELERATION = 0.8
        self.rgb_out = np.ones((self.WIDTH, self.HEIGHT))
        self.mask_out = np.ones((self.WIDTH, self.HEIGHT))
        self.CAMERA = rs.pipeline()
        config = rs.config()

        # Start streaming
        self.CAMERA.start(config)
        den = 1

        self.robot_pos_bound = np.array(
            [
                [-6.28, 6.28],
                [-6.28, 6.28],
                [-6.28, 6.28],
                [-6.28, 6.28],
                [-6.28, 6.28],
                [-0.7, 0.7],
                [0, 0.8],
            ]
        )

        self.robot_vel_bound = np.array(
            [
                [-2.09 / den, 2.09 / den],
                [-2.09 / den, 2.09 / den],
                [-1.57 / den, 1.57 / den],
                [-1.57 / den, 1.57 / den],
                [-1.57 / den, 1.57 / den],
                [-1.57 / den, 1.57 / den],
                [-3, 3],
            ]
        )

        self.control = rtde_control.RTDEControlInterface(self.HOST)
        self.receive = rtde_receive.RTDEReceiveInterface(self.HOST)
        self.dashboard = dashboard_client.DashboardClient(self.HOST)
        self.channel = channel
        self.grip_status = None

        print("Creating gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        self.gripper.connect(self.HOST, 63352)

        print("Activating gripper...")
        self.gripper.activate()
        # [4.549111843109131, -0.7404842537692566, 1.6890791098224085, 2.703993006343506, -1.430075470601217, -3.2127202192889612]

        self.mask_model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "GroundingDINO/weights/groundingdino_swint_ogc.pth",
        )
        self.BOX_THRESHOLD = 0.55
        self.TEXT_THRESHOLD = 0.25
        self.TEXT_PROMPT = "toy rubber duck"
        self.GDINO_Coord = [0, 0]
        self.GDINO_array = []

        self.command_queue = queue.Queue()
        self.ur10_lock = threading.Lock()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.shutdown_flag = threading.Event()  # Used to signal the thread to stop
        self.command_thread = threading.Thread(target=self.command_sender)
        self.command_thread.daemon = True
        self.command_thread.start()
        self.last_command = None
        self.target_pos = None

        obs_range = (-10, 10)
        self.done = False
        self.obs_dict = {}
        # Define OpenAI Gym action and state spaces
        self.normalize_act = True
        self.current_image = None
        self.mask_image = np.ones(
            (
                self.WIDTH,
                self.HEIGHT,
            )
        )
        self.STEPS_IN_EPISODE = 100
        self.DT = 0.006
        self.reset_status = False
        self.vel_action = self.receive.getActualQd()
        self.target_x, self.target_y = 0, 0
        self.step_count = 0
        self.duck_pos = [-0.19879814317014585, 1.1419941786129197, 0.09736425260562352]
        self.frog_pos = [-0.45390599718515445, 1.002975621079947, 0.05415598941643378]
        act_low = -np.ones(7)
        act_high = np.ones(7)
        self.nwidth = 120
        self.nheight = 212
        self.action_space = gym.spaces.Box(
            act_low, act_high, dtype=np.float64
        )  # clockwise or counterclockwise, for each of the 5 moving joints
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.nwidth, self.nheight, self.channel),
                    dtype=np.float64,
                ),  # Use np.float32 here
                "vector": gym.spaces.Box(
                    obs_range[0] * np.ones(14),
                    obs_range[1] * np.ones(14),
                    dtype=np.float64,
                ),  # Ensure consistency in dtype usage
            }
        )

    def check_boundary(self):
        tcp_pose = self.receive.getActualTCPPose()
        if tcp_pose[2] < 0.070 and not self.re_pos:  # Warning plane in Z
            self.hard_grip = True
            self.clear_queue(self.command_queue)
            self.last_command = None
            self.gripper.move(255, 255, 25)
            self.done = True
            time.sleep(3)
            """
            self.command_queue.put(
                {
                    "ctrl_desired": [0, -0.3, 0, 0, 0, 0, 0],
                    "last_qpos": self.receive.getActualQ() + [0],
                }
            )
            """
            print("Hard Gripping Encoded")
            # print("WARNING, robot end effector too close to table, issuing reset.")

    def get_obs_dict(self):
        self.obs_dict["qp_robot"] = np.concatenate(
            [
                self.receive.getActualQ(),
                np.array(
                    [0.0]
                ),  # np.array([self.gripper.get_current_position() / 255]),
            ]
        )
        self.obs_dict["qv_robot"] = np.concatenate((self.vel_action[:6].copy(), [0]))
        self.obs_dict["reach_err"] = np.array(self.target_pos) - np.array(
            self.receive.getActualTCPPose()[:3]
        )  # use the end_effector to check
        # self.current_observation = self.get_observation(show=True)

        return self.obs_dict

    def command_sender(self):
        print("Command sender is now running...")
        while not self.shutdown_flag.is_set():
            # self.pause_event.wait()
            # command_received_time = time.time()  # Time before trying to get a command
            try:
                command = self.command_queue.get(timeout=0.002)
                if command is None:
                    print("No more commands. Shutting down.")
                    break
                with self.ur10_lock:
                    self.last_ctrl, self.vel_action = self.vtp_step(
                        ctrl_desired=command["ctrl_desired"],
                        last_qpos=command["last_qpos"],
                    )
                self.last_command = command
            except queue.Empty:
                if not self.done or not self.hard_grip:
                    self.resend_last_command()
            except Exception as e:
                print(f"Error processing command: {e}")

    def resend_last_command(self):
        if self.last_command is not None:
            # print(f"Resending last command: {self.last_command}")
            with self.ur10_lock:
                if self.reset_status:
                    last_qpos = self.last_command["last_qpos"]
                else:
                    last_qpos = self.receive.getActualQ() + [
                        self.gripper.get_current_position()
                    ]
                self.last_ctrl, self.vel_action = self.vtp_step(
                    ctrl_desired=self.last_command["ctrl_desired"], last_qpos=last_qpos
                )

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict["reach_err"], axis=-1)
        self.depth = reach_dist
        contact = self.grip_status == 2
        if contact == True:
            if self.touch_success == 1:
                print("grasping")
            self.touch_success += 1
        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys[]
                ("reach", np.array([reach_dist])),
                ("contact", np.array([contact == True])),
                ("penalty", np.array([-1])),
                ("sparse", np.array([0.0])),
                ("solved", np.array([reach_dist < 0.08])),
                (
                    "done",
                    np.array([reach_dist < 0.08]),
                ),  #    obj_height  - self.obj_init_z > 0.2, #reach_dist > far_th
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def vtp_step(self, ctrl_desired, last_qpos, n_frames=1):
        """
        Apply controls and step forward in time
        INPUTS:
            ctrl_desired:       Desired control to be applied(sim_space)
            step_duration:      Step duration (seconds)
            ctrl_normalized:    is the ctrl normalized to [-1, 1]
            realTimeSim:        run simulate real world speed via sim
        """
        dt = self.DT
        control_v = (
            self.robot_vel_bound[:6, 1] + self.robot_vel_bound[:6, 0]
        ) / 2.0 + ctrl_desired[:6] * (
            self.robot_vel_bound[:6, 1] - self.robot_vel_bound[:6, 0]
        ) / 2.0
        control = last_qpos[:6] + control_v[:6] * dt
        ctrl_feasible = np.clip(
            control, self.robot_pos_bound[:6, 0], self.robot_pos_bound[:6, 1]
        )

        gripper_pos = int(ctrl_desired[-1])

        self.check_boundary()

        """
        if gripper_pos > 200:
            self.gripper_cd = self.GRIPPER_COUNT

        if self.gripper_cd > 0:
            gripper_pos = 207
            self.gripper_cd -= 1
        """

        for _ in range(n_frames):
            protective_stop = self.servoJ(ctrl_feasible[:6])
            # self.gripper.move(gripper_pos, 255, 25)

        return ctrl_feasible, control_v

    def servoJ(self, new_pos):
        success = self.control.servoJ(
            new_pos, self.SPEED, self.ACCELERATION, 0.002, 0.1, 300
        )
        # time.sleep(0.01)
        while not success:
            self.pause_event.clear()
            if self.receive.isProtectiveStopped():
                self.reconnect()
                protective_stop = True
                # time.sleep(
                #     3
                # )  # cannot unlock protective stop before it has been stopped for 5 seconds
                print(self.dashboard.safetystatus())
                self.dashboard.unlockProtectiveStop()
                time.sleep(1)
                print(self.dashboard.safetystatus())
                self.dashboard.closeSafetyPopup()
                # self.reconnect()

                if not self.receive.isProtectiveStopped():
                    self.episode_count += 1
                    self.done = True
                    print("epsiode ends with protective stop")

                self.pause_event.set()  # Resume threads
                self.control.reuploadScript()
                print(
                    "Thread status after handling protective stop:",
                    self.command_thread.is_alive(),
                )

                return protective_stop
            else:
                self.reconnect()
                success = self.control.servoJ(
                    new_pos, self.SPEED, self.ACCELERATION, 0.002, 0.1, 300
                )
                time.sleep(0.4)

    def moveJ(self, new_pos):
        success = self.control.moveJ(new_pos, speed=1.25, acceleration=0.8)
        # time.sleep(0.01)
        while not success:
            if self.receive.isProtectiveStopped():
                self.reconnect()
                protective_stop = True
                time.sleep(
                    6
                )  # cannot unlock protective stop before it has been stopped for 5 seconds
                print(self.dashboard.safetystatus())
                self.dashboard.unlockProtectiveStop()
                time.sleep(2)
                print(self.dashboard.safetystatus())
                self.dashboard.closeSafetyPopup()
                # self.reconnect()

                if not self.receive.isProtectiveStopped():
                    self.episode_count += 1
                    self.done = True
                    print("epsiode ends with protective stop")

                return protective_stop
            else:
                self.reconnect()
                # print('reconnecting')
                success = self.control.moveJ(new_pos, speed=0.25, acceleration=0.01)
                time.sleep(0.4)
        return success

    def step(self, a, sleep=None):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        change control method here if needed
        """
        self.step_count += 1
        self.state = self.receive.getActualQ()
        self.LAST_STATE = np.append(
            self.state.copy(), self.gripper.get_current_position()
        )  # save the last state before we take the action. and normalized the gripper to [-1, 1]

        frames = self.CAMERA.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        self.image_state = np.asanyarray(color_frame.get_data())
        self.image_state = cv.cvtColor(self.image_state, cv.COLOR_BGR2RGB)

        # self.LAST_IMAGE_STATE = self.image_state.copy()

        a = np.clip(a, self.action_space.low, self.action_space.high)
        a[-1] = (a[-1] > 0).astype(int) * 225 + 3
        self.LAST_ACTION = a
        if np.any(a != self.last_command):
            self.command_queue.put(
                {"ctrl_desired": self.LAST_ACTION, "last_qpos": self.LAST_STATE}
            )

        # self.check_boundary()
        self.contact = self.gripper._get_var("OBJ")
        return self.forward()

    def forward(self):
        """
        Forward propagate env to recover env details
        Returns current obs(t), rwd(t), done(t), info(t)
        """

        self.obs_dict = self.get_obs_dict()
        self.rwd_dict = self.get_reward_dict(self.obs_dict)

        env_info = {
            "rwd_dense": self.rwd_dict["dense"][()],  # MDP(t)
            "rwd_sparse": self.rwd_dict["sparse"][()],  # MDP(t)
            "solved": self.rwd_dict["solved"][()],  # MDP(t)
            "done": self.rwd_dict["done"][()],  # MDP(t)
            "obs_dict": self.obs_dict,  # MDP(t)
            "rwd_dict": self.rwd_dict,  # MDP(t)
        }

        if self.rwd_dict["solved"]:
            self.step_count = 0
            self.episode_count += 1
            self.done = True

        if self.control == 2:
            old_pos = self.receive.getActualQ()
            new_pos = old_pos
            new_pos[1] = new_pos[1] - 0.3
            self.control.moveJ(new_pos)
            self.episode_count += 1
            if self.gripper._get_var("OBJ") == 2:
                self.rwd_dict["dense"] += 10
                print("episode ends with successful gripping")
            else:
                print("gripping not successful")
            self.done = True
            self.contact.moveJ(old_pos)
            self.gripper.move_and_wait_for_pos(0, 25, 25)
            return self.forward()

        if self.done:
            self.episode_count += 1
            self.clear_queue(self.command_queue)
        elif self.step_count == self.STEPS_IN_EPISODE:
            self.clear_queue(self.command_queue)
            print("episode " + str(self.episode_count) + " terminated")
            self.episode_count += 1
            self.done = True
            env_info["TimeLimit.truncated"] = True
            return self.forward()

        obsvec = np.zeros(0)
        for keys in self.DEFAULT_OBS_KEYS:
            obsvec = np.concatenate([obsvec, self.obs_dict[keys]])

        ##
        self.mask_image = self.get_mask(self.image_state)
        if self.channel == 4:
            self.current_image = np.concatenate(
                (
                    self.image_state / 255,
                    np.expand_dims(self.mask_image / 255, axis=-1),
                ),
                axis=2,
            )
        else:
            self.current_image = np.expand_dims(self.mask_image / 255, axis=-1)
        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        # print(image.size)
        obs = {
            "image": self.current_image.reshape(
                (self.nwidth, self.nheight, self.channel)
            ),
            "vector": obsvec,
        }
        obs = {
            "image": self.current_image.reshape(
                (self.nwidth, self.nheight, self.channel)
            ),
            "vector": obsvec,
        }

        return obs, env_info["rwd_dense"], self.done, None, env_info

    def get_mask(self, rgb):
        mask = np.zeros((self.WIDTH, self.HEIGHT), dtype=np.uint8)

        # rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        boxes, logits, phrases = predict(
            model=self.mask_model,
            image=self.load_image2(pil_image),
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
        )
        self.rgb_out = annotate(
            image_source=rgb, boxes=boxes, logits=logits, phrases=phrases
        )
        if logits.nelement() > 0:
            max, indices = torch.max(logits, dim=0)
            boxes = boxes.numpy()
            boxes = boxes[indices]

        mask = np.zeros((self.WIDTH, self.HEIGHT), dtype=np.uint8)

        mask = self.create_mask(mask, boxes=boxes)

        self.mask_out = mask

        mask = cv.resize(
            mask, dsize=(self.nheight, self.nwidth), interpolation=cv.INTER_CUBIC
        )

        self.image_state = cv.resize(
            rgb, dsize=(self.nheight, self.nwidth), interpolation=cv.INTER_CUBIC
        )

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
            cv.rectangle(
                mask, top_left, bottom_right, (255), thickness=-1
            )  # Fill the rectangle
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

    def clear_queue(self, q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def reset(self, seed=None):
        print("reset has been called")

        self.done = False
        if self.hard_grip:
            self.re_pos = True
            grasp_q = np.deg2rad([268.60, -87.35, 131.25, 177.02, -94.72, 0])
            while (
                np.linalg.norm(
                    np.asarray(self.receive.getActualQ()) - np.asanyarray(grasp_q)
                )
                > 1e-4
            ):
                # print(np.linalg.norm(np.asarray(self.receive.getActualQ())- np.asanyarray(start_q)))
                self.command_queue.put(
                    {"ctrl_desired": [0] * 7, "last_qpos": list(grasp_q) + [0]}
                )

            time.sleep(1)
            self.clear_queue(self.command_queue)

            new_q = np.deg2rad([250, -60.18, 126.97, 160, -92.21, 0]).tolist()

            while (
                np.linalg.norm(
                    np.asarray(self.receive.getActualQ()) - np.asanyarray(new_q)
                )
                > 1e-4
            ):
                # print(np.linalg.norm(np.asarray(self.receive.getActualQ())- np.asanyarray(start_q)))
                self.command_queue.put(
                    {"ctrl_desired": [0] * 7, "last_qpos": list(new_q) + [0]}
                )
            self.clear_queue(self.command_queue)
            self.gripper.move(0, 225, 25)

            time.sleep(1)

        start_q = FIXED_START_2
        # time.sleep(5)

        self.reset_status = True
        self.done = False
        self.gripper_cd = self.GRIPPER_COUNT
        self.hard_grip = False

        self.clear_queue(self.command_queue)

        while (
            np.linalg.norm(
                np.asarray(self.receive.getActualQ()) - np.asanyarray(start_q)
            )
            > 1e-4
        ):
            # print(np.linalg.norm(np.asarray(self.receive.getActualQ())- np.asanyarray(start_q)))
            self.command_queue.put(
                {"ctrl_desired": [0] * 7, "last_qpos": list(start_q) + [0]}
            )
            self.gripper.move(0, 225, 25)
            time.sleep(0.1)

        self.re_pos = False
        self.clear_queue(self.command_queue)
        self.gripper.move(0, 225, 25)
        # self.pause_event.set()
        self.reset_status = False
        target_names = [
            "plastic cup",
            "red cube",
            "green apple",
            "yellow rubber duck",
        ]  #'toy robot', 'toy plush dinosaur', 'water bottle'] #'toy rubber duck', 'plastic cylinder',
        pos_1 = [0.2032783405727848, 1.0212731987775898, 0.12076091415859008]
        pos_2 = [-0.21295067934243966, 1.119986777469685, 0.08473901185316574]
        pos_3 = [0.16557114645698465, 1.0496839090325065, 0.1182154298508489]
        target_pos_list = [pos_1, pos_2, pos_3]
        chosen_target = random.randint(0, 2)
        chosen_target = 0

        self.TEXT_PROMPT = target_names[chosen_target]
        self.target_pos = target_pos_list[chosen_target]
        print(self.TEXT_PROMPT)

        frames = self.CAMERA.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        self.image_state = np.asanyarray(color_frame.get_data())

        # obs = super().reset(reset_qpos = reset_qpos, reset_qvel = None, **kwargs)
        # self._last_robot_qpos = self.sim.model.key_qpos[0].copy()
        self.final_image = np.ones(
            (self.nwidth, self.nheight, self.channel), dtype=np.uint8
        )
        self.step_count = 0
        obs = (
            self.receive.getActualQ()
            + [self.gripper.get_current_position()]
            + self.receive.getActualQd()
            + [0]
        )
        return {"image": np.array(self.final_image), "vector": np.array(obs)}

    def render(self, mode="human"):
        # concatenate the arrays so the display is side by side
        if mode == "human":
            cv.imshow("UR10e View", self.rgb_out)
            cv.imshow("observation", self.mask_out)
            cv.waitKey(2)
        elif mode == "rgb_array":
            cv.imshow("observation", self.image_state)
            cv.waitKey(5)
            return self.image_state

    def close(self):
        self.shutdown_flag.set()
        self.command_queue.put(None)  # Unblock the thread if waiting
        self.command_thread.join()
        self.CAMERA.stop()
        self.control.disconnect()
        self.receive.disconnect()
        self.dashboard.disconnect()
