import numpy as np
import gymnasium as gym
from gymnasium import spaces  
from gymnasium.spaces import Box
from collections import deque
import mujoco
from jsac.envs.rl_chemist.rotation import quat2euler
import cv2
import os


script_dir = os.path.dirname(__file__)
MODEL_PATH = xml_path = os.path.join(script_dir, 'env.xml')
# ARM_ACTUATOR_NAMES = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7']
FINGER_ACTUATOR_NAME = 'actuator8'
ARM_DOF = 7
ARM_VEL_LIMITS = np.array([2.61799, 2.61799, 2.61799, 2.61799, 3.14159, 3.14159, 3.14159])
GRAB_Z_RANGE = [0.275, 0.313]
LIFT_Z_POS = 0.45
BOUNDING_BOX = [0.25, 1, -0.5, 0.5, 0.15, 1]
ANGLE_TOL = 0.261799
STATE1 = 0
STATE2 = 1
STATE3 = 2
STATE_OH = np.array([[1,0,0], [0,1,0], [0,0,1]])
TARGET_CROP_RATIO = [0.3625, 0.5125, 0.12, 0.88]
GREEN_L = np.array([35, 100, 100])
GREEN_U = np.array([85, 255, 255])
STATE1_MIN_COVER = 0.9 
HAND_TARGET_ANGLES = np.array([-np.pi, 0, 0])
FINGER_GRAB_POS_MIN = 0.017
FINGER_GRAB_POS_MAX = 0.02
DEFAULT_CAM_NAME = 'camera_1'

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1 
    delta_theta_normalized = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    
    return abs(delta_theta_normalized)

class RLChemistEnv(gym.Wrapper):
    def __init__(self, 
                 image_width=84, 
                 image_height=84, 
                 image_history=3,
                 frame_skip=20, 
                 max_epi_steps=500):
        self._image_shape = (image_height, image_width, image_history * 3)
        self._image_history = image_history
        self._image_buffer = deque([], maxlen=image_history)
                
        self._model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=image_height, width=image_width)
        
        self._frame_skip = frame_skip
        self._dt = self._model.opt.timestep * self._frame_skip

        self._default_kf = self._model.keyframe('default')
        # self._default_kf = self._model.keyframe('home')
        
        # self._arm_actuator_idx = [self._model.actuator(name).id for name in ARM_ACTUATOR_NAMES]
        self._finger_actuator_idx = self._model.actuator(FINGER_ACTUATOR_NAME).id  
        
        self._arm_ctrlrange_min = self._model.actuator_ctrlrange[:, 0][:ARM_DOF]
        self._arm_ctrlrange_max = self._model.actuator_ctrlrange[:, 1][:ARM_DOF] 
        
        self._finger_ctrlrange_min = self._model.actuator_ctrlrange[:, 0][ARM_DOF]
        self._finger_ctrlrange_max = self._model.actuator_ctrlrange[:, 1][ARM_DOF]
        
        self._r1 = int(TARGET_CROP_RATIO[0] * image_height) 
        self._r2 = int(TARGET_CROP_RATIO[1] * image_height) + 1 
        self._c1  = int(TARGET_CROP_RATIO[2] * image_width) 
        self._c2 = int(TARGET_CROP_RATIO[3] * image_width) + 1 
        self._cropped_area = (self._r2 - self._r1) * (self._c2 - self._c1)
        
        self._hand_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self._tube_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'tube')
        
        self._episode = -1
        self._reset_flag = 1
        self._max_epi_steps = max_epi_steps
        
        
    def reset(self):
        self._reset_flag = 0
        self._epi_steps = 0
        
        init_q_pos = self._default_kf.qpos.copy()
        
        # Randomize initial arm position
        init_q_pos[2] = np.random.uniform(-0.6, 0)
        init_q_pos[3] = np.random.uniform(-2, -1.6)
        init_q_pos[5] = np.random.uniform(1.3, 1.8)
        
        self._data.qpos = init_q_pos
        mujoco.mj_forward(self._model, self._data)
        
        ## CTRL value for the fingers is in the range 0-255, 0 is closed and 255 is open
        ## CTRL index for the fingers is 7
        init_q_pos[7] = 255
        self._data.ctrl[:8] = init_q_pos[:8] 
        mujoco.mj_step(self._model, self._data, 1000) 
        
        self._tube_initial_pos, self._tube_initial_rot = self._get_pos_rot(self._tube_id)        
        
        self._arm_last_qpos = self._data.qpos[:ARM_DOF]
        self._state = STATE1
        
        self._last_cover = 0
        
        self._episode += 1
        
        hand_pos, hand_rot = self._get_pos_rot(self._hand_id)
        
        image, _ = self.get_image(DEFAULT_CAM_NAME, first_step=True)
        propriocption = np.concatenate((self._arm_last_qpos, hand_pos, hand_rot, STATE_OH[STATE1]))
        
        return (image, propriocption)
        
        
    def get_image(self, cam_name, first_step=False, add_to_buffer=True):
        self._renderer.update_scene(self._data, camera=cam_name)
        image = None
        image_single = self._renderer.render()
        if add_to_buffer:
            if first_step:
                for _ in range(self._image_history):
                    self._image_buffer.append(image_single)
            else:
                self._image_buffer.append(image_single)
            
            image = np.concatenate(self._image_buffer, axis=-1)
        return image, image_single
    
    def step(self, action):
        assert self._reset_flag == 0
        self._epi_steps += 1
        
        action = np.clip(action, -1.0, 1.0) 
        action = (((action + 1) * 0.5) * ARM_VEL_LIMITS * 2) - ARM_VEL_LIMITS
        action = action * self._dt
        
        # if self._state == STATE2:
        #     action *= 0 
        # else:
        if self._state == STATE1:
            action *= 0.4
        else:
            action *= 0.2
                
        new_qpos = self._arm_last_qpos + action
        clipped_new_qpos = np.clip(new_qpos, self._arm_ctrlrange_min, self._arm_ctrlrange_max)
        
        self._data.ctrl[:ARM_DOF] = clipped_new_qpos
        mujoco.mj_step(self._model, self._data, self._frame_skip)
        
        ## Check if hand is outside the bounding box
        hand_pos, hand_rot = self._get_pos_rot(self._hand_id)
        if not self._check_hand_in_bounding_box(hand_pos):
            new_qpos = new_qpos - (action * 2)
            clipped_new_qpos = np.clip(new_qpos, self._arm_ctrlrange_min, self._arm_ctrlrange_max)
            
            self._data.ctrl[:ARM_DOF] = clipped_new_qpos
            mujoco.mj_step(self._model, self._data, self._frame_skip)
            hand_pos, hand_rot = self._get_pos_rot(self._hand_id)
            
        self._arm_last_qpos = self._data.qpos[:ARM_DOF]

        image, image_single = self.get_image(DEFAULT_CAM_NAME)
        propriocption = np.concatenate((self._arm_last_qpos, hand_pos, hand_rot))
        reward = -1
        done = False
        info = {}
        
        if self._state == STATE1:
            if self._tube_moved():
                propriocption = np.concatenate((propriocption, STATE_OH[STATE1]))
                reward = -2
                # self._reset_flag = 1
                # done = True
                # info['truncated'] = True
                if self._epi_steps == self._max_epi_steps:
                    self._reset_flag = 1
                    info['truncated'] = True
                
            else:
                cover = self._get_img_cover(image_single)
                self._last_cover = cover
                hand_z = hand_pos[-1]
                if cover > 0.8 and hand_z >= GRAB_Z_RANGE[0] and hand_z <= GRAB_Z_RANGE[1]:
                    # propriocption = np.concatenate((propriocption, STATE_OH[STATE2]))
                    # reward = 1
                    # self._state = STATE2
                    finger_ctrl = 255
                    for _ in range(30):
                        self._data.ctrl[:ARM_DOF] = clipped_new_qpos
                        self._data.ctrl[7] = finger_ctrl
                        finger_ctrl -= 10
                        finger_ctrl = max(finger_ctrl, 0)
                        mujoco.mj_step(self._model, self._data, self._frame_skip)
                    
                    image, image_single = self.get_image(DEFAULT_CAM_NAME)
                    
                    if self._is_grabbed():
                        propriocption = np.concatenate((propriocption, STATE_OH[STATE2]))
                        reward = 75
                        self._state = STATE2
                        # self._reset_flag = 1
                        # done = True
                        # info['truncated'] = True
                        
                        with open('grabs.txt', 'a') as file:
                            file.write(f'Grabbed successfully. Steps: {self._epi_steps}, Episode: {self._episode}\n') 
                        
                        # if self._epi_steps == self._max_epi_steps:
                        #     self._reset_flag = 1
                        #     info['truncated'] = True
                        
                        return (image, propriocption), reward, done, info
                    
                    else:
                        # finger_ctrl = 0
                        # for _ in range(30):
                        #     self._data.ctrl[:ARM_DOF] = clipped_new_qpos
                        #     self._data.ctrl[7] = finger_ctrl
                        #     finger_ctrl += 10
                        #     finger_ctrl = min(finger_ctrl, 255)
                        #     mujoco.mj_step(self._model, self._data, self._frame_skip)
                        
                        # image, image_single = self.get_image(DEFAULT_CAM_NAME)
                        # self._arm_last_qpos = self._data.qpos[:ARM_DOF]
                        # hand_pos, hand_rot = self._get_pos_rot(self._hand_id)
                        # propriocption = np.concatenate((self._arm_last_qpos, hand_pos, hand_rot)) 
                        
                        reward = -10
                        self._reset_flag = 1
                        done = True
                        info['truncated'] = True
                
                if self._epi_steps == self._max_epi_steps:
                    self._reset_flag = 1
                    info['truncated'] = True
                if self._state == STATE1:
                    propriocption = np.concatenate((propriocption, STATE_OH[STATE1]))
                    if cover > 0.01:
                        pos_reward = self._hand_pos_reward(hand_pos)
                        rot_reward = self._hand_rot_reward(hand_rot)
                        reward = ((cover - 1) * 0.7) + (pos_reward * 0.15) + (rot_reward * 0.15)
                        reward = int(reward * 100) / 100.0
            return (image, propriocption), reward, done, info
        
        # elif self._state == STATE2:
        #     finger_ctrl = 255
        #     for _ in range(30):
        #         self._data.ctrl[7] = finger_ctrl
        #         finger_ctrl -= 10
        #         finger_ctrl = max(finger_ctrl, 0)
        #         mujoco.mj_step(self._model, self._data, self._frame_skip)
            
        #     image, image_single = self.get_image(DEFAULT_CAM_NAME)
            
        #     if self._is_grabbed():
        #         propriocption = np.concatenate((propriocption, STATE_OH[STATE2]))
        #         reward = 20
        #         # self._state = STATE3
        #         self._reset_flag = 1
        #         done = True
                
        #         with open('grabs.txt', 'a') as file:
        #             file.write('Grabbed successfully.\n') 
                
        #         # if self._epi_steps == self._max_epi_steps:
        #         #     self._reset_flag = 1
        #         #     info['truncated'] = True
        #     else:
        #         propriocption = np.concatenate((propriocption, STATE_OH[STATE2]))
        #         reward = -1
        #         self._reset_flag = 1
        #         done = True
                
        #     return (image, propriocption), reward, done, info
        
        elif self._state == STATE2:
            propriocption = np.concatenate((propriocption, STATE_OH[STATE2]))

            z_diff = abs(LIFT_Z_POS - hand_pos[-1])
            if not self._is_grabbed():
                reward = -75
                self._reset_flag = 1
                done = True
            else:
                if z_diff < 0.01:
                    reward = 20
                    self._reset_flag = 1
                    done = True
                    
                    with open('grabs.txt', 'a') as file: 
                        file.write(f'\tLifted  successfully. Steps: {self._epi_steps}, Episode: {self._episode}\n') 
                    
                else:
                    hand_rot_reward = max(self._hand_rot_reward(hand_rot) * 2, -0.4)
                    z_reward = max(z_diff * -3, -0.6)
                    
                    reward = hand_rot_reward + z_reward
                    reward = int(reward * 100) / 100.0
                    
                    if self._epi_steps == self._max_epi_steps:
                        self._reset_flag = 1
                        info['truncated'] = True
                
            return (image, propriocption), reward, done, info
        
            
    def _get_pos_rot(self, id):
        pos = self._data.xpos[id]
        rot = quat2euler(self._data.xquat[id])
        return pos, rot
    
    def _get_img_cover(self, image):
        cropped_image = image[self._r1:self._r2, self._c1:self._c2, :]
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, GREEN_L, GREEN_U)
        cover = np.sum((mask / 255)) / self._cropped_area
        return cover
    
    def _hand_rot_reward(self, hand_curr_rot):
        diff1 = angle_difference(hand_curr_rot[0], HAND_TARGET_ANGLES[0])
        diff2 = angle_difference(hand_curr_rot[1], HAND_TARGET_ANGLES[1])
        
        # angle_tol = np.pi / div
        
        # if diff1 < angle_tol:
        #     r1 = ((angle_tol - diff1) / angle_tol) * 0.5
        # else:
        #     r1 = 0
        
        # if diff2 < angle_tol:
        #     r2 = ((angle_tol - diff2) / angle_tol) * 0.5
        # else:
        #     r2 = 0
        
        # return r1 + r2
        
        r = max((diff1 + diff2) / (2 * np.pi), 1)
        return -r
    
    def _check_hand_in_bounding_box(self, hand_pos):
        xmin = BOUNDING_BOX[0]
        xmax = BOUNDING_BOX[1]
        ymin = BOUNDING_BOX[2]
        ymax = BOUNDING_BOX[3]
        zmin = BOUNDING_BOX[4]
        zmax = BOUNDING_BOX[5]
        
        hand_pos_x, hand_pos_y, hand_pos_z = hand_pos
        
        if hand_pos_x >= xmin and hand_pos_x <= xmax:
            if hand_pos_y >= ymin and hand_pos_y <= ymax:
                if hand_pos_z >= zmin and hand_pos_z <= zmax:
                    return True
        return False

    def _hand_pos_reward(self, hand_curr_pos):
        curr_z_pos = hand_curr_pos[-1]
        if curr_z_pos < GRAB_Z_RANGE[0]:
            return -1
        if curr_z_pos > GRAB_Z_RANGE[1]:
            diff = max(abs(curr_z_pos - GRAB_Z_RANGE[1]) / 0.7, 1)
            return -diff
        return 0
    
    def _tube_moved(self):
        tol = np.pi / 18
        curr_tube_pos, curr_tube_rot = self._get_pos_rot(self._tube_id)
        
        for i in range(3):
            diff = angle_difference(self._tube_initial_rot[i], curr_tube_rot[i])
            if diff > tol:
                return True
            
        tol = 0.01
        for i in range(3):
            diff = abs(self._tube_initial_pos[i] - curr_tube_pos[i])
            if diff > tol:
                return True
            
        return False 
    
    def _is_grabbed(self):
        finger1_pos = self._data.qpos[7] 
        finger2_pos = self._data.qpos[8] 
        
        if finger1_pos > FINGER_GRAB_POS_MIN and finger1_pos < FINGER_GRAB_POS_MAX:
            if finger2_pos > FINGER_GRAB_POS_MIN and finger2_pos < FINGER_GRAB_POS_MAX:
                return True
        return False         
         
    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(ARM_DOF,))
    
    @property
    def image_space(self):
        return Box(low=0, high=255, shape=self._image_shape, dtype=np.uint8)
    
    @property
    def proprioception_space(self):
        return Box(low=-20, high=20, shape=(16,))
    
    def seed(self, seed):
        pass
    
    
    def close(self):
        pass
    

# env = RLChemistEnv(image_height=400, image_width=400)
# image, prop = env.reset()
# import time

# k = 0

# ac1 = 0
# ac3 = 0
# ac5 = 0

# for i in range(1000):
#     action = np.array([0.0] * 7)
    
#     if env._state == 0:
#         action[3] = -0.003
#         action[5] = 0.0001 
    
#     if env._state == 2: 
#         action[1] = -0.3 
        
        
#     # print(action)
#     (image, prop), reward, done, info = env.step(action)

#     print(i, k, env._state, reward, env._get_pos_rot(env._hand_id)[0][-1])
        
#     img1 = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_RGB2BGR)
#     _, img2 = env.get_image('cam', add_to_buffer=False)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
#     img = np.hstack((img1, img2))
    
#     cv2.imshow("a", img) 
#     cv2.waitKey(1)
#     time.sleep(0.5)
    
    
#     if done or 'truncated' in info:
#         k+= 1
#         print(k)
#         time.sleep(0.5)
#         image, prop = env.reset() 
    
#     # print(image.shape, prop, reward)
