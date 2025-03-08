import cv2
import numpy as np 
import gymnasium as gym
from collections import deque
from gymnasium.spaces import Box


class UR10_4C_ENV(gym.Wrapper):
    def __init__(self, 
                 env_name,  
                 image_history=2, 
                 image_width=128, 
                 image_height=128, 
                 eval_mode=False,
                 video_path="."):

        super().__init__(gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=eval_mode))

        self._env_name = env_name
        self._image_history = image_history
        self._video_path = video_path
        
        state = self.env.reset() 
        channels = state['image'].shape[-1]
        self._single_image_shape = (image_width, image_height, channels)
        self._image_shape = (image_width, image_height, channels * self._image_history)
        self._image_buffer = deque([], maxlen=self._image_history)
        self._obs_dim = state['vector'].shape[0] 
        self._action_dim = self.env.action_space.shape[0]
        
        self._latest_image = None
        self._reset = False
        self._create_video = False
        

    @property
    def image_space(self):
        return Box(low=0, high=255, shape=self._image_shape, dtype=np.uint8)

    @property
    def proprioception_space(self):
        return self.observation_space
    
    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10, dtype=np.float32)
    
    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1, dtype=np.float32)


    def step(self, a):
        assert self._reset
        ob, reward, terminated, truncated, info = self.env.step(a)
        
        
        new_img = ob['image']
        prop = ob['vector']
        done = terminated 
        
        msk = (new_img[:, :, 3:4].squeeze(-1),)*3
        # cv2.imshow("w1", np.concatenate((new_img[:, :, 0:3], np.stack(msk, axis=-1)), axis=1))
        # cv2.waitKey(10) 
        
        if self._create_video: 
            frame = np.concatenate((new_img[:, :, 0:3][..., ::-1], np.stack(msk, axis=-1)), axis=1)
            self._video_buffer.append(frame)
        
        if truncated:
            info['truncated'] = True
        
        if done or truncated:
            self._reset = False
            if self._create_video: 
                self._save_video()
                self._create_video = False
                self._video_buffer = []
                
        new_img = cv2.resize(new_img, self._single_image_shape[0:2], interpolation=cv2.INTER_LINEAR)
        self._image_buffer.append(new_img)
        self._latest_image = np.concatenate(self._image_buffer, axis=-1)
 
        return (self._latest_image, prop), reward, done, info 

    def reset(self, create_vid=False):
        if self._create_video and len(self._video_buffer) > 0: 
            self._save_video()
            self._create_video = False
            self._video_buffer = []
        
        ob = self.env.reset() 
        new_img = ob['image']
        prop = ob['vector']
        
        msk = (new_img[:, :, 3:4].squeeze(-1),)*3
        # cv2.imshow("w1", np.concatenate((new_img[:, :, 0:3], np.stack(msk, axis=-1)), axis=1))
        # cv2.waitKey(1) 
        
        if create_vid: 
            print("Video will be created. ")
            self._create_video = True
            self._video_buffer = []
            frame = np.concatenate((new_img[:, :, 0:3][..., ::-1], np.stack(msk, axis=-1)), axis=1)
            self._video_buffer.append(frame) 
        
        new_img = cv2.resize(new_img, self._single_image_shape[0:2], interpolation=cv2.INTER_LINEAR)
        for _ in range(self._image_buffer.maxlen):
            self._image_buffer.append(new_img)
        self._latest_image = np.concatenate(self._image_buffer, axis=-1)
        
        self._reset = True
        return (self._latest_image, prop)

    def _save_video(self):
        import os
        from moviepy.editor import ImageSequenceClip
        
        print("Saving video...")
        
        num_files = len(os.listdir(self._video_path))
        vid_name = f'{self._env_name}_{num_files+1}.mp4'
        print("vid_name: ", vid_name)
        
        clip = ImageSequenceClip(self._video_buffer , fps=30)
        clip.write_videofile(f'{self._video_path}/{vid_name}', codec='libx264')
        
        del self._video_buffer
        
    def close(self):
        super().close()
        del self


