import os
import cv2
import numpy as np 
import gymnasium as gym
from collections import deque
from gymnasium.spaces import Box
import mujoco

from jsac.helpers.utils import render_interactive

class RLC_Env(gym.Wrapper):
    def __init__(self, 
                 env_name,  
                 image_history=2, 
                 image_width=128, 
                 image_height=128, 
                 env_mode="train",
                 mask_type="ground_truth",
                 target_obj_num=-1,
                 mask_delay_type="none",
                 mask_delay_steps=2,
                 step_time=None,
                 video_path=".",
                 render_interactive=False):

        if target_obj_num >= 0:
            super().__init__(gym.make(f'robohive.envs:{env_name}', 
                                      env_mode=env_mode, 
                                      target_obj_num=target_obj_num))
        else:
            if step_time:
                super().__init__(gym.make(f'robohive.envs:{env_name}', 
                                        env_mode=env_mode,
                                        mask_type=mask_type,
                                        mask_delay_type=mask_delay_type,
                                        mask_delay_steps=mask_delay_steps,
                                        step_time=step_time))
            else:
                super().__init__(gym.make(f'robohive.envs:{env_name}', 
                                        env_mode=env_mode,
                                        mask_type=mask_type,
                                        mask_delay_type=mask_delay_type,
                                        mask_delay_steps=mask_delay_steps))

        self._env_name = env_name
        self._image_history = image_history
        self._video_path = video_path
        
        state = self.env.reset() 
        channels = state['image'].shape[-1]
        self._single_image_shape = (image_height, image_width, channels)
        self._image_shape = (image_height, image_width, channels * self._image_history)
        self._image_buffer = deque([], maxlen=self._image_history)
        self._obs_dim = state['vector'].shape[0] 
        self._action_dim = self.env.action_space.shape[0]
        
        self._latest_image = None
        self._reset = False
        self._create_video = False
        if render_interactive:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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

        # path = '/home/fahim/project/imgs_dump/'
        # ln = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        # img_name = f'{path}{ln}.png'
        # cv2.imshow("w1", np.concatenate((new_img[:, :, 0:3], np.stack(msk, axis=-1)), axis=1))
        # cv2.waitKey(1)
        # print(info['prompt'])
        
        if self._create_video: 
            self.add_frame_to_video_buffer(self.env.target_name.title(), new_img, msk)
        
        if truncated:
            info['truncated'] = True
        
        if done or truncated:
            self._reset = False
            if self._create_video: 
                self._save_video()
                self._create_video = False
                self._video_buffer = []
                
        new_img = cv2.resize(new_img, self._single_image_shape[0:2][::-1], interpolation=cv2.INTER_AREA)
        self._image_buffer.append(new_img)
        self._latest_image = np.concatenate(self._image_buffer, axis=-1)
 
        return (self._latest_image, prop), reward, done, info 

    def reset(self, create_vid=False, **kwargs):
        if self._create_video and len(self._video_buffer) > 0: 
            self._save_video()
            self._create_video = False
            self._video_buffer = []
        
        ob = self.env.reset(**kwargs) 

        new_img = ob['image']
        prop = ob['vector']
        
        msk = (new_img[:, :, 3:4].squeeze(-1),)*3
        
        # path = '/home/fahim/project/imgs_dump/'
        # ln = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        # img_name = f'{path}{ln}.png'
        # cv2.imshow("w1", np.concatenate((new_img[:, :, 0:3], np.stack(msk, axis=-1)), axis=1)) 
        # cv2.waitKey(1)
        
        if create_vid: 
            print("Video will be created. ")
            self._create_video = True
            self._video_buffer = []
            self.add_frame_to_video_buffer(self.env.target_name.title(), new_img, msk)
        
        new_img = cv2.resize(new_img, self._single_image_shape[0:2][::-1], interpolation=cv2.INTER_AREA)
        for _ in range(self._image_buffer.maxlen):
            self._image_buffer.append(new_img)
        self._latest_image = np.concatenate(self._image_buffer, axis=-1)
        
        self._reset = True
        
        return (self._latest_image, prop)
    
    def add_frame_to_video_buffer(self, text, new_img, msk): 
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        frame = np.concatenate((new_img[:, :, 0:3][..., ::-1], np.stack(msk, axis=-1)), axis=1).copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 2 
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Create a black image for the text
        banner_height = text_height + baseline + 20  # Adjust height for padding
        banner = np.zeros((banner_height, frame.shape[1], 3), dtype=np.uint8)

        # Calculate the center position for the text
        center_x = (banner.shape[1] - text_width) // 2
        center_y = (banner.shape[0] + text_height) // 2  # Adjust for baseline

        # Add text to the black image
        cv2.putText(banner, text, (center_x, center_y), font, font_scale, font_color, thickness)
        
        combined_image = np.vstack((frame, banner))
        # cv2.imshow("w1", combined_image)
        # cv2.waitKey(1)
        self._video_buffer.append(combined_image) 


    # def _save_video(self):
    #     import os
    #     from moviepy.editor import ImageSequenceClip
        
    #     print("Saving video...")
        
    #     num_files = len(os.listdir(self._video_path))
    #     vid_name = f'{self._env_name}_{num_files+1}.mp4'
    #     print("vid_name: ", vid_name)
        
    #     last = self._video_buffer[-1]
    #     for i in range(5):
    #         self._video_buffer.append(last)
        
    #     try:
    #         clip = ImageSequenceClip(self._video_buffer, fps=30)
    #         output_path = os.path.join(self._video_path, vid_name)
    #         clip.write_videofile(output_path, codec='libx264')
    #     except Exception as e:
    #         print(f"Error saving video: {e}")
    #     finally:
    #         self._video_buffer = []
    
    def _save_video(self): 
        print("Saving video...")
 
        num_files = len(os.listdir(self._video_path))
        vid_name = f'{self._env_name}_{num_files + 1}.mp4'
        print("vid_name: ", vid_name)
 
        last_frame = self._video_buffer[-1]
        for _ in range(5):
            self._video_buffer.append(last_frame)
 
        if not self._video_buffer:
            print("Error: Video buffer is empty.")
            return
 
        height, width, channels = self._video_buffer[0].shape
        output_path = os.path.join(self._video_path, vid_name)
 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = 30 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try: 
            for frame in self._video_buffer: 
                out.write(frame)
            print(f"Video saved at: {output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
        finally:
            out.release()  # Release the video writer
            self._video_buffer = []
    
    def sync_view(self):
        '''
            This function keeps rendering the interactive view
        '''
        if self.viewer and self.viewer.is_running():
            with self.viewer.lock():
                self.viewer.sync()
        
    def close(self):
        super().close()

        del self


### TEST ENVIRONMENT ###


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='FrankaEnv-v0', type=str, help="Two envs: FrankaEnv-v0, UR10eEnv-v0")
    args = parser.parse_args()
    env = RLC_Env(env_name=args.env_name, render_interactive=True) # replace model path accordingly
    render_interactive(env)