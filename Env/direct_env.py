import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

from Data.data_process import load_stroke_png, preprocess_stroke_png, extract_skeleton_trace
from utils.custom_rewards import cos_sim_reward
from utils.mypaint_agent import MypaintAgent


class DirectCnnEnv(gym.Env):
    def __init__(self, env_config: dict):
        self.config = env_config
        self.image_size = self.config['image_size']
        self.roi_grid_size = self.config['roi_grid_size']
        self.pixels_per_grid = self.config['pixels_per_grid']
        self.brush_name = self.config['brush_name']
        self.num_keypoints = self.config['num_keypoints']
        self.target_image = self.config['target_image']
        self.z_grid_size = self.config['z_grid_size']

        # Spaces
        num_grids = int(self.image_size / self.roi_grid_size)
        self.num_pixels = self.pixels_per_grid * num_grids
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.num_pixels,
                                                       self.num_pixels,
                                                       1),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([num_grids, num_grids, int(1 / self.z_grid_size) + 1] *
                                                     self.num_keypoints)

        self.agent = MypaintAgent(env_config)

    def step(self, action: list):
        self.action = action
        wps = np.array(action, dtype=np.float).reshape(self.num_keypoints, 3)
        wps *= np.array([
            self.roi_grid_size / self.image_size,
            self.roi_grid_size / self.image_size, int(1 / self.z_grid_size) + 1
        ])
        self.agent.paint(wps[0, 0], wps[0, 1], 0)  # Draw first point without z
        for wp in wps:
            x, y, z = wp
            self.agent.paint(x, y, z)

        self.result_img = self.agent.get_img(
            (self.image_size, self.image_size))
        # Calculate reward
        reward = cos_sim_reward(self.result_img, self.target_image)

        return None, reward, True, {}  # obs, reward, done, info

    def reset(self):
        num_grids = int(self.image_size / self.roi_grid_size)
        num_pixels = self.pixels_per_grid * num_grids
        self.action = None
        self.agent.reset()

        obs = cv2.resize(self.target_image, (self.num_pixels, self.num_pixels))
        obs = np.expand_dims(obs, axis=-1)
        return obs

    def render(self, **kwargs):
        wps_frame = np.zeros((self.image_size, self.image_size))
        wps = np.array(self.action, dtype=np.float).reshape(self.num_keypoints, 3)
        wps *= np.array([
            self.roi_grid_size / self.image_size,
            self.roi_grid_size / self.image_size, int(1 / self.z_grid_size) + 1
        ])
        for wp in wps:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((self.roi_grid_size, self.roi_grid_size))
        wps_frame = scipy.ndimage.convolve(wps_frame, kernel, mode='constant', cval=0.0)
        plt.imshow(np.stack([wps_frame, self.target_image, self.result_img], axis=-1))
        plt.show()


if __name__ == '__main__':
    # Settings
    image_size = 192 * 4
    roi_grid_size = 16 * 2  # Each roi contains 3*3 roi grids
    pixels_per_grid = 3
    z_grid_size = 1 / 10
    num_keypoints = 6

    ori_img = load_stroke_png(11)
    print(f'Shape of origin image is {ori_img.shape}')

    preprocessed_img = preprocess_stroke_png(ori_img, image_size=image_size)
    print(f'Shape of preprocessed image is {preprocessed_img.shape}')

    reference_path = extract_skeleton_trace(preprocessed_img, roi_grid_size)

    env_config = {
        'image_size': image_size,
        'roi_grid_size': roi_grid_size,
        'pixels_per_grid': pixels_per_grid,
        'z_grid_size': z_grid_size,
        'brush_name': 'custom/slow_ink',
        'num_keypoints': num_keypoints,
        'target_image': preprocessed_img,
    }

    direct_cnn_env = DirectCnnEnv(env_config)
