import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

from Data.data_process_lib import load_stroke_png, preprocess_stroke_png
from Env.core_config import experimental_config
from utils.custom_rewards import img_cosine_reward
from utils.mypaint_agent import MypaintAgent


class DirectCnnEnv(gym.Env):
    def __init__(self, env_config: dict):
        self.config = env_config
        self.image_size = self.config['image_size']
        self.stride_size = self.config['stride_size']
        self.stride_amplify = self.config['stride_amplify']
        self.brush_name = self.config['brush_name']
        self.num_keypoints = self.config['num_keypoints']
        self.image_nums = self.config['image_nums']
        self.z_size = self.config['z_size']

        # Spaces
        num_strides = int(self.image_size / self.stride_size)
        self.num_pixels = self.stride_amplify * num_strides
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.num_pixels,
                                                       self.num_pixels,
                                                       1),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([num_strides, num_strides, int(1 / self.z_size) + 1] *
                                                     self.num_keypoints)

        self.agent = MypaintAgent(env_config)

    def step(self, action: list):
        self.action = action
        wps = np.array(action, dtype=np.float).reshape(self.num_keypoints, 3)
        wps *= np.array([
            self.stride_size / self.image_size,
            self.stride_size / self.image_size,
            self.z_size
        ])
        self.agent.paint(wps[0, 0], wps[0, 1], 0)  # Draw first point without z
        for wp in wps:
            x, y, z = wp
            self.agent.paint(x, y, z)

        self.result_img = self.agent.get_img((self.image_size, self.image_size))
        # Calculate reward
        reward = img_cosine_reward(self.result_img, self.target_image)

        # Resize to obs
        obs = cv2.resize(self.result_img, (self.num_pixels, self.num_pixels))
        obs = np.expand_dims(obs, axis=-1)

        return obs, reward, True, {}  # obs, reward, done, info

    def reset(self):
        image_num = np.random.choice(self.image_nums)
        ori_img = load_stroke_png(image_num)
        self.target_image = preprocess_stroke_png(ori_img, image_size=self.image_size)

        self.action = None
        self.agent.reset()

        obs = cv2.resize(self.target_image, (self.num_pixels, self.num_pixels))
        obs = np.expand_dims(obs, axis=-1)
        return obs

    def render(self, **kwargs):
        wps_frame = np.zeros((self.image_size, self.image_size))
        wps = np.array(self.action, dtype=np.float).reshape(self.num_keypoints, 3)
        wps = wps * np.array([self.stride_size, self.stride_size, int(1 / self.z_size) + 1
                              ]) + np.array([self.stride_size/2, self.stride_size/2, 0])
        for wp in wps:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((self.stride_size, self.stride_size))
        wps_frame = scipy.ndimage.convolve(wps_frame, kernel, mode='constant', cval=0.0)
        plt.imshow(np.stack([wps_frame, self.target_image, self.result_img], axis=-1))
        plt.show()


if __name__ == '__main__':
    # Settings
    env_config = {
        'image_size': experimental_config.image_size,
        'stride_size': experimental_config.stride_size,
        'stride_amplify': experimental_config.stride_amplify,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'num_keypoints': experimental_config.num_keypoints,
        'image_nums': experimental_config.image_nums,
    }

    direct_cnn_env = DirectCnnEnv(env_config)
    for _ in range(10):
        obs = direct_cnn_env.reset()
        action = direct_cnn_env.action_space.sample()
        direct_cnn_env.step(action)
        direct_cnn_env.render()
