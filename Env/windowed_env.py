import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage import draw

from Data.data_process import load_stroke_png, preprocess_stroke_png, extract_skeleton_trace, cut_roi, \
    refpath_to_actions
from Env.core_config import experimental_config
from utils.custom_rewards import cos_sim_reward
from utils.mypaint_agent import MypaintAgent


class WindowedCnnEnv(gym.Env):
    def __init__(self, env_config: dict):
        self.config = env_config
        self.image_size = self.config['image_size']
        self.window_size = self.config['window_size']
        self.brush_name = self.config['brush_name']
        self.image_nums = self.config['image_nums']
        self.z_size = self.config['z_size']

        self.all_ref_paths = {}

        # Spaces
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.window_size,
                                                       self.window_size,
                                                       3),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([5, 5, 5])  # 0 for left/down, 1 for stay, 2 for right/up
        self.agent = MypaintAgent(env_config)

    def step(self, action: list):
        action = np.array(action) - np.array([2, 2, 2])
        self.cur_pos += action * np.array([0.5 * self.window_size / self.image_size,
                                           0.5 * self.window_size / self.image_size,
                                           self.z_size])
        self.cur_pos = np.clip(self.cur_pos, 0, 1)

        # prev
        prev_img = self.agent.get_img((self.image_size, self.image_size))
        # current
        self.agent.paint(self.cur_pos[0], self.cur_pos[1], self.cur_pos[2])  # Draw first point without z
        cur_img = self.agent.get_img((self.image_size, self.image_size))

        # todo: instant reward

        # observation
        position = (self.cur_pos * np.array([self.image_size - 1, self.image_size - 1, 1]))
        self.history.append(position)

        cur = cut_roi(cur_img, position, self.window_size)
        prev = cut_roi(prev_img, position, self.window_size)
        tar = cut_roi(self.target_image, position, self.window_size)
        obs = np.stack([cur, prev, tar], axis=-1)

        if self.cur_pos[-1] == 0:
            done = True
            reward = cos_sim_reward(cur_img, self.target_image)
        else:
            done = False
            reward = 0

        return obs, reward, done, {'history': self.history}  # obs, reward, done, info

    def reset(self):
        image_num = np.random.choice(self.image_nums)
        # print(f'Image No.{image_num}\n')
        ori_img = load_stroke_png(image_num)
        self.target_image = preprocess_stroke_png(ori_img, image_size=self.image_size)

        if image_num in self.all_ref_paths:
            self.cur_ref_path = self.all_ref_paths[image_num]
        else:
            self.cur_ref_path = extract_skeleton_trace(self.target_image, self.window_size)
            self.all_ref_paths[image_num] = self.cur_ref_path
        start_point = self.cur_ref_path[0]

        self.cur_pos = np.array([start_point[0] / self.image_size, start_point[1] / self.image_size, 0.0])

        prev_img = np.zeros((self.window_size, self.window_size))
        cur_img = np.zeros((self.window_size, self.window_size))

        position = (self.cur_pos * np.array([self.image_size - 1, self.image_size - 1, 1]))
        tar_img = cut_roi(self.target_image, position, self.window_size)
        obs = np.stack([cur_img, prev_img, tar_img], axis=-1)

        self.agent.reset()
        self.history = [position]
        return obs

    def render(self, **kwargs):
        result_img = self.agent.get_img((self.image_size, self.image_size))
        tar_img = self.target_image

        # paint Waypoints
        wps_img = np.zeros((self.image_size, self.image_size))
        for wp in self.history:
            wps_img[int(wp[0]), int(wp[1])] = wp[2]
        wps_img = scipy.ndimage.convolve(wps_img, np.ones((9, 9)), mode='constant', cval=0.0)

        for wp0, wp1 in zip(self.history, self.history[1:]):
            rr, cc, val = draw.line_aa(int(wp0[0]), int(wp0[1]), int(wp1[0]), int(wp1[1]))
            wps_img[rr, cc] = val

        plt.imshow(np.stack([wps_img, result_img, tar_img], axis=-1))
        plt.show()


if __name__ == '__main__':
    # Settings

    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
    }

    windowed_env = WindowedCnnEnv(env_config)

    fake_actions = np.random.choice(np.arange(5), size=(30, 3))
    fake_actions[:5, -1] = 3

    for i in range(10):
        obs = windowed_env.reset()
        fake_path = windowed_env.cur_ref_path
        fake_actions = refpath_to_actions(fake_path, experimental_config.window_size, experimental_config.action_shape)
        for action in fake_actions:
            obs, reward, done, info = windowed_env.step(action)
            if done:
                break

        # print(np.stack(windowed_env.history, axis=0))

        windowed_env.render()
