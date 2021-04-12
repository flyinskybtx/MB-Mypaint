import os
from glob import glob

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage import transform

from Data import DATA_DIR
from Data.data_process_lib import load_imgs_and_refpaths, load_all_stroke_pngs
from Model import AttrDict
from script.main_procs.hparams import define_hparams
from utils.custom_rewards import rewards_dict
from utils.mypaint_agent import MypaintPainter
from Env.agent import Agent


class RobotDirectEnv(gym.Env):
    def __init__(self, env_config: AttrDict):
        self.obs_size = env_config['obs_size']
        self.num_xy = env_config['num_xy']
        self.num_z = env_config['num_z']
        self.num_images = env_config['num_images']
        self.num_waypoints = env_config['num_waypoints']
        self.reward_names = env_config['rewards']
        self.split_view = env_config['split_view']

        # --- Spaces
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(self.obs_size, self.obs_size, 1),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([self.num_xy, self.num_xy, self.num_z + 1] * self.num_waypoints)

        # --- Load Images
        self.images = load_all_stroke_pngs()

        # --- Agent
        self.agent = Agent(env_config["brush_config"])

        # --- Reward_fn
        self.reward_fn = self.make_reward_fn(self.reward_names)

    def reset(self, **kwargs):
        """
        obs: cur, prev, tar, zs
        :return:
        """
        image_num = kwargs.setdefault('image_num', None)
        if image_num is None:
            image_num = np.random.randint(low=0, high=self.num_images - 1)

        # print(f'Image No.{image_num}\n')
        self.target_image = self.to_obs(self.images[image_num])
        return self.target_image

    def step(self, action: list):
        actions = np.array(action, dtype=np.float).reshape(-1, 3)
        start_point = actions[0]
        start_point[-1] = 0
        actions = np.concatenate([np.expand_dims(start_point, axis=0), actions], axis=0)
        self.waypoints = self.actions_to_waypoints(actions)
        img = self.agent.execute(self.waypoints)
        self.obs = self.to_obs(img)

        rewards = self.reward_fn(self.target_image, self.obs)
        reward = 0 + np.sum(list(rewards.values()))

        return self.obs, reward, True, {}  # obs, accu_reward, done, info

    def render(self, **kwargs):
        wps_frame = np.zeros((self.target_image.shape[0], self.target_image.shape[1]))
        wps = self.waypoints * np.array([wps_frame.shape[0], wps_frame.shape[0], 1])
        for wp in wps:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((3, 3))
        wps_frame = scipy.ndimage.convolve(wps_frame, kernel, mode='constant', cval=0.0)
        wps_frame = np.expand_dims(wps_frame, axis=-1)

        if not self.split_view:
            plt.imshow(np.concatenate([wps_frame, self.target_image, self.obs], axis=-1))
            plt.show()
        else:
            plt.imshow(np.concatenate([wps_frame, self.target_image, self.obs], axis=1))
            plt.show()

    def actions_to_waypoints(self, actions):
        waypoints = (actions + np.array([0.5, 0.5, 0])) / np.array([self.num_xy, self.num_xy, self.num_z])
        return waypoints

    def to_obs(self, img):
        img = img.astype(np.float)
        img = transform.resize(img, (self.obs_size, self.obs_size))  # Resize to a smaller obs size
        img = img.reshape((self.obs_size, self.obs_size, 1))
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def make_reward_fn(reward_names):
        def calculate_reward(target, obs):
            rewards = {}
            for name in reward_names:
                rewards[name] = rewards_dict[name](target, obs)
            return rewards

        return calculate_reward


class DirectEnv(gym.Env):
    def __init__(self, env_config: dict):
        self.image_size = env_config['image_size']
        self.obs_size = env_config['obs_size']
        self.stride_size = env_config['stride_size']
        self.stride_amplify = env_config['stride_amplify']
        self.num_waypoints = env_config['num_waypoints']
        self.num_images = env_config['num_images']
        self.z_grid = env_config['z_grid']
        self.reward_names = env_config['rewards']

        # --- Spaces
        num_strides = int(self.image_size / self.stride_size)
        self.observation_space = gym.spaces.Box(low=0.,
                                                high=1.,
                                                shape=(self.obs_size,
                                                       self.obs_size,
                                                       1),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([num_strides, num_strides, int(1 / self.z_grid) + 1] *
                                                     self.num_waypoints)
        # --- Load Images
        self.images, self.ref_paths = load_imgs_and_refpaths()

        # --- Agent
        self.mypaint_painter = MypaintPainter(env_config)

    def step(self, action: list):
        actions = np.array(action, dtype=np.float).reshape(-1, 3)
        x, y, _ = self.action_to_paint_position(actions[0])
        self.mypaint_painter.paint(x, y, 0)

        for action in actions:
            x, y, z = self.action_to_paint_position(action)
            self.mypaint_painter.paint(x, y, z)

        rewards = self.reward_fn(self.cur_img)
        reward = 0 + np.sum(list(rewards.values()))

        self.actions = actions

        return self.to_obs(self.cur_img), reward, True, {}  # obs, accu_reward, done, info

    def reset(self, **kwargs):
        """
        obs: cur, prev, tar, zs
        :return:
        """
        image_num = kwargs.setdefault('image_num', None)
        if image_num is None:
            image_num = np.random.randint(low=0, high=self.num_images - 1)

        # print(f'Image No.{image_num}\n')
        self.target_image = self.images[image_num]
        obs = self.to_obs(self.target_image)

        # --- Reset
        self.mypaint_painter.reset()
        self.reward_fn = self.make_reward_fn(self.target_image, self.reward_names)

        return obs

    def render(self, **kwargs):
        SPLIT_VIEW = True

        wps_frame = np.zeros((self.image_size, self.image_size))
        wps = self.actions * np.array([self.stride_size, self.stride_size, self.z_grid]) + np.array(
            [self.stride_size / 2, self.stride_size / 2, 0])
        for wp in wps:
            wps_frame[int(wp[0]), int(wp[1])] = wp[2]
        kernel = np.ones((self.stride_size, self.stride_size))
        wps_frame = scipy.ndimage.convolve(wps_frame, kernel, mode='constant', cval=0.0)

        # if not SPLIT_VIEW:
        #     plt.imshow(np.concatenate([self.to_obs(wps_frame), self.to_obs(self.target_image), self.to_obs(
        #         self.cur_img)], axis=-1))
        #     plt.show()
        # else:
        #     plt.imshow(np.concatenate([self.to_obs(wps_frame), self.to_obs(self.target_image), self.to_obs(
        #         self.cur_img)], axis=1))
        #     plt.show()

        plt.imshow(np.concatenate([wps_frame, self.target_image, self.cur_img], axis=1))
        plt.show()

    def action_to_paint_position(self, action):
        x, y, z = action
        x = (x * self.stride_size + self.stride_size / 2) / self.image_size  # center at each grid
        y = (y * self.stride_size + self.stride_size / 2) / self.image_size
        z = z * self.z_grid
        return x, y, z

    @property
    def cur_img(self):
        return self.mypaint_painter.get_img(shape=(self.image_size, self.image_size))

    def to_obs(self, img):
        img = img.astype(np.float)
        img = transform.resize(img, (self.obs_size, self.obs_size))  # Resize to a smaller obs size
        img = img.reshape((self.obs_size, self.obs_size, 1))
        return np.clip(img, 0, 1)

    @staticmethod
    def make_reward_fn(target_image, reward_names):
        def calculate_reward(cur_img):
            rewards = {}
            for name in reward_names:
                rewards[name] = rewards_dict[name](target_image, cur_img)
            return rewards

        return calculate_reward


def test_directenv():
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = DirectEnv(env_config)
    for _ in range(10):
        obs = env.reset()
        action = env.action_space.sample()
        env.step(action)
        env.render()


def test_robot_directenv():
    config = AttrDict()

    config.brush_config = AttrDict()
    config.brush_config.brush_name = 'custom/slow_ink'
    config.brush_config.agent_name = 'Physics'

    config.direct_env_config = AttrDict()
    config.direct_env_config.num_xy = 24
    config.direct_env_config.num_z = 10
    config.direct_env_config.num_waypoints = 6
    config.direct_env_config.obs_size = 64
    config.direct_env_config.num_images = 64
    config.direct_env_config.rewards = [
        'img_cosine_reward',
        # 'img_mse_loss',
        # 'scale_loss',
        # 'curvature_loss',
        # 'iou_reward',
        # 'incremental_reward',
        # 'incremental_loss'
    ]
    config.direct_env_config.split_view = False
    config.direct_env_config.brush_config = config.brush_config



    env = RobotDirectEnv(config.direct_env_config)
    for _ in range(10):
        obs = env.reset()
        action = env.action_space.sample()
        env.step(action)
        env.render()


if __name__ == '__main__':
    test_robot_directenv()
