import os
from glob import glob

import gym
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import draw, transform

from Data import DATA_DIR
from Data.data_process_lib import cut_roi, refpath_to_actions, load_imgs_and_refpaths
from Model import AttrDict
from script.main_procs.hparams import define_hparams
from utils.custom_rewards import rewards_dict
from utils.mypaint_agent import MypaintPainter
from Env.canvas import Canvas


class MainEnv(gym.Env):
    def __init__(self, env_config: AttrDict):
        # --- Spaces
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config.obs_size, env_config.obs_size, 4),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([env_config.action_shape] * 3)  # 0 for left/down, 1 for stay,

        # --- Make function
        self.step_fn = self.make_step_fn(env_config)
        self.mypaint_painter = MypaintPainter(env_config)

        # --- Load Images
        self.images, self.ref_paths = load_imgs_and_refpaths()

        self.image_size = env_config.image_size
        self.obs_size = env_config.obs_size
        self.window_size = env_config.window_size
        self.reward_names = env_config.rewards
        self.num_images = env_config.num_images
        self.z_grid = env_config.z_grid

        self.reward_fn = None
        self.cur_ref_path = None
        self.target_image = None
        self._x = None
        self._y = None
        self._z = None
        self.history = []
        self._prev_img = None
        self.done = None

    def step(self, action: list):
        """

        Args:
            action:

        Returns:

        """
        # --- Update states ---
        self.position += self.step_fn(action)
        self.history.append(self.position)

        # --- Step
        self._prev_img = np.copy(self.cur_img)
        self.mypaint_painter.paint(*self.painter_position)

        # --- Check done
        self.done = True if self.position[-1] == 0 else False

        # --- Get obs
        obs = self.obs
        if self.done:
            rewards = self.reward_fn(self.cur_img, None, self.history)
        else:
            rewards = self.reward_fn(self.cur_img, self._prev_img, None)

        reward = 0 + np.sum(list(rewards.values()))

        return obs, reward, self.done, {'history': self.history,
                                        'rewards': rewards, }

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
        self.cur_ref_path = self.ref_paths[image_num]

        # --- Start point
        self.position = np.concatenate([self.cur_ref_path[0], [0]], axis=-1)

        # --- Reset
        self.done = False
        self.history = []
        self.mypaint_painter.reset()
        self.mypaint_painter.paint(*self.painter_position)
        self._prev_img = np.zeros((self.image_size, self.image_size))

        # --- Observation
        obs = self.obs

        # --- Make reward function
        self.reward_fn = self.make_reward_fn(self.cur_ref_path, self.target_image, self.reward_names)
        return obs

    def render(self, **kwargs):
        wps_img = np.zeros((self.image_size, self.image_size))

        if self.done:
            # paint Waypoints
            wps_img = np.zeros((self.image_size, self.image_size))
            for wp in self.history:
                wps_img[int(wp[0]), int(wp[1])] = wp[2]
            wps_img = scipy.ndimage.convolve(wps_img, np.ones((9, 9)), mode='constant', cval=0.0)

            for wp0, wp1 in zip(self.history, self.history[1:]):
                rr, cc, val = draw.line_aa(int(wp0[0]), int(wp0[1]), int(wp1[0]), int(wp1[1]))
                wps_img[rr, cc] = val

        fig = plt.figure()
        plt.imshow(np.stack([wps_img, self.cur_img, self.target_image], axis=-1))
        plt.show()
        plt.close(fig)

    @property
    def position(self):
        return np.array([self._x, self._y, self._z])

    @position.setter
    def position(self, value):
        self._x = np.clip(value[0], 0., self.image_size)
        self._y = np.clip(value[1], 0., self.image_size)
        self._z = np.clip(value[2], 0., 1.)

    @property
    def painter_position(self):
        return [self._x / self.image_size, self._y / self.image_size, self._z]

    @property
    def cur_img(self):
        return self.mypaint_painter.get_img(shape=(self.image_size, self.image_size))

    @property
    def ref_path(self):
        return self.cur_ref_path

    def to_canvas(self):
        obs = self.obs
        delta = obs[:, :, 0] - obs[:, :, 1]
        canvas = Canvas(self.cur_img, delta, self._x, self._y, self._z, self.window_size)
        return canvas

    @property
    def obs(self):
        cur = cut_roi(self.cur_img, self.position, self.window_size)
        prev = cut_roi(self._prev_img, self.position, self.window_size)
        tar = cut_roi(self.target_image, self.position, self.window_size)
        z = self._z * np.ones_like(tar)
        obs = np.stack([cur, prev, tar, z], axis=-1)
        obs = transform.resize(obs, (self.obs_size, self.obs_size, 4))  # Resize to a smaller obs size
        obs = np.round(obs, decimals=0)  # BW-wize the obs
        return obs

    @staticmethod
    def make_reward_fn(ref_path, target_image, reward_names):
        def calculate_reward(cur_img, prev_img=None, history=None):
            rewards = {}
            if prev_img is None:
                for name in reward_names:
                    if name == 'curvature_loss' and history is not None:
                        cur_path = np.array([position[:2] for position in history])
                        rewards[name] = rewards_dict[name](ref_path, cur_path)
                    else:
                        rewards[name] = rewards_dict[name](target_image, cur_img)

            else:
                for name in reward_names:
                    if 'incremental' in name:
                        delta = cur_img - prev_img
                        rewards[name] = rewards_dict[name](target_image, delta)
            return rewards

        return calculate_reward

    @staticmethod
    def make_step_fn(config):
        zeros_disp = config.action_shape // 2
        xy_grid = config.xy_grid
        z_grid = config.z_grid

        def step_fn(action):
            step = np.array(action) / zeros_disp - np.ones((3,))
            return step * np.array([xy_grid, xy_grid, z_grid])

        return step_fn


def test():
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = MainEnv(env_config)

    for i in range(5):
        env.reset()
        fake_actions = refpath_to_actions(env.cur_ref_path,
                                          env_config.xy_grid,
                                          env_config.action_shape)

        for action in fake_actions:
            obs, reward, done, info = env.step(action)
            print(info['rewards'])
            if done:
                break
            # print(np.stack(env.history, axis=0))
            # delta = obs[:, :, 0] - obs[:, :, 1]
            # plt.imshow(delta)
            # plt.show()

        env.render()

        print()


if __name__ == '__main__':
    test()
