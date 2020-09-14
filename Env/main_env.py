import gym
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import draw, transform

from Data.data_process_lib import cut_roi, \
    refpath_to_actions
from Model import AttrDict
from script.main_procs.data_preprocessing import load_imgs_and_refpaths
from script.main_procs.hparams import define_hparams
from utils.custom_rewards import rewards_dict
from utils.mypaint_agent import MypaintPainter


class Canvas:
    def __init__(self, config):
        self.image_size = config.image_size
        self.frame = np.zeros((self.image_size, self.image_size))
        self.window_size = config.window_size

    def resize(self, obs):
        return transform.resize(obs, (self.window_size, self.window_size))

    def reset(self):
        self.frame = np.zeros((self.image_size, self.image_size))

    def set_image(self, image):
        assert image.shape == self.frame.shape
        self.frame = image.astype(np.float32)

    def place_delta(self, delta, position):
        window = self.resize(delta)
        half_width = int(self.window_size / 2)
        x, y = int(position[0]), int(position[1])
        img = np.pad(self.frame, (half_width, half_width), 'constant')
        img[x:x + 2 * half_width, y:y + 2 * half_width] += window
        self.frame = img[half_width:-half_width, half_width:-half_width]

    def render(self):
        plt.imshow(self.frame)
        plt.show()


class MainEnv(gym.Env):
    def __init__(self, env_config: AttrDict):
        self.config = env_config
        # Spaces
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.config.obs_size,
                                                       self.config.obs_size,
                                                       4),
                                                dtype=np.float)
        self.action_space = gym.spaces.MultiDiscrete([self.config.action_shape, self.config.action_shape,
                                                      self.config.action_shape])  # 0 for left/down, 1 for stay,
        self.mypaint_painter = MypaintPainter(env_config)

        self.images, self.ref_paths = load_imgs_and_refpaths()
        self.target_image = None
        self.accu_reward = None
        self.cur_ref_path = None
        self.cur_pos = None
        self.done = None
        self.history = None

    def step(self, action: list):
        # --- Update states ---
        zero_disp = self.config.action_shape // 2
        action = np.array(action) - np.array([zero_disp, zero_disp, zero_disp])

        xy_stride = 1 / zero_disp * self.config.xy_grid / self.config.image_size
        z_stride = 1 / zero_disp * self.config.z_grid
        self.cur_pos += action * np.array([xy_stride, xy_stride, z_stride])  # 0.5*xy_grid is minimum step
        self.cur_pos = np.clip(self.cur_pos, 0, 1)

        # prev
        prev_img = self.mypaint_painter.get_img(shape=(self.config.image_size, self.config.image_size))
        # current
        self.mypaint_painter.paint(self.cur_pos[0], self.cur_pos[1], self.cur_pos[2])
        cur_img = self.mypaint_painter.get_img(shape=(self.config.image_size, self.config.image_size))

        # --- observation: Cur/Prev/Tar/Z ---
        position = self.cur_pos * np.array([self.config.image_size - 1,
                                            self.config.image_size - 1,
                                            1])  # Decode to pixel-level
        position = np.array([round(position[0]), round(position[1]), position[2]])
        self.history.append(position)

        cur = cut_roi(cur_img, position, self.config.window_size)
        prev = cut_roi(prev_img, position, self.config.window_size)
        tar = cut_roi(self.target_image, position, self.config.window_size)
        z = self.cur_pos[-1] * np.ones_like(tar)
        obs = np.stack([cur, prev, tar, z], axis=-1)
        obs = transform.resize(obs, (self.config.obs_size, self.config.obs_size, 4))  # Resize to a smaller obs size

        # --- update Done status and calculate rewards---
        rewards = {}
        if self.cur_pos[-1] == 0:
            done = True
            for reward_name in self.config.rewards:
                if reward_name == 'curvature_loss':
                    cur_path = np.array([[x, y] for (x, y, z) in self.history])
                    rewards[reward_name] = rewards_dict[reward_name](self.cur_ref_path, cur_path)
                else:
                    rewards[reward_name] = rewards_dict[reward_name](self.target_image, cur_img)

        else:
            done = False
            for reward_name in self.config.rewards:
                if 'incremental' in reward_name:
                    delta = cur_img - prev_img
                    rewards[reward_name] = rewards_dict[reward_name](self.target_image, delta)

        reward = 0 + np.sum(list(rewards.values()))
        self.accu_reward += reward
        self.done = done
        return obs, reward, done, {'history': self.history,
                                   'rewards': rewards, }

    def reset(self):
        """
        obs: cur, prev, tar, z
        :return:
        """
        image_num = np.random.randint(low=0, high=self.config.num_images - 1)
        # print(f'Image No.{image_num}\n')
        self.target_image = self.images[image_num]
        self.cur_ref_path = self.ref_paths[image_num]

        # Start point
        start_point = self.cur_ref_path[0]
        self.cur_pos = np.array([start_point[0] / self.config.image_size, start_point[1] / self.config.image_size, 0.0])

        # Observation
        prev = np.zeros((self.config.window_size, self.config.window_size))
        cur = np.zeros((self.config.window_size, self.config.window_size))

        position = self.cur_pos * np.array([self.config.image_size - 1, self.config.image_size - 1, 1])
        position = np.array([round(position[0]), round(position[1]), position[2]])
        tar = cut_roi(self.target_image, position, self.config.window_size)
        z = self.cur_pos[-1] * np.ones_like(tar)
        obs = np.stack([cur, prev, tar, z], axis=-1)
        obs = transform.resize(obs, (self.config.obs_size, self.config.obs_size, 4))  # Resize to a obs size

        self.mypaint_painter.reset()
        self.done = False
        self.accu_reward = 0
        self.history = [position]
        return obs

    def render(self, **kwargs):
        result_img = self.mypaint_painter.get_img((self.config.image_size, self.config.image_size))
        tar_img = self.target_image

        # paint Waypoints
        wps_img = np.zeros((self.config.image_size, self.config.image_size))
        for wp in self.history:
            wps_img[int(wp[0]), int(wp[1])] = wp[2]
        wps_img = scipy.ndimage.convolve(wps_img, np.ones((9, 9)), mode='constant', cval=0.0)

        for wp0, wp1 in zip(self.history, self.history[1:]):
            rr, cc, val = draw.line_aa(int(wp0[0]), int(wp0[1]), int(wp1[0]), int(wp1[1]))
            wps_img[rr, cc] = val

        if self.done:
            fig = plt.figure()
            fig.suptitle(self.accu_reward)
            plt.imshow(np.stack([wps_img, result_img, tar_img], axis=-1))
            plt.show()
            plt.close(fig)

    @property
    def xy(self):
        return self.history[-1][0], self.history[-1][1]

    @property
    def canvas(self):
        image = self.mypaint_painter.get_img(shape=(self.config.image_size, self.config.image_size))
        canvas = Canvas(self.config)
        canvas.set_image(image)
        return canvas


if __name__ == '__main__':
    # Settings
    env_config = define_hparams()
    env_config.brush_name = 'custom/slow_ink'
    env_config.brush_factor = np.random.uniform(0.8, 1.2, (4,))

    env = MainEnv(env_config)

    for i in range(1):
        obs = env.reset()
        fake_actions = refpath_to_actions(env.cur_ref_path,
                                          env_config.xy_grid,
                                          env_config.action_shape)

        for action in fake_actions:
            obs, reward, done, info = env.step(action)
            print(info['rewards'])
            if done:
                break
            # print(np.stack(env.history, axis=0))
            delta = obs[:, :, 0] - obs[:, :, 1]
            plt.imshow(delta)
            plt.show()

        # env.render()
